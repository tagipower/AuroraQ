import os
import sys
import logging
import pandas as pd
from datetime import timedelta, datetime

# ✅ 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ppo_agent_trainable import PPOAgentTrainable
from envs.env_router import get_env
from utils.reward_config_loader import load_reward_config
from utils.reward_calculator import RewardCalculator
from config.rule_param_loader import get_rule_params
from report.strategy_param_logger import log_strategy_params
from report.html_report_generator import generate_html_report
from strategy_score_manager import update_score  # ✅ 전략 점수 업데이트

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainLoop")

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def train():
    device = get_device()
    print(f"Device set to use {device}")

    # ✅ 설정값
    env_name = "rule_e"
    window_size = 10
    timesteps = 100_000
    model_path_pt = "models/ppo_latest.pt"
    model_path_zip = "models/ppo_latest.zip"
    csv_path = "data/train_data.csv"

    logger.info(f"학습 데이터: {csv_path}")
    logger.info(f"학습 에피소드 수: {timesteps}")
    logger.info(f"저장 위치 (.pt): {model_path_pt}")
    logger.info(f"저장 위치 (.zip): {model_path_zip}")

    # ✅ 파라미터 기록
    if env_name.startswith("rule_"):
        params = get_rule_params(env_name.upper().replace("RULE_", "Rule"))
        log_strategy_params(env_name, params)

    # ✅ 데이터 로딩
    data = pd.read_csv(csv_path)
    if len(data) < window_size:
        raise ValueError(f"학습 데이터가 너무 적습니다 (최소 {window_size}행 필요)")

    # ✅ 결측 컬럼 보완
    data["sentiment_score"] = data.get("sentiment_score", 0.0)
    data["event_risk"] = data.get("event_risk", False)
    data["regime"] = data.get("regime", "neutral")
    data["long_term_trend"] = data.get("long_term_trend", "sideways")
    data["sentiment_delta"] = data.get("sentiment_delta", 0.0)
    if "timestamp" not in data.columns:
        data["timestamp"] = [datetime(2025, 1, 1) + timedelta(minutes=5 * i) for i in range(len(data))]
    data["news_text"] = data.get("news_text", "")

    # ✅ 보상 계산기 구성
    reward_config = load_reward_config()
    reward_calculator = RewardCalculator(config=reward_config, sentiment_mode="backtest")

    # ✅ 환경 생성
    env = get_env(
        env_name=env_name,
        data=data,
        reward_calculator=reward_calculator,
        config={"reward": reward_config},
        window_size=window_size
    )

    # ✅ PPO 학습
    agent = PPOAgentTrainable(env)
    logger.info("📚 PPO 학습 시작")
    training_info = agent.train(timesteps)

    # ✅ 모델 저장 (선택)
    agent.save(model_path_pt, model_path_zip)

    # ✅ 전략 점수 평가 및 저장
    try:
        ppo_score = agent.score()  # PPOAgentTrainable에 .score() 메서드 구현 필요
        update_score("PPOStrategy", ppo_score, is_real_trade=False)
        logger.info(f"📊 PPO 점수화 완료: {ppo_score}")

        # ✅ 리포트 생성
        result = {
            "pnl_pct": round(ppo_score.get("total_score", 0.0) * 100, 2),
            "action": "N/A (train)",
            "leverage": 1
        }
        generate_html_report("ppo", result, output_dir="report/train/")

    except Exception as e:
        logger.warning(f"⚠️ PPO 점수화 실패: {e}")

if __name__ == "__main__":
    train()
