import os
import yaml
import numpy as np
from datetime import datetime, timedelta

from core.strategy_score_manager import load_score_history
from config.trade_config_loader import save_yaml_config

STRATEGY_LIST = ["RuleA", "RuleB", "RuleC", "RuleD", "PPOStrategy"]
WEIGHT_FILE_PATH = "config/strategy_weight.yaml"

# 설정: 가중치의 변화 한계
MIN_WEIGHT = 0.1
MAX_WEIGHT = 1.5
BASE_WEIGHT = 1.0

# 최근 평균 점수 계산 기준 기간 (예: 최근 5일치)
RECENT_DAYS = 5


def adjust_weight_from_score(score: float) -> float:
    """
    전략 점수를 기반으로 가중치 산정
    - 점수 1.0 기준으로 오차 범위에 따라 ± 0.1~0.3 가중
    """
    if score >= 1.2:
        return 1.3
    elif score >= 1.1:
        return 1.2
    elif score >= 1.0:
        return 1.1
    elif score >= 0.9:
        return 1.0
    elif score >= 0.8:
        return 0.9
    elif score >= 0.7:
        return 0.8
    else:
        return 0.7


def load_existing_weights(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {name: BASE_WEIGHT for name in STRATEGY_LIST}
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_updated_weights(updated_weights: dict):
    save_yaml_config(updated_weights, WEIGHT_FILE_PATH)
    print(f"✅ strategy_weight.yaml 업데이트 완료 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")


def calculate_recent_average(score_list: list, days: int = RECENT_DAYS) -> float:
    if not score_list:
        return BASE_WEIGHT
    now = datetime.now()
    recent_scores = [
        item["score"]
        for item in score_list
        if "timestamp" in item and datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S") >= now - timedelta(days=days)
    ]
    return np.mean(recent_scores) if recent_scores else BASE_WEIGHT


def update_weights():
    print(f"\n📈 전략 가중치 자동 업데이트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    score_history = load_score_history()
    current_weights = load_existing_weights(WEIGHT_FILE_PATH)
    updated_weights = {}

    for strategy in STRATEGY_LIST:
        score_list = score_history.get(strategy, [])
        recent_avg = calculate_recent_average(score_list)

        new_weight = adjust_weight_from_score(recent_avg)
        new_weight = max(min(new_weight, MAX_WEIGHT), MIN_WEIGHT)

        old_weight = current_weights.get(strategy, BASE_WEIGHT)
        updated_weights[strategy] = round(new_weight, 2)

        print(f"🔄 {strategy}: 평균 점수 = {recent_avg:.3f} → weight {old_weight:.2f} → {new_weight:.2f}")

    save_updated_weights(updated_weights)


if __name__ == "__main__":
    update_weights()
