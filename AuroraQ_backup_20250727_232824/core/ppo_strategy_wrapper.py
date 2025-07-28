import os
import pandas as pd
from datetime import datetime, timedelta

# Stable Baselines3 PPO 로드 지원
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # PPO 로드 실패 시 처리

class PPOStrategy:
    def __init__(self, model_path: str = "models/ppo_latest.zip", env=None, reward_log_path="logs/reward_log.csv"):
        """
        PPO 기반 전략 클래스.
        - model_path: 학습된 PPO 모델 경로 (시나리오마다 동적으로 지정 가능)
        - env: PPO 환경 (백테스트/실시간 공용)
        - reward_log_path: 보상 로그 경로 (성과 점수 계산용)
        """
        self.model_path = model_path
        self.env = env
        self.reward_log_path = reward_log_path
        self.model = self._load_model(self.model_path)

    def _load_model(self, path: str):
        """PPO 모델을 지정 경로에서 로드. 실패 시 None 반환."""
        if PPO is None:
            print("⚠️ stable-baselines3 라이브러리가 설치되어 있지 않습니다.")
            return None

        if not os.path.exists(path):
            print(f"⚠️ PPO 모델 경로를 찾을 수 없습니다: {path}")
            return None

        try:
            return PPO.load(path)
        except Exception as e:
            print(f"⚠️ PPO 모델 로드 실패 ({path}): {e}")
            return None

    def predict(self, obs):
        """
        관측값(obs)을 기반으로 PPO 모델로 액션을 예측.
        모델이 없을 경우 기본 0(중립) 액션 반환.
        """
        if self.model:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                return action
            except Exception as e:
                print(f"⚠️ PPO 예측 오류: {e}")
                return 0
        return 0

    def evaluate_segment(self, df_segment: pd.DataFrame):
        """
        시나리오 백테스트용: 주어진 세그먼트 DataFrame(df_segment)에서 PPO 전략을 평가.
        - 관측값 변환 후 PPO 액션을 예측하여 결과를 리스트로 반환.
        """
        results = []
        for _, row in df_segment.iterrows():
            obs = self._row_to_obs(row)
            action = self.predict(obs)
            results.append({
                "datetime": row["datetime"],
                "close": row.get("close", 0),
                "sentiment_score": row.get("sentiment_score", 0),
                "regime_score": row.get("regime_score", 0),
                "action": action
            })
        return results

    def _row_to_obs(self, row):
        """
        DataFrame 행(row)을 PPO 관측값으로 변환.
        필요한 경우 가격, 거래량, 감정 점수, 레짐 점수를 포함.
        """
        return [
            row.get("close", 0),
            row.get("volume", 0),
            row.get("sentiment_score", 0),
            row.get("regime_score", 0)
        ]

    def score(self, price_data_window=None):
        """
        PPO 전략의 점수 산정.
        최근 7일간 reward 평균을 기반으로 점수 계산.
        """
        try:
            if not os.path.exists(self.reward_log_path):
                return -999  # 보상 로그 없으면 기본 낮은 점수

            df = pd.read_csv(self.reward_log_path)
            df = df.dropna(subset=["reward"])

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_rewards = df[df["timestamp"] >= recent_cutoff]["reward"]

            if recent_rewards.empty:
                return -999

            avg_reward = recent_rewards.mean()
            return round(avg_reward, 4)

        except Exception as e:
            print(f"[PPOStrategy] 점수 계산 실패: {e}")
            return -999
