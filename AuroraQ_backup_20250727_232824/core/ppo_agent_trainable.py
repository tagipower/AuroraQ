import os
import torch
from stable_baselines3 import PPO

class PPOAgentTrainable:
    def __init__(self, env):
        """
        PPOAgentTrainable 객체를 초기화하며 환경 env를 인자로 받습니다.
        """
        self.name = "PPO"
        self.model = None
        self.env = env

    def train(self, timesteps=10000):
        """
        PPO 모델을 지정된 환경에서 학습합니다.
        학습 완료 후 평균 보상값(mean_reward)을 반환합니다.
        """
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=timesteps)

        # 에피소드 보상 평균 계산
        if hasattr(self.env, 'get_episode_rewards'):
            rewards = self.env.get_episode_rewards()
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                return {"mean_reward": mean_reward}

        return {"mean_reward": None}

    def save_model(self, zip_path, pt_path=None):
        """
        학습된 PPO 모델을 zip 및 선택적으로 pt 형식으로 저장합니다.
        """
        if self.model:
            self.model.save(zip_path)
            print(f"[PPOAgentTrainable] ✅ zip 저장됨: {zip_path}")
            if pt_path:
                torch.save(self.model.policy.state_dict(), pt_path)
                print(f"[PPOAgentTrainable] ✅ pt 저장됨: {pt_path}")
        else:
            raise RuntimeError("[PPOAgentTrainable] 모델이 존재하지 않아 저장할 수 없습니다.")

    def load_model(self, zip_path):
        """
        저장된 zip 모델을 불러옵니다.
        """
        if os.path.exists(zip_path):
            self.model = PPO.load(zip_path)
            print(f"[PPOAgentTrainable] 📦 zip 모델 로드됨: {zip_path}")
        else:
            raise FileNotFoundError(f"[PPOAgentTrainable] ❌ zip 파일 없음: {zip_path}")

    def predict(self, obs):
        """
        현재 상태에 대해 예측된 행동을 반환합니다.
        """
        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        else:
            raise RuntimeError("[PPOAgentTrainable] 모델이 로드되지 않았습니다.")

    def score(self):
        """
        PPO 전략 점수화 함수.
        학습된 환경의 평균 보상 등을 기반으로 점수 계산.
        """
        if hasattr(self.env, 'get_episode_rewards'):
            rewards = self.env.get_episode_rewards()
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                total_score = round(mean_reward, 4)
                return {
                    "total_score": total_score,
                    "mean_reward": round(mean_reward, 4),
                    "score_type": "mean_reward"
                }
        return {
            "total_score": 0.0,
            "mean_reward": 0.0,
            "score_type": "none"
        }
