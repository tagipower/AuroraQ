import os
import numpy as np
from stable_baselines3 import PPO
from core.state_preprocessor import StatePreprocessor

class PPOAgentProxy:
    def __init__(self, model_path=None):
        self.name = "PPO"
        self.model_loaded = False
        self.model_path = model_path or os.getenv("PPO_MODEL_ZIP_PATH", "models/ppo_latest.zip")
        self.model = None
        self.preprocessor = StatePreprocessor(normalize=True)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PPO 모델 파일을 찾을 수 없습니다: {self.model_path}")
        self.model = PPO.load(self.model_path, device="cpu")
        self.model_loaded = True

    def generate_signal(self, price_data):
        if not self.model_loaded:
            self.load_model()
        state = self._preprocess(price_data)
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def _preprocess(self, price_data):
        np_state = self.preprocessor.preprocess(price_data)
        return np_state  # Stable-Baselines3 모델은 numpy array 입력 허용