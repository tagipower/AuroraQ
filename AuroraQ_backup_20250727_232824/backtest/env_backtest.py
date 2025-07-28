# backtest/env_backtest.py

import numpy as np
import gym
from gym import spaces
import pandas as pd

class SentimentTradingEnvBacktest(gym.Env):
    def __init__(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame,
                 strategy=None, reward_fn=None):
        super().__init__()
        # 🔹 price 컬럼 없으면 close로 대체
        if "price" not in price_data.columns:
            if "close" in price_data.columns:
                price_data = price_data.copy()
                price_data["price"] = price_data["close"]
            else:
                raise KeyError("price_data에 'price' 또는 'close' 컬럼이 필요합니다.")

        self.price_data = price_data.reset_index(drop=True)
        self.sentiment_data = sentiment_data.reset_index(drop=True)
        self.strategy = strategy
        self.reward_fn = reward_fn

        # PPO 학습 및 전략 평가를 위한 action/observation space 정의
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.current_step = 0
        self.done = False
        self.entry_count = 0
        self.logs = []

    def reset(self):
        self.current_step = 0
        self.done = False
        self.entry_count = 0
        self.logs.clear()
        return self._get_observation()

    def step(self, action):
        reward = 0.0

        if self.reward_fn:
            reward = self.reward_fn(
                price=self.price_data.loc[self.current_step, "price"],
                sentiment=self._get_sentiment(),
                action=action
            )

        # 로그 기록
        self.logs.append({
            "step": self.current_step,
            "price": self.price_data.loc[self.current_step, "price"],
            "sentiment": self._get_sentiment(),
            "action": action,
            "reward": reward
        })

        self.entry_count += int(action != 0)  # 매수/매도 시 카운트

        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        price = self.price_data.loc[self.current_step, "price"]
        sentiment = self._get_sentiment()
        return np.array([price, sentiment], dtype=np.float32)

    def _get_sentiment(self):
        if self.current_step < len(self.sentiment_data):
            return self.sentiment_data.loc[self.current_step, "sentiment_score"]
        return 0.0

    def get_entry_count(self):
        return self.entry_count

    def get_log_data(self):
        return self.logs
