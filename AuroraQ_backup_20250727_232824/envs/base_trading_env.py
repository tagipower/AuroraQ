import gym
import numpy as np
import pandas as pd
from gym import spaces

class BaseTradingEnv(gym.Env):
    def __init__(self, df, reward_calculator, window_size=30, config=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.reward_calculator = reward_calculator
        self.window_size = window_size
        self.current_step = self.window_size
        self.position = None
        self.entry_price = None
        self.entry_step = None
        self.config = config or {}

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Long, 2 = Short
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self._get_observation_dim()), dtype=np.float32
        )

    def reset(self):
        self.current_step = self.window_size
        self.position = None
        self.entry_price = None
        self.entry_step = None
        return self._get_observation()

    def step(self, action):
        reward = 0
        done = False
        info = {}

        price = self.df.loc[self.current_step, 'close']

        if self._check_exit_condition(action):
            reward += self._calculate_reward(price)
            self.position = None
            self.entry_price = None
            self.entry_step = None

        elif self._check_entry_condition(action):
            self.position = 'long' if action == 1 else 'short'
            self.entry_price = price
            self.entry_step = self.current_step

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        window_df = self.df.iloc[self.current_step - self.window_size : self.current_step]
        if hasattr(self, "custom_observation"):
            return self.custom_observation(window_df)
        return window_df[['close', 'high', 'low', 'volume', 'sentiment_score']].values

    def _check_entry_condition(self, action):
        if hasattr(self, "custom_entry_condition"):
            return self.custom_entry_condition(action)
        return self.position is None and action in [1, 2]

    def _check_exit_condition(self, action):
        if hasattr(self, "custom_exit_condition"):
            return self.custom_exit_condition(action)
        return (
            (self.position == 'long' and action == 2) or
            (self.position == 'short' and action == 1)
        )

    def _calculate_reward(self, price):
        if hasattr(self, "custom_reward"):
            return self.custom_reward(price)

        if self.position == 'long':
            pnl = price - self.entry_price
        elif self.position == 'short':
            pnl = self.entry_price - price
        else:
            pnl = 0

        return self.reward_calculator(pnl)

    def render(self, mode='human'):
        pass
