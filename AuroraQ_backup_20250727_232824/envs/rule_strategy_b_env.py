import numpy as np
import pandas as pd
from envs.base_trading_env import BaseTradingEnv

class RuleStrategyBEnv(BaseTradingEnv):
    def __init__(self, df, reward_calculator, config=None, window_size=60):
        super().__init__(df, window_size)
        self.reward_calculator = reward_calculator
        self.config = config or {}
        self.position = None
        self.entry_price = None
        self.entry_step = None

    def custom_observation(self):
        idx = self.current_step
        window = self.df.iloc[max(0, idx - self.window_size):idx + 1]

        close = window['close'].values
        volume = window['volume'].values
        sentiment = self.df['sentiment_score'].iloc[idx]
        regime = self.df['regime'].iloc[idx]
        is_event = self.df.get('is_event', pd.Series([0] * len(self.df))).iloc[idx]

        ema60 = np.mean(close[-60:]) if len(close) >= 60 else np.mean(close)
        ema120 = np.mean(close[-120:]) if idx >= 120 else ema60
        trend_score = 1.0 if ema60 > ema120 else -1.0

        price_norm = (close[-1] - np.mean(close)) / (np.std(close) + 1e-6)
        vol_ratio = volume[-1] / (np.mean(volume) + 1e-6)
        vol_std = np.std(close[-20:]) if len(close) >= 20 else 0

        obs = np.array([
            price_norm,     # 가격 정상화
            vol_std,        # 단기 변동성
            vol_ratio,      # 거래량 비율
            sentiment,      # 감정 점수
            regime,         # 시장 상태
            is_event,       # 이벤트 여부
            trend_score     # 추세 기반 점수
        ], dtype=np.float32)
        return obs

    def custom_reward(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        if action == 1 and self.position is None:  # 롱 진입
            self.position = "long"
            self.entry_price = current_price
            self.entry_step = self.current_step

        elif action == 2 and self.position == "long":  # 익절/손절
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl
            self._close_position()

        elif action == 0 and self.position == "long":  # 유지 or 조건부 청산
            pnl = (current_price - self.entry_price) / self.entry_price
            holding_period = self.current_step - self.entry_step

            if pnl >= 0.015 or pnl <= -0.01 or holding_period >= 20:
                reward = pnl
                self._close_position()

        return np.clip(reward * 10, -1.0, 1.0)

    def custom_entry_rule(self):
        if self.position is None:
            sentiment = self.df['sentiment_score'].iloc[self.current_step]
            trend = self.df['regime'].iloc[self.current_step]
            return sentiment > 0.2 and trend != "bear"
        return False

    def custom_exit_rule(self):
        if self.position == "long":
            current_price = self.df['close'].iloc[self.current_step]
            pnl = (current_price - self.entry_price) / self.entry_price
            holding_period = self.current_step - self.entry_step
            return pnl >= 0.015 or pnl <= -0.01 or holding_period >= 20
        return False

    def _close_position(self):
        self.position = None
        self.entry_price = None
        self.entry_step = None
