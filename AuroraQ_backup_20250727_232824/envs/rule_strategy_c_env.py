import numpy as np
import pandas as pd
from envs.base_trading_env import BaseTradingEnv

class RuleStrategyCEnv(BaseTradingEnv):
    def __init__(self, df, reward_calculator=None, config=None, window_size=30):
        super().__init__(df, window_size)
        self.reward_calculator = reward_calculator
        self.config = config or {}
        self.position = None
        self.entry_price = None
        self.entry_step = None

    def custom_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]

        close = window['close'].values
        high = window['high'].values
        low = window['low'].values
        sentiment = self.df['sentiment'].iloc[self.current_step]
        prev_sentiment = self.df['sentiment'].iloc[self.current_step - 1] if self.current_step > 0 else sentiment
        regime = self.df['regime'].iloc[self.current_step]
        is_event = self.df.get('is_event', pd.Series(0, index=self.df.index)).iloc[self.current_step]

        # 📈 특징 추출
        normalized_close = (close[-1] - np.mean(close)) / (np.std(close) + 1e-6)
        price_deviation = (close[-1] - np.max(high)) / (np.max(high) + 1e-6)
        drawdown_expectation = np.mean(np.diff(close)[-5:])
        recent_volatility = np.std(close[-5:]) / (np.mean(close) + 1e-6)
        sentiment_change = sentiment - prev_sentiment

        return np.array([
            normalized_close,
            price_deviation,
            drawdown_expectation,
            recent_volatility,
            sentiment_change,
            regime,
            is_event
        ], dtype=np.float32)

    def custom_reward(self, action):
        reward = 0.0
        current_price = self.df['close'].iloc[self.current_step]

        if action == 1 and self.position is None:
            self.position = "short"
            self.entry_price = current_price
            self.entry_step = self.current_step

        elif action == 2 and self.position == "short":
            pnl = (self.entry_price - current_price) / self.entry_price
            reward = pnl
            self._close_position()

        elif action == 0 and self.position == "short":
            pnl = (self.entry_price - current_price) / self.entry_price
            holding = self.current_step - self.entry_step

            close = self.df['close'].iloc[max(0, self.current_step - 30):self.current_step + 1].values
            sentiment = self.df['sentiment'].iloc[self.current_step]
            prev_sentiment = self.df['sentiment'].iloc[self.current_step - 1] if self.current_step > 0 else sentiment
            sentiment_jump = sentiment - prev_sentiment
            volatility_jump = np.std(close[-5:]) > 1.8 * np.std(close)
            bullish_candles = sum(close[-i] > close[-i - 1] for i in range(1, 4))

            if (
                pnl >= 0.012 or
                pnl <= -0.012 or
                holding >= 18 or
                volatility_jump or
                sentiment_jump > 0.25 or
                bullish_candles >= 2
            ):
                reward = pnl
                self._close_position()

        return np.clip(reward * 10, -1.0, 1.0)

    def _close_position(self):
        self.position = None
        self.entry_price = None
        self.entry_step = None
