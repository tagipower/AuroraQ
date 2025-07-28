import numpy as np
from envs.base_trading_env import BaseTradingEnv
import ta.trend
import ta.volatility

class RuleStrategyAEnv(BaseTradingEnv):
    def __init__(self, df, reward_calculator=None, config=None, window_size=20):
        super().__init__(df, reward_calculator, window_size, config)
        self.position = None
        self.entry_price = None

    def custom_observation(self, window_df):
        close_prices = window_df['close'].values
        high_prices = window_df['high'].values
        low_prices = window_df['low'].values

        normalized_close = (close_prices[-1] - np.mean(close_prices)) / (np.std(close_prices) + 1e-6)
        ema_short = np.mean(close_prices[-5:])
        ema_long = np.mean(close_prices)
        ema_gap = (ema_short - ema_long) / (ema_long + 1e-6)
        bb_width = (np.std(close_prices) * 4) / (np.mean(close_prices) + 1e-6)

        # 💡 ADX 계산
        try:
            adx = ta.trend.ADXIndicator(high=high_prices, low=low_prices, close=close_prices, window=14).adx()
            adx_value = adx.iloc[-1] / 100  # 정규화
        except:
            adx_value = 0.0

        # 💡 볼린저 밴드 하단 돌파 여부 (이진값)
        try:
            bb = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2)
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_break = float(close_prices[-1] < bb_lower)
        except:
            bb_break = 0.0

        # 감정 점수, 레짐, 이벤트
        sentiment = self.df['sentiment'].iloc[self.current_step]
        regime_score = self.df['regime_score'].iloc[self.current_step]
        event_flag = self.df['event_flag'].iloc[self.current_step]

        return np.array([
            normalized_close,
            ema_gap,
            bb_width,
            adx_value,
            bb_break,
            sentiment,
            regime_score,
            event_flag
        ], dtype=np.float32)

    def custom_entry_condition(self, action):
        if self.position is not None or action != 1:
            return False

        window_df = self.df.iloc[self.current_step - self.window_size:self.current_step]
        close_prices = window_df['close'].values
        ema_short = np.mean(close_prices[-5:])
        ema_long = np.mean(close_prices)
        if ema_short <= ema_long:
            return False

        if self.df['event_flag'].iloc[self.current_step] == 1:
            return False

        return True

    def custom_exit_condition(self, action):
        if self.position is None:
            return False
        if action == 2:
            return True
        if self.df['event_flag'].iloc[self.current_step] == 1:
            return True
        return False

    def custom_reward(self, price):
        if self.position is None or self.entry_price is None:
            return 0.0
        pnl = (price - self.entry_price) / self.entry_price
        return np.clip(pnl * 10, -1.0, 1.0)
