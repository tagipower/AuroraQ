import numpy as np
from envs.base_trading_env import BaseTradingEnv
from utils.logger import get_logger

logger = get_logger("RuleStrategyEEnv")

class RuleStrategyEEnv(BaseTradingEnv):
    def __init__(self, df, reward_calculator=None, config=None, window_size=30):
        super().__init__(df, window_size)
        self.df = df
        self.reward_calculator = reward_calculator
        self.config = config or {}
        self.position = None
        self.entry_price = None
        self.entry_step = None

    def custom_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        close = window["close"].values
        volume = window["volume"].values
        sentiment = self.df["sentiment_score"].iloc[self.current_step]
        regime = self.df["regime"].iloc[self.current_step]

        price_std = np.std(close)
        last_price = self.df["close"].iloc[self.current_step]
        recent_max = np.max(close[-10:])  # breakout 기준
        volume_ratio = self.df["volume"].iloc[self.current_step] / (np.mean(volume) + 1e-6)

        return np.array([
            (last_price - np.mean(close)) / (np.std(close) + 1e-6),  # 가격 정규화
            price_std,                # 가격 변동성
            volume_ratio,             # 거래량 급증 지표
            sentiment,                # 감정 점수
            regime,                   # 시장 정권
            (last_price - recent_max) / (recent_max + 1e-6)  # 돌파 강도
        ], dtype=np.float32)

    def custom_reward(self, action):
        current_price = self.df["close"].iloc[self.current_step]
        reward = 0.0

        if action == 1 and self.position is None:
            if self._generate_entry_signal():
                self.position = "long"
                self.entry_price = current_price
                self.entry_step = self.current_step
                logger.debug(f"[RuleE] 진입: {self.entry_price}")

        elif action == 2 and self.position == "long":
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl
            logger.debug(f"[RuleE] 강제청산: {current_price}, 보상: {reward}")
            self._close_position()

        elif action == 0 and self.position == "long":
            pnl = (current_price - self.entry_price) / self.entry_price
            time_held = self.current_step - self.entry_step

            if pnl >= 0.02 or pnl <= -0.008 or time_held >= 3:
                reward = pnl
                logger.debug(f"[RuleE] 조건청산: {current_price}, 보상: {reward}")
                self._close_position()

        return np.clip(reward * 10, -1.0, 1.0)

    def _generate_entry_signal(self):
        if self.current_step < 20:
            return False

        window = self.df.iloc[self.current_step - 20:self.current_step]
        current = self.df.iloc[self.current_step]

        price_std = np.std(window['close'])
        recent_max = max(window['close'][:-1])
        last_close = current['close']
        last_volume = current['volume']
        mean_volume = np.mean(window['volume'])
        sentiment = current.get('sentiment_score', 0.0)

        return (
            price_std < 0.002 and
            last_close > recent_max and
            last_volume > 1.2 * mean_volume and
            sentiment > 0.2
        )

    def _close_position(self):
        self.position = None
        self.entry_price = None
        self.entry_step = None
