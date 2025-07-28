import numpy as np
from envs.base_trading_env import BaseTradingEnv
from utils.logger import get_logger
from strategy.rule_strategy_d import RuleStrategyD

logger = get_logger("RuleStrategyDEnv")

class RuleStrategyDEnv(BaseTradingEnv):
    def __init__(self, price_data, window_size=30):
        super().__init__(price_data, window_size)
        self.strategy = RuleStrategyD()
        self.position = None
        self.done = False
        self.reward = 0

    def reset(self):
        self.current_step = self.window_size
        self.position = None
        self.done = False
        self.reward = 0
        self.strategy.reset()
        return self._get_observation()

    def step(self, action):
        current_data = self._get_price_data()
        signal = self.strategy.generate_signal(current_data)

        if signal and action == 1 and not self.position:
            self.position = signal
            logger.debug(f"[EnvD] ✅ 진입: {self.position}")

        elif self.position:
            exit_type = self.strategy.check_exit(self.position, current_data)
            if exit_type:
                self.reward = self._calculate_reward(exit_type)
                self.done = True
                logger.debug(f"[EnvD] 💰 청산: {exit_type}, 보상: {self.reward}")

        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            self.done = True

        return self._get_observation(), self.reward, self.done, {}

    def _calculate_reward(self, exit_type):
        price_now = self.price_data["close"][self.current_step]
        price_entry = self.position["price"]
        pnl = (price_now - price_entry) / price_entry

        if exit_type == "take_profit":
            return pnl * 100
        elif exit_type == "stop_loss":
            return pnl * 100
        else:
            return pnl * 100 * 0.5  # time_exit 패널티

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step

        close = self.price_data["close"][start:end]
        volume = self.price_data["volume"][start:end]
        sentiment = self.price_data["sentiment"][end - 1]
        regime = self.price_data.get("regime", ["neutral"] * len(close))[end - 1]
        adx = self.price_data.get("adx", [20] * len(close))[end - 1]

        bb_mean = np.mean(close)
        bb_std = np.std(close)
        bb_lower = bb_mean - 2 * bb_std
        bb_break = int(close[-1] < bb_lower)

        obs = np.array([
            close[-1],
            np.mean(volume),
            sentiment,
            1 if regime == "bull" else (-1 if regime == "bear" else 0),
            adx,
            bb_break
        ], dtype=np.float32)

        return obs
