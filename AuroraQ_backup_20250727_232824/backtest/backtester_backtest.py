# backtest/backtester_backtest.py

import pandas as pd
from backtest.env_backtest import SentimentTradingEnvBacktest
from backtest.model_loader_backtest import load_ppo_model as load_model
from backtest.reward_backtest import RewardCalculator
from stable_baselines3 import PPO

from strategy import (
    rule_strategy_a,
    rule_strategy_b,
    rule_strategy_c,
    rule_strategy_d,
    rule_strategy_e
)

class BacktesterBacktest:
    def __init__(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame, strategy=None, model_path: str = None):
        # 🔹 price 없으면 close로 대체
        if "price" not in price_data.columns:
            if "close" in price_data.columns:
                price_data = price_data.copy()
                price_data["price"] = price_data["close"]
            else:
                raise KeyError("price_data에 'price' 또는 'close' 컬럼이 필요합니다.")

        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.strategy = strategy
        self.model_path = model_path
        self.reward_calculator = RewardCalculator()  # 호출 가능 객체

    def run(self):
        if isinstance(self.strategy, PPO):
            return self._run_ppo_model()
        elif self.strategy:
            return self._run_custom_strategy()
        elif self.model_path:
            return self._run_ppo()
        else:
            raise ValueError("전략 객체 또는 PPO 모델 경로가 필요합니다.")

    def _run_custom_strategy(self):
        env = SentimentTradingEnvBacktest(
            price_data=self.price_data,
            sentiment_data=self.sentiment_data,
            strategy=self.strategy,
            reward_fn=self.reward_calculator
        )

        obs = env.reset()
        done = False
        rewards = []

        while not done:
            if hasattr(self.strategy, "decide"):
                action = self.strategy.decide(price=obs[0], sentiment=obs[1], timestamp=env.current_step)
            else:
                action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        return self._calculate_metrics(rewards, env, name=self.strategy.__class__.__name__)

    def _run_ppo_model(self):
        env = SentimentTradingEnvBacktest(
            price_data=self.price_data,
            sentiment_data=self.sentiment_data,
            strategy=None,
            reward_fn=self.reward_calculator
        )

        obs = env.reset()
        done = False
        rewards = []

        while not done:
            action, _ = self.strategy.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        return self._calculate_metrics(rewards, env, name="PPO")

    def _run_ppo(self):
        model = load_model(self.model_path)
        env = SentimentTradingEnvBacktest(
            price_data=self.price_data,
            sentiment_data=self.sentiment_data,
            strategy=None,
            reward_fn=self.reward_calculator
        )

        obs = env.reset()
        done = False
        rewards = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        return self._calculate_metrics(rewards, env, name="PPO")

    def _calculate_metrics(self, rewards, env, name="Strategy"):
        rewards_series = pd.Series(rewards)
        total_return = rewards_series.sum() if not rewards_series.empty else 0
        trades = env.get_entry_count()
        win_rate = (
            (sum(r for r in rewards if r > 0) / (abs(sum(rewards)) + 1e-6)) * 100
            if rewards else 0
        )
        mdd = rewards_series.min() if not rewards_series.empty else 0
        sharpe = (
            (rewards_series.mean() / (rewards_series.std() + 1e-6)) * (252 ** 0.5)
            if not rewards_series.empty else 0
        )

        return {
            "strategy": name,
            "total_return": round(total_return, 4),
            "mdd": round(mdd, 4),
            "sharpe": round(sharpe, 4),
            "trades": trades,
            "win_rate": round(win_rate, 2),
            "logs": env.get_log_data()
        }

def run_backtest_for_scenario(price_data, sentiment_data, strategy=None, model_path: str = None):
    backtester = BacktesterBacktest(price_data, sentiment_data, strategy, model_path)
    return backtester.run()
