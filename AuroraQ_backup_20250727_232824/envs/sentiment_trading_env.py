import gym
import numpy as np
from gym import spaces
from utils.reward_calculator import RewardCalculator
from sentiment.sentiment_router import SentimentScoreRouter

class SentimentTradingEnv(gym.Env):
    def __init__(self, data, window_size=10, reward_calculator=None, config=None):
        super(SentimentTradingEnv, self).__init__()

        self.window_size = window_size
        self.data = data
        self.price_data = data['price']
        self.sentiment_data = data['sentiment']
        self.trend_data = data['trend']
        self.event_flags = data['event_flag']
        self.regime_data = data.get('regime', ["neutral"] * len(self.price_data))
        self.timestamp_data = data.get('timestamp', [None] * len(self.price_data))
        self.news_data = data.get('news_text', [""] * len(self.price_data))
        self.config = config or {}

        # 보상 계산기와 감정 점수 라우터
        self.reward_calculator = reward_calculator or RewardCalculator(
            self.config.get("reward", {}),
            sentiment_mode="backtest"
        )
        self.sentiment_router = SentimentScoreRouter(mode="backtest")
        self.log_reward_components = self.config.get("log_reward", False)

        self.current_step = self.window_size
        self.done = False
        self.position = None
        self.entry_price = 0.0

        # 관측 공간: [가격, 감정 점수, 추세]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: 관망, 1: 매수, 2: 매도

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.position = None
        self.entry_price = 0.0
        self.reward_calculator.reset()
        return self._get_observation()

    def _get_observation(self):
            price = self.price_data[self.current_step]
            sentiment = self.sentiment_data[self.current_step]["sentiment_score"]
            confidence = self.sentiment_data[self.current_step]["confidence"]
            scenario = self._encode_scenario(self.sentiment_data[self.current_step]["scenario_tag"])
            trend = 1.0 if self.trend_data[self.current_step] == "up" else -1.0 if self.trend_data[self.current_step] == "down" else 0.0
            return np.array([price, sentiment, confidence, scenario, trend], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Please reset the environment.")

        current_price = self.price_data[self.current_step]
        sentiment_score = self.sentiment_data[self.current_step]
        sentiment_delta = self.sentiment_data[self.current_step] - self.sentiment_data[self.current_step - 1]
        regime = self.regime_data[self.current_step]
        long_term_trend = self.trend_data[self.current_step]
        event_risk = self.event_flags[self.current_step]
        timestamp = self.timestamp_data[self.current_step]
        news_text = self.news_data[self.current_step]

        reward = self.reward_calculator.calculate_reward(
            current_price=current_price,
            timestamp=timestamp,
            event_flag=event_risk,
            regime_state=regime,
            long_term_trend=long_term_trend,
            action=action,
            news_text=news_text
        )

        # 선택적 로그 기록
        if self.log_reward_components:
            self.reward_calculator.log_reward(self.current_step)

        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            self.done = True

        next_observation = self._get_observation()
        return next_observation, reward, self.done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.reward_calculator.position}")
