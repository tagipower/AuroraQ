# reward/reward_calculator.py

import numpy as np
import os
import datetime
from sentiment.sentiment_router import SentimentScoreRouter
from core.score_feedback import apply_reward_to_score
from utils.logger import get_logger
from utils.reward_config_loader import load_reward_config
from utils.reward_shaping_loader import load_reward_shaping_config

logger = get_logger("RewardCalculator")


class RewardCalculator:
    def __init__(self, sentiment_mode="backtest"):
        self.config = load_reward_config()
        self.shaping = load_reward_shaping_config()

        self.prev_price = None
        self.entry_price = None
        self.position = None
        self.holding_period = 0
        self.total_trades = 0
        self.reward_log = {}

        self.transaction_fee = self.config.get("fee_market_order", 0.0004) \
            if self.config.get("fee_type") == "market" else self.config.get("fee_limit_order", 0.0002)

        self.enable_logging = self.config.get("enable_reward_logging", False)
        self.log_path = self.config.get("reward_log_path", "logs/reward_log.csv")

        if self.enable_logging and not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,reward,base_reward,sentiment_bonus,sentiment_risk_penalty,event_penalty,"
                        "regime_penalty,trend_penalty,alignment_bonus,idle_penalty,over_hold_penalty,freq_penalty\n")

        self.sentiment_router = SentimentScoreRouter(mode=sentiment_mode)

    def reset(self):
        self.prev_price = None
        self.entry_price = None
        self.position = None
        self.holding_period = 0
        self.total_trades = 0
        self.reward_log = {}

    def calculate_reward(self, current_price, timestamp, event_flag, regime_state, long_term_trend,
                         action, news_text=None, sentiment_score=None, sentiment_delta=None, is_real_trade=False):

        reward = 0.0
        log = {key: 0.0 for key in [
            "base_reward", "sentiment_bonus", "sentiment_risk_penalty", "event_penalty", "regime_penalty",
            "trend_penalty", "alignment_bonus", "idle_penalty", "over_hold_penalty", "freq_penalty"
        ]}

        # ✅ 감정 점수 자동 추출 (다중 소스 통합)
        if sentiment_score is None:
            sentiment_score = self.sentiment_router.get_score(news_text=news_text, timestamp=timestamp)
        if sentiment_delta is None:
            sentiment_delta = self.sentiment_router.get_delta(timestamp)

        # 진입
        if action == 1:
            self.position = "long"
            self.entry_price = current_price
            self.holding_period = 0
            self.total_trades += 1
        elif action == 2:
            self.position = "short"
            self.entry_price = current_price
            self.holding_period = 0
            self.total_trades += 1

        # 청산
        elif action == 3 and self.position:
            price_change = (current_price - self.entry_price) / self.entry_price
            if self.position == "short":
                price_change *= -1
            pnl = price_change - self.transaction_fee
            reward += pnl
            log["base_reward"] = pnl

            if self.holding_period < self.config.get("min_hold_threshold", 2):
                penalty = self.config.get("too_short_holding_penalty", 0.5)
                reward -= penalty
                log["idle_penalty"] = -penalty

            self.position = None
            self.holding_period = 0

        # 보유 중
        elif self.position:
            self.holding_period += 1
            penalty = self.config.get("over_hold_penalty", 0.001) * self.holding_period
            reward -= penalty
            log["over_hold_penalty"] = -penalty

        # ✅ reward_schema.yaml 기반 shaping
        shaping_cfg = self.shaping["reward_shaping"]
        weights = self.shaping["weights"]
        thresholds = self.shaping["thresholds"]

        if shaping_cfg.get("use_sentiment_delta", False):
            reward += weights["sentiment_delta"] * sentiment_delta
            log["sentiment_bonus"] += weights["sentiment_delta"] * sentiment_delta

        if shaping_cfg.get("use_position_duration", False):
            norm_dur = min(self.holding_period / 100, 1.0)
            reward += weights["position_duration"] * norm_dur
            log["sentiment_bonus"] += weights["position_duration"] * norm_dur

        if shaping_cfg.get("use_normalized_profit", False) and self.entry_price:
            raw_profit = (current_price - self.entry_price) / self.entry_price
            norm_profit = max(min((raw_profit - thresholds["min_profit"]) / (0.1 - thresholds["min_profit"]), 1.0), -1.0)
            reward += weights["normalized_profit"] * norm_profit
            log["sentiment_bonus"] += weights["normalized_profit"] * norm_profit

        # ✅ 정책성 penalty
        if abs(sentiment_delta) > self.config.get("sentiment_delta_threshold", 0.7):
            penalty = self.config.get("sentiment_risk_penalty", 0.3)
            reward -= penalty
            log["sentiment_risk_penalty"] = -penalty

        if event_flag:
            penalty = self.config.get("event_penalty", 0.2)
            reward -= penalty
            log["event_penalty"] = -penalty

        if regime_state == "volatile":
            penalty = self.config.get("regime_penalty", 0.3)
            reward -= penalty
            log["regime_penalty"] = -penalty

        if self.position == "long" and long_term_trend == "down":
            penalty = self.config.get("trend_penalty", 0.2)
            reward -= penalty
            log["trend_penalty"] = -penalty
        elif self.position == "short" and long_term_trend == "up":
            penalty = self.config.get("trend_penalty", 0.2)
            reward -= penalty
            log["trend_penalty"] = -penalty

        align_bonus = 0.0
        if self.position == "long" and sentiment_score > 0:
            align_bonus += self.config.get("position_sentiment_alignment_bonus", 0.2)
        elif self.position == "short" and sentiment_score < 0:
            align_bonus += self.config.get("position_sentiment_alignment_bonus", 0.2)

        if self.position == "long" and long_term_trend == "up":
            align_bonus += self.config.get("position_trend_alignment_bonus", 0.2)
        elif self.position == "short" and long_term_trend == "down":
            align_bonus += self.config.get("position_trend_alignment_bonus", 0.2)

        reward += align_bonus
        log["alignment_bonus"] = align_bonus

        if self.total_trades > self.config.get("max_trade_frequency", 15):
            penalty = self.config.get("freq_penalty", 0.5)
            reward -= penalty
            log["freq_penalty"] = -penalty

        self.reward_log = log

        # ✅ CSV 로그 저장
        if self.enable_logging:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp_str},{reward}," + ",".join(str(log[k]) for k in log) + "\n")

        # ✅ PPO 전략 점수 자동 반영
        if is_real_trade:
            try:
                apply_reward_to_score("PPOStrategy", reward)
                logger.info(f"[RewardCalculator] ✅ PPO 점수 자동 반영: {reward:.4f}")
            except Exception as e:
                logger.warning(f"[RewardCalculator] ⚠️ PPO 점수 반영 실패: {e}")

        return round(reward, 6)

    def get_reward_log(self):
        return self.reward_log
