import os
import time
import csv
import pandas as pd
from datetime import datetime
from strategy import (
    rule_strategy_a, rule_strategy_b, rule_strategy_c,
    rule_strategy_d, rule_strategy_e
)
from strategy.mab_selector import MABSelector
from core.strategy_score_manager import get_all_current_scores, update_strategy_metrics
from config.trade_config_loader import load_yaml_config
from utils.logger import get_logger

logger = get_logger("StrategySelectorRuleOnly")


class StrategySelectorRuleOnly:
    """
    AuroraQ 메타 전략 선택기 (Rule 전략 전용)
    - Rule A~E만 사용 (PPO 제거)
    - 감정 점수와 레짐 기반 가중치 없음 (단순 점수 기반)
    - Sharpe/ROI/승률/보상 등 score 기반 MAB 선택
    """

    def __init__(self):
        # Rule 기반 전략만 초기화
        self.strategies = {
            "RuleStrategyA": rule_strategy_a.RuleStrategyA(),
            "RuleStrategyB": rule_strategy_b.RuleStrategyB(),
            "RuleStrategyC": rule_strategy_c.RuleStrategyC(),
            "RuleStrategyD": rule_strategy_d.RuleStrategyD(),
            "RuleStrategyE": rule_strategy_e.RuleStrategyE(),
        }
        self.strategy_names = list(self.strategies.keys())

        # 설정 로드
        self.config_path = "config/strategy_weight.yaml"
        self.mab_config_path = "config/mab_config.yaml"
        self.last_mtime = None
        self.config = self._load_config()
        self.mab_config = self._load_mab_config()

        epsilon = self.mab_config.get("epsilon", 0.1)
        self.mab_selector = MABSelector(self.strategy_names, epsilon=epsilon)

        # 로그 초기화
        self.mab_log_path = "logs/mab_score_log.csv"
        os.makedirs(os.path.dirname(self.mab_log_path), exist_ok=True)
        if not os.path.exists(self.mab_log_path):
            with open(self.mab_log_path, mode="w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "strategy", "reward"])

    def _load_config(self):
        self.last_mtime = os.path.getmtime(self.config_path)
        return load_yaml_config(self.config_path)

    def _load_mab_config(self):
        try:
            return load_yaml_config(self.mab_config_path)
        except Exception as e:
            logger.warning(f"⚠️ mab_config.yaml 로드 실패: {e} → 기본값 사용")
            return {"epsilon": 0.1}

    def reload_config_if_changed(self):
        try:
            current_mtime = os.path.getmtime(self.config_path)
            if self.last_mtime != current_mtime:
                logger.info("🔄 strategy_weight.yaml 변경 감지 → config reload")
                self.config = self._load_config()
        except Exception as e:
            logger.warning(f"⚠️ config reload 실패: {e}")

    def select(self, price_data_window: dict) -> dict:
        """단순 MAB 기반 Rule 전략 선택 및 실행"""
        try:
            self.reload_config_if_changed()

            # MAB로 전략 선택
            chosen_name = self.mab_selector.select()
            strategy = self.strategies.get(chosen_name)
            if not strategy:
                raise ValueError(f"전략 인스턴스 로드 실패: {chosen_name}")

            # 점수 불러오기 (Sharpe, ROI 등 포함)
            all_scores = get_all_current_scores()
            base_score = all_scores.get(chosen_name, 0.0)

            # 시그널 생성
            try:
                signal = strategy.generate_signal(price_data_window)
            except Exception as e:
                logger.warning(f"⚠️ {chosen_name} 시그널 생성 실패: {e}")
                signal = None

            # 점수 업데이트
            reward = base_score
            update_strategy_metrics(chosen_name, {"reward_shaping_score": reward})
            self.mab_selector.update(chosen_name, reward)
            self._log_mab_score(chosen_name, reward)

            return {
                "strategy": chosen_name,
                "strategy_object": strategy,
                "signal": signal,
                "score": base_score
            }

        except Exception as e:
            logger.error(f"[SelectorRuleOnly] 전략 선택 실패: {e}")
            raise

    def _log_mab_score(self, strategy_name, reward):
        try:
            with open(self.mab_log_path, mode="a", newline="") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), strategy_name, reward])
        except Exception as e:
            logger.warning(f"⚠️ MAB 로그 저장 실패: {e}")
