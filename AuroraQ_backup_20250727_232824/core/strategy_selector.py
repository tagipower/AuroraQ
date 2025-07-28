import os
import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from collections import deque
import threading

from core.path_config import get_log_path, get_data_path, get_config_path

from strategy import (
    rule_strategy_a, rule_strategy_b, rule_strategy_c,
    rule_strategy_d, rule_strategy_e
)
from core.ppo_strategy_wrapper import PPOStrategy
from strategy.mab_selector import MABSelector
from core.strategy_score_manager import get_all_current_scores, update_strategy_metrics, get_strategy_summary
from config.trade_config_loader import load_yaml_config
from utils.logger import get_logger
from utils.optimized_sentiment_aligner import get_sentiment_aligner

logger = get_logger("StrategySelector")


class StrategySelector:
    """
    AuroraQ 메타 전략 선택기 (개선 버전)
    - Rule A~E + PPO 전략 지원
    - 감정 점수 통합 및 자동 정렬
    - 동적 전략 선택 with MAB
    - 성과 기반 자동 가중치 조정
    """

    def __init__(
        self, 
        sentiment_file: str = "data/sentiment/news_sentiment_log.csv",
        enable_ppo: bool = True,
        cache_size: int = 1000
    ):
        """
        Args:
            sentiment_file: 감정 점수 CSV 경로
            enable_ppo: PPO 전략 활성화 여부
            cache_size: 감정 점수 캐시 크기
        """
        # 전략 인스턴스 초기화
        self.strategies = {
            "RuleStrategyA": rule_strategy_a.RuleStrategyA(),
            "RuleStrategyB": rule_strategy_b.RuleStrategyB(),
            "RuleStrategyC": rule_strategy_c.RuleStrategyC(),
            "RuleStrategyD": rule_strategy_d.RuleStrategyD(),
            "RuleStrategyE": rule_strategy_e.RuleStrategyE(),
        }
        
        if enable_ppo:
            try:
                self.strategies["PPOStrategy"] = PPOStrategy()
            except Exception as e:
                logger.warning(f"PPO 전략 로드 실패: {e}")
        
        self.strategy_names = list(self.strategies.keys())
        
        # 감정 점수 관리 - 최적화된 정렬기 사용
        self.sentiment_file = sentiment_file if sentiment_file else str(get_data_path("sentiment"))
        self.sentiment_df = self._ensure_sentiment_file()
        self.sentiment_aligner = get_sentiment_aligner()
        self.sentiment_aligner.update_sentiment_data(self.sentiment_df)
        self.cache_size = cache_size
        
        # 설정 관리
        self.config_path = str(get_config_path("strategy_weight"))
        self.mab_config_path = str(get_config_path("mab_config"))
        self.last_config_mtime = None
        self.config = self._load_config()
        self.mab_config = self._load_mab_config()
        
        # MAB 선택기 초기화
        epsilon = self.mab_config.get("epsilon", 0.1)
        decay_rate = self.mab_config.get("decay_rate", 0.995)
        self.mab_selector = MABSelector(
            self.strategy_names, 
            epsilon=epsilon,
            decay_rate=decay_rate
        )
        
        # 성과 추적
        self.performance_window = deque(maxlen=100)
        self.strategy_performance = {name: deque(maxlen=50) for name in self.strategy_names}
        
        # 로깅 설정
        self._init_logging()
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
        logger.info(f"전략 선택기 초기화 완료 - 전략: {len(self.strategies)}개")

    def _init_logging(self):
        """로깅 초기화"""
        self.mab_log_path = get_log_path("mab_score")
        self.selection_log_path = get_log_path("strategy_selection")
        
        # 디렉토리는 path_config에서 자동 생성됨
        
        # MAB 로그
        if not os.path.exists(self.mab_log_path):
            with open(self.mab_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "strategy", "reward", "epsilon"])
        
        # 선택 로그
        if not os.path.exists(self.selection_log_path):
            with open(self.selection_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "strategy", "base_score", "adjusted_score",
                    "sentiment", "regime", "signal", "reason"
                ])

    def _ensure_sentiment_file(self) -> pd.DataFrame:
        """감정 점수 파일 확인 및 로드"""
        os.makedirs(os.path.dirname(self.sentiment_file), exist_ok=True)
        
        if not os.path.exists(self.sentiment_file):
            logger.warning("감정 점수 파일 없음 → 기본 데이터 생성")
            
            # 현실적인 더미 데이터 생성
            periods = 2000
            base_sentiment = 0.5
            
            # 시간에 따른 감정 변화 시뮬레이션
            timestamps = pd.date_range(
                datetime.now() - timedelta(days=7), 
                periods=periods, 
                freq="5T"
            )
            
            # 사인파 + 노이즈로 현실적인 감정 변화 생성
            t = np.linspace(0, 4 * np.pi, periods)
            sentiment_scores = base_sentiment + 0.3 * np.sin(t) + 0.1 * np.random.randn(periods)
            sentiment_scores = np.clip(sentiment_scores, 0, 1)
            
            dummy_df = pd.DataFrame({
                "timestamp": timestamps,
                "sentiment_score": sentiment_scores,
                "confidence": 0.8 + 0.2 * np.random.rand(periods),
                "label": pd.cut(sentiment_scores, 
                               bins=[0, 0.3, 0.7, 1.0], 
                               labels=["negative", "neutral", "positive"])
            })
            
            dummy_df.to_csv(self.sentiment_file, index=False)
            logger.info(f"더미 감정 데이터 생성 완료: {periods}개 레코드")
            
            return dummy_df
        
        # 기존 파일 로드
        try:
            df = pd.read_csv(self.sentiment_file, parse_dates=["timestamp"])
            logger.info(f"감정 점수 로드 완료: {len(df)}개 레코드")
            return df
        except Exception as e:
            logger.error(f"감정 점수 파일 로드 실패: {e}")
            return pd.DataFrame()

    def _align_sentiment(self, timestamps: pd.Series) -> pd.Series:
        """가격 데이터와 감정 점수 정렬 (최적화 버전)"""
        # 최적화된 정렬기 사용
        return self.sentiment_aligner.align_batch(timestamps)

    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        try:
            config = load_yaml_config(self.config_path)
            self.last_config_mtime = os.path.getmtime(self.config_path)
            
            # 기본값 설정
            config.setdefault("strategy_weights", {})
            config.setdefault("sentiment_weights", {
                "positive": 1.2,
                "neutral": 1.0,
                "negative": 0.8
            })
            config.setdefault("regime_weights", {
                "bull": 1.2,
                "bear": 0.8,
                "neutral": 1.0,
                "volatile": 0.6
            })
            config.setdefault("min_score_threshold", 0.3)
            config.setdefault("ppo_min_score_threshold", 0.5)
            
            return config
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return {}

    def _load_mab_config(self) -> Dict:
        """MAB 설정 로드"""
        try:
            config = load_yaml_config(self.mab_config_path)
            return config
        except Exception as e:
            logger.warning(f"MAB 설정 로드 실패: {e} - 기본값 사용")
            return {
                "epsilon": 0.1,
                "decay_rate": 0.995,
                "min_epsilon": 0.01
            }

    def reload_config_if_changed(self):
        """설정 파일 변경 감지 및 리로드"""
        with self.lock:  # 스레드 안전성 추가
            try:
                current_mtime = os.path.getmtime(self.config_path)
                if self.last_config_mtime and current_mtime != self.last_config_mtime:
                    logger.info("설정 파일 변경 감지 - 리로드")
                    self.config = self._load_config()
                    
                    # MAB 파라미터 업데이트
                    new_mab_config = self._load_mab_config()
                    if new_mab_config != self.mab_config:
                        self.mab_config = new_mab_config
                        self.mab_selector.epsilon = new_mab_config.get("epsilon", 0.1)
                        
            except Exception as e:
                logger.error(f"설정 리로드 실패: {e}")

    def get_sentiment_score(
        self, 
        timestamp: pd.Timestamp, 
        price_timestamps: Optional[pd.Series] = None
    ) -> float:
        """감정 점수 조회 (최적화된 정렬기 사용)"""
        try:
            if price_timestamps is not None and len(price_timestamps) > 0:
                # 전체 시계열 정렬
                aligned_scores = self._align_sentiment(price_timestamps)
                score = float(aligned_scores.iloc[-1])
            else:
                # 단일 시점 조회 - 최적화된 정렬기 사용
                score = self.sentiment_aligner.align_single(timestamp)
            
            return score
            
        except Exception as e:
            logger.error(f"감정 점수 조회 실패: {e}")
            return 0.5

    def adjust_score(
        self, 
        base_score: float, 
        sentiment_score: float, 
        regime: str, 
        strategy_name: str
    ) -> float:
        """점수 조정 (다중 요인 고려)"""
        if base_score is None or base_score <= 0:
            return 0.0
        
        # 전략별 기본 가중치
        strategy_weight = self.config.get("strategy_weights", {}).get(strategy_name, 1.0)
        
        # 감정 점수 기반 가중치
        if sentiment_score > 0.7:
            sentiment_category = "positive"
        elif sentiment_score < 0.3:
            sentiment_category = "negative"
        else:
            sentiment_category = "neutral"
        
        sentiment_weight = self.config.get("sentiment_weights", {}).get(sentiment_category, 1.0)
        
        # 시장 레짐 가중치
        regime_weight = self.config.get("regime_weights", {}).get(regime, 1.0)
        
        # 최근 성과 기반 조정
        performance_weight = self._get_performance_weight(strategy_name)
        
        # 최종 점수 계산
        adjusted_score = base_score * strategy_weight * sentiment_weight * regime_weight * performance_weight
        
        # 범위 제한
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        return round(adjusted_score, 4)

    def _get_performance_weight(self, strategy_name: str) -> float:
        """최근 성과 기반 가중치 계산"""
        with self.lock:  # 스레드 안전성 추가
            if strategy_name not in self.strategy_performance:
                return 1.0
            
            recent_performance = list(self.strategy_performance[strategy_name])
            if len(recent_performance) < 5:
                return 1.0
        
        # 최근 성과 평균
        avg_performance = np.mean(recent_performance[-10:])
        
        # 성과에 따른 가중치 (0.8 ~ 1.2)
        if avg_performance > 0.7:
            return 1.2
        elif avg_performance > 0.5:
            return 1.1
        elif avg_performance < 0.3:
            return 0.8
        else:
            return 1.0

    def select(self, price_data_window: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """전략 선택 및 실행 (메인 메서드)"""
        try:
            # 설정 리로드 확인
            self.reload_config_if_changed()
            
            # 데이터 준비
            if isinstance(price_data_window, dict):
                price_df = pd.DataFrame(price_data_window)
            else:
                price_df = price_data_window
            
            # 타임스탬프 추출
            if "timestamp" in price_df.columns:
                timestamps = pd.to_datetime(price_df["timestamp"])
            else:
                timestamps = pd.to_datetime(price_df.index)
            
            current_time = timestamps.iloc[-1] if len(timestamps) > 0 else pd.Timestamp.now()
            
            # 컨텍스트 정보 수집
            context = self._gather_context(price_df, timestamps, current_time)
            
            # 전략 선택
            selection_result = self._select_strategy(context, price_df)
            
            # 결과 로깅
            self._log_selection(selection_result, context)
            
            return selection_result
            
        except Exception as e:
            logger.error(f"전략 선택 중 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 비상 폴백
            return self._emergency_fallback(price_data_window)

    def _gather_context(
        self, 
        price_df: pd.DataFrame, 
        timestamps: pd.Series, 
        current_time: pd.Timestamp
    ) -> Dict[str, Any]:
        """컨텍스트 정보 수집"""
        # 감정 점수
        sentiment_score = self.get_sentiment_score(current_time, timestamps)
        
        # 시장 레짐
        regime = self._detect_regime(price_df)
        
        # 변동성
        volatility = self._calculate_volatility(price_df)
        
        # 추세
        trend = self._detect_trend(price_df)
        
        return {
            "timestamp": current_time,
            "sentiment_score": sentiment_score,
            "regime": regime,
            "volatility": volatility,
            "trend": trend
        }

    def _detect_regime(self, price_df: pd.DataFrame) -> str:
        """시장 레짐 감지"""
        if len(price_df) < 20:
            return "neutral"
        
        try:
            # 단기/장기 이동평균
            ma_short = price_df["close"].rolling(10).mean().iloc[-1]
            ma_long = price_df["close"].rolling(20).mean().iloc[-1]
            
            # 변동성
            volatility = price_df["close"].pct_change().std()
            
            if volatility > 0.05:
                return "volatile"
            elif ma_short > ma_long * 1.02:
                return "bull"
            elif ma_short < ma_long * 0.98:
                return "bear"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"

    def _calculate_volatility(self, price_df: pd.DataFrame) -> float:
        """변동성 계산"""
        if len(price_df) < 10:
            return 0.02
        
        try:
            returns = price_df["close"].pct_change().dropna()
            return float(returns.std())
        except Exception:
            return 0.02

    def _detect_trend(self, price_df: pd.DataFrame) -> str:
        """추세 감지"""
        if len(price_df) < 20:
            return "sideways"
        
        try:
            # 선형 회귀
            from scipy import stats
            x = np.arange(len(price_df))
            y = price_df["close"].values
            slope, _, _, _, _ = stats.linregress(x[-20:], y[-20:])
            
            if slope > 0.001:
                return "uptrend"
            elif slope < -0.001:
                return "downtrend"
            else:
                return "sideways"
                
        except Exception:
            return "sideways"

    def _select_strategy(self, context: Dict[str, Any], price_df: pd.DataFrame) -> Dict[str, Any]:
        """전략 선택 로직"""
        # MAB로 초기 선택
        chosen_name = self.mab_selector.select()
        
        # 전략 검증 및 실행
        max_attempts = len(self.strategy_names)
        attempted_strategies = set()
        
        while len(attempted_strategies) < max_attempts:
            if chosen_name in attempted_strategies:
                # 다음 전략 선택
                remaining = [s for s in self.strategy_names if s not in attempted_strategies]
                if not remaining:
                    break
                chosen_name = self._select_best_remaining(remaining, context)
            
            attempted_strategies.add(chosen_name)
            
            # 전략 실행 시도
            result = self._execute_strategy(chosen_name, context, price_df)
            
            if result["success"]:
                return result
            
            # 실패 시 MAB 업데이트
            self.mab_selector.update(chosen_name, 0.0)
            logger.warning(f"{chosen_name} 실행 실패: {result.get('error')}")
        
        # 모든 전략 실패
        return self._emergency_fallback(price_df)

    def _select_best_remaining(
        self, 
        remaining_strategies: List[str], 
        context: Dict[str, Any]
    ) -> str:
        """남은 전략 중 최선 선택"""
        scores = {}
        
        for strategy_name in remaining_strategies:
            summary = get_strategy_summary(strategy_name)
            base_score = summary.get("current_score", 0)
            adjusted_score = self.adjust_score(
                base_score,
                context["sentiment_score"],
                context["regime"],
                strategy_name
            )
            scores[strategy_name] = adjusted_score
        
        # 최고 점수 전략 선택
        best_strategy = max(scores, key=scores.get)
        return best_strategy

    def _execute_strategy(
        self, 
        strategy_name: str, 
        context: Dict[str, Any], 
        price_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """전략 실행"""
        try:
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                return {
                    "success": False,
                    "error": "전략 인스턴스 없음"
                }
            
            # 점수 계산
            all_scores = get_all_current_scores()
            base_score = all_scores.get(strategy_name, 0)
            adjusted_score = self.adjust_score(
                base_score,
                context["sentiment_score"],
                context["regime"],
                strategy_name
            )
            
            # PPO 특별 처리
            if strategy_name == "PPOStrategy":
                min_score = self.config.get("ppo_min_score_threshold", 0.5)
                if adjusted_score < min_score:
                    return {
                        "success": False,
                        "error": f"PPO 점수 부족: {adjusted_score:.3f} < {min_score}"
                    }
            
            # 최소 점수 확인
            min_score = self.config.get("min_score_threshold", 0.3)
            if adjusted_score < min_score:
                return {
                    "success": False,
                    "error": f"점수 부족: {adjusted_score:.3f} < {min_score}"
                }
            
            # 시그널 생성
            signal = strategy.generate_signal(price_df)
            
            # 시그널 검증
            if not signal or signal.get("action") == "HOLD":
                # HOLD는 실패가 아님
                pass
            
            # 성과 기록 (스레드 안전)
            with self.lock:
                self.strategy_performance[strategy_name].append(adjusted_score)
            
            # MAB 업데이트 (스레드 안전)
            reward = adjusted_score
            with self.lock:
                self.mab_selector.update(strategy_name, reward)
            self._log_mab_score(strategy_name, reward)
            
            # 메트릭 업데이트
            update_strategy_metrics(
                strategy_name, 
                {
                    "reward_shaping_score": reward,
                    "sentiment_score": context["sentiment_score"]
                }
            )
            
            return {
                "success": True,
                "strategy": strategy_name,
                "strategy_object": strategy,
                "signal": signal,
                "score": adjusted_score,
                "base_score": base_score,
                "sentiment_score": context["sentiment_score"],
                "regime": context["regime"],
                "volatility": context["volatility"],
                "trend": context["trend"]
            }
            
        except Exception as e:
            logger.error(f"{strategy_name} 실행 중 오류: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _emergency_fallback(self, price_data: Any) -> Dict[str, Any]:
        """비상 폴백 전략"""
        logger.error("모든 전략 실패 - 비상 폴백 사용")
        
        fallback_strategy = self.strategies.get("RuleStrategyA")
        
        return {
            "strategy": "RuleStrategyA",
            "strategy_object": fallback_strategy,
            "signal": {"action": "HOLD", "price": 0},
            "score": 0.0,
            "sentiment_score": 0.5,
            "regime": "neutral",
            "error": "Emergency fallback"
        }

    def _log_selection(self, result: Dict[str, Any], context: Dict[str, Any]):
        """전략 선택 로깅"""
        try:
            with open(self.selection_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    context["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    result.get("strategy", "Unknown"),
                    result.get("base_score", 0),
                    result.get("score", 0),
                    context["sentiment_score"],
                    context["regime"],
                    result.get("signal", {}).get("action", "HOLD"),
                    result.get("error", "")
                ])
        except Exception as e:
            logger.error(f"선택 로깅 실패: {e}")

    def _log_mab_score(self, strategy_name: str, reward: float):
        """MAB 점수 로깅"""
        try:
            with open(self.mab_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    strategy_name,
                    reward,
                    self.mab_selector.epsilon
                ])
        except Exception as e:
            logger.error(f"MAB 로깅 실패: {e}")

    def get_strategy_stats(self) -> Dict[str, Any]:
        """전략 통계 조회"""
        stats = {}
        
        for strategy_name in self.strategy_names:
            summary = get_strategy_summary(strategy_name)
            performance = list(self.strategy_performance[strategy_name])
            
            stats[strategy_name] = {
                "current_score": summary.get("current_score", 0),
                "average_score": summary.get("average_score", 0),
                "recent_performance": np.mean(performance[-10:]) if performance else 0,
                "selection_count": self.mab_selector.counts.get(strategy_name, 0),
                "performance_metrics": summary.get("performance", {})
            }
        
        return stats

    def reset_performance_tracking(self):
        """성과 추적 초기화"""
        with self.lock:  # 스레드 안전성 추가
            self.performance_window.clear()
            for deque_obj in self.strategy_performance.values():
                deque_obj.clear()
            logger.info("성과 추적 데이터 초기화 완료")