"""
Enhanced BaseRuleStrategy with advanced caching and filtering
향상된 캐싱과 필터링이 적용된 기본 룰 전략 클래스
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from core.standardized_metrics import StandardizedMetrics, MetricResult, TradeRecord
from core.strategy_filter_manager import get_filter_manager, FilterCheckResult
from utils.enhanced_indicator_cache import get_enhanced_cache
from utils.logger import get_logger

logger = get_logger("EnhancedBaseRuleStrategy")


@dataclass
class EnhancedPosition:
    """향상된 포지션 정보"""
    side: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    quantity: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_reason: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 성과 추적
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    
    @property
    def holding_time(self) -> timedelta:
        """보유 시간 계산"""
        return datetime.now() - self.entry_time
    
    @property
    def current_pnl_pct(self) -> float:
        """현재 PnL % (현재가 필요)"""
        # 실제 구현에서는 현재가를 받아서 계산
        return 0.0
    
    def update_extremes(self, current_price: float):
        """최고/최저가 업데이트"""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_reason": self.entry_reason,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price
        }


@dataclass
class EnhancedSignal:
    """향상된 시그널 정보"""
    action: str  # BUY, SELL, HOLD
    price: float
    confidence: float = 0.0
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    filters_passed: List[FilterCheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "action": self.action,
            "price": self.price,
            "confidence": self.confidence,
            "reason": self.reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "metadata": self.metadata,
            "filters_passed": len(self.filters_passed),
            "filter_score": sum(f.score for f in self.filters_passed) / len(self.filters_passed) if self.filters_passed else 0.0
        }


class BaseRuleStrategy(ABC):
    """
    향상된 베이스 룰 전략 클래스
    - 강화된 지표 캐싱
    - 통합 필터 시스템
    - 표준화된 메트릭
    - 성과 추적 개선
    """
    
    def __init__(
        self, 
        name: str = "UnnamedStrategy", 
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.config = config or self._default_config()
        self.position: Optional[EnhancedPosition] = None
        self.trades: List[TradeRecord] = []
        
        # 향상된 구성 요소
        self.cache = get_enhanced_cache()
        self.filter_manager = get_filter_manager()
        self.metrics_calculator = StandardizedMetrics()
        
        # 성과 추적
        self.current_metrics: Optional[MetricResult] = None
        self.performance_history: List[Dict[str, Any]] = []
        
        # 상태 추적
        self.last_signal_time = None
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        
        # 캐시 설정
        self.indicator_cache_ttl = 300  # 5분
        
        logger.info(f"{self.name} 향상된 전략 초기화 완료")
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정값"""
        return {
            "max_position_size": 1.0,
            "default_stop_loss": 0.025,  # 2.5%
            "default_take_profit": 0.05,  # 5%
            "max_holding_days": 30,
            "max_daily_trades": 8,  # 줄어든 한도
            "max_consecutive_losses": 3,  # 더 엄격
            "min_confidence": 0.6,  # 더 높은 신뢰도 요구
            "enable_trailing_stop": True,
            "trailing_stop_distance": 0.02,  # 2%
            "enable_enhanced_filters": True,
            "cache_indicators": True
        }
    
    # ========== 캐시된 지표 계산 ==========
    
    def get_cached_indicator(self,
                           indicator: str,
                           data: pd.DataFrame,
                           **params) -> Any:
        """
        캐시된 지표 조회/계산
        
        Args:
            indicator: 지표 이름 (ema, rsi, macd 등)
            data: 가격 데이터
            **params: 지표 파라미터
            
        Returns:
            계산된 지표 값
        """
        if not self.config.get("cache_indicators", True):
            # 캐시 비활성화 시 직접 계산
            return self._calculate_indicator_direct(indicator, data, **params)
        
        return self.cache.get(
            indicator=indicator,
            data=data,
            params=params,
            ttl=self.indicator_cache_ttl
        )
    
    def _calculate_indicator_direct(self, 
                                  indicator: str,
                                  data: pd.DataFrame,
                                  **params) -> Any:
        """직접 지표 계산 (캐시 우회)"""
        # 이 메서드는 하위 클래스에서 필요시 오버라이드
        logger.warning(f"직접 계산 요청: {indicator} (캐시 우회)")
        return None
    
    # ========== 향상된 필터 시스템 ==========
    
    def check_enhanced_filters(self,
                             price_data: pd.DataFrame,
                             indicators: Dict[str, Any],
                             sentiment_score: float = 0.5) -> Tuple[bool, float, List[FilterCheckResult]]:
        """
        향상된 필터 체크
        
        Returns:
            (통과 여부, 종합 점수, 필터 결과 리스트)
        """
        if not self.config.get("enable_enhanced_filters", True):
            # 필터 비활성화 시 통과
            return True, 1.0, []
        
        current_positions = [self.position.to_dict()] if self.position else []
        
        return self.filter_manager.check_entry_conditions(
            strategy_name=self.name,
            price_data=price_data,
            indicators=indicators,
            sentiment_score=sentiment_score,
            current_positions=current_positions
        )
    
    # ========== 표준화된 메트릭 ==========
    
    def calculate_standardized_metrics(self) -> MetricResult:
        """표준화된 메트릭 계산"""
        if not self.trades:
            return MetricResult()
        
        # TradeRecord를 딕셔너리로 변환
        trades_dict = [trade.to_dict() if hasattr(trade, 'to_dict') else trade for trade in self.trades]
        
        metrics = self.metrics_calculator.calculate_metrics(
            trades=trades_dict,
            initial_capital=1000000  # 기본값
        )
        
        self.current_metrics = metrics
        return metrics
    
    def get_mab_reward(self) -> float:
        """MAB 시스템을 위한 보상 값"""
        if not self.current_metrics:
            self.calculate_standardized_metrics()
        
        return self.current_metrics.mab_reward_score if self.current_metrics else 0.0
    
    # ========== 추상 메서드 ==========
    
    @abstractmethod
    def should_enter(self, price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """진입 조건 확인 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def should_exit(self, position: EnhancedPosition, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """전략별 지표 계산 (캐시 활용 권장)"""
        pass
    
    # ========== 메인 시그널 생성 ==========
    
    def generate_signal(self, price_data_window: Any) -> Dict[str, Any]:
        """향상된 시그널 생성 메서드"""
        try:
            # 데이터 준비
            price_data = self._prepare_data(price_data_window)
            
            # 일일 한도 체크
            if not self._check_daily_limits():
                return EnhancedSignal(
                    action="HOLD",
                    price=self.safe_last(price_data, "close"),
                    reason="일일 한도 초과"
                ).to_dict()
            
            # 지표 계산 (캐시 활용)
            indicators = self.calculate_indicators(price_data)
            
            # 감정 점수 추출
            sentiment_score = float(price_data_window.get("sentiment", 0.5))
            
            # 포지션 상태에 따른 처리
            if self.position is None:
                return self._handle_no_position(price_data, indicators, sentiment_score)
            else:
                return self._handle_existing_position(price_data, indicators)
                
        except Exception as e:
            logger.error(f"{self.name} 향상된 시그널 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            
            return EnhancedSignal(
                action="HOLD",
                price=0.0,
                reason=f"오류: {str(e)}"
            ).to_dict()
    
    def _handle_no_position(self,
                           price_data: pd.DataFrame,
                           indicators: Dict[str, Any],
                           sentiment_score: float) -> Dict[str, Any]:
        """포지션 없을 때 향상된 처리"""
        # 1. 기본 진입 조건 확인
        entry_info = self.should_enter(price_data)
        
        if not entry_info:
            return EnhancedSignal(
                action="HOLD",
                price=self.safe_last(price_data, "close"),
                reason="기본 진입 조건 미충족"
            ).to_dict()
        
        # 2. 향상된 필터 체크
        filter_passed, filter_score, filter_results = self.check_enhanced_filters(
            price_data, indicators, sentiment_score
        )
        
        if not filter_passed:
            failed_filters = [f.filter_name for f in filter_results if f.result.value == "fail"]
            return EnhancedSignal(
                action="HOLD",
                price=self.safe_last(price_data, "close"),
                reason=f"필터 실패: {', '.join(failed_filters[:3])}",
                filters_passed=filter_results
            ).to_dict()
        
        # 3. 최종 신뢰도 계산
        base_confidence = entry_info.get("confidence", 0.5)
        adjusted_confidence = (base_confidence + filter_score) / 2
        
        # 최소 신뢰도 확인
        min_confidence = self.config["min_confidence"]
        if adjusted_confidence < min_confidence:
            return EnhancedSignal(
                action="HOLD",
                price=self.safe_last(price_data, "close"),
                reason=f"신뢰도 부족: {adjusted_confidence:.3f} < {min_confidence}",
                confidence=adjusted_confidence,
                filters_passed=filter_results
            ).to_dict()
        
        # 4. 진입 실행
        current_price = self.safe_last(price_data, "close")
        
        # 손절/익절 설정
        stop_loss = entry_info.get("stop_loss") or current_price * (1 - self.config["default_stop_loss"])
        take_profit = entry_info.get("take_profit") or current_price * (1 + self.config["default_take_profit"])
        
        # 포지션 생성
        self.position = EnhancedPosition(
            side=entry_info.get("side", "LONG"),
            entry_price=current_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=entry_info.get("reason", ""),
            confidence=adjusted_confidence,
            metadata={"indicators": indicators, "filter_score": filter_score}
        )
        
        # 극값 초기화
        self.position.update_extremes(current_price)
        
        # 시그널 생성
        signal = EnhancedSignal(
            action="BUY",
            price=current_price,
            confidence=adjusted_confidence,
            reason=entry_info.get("reason", "향상된 진입 조건 충족"),
            stop_loss=stop_loss,
            take_profit=take_profit,
            filters_passed=filter_results
        )
        
        # 상태 업데이트
        self.last_signal_time = datetime.now()
        self.daily_trades += 1
        self._update_daily_date()
        
        logger.info(
            f"{self.name} 향상된 진입: "
            f"{signal.action} @ {signal.price:.2f} "
            f"(신뢰도: {signal.confidence:.3f}, 필터점수: {filter_score:.3f})"
        )
        
        return signal.to_dict()
    
    def _handle_existing_position(self,
                                price_data: pd.DataFrame,
                                indicators: Dict[str, Any]) -> Dict[str, Any]:
        """기존 포지션 향상된 처리"""
        current_price = self.safe_last(price_data, "close")
        
        # 극값 업데이트
        self.position.update_extremes(current_price)
        
        # 1. 리스크 관리 체크
        risk_exit = self._check_enhanced_risk_exit(current_price)
        if risk_exit:
            return self._create_enhanced_exit_signal(current_price, risk_exit)
        
        # 2. 전략별 청산 조건
        exit_reason = self.should_exit(self.position, price_data)
        if exit_reason:
            return self._create_enhanced_exit_signal(current_price, exit_reason)
        
        # 3. 향상된 Trailing Stop
        if self.config["enable_trailing_stop"]:
            self._update_enhanced_trailing_stop(current_price)
        
        # 홀드 시그널
        pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price
        
        return EnhancedSignal(
            action="HOLD",
            price=current_price,
            reason="포지션 유지",
            metadata={
                "pnl_pct": pnl_pct,
                "holding_time": str(self.position.holding_time),
                "highest_price": self.position.highest_price,
                "lowest_price": self.position.lowest_price
            }
        ).to_dict()
    
    def _check_enhanced_risk_exit(self, current_price: float) -> Optional[str]:
        """향상된 리스크 기반 청산 확인"""
        if not self.position:
            return None
        
        pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price
        
        # 기본 손절/익절
        if self.position.stop_loss and current_price <= self.position.stop_loss:
            return f"손절 ({pnl_pct:.1%})"
        
        if self.position.take_profit and current_price >= self.position.take_profit:
            return f"익절 ({pnl_pct:.1%})"
        
        # 시간 기반 청산
        holding_days = self.position.holding_time.days
        if holding_days > self.config["max_holding_days"]:
            return f"시간 초과 ({holding_days}일)"
        
        # 신뢰도 기반 청산 (낮은 신뢰도 포지션은 빨리 청산)
        if self.position.confidence < 0.4 and pnl_pct < -0.01:
            return f"낮은 신뢰도 손절 (신뢰도: {self.position.confidence:.2f})"
        
        return None
    
    def _update_enhanced_trailing_stop(self, current_price: float):
        """향상된 Trailing Stop 업데이트"""
        if not self.position or not self.position.stop_loss:
            return
        
        # 수익 중인 경우만
        if current_price > self.position.entry_price:
            trailing_distance = self.config["trailing_stop_distance"]
            
            # 신뢰도에 따른 조정
            confidence_multiplier = 1.0 + (self.position.confidence - 0.5)
            adjusted_distance = trailing_distance * confidence_multiplier
            
            new_stop = current_price * (1 - adjusted_distance)
            
            # 기존 손절가보다 높은 경우만 업데이트
            if new_stop > self.position.stop_loss:
                old_stop = self.position.stop_loss
                self.position.stop_loss = new_stop
                logger.debug(
                    f"{self.name}: 향상된 Trailing stop 업데이트 "
                    f"{old_stop:.2f} → {new_stop:.2f} (신뢰도: {self.position.confidence:.2f})"
                )
    
    def _create_enhanced_exit_signal(self, current_price: float, reason: str) -> Dict[str, Any]:
        """향상된 청산 시그널 생성"""
        if not self.position:
            return EnhancedSignal(action="HOLD", price=current_price).to_dict()
        
        # PnL 계산
        pnl = (current_price - self.position.entry_price) * self.position.quantity
        pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price
        
        # 거래 기록 생성
        trade_record = TradeRecord(
            timestamp=datetime.now(),
            action="SELL",
            price=current_price,
            quantity=self.position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_time=self.position.holding_time.total_seconds(),
            entry_reason=self.position.entry_reason,
            exit_reason=reason,
            strategy=self.name
        )
        
        self.trades.append(trade_record)
        
        # 필터 매니저 업데이트
        self.filter_manager.update_trade_result(pnl > 0)
        
        # 연속 손실 추적
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # 일일 PnL 업데이트
        self.daily_pnl += pnl
        
        # 포지션 정리
        old_position = self.position
        self.position = None
        
        # 메트릭 업데이트
        self.calculate_standardized_metrics()
        
        # 시그널 생성
        signal = EnhancedSignal(
            action="SELL",
            price=current_price,
            confidence=1.0,
            reason=reason,
            metadata={
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry_confidence": old_position.confidence,
                "holding_time_hours": old_position.holding_time.total_seconds() / 3600,
                "highest_price": old_position.highest_price,
                "lowest_price": old_position.lowest_price
            }
        )
        
        logger.info(
            f"{self.name} 향상된 청산: "
            f"{signal.action} @ {signal.price:.2f} "
            f"(PnL: {pnl_pct:.1%}, 이유: {reason})"
        )
        
        return signal.to_dict()
    
    # ========== 유틸리티 메서드 ==========
    
    def _prepare_data(self, price_data_window: Any) -> pd.DataFrame:
        """데이터 준비 및 검증"""
        if isinstance(price_data_window, list):
            df = pd.DataFrame(price_data_window)
        elif isinstance(price_data_window, dict):
            df = pd.DataFrame([price_data_window])
        elif isinstance(price_data_window, pd.DataFrame):
            df = price_data_window.copy()
        else:
            raise ValueError(f"지원하지 않는 데이터 타입: {type(price_data_window)}")
        
        # 필수 컬럼 확인
        required_columns = ["close", "high", "low", "volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")
        
        df = df.reset_index(drop=True)
        
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="5T")
        
        return df
    
    def _check_daily_limits(self) -> bool:
        """일일 거래 한도 확인"""
        self._update_daily_date()
        
        if self.daily_trades >= self.config["max_daily_trades"]:
            logger.warning(f"{self.name}: 일일 거래 한도 도달 ({self.daily_trades})")
            return False
        
        if self.consecutive_losses >= self.config["max_consecutive_losses"]:
            logger.warning(f"{self.name}: 연속 손실 한도 도달 ({self.consecutive_losses})")
            return False
        
        return True
    
    def _update_daily_date(self):
        """일일 통계 날짜 업데이트"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
    
    def safe_last(self, data: Any, column: Optional[str] = None) -> float:
        """마지막 값 안전하게 추출"""
        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                if column and column in data.columns:
                    return float(data[column].iloc[-1])
                return float(data.iloc[-1, 0])
            elif isinstance(data, pd.Series) and not data.empty:
                return float(data.iloc[-1])
            elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
                return float(data[-1])
            else:
                return float(data)
        except Exception as e:
            logger.debug(f"safe_last 변환 오류: {e}")
            return 0.0
    
    # ========== 성과 관리 ==========
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약 반환"""
        metrics = self.calculate_standardized_metrics()
        
        return {
            "strategy_name": self.name,
            "total_trades": len(self.trades),
            "current_position": self.position.to_dict() if self.position else None,
            "daily_stats": {
                "daily_trades": self.daily_trades,
                "daily_pnl": self.daily_pnl,
                "consecutive_losses": self.consecutive_losses
            },
            "metrics": metrics.to_dict(),
            "cache_stats": self.cache.get_stats(),
            "filter_stats": self.filter_manager.get_filter_stats()
        }
    
    def reset(self):
        """전략 초기화"""
        self.position = None
        self.trades.clear()
        self.current_metrics = None
        self.performance_history.clear()
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.last_signal_time = None
        
        # 캐시는 전략별로 초기화하지 않음 (공유 캐시)
        # self.cache.clear(self.name)  # 특정 전략 지표만 초기화 시
        
        logger.info(f"{self.name} 향상된 전략 초기화 완료")
    
    # ========== PPO 인터페이스 ==========
    
    def custom_observation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """PPO 학습용 관측값 (향상된 버전)"""
        indicators = self.calculate_indicators(market_data)
        
        base_obs = {
            "price_change": market_data["close"].pct_change().iloc[-1],
            "volume_ratio": market_data["volume"].iloc[-1] / market_data["volume"].mean(),
            "position_held": self.position is not None,
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades
        }
        
        # 지표 추가
        base_obs.update(indicators)
        
        # 필터 상태 추가
        if self.position is None:
            filter_passed, filter_score, _ = self.check_enhanced_filters(
                market_data, indicators
            )
            base_obs["filter_passed"] = filter_passed
            base_obs["filter_score"] = filter_score
        
        return base_obs
    
    def custom_reward(self, position: EnhancedPosition, price_data: pd.DataFrame) -> float:
        """PPO 학습용 보상 (향상된 버전)"""
        if not position:
            return 0.0
        
        current_price = self.safe_last(price_data, "close")
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # 기본 보상
        reward = pnl_pct
        
        # 신뢰도 보너스
        reward += (position.confidence - 0.5) * 0.1
        
        # 리스크 조정
        if pnl_pct < -self.config["default_stop_loss"]:
            reward -= 0.1  # 손절 미실행 패널티
        
        # 보유 시간 패널티 (신뢰도 고려)
        holding_hours = position.holding_time.total_seconds() / 3600
        if holding_hours > 24:
            time_penalty = (holding_hours - 24) * 0.001 * (1 - position.confidence)
            reward -= time_penalty
        
        return float(reward)
    
    # ========== 백테스트 호환성 메소드 ==========
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """백테스트용 거래 기록 (호환성 메소드)"""
        try:
            # TradeRecord 생성
            trade_record = TradeRecord(
                timestamp=trade_info.get("timestamp", datetime.now()),
                action=trade_info.get("action", "SELL"),
                price=trade_info.get("price", 0.0),
                quantity=trade_info.get("quantity", 1.0),
                pnl=trade_info.get("pnl", 0.0),
                pnl_pct=trade_info.get("pnl", 0.0),
                holding_time=0.0,
                entry_reason=trade_info.get("reason", ""),
                exit_reason=trade_info.get("exit_reason", ""),
                strategy=trade_info.get("strategy", self.name)
            )
            
            self.trades.append(trade_record)
            logger.debug(f"{self.name} 거래 기록: PnL={trade_record.pnl:.4f}")
            
        except Exception as e:
            logger.error(f"{self.name} log_trade 오류: {e}")
    
    def evaluate_result(self, price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """백테스트용 결과 평가 (호환성 메소드)"""
        try:
            # 표준화된 메트릭 계산
            metrics = self.calculate_standardized_metrics()
            
            # 백테스트 스크립트 호환 형식 반환
            result = {
                "roi": metrics.roi,
                "sharpe": metrics.sharpe_ratio,
                "win_rate": metrics.win_rate,
                "mdd": metrics.max_drawdown,
                "profit_factor": metrics.profit_factor,
                "baseline_roi": 0.0,  # 기본값
                "volatility": metrics.downside_deviation,  # 변동성 대체 지표
                "composite_score": metrics.mab_reward_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"{self.name} evaluate_result 오류: {e}")
            return {
                "roi": 0.0, "sharpe": 0.0, "win_rate": 0.0,
                "mdd": 0.0, "profit_factor": 0.0,
                "baseline_roi": 0.0, "volatility": 0.0,
                "composite_score": 0.0
            }