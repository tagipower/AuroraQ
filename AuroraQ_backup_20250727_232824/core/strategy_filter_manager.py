"""
전략 필터 매니저
강화된 진입 조건을 통합 관리하는 시스템
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from config.trade_config_loader import load_yaml_config

logger = get_logger("StrategyFilterManager")


class FilterResult(Enum):
    """필터 결과 타입"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class FilterCheckResult:
    """필터 체크 결과"""
    filter_name: str
    result: FilterResult
    score: float  # 0.0 ~ 1.0
    reason: str
    details: Dict[str, Any]


class StrategyFilterManager:
    """
    전략 진입 필터 통합 관리자
    - 전역 필터와 전략별 필터 적용
    - 다층 검증 시스템
    - 동적 조정 지원
    """
    
    def __init__(self, config_path: str = "config/enhanced_strategy_filters.yaml"):
        """필터 매니저 초기화"""
        self.config = load_yaml_config(config_path)
        self.global_filters = self.config.get("global_filters", {})
        self.strategy_filters = self.config.get("strategy_specific", {})
        self.risk_filters = self.config.get("risk_filters", {})
        self.filter_logic = self.config.get("filter_logic", {})
        self.override_conditions = self.config.get("override_conditions", {})
        
        # 상태 추적
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.position_history = []
        
        logger.info("전략 필터 매니저 초기화 완료")
    
    def check_entry_conditions(self,
                             strategy_name: str,
                             price_data: pd.DataFrame,
                             indicators: Dict[str, Any],
                             sentiment_score: float = 0.5,
                             current_positions: List[Dict] = None) -> Tuple[bool, float, List[FilterCheckResult]]:
        """
        통합 진입 조건 체크
        
        Returns:
            (통과 여부, 종합 점수, 세부 결과 리스트)
        """
        results = []
        
        # 1. 전역 필터 체크
        global_results = self._check_global_filters(price_data, indicators, sentiment_score)
        results.extend(global_results)
        
        # 2. 전략별 필터 체크
        if strategy_name in self.strategy_filters:
            strategy_results = self._check_strategy_filters(
                strategy_name, price_data, indicators
            )
            results.extend(strategy_results)
        
        # 3. 리스크 필터 체크
        risk_results = self._check_risk_filters(current_positions)
        results.extend(risk_results)
        
        # 4. 종합 평가
        passed, total_score = self._evaluate_results(results, strategy_name)
        
        # 5. 오버라이드 조건 체크
        if not passed:
            passed, total_score = self._check_override_conditions(
                price_data, indicators, sentiment_score, results
            )
        
        return passed, total_score, results
    
    def _check_global_filters(self,
                            price_data: pd.DataFrame,
                            indicators: Dict[str, Any],
                            sentiment_score: float) -> List[FilterCheckResult]:
        """전역 필터 체크"""
        results = []
        
        # 시장 체제 필터
        if self.global_filters.get("market_regime", {}).get("enabled", True):
            result = self._check_market_regime(price_data, indicators)
            results.append(result)
        
        # 감정 점수 필터
        if self.global_filters.get("sentiment", {}).get("enabled", True):
            result = self._check_sentiment(sentiment_score)
            results.append(result)
        
        # 변동성 필터
        if self.global_filters.get("volatility", {}).get("enabled", True):
            result = self._check_volatility(price_data, indicators)
            results.append(result)
        
        # 거래량 필터
        if self.global_filters.get("volume", {}).get("enabled", True):
            result = self._check_volume(price_data)
            results.append(result)
        
        # 시간 필터
        if self.global_filters.get("time", {}).get("enabled", True):
            result = self._check_time_filter(price_data)
            results.append(result)
        
        return results
    
    def _check_market_regime(self, 
                           price_data: pd.DataFrame,
                           indicators: Dict[str, Any]) -> FilterCheckResult:
        """시장 체제 체크"""
        try:
            config = self.global_filters.get("market_regime", {})
            
            # 간단한 체제 분류 (실제로는 더 복잡한 로직 필요)
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1) if len(price_data) >= 20 else 0
            
            # 체제 결정
            if volatility > 0.5:
                regime = "high_volatility_extreme"
                confidence = min(volatility / 0.5, 1.0)
            elif volatility > 0.3:
                regime = "high_volatility"
                confidence = 0.7
            elif trend > 0.1:
                regime = "bull"
                confidence = min(abs(trend) * 5, 1.0)
            elif trend < -0.1:
                regime = "bear"
                confidence = min(abs(trend) * 5, 1.0)
            else:
                regime = "sideways"
                confidence = 0.6
            
            # 허용/차단 체제 확인
            allowed = config.get("allowed_regimes", [])
            blocked = config.get("blocked_regimes", [])
            min_confidence = config.get("confidence_threshold", 0.7)
            
            if regime in blocked:
                return FilterCheckResult(
                    filter_name="market_regime",
                    result=FilterResult.FAIL,
                    score=0.0,
                    reason=f"차단된 시장 체제: {regime}",
                    details={"regime": regime, "confidence": confidence}
                )
            
            if allowed and regime not in allowed:
                return FilterCheckResult(
                    filter_name="market_regime",
                    result=FilterResult.FAIL,
                    score=0.3,
                    reason=f"허용되지 않은 시장 체제: {regime}",
                    details={"regime": regime, "confidence": confidence}
                )
            
            if confidence < min_confidence:
                return FilterCheckResult(
                    filter_name="market_regime",
                    result=FilterResult.WARNING,
                    score=0.5,
                    reason=f"낮은 체제 신뢰도: {confidence:.2f}",
                    details={"regime": regime, "confidence": confidence}
                )
            
            return FilterCheckResult(
                filter_name="market_regime",
                result=FilterResult.PASS,
                score=confidence,
                reason=f"정상 시장 체제: {regime}",
                details={"regime": regime, "confidence": confidence}
            )
            
        except Exception as e:
            logger.error(f"시장 체제 체크 오류: {e}")
            return FilterCheckResult(
                filter_name="market_regime",
                result=FilterResult.WARNING,
                score=0.5,
                reason="체제 체크 오류",
                details={"error": str(e)}
            )
    
    def _check_sentiment(self, sentiment_score: float) -> FilterCheckResult:
        """감정 점수 체크"""
        config = self.global_filters.get("sentiment", {})
        
        min_score = config.get("min_score", 0.3)
        max_score = config.get("max_score", 0.8)
        extreme_fear = config.get("extreme_fear_threshold", 0.2)
        extreme_greed = config.get("extreme_greed_threshold", 0.85)
        
        # 극단적 감정 체크
        if sentiment_score < extreme_fear:
            return FilterCheckResult(
                filter_name="sentiment",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"극단적 공포: {sentiment_score:.3f}",
                details={"sentiment": sentiment_score, "threshold": extreme_fear}
            )
        
        if sentiment_score > extreme_greed:
            return FilterCheckResult(
                filter_name="sentiment",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"극단적 탐욕: {sentiment_score:.3f}",
                details={"sentiment": sentiment_score, "threshold": extreme_greed}
            )
        
        # 일반 범위 체크
        if sentiment_score < min_score or sentiment_score > max_score:
            return FilterCheckResult(
                filter_name="sentiment",
                result=FilterResult.WARNING,
                score=0.3,
                reason=f"감정 점수 범위 벗어남: {sentiment_score:.3f}",
                details={"sentiment": sentiment_score, "range": [min_score, max_score]}
            )
        
        # 정상 범위
        # 0.5에 가까울수록 높은 점수
        distance_from_neutral = abs(sentiment_score - 0.5)
        score = 1.0 - (distance_from_neutral * 2)  # 최대 거리 0.5
        
        return FilterCheckResult(
            filter_name="sentiment",
            result=FilterResult.PASS,
            score=max(0.5, score),
            reason=f"정상 감정 점수: {sentiment_score:.3f}",
            details={"sentiment": sentiment_score}
        )
    
    def _check_volatility(self,
                         price_data: pd.DataFrame,
                         indicators: Dict[str, Any]) -> FilterCheckResult:
        """변동성 체크"""
        config = self.global_filters.get("volatility", {})
        
        # 변동성 계산
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # ATR이 있으면 사용
        atr = indicators.get("atr")
        if atr is not None and not pd.isna(atr):
            atr_ratio = atr / price_data['close'].iloc[-1]
            volatility = max(volatility, atr_ratio)
        
        min_vol = config.get("min_volatility", 0.005)
        max_vol = config.get("max_volatility", 0.10)
        
        if volatility < min_vol:
            return FilterCheckResult(
                filter_name="volatility",
                result=FilterResult.FAIL,
                score=0.2,
                reason=f"변동성 너무 낮음: {volatility:.4f}",
                details={"volatility": volatility, "min": min_vol}
            )
        
        if volatility > max_vol:
            return FilterCheckResult(
                filter_name="volatility",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"변동성 너무 높음: {volatility:.4f}",
                details={"volatility": volatility, "max": max_vol}
            )
        
        # 정상 범위 - 적정 변동성일수록 높은 점수
        optimal_vol = 0.02  # 2% 정도가 적정
        distance = abs(volatility - optimal_vol)
        score = max(0.5, 1.0 - (distance / optimal_vol))
        
        return FilterCheckResult(
            filter_name="volatility",
            result=FilterResult.PASS,
            score=score,
            reason=f"정상 변동성: {volatility:.4f}",
            details={"volatility": volatility}
        )
    
    def _check_volume(self, price_data: pd.DataFrame) -> FilterCheckResult:
        """거래량 체크"""
        config = self.global_filters.get("volume", {})
        
        if 'volume' not in price_data.columns:
            return FilterCheckResult(
                filter_name="volume",
                result=FilterResult.WARNING,
                score=0.5,
                reason="거래량 데이터 없음",
                details={}
            )
        
        lookback = config.get("lookback_period", 20)
        if len(price_data) < lookback:
            return FilterCheckResult(
                filter_name="volume",
                result=FilterResult.WARNING,
                score=0.5,
                reason="데이터 부족",
                details={"required": lookback, "available": len(price_data)}
            )
        
        current_volume = price_data['volume'].iloc[-1]
        avg_volume = price_data['volume'].tail(lookback).mean()
        
        if avg_volume == 0:
            return FilterCheckResult(
                filter_name="volume",
                result=FilterResult.FAIL,
                score=0.0,
                reason="평균 거래량 0",
                details={}
            )
        
        volume_ratio = current_volume / avg_volume
        min_ratio = config.get("min_ratio", 0.8)
        max_ratio = config.get("max_ratio", 5.0)
        
        if volume_ratio < min_ratio:
            return FilterCheckResult(
                filter_name="volume",
                result=FilterResult.WARNING,
                score=0.4,
                reason=f"낮은 거래량: {volume_ratio:.2f}x",
                details={"ratio": volume_ratio, "min": min_ratio}
            )
        
        if volume_ratio > max_ratio:
            return FilterCheckResult(
                filter_name="volume",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"비정상적 거래량 급증: {volume_ratio:.2f}x",
                details={"ratio": volume_ratio, "max": max_ratio}
            )
        
        # 정상 범위 - 1.0~2.0이 이상적
        if 1.0 <= volume_ratio <= 2.0:
            score = 1.0
        else:
            score = max(0.5, 1.0 - abs(volume_ratio - 1.5) / 3.0)
        
        return FilterCheckResult(
            filter_name="volume",
            result=FilterResult.PASS,
            score=score,
            reason=f"정상 거래량: {volume_ratio:.2f}x",
            details={"ratio": volume_ratio}
        )
    
    def _check_time_filter(self, price_data: pd.DataFrame) -> FilterCheckResult:
        """시간 필터 체크"""
        config = self.global_filters.get("time", {})
        
        # 현재 시간 (UTC)
        now = datetime.utcnow()
        current_hour = now.hour
        
        blocked_hours = config.get("blocked_hours", [])
        if current_hour in blocked_hours:
            return FilterCheckResult(
                filter_name="time",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"차단된 시간대: {current_hour}시 UTC",
                details={"hour": current_hour, "blocked": blocked_hours}
            )
        
        # 최소 데이터 요구량
        min_bars = config.get("min_bars_required", 100)
        if len(price_data) < min_bars:
            return FilterCheckResult(
                filter_name="time",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"데이터 부족: {len(price_data)}/{min_bars}",
                details={"available": len(price_data), "required": min_bars}
            )
        
        return FilterCheckResult(
            filter_name="time",
            result=FilterResult.PASS,
            score=1.0,
            reason="시간 필터 통과",
            details={"hour": current_hour}
        )
    
    def _check_strategy_filters(self,
                              strategy_name: str,
                              price_data: pd.DataFrame,
                              indicators: Dict[str, Any]) -> List[FilterCheckResult]:
        """전략별 필터 체크"""
        results = []
        strategy_config = self.strategy_filters.get(strategy_name, {})
        
        # 전략별 맞춤 체크 로직
        if strategy_name == "RuleStrategyA":
            results.extend(self._check_strategy_a_filters(strategy_config, indicators))
        elif strategy_name == "RuleStrategyB":
            results.extend(self._check_strategy_b_filters(strategy_config, indicators))
        # ... 다른 전략들도 구현
        
        # 공통: 최소 신뢰도 체크
        min_confidence = strategy_config.get("entry_confidence", {}).get("min_confidence", 0.5)
        return results
    
    def _check_strategy_a_filters(self,
                                config: Dict[str, Any],
                                indicators: Dict[str, Any]) -> List[FilterCheckResult]:
        """전략 A 전용 필터"""
        results = []
        
        # EMA 크로스 체크
        ema_config = config.get("ema_cross", {})
        if "ema_short" in indicators and "ema_long" in indicators:
            ema_diff = indicators["ema_short"] - indicators["ema_long"]
            ema_diff_pct = ema_diff / indicators["ema_long"] if indicators["ema_long"] > 0 else 0
            
            min_diff = ema_config.get("min_diff_pct", 0.002)
            if abs(ema_diff_pct) < min_diff:
                results.append(FilterCheckResult(
                    filter_name="ema_cross_strength",
                    result=FilterResult.WARNING,
                    score=0.4,
                    reason=f"약한 EMA 크로스: {ema_diff_pct:.4f}",
                    details={"diff_pct": ema_diff_pct, "min": min_diff}
                ))
            else:
                results.append(FilterCheckResult(
                    filter_name="ema_cross_strength",
                    result=FilterResult.PASS,
                    score=min(1.0, abs(ema_diff_pct) / min_diff),
                    reason=f"강한 EMA 크로스: {ema_diff_pct:.4f}",
                    details={"diff_pct": ema_diff_pct}
                ))
        
        # ADX 체크
        adx_config = config.get("adx", {})
        if "adx" in indicators:
            adx = indicators["adx"]
            min_adx = adx_config.get("min_threshold", 20)
            
            if adx < min_adx:
                results.append(FilterCheckResult(
                    filter_name="adx_strength",
                    result=FilterResult.FAIL,
                    score=0.0,
                    reason=f"약한 트렌드: ADX={adx:.1f}",
                    details={"adx": adx, "min": min_adx}
                ))
            else:
                score = min(1.0, (adx - min_adx) / 20)  # 20~40 범위에서 스케일
                results.append(FilterCheckResult(
                    filter_name="adx_strength", 
                    result=FilterResult.PASS,
                    score=score,
                    reason=f"강한 트렌드: ADX={adx:.1f}",
                    details={"adx": adx}
                ))
        
        return results
    
    def _check_strategy_b_filters(self,
                                config: Dict[str, Any],
                                indicators: Dict[str, Any]) -> List[FilterCheckResult]:
        """전략 B 전용 필터"""
        results = []
        
        # RSI 체크
        rsi_config = config.get("rsi", {})
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            oversold = rsi_config.get("oversold", 25)
            overbought = rsi_config.get("overbought", 75)
            neutral_zone = rsi_config.get("neutral_zone", [40, 60])
            
            if neutral_zone[0] <= rsi <= neutral_zone[1]:
                results.append(FilterCheckResult(
                    filter_name="rsi_position",
                    result=FilterResult.WARNING,
                    score=0.3,
                    reason=f"중립 구간 RSI: {rsi:.1f}",
                    details={"rsi": rsi, "neutral_zone": neutral_zone}
                ))
            elif rsi < oversold or rsi > overbought:
                results.append(FilterCheckResult(
                    filter_name="rsi_position",
                    result=FilterResult.PASS,
                    score=1.0,
                    reason=f"극단 구간 RSI: {rsi:.1f}",
                    details={"rsi": rsi}
                ))
        
        return results
    
    def _check_risk_filters(self, current_positions: List[Dict]) -> List[FilterCheckResult]:
        """리스크 필터 체크"""
        results = []
        
        # 연속 손실 체크
        loss_config = self.risk_filters.get("consecutive_losses", {})
        max_losses = loss_config.get("max_allowed", 3)
        
        if self.consecutive_losses >= max_losses:
            results.append(FilterCheckResult(
                filter_name="consecutive_losses",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"연속 손실 한도 초과: {self.consecutive_losses}",
                details={"losses": self.consecutive_losses, "max": max_losses}
            ))
        
        # 일일 한도 체크
        daily_config = self.risk_filters.get("daily_limits", {})
        max_daily_trades = daily_config.get("max_trades", 10)
        
        # 날짜 변경 체크
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        if self.daily_trades >= max_daily_trades:
            results.append(FilterCheckResult(
                filter_name="daily_trades",
                result=FilterResult.FAIL,
                score=0.0,
                reason=f"일일 거래 한도 초과: {self.daily_trades}",
                details={"trades": self.daily_trades, "max": max_daily_trades}
            ))
        
        # 상관 포지션 체크
        if current_positions:
            corr_config = self.risk_filters.get("correlated_positions", {})
            max_correlated = corr_config.get("max_correlated", 2)
            
            if len(current_positions) >= max_correlated:
                results.append(FilterCheckResult(
                    filter_name="correlated_positions",
                    result=FilterResult.WARNING,
                    score=0.3,
                    reason=f"상관 포지션 많음: {len(current_positions)}",
                    details={"positions": len(current_positions), "max": max_correlated}
                ))
        
        return results
    
    def _evaluate_results(self,
                         results: List[FilterCheckResult],
                         strategy_name: str) -> Tuple[bool, float]:
        """필터 결과 종합 평가"""
        if not results:
            return True, 1.0
        
        # 필터 로직 설정
        logic = self.filter_logic
        required_filters = logic.get("required_filters", [])
        weights = logic.get("filter_weights", {})
        min_score = logic.get("min_combined_score", 0.7)
        
        # FAIL이 하나라도 있으면 실패
        for result in results:
            if result.result == FilterResult.FAIL:
                if result.filter_name in required_filters:
                    return False, 0.0
        
        # 가중 평균 점수 계산
        total_weight = 0
        weighted_score = 0
        
        for result in results:
            weight = weights.get(result.filter_name, 1.0)
            total_weight += weight
            weighted_score += result.score * weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5
        
        passed = final_score >= min_score
        
        return passed, final_score
    
    def _check_override_conditions(self,
                                 price_data: pd.DataFrame,
                                 indicators: Dict[str, Any],
                                 sentiment_score: float,
                                 results: List[FilterCheckResult]) -> Tuple[bool, float]:
        """오버라이드 조건 체크"""
        # 극단적 시장 상황 체크
        extreme_config = self.override_conditions.get("extreme_market", {})
        
        # 변동성 계산
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # 극단적 상황 판단
        is_extreme = (
            volatility > 0.15 or
            abs(sentiment_score - 0.5) > 0.4
        )
        
        if is_extreme:
            # 일부 필터 완화
            multiplier = extreme_config.get("multiplier", 0.8)
            
            # 점수 재계산 (완화된 기준)
            relaxed_score = 0
            total_weight = 0
            
            for result in results:
                if result.filter_name in extreme_config.get("relaxed_filters", []):
                    # 완화된 점수
                    relaxed = min(1.0, result.score / multiplier)
                    relaxed_score += relaxed
                else:
                    relaxed_score += result.score
                total_weight += 1
            
            if total_weight > 0:
                final_score = relaxed_score / total_weight
                passed = final_score >= self.filter_logic.get("min_combined_score", 0.7) * multiplier
                return passed, final_score
        
        return False, 0.0
    
    def update_trade_result(self, is_profit: bool):
        """거래 결과 업데이트"""
        self.daily_trades += 1
        
        if is_profit:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 반환"""
        return {
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades,
            "last_trade_date": self.last_trade_date,
            "active_filters": len(self.global_filters),
            "strategy_filters": len(self.strategy_filters)
        }


# 전역 인스턴스
_filter_manager = None


def get_filter_manager() -> StrategyFilterManager:
    """전역 필터 매니저 인스턴스 반환"""
    global _filter_manager
    if _filter_manager is None:
        _filter_manager = StrategyFilterManager()
    return _filter_manager