#!/usr/bin/env python3
"""
비트코인 선물시장 전용 동적 레버리지 관리 시스템
AuroraQ VPS Deployment - 실시간 리스크 기반 레버리지 자동 조정
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# VPS 통합 로깅 시스템
try:
    from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
except ImportError:
    try:
        from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
    except ImportError:
        from vps_deployment.vps_logging import get_vps_log_integrator, LogCategory, LogLevel

class MarketCondition(Enum):
    """시장 상황"""
    EXTREMELY_VOLATILE = "extremely_volatile"  # 매우 변동성 높음
    HIGH_VOLATILE = "high_volatile"            # 높은 변동성
    NORMAL = "normal"                          # 정상
    LOW_VOLATILE = "low_volatile"              # 낮은 변동성
    CONSOLIDATION = "consolidation"            # 횡보

class RiskLevel(Enum):
    """리스크 수준"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6

@dataclass
class FuturesLeverageConfig:
    """선물 레버리지 설정"""
    # 기본 레버리지 설정
    min_leverage: float = 1.0          # 최소 레버리지
    max_leverage: float = 10.0         # 최대 레버리지
    default_leverage: float = 3.0      # 기본 레버리지
    
    # 동적 조정 파라미터
    volatility_threshold_high: float = 0.05    # 5% 고변동성 임계값
    volatility_threshold_low: float = 0.02     # 2% 저변동성 임계값
    
    # 리스크 기반 조정
    max_portfolio_risk: float = 0.02           # 2% 최대 포트폴리오 리스크
    leverage_reduction_factor: float = 0.8     # 리스크 초과 시 레버리지 감소 비율
    leverage_increase_factor: float = 1.2      # 저리스크 시 레버리지 증가 비율
    
    # 마진 관리
    initial_margin_ratio: float = 0.1          # 10% 초기 마진 비율
    maintenance_margin_ratio: float = 0.05     # 5% 유지 마진 비율
    liquidation_buffer: float = 0.02           # 2% 청산 버퍼
    
    # 동적 조정 주기
    adjustment_interval_seconds: int = 30      # 30초마다 조정 검토
    
    # 성능 기반 조정
    performance_lookback_trades: int = 20      # 최근 20거래 성과 기반
    performance_adjustment_factor: float = 0.1  # 성과 기반 조정 비율
    
    # 시간대별 조정
    enable_time_based_adjustment: bool = True
    low_liquidity_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # UTC 기준 저유동성 시간
    low_liquidity_leverage_factor: float = 0.7  # 저유동성 시간 레버리지 감소

@dataclass
class MarketMetrics:
    """시장 지표"""
    price: float
    volatility_1h: float = 0.0
    volatility_4h: float = 0.0
    volatility_24h: float = 0.0
    volume_24h: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    
    # 기술적 지표
    rsi_14: float = 50.0
    atr_14: float = 0.0
    bb_width: float = 0.0
    
    # 트렌드 지표
    trend_strength: float = 0.0
    market_condition: MarketCondition = MarketCondition.NORMAL

@dataclass
class PositionMetrics:
    """포지션 지표"""
    symbol: str
    side: str  # LONG, SHORT
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin_ratio: float
    
    # 리스크 지표
    position_risk: float = 0.0
    liquidation_price: float = 0.0
    margin_available: float = 0.0

class FuturesLeverageManager:
    """비트코인 선물 동적 레버리지 관리자"""
    
    def __init__(self, config: FuturesLeverageConfig, enable_logging: bool = True):
        """
        선물 레버리지 관리자 초기화
        
        Args:
            config: 레버리지 설정
            enable_logging: 통합 로깅 활성화
        """
        self.config = config
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("futures_leverage_manager")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 현재 상태
        self.current_leverage = config.default_leverage
        self.current_market_condition = MarketCondition.NORMAL
        self.current_risk_level = RiskLevel.MODERATE
        
        # 히스토리컬 데이터
        self.volatility_history = []
        self.performance_history = []
        self.leverage_adjustments = []
        
        # 실시간 모니터링
        self.is_monitoring = False
        self.monitoring_task = None
        
        # 통계
        self.stats = {
            "total_adjustments": 0,
            "leverage_increases": 0,
            "leverage_decreases": 0,
            "avg_leverage": config.default_leverage,
            "risk_events": 0,
            "liquidation_warnings": 0
        }
        
        if self.logger:
            self.logger.info(f"FuturesLeverageManager initialized - Default leverage: {config.default_leverage}x")
    
    async def calculate_optimal_leverage(self, 
                                       market_metrics: MarketMetrics,
                                       position_metrics: Optional[PositionMetrics] = None,
                                       strategy_confidence: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        최적 레버리지 계산
        
        Args:
            market_metrics: 시장 지표
            position_metrics: 현재 포지션 지표
            strategy_confidence: 전략 신뢰도 (0-1)
            
        Returns:
            Tuple[float, Dict[str, Any]]: (최적 레버리지, 계산 세부사항)
        """
        try:
            calculation_details = {}
            
            # 1. 기본 레버리지에서 시작
            base_leverage = self.config.default_leverage
            calculation_details['base_leverage'] = base_leverage
            
            # 2. 변동성 기반 조정
            volatility_factor = await self._calculate_volatility_factor(market_metrics)
            calculation_details['volatility_factor'] = volatility_factor
            
            # 3. 시장 상황 기반 조정
            market_factor = await self._calculate_market_factor(market_metrics)
            calculation_details['market_factor'] = market_factor
            
            # 4. 전략 신뢰도 기반 조정
            confidence_factor = self._calculate_confidence_factor(strategy_confidence)
            calculation_details['confidence_factor'] = confidence_factor
            
            # 5. 시간대 기반 조정
            time_factor = self._calculate_time_factor()
            calculation_details['time_factor'] = time_factor
            
            # 6. 성과 기반 조정
            performance_factor = self._calculate_performance_factor()
            calculation_details['performance_factor'] = performance_factor
            
            # 7. 현재 포지션 리스크 기반 조정
            position_factor = 1.0
            if position_metrics:
                position_factor = await self._calculate_position_factor(position_metrics)
            calculation_details['position_factor'] = position_factor
            
            # 8. 종합 레버리지 계산
            optimal_leverage = (base_leverage * 
                              volatility_factor * 
                              market_factor * 
                              confidence_factor * 
                              time_factor * 
                              performance_factor * 
                              position_factor)
            
            # 9. 한도 적용
            optimal_leverage = max(self.config.min_leverage, 
                                 min(optimal_leverage, self.config.max_leverage))
            
            calculation_details['optimal_leverage'] = optimal_leverage
            calculation_details['current_leverage'] = self.current_leverage
            calculation_details['adjustment_needed'] = abs(optimal_leverage - self.current_leverage) > 0.1
            
            return optimal_leverage, calculation_details
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Optimal leverage calculation error: {e}")
            return self.config.default_leverage, {"error": str(e)}
    
    async def _calculate_volatility_factor(self, market_metrics: MarketMetrics) -> float:
        """변동성 기반 레버리지 조정 계수"""
        try:
            # 다중 시간대 변동성 가중 평균
            vol_1h_weight = 0.5
            vol_4h_weight = 0.3
            vol_24h_weight = 0.2
            
            weighted_volatility = (market_metrics.volatility_1h * vol_1h_weight +
                                 market_metrics.volatility_4h * vol_4h_weight +
                                 market_metrics.volatility_24h * vol_24h_weight)
            
            # 변동성에 따른 레버리지 조정
            if weighted_volatility > self.config.volatility_threshold_high:
                # 고변동성: 레버리지 감소
                factor = 0.5 + (0.5 * (self.config.volatility_threshold_high / weighted_volatility))
                factor = max(0.3, min(factor, 1.0))  # 0.3 ~ 1.0 범위
            elif weighted_volatility < self.config.volatility_threshold_low:
                # 저변동성: 레버리지 증가 가능
                factor = 1.0 + (0.5 * (self.config.volatility_threshold_low / weighted_volatility - 1))
                factor = max(1.0, min(factor, 1.5))  # 1.0 ~ 1.5 범위
            else:
                # 정상 변동성
                factor = 1.0
            
            return factor
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Volatility factor calculation error: {e}")
            return 1.0
    
    async def _calculate_market_factor(self, market_metrics: MarketMetrics) -> float:
        """시장 상황 기반 레버리지 조정 계수"""
        try:
            factor = 1.0
            
            # 펀딩비율 기반 조정
            if abs(market_metrics.funding_rate) > 0.01:  # 1% 이상
                factor *= 0.8  # 극단적 펀딩비율 시 레버리지 감소
            
            # RSI 기반 조정
            if market_metrics.rsi_14 > 80 or market_metrics.rsi_14 < 20:
                factor *= 0.9  # 과매수/과매도 시 레버리지 감소
            
            # 볼린저 밴드 폭 기반 조정
            if market_metrics.bb_width > 0.1:  # 10% 이상
                factor *= 0.85  # 밴드 확장 시 레버리지 감소
            
            # 트렌드 강도 기반 조정
            if market_metrics.trend_strength > 0.7:
                factor *= 1.1  # 강한 트렌드 시 레버리지 증가
            elif market_metrics.trend_strength < 0.3:
                factor *= 0.9  # 약한 트렌드 시 레버리지 감소
            
            # 시장 상황별 조정
            condition_factors = {
                MarketCondition.EXTREMELY_VOLATILE: 0.3,
                MarketCondition.HIGH_VOLATILE: 0.6,
                MarketCondition.NORMAL: 1.0,
                MarketCondition.LOW_VOLATILE: 1.2,
                MarketCondition.CONSOLIDATION: 0.8
            }
            
            factor *= condition_factors.get(market_metrics.market_condition, 1.0)
            
            return max(0.3, min(factor, 1.5))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Market factor calculation error: {e}")
            return 1.0
    
    def _calculate_confidence_factor(self, strategy_confidence: float) -> float:
        """전략 신뢰도 기반 레버리지 조정 계수"""
        try:
            # 신뢰도가 높을수록 높은 레버리지 허용
            if strategy_confidence > 0.8:
                return 1.2
            elif strategy_confidence > 0.6:
                return 1.0
            elif strategy_confidence > 0.4:
                return 0.8
            else:
                return 0.6
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Confidence factor calculation error: {e}")
            return 1.0
    
    def _calculate_time_factor(self) -> float:
        """시간대 기반 레버리지 조정 계수"""
        try:
            if not self.config.enable_time_based_adjustment:
                return 1.0
            
            current_hour = datetime.utcnow().hour
            
            # 저유동성 시간대 확인
            if current_hour in self.config.low_liquidity_hours:
                return self.config.low_liquidity_leverage_factor
            
            # 고유동성 시간대 (아시아/유럽/미국 개장 시간)
            high_liquidity_hours = [8, 9, 10, 14, 15, 16, 20, 21, 22]
            if current_hour in high_liquidity_hours:
                return 1.1
            
            return 1.0
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Time factor calculation error: {e}")
            return 1.0
    
    def _calculate_performance_factor(self) -> float:
        """최근 성과 기반 레버리지 조정 계수"""
        try:
            if len(self.performance_history) < 5:
                return 1.0
            
            # 최근 거래들의 성과 분석
            recent_trades = self.performance_history[-self.config.performance_lookback_trades:]
            
            # 승률 계산
            winning_trades = sum(1 for trade in recent_trades if trade > 0)
            win_rate = winning_trades / len(recent_trades)
            
            # 평균 수익률 계산
            avg_return = sum(recent_trades) / len(recent_trades)
            
            # 성과 기반 조정
            performance_score = (win_rate * 0.6) + ((avg_return + 1) * 0.4)
            
            if performance_score > 1.2:
                return 1.1  # 좋은 성과 시 레버리지 증가
            elif performance_score < 0.8:
                return 0.8  # 나쁜 성과 시 레버리지 감소
            else:
                return 1.0
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Performance factor calculation error: {e}")
            return 1.0
    
    async def _calculate_position_factor(self, position_metrics: PositionMetrics) -> float:
        """현재 포지션 리스크 기반 레버리지 조정 계수"""
        try:
            factor = 1.0
            
            # 마진 비율 기반 조정
            if position_metrics.margin_ratio < self.config.maintenance_margin_ratio * 2:
                factor *= 0.5  # 마진 부족 시 급격한 레버리지 감소
            elif position_metrics.margin_ratio < self.config.maintenance_margin_ratio * 3:
                factor *= 0.7  # 마진 주의 시 레버리지 감소
            
            # 미실현 손익 기반 조정
            unrealized_pnl_pct = position_metrics.unrealized_pnl / abs(position_metrics.size * position_metrics.entry_price)
            
            if unrealized_pnl_pct < -0.05:  # -5% 이하
                factor *= 0.6  # 손실 포지션 시 레버리지 감소
            elif unrealized_pnl_pct > 0.05:  # +5% 이상
                factor *= 1.1  # 수익 포지션 시 레버리지 소폭 증가
            
            # 청산가 근접도 체크
            if position_metrics.liquidation_price > 0:
                price_to_liquidation = abs(position_metrics.mark_price - position_metrics.liquidation_price) / position_metrics.mark_price
                
                if price_to_liquidation < 0.1:  # 청산가 10% 이내
                    factor *= 0.3  # 극도로 위험한 상황
                elif price_to_liquidation < 0.2:  # 청산가 20% 이내
                    factor *= 0.6  # 위험한 상황
            
            return max(0.1, min(factor, 1.2))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position factor calculation error: {e}")
            return 1.0
    
    async def adjust_leverage(self, 
                            optimal_leverage: float,
                            current_position: Optional[PositionMetrics] = None,
                            force_adjustment: bool = False) -> bool:
        """
        레버리지 조정 실행
        
        Args:
            optimal_leverage: 최적 레버리지
            current_position: 현재 포지션 정보
            force_adjustment: 강제 조정 여부
            
        Returns:
            bool: 조정 성공 여부
        """
        try:
            # 조정 필요성 확인
            leverage_diff = abs(optimal_leverage - self.current_leverage)
            
            if not force_adjustment and leverage_diff < 0.1:
                return True  # 조정 불필요
            
            # 포지션이 있을 때는 신중하게 조정
            if current_position and leverage_diff > 1.0:
                # 포지션이 있을 때 급격한 변경은 위험
                adjustment_ratio = 0.5
                optimal_leverage = self.current_leverage + (optimal_leverage - self.current_leverage) * adjustment_ratio
            
            # 실제 레버리지 조정 (시뮬레이션)
            old_leverage = self.current_leverage
            self.current_leverage = optimal_leverage
            
            # 조정 히스토리 기록
            adjustment_record = {
                "timestamp": datetime.now(),
                "old_leverage": old_leverage,
                "new_leverage": optimal_leverage,
                "adjustment_reason": "dynamic_optimization",
                "has_position": current_position is not None
            }
            self.leverage_adjustments.append(adjustment_record)
            
            # 통계 업데이트
            self.stats["total_adjustments"] += 1
            if optimal_leverage > old_leverage:
                self.stats["leverage_increases"] += 1
            else:
                self.stats["leverage_decreases"] += 1
            
            # 평균 레버리지 업데이트
            total_adjustments = self.stats["total_adjustments"]
            current_avg = self.stats["avg_leverage"]
            self.stats["avg_leverage"] = ((current_avg * (total_adjustments - 1)) + optimal_leverage) / total_adjustments
            
            # 로깅
            if self.logger:
                self.logger.info(
                    f"Leverage adjusted: {old_leverage:.2f}x → {optimal_leverage:.2f}x",
                    adjustment_ratio=leverage_diff,
                    has_position=current_position is not None
                )
            
            # 중요한 조정 시 Tagged 로깅
            if self.log_integrator and leverage_diff > 1.0:
                await self.log_integrator.log_security_event(
                    event_type="leverage_adjustment",
                    severity="medium" if leverage_diff < 2.0 else "high",
                    description=f"Leverage adjusted: {old_leverage:.2f}x → {optimal_leverage:.2f}x",
                    old_leverage=old_leverage,
                    new_leverage=optimal_leverage,
                    adjustment_difference=leverage_diff,
                    has_position=current_position is not None
                )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Leverage adjustment error: {e}")
            
            # 에러 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="leverage_adjustment_failed",
                    severity="high",
                    description=f"Failed to adjust leverage: {str(e)}",
                    target_leverage=optimal_leverage,
                    current_leverage=self.current_leverage,
                    error_details=str(e)
                )
            
            return False
    
    async def calculate_position_size(self, 
                                    account_balance: float,
                                    entry_price: float,
                                    risk_per_trade: float = 0.02) -> Tuple[float, Dict[str, Any]]:
        """
        레버리지를 고려한 포지션 크기 계산
        
        Args:
            account_balance: 계좌 잔고
            entry_price: 진입 가격
            risk_per_trade: 거래당 리스크 비율 (2%)
            
        Returns:
            Tuple[float, Dict[str, Any]]: (포지션 크기, 계산 세부사항)
        """
        try:
            calculation_details = {}
            
            # 1. 위험 허용 금액 계산
            risk_amount = account_balance * risk_per_trade
            calculation_details['risk_amount'] = risk_amount
            calculation_details['account_balance'] = account_balance
            calculation_details['risk_per_trade'] = risk_per_trade
            
            # 2. 현재 레버리지 적용
            leveraged_buying_power = account_balance * self.current_leverage
            calculation_details['leveraged_buying_power'] = leveraged_buying_power
            calculation_details['current_leverage'] = self.current_leverage
            
            # 3. 초기 마진 필요 금액
            initial_margin_required = leveraged_buying_power * self.config.initial_margin_ratio
            calculation_details['initial_margin_required'] = initial_margin_required
            
            # 4. 사용 가능한 최대 포지션 가치
            max_position_value = min(leveraged_buying_power, 
                                   account_balance / self.config.initial_margin_ratio)
            calculation_details['max_position_value'] = max_position_value
            
            # 5. 리스크 기반 포지션 크기 계산
            # ATR 기반 스톱로스 가정 (5% 기본)
            estimated_stop_loss_pct = 0.05
            risk_based_position_value = risk_amount / estimated_stop_loss_pct
            calculation_details['risk_based_position_value'] = risk_based_position_value
            calculation_details['estimated_stop_loss_pct'] = estimated_stop_loss_pct
            
            # 6. 최종 포지션 크기 결정 (더 작은 값 선택)
            final_position_value = min(max_position_value, risk_based_position_value)
            position_size = final_position_value / entry_price
            
            calculation_details['final_position_value'] = final_position_value
            calculation_details['position_size'] = position_size
            calculation_details['entry_price'] = entry_price
            
            # 7. 마진 및 청산가 계산
            required_margin = final_position_value / self.current_leverage
            calculation_details['required_margin'] = required_margin
            
            # 청산가 추정 (LONG 포지션 기준)
            liquidation_price = entry_price * (1 - (1 / self.current_leverage) + self.config.liquidation_buffer)
            calculation_details['estimated_liquidation_price'] = liquidation_price
            
            # 8. 안전성 검증
            margin_utilization = required_margin / account_balance
            calculation_details['margin_utilization'] = margin_utilization
            
            if margin_utilization > 0.8:  # 80% 이상 마진 사용
                calculation_details['warning'] = "High margin utilization - consider reducing position size"
            
            return position_size, calculation_details
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position size calculation error: {e}")
            return 0.0, {"error": str(e)}
    
    async def check_liquidation_risk(self, position_metrics: PositionMetrics) -> Dict[str, Any]:
        """청산 리스크 체크"""
        try:
            risk_assessment = {
                "risk_level": RiskLevel.MODERATE,
                "liquidation_distance_pct": 0.0,
                "recommended_action": "monitor",
                "warnings": []
            }
            
            if position_metrics.liquidation_price <= 0:
                return risk_assessment
            
            # 청산가까지의 거리 계산
            if position_metrics.side == "LONG":
                distance_pct = (position_metrics.mark_price - position_metrics.liquidation_price) / position_metrics.mark_price
            else:
                distance_pct = (position_metrics.liquidation_price - position_metrics.mark_price) / position_metrics.mark_price
            
            risk_assessment["liquidation_distance_pct"] = distance_pct
            
            # 리스크 수준 평가
            if distance_pct < 0.05:  # 5% 이내
                risk_assessment["risk_level"] = RiskLevel.EXTREME
                risk_assessment["recommended_action"] = "emergency_close"
                risk_assessment["warnings"].append("CRITICAL: Liquidation imminent")
                
                # 긴급 알림
                if self.log_integrator:
                    await self.log_integrator.log_security_event(
                        event_type="liquidation_warning_critical",
                        severity="critical",
                        description=f"CRITICAL: Position near liquidation - {distance_pct:.2%} away",
                        symbol=position_metrics.symbol,
                        side=position_metrics.side,
                        liquidation_distance=distance_pct,
                        mark_price=position_metrics.mark_price,
                        liquidation_price=position_metrics.liquidation_price
                    )
                
                self.stats["liquidation_warnings"] += 1
                
            elif distance_pct < 0.1:  # 10% 이내
                risk_assessment["risk_level"] = RiskLevel.VERY_HIGH
                risk_assessment["recommended_action"] = "reduce_position"
                risk_assessment["warnings"].append("HIGH RISK: Close to liquidation")
                
            elif distance_pct < 0.2:  # 20% 이내
                risk_assessment["risk_level"] = RiskLevel.HIGH
                risk_assessment["recommended_action"] = "monitor_closely"
                risk_assessment["warnings"].append("WARNING: Approaching liquidation zone")
                
            elif distance_pct < 0.3:  # 30% 이내
                risk_assessment["risk_level"] = RiskLevel.MODERATE
                risk_assessment["recommended_action"] = "monitor"
                
            else:
                risk_assessment["risk_level"] = RiskLevel.LOW
                risk_assessment["recommended_action"] = "normal"
            
            # 마진 비율 추가 체크
            if position_metrics.margin_ratio < self.config.maintenance_margin_ratio * 1.5:
                risk_assessment["warnings"].append("Low margin ratio detected")
                if risk_assessment["risk_level"].value < RiskLevel.HIGH.value:
                    risk_assessment["risk_level"] = RiskLevel.HIGH
            
            return risk_assessment
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Liquidation risk check error: {e}")
            return {"error": str(e), "risk_level": RiskLevel.EXTREME}
    
    def get_leverage_stats(self) -> Dict[str, Any]:
        """레버리지 관리 통계"""
        try:
            stats = self.stats.copy()
            
            # 추가 통계 계산
            stats.update({
                "current_leverage": self.current_leverage,
                "current_market_condition": self.current_market_condition.value,
                "current_risk_level": self.current_risk_level.value,
                "volatility_history_size": len(self.volatility_history),
                "performance_history_size": len(self.performance_history),
                "adjustment_history_size": len(self.leverage_adjustments),
                "config": {
                    "min_leverage": self.config.min_leverage,
                    "max_leverage": self.config.max_leverage,
                    "default_leverage": self.config.default_leverage
                }
            })
            
            # 최근 조정 정보
            if self.leverage_adjustments:
                latest_adjustment = self.leverage_adjustments[-1]
                stats["latest_adjustment"] = {
                    "timestamp": latest_adjustment["timestamp"].isoformat(),
                    "old_leverage": latest_adjustment["old_leverage"],
                    "new_leverage": latest_adjustment["new_leverage"]
                }
            
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Leverage stats error: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """동적 레버리지 모니터링 시작"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.logger:
                self.logger.info("Dynamic leverage monitoring started")
    
    async def stop_monitoring(self):
        """동적 레버리지 모니터링 중지"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.logger:
            self.logger.info("Dynamic leverage monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        try:
            while self.is_monitoring:
                # 주기적으로 레버리지 최적화 검토
                # 실제 구현에서는 시장 데이터를 가져와서 처리
                await asyncio.sleep(self.config.adjustment_interval_seconds)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Monitoring loop error: {e}")

# 팩토리 함수
def create_futures_leverage_manager(config: Optional[FuturesLeverageConfig] = None) -> FuturesLeverageManager:
    """VPS 최적화된 선물 레버리지 관리자 생성"""
    if config is None:
        config = FuturesLeverageConfig()
    
    return FuturesLeverageManager(config, enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_leverage_manager():
        config = FuturesLeverageConfig(
            min_leverage=1.0,
            max_leverage=10.0,
            default_leverage=3.0
        )
        
        manager = create_futures_leverage_manager(config)
        
        # 테스트 시장 데이터
        market_metrics = MarketMetrics(
            price=50000.0,
            volatility_1h=0.03,
            volatility_4h=0.04,
            volatility_24h=0.05,
            volume_24h=1000000.0,
            rsi_14=65.0,
            trend_strength=0.6,
            market_condition=MarketCondition.NORMAL
        )
        
        # 최적 레버리지 계산
        optimal_leverage, details = await manager.calculate_optimal_leverage(
            market_metrics=market_metrics,
            strategy_confidence=0.7
        )
        
        print(f"Optimal leverage: {optimal_leverage:.2f}x")
        print("Calculation details:", json.dumps(details, indent=2, default=str))
        
        # 레버리지 조정
        success = await manager.adjust_leverage(optimal_leverage)
        print(f"Leverage adjustment: {'Success' if success else 'Failed'}")
        
        # 포지션 크기 계산
        position_size, pos_details = await manager.calculate_position_size(
            account_balance=10000.0,
            entry_price=50000.0,
            risk_per_trade=0.02
        )
        
        print(f"Recommended position size: {position_size:.6f} BTC")
        print("Position calculation:", json.dumps(pos_details, indent=2, default=str))
        
        # 통계 확인
        stats = manager.get_leverage_stats()
        print("Leverage stats:", json.dumps(stats, indent=2, default=str))
    
    asyncio.run(test_leverage_manager())