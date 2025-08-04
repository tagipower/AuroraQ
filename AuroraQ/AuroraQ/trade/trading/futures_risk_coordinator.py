#!/usr/bin/env python3
"""
비트코인 선물 통합 리스크 관리 코디네이터
AuroraQ VPS Deployment - 레버리지, 마진, 포지션 통합 관리
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# VPS 통합 로깅 시스템
from vps_logging import get_vps_log_integrator, LogCategory, LogLevel

# 선물 관리 모듈들
from trading.futures_leverage_manager import (
    FuturesLeverageManager, FuturesLeverageConfig, 
    MarketMetrics, MarketCondition, RiskLevel
)
from trading.futures_margin_manager import (
    FuturesMarginManager, MarginConfig, MarginStatus, 
    MarginAction, PositionMetrics
)

class SystemRiskLevel(Enum):
    """시스템 전체 리스크 수준"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"  
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskEvent(Enum):
    """리스크 이벤트 유형"""
    LEVERAGE_ADJUSTED = "leverage_adjusted"
    MARGIN_WARNING = "margin_warning"
    POSITION_REDUCED = "position_reduced"
    EMERGENCY_CLOSE = "emergency_close"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDATION_RISK = "liquidation_risk"

@dataclass
class SystemRiskState:
    """시스템 리스크 상태"""
    overall_risk_level: SystemRiskLevel
    leverage_risk: RiskLevel
    margin_risk: MarginStatus
    market_risk: MarketCondition
    
    # 리스크 점수 (0-100)
    overall_risk_score: float = 0.0
    leverage_risk_score: float = 0.0
    margin_risk_score: float = 0.0
    market_risk_score: float = 0.0
    
    # 권장 조치
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 제한사항
    trading_restrictions: Dict[str, bool] = field(default_factory=dict)

@dataclass 
class FuturesRiskConfig:
    """선물 리스크 통합 설정"""
    # 레버리지 관리
    leverage_config: FuturesLeverageConfig = field(default_factory=FuturesLeverageConfig)
    
    # 마진 관리  
    margin_config: MarginConfig = field(default_factory=MarginConfig)
    
    # 통합 리스크 임계값
    critical_risk_score: float = 80.0        # 80점 이상 임계
    high_risk_score: float = 65.0            # 65점 이상 위험
    moderate_risk_score: float = 45.0        # 45점 이상 보통
    
    # 자동 대응 설정
    enable_auto_leverage_adjustment: bool = True
    enable_auto_margin_management: bool = True
    enable_emergency_protocols: bool = True
    
    # 모니터링 주기
    risk_assessment_interval: int = 15       # 15초마다 리스크 평가
    
    # 거래 제한 설정
    restrict_new_positions_risk_score: float = 70.0    # 70점 이상 신규 포지션 제한
    force_position_reduction_risk_score: float = 85.0  # 85점 이상 강제 포지션 축소
    emergency_close_risk_score: float = 95.0           # 95점 이상 긴급 청산

class FuturesRiskCoordinator:
    """비트코인 선물 통합 리스크 관리 코디네이터"""
    
    def __init__(self, config: FuturesRiskConfig, enable_logging: bool = True):
        """
        선물 리스크 코디네이터 초기화
        
        Args:
            config: 통합 리스크 설정
            enable_logging: 통합 로깅 활성화
        """
        self.config = config
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("futures_risk_coordinator")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 하위 관리자 초기화
        self.leverage_manager = FuturesLeverageManager(config.leverage_config, enable_logging)
        self.margin_manager = FuturesMarginManager(config.margin_config, enable_logging)
        
        # 현재 상태
        self.current_risk_state = SystemRiskState(
            overall_risk_level=SystemRiskLevel.MODERATE,
            leverage_risk=RiskLevel.MODERATE,
            margin_risk=MarginStatus.HEALTHY,
            market_risk=MarketCondition.NORMAL
        )
        
        # 히스토리
        self.risk_history = []
        self.event_history = []
        
        # 모니터링
        self.is_monitoring = False
        self.monitoring_task = None
        
        # 통계
        self.stats = {
            "risk_assessments": 0,
            "leverage_adjustments": 0,
            "margin_actions": 0,
            "emergency_interventions": 0,
            "positions_protected": 0,
            "total_risk_events": 0
        }
        
        if self.logger:
            self.logger.info("FuturesRiskCoordinator initialized")
    
    async def assess_system_risk(self, 
                                market_metrics: MarketMetrics,
                                position_metrics: Optional[PositionMetrics],
                                account_balance: float,
                                current_price: float) -> SystemRiskState:
        """
        시스템 전체 리스크 평가
        
        Args:
            market_metrics: 시장 지표
            position_metrics: 포지션 지표
            account_balance: 계좌 잔고
            current_price: 현재 가격
            
        Returns:
            SystemRiskState: 시스템 리스크 상태
        """
        try:
            self.stats["risk_assessments"] += 1
            
            # 1. 레버리지 리스크 평가
            leverage_risk_score = await self._assess_leverage_risk(market_metrics, position_metrics)
            
            # 2. 마진 리스크 평가
            margin_risk_score = 0.0
            margin_status = MarginStatus.HEALTHY
            
            if position_metrics:
                margin_status, margin_info, _ = await self.margin_manager.analyze_margin_status(
                    position_metrics, account_balance, current_price
                )
                margin_risk_score = await self._assess_margin_risk(margin_status, margin_info)
            
            # 3. 시장 리스크 평가
            market_risk_score = await self._assess_market_risk(market_metrics)
            
            # 4. 종합 리스크 점수 계산
            overall_risk_score = await self._calculate_overall_risk_score(
                leverage_risk_score, margin_risk_score, market_risk_score
            )
            
            # 5. 시스템 리스크 수준 결정
            overall_risk_level = await self._determine_risk_level(overall_risk_score)
            
            # 6. 권장 조치 생성
            recommended_actions = await self._generate_risk_actions(
                overall_risk_score, leverage_risk_score, margin_risk_score, market_risk_score
            )
            
            # 7. 거래 제한사항 결정
            trading_restrictions = await self._determine_trading_restrictions(overall_risk_score)
            
            # 8. 리스크 상태 생성
            risk_state = SystemRiskState(
                overall_risk_level=overall_risk_level,
                leverage_risk=self.leverage_manager.current_risk_level,
                margin_risk=margin_status,
                market_risk=market_metrics.market_condition,
                overall_risk_score=overall_risk_score,
                leverage_risk_score=leverage_risk_score,
                margin_risk_score=margin_risk_score,
                market_risk_score=market_risk_score,
                recommended_actions=recommended_actions,
                trading_restrictions=trading_restrictions
            )
            
            # 9. 상태 업데이트 및 히스토리 기록
            await self._update_risk_state(risk_state)
            
            return risk_state
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"System risk assessment error: {e}")
            
            # 에러 시 안전한 기본값 반환
            return SystemRiskState(
                overall_risk_level=SystemRiskLevel.CRITICAL,
                leverage_risk=RiskLevel.EXTREME,
                margin_risk=MarginStatus.CRITICAL,
                market_risk=MarketCondition.EXTREMELY_VOLATILE,
                overall_risk_score=100.0
            )
    
    async def _assess_leverage_risk(self, 
                                  market_metrics: MarketMetrics,
                                  position_metrics: Optional[PositionMetrics]) -> float:
        """레버리지 리스크 점수 계산 (0-100)"""
        try:
            risk_score = 0.0
            
            # 현재 레버리지 수준 (40%)
            current_leverage = self.leverage_manager.current_leverage
            max_leverage = self.config.leverage_config.max_leverage
            leverage_ratio = current_leverage / max_leverage
            risk_score += leverage_ratio * 40
            
            # 시장 변동성 기반 (30%)
            volatility_risk = min(market_metrics.volatility_1h / 0.1, 1.0)  # 10% 변동성을 100%로 정규화
            risk_score += volatility_risk * 30
            
            # 포지션 크기 기반 (20%) 
            if position_metrics:
                position_size_ratio = abs(position_metrics.size * position_metrics.mark_price) / 100000  # 10만 달러 기준
                position_risk = min(position_size_ratio, 1.0)
                risk_score += position_risk * 20
            
            # 최근 성과 기반 (10%)
            if self.leverage_manager.performance_history:
                recent_performance = np.mean(self.leverage_manager.performance_history[-10:])
                performance_risk = max(0, (0.5 - recent_performance) * 2)  # 50% 이하 성과를 리스크로 변환
                risk_score += performance_risk * 10
            
            return min(100.0, max(0.0, risk_score))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Leverage risk assessment error: {e}")
            return 50.0  # 에러 시 중간값
    
    async def _assess_margin_risk(self, margin_status: MarginStatus, margin_info) -> float:
        """마진 리스크 점수 계산 (0-100)"""
        try:
            # 마진 상태별 기본 점수
            status_scores = {
                MarginStatus.HEALTHY: 10.0,
                MarginStatus.WARNING: 30.0,
                MarginStatus.DANGER: 60.0,  
                MarginStatus.CRITICAL: 85.0,
                MarginStatus.LIQUIDATION_RISK: 100.0
            }
            
            base_score = status_scores.get(margin_status, 50.0)
            
            # 마진 비율 기반 조정
            if hasattr(margin_info, 'margin_ratio'):
                if margin_info.margin_ratio < 0.05:  # 5% 미만
                    base_score = max(base_score, 95.0)
                elif margin_info.margin_ratio < 0.1:  # 10% 미만
                    base_score = max(base_score, 80.0)
                elif margin_info.margin_ratio < 0.2:  # 20% 미만
                    base_score = max(base_score, 60.0)
            
            # 청산가 거리 기반 조정
            if hasattr(margin_info, 'liquidation_distance_pct'):
                if margin_info.liquidation_distance_pct < 0.05:  # 5% 미만
                    base_score = max(base_score, 100.0)
                elif margin_info.liquidation_distance_pct < 0.1:  # 10% 미만
                    base_score = max(base_score, 90.0)
                elif margin_info.liquidation_distance_pct < 0.2:  # 20% 미만
                    base_score = max(base_score, 70.0)
            
            return min(100.0, base_score)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin risk assessment error: {e}")
            return 50.0
    
    async def _assess_market_risk(self, market_metrics: MarketMetrics) -> float:
        """시장 리스크 점수 계산 (0-100)"""
        try:
            risk_score = 0.0
            
            # 변동성 기반 (40%)
            volatility_risk = min(market_metrics.volatility_1h / 0.1, 1.0)
            risk_score += volatility_risk * 40
            
            # 시장 상황 기반 (30%)
            condition_scores = {
                MarketCondition.EXTREMELY_VOLATILE: 100.0,
                MarketCondition.HIGH_VOLATILE: 70.0,
                MarketCondition.NORMAL: 30.0,
                MarketCondition.LOW_VOLATILE: 15.0,
                MarketCondition.CONSOLIDATION: 20.0
            }
            risk_score += condition_scores.get(market_metrics.market_condition, 50.0) * 0.3
            
            # RSI 극단치 기반 (15%)
            rsi_risk = 0.0
            if market_metrics.rsi_14 > 80 or market_metrics.rsi_14 < 20:
                rsi_risk = min(abs(market_metrics.rsi_14 - 50) / 30, 1.0)
            risk_score += rsi_risk * 15
            
            # 펀딩비율 기반 (10%)
            funding_risk = min(abs(market_metrics.funding_rate) / 0.01, 1.0)  # 1% 펀딩비율을 100%로 정규화
            risk_score += funding_risk * 10
            
            # ATR 기반 (5%)
            atr_risk = min(market_metrics.atr_14 / (market_metrics.price * 0.05), 1.0)  # 5% ATR을 100%로 정규화
            risk_score += atr_risk * 5
            
            return min(100.0, max(0.0, risk_score))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Market risk assessment error: {e}")
            return 50.0
    
    async def _calculate_overall_risk_score(self, 
                                          leverage_risk: float,
                                          margin_risk: float, 
                                          market_risk: float) -> float:
        """종합 리스크 점수 계산"""
        try:
            # 가중 평균 계산
            weights = {
                'margin_risk': 0.5,      # 마진 리스크가 가장 중요
                'leverage_risk': 0.3,    # 레버리지 리스크
                'market_risk': 0.2       # 시장 리스크
            }
            
            overall_score = (
                margin_risk * weights['margin_risk'] +
                leverage_risk * weights['leverage_risk'] +
                market_risk * weights['market_risk']
            )
            
            # 최대값 조정 (가장 높은 개별 리스크가 전체에 미치는 영향)
            max_individual_risk = max(leverage_risk, margin_risk, market_risk)
            if max_individual_risk > 80:
                overall_score = max(overall_score, max_individual_risk * 0.9)
            
            return min(100.0, max(0.0, overall_score))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Overall risk calculation error: {e}")
            return 75.0  # 에러 시 높은 값으로 안전하게 설정
    
    async def _determine_risk_level(self, overall_risk_score: float) -> SystemRiskLevel:
        """리스크 점수 기반 시스템 리스크 수준 결정"""
        if overall_risk_score >= 95:
            return SystemRiskLevel.EMERGENCY
        elif overall_risk_score >= self.config.critical_risk_score:
            return SystemRiskLevel.CRITICAL
        elif overall_risk_score >= self.config.high_risk_score:
            return SystemRiskLevel.VERY_HIGH
        elif overall_risk_score >= 55:
            return SystemRiskLevel.HIGH
        elif overall_risk_score >= self.config.moderate_risk_score:
            return SystemRiskLevel.MODERATE
        elif overall_risk_score >= 25:
            return SystemRiskLevel.LOW
        else:
            return SystemRiskLevel.VERY_LOW
    
    async def _generate_risk_actions(self, 
                                   overall_risk: float,
                                   leverage_risk: float,
                                   margin_risk: float, 
                                   market_risk: float) -> List[Dict[str, Any]]:
        """리스크 수준별 권장 조치 생성"""
        try:
            actions = []
            
            # 긴급 상황 조치
            if overall_risk >= self.config.emergency_close_risk_score:
                actions.append({
                    "priority": "EMERGENCY",
                    "action": "emergency_close_all_positions",
                    "reason": f"시스템 리스크 임계 초과: {overall_risk:.1f}점",
                    "urgency": 10,
                    "auto_executable": self.config.enable_emergency_protocols
                })
            
            # 포지션 축소 조치
            elif overall_risk >= self.config.force_position_reduction_risk_score:
                actions.append({
                    "priority": "CRITICAL",
                    "action": "reduce_position_size",
                    "reason": f"높은 리스크 수준: {overall_risk:.1f}점",
                    "urgency": 9,
                    "auto_executable": self.config.enable_auto_margin_management
                })
            
            # 레버리지 조정 조치
            if leverage_risk > 60:
                actions.append({
                    "priority": "HIGH",
                    "action": "reduce_leverage",
                    "reason": f"레버리지 리스크 과다: {leverage_risk:.1f}점",
                    "urgency": 7,
                    "auto_executable": self.config.enable_auto_leverage_adjustment
                })
            
            # 마진 추가 조치
            if margin_risk > 70:
                actions.append({
                    "priority": "HIGH", 
                    "action": "add_margin",
                    "reason": f"마진 리스크 증가: {margin_risk:.1f}점",
                    "urgency": 8,
                    "auto_executable": self.config.enable_auto_margin_management
                })
            
            # 시장 리스크 대응
            if market_risk > 80:
                actions.append({
                    "priority": "MEDIUM",
                    "action": "pause_new_positions",
                    "reason": f"시장 변동성 과다: {market_risk:.1f}점",
                    "urgency": 6,
                    "auto_executable": True
                })
            
            # 모니터링 강화
            if overall_risk > 50:
                actions.append({
                    "priority": "LOW",
                    "action": "increase_monitoring_frequency",
                    "reason": f"리스크 수준 상승: {overall_risk:.1f}점",
                    "urgency": 3,
                    "auto_executable": True
                })
            
            # 우선순위 정렬
            actions.sort(key=lambda x: x["urgency"], reverse=True)
            
            return actions
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk actions generation error: {e}")
            return []
    
    async def _determine_trading_restrictions(self, overall_risk_score: float) -> Dict[str, bool]:
        """리스크 수준별 거래 제한사항 결정"""
        try:
            restrictions = {
                "new_positions_allowed": True,
                "position_increase_allowed": True,
                "leverage_increase_allowed": True,
                "high_risk_strategies_allowed": True,
                "emergency_only_mode": False
            }
            
            if overall_risk_score >= self.config.emergency_close_risk_score:
                # 긴급 상황: 모든 거래 금지
                restrictions.update({
                    "new_positions_allowed": False,
                    "position_increase_allowed": False,
                    "leverage_increase_allowed": False,
                    "high_risk_strategies_allowed": False,
                    "emergency_only_mode": True
                })
            
            elif overall_risk_score >= self.config.force_position_reduction_risk_score:
                # 임계 상황: 신규 포지션 및 증가 금지
                restrictions.update({
                    "new_positions_allowed": False,
                    "position_increase_allowed": False,
                    "leverage_increase_allowed": False,
                    "high_risk_strategies_allowed": False
                })
            
            elif overall_risk_score >= self.config.restrict_new_positions_risk_score:
                # 위험 상황: 신규 포지션 금지
                restrictions.update({
                    "new_positions_allowed": False,
                    "leverage_increase_allowed": False,
                    "high_risk_strategies_allowed": False
                })
            
            elif overall_risk_score >= self.config.high_risk_score:
                # 주의 상황: 고위험 전략 제한
                restrictions.update({
                    "high_risk_strategies_allowed": False,
                    "leverage_increase_allowed": False
                })
            
            return restrictions
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Trading restrictions determination error: {e}")
            
            # 에러 시 모든 거래 제한
            return {
                "new_positions_allowed": False,
                "position_increase_allowed": False,
                "leverage_increase_allowed": False,
                "high_risk_strategies_allowed": False,
                "emergency_only_mode": True
            }
    
    async def _update_risk_state(self, risk_state: SystemRiskState):
        """리스크 상태 업데이트 및 히스토리 기록"""
        try:
            # 상태 변화 확인
            state_changed = (
                self.current_risk_state.overall_risk_level != risk_state.overall_risk_level or
                abs(self.current_risk_state.overall_risk_score - risk_state.overall_risk_score) > 5.0
            )
            
            # 상태 업데이트
            self.current_risk_state = risk_state
            
            # 히스토리 기록
            self.risk_history.append({
                "timestamp": datetime.now(),
                "risk_state": risk_state,
                "state_changed": state_changed
            })
            
            # 히스토리 크기 제한
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            # 중요한 상태 변화 시 로깅
            if state_changed and self.logger:
                self.logger.info(
                    f"Risk state changed: {risk_state.overall_risk_level.value} "
                    f"(score: {risk_state.overall_risk_score:.1f})",
                    leverage_risk=risk_state.leverage_risk_score,
                    margin_risk=risk_state.margin_risk_score,
                    market_risk=risk_state.market_risk_score
                )
            
            # 임계 상황 시 Tagged 로깅
            if (risk_state.overall_risk_level in [SystemRiskLevel.CRITICAL, SystemRiskLevel.EMERGENCY] 
                and self.log_integrator):
                
                severity = "critical" if risk_state.overall_risk_level == SystemRiskLevel.EMERGENCY else "high"
                
                await self.log_integrator.log_security_event(
                    event_type="system_risk_alert",
                    severity=severity,
                    description=f"System risk level: {risk_state.overall_risk_level.value}",
                    overall_risk_score=risk_state.overall_risk_score,
                    leverage_risk_score=risk_state.leverage_risk_score,
                    margin_risk_score=risk_state.margin_risk_score,
                    market_risk_score=risk_state.market_risk_score,
                    recommended_actions_count=len(risk_state.recommended_actions),
                    trading_restrictions=risk_state.trading_restrictions
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk state update error: {e}")
    
    async def execute_risk_actions(self, 
                                 risk_state: SystemRiskState,
                                 market_metrics: MarketMetrics,
                                 position_metrics: Optional[PositionMetrics],
                                 account_balance: float) -> List[Dict[str, Any]]:
        """리스크 관리 조치 실행"""
        try:
            execution_results = []
            
            for action in risk_state.recommended_actions:
                if not action.get("auto_executable", False):
                    continue
                
                try:
                    result = await self._execute_single_action(
                        action, market_metrics, position_metrics, account_balance
                    )
                    
                    execution_results.append({
                        "action": action["action"],
                        "success": result["success"],
                        "message": result["message"],
                        "timestamp": datetime.now()
                    })
                    
                    if result["success"]:
                        # 성공 시 통계 업데이트
                        if "leverage" in action["action"]:
                            self.stats["leverage_adjustments"] += 1
                        elif "margin" in action["action"]:
                            self.stats["margin_actions"] += 1
                        elif "emergency" in action["action"]:
                            self.stats["emergency_interventions"] += 1
                    
                    # 액션 간 지연
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    execution_results.append({
                        "action": action["action"],
                        "success": False,
                        "message": f"실행 오류: {str(e)}",
                        "timestamp": datetime.now()
                    })
            
            return execution_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk actions execution error: {e}")
            return []
    
    async def _execute_single_action(self, 
                                   action: Dict[str, Any],
                                   market_metrics: MarketMetrics,
                                   position_metrics: Optional[PositionMetrics],
                                   account_balance: float) -> Dict[str, Any]:
        """개별 리스크 조치 실행"""
        try:
            action_type = action["action"]
            
            if action_type == "reduce_leverage":
                # 레버리지 감소
                target_leverage = self.leverage_manager.current_leverage * 0.8
                optimal_leverage, _ = await self.leverage_manager.calculate_optimal_leverage(
                    market_metrics, position_metrics, 0.5
                )
                
                success = await self.leverage_manager.adjust_leverage(
                    min(target_leverage, optimal_leverage), position_metrics
                )
                
                return {
                    "success": success,
                    "message": f"레버리지 조정: {target_leverage:.2f}x"
                }
            
            elif action_type == "add_margin" and position_metrics:
                # 마진 추가
                success, message = await self.margin_manager.execute_margin_action(
                    MarginAction.ADD_MARGIN, position_metrics, account_balance
                )
                
                return {"success": success, "message": message}
            
            elif action_type == "reduce_position_size" and position_metrics:
                # 포지션 축소
                success, message = await self.margin_manager.execute_margin_action(
                    MarginAction.REDUCE_POSITION, position_metrics, account_balance
                )
                
                return {"success": success, "message": message}
            
            elif action_type == "emergency_close_all_positions" and position_metrics:
                # 긴급 청산
                success, message = await self.margin_manager.execute_margin_action(
                    MarginAction.EMERGENCY_CLOSE, position_metrics, account_balance
                )
                
                if success:
                    self.stats["positions_protected"] += 1
                
                return {"success": success, "message": message}
            
            else:
                return {
                    "success": False,
                    "message": f"지원하지 않는 액션: {action_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"액션 실행 오류: {str(e)}"
            }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """리스크 관리 요약 정보"""
        try:
            current_state = self.current_risk_state
            
            summary = {
                "current_risk_level": current_state.overall_risk_level.value,
                "overall_risk_score": current_state.overall_risk_score,
                "component_scores": {
                    "leverage_risk": current_state.leverage_risk_score,
                    "margin_risk": current_state.margin_risk_score,
                    "market_risk": current_state.market_risk_score
                },
                "active_restrictions": current_state.trading_restrictions,
                "pending_actions": len(current_state.recommended_actions),
                "high_priority_actions": len([
                    a for a in current_state.recommended_actions 
                    if a.get("urgency", 0) >= 8
                ]),
                "statistics": self.stats.copy(),
                "leverage_info": self.leverage_manager.get_leverage_stats(),
                "margin_info": self.margin_manager.get_margin_stats()
            }
            
            # 최근 리스크 트렌드
            if len(self.risk_history) >= 10:
                recent_scores = [h["risk_state"].overall_risk_score for h in self.risk_history[-10:]]
                summary["risk_trend"] = {
                    "current": recent_scores[-1],
                    "avg_10": np.mean(recent_scores),
                    "trend_direction": "increasing" if recent_scores[-1] > recent_scores[0] else "decreasing"
                }
            
            return summary
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk summary error: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """통합 리스크 모니터링 시작"""
        if not self.is_monitoring:
            self.is_monitoring = True
            
            # 하위 관리자들 모니터링 시작
            await self.leverage_manager.start_monitoring()
            await self.margin_manager.start_monitoring()
            
            # 통합 모니터링 시작
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.logger:
                self.logger.info("Integrated risk monitoring started")
    
    async def stop_monitoring(self):
        """통합 리스크 모니터링 중지"""
        self.is_monitoring = False
        
        # 하위 관리자들 모니터링 중지
        await self.leverage_manager.stop_monitoring()
        await self.margin_manager.stop_monitoring()
        
        # 통합 모니터링 중지
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.logger:
            self.logger.info("Integrated risk monitoring stopped")
    
    async def _monitoring_loop(self):
        """통합 모니터링 루프"""
        try:
            while self.is_monitoring:
                # 주기적으로 시스템 리스크 평가
                # 실제 구현에서는 시장 데이터와 포지션 정보를 가져와서 처리
                await asyncio.sleep(self.config.risk_assessment_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk monitoring loop error: {e}")

# 팩토리 함수
def create_futures_risk_coordinator(config: Optional[FuturesRiskConfig] = None) -> FuturesRiskCoordinator:
    """VPS 최적화된 선물 리스크 코디네이터 생성"""
    if config is None:
        config = FuturesRiskConfig()
    
    return FuturesRiskCoordinator(config, enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_risk_coordinator():
        config = FuturesRiskConfig()
        coordinator = create_futures_risk_coordinator(config)
        
        # 테스트 데이터
        market_metrics = MarketMetrics(
            price=50000.0,
            volatility_1h=0.08,  # 높은 변동성
            volatility_4h=0.06,
            volatility_24h=0.05,
            rsi_14=75.0,  # 과매수
            funding_rate=0.015,  # 높은 펀딩비율
            market_condition=MarketCondition.HIGH_VOLATILE
        )
        
        position_metrics = PositionMetrics(
            symbol="BTCUSDT",
            side="LONG",
            size=0.2,  # 큰 포지션
            entry_price=50000.0,
            mark_price=49000.0,  # 손실 상태
            unrealized_pnl=-200.0,
            margin_ratio=0.08,  # 낮은 마진
            liquidation_price=47000.0
        )
        
        # 리스크 평가
        risk_state = await coordinator.assess_system_risk(
            market_metrics=market_metrics,
            position_metrics=position_metrics,
            account_balance=10000.0,
            current_price=49000.0
        )
        
        print(f"Overall risk level: {risk_state.overall_risk_level.value}")
        print(f"Risk score: {risk_state.overall_risk_score:.1f}")
        print(f"Component scores:")
        print(f"  - Leverage: {risk_state.leverage_risk_score:.1f}")
        print(f"  - Margin: {risk_state.margin_risk_score:.1f}")
        print(f"  - Market: {risk_state.market_risk_score:.1f}")
        
        print(f"\nTrading restrictions:")
        for restriction, allowed in risk_state.trading_restrictions.items():
            print(f"  - {restriction}: {'Allowed' if allowed else 'Restricted'}")
        
        print(f"\nRecommended actions ({len(risk_state.recommended_actions)}):")
        for action in risk_state.recommended_actions:
            print(f"  - {action['priority']}: {action['action']} (urgency: {action['urgency']})")
            print(f"    Reason: {action['reason']}")
        
        # 요약 정보
        summary = coordinator.get_risk_summary()
        print("\nRisk summary:", json.dumps(summary, indent=2, default=str))
    
    asyncio.run(test_risk_coordinator())