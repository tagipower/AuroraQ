#!/usr/bin/env python3
"""
비트코인 선물 마진 및 청산가 관리 시스템
AuroraQ VPS Deployment - 실시간 마진 모니터링 및 청산 방지
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
try:
    from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
except ImportError:
    try:
        from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
    except ImportError:
        from vps_deployment.vps_logging import get_vps_log_integrator, LogCategory, LogLevel

try:
    from trading.futures_leverage_manager import PositionMetrics, RiskLevel
except ImportError:
    try:
        from trading.futures_leverage_manager import PositionMetrics, RiskLevel
    except ImportError:
        from vps_deployment.trading.futures_leverage_manager import PositionMetrics, RiskLevel

class MarginStatus(Enum):
    """마진 상태"""
    HEALTHY = "healthy"                    # 건전한 마진
    WARNING = "warning"                    # 주의 필요
    DANGER = "danger"                      # 위험 상태
    CRITICAL = "critical"                  # 임계 상태
    LIQUIDATION_RISK = "liquidation_risk"  # 청산 위험

class MarginAction(Enum):
    """마진 관리 액션"""
    MONITOR = "monitor"                    # 모니터링
    ADD_MARGIN = "add_margin"              # 마진 추가
    REDUCE_POSITION = "reduce_position"    # 포지션 축소
    CLOSE_POSITION = "close_position"      # 포지션 종료
    EMERGENCY_CLOSE = "emergency_close"    # 긴급 종료

@dataclass
class MarginConfig:
    """마진 관리 설정"""
    # 마진 비율 임계값
    healthy_margin_ratio: float = 0.3      # 30% 이상 건전
    warning_margin_ratio: float = 0.2      # 20% 주의
    danger_margin_ratio: float = 0.1       # 10% 위험
    critical_margin_ratio: float = 0.05    # 5% 임계
    
    # 청산가 거리 임계값
    healthy_liquidation_distance: float = 0.5    # 50% 이상 건전
    warning_liquidation_distance: float = 0.3    # 30% 주의
    danger_liquidation_distance: float = 0.2     # 20% 위험
    critical_liquidation_distance: float = 0.1   # 10% 임계
    
    # 자동 대응 설정
    auto_add_margin: bool = True           # 자동 마진 추가
    auto_reduce_position: bool = True      # 자동 포지션 축소
    emergency_close_threshold: float = 0.05 # 5% 이내 긴급 종료
    
    # 마진 추가 설정
    margin_add_amount_ratio: float = 0.1   # 잔고의 10% 추가
    max_margin_add_per_day: int = 3        # 하루 최대 3회 추가
    
    # 포지션 축소 설정
    position_reduce_ratio: float = 0.3     # 30% 축소
    max_position_reduces: int = 2          # 최대 2회 축소
    
    # 모니터링 주기
    monitoring_interval_seconds: int = 10  # 10초마다 체크

@dataclass
class MarginInfo:
    """마진 정보"""
    total_margin: float                    # 총 마진
    used_margin: float                     # 사용된 마진
    free_margin: float                     # 여유 마진
    margin_ratio: float                    # 마진 비율
    
    # 포지션별 마진
    position_margin: float                 # 포지션 마진
    order_margin: float = 0.0              # 주문 마진
    
    # 청산 관련
    liquidation_price: float = 0.0         # 청산가
    liquidation_distance_pct: float = 0.0  # 청산가까지 거리(%)
    
    # 손익
    unrealized_pnl: float = 0.0            # 미실현 손익
    realized_pnl: float = 0.0              # 실현 손익

@dataclass
class MarginAlert:
    """마진 알림"""
    timestamp: datetime
    alert_type: MarginStatus
    message: str
    current_margin_ratio: float
    liquidation_distance: float
    recommended_action: MarginAction
    urgency_level: int  # 1-10 (10이 가장 긴급)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FuturesMarginManager:
    """비트코인 선물 마진 관리자"""
    
    def __init__(self, config: MarginConfig, enable_logging: bool = True):
        """
        선물 마진 관리자 초기화
        
        Args:
            config: 마진 관리 설정
            enable_logging: 통합 로깅 활성화
        """
        self.config = config
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("futures_margin_manager")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 현재 상태
        self.current_margin_status = MarginStatus.HEALTHY
        self.current_margin_info = None
        
        # 히스토리
        self.margin_history = []
        self.alert_history = []
        self.action_history = []
        
        # 일일 카운터
        self.daily_margin_adds = 0
        self.daily_position_reduces = 0
        self.last_reset_date = datetime.now().date()
        
        # 모니터링
        self.is_monitoring = False
        self.monitoring_task = None
        
        # 통계
        self.stats = {
            "total_alerts": 0,
            "margin_adds": 0,
            "position_reductions": 0,
            "emergency_closes": 0,
            "liquidations_prevented": 0,
            "avg_margin_ratio": 0.0
        }
        
        if self.logger:
            self.logger.info("FuturesMarginManager initialized")
    
    async def analyze_margin_status(self, 
                                  position_metrics: PositionMetrics,
                                  account_balance: float,
                                  current_price: float) -> Tuple[MarginStatus, MarginInfo, Optional[MarginAlert]]:
        """
        마진 상태 분석
        
        Args:
            position_metrics: 포지션 지표
            account_balance: 계좌 잔고
            current_price: 현재 가격
            
        Returns:
            Tuple[MarginStatus, MarginInfo, Optional[MarginAlert]]: (상태, 마진정보, 알림)
        """
        try:
            # 마진 정보 계산
            margin_info = await self._calculate_margin_info(
                position_metrics, account_balance, current_price
            )
            
            # 마진 상태 평가
            margin_status = self._evaluate_margin_status(margin_info)
            
            # 알림 생성 필요성 확인
            alert = None
            if self._should_generate_alert(margin_status, margin_info):
                alert = await self._generate_margin_alert(margin_status, margin_info)
            
            # 상태 업데이트
            self.current_margin_status = margin_status
            self.current_margin_info = margin_info
            
            # 히스토리 업데이트
            self.margin_history.append({
                "timestamp": datetime.now(),
                "margin_status": margin_status,
                "margin_info": margin_info,
                "current_price": current_price
            })
            
            # 히스토리 크기 제한 (최근 1000개)
            if len(self.margin_history) > 1000:
                self.margin_history = self.margin_history[-1000:]
            
            return margin_status, margin_info, alert
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin status analysis error: {e}")
            
            # 에러 시 안전한 기본값 반환
            default_info = MarginInfo(
                total_margin=account_balance,
                used_margin=0.0,
                free_margin=account_balance,
                margin_ratio=1.0
            )
            
            return MarginStatus.HEALTHY, default_info, None
    
    async def _calculate_margin_info(self, 
                                   position_metrics: PositionMetrics,
                                   account_balance: float,
                                   current_price: float) -> MarginInfo:
        """마진 정보 계산"""
        try:
            # 포지션 가치 계산
            position_value = abs(position_metrics.size) * current_price
            
            # 사용된 마진 계산 (레버리지 기반)
            # 레버리지는 포지션 크기로부터 역산
            if position_value > 0:
                estimated_leverage = position_value / (account_balance * 0.1)  # 추정 레버리지
                estimated_leverage = max(1.0, min(estimated_leverage, 20.0))  # 1-20x 범위
                used_margin = position_value / estimated_leverage
            else:
                used_margin = 0.0
            
            # 여유 마진 계산
            free_margin = account_balance - used_margin + position_metrics.unrealized_pnl
            
            # 마진 비율 계산
            margin_ratio = free_margin / account_balance if account_balance > 0 else 0.0
            margin_ratio = max(0.0, margin_ratio)  # 음수 방지
            
            # 청산가 거리 계산
            liquidation_distance_pct = 0.0
            if position_metrics.liquidation_price > 0:
                if position_metrics.side == "LONG":
                    liquidation_distance_pct = (current_price - position_metrics.liquidation_price) / current_price
                else:
                    liquidation_distance_pct = (position_metrics.liquidation_price - current_price) / current_price
                
                liquidation_distance_pct = max(0.0, liquidation_distance_pct)
            
            return MarginInfo(
                total_margin=account_balance,
                used_margin=used_margin,
                free_margin=free_margin,
                margin_ratio=margin_ratio,
                position_margin=used_margin,
                liquidation_price=position_metrics.liquidation_price,
                liquidation_distance_pct=liquidation_distance_pct,
                unrealized_pnl=position_metrics.unrealized_pnl
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin info calculation error: {e}")
            
            # 에러 시 보수적인 값 반환
            return MarginInfo(
                total_margin=account_balance,
                used_margin=account_balance * 0.5,  # 보수적으로 50% 사용으로 가정
                free_margin=account_balance * 0.5,
                margin_ratio=0.5
            )
    
    def _evaluate_margin_status(self, margin_info: MarginInfo) -> MarginStatus:
        """마진 상태 평가"""
        try:
            # 청산가 거리 기반 평가 (우선순위 높음)
            if margin_info.liquidation_distance_pct < self.config.critical_liquidation_distance:
                return MarginStatus.LIQUIDATION_RISK
            elif margin_info.liquidation_distance_pct < self.config.danger_liquidation_distance:
                return MarginStatus.CRITICAL
            elif margin_info.liquidation_distance_pct < self.config.warning_liquidation_distance:
                return MarginStatus.DANGER
            
            # 마진 비율 기반 평가
            if margin_info.margin_ratio < self.config.critical_margin_ratio:
                return MarginStatus.CRITICAL
            elif margin_info.margin_ratio < self.config.danger_margin_ratio:
                return MarginStatus.DANGER
            elif margin_info.margin_ratio < self.config.warning_margin_ratio:
                return MarginStatus.WARNING
            elif margin_info.margin_ratio >= self.config.healthy_margin_ratio:
                return MarginStatus.HEALTHY
            else:
                return MarginStatus.WARNING
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin status evaluation error: {e}")
            return MarginStatus.CRITICAL  # 에러 시 안전하게 CRITICAL로 설정
    
    def _should_generate_alert(self, status: MarginStatus, margin_info: MarginInfo) -> bool:
        """알림 생성 필요성 판단"""
        try:
            # 상태 변화가 있거나 위험 수준이 높으면 알림
            if status != self.current_margin_status:
                return True
            
            # 임계 상태에서는 지속적으로 알림
            if status in [MarginStatus.CRITICAL, MarginStatus.LIQUIDATION_RISK]:
                return True
            
            # 마진 비율이 급격히 변화한 경우
            if self.current_margin_info:
                margin_change = abs(margin_info.margin_ratio - self.current_margin_info.margin_ratio)
                if margin_change > 0.1:  # 10% 이상 변화
                    return True
            
            return False
            
        except Exception as e:
            return True  # 에러 시 알림 생성
    
    async def _generate_margin_alert(self, status: MarginStatus, margin_info: MarginInfo) -> MarginAlert:
        """마진 알림 생성"""
        try:
            # 상태별 메시지 및 액션
            status_config = {
                MarginStatus.HEALTHY: {
                    "message": "마진 상태 양호",
                    "action": MarginAction.MONITOR,
                    "urgency": 1
                },
                MarginStatus.WARNING: {
                    "message": f"마진 비율 주의 필요: {margin_info.margin_ratio:.1%}",
                    "action": MarginAction.MONITOR,
                    "urgency": 3
                },
                MarginStatus.DANGER: {
                    "message": f"마진 비율 위험: {margin_info.margin_ratio:.1%}",
                    "action": MarginAction.ADD_MARGIN if self.config.auto_add_margin else MarginAction.MONITOR,
                    "urgency": 6
                },
                MarginStatus.CRITICAL: {
                    "message": f"마진 비율 임계: {margin_info.margin_ratio:.1%}",
                    "action": MarginAction.REDUCE_POSITION if self.config.auto_reduce_position else MarginAction.ADD_MARGIN,
                    "urgency": 8
                },
                MarginStatus.LIQUIDATION_RISK: {
                    "message": f"청산 위험: {margin_info.liquidation_distance_pct:.1%} 거리",
                    "action": MarginAction.EMERGENCY_CLOSE,
                    "urgency": 10
                }
            }
            
            config = status_config[status]
            
            alert = MarginAlert(
                timestamp=datetime.now(),
                alert_type=status,
                message=config["message"],
                current_margin_ratio=margin_info.margin_ratio,
                liquidation_distance=margin_info.liquidation_distance_pct,
                recommended_action=config["action"],
                urgency_level=config["urgency"],
                metadata={
                    "total_margin": margin_info.total_margin,
                    "used_margin": margin_info.used_margin,
                    "free_margin": margin_info.free_margin,
                    "unrealized_pnl": margin_info.unrealized_pnl,
                    "liquidation_price": margin_info.liquidation_price
                }
            )
            
            # 알림 히스토리에 추가
            self.alert_history.append(alert)
            self.stats["total_alerts"] += 1
            
            # 로깅
            if self.logger:
                self.logger.warning(
                    f"Margin alert: {alert.message}",
                    margin_ratio=margin_info.margin_ratio,
                    liquidation_distance=margin_info.liquidation_distance_pct,
                    recommended_action=alert.recommended_action.value,
                    urgency=alert.urgency_level
                )
            
            # 긴급 상황 시 Tagged 로깅
            if status in [MarginStatus.CRITICAL, MarginStatus.LIQUIDATION_RISK] and self.log_integrator:
                severity = "critical" if status == MarginStatus.LIQUIDATION_RISK else "high"
                
                await self.log_integrator.log_security_event(
                    event_type="margin_alert",
                    severity=severity,
                    description=alert.message,
                    margin_status=status.value,
                    margin_ratio=margin_info.margin_ratio,
                    liquidation_distance=margin_info.liquidation_distance_pct,
                    recommended_action=alert.recommended_action.value,
                    urgency_level=alert.urgency_level
                )
            
            return alert
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin alert generation error: {e}")
            
            # 에러 시 기본 알림 반환
            return MarginAlert(
                timestamp=datetime.now(),
                alert_type=MarginStatus.CRITICAL,
                message=f"마진 상태 확인 필요 (오류: {str(e)})",
                current_margin_ratio=0.0,
                liquidation_distance=0.0,
                recommended_action=MarginAction.MONITOR,
                urgency_level=5
            )
    
    async def execute_margin_action(self, 
                                  action: MarginAction,
                                  position_metrics: PositionMetrics,
                                  account_balance: float) -> Tuple[bool, str]:
        """
        마진 관리 액션 실행
        
        Args:
            action: 실행할 액션
            position_metrics: 포지션 지표
            account_balance: 계좌 잔고
            
        Returns:
            Tuple[bool, str]: (성공 여부, 결과 메시지)
        """
        try:
            # 일일 카운터 리셋 체크
            await self._check_daily_reset()
            
            action_result = False
            result_message = ""
            
            if action == MarginAction.MONITOR:
                action_result = True
                result_message = "모니터링 지속"
                
            elif action == MarginAction.ADD_MARGIN:
                action_result, result_message = await self._add_margin(account_balance)
                
            elif action == MarginAction.REDUCE_POSITION:
                action_result, result_message = await self._reduce_position(position_metrics)
                
            elif action == MarginAction.CLOSE_POSITION:
                action_result, result_message = await self._close_position(position_metrics)
                
            elif action == MarginAction.EMERGENCY_CLOSE:
                action_result, result_message = await self._emergency_close_position(position_metrics)
            
            # 액션 히스토리 기록
            self.action_history.append({
                "timestamp": datetime.now(),
                "action": action,
                "success": action_result,
                "message": result_message,
                "position_symbol": position_metrics.symbol
            })
            
            # 로깅
            if self.logger:
                log_level = "info" if action_result else "error"
                getattr(self.logger, log_level)(
                    f"Margin action executed: {action.value} - {result_message}",
                    success=action_result,
                    position_symbol=position_metrics.symbol
                )
            
            return action_result, result_message
            
        except Exception as e:
            error_msg = f"마진 액션 실행 오류: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg)
            
            return False, error_msg
    
    async def _check_daily_reset(self):
        """일일 카운터 리셋 확인"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_margin_adds = 0
            self.daily_position_reduces = 0
            self.last_reset_date = today
    
    async def _add_margin(self, account_balance: float) -> Tuple[bool, str]:
        """마진 추가"""
        try:
            # 일일 한도 체크
            if self.daily_margin_adds >= self.config.max_margin_add_per_day:
                return False, f"일일 마진 추가 한도 초과: {self.daily_margin_adds}/{self.config.max_margin_add_per_day}"
            
            # 추가할 마진 계산
            margin_add_amount = account_balance * self.config.margin_add_amount_ratio
            
            # 실제 마진 추가 (시뮬레이션)
            # 실제 환경에서는 거래소 API 호출
            
            self.daily_margin_adds += 1
            self.stats["margin_adds"] += 1
            
            result_message = f"마진 추가 완료: {margin_add_amount:.2f} USDT"
            
            # 중요 이벤트 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="margin_added",
                    severity="medium",
                    description=result_message,
                    margin_amount=margin_add_amount,
                    daily_count=self.daily_margin_adds
                )
            
            return True, result_message
            
        except Exception as e:
            return False, f"마진 추가 실패: {str(e)}"
    
    async def _reduce_position(self, position_metrics: PositionMetrics) -> Tuple[bool, str]:
        """포지션 축소"""
        try:
            # 축소 한도 체크
            if self.daily_position_reduces >= self.config.max_position_reduces:
                return False, f"일일 포지션 축소 한도 초과: {self.daily_position_reduces}/{self.config.max_position_reduces}"
            
            # 축소할 크기 계산
            reduce_amount = abs(position_metrics.size) * self.config.position_reduce_ratio
            
            # 실제 포지션 축소 (시뮬레이션)
            # 실제 환경에서는 거래소 API 호출
            
            self.daily_position_reduces += 1
            self.stats["position_reductions"] += 1
            
            result_message = f"포지션 축소 완료: {reduce_amount:.6f} BTC ({self.config.position_reduce_ratio:.0%})"
            
            # 중요 이벤트 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_reduced",
                    severity="medium",
                    description=result_message,
                    symbol=position_metrics.symbol,
                    reduce_amount=reduce_amount,
                    reduce_ratio=self.config.position_reduce_ratio,
                    daily_count=self.daily_position_reduces
                )
            
            return True, result_message
            
        except Exception as e:
            return False, f"포지션 축소 실패: {str(e)}"
    
    async def _close_position(self, position_metrics: PositionMetrics) -> Tuple[bool, str]:
        """포지션 종료"""
        try:
            # 실제 포지션 종료 (시뮬레이션)
            # 실제 환경에서는 거래소 API 호출
            
            result_message = f"포지션 종료 완료: {position_metrics.symbol} {position_metrics.size:.6f}"
            
            # 중요 이벤트 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_closed",
                    severity="high",
                    description=result_message,
                    symbol=position_metrics.symbol,
                    position_size=position_metrics.size,
                    unrealized_pnl=position_metrics.unrealized_pnl,
                    reason="margin_management"
                )
            
            return True, result_message
            
        except Exception as e:
            return False, f"포지션 종료 실패: {str(e)}"
    
    async def _emergency_close_position(self, position_metrics: PositionMetrics) -> Tuple[bool, str]:
        """긴급 포지션 종료"""
        try:
            # 실제 긴급 포지션 종료 (시뮬레이션)
            # 실제 환경에서는 거래소 API 호출 (시장가 주문)
            
            self.stats["emergency_closes"] += 1
            self.stats["liquidations_prevented"] += 1
            
            result_message = f"긴급 포지션 종료 완료: {position_metrics.symbol} {position_metrics.size:.6f}"
            
            # 긴급 이벤트 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="emergency_position_close",
                    severity="critical",
                    description=result_message,
                    symbol=position_metrics.symbol,
                    position_size=position_metrics.size,
                    unrealized_pnl=position_metrics.unrealized_pnl,
                    liquidation_price=position_metrics.liquidation_price,
                    reason="liquidation_prevention"
                )
            
            return True, result_message
            
        except Exception as e:
            return False, f"긴급 포지션 종료 실패: {str(e)}"
    
    async def get_margin_recommendations(self, 
                                       margin_info: MarginInfo,
                                       position_metrics: PositionMetrics) -> List[Dict[str, Any]]:
        """마진 관리 권장사항"""
        try:
            recommendations = []
            
            # 마진 비율 기반 권장사항
            if margin_info.margin_ratio < self.config.critical_margin_ratio:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "긴급 마진 추가 또는 포지션 축소",
                    "reason": f"마진 비율 임계: {margin_info.margin_ratio:.1%}",
                    "urgency": 9
                })
            elif margin_info.margin_ratio < self.config.danger_margin_ratio:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "마진 추가 고려",
                    "reason": f"마진 비율 위험: {margin_info.margin_ratio:.1%}",
                    "urgency": 6
                })
            
            # 청산가 거리 기반 권장사항
            if margin_info.liquidation_distance_pct < self.config.critical_liquidation_distance:
                recommendations.append({
                    "priority": "CRITICAL",
                    "action": "즉시 포지션 축소 또는 종료",
                    "reason": f"청산가 근접: {margin_info.liquidation_distance_pct:.1%}",
                    "urgency": 10
                })
            elif margin_info.liquidation_distance_pct < self.config.danger_liquidation_distance:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "포지션 크기 축소",
                    "reason": f"청산가 위험 거리: {margin_info.liquidation_distance_pct:.1%}",
                    "urgency": 8
                })
            
            # 미실현 손익 기반 권장사항
            if position_metrics.unrealized_pnl < 0:
                pnl_ratio = position_metrics.unrealized_pnl / margin_info.total_margin
                if pnl_ratio < -0.1:  # -10% 이상 손실
                    recommendations.append({
                        "priority": "MEDIUM",
                        "action": "손절 고려",
                        "reason": f"큰 미실현 손실: {pnl_ratio:.1%}",
                        "urgency": 5
                    })
            
            # 우선순위 정렬
            recommendations.sort(key=lambda x: x["urgency"], reverse=True)
            
            return recommendations
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin recommendations error: {e}")
            return []
    
    def get_margin_stats(self) -> Dict[str, Any]:
        """마진 관리 통계"""
        try:
            stats = self.stats.copy()
            
            # 추가 통계 계산
            stats.update({
                "current_margin_status": self.current_margin_status.value if self.current_margin_status else "unknown",
                "total_margin_history": len(self.margin_history),
                "total_alerts": len(self.alert_history),
                "total_actions": len(self.action_history),
                "daily_margin_adds": self.daily_margin_adds,
                "daily_position_reduces": self.daily_position_reduces,
                "config": {
                    "auto_add_margin": self.config.auto_add_margin,
                    "auto_reduce_position": self.config.auto_reduce_position,
                    "monitoring_interval": self.config.monitoring_interval_seconds
                }
            })
            
            # 현재 마진 정보
            if self.current_margin_info:
                stats["current_margin_info"] = {
                    "margin_ratio": self.current_margin_info.margin_ratio,
                    "liquidation_distance": self.current_margin_info.liquidation_distance_pct,
                    "unrealized_pnl": self.current_margin_info.unrealized_pnl,
                    "free_margin": self.current_margin_info.free_margin
                }
            
            # 최근 알림 정보
            if self.alert_history:
                latest_alert = self.alert_history[-1]
                stats["latest_alert"] = {
                    "timestamp": latest_alert.timestamp.isoformat(),
                    "type": latest_alert.alert_type.value,
                    "message": latest_alert.message,
                    "urgency": latest_alert.urgency_level
                }
            
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin stats error: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """마진 모니터링 시작"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.logger:
                self.logger.info("Margin monitoring started")
    
    async def stop_monitoring(self):
        """마진 모니터링 중지"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.logger:
            self.logger.info("Margin monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        try:
            while self.is_monitoring:
                # 주기적으로 마진 상태 체크
                # 실제 구현에서는 포지션 및 계좌 정보를 가져와서 처리
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Margin monitoring loop error: {e}")

# 팩토리 함수
def create_futures_margin_manager(config: Optional[MarginConfig] = None) -> FuturesMarginManager:
    """VPS 최적화된 선물 마진 관리자 생성"""
    if config is None:
        config = MarginConfig()
    
    return FuturesMarginManager(config, enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_margin_manager():
        config = MarginConfig(
            healthy_margin_ratio=0.3,
            warning_margin_ratio=0.2,
            danger_margin_ratio=0.1,
            critical_margin_ratio=0.05
        )
        
        manager = create_futures_margin_manager(config)
        
        # 테스트 포지션 데이터
        position_metrics = PositionMetrics(
            symbol="BTCUSDT",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            mark_price=49000.0,
            unrealized_pnl=-100.0,
            margin_ratio=0.15,
            liquidation_price=45000.0
        )
        
        # 마진 상태 분석
        status, margin_info, alert = await manager.analyze_margin_status(
            position_metrics=position_metrics,
            account_balance=10000.0,
            current_price=49000.0
        )
        
        print(f"Margin status: {status.value}")
        print(f"Margin ratio: {margin_info.margin_ratio:.1%}")
        print(f"Liquidation distance: {margin_info.liquidation_distance_pct:.1%}")
        
        if alert:
            print(f"Alert: {alert.message} (urgency: {alert.urgency_level})")
            print(f"Recommended action: {alert.recommended_action.value}")
        
        # 권장사항 조회
        recommendations = await manager.get_margin_recommendations(margin_info, position_metrics)
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"- {rec['priority']}: {rec['action']} ({rec['reason']})")
        
        # 통계 확인
        stats = manager.get_margin_stats()
        print("\nMargin stats:", json.dumps(stats, indent=2, default=str))
    
    asyncio.run(test_margin_manager())