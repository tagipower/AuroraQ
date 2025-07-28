#!/usr/bin/env python3
"""
실시간 거래 시스템 통합 모듈
AuroraQ_Production과 AuroraQ_Shared 모듈의 통합
"""

import sys
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import asyncio
import logging

# 상대 경로로 프로덕션 모듈 임포트
try:
    from ...AuroraQ_Production.core.realtime_system import RealtimeSystem
    from ...AuroraQ_Production.core.position_manager import PositionManager as ProductionPositionManager
    from ...AuroraQ_Production.strategies.strategy_manager import StrategyManager
except ImportError:
    # 개발 환경에서의 대체 임포트
    import warnings
    warnings.warn("프로덕션 모듈을 직접 임포트할 수 없습니다. 통합 모드에서 실행하세요.")
    
    class RealtimeSystem:
        def __init__(self): pass
    class ProductionPositionManager:
        def __init__(self): pass
    class StrategyManager:
        def __init__(self): pass

from ..risk_management import AdvancedRiskManager, RiskConfig, RiskMetrics
from ..position_management import UnifiedPositionManager


class ProductionRiskIntegration:
    """실시간 거래 시스템 리스크 관리 통합"""
    
    def __init__(self,
                 realtime_system: RealtimeSystem,
                 risk_config: Optional[RiskConfig] = None):
        
        self.realtime_system = realtime_system
        self.risk_config = risk_config or RiskConfig()
        self.logger = logging.getLogger(__name__)
        
        # 기존 프로덕션 포지션 관리자를 통합 관리자로 교체/통합
        self.unified_position_manager = self._integrate_position_managers()
        
        # 리스크 관리자
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.unified_position_manager,
            config=self.risk_config
        )
        
        # 실시간 모니터링 상태
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 성과 추적
        self.trading_session_start = datetime.now()
        self.session_metrics: Dict[str, Any] = {}
        
        # 콜백 및 이벤트 핸들러
        self._setup_risk_callbacks()
        self._setup_trading_callbacks()
    
    def _integrate_position_managers(self) -> UnifiedPositionManager:
        """기존 프로덕션 포지션 관리자와 통합"""
        
        # 실시간 시스템의 기존 설정 가져오기
        if hasattr(self.realtime_system, 'position_manager'):
            existing_manager = self.realtime_system.position_manager
            
            # 기존 설정으로 통합 관리자 초기화
            unified_manager = UnifiedPositionManager(
                initial_capital=getattr(existing_manager, 'initial_capital', 100000),
                commission_rate=getattr(existing_manager, 'commission_rate', 0.001),
                slippage_rate=getattr(existing_manager, 'slippage_rate', 0.0005)
            )
            
            # 기존 포지션 데이터 마이그레이션
            self._migrate_existing_positions(existing_manager, unified_manager)
            
        else:
            # 새로운 통합 관리자 생성
            unified_manager = UnifiedPositionManager(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
        
        return unified_manager
    
    def _migrate_existing_positions(self, 
                                  old_manager: ProductionPositionManager,
                                  new_manager: UnifiedPositionManager):
        """기존 포지션 데이터를 통합 관리자로 마이그레이션"""
        
        try:
            # 기존 포지션 정보 복사
            if hasattr(old_manager, 'positions'):
                for symbol, position_data in old_manager.positions.items():
                    # 기존 포지션을 통합 관리자 형식으로 변환
                    if hasattr(position_data, 'size') and position_data.size != 0:
                        # Trade 객체 생성하여 포지션 재구성
                        from ..position_management.position_models import Trade, OrderSide, TradeStatus
                        
                        trade = Trade(
                            symbol=symbol,
                            side=OrderSide.BUY if position_data.size > 0 else OrderSide.SELL,
                            size=abs(position_data.size),
                            price=getattr(position_data, 'avg_price', 0),
                            timestamp=datetime.now(),
                            status=TradeStatus.EXECUTED
                        )
                        
                        new_manager._update_position(trade)
            
            # 현금 및 자본 정보 복사
            if hasattr(old_manager, 'cash'):
                new_manager.cash = old_manager.cash
            
            if hasattr(old_manager, 'equity'):
                equity = old_manager.get_equity() if callable(getattr(old_manager, 'get_equity', None)) else old_manager.equity
                new_manager.current_capital = equity
            
            self.logger.info("기존 포지션 데이터 마이그레이션 완료")
            
        except Exception as e:
            self.logger.error(f"포지션 마이그레이션 실패: {e}")
    
    def _setup_risk_callbacks(self):
        """리스크 관리 콜백 설정"""
        
        def realtime_risk_callback(metrics: RiskMetrics, alerts):
            """실시간 리스크 모니터링 콜백"""
            
            # 로깅
            self.logger.info(f"리스크 지표 업데이트 - VaR: {metrics.var_95_pct:.2%}, 낙폭: {metrics.current_drawdown:.2%}")
            
            # 긴급 상황 체크
            emergency_actions = self._check_emergency_conditions(metrics, alerts)
            if emergency_actions:
                self._execute_emergency_actions(emergency_actions)
            
            # 세션 메트릭스 업데이트
            self._update_session_metrics(metrics)
        
        def emergency_callback(critical_alerts):
            """긴급 상황 콜백"""
            self.logger.critical(f"🚨 긴급 상황: {len(critical_alerts)}개의 심각한 리스크 알림")
            
            # 거래 중단
            self._emergency_trading_halt()
            
            # 알림 발송 (이메일, SMS 등)
            self._send_emergency_notifications(critical_alerts)
        
        self.risk_manager.add_risk_callback(realtime_risk_callback)
        self.risk_manager.add_emergency_callback(emergency_callback)
    
    def _setup_trading_callbacks(self):
        """거래 관련 콜백 설정"""
        
        # 실시간 시스템의 거래 콜백에 리스크 관리 추가
        if hasattr(self.realtime_system, 'add_trade_callback'):
            self.realtime_system.add_trade_callback(self._on_trade_executed)
        
        # 가격 업데이트 콜백
        if hasattr(self.realtime_system, 'add_price_callback'):
            self.realtime_system.add_price_callback(self._on_price_update)
    
    def _on_trade_executed(self, trade_data: Dict[str, Any]):
        """거래 실행 시 콜백"""
        
        try:
            # 통합 포지션 관리자 업데이트는 이미 내부적으로 처리됨
            # 추가 리스크 체크 및 로깅만 수행
            
            symbol = trade_data.get('symbol', '')
            side = trade_data.get('side', '')
            size = trade_data.get('size', 0)
            price = trade_data.get('price', 0)
            
            self.logger.info(f"거래 실행: {symbol} {side} {size}@{price}")
            
            # 실시간 리스크 체크
            self._realtime_risk_check()
            
        except Exception as e:
            self.logger.error(f"거래 실행 콜백 오류: {e}")
    
    def _on_price_update(self, price_data: Dict[str, float]):
        """가격 업데이트 시 콜백"""
        
        try:
            # 포지션 가격 업데이트
            self.unified_position_manager.update_multiple_prices(price_data)
            
            # 리스크 지표 업데이트 (주기적으로)
            if self._should_update_risk_metrics():
                self._realtime_risk_check()
                
        except Exception as e:
            self.logger.error(f"가격 업데이트 콜백 오류: {e}")
    
    def _should_update_risk_metrics(self) -> bool:
        """리스크 지표 업데이트 주기 확인"""
        # 예: 1분마다 또는 특정 조건에서 업데이트
        return True  # 단순화된 예시
    
    def _realtime_risk_check(self):
        """실시간 리스크 체크"""
        
        try:
            # 현재 포트폴리오 상태로 스냅샷 생성
            equity = self.unified_position_manager.get_equity()
            cash = self.unified_position_manager.cash
            positions = {
                symbol: pos.get_position_info() 
                for symbol, pos in self.unified_position_manager.positions.items()
            }
            prices = {
                symbol: pos.state.current_price 
                for symbol, pos in self.unified_position_manager.positions.items()
            }
            
            # 포트폴리오 스냅샷 업데이트
            snapshot = self.risk_manager.update_portfolio_snapshot(
                total_equity=equity,
                cash=cash,
                positions=positions,
                prices=prices
            )
            
            # 리스크 지표 계산 및 알림 체크
            metrics = self.risk_manager.calculate_risk_metrics(snapshot)
            alerts = self.risk_manager.check_risk_limits(metrics)
            
            # 새로운 알림 처리
            if alerts:
                self._handle_risk_alerts(alerts)
                
        except Exception as e:
            self.logger.error(f"실시간 리스크 체크 오류: {e}")
    
    def _handle_risk_alerts(self, alerts):
        """리스크 알림 처리"""
        
        for alert in alerts:
            if alert.risk_level.value == 'critical':
                self.logger.critical(f"🚨 심각한 리스크: {alert.title} - {alert.description}")
                # 즉시 대응 조치
                self._immediate_risk_response(alert)
            
            elif alert.risk_level.value == 'high':
                self.logger.warning(f"⚠️ 높은 리스크: {alert.title} - {alert.description}")
                # 예방적 조치
                self._preventive_risk_response(alert)
            
            else:
                self.logger.info(f"ℹ️ 리스크 알림: {alert.title} - {alert.description}")
    
    def _immediate_risk_response(self, alert):
        """즉시 리스크 대응"""
        
        if alert.alert_type.value == 'drawdown_limit':
            # 낙폭 한도 도달 - 포지션 축소
            reduction_factor = self.risk_config.drawdown_position_reduction
            self._reduce_all_positions(reduction_factor, "낙폭 한도 도달")
        
        elif alert.alert_type.value == 'var_breach':
            # VaR 한도 초과 - 고위험 포지션 축소
            self._reduce_high_risk_positions(0.3, "VaR 한도 초과")
    
    def _preventive_risk_response(self, alert):
        """예방적 리스크 대응"""
        
        if alert.alert_type.value == 'concentration_risk':
            # 집중도 위험 - 새로운 포지션 진입 제한
            self._limit_new_positions()
        
        elif alert.alert_type.value == 'volatility_spike':
            # 변동성 급등 - 포지션 크기 축소
            self._reduce_position_sizes(0.2, "변동성 급등")
    
    def _reduce_all_positions(self, reduction_factor: float, reason: str):
        """모든 포지션 축소"""
        
        for symbol in list(self.unified_position_manager.positions.keys()):
            try:
                position = self.unified_position_manager.positions[symbol]
                reduce_size = abs(position.size) * reduction_factor
                
                self.unified_position_manager.close_position(
                    symbol, 
                    size=reduce_size,
                    reason=f"Risk management: {reason}"
                )
                
                self.logger.info(f"포지션 축소: {symbol} -{reduce_size} ({reason})")
                
            except Exception as e:
                self.logger.error(f"포지션 축소 실패 {symbol}: {e}")
    
    def _reduce_high_risk_positions(self, reduction_factor: float, reason: str):
        """고위험 포지션 축소"""
        
        # 포지션별 리스크 기여도 계산하여 상위 리스크 포지션 식별
        # 간단한 예시: 큰 포지션부터 축소
        positions_by_size = sorted(
            self.unified_position_manager.positions.items(),
            key=lambda x: abs(x[1].size * x[1].state.current_price),
            reverse=True
        )
        
        # 상위 30% 포지션 축소
        top_positions_count = max(1, len(positions_by_size) // 3)
        for symbol, position in positions_by_size[:top_positions_count]:
            try:
                reduce_size = abs(position.size) * reduction_factor
                
                self.unified_position_manager.close_position(
                    symbol,
                    size=reduce_size,
                    reason=f"High risk reduction: {reason}"
                )
                
                self.logger.info(f"고위험 포지션 축소: {symbol} -{reduce_size}")
                
            except Exception as e:
                self.logger.error(f"고위험 포지션 축소 실패 {symbol}: {e}")
    
    def _emergency_trading_halt(self):
        """긴급 거래 중단"""
        
        try:
            # 실시간 시스템 거래 중단
            if hasattr(self.realtime_system, 'stop_trading'):
                self.realtime_system.stop_trading()
            
            # 모든 펜딩 주문 취소
            if hasattr(self.realtime_system, 'cancel_all_orders'):
                self.realtime_system.cancel_all_orders()
            
            self.logger.critical("🛑 긴급 거래 중단 실행됨")
            
        except Exception as e:
            self.logger.error(f"긴급 거래 중단 실패: {e}")
    
    def _send_emergency_notifications(self, alerts):
        """긴급 알림 발송"""
        
        # 이메일, SMS, 슬랙 등으로 알림 발송
        # 실제 구현에서는 외부 알림 서비스 연동
        
        message = f"🚨 AuroraQ 긴급 알림 🚨\n"
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"심각한 리스크 알림 {len(alerts)}개 발생\n\n"
        
        for alert in alerts:
            message += f"- {alert.title}: {alert.description}\n"
        
        # 로그에 기록 (실제로는 외부 알림 서비스로 전송)
        self.logger.critical(f"긴급 알림 메시지:\n{message}")
    
    def _check_emergency_conditions(self, metrics: RiskMetrics, alerts) -> List[str]:
        """긴급 상황 확인"""
        
        emergency_actions = []
        
        # 1. 극심한 낙폭
        if metrics.current_drawdown > self.risk_config.max_drawdown_limit * 1.2:
            emergency_actions.append("extreme_drawdown")
        
        # 2. VaR 극한 초과
        if metrics.var_95_pct > self.risk_config.var_limit_pct * 2:
            emergency_actions.append("extreme_var")
        
        # 3. 다중 심각한 알림
        critical_alerts = [a for a in alerts if a.risk_level.value == 'critical']
        if len(critical_alerts) >= 3:
            emergency_actions.append("multiple_critical_alerts")
        
        # 4. 시스템 오류
        if self._detect_system_anomalies():
            emergency_actions.append("system_anomaly")
        
        return emergency_actions
    
    def _detect_system_anomalies(self) -> bool:
        """시스템 이상 감지"""
        
        try:
            # 1. 포지션 관리자 상태 체크
            if not self.unified_position_manager:
                return True
            
            # 2. 실시간 시스템 연결 상태 체크
            if hasattr(self.realtime_system, 'is_connected'):
                if not self.realtime_system.is_connected():
                    return True
            
            # 3. 데이터 무결성 체크
            equity = self.unified_position_manager.get_equity()
            if equity <= 0 or np.isnan(equity) or np.isinf(equity):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"시스템 이상 감지 오류: {e}")
            return True
    
    def _execute_emergency_actions(self, actions: List[str]):
        """긴급 조치 실행"""
        
        for action in actions:
            try:
                if action == "extreme_drawdown":
                    self._reduce_all_positions(0.8, "극심한 낙폭")
                    
                elif action == "extreme_var":
                    self._reduce_all_positions(0.6, "극한 VaR 초과")
                    
                elif action == "multiple_critical_alerts":
                    self._emergency_trading_halt()
                    
                elif action == "system_anomaly":
                    self._emergency_trading_halt()
                    self._system_safety_check()
                
                self.logger.critical(f"긴급 조치 실행: {action}")
                
            except Exception as e:
                self.logger.error(f"긴급 조치 실행 실패 {action}: {e}")
    
    def _system_safety_check(self):
        """시스템 안전성 점검"""
        
        safety_report = {
            'timestamp': datetime.now(),
            'position_manager_status': self.unified_position_manager is not None,
            'risk_manager_status': self.risk_manager is not None,
            'equity': self.unified_position_manager.get_equity() if self.unified_position_manager else 0,
            'position_count': len(self.unified_position_manager.positions) if self.unified_position_manager else 0,
            'cash': self.unified_position_manager.cash if self.unified_position_manager else 0
        }
        
        self.logger.critical(f"시스템 안전성 점검 결과: {safety_report}")
    
    def _update_session_metrics(self, metrics: RiskMetrics):
        """세션 메트릭스 업데이트"""
        
        session_duration = (datetime.now() - self.trading_session_start).total_seconds() / 3600  # 시간 단위
        
        self.session_metrics.update({
            'session_duration_hours': session_duration,
            'current_equity': self.unified_position_manager.get_equity(),
            'max_var_95': max(self.session_metrics.get('max_var_95', 0), metrics.var_95_pct),
            'max_drawdown': max(self.session_metrics.get('max_drawdown', 0), metrics.current_drawdown),
            'total_trades': len(self.unified_position_manager.all_trades),
            'last_update': datetime.now()
        })
    
    async def start_monitoring(self):
        """실시간 리스크 모니터링 시작"""
        
        if self.is_monitoring:
            self.logger.warning("리스크 모니터링이 이미 실행 중입니다.")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("실시간 리스크 모니터링 시작")
    
    async def stop_monitoring(self):
        """실시간 리스크 모니터링 중단"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("실시간 리스크 모니터링 중단")
    
    async def _monitoring_loop(self):
        """리스크 모니터링 루프"""
        
        try:
            while self.is_monitoring:
                # 주기적 리스크 체크
                self._realtime_risk_check()
                
                # 5초마다 체크 (설정 가능)
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            self.logger.info("리스크 모니터링 루프 종료")
        except Exception as e:
            self.logger.error(f"리스크 모니터링 루프 오류: {e}")
    
    def get_realtime_dashboard(self) -> Dict[str, Any]:
        """실시간 리스크 대시보드"""
        
        # 기본 리스크 대시보드
        dashboard = self.risk_manager.get_risk_dashboard()
        
        # 실시간 특화 정보 추가
        dashboard.update({
            'session_metrics': self.session_metrics,
            'system_status': {
                'monitoring_active': self.is_monitoring,
                'realtime_system_connected': hasattr(self.realtime_system, 'is_connected') and 
                                           (self.realtime_system.is_connected() if callable(getattr(self.realtime_system, 'is_connected', None)) else True),
                'position_manager_healthy': self.unified_position_manager is not None,
                'last_risk_check': datetime.now()
            },
            'trading_status': {
                'total_trades_today': len(self.unified_position_manager.all_trades),
                'active_positions': len(self.unified_position_manager.positions),
                'current_equity': self.unified_position_manager.get_equity(),
                'available_capital': self.unified_position_manager.get_available_capital()
            }
        })
        
        return dashboard
    
    def export_session_report(self, output_path: str = None):
        """거래 세션 보고서 생성"""
        
        if output_path is None:
            output_path = f"trading_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 세션 보고서 데이터
        report_data = {
            'session_info': {
                'start_time': self.trading_session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': self.session_metrics.get('session_duration_hours', 0)
            },
            'performance_summary': self.unified_position_manager.get_performance_summary(),
            'risk_summary': self.session_metrics,
            'current_dashboard': self.get_realtime_dashboard(),
            'integration_status': self.get_integration_status()
        }
        
        # JSON 파일로 저장
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"거래 세션 보고서 생성: {output_path}")
        
        return report_data
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 확인"""
        
        status = {
            'unified_position_manager': {
                'initialized': self.unified_position_manager is not None,
                'equity': self.unified_position_manager.get_equity() if self.unified_position_manager else 0,
                'positions': len(self.unified_position_manager.positions) if self.unified_position_manager else 0,
                'trades': len(self.unified_position_manager.all_trades) if self.unified_position_manager else 0
            },
            'risk_manager': {
                'initialized': self.risk_manager is not None,
                'active_alerts': len(self.risk_manager.active_alerts) if self.risk_manager else 0,
                'snapshots': len(self.risk_manager.portfolio_snapshots) if self.risk_manager else 0
            },
            'realtime_system': {
                'connected': self.realtime_system is not None,
                'monitoring_active': self.is_monitoring
            },
            'session': {
                'start_time': self.trading_session_start,
                'metrics': self.session_metrics
            }
        }
        
        return status


# 편의 함수들
def create_production_risk_integration(realtime_system: RealtimeSystem,
                                     risk_config: Optional[RiskConfig] = None) -> ProductionRiskIntegration:
    """프로덕션 리스크 통합 시스템 생성"""
    
    if risk_config is None:
        risk_config = RiskConfig(
            var_limit_pct=0.03,  # 실시간에서는 더 보수적
            max_drawdown_limit=0.10,  # 10% 낙폭 한도
            drawdown_position_reduction=0.7,  # 70% 축소
            var_lookback_period=60,  # 짧은 기간
            correlation_lookback_period=30
        )
    
    integration = ProductionRiskIntegration(
        realtime_system=realtime_system,
        risk_config=risk_config
    )
    
    return integration


async def start_risk_aware_trading(realtime_system: RealtimeSystem,
                                 risk_config: Optional[RiskConfig] = None) -> ProductionRiskIntegration:
    """리스크 관리가 통합된 실시간 거래 시작"""
    
    # 통합 시스템 생성
    integration = create_production_risk_integration(realtime_system, risk_config)
    
    # 리스크 모니터링 시작
    await integration.start_monitoring()
    
    # 실시간 거래 시작
    if hasattr(realtime_system, 'start_trading'):
        await realtime_system.start_trading()
    
    return integration


class ProductionIntegration:
    """실시간 거래 시스템 통합 관리자
    
    ProductionRiskIntegration을 기반으로 한 통합 실시간 거래 시스템
    테스트 호환성을 위한 Wrapper 클래스
    """
    
    def __init__(self,
                 realtime_system: Optional[RealtimeSystem] = None,
                 position_manager: Optional[UnifiedPositionManager] = None,
                 risk_manager: Optional[AdvancedRiskManager] = None,
                 risk_config: Optional[RiskConfig] = None):
        
        # 실시간 시스템 설정
        if realtime_system is None:
            realtime_system = RealtimeSystem()
        
        # 포지션 관리자 설정
        if position_manager is None:
            position_manager = UnifiedPositionManager(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
        
        # 리스크 관리자 설정
        if risk_manager is None:
            if risk_config is None:
                risk_config = RiskConfig(
                    var_limit_pct=0.03,
                    max_drawdown_limit=0.10,
                    drawdown_position_reduction=0.7
                )
            risk_manager = AdvancedRiskManager(
                position_manager=position_manager,
                config=risk_config
            )
        
        # 프로덕션 리스크 통합 시스템을 내부적으로 사용
        self.risk_integration = ProductionRiskIntegration(
            realtime_system=realtime_system,
            risk_config=risk_manager.config if risk_manager else risk_config
        )
        
        # 외부 인터페이스
        self.realtime_system = realtime_system
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ProductionIntegration 초기화 완료")
    
    async def start_trading(self, **kwargs) -> Dict[str, Any]:
        """실시간 거래 시작"""
        return await start_risk_aware_trading(
            realtime_system=self.realtime_system,
            risk_config=self.risk_manager.config if self.risk_manager else None
        )
    
    async def stop_trading(self):
        """실시간 거래 중단"""
        await self.risk_integration.stop_monitoring()
        
        if hasattr(self.realtime_system, 'stop_trading'):
            await self.realtime_system.stop_trading()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 확인"""
        return self.risk_integration.get_integration_status()
    
    def get_realtime_dashboard(self) -> Dict[str, Any]:
        """실시간 대시보드"""
        return self.risk_integration.get_realtime_dashboard()
    
    def export_session_report(self, output_path: str = None):
        """세션 보고서 생성"""
        return self.risk_integration.export_session_report(output_path)


# 편의 함수 (기존 create_simple_production 등)
def create_simple_production(initial_capital: float = 100000) -> ProductionIntegration:
    """간단한 실시간 거래 시스템 생성"""
    
    # 기본 실시간 시스템
    realtime_system = RealtimeSystem()
    
    # 기본 포지션 관리자
    position_manager = UnifiedPositionManager(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # 기본 리스크 관리자
    risk_config = RiskConfig(
        var_limit_pct=0.03,
        max_drawdown_limit=0.10,
        drawdown_position_reduction=0.7
    )
    
    risk_manager = AdvancedRiskManager(
        position_manager=position_manager,
        config=risk_config
    )
    
    return ProductionIntegration(
        realtime_system=realtime_system,
        position_manager=position_manager,
        risk_manager=risk_manager,
        risk_config=risk_config
    )