#!/usr/bin/env python3
"""
실시간 하이브리드 시스템
PPO 강화학습 + 리스크 관리 + 실시간 보정을 통합한 완전한 실거래 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import asyncio
import threading
import time
import json
from pathlib import Path

from ..position_management.unified_position_manager import UnifiedPositionManager
from ..risk_management.advanced_risk_manager import AdvancedRiskManager
from ..risk_management.risk_models import RiskConfig, RiskLevel
from .realtime_calibration_system import RealtimeCalibrationSystem, RealtimeCalibrationConfig
from .backtest_integration import BacktestIntegration


@dataclass
class RealtimeSystemConfig:
    """실시간 시스템 설정"""
    # 기본 설정
    initial_capital: float = 1000000.0
    max_portfolio_risk: float = 0.02  # 2% VaR 한도
    position_sizing_method: str = "var_based"  # "fixed", "var_based", "kelly"
    
    # PPO 관련 설정
    ppo_model_path: Optional[str] = None
    strategy_selection_interval: int = 300  # 5분마다 전략 재평가
    performance_lookback_periods: int = 100
    
    # 실시간 데이터 설정
    data_update_interval: int = 1  # 1초마다 데이터 업데이트
    signal_generation_interval: int = 60  # 1분마다 신호 생성
    
    # 보정 시스템 설정
    enable_realtime_calibration: bool = True
    calibration_config: Optional[RealtimeCalibrationConfig] = None
    
    # 안전 장치
    daily_loss_limit: float = 0.05  # 일일 손실 한도 5%
    max_drawdown_limit: float = 0.10  # 최대 낙폭 한도 10%
    emergency_stop_enabled: bool = True
    
    # 로깅 및 모니터링
    log_directory: str = "logs/realtime"
    enable_performance_tracking: bool = True
    enable_real_time_dashboard: bool = True


@dataclass
class TradingSignal:
    """거래 신호"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    action: str = ""  # buy, sell, hold
    confidence: float = 0.0
    size_ratio: float = 0.0  # 포트폴리오 대비 비율
    reason: str = ""
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """시스템 상태"""
    running: bool = False
    start_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    last_data_update: Optional[datetime] = None
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    current_positions_count: int = 0
    risk_alerts: List[str] = field(default_factory=list)
    system_alerts: List[str] = field(default_factory=list)


class RealtimeHybridSystem:
    """실시간 하이브리드 거래 시스템"""
    
    def __init__(self, config: Optional[RealtimeSystemConfig] = None):
        self.config = config or RealtimeSystemConfig()
        self.state = SystemState()
        
        # 핵심 컴포넌트 초기화
        self._initialize_components()
        
        # 실시간 스레드 관리
        self.running = False
        self.data_thread = None
        self.signal_thread = None
        self.monitoring_thread = None
        
        # 콜백 시스템
        self.signal_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.risk_callbacks: List[Callable] = []
        
        # 성과 추적
        self.performance_history: List[Dict[str, Any]] = []
        self.signal_history: List[TradingSignal] = []
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # PPO 모델 (향후 로드)
        self.ppo_model = None
        self.strategy_selector = None
    
    def _initialize_components(self):
        """핵심 컴포넌트 초기화"""
        # 포지션 관리자
        self.position_manager = UnifiedPositionManager(
            initial_capital=self.config.initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # 리스크 관리자
        risk_config = RiskConfig()
        risk_config.var_limit_pct = self.config.max_portfolio_risk
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.position_manager,
            config=risk_config
        )
        
        # 백테스트 통합 시스템
        self.backtest_integration = BacktestIntegration(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager
        )
        
        # 실시간 보정 시스템
        if self.config.enable_realtime_calibration:
            calibration_config = self.config.calibration_config or RealtimeCalibrationConfig()
            self.calibration_system = RealtimeCalibrationSystem(
                position_manager=self.position_manager,
                risk_manager=self.risk_manager,
                config=calibration_config,
                log_directory=self.config.log_directory
            )
            
            # 보정 시스템 콜백 설정
            self.calibration_system.add_parameter_callback(self._on_parameter_calibrated)
            self.calibration_system.add_risk_adjustment_callback(self._on_risk_adjusted)
            self.calibration_system.add_emergency_callback(self._on_emergency_triggered)
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """신호 생성 콜백 추가"""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """거래 실행 콜백 추가"""
        self.trade_callbacks.append(callback)
    
    def add_risk_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """리스크 이벤트 콜백 추가"""
        self.risk_callbacks.append(callback)
    
    def start_system(self):
        """시스템 시작"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.running = True
        self.state.running = True
        self.state.start_time = datetime.now()
        
        self.logger.info("Starting Realtime Hybrid System")
        
        try:
            # 실시간 보정 시스템 시작
            if hasattr(self, 'calibration_system'):
                self.calibration_system.start_realtime_calibration()
            
            # 데이터 업데이트 스레드
            self.data_thread = threading.Thread(target=self._data_update_loop, daemon=True)
            self.data_thread.start()
            
            # 신호 생성 스레드
            self.signal_thread = threading.Thread(target=self._signal_generation_loop, daemon=True)
            self.signal_thread.start()
            
            # 모니터링 스레드
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Realtime Hybrid System started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """시스템 중지"""
        if not self.running:
            return
        
        self.running = False
        self.state.running = False
        
        self.logger.info("Stopping Realtime Hybrid System")
        
        try:
            # 실시간 보정 시스템 중지
            if hasattr(self, 'calibration_system'):
                self.calibration_system.stop_realtime_calibration()
            
            # 스레드 종료 대기
            for thread in [self.data_thread, self.signal_thread, self.monitoring_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=5)
            
            # 최종 상태 저장
            self._save_performance_data()
            
            self.logger.info("Realtime Hybrid System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def _data_update_loop(self):
        """데이터 업데이트 루프"""
        while self.running:
            try:
                # 실제 구현에서는 실시간 데이터 소스에서 데이터 수신
                current_time = datetime.now()
                
                # 포트폴리오 상태 업데이트
                self._update_portfolio_state()
                
                # 시장 데이터 업데이트 (모의)
                self._update_market_data()
                
                self.state.last_data_update = current_time
                
                time.sleep(self.config.data_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                time.sleep(5)
    
    def _signal_generation_loop(self):
        """신호 생성 루프"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # PPO 모델을 통한 신호 생성
                signals = self._generate_trading_signals()
                
                # 신호 처리
                for signal in signals:
                    self._process_trading_signal(signal)
                
                self.state.last_signal_time = current_time
                
                time.sleep(self.config.signal_generation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                time.sleep(30)
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 리스크 체크
                self._check_risk_limits()
                
                # 성과 추적
                self._track_performance()
                
                # 안전 장치 체크
                self._check_safety_limits()
                
                time.sleep(30)  # 30초마다 모니터링
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _update_portfolio_state(self):
        """포트폴리오 상태 업데이트"""
        try:
            self.state.current_equity = self.position_manager.get_equity()
            self.state.current_positions_count = len(self.position_manager.positions)
            self.state.total_trades = len(self.position_manager.trade_history)
            
            # 일일 손익 계산
            if self.state.start_time:
                start_equity = self.config.initial_capital
                self.state.daily_pnl = (self.state.current_equity - start_equity) / start_equity
            
            # 최대 낙폭 업데이트
            if hasattr(self, '_peak_equity'):
                if self.state.current_equity > self._peak_equity:
                    self._peak_equity = self.state.current_equity
                
                current_drawdown = (self._peak_equity - self.state.current_equity) / self._peak_equity
                self.state.max_drawdown = max(self.state.max_drawdown, current_drawdown)
            else:
                self._peak_equity = self.state.current_equity
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")
    
    def _update_market_data(self):
        """시장 데이터 업데이트 (모의)"""
        # 실제 구현에서는 실시간 시장 데이터 API 연동
        pass
    
    def _generate_trading_signals(self) -> List[TradingSignal]:
        """거래 신호 생성"""
        signals = []
        
        try:
            # 현재는 모의 신호 생성
            # 실제 구현에서는 PPO 모델과 전략 선택기 사용
            
            # 예시: 간단한 모의 신호
            if np.random.random() > 0.95:  # 5% 확률로 신호 생성
                signal = TradingSignal(
                    symbol="AAPL",
                    action=np.random.choice(["buy", "sell"]),
                    confidence=np.random.uniform(0.6, 0.9),
                    size_ratio=0.05,  # 포트폴리오의 5%
                    reason="PPO model signal",
                    strategy_name="PPO_Strategy_1"
                )
                signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
        
        return signals
    
    def _process_trading_signal(self, signal: TradingSignal):
        """거래 신호 처리"""
        try:
            # 신호 유효성 검증
            if not self._validate_signal(signal):
                self.logger.warning(f"Invalid signal rejected: {signal.symbol} {signal.action}")
                return
            
            # 리스크 기반 포지션 사이징
            recommended_sizing = self.risk_manager.get_position_sizing_recommendation(
                symbol=signal.symbol,
                current_price=100.0,  # 실제로는 현재 시장가격
                signal_confidence=signal.confidence
            )
            
            # 실시간 보정 적용
            if hasattr(self, 'calibration_system'):
                adjustments = self.calibration_system.get_current_adjustments()
                position_multiplier = adjustments['active_adjustments'].get('position_size_multiplier', 1.0)
                recommended_sizing['recommended_size'] *= position_multiplier
            
            # 거래 실행
            from ..position_management.position_models import OrderSignal, OrderType
            
            order_signal = OrderSignal(
                action=signal.action,
                symbol=signal.symbol,
                size=recommended_sizing['recommended_size'],
                order_type=OrderType.MARKET,
                confidence=signal.confidence,
                reason=signal.reason
            )
            
            trade = self.position_manager.execute_trade(
                order_signal, 
                100.0,  # 현재 가격
                signal.strategy_name
            )
            
            if trade:
                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} {trade.size}")
                
                # 신호 기록
                self.signal_history.append(signal)
                
                # 콜백 실행
                for callback in self.trade_callbacks:
                    try:
                        callback({
                            'signal': signal,
                            'trade': trade,
                            'sizing_info': recommended_sizing
                        })
                    except Exception as e:
                        self.logger.error(f"Trade callback failed: {e}")
            
            # 신호 콜백 실행
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"Signal callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """신호 유효성 검증"""
        # 기본 유효성 검사
        if not signal.symbol or signal.action not in ['buy', 'sell']:
            return False
        
        if signal.confidence < 0.5:  # 최소 신뢰도
            return False
        
        # 리스크 한도 체크
        current_metrics = self.risk_manager.calculate_risk_metrics()
        if current_metrics.overall_risk_score > 80:  # 높은 리스크시 신호 거부
            return False
        
        return True
    
    def _check_risk_limits(self):
        """리스크 한도 체크"""
        try:
            current_metrics = self.risk_manager.calculate_risk_metrics()
            alerts = self.risk_manager.check_risk_limits(current_metrics)
            
            if alerts:
                for alert in alerts:
                    if alert.risk_level == RiskLevel.CRITICAL:
                        self.state.risk_alerts.append(f"CRITICAL: {alert.title}")
                        self.logger.critical(f"Risk alert: {alert.title}")
                    else:
                        self.state.risk_alerts.append(f"{alert.risk_level.value}: {alert.title}")
                
                # 리스크 콜백 실행
                for callback in self.risk_callbacks:
                    try:
                        callback({
                            'metrics': current_metrics,
                            'alerts': alerts
                        })
                    except Exception as e:
                        self.logger.error(f"Risk callback failed: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def _track_performance(self):
        """성과 추적"""
        try:
            performance_data = {
                'timestamp': datetime.now(),
                'equity': self.state.current_equity,
                'daily_pnl': self.state.daily_pnl,
                'max_drawdown': self.state.max_drawdown,
                'positions_count': self.state.current_positions_count,
                'total_trades': self.state.total_trades
            }
            
            self.performance_history.append(performance_data)
            
            # 메모리 관리 (최근 1000개만 유지)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
    
    def _check_safety_limits(self):
        """안전 장치 체크"""
        try:
            # 일일 손실 한도 체크
            if abs(self.state.daily_pnl) > self.config.daily_loss_limit:
                self._trigger_emergency_stop(f"Daily loss limit exceeded: {self.state.daily_pnl:.2%}")
            
            # 최대 낙폭 한도 체크
            if self.state.max_drawdown > self.config.max_drawdown_limit:
                self._trigger_emergency_stop(f"Max drawdown limit exceeded: {self.state.max_drawdown:.2%}")
                
        except Exception as e:
            self.logger.error(f"Error checking safety limits: {e}")
    
    def _trigger_emergency_stop(self, reason: str):
        """긴급 정지 발동"""
        if not self.config.emergency_stop_enabled:
            return
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        try:
            # 모든 포지션 청산
            for symbol in list(self.position_manager.positions.keys()):
                position = self.position_manager.positions[symbol]
                if not position.is_flat:
                    self.position_manager.close_position(symbol, reason="Emergency stop")
            
            # 시스템 중지
            self.stop_system()
            
            self.state.system_alerts.append(f"Emergency stop: {reason}")
            
        except Exception as e:
            self.logger.error(f"Emergency stop execution failed: {e}")
    
    def _save_performance_data(self):
        """성과 데이터 저장"""
        try:
            if not self.performance_history:
                return
            
            # 성과 데이터 DataFrame 생성
            df = pd.DataFrame(self.performance_history)
            
            # 파일 저장
            log_dir = Path(self.config.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = log_dir / f"performance_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Performance data saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def _on_parameter_calibrated(self, parameters: Dict[str, Dict[str, Any]]):
        """파라미터 보정 콜백"""
        self.logger.info(f"Parameters calibrated for {len(parameters)} symbols")
    
    def _on_risk_adjusted(self, regime: str, adjustments: Dict[str, Any]):
        """리스크 조정 콜백"""
        self.logger.info(f"Risk adjustments applied for regime: {regime}")
        self.state.system_alerts.append(f"Risk adjusted for {regime}: {adjustments}")
    
    def _on_emergency_triggered(self, event_type: str, data: Dict[str, Any]):
        """긴급상황 콜백"""
        self.logger.critical(f"Emergency event: {event_type}")
        self.state.system_alerts.append(f"Emergency: {event_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        status = {
            'system_state': {
                'running': self.state.running,
                'start_time': self.state.start_time,
                'uptime_seconds': (datetime.now() - self.state.start_time).total_seconds() if self.state.start_time else 0
            },
            'portfolio_state': {
                'current_equity': self.state.current_equity,
                'daily_pnl': self.state.daily_pnl,
                'max_drawdown': self.state.max_drawdown,
                'positions_count': self.state.current_positions_count,
                'total_trades': self.state.total_trades
            },
            'recent_performance': self.performance_history[-10:] if self.performance_history else [],
            'recent_signals': [s.__dict__ for s in self.signal_history[-5:]] if self.signal_history else [],
            'risk_alerts': self.state.risk_alerts[-5:],
            'system_alerts': self.state.system_alerts[-5:],
            'threads_status': {
                'data_thread': self.data_thread.is_alive() if self.data_thread else False,
                'signal_thread': self.signal_thread.is_alive() if self.signal_thread else False,
                'monitoring_thread': self.monitoring_thread.is_alive() if self.monitoring_thread else False
            }
        }
        
        # 보정 시스템 상태 추가
        if hasattr(self, 'calibration_system'):
            status['calibration_system'] = self.calibration_system.get_system_status()
        
        return status
    
    def force_calibration(self):
        """강제 보정 실행"""
        if hasattr(self, 'calibration_system'):
            self.calibration_system.force_full_calibration()
    
    def execute_manual_trade(self, symbol: str, action: str, size: float, reason: str = "Manual trade"):
        """수동 거래 실행"""
        signal = TradingSignal(
            symbol=symbol,
            action=action,
            confidence=1.0,
            size_ratio=size,
            reason=reason,
            strategy_name="Manual"
        )
        
        self._process_trading_signal(signal)