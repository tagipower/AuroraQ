#!/usr/bin/env python3
"""
실시간 보정 시스템
실거래 루프와 CalibrationManager를 완전 통합하여 실시간 파라미터 보정 구현
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
from pathlib import Path

from ..calibration.calibration_manager import CalibrationManager, CalibrationConfig
from ..calibration.execution_analyzer import ExecutionAnalyzer
from ..calibration.market_condition_detector import MarketConditionDetector
from ..position_management.unified_position_manager import UnifiedPositionManager
from ..risk_management.advanced_risk_manager import AdvancedRiskManager
from ..risk_management.risk_models import RiskConfig, RiskLevel


@dataclass
class RealtimeCalibrationConfig:
    """실시간 보정 설정"""
    # 보정 주기 설정
    calibration_interval_minutes: int = 30  # 30분마다 보정
    quick_calibration_interval_minutes: int = 5  # 5분마다 빠른 체크
    market_condition_check_interval_seconds: int = 60  # 1분마다 시장 상황 체크
    
    # 파라미터 조정 임계값
    slippage_adjustment_threshold: float = 0.1  # 10% 이상 차이시 조정
    commission_adjustment_threshold: float = 0.05  # 5% 이상 차이시 조정
    fill_rate_adjustment_threshold: float = 0.05  # 5% 이상 차이시 조정
    
    # 시장 레짐별 리스크 조정
    high_volatility_var_multiplier: float = 0.7  # 고변동성시 VaR 한도 30% 축소
    low_liquidity_position_multiplier: float = 0.8  # 저유동성시 포지션 20% 축소
    extreme_condition_emergency_reduction: float = 0.5  # 극한 상황시 50% 축소
    
    # 자동 조정 활성화
    enable_auto_slippage_adjustment: bool = True
    enable_auto_commission_adjustment: bool = True
    enable_auto_risk_adjustment: bool = True
    enable_emergency_protection: bool = True
    
    # 로깅 및 알림
    enable_detailed_logging: bool = True
    enable_real_time_alerts: bool = True


@dataclass
class RealtimeCalibrationState:
    """실시간 보정 상태"""
    last_calibration: Optional[datetime] = None
    last_quick_check: Optional[datetime] = None
    last_market_condition_check: Optional[datetime] = None
    current_market_regime: str = "normal"
    active_adjustments: Dict[str, float] = field(default_factory=dict)
    adjustment_history: List[Dict[str, Any]] = field(default_factory=list)
    emergency_mode: bool = False
    system_alerts: List[str] = field(default_factory=list)


class RealtimeCalibrationSystem:
    """실시간 보정 시스템"""
    
    def __init__(self,
                 position_manager: UnifiedPositionManager,
                 risk_manager: AdvancedRiskManager,
                 config: Optional[RealtimeCalibrationConfig] = None,
                 log_directory: str = "logs"):
        
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.config = config or RealtimeCalibrationConfig()
        self.state = RealtimeCalibrationState()
        
        # 보정 관련 컴포넌트
        calibration_config = CalibrationConfig()
        calibration_config.log_directory = log_directory
        self.calibration_manager = CalibrationManager(calibration_config)
        self.execution_analyzer = ExecutionAnalyzer(log_directory)
        self.market_condition_detector = MarketConditionDetector()
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # 실시간 스레드 관리
        self.running = False
        self.calibration_thread = None
        self.market_monitor_thread = None
        
        # 콜백 시스템
        self.parameter_callbacks: List[Callable] = []
        self.risk_adjustment_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # 초기 파라미터 백업
        self.original_parameters = self._backup_current_parameters()
    
    def add_parameter_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """파라미터 변경 콜백 추가"""
        self.parameter_callbacks.append(callback)
    
    def add_risk_adjustment_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """리스크 조정 콜백 추가"""
        self.risk_adjustment_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """긴급상황 콜백 추가"""
        self.emergency_callbacks.append(callback)
    
    def start_realtime_calibration(self):
        """실시간 보정 시작"""
        if self.running:
            self.logger.warning("Realtime calibration is already running")
            return
        
        self.running = True
        self.logger.info("Starting realtime calibration system")
        
        # 보정 스레드 시작
        self.calibration_thread = threading.Thread(target=self._calibration_loop, daemon=True)
        self.calibration_thread.start()
        
        # 시장 모니터링 스레드 시작
        self.market_monitor_thread = threading.Thread(target=self._market_monitoring_loop, daemon=True)
        self.market_monitor_thread.start()
        
        self.logger.info("Realtime calibration system started successfully")
    
    def stop_realtime_calibration(self):
        """실시간 보정 중지"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping realtime calibration system")
        
        # 스레드 종료 대기
        if self.calibration_thread:
            self.calibration_thread.join(timeout=5)
        if self.market_monitor_thread:
            self.market_monitor_thread.join(timeout=5)
        
        self.logger.info("Realtime calibration system stopped")
    
    def _calibration_loop(self):
        """보정 루프"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # 전체 보정 실행 체크
                if self._should_run_full_calibration(current_time):
                    self._run_full_calibration()
                    self.state.last_calibration = current_time
                
                # 빠른 보정 체크
                elif self._should_run_quick_calibration(current_time):
                    self._run_quick_calibration()
                    self.state.last_quick_check = current_time
                
                # 대기
                time.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                self.logger.error(f"Error in calibration loop: {e}")
                time.sleep(300)  # 에러시 5분 대기
    
    def _market_monitoring_loop(self):
        """시장 모니터링 루프"""
        while self.running:
            try:
                current_time = datetime.now()
                
                if self._should_check_market_condition(current_time):
                    self._check_and_adjust_market_conditions()
                    self.state.last_market_condition_check = current_time
                
                # 대기
                time.sleep(self.config.market_condition_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in market monitoring loop: {e}")
                time.sleep(120)  # 에러시 2분 대기
    
    def _should_run_full_calibration(self, current_time: datetime) -> bool:
        """전체 보정 실행 여부 판단"""
        if self.state.last_calibration is None:
            return True
        
        time_diff = current_time - self.state.last_calibration
        return time_diff.total_seconds() >= self.config.calibration_interval_minutes * 60
    
    def _should_run_quick_calibration(self, current_time: datetime) -> bool:
        """빠른 보정 실행 여부 판단"""
        if self.state.last_quick_check is None:
            return True
        
        time_diff = current_time - self.state.last_quick_check
        return time_diff.total_seconds() >= self.config.quick_calibration_interval_minutes * 60
    
    def _should_check_market_condition(self, current_time: datetime) -> bool:
        """시장 상황 체크 여부 판단"""
        if self.state.last_market_condition_check is None:
            return True
        
        time_diff = current_time - self.state.last_market_condition_check
        return time_diff.total_seconds() >= self.config.market_condition_check_interval_seconds
    
    def _run_full_calibration(self):
        """전체 보정 실행"""
        self.logger.info("Running full calibration")
        
        try:
            # 모든 심볼에 대해 보정 실행
            calibration_result = self.calibration_manager.calibrate_parameters()
            
            if calibration_result.success:
                # 파라미터 적용
                self._apply_calibration_results(calibration_result)
                
                # 결과 로깅
                self.logger.info(f"Full calibration completed: {len(calibration_result.parameters)} symbols updated")
                
                # 콜백 실행
                for callback in self.parameter_callbacks:
                    try:
                        callback(calibration_result.parameters)
                    except Exception as e:
                        self.logger.error(f"Parameter callback failed: {e}")
            else:
                self.logger.warning(f"Full calibration failed: {calibration_result.message}")
                
        except Exception as e:
            self.logger.error(f"Full calibration error: {e}")
    
    def _run_quick_calibration(self):
        """빠른 보정 실행 (주요 심볼만)"""
        self.logger.debug("Running quick calibration")
        
        try:
            # 주요 심볼들에 대해서만 빠른 보정
            active_symbols = list(self.position_manager.positions.keys())
            if not active_symbols:
                return
            
            for symbol in active_symbols[:5]:  # 최대 5개 심볼만
                try:
                    result = self.calibration_manager.calibrate_parameters(symbol, force_calibration=True)
                    if result.success and symbol in result.parameters:
                        self._apply_symbol_calibration(symbol, result.parameters[symbol])
                except Exception as e:
                    self.logger.error(f"Quick calibration failed for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Quick calibration error: {e}")
    
    def _check_and_adjust_market_conditions(self):
        """시장 상황 체크 및 조정"""
        try:
            # 활성 심볼들의 시장 상황 체크
            active_symbols = list(self.position_manager.positions.keys())
            if not active_symbols:
                return
            
            market_conditions = {}
            overall_condition = "normal"
            
            for symbol in active_symbols:
                try:
                    condition = self.market_condition_detector.detect_current_condition(symbol)
                    market_conditions[symbol] = condition
                    
                    # 전체 시장 상황 결정 (가장 위험한 상황으로)
                    if condition in ["high_volatility", "low_liquidity"] and overall_condition == "normal":
                        overall_condition = condition
                    elif condition == "extreme" and overall_condition != "extreme":
                        overall_condition = "extreme"
                        
                except Exception as e:
                    self.logger.error(f"Market condition check failed for {symbol}: {e}")
            
            # 시장 레짐 변경 감지
            if overall_condition != self.state.current_market_regime:
                self.logger.info(f"Market regime changed: {self.state.current_market_regime} -> {overall_condition}")
                self._handle_market_regime_change(overall_condition, market_conditions)
                self.state.current_market_regime = overall_condition
            
        except Exception as e:
            self.logger.error(f"Market condition monitoring error: {e}")
    
    def _handle_market_regime_change(self, new_regime: str, market_conditions: Dict[str, str]):
        """시장 레짐 변경 처리"""
        adjustments = {}
        
        if new_regime == "high_volatility":
            # 고변동성시 리스크 한도 축소
            if self.config.enable_auto_risk_adjustment:
                adjustments['var_limit_multiplier'] = self.config.high_volatility_var_multiplier
                adjustments['position_size_multiplier'] = 0.9
                self._apply_risk_adjustments(adjustments)
                
        elif new_regime == "low_liquidity":
            # 저유동성시 포지션 크기 축소
            if self.config.enable_auto_risk_adjustment:
                adjustments['position_size_multiplier'] = self.config.low_liquidity_position_multiplier
                adjustments['slippage_multiplier'] = 1.2  # 슬리피지 20% 증가
                self._apply_risk_adjustments(adjustments)
                
        elif new_regime == "extreme":
            # 극한 상황시 긴급 대응
            if self.config.enable_emergency_protection:
                self._activate_emergency_mode(market_conditions)
                
        elif new_regime == "normal":
            # 정상 상황시 원래 파라미터 복구
            self._restore_normal_parameters()
        
        # 상태 기록
        adjustment_record = {
            'timestamp': datetime.now(),
            'regime_change': f"{self.state.current_market_regime} -> {new_regime}",
            'adjustments': adjustments,
            'market_conditions': market_conditions
        }
        self.state.adjustment_history.append(adjustment_record)
        
        # 콜백 실행
        for callback in self.risk_adjustment_callbacks:
            try:
                callback(new_regime, adjustments)
            except Exception as e:
                self.logger.error(f"Risk adjustment callback failed: {e}")
    
    def _apply_calibration_results(self, calibration_result):
        """보정 결과 적용"""
        for symbol, params in calibration_result.parameters.items():
            self._apply_symbol_calibration(symbol, params)
    
    def _apply_symbol_calibration(self, symbol: str, params: Dict[str, Any]):
        """심볼별 보정 적용"""
        try:
            # 슬리피지 조정
            if self.config.enable_auto_slippage_adjustment and 'slippage' in params:
                new_slippage = params['slippage']
                current_slippage = self.position_manager.slippage_rate
                
                if abs(new_slippage - current_slippage) / current_slippage > self.config.slippage_adjustment_threshold:
                    self.position_manager.slippage_rate = new_slippage
                    self.logger.info(f"Slippage adjusted for {symbol}: {current_slippage:.4f} -> {new_slippage:.4f}")
            
            # 수수료 조정
            if self.config.enable_auto_commission_adjustment and 'commission' in params:
                new_commission = params['commission']
                current_commission = self.position_manager.commission_rate
                
                if abs(new_commission - current_commission) / current_commission > self.config.commission_adjustment_threshold:
                    self.position_manager.commission_rate = new_commission
                    self.logger.info(f"Commission adjusted for {symbol}: {current_commission:.4f} -> {new_commission:.4f}")
            
            # 체결률 정보 저장 (향후 포지션 사이징에 활용)
            if 'fill_rate' in params:
                self.state.active_adjustments[f"{symbol}_fill_rate"] = params['fill_rate']
                
        except Exception as e:
            self.logger.error(f"Failed to apply calibration for {symbol}: {e}")
    
    def _apply_risk_adjustments(self, adjustments: Dict[str, float]):
        """리스크 조정 적용"""
        try:
            # VaR 한도 조정
            if 'var_limit_multiplier' in adjustments:
                multiplier = adjustments['var_limit_multiplier']
                original_var_limit = self.risk_manager.config.var_limit_pct
                new_var_limit = original_var_limit * multiplier
                self.risk_manager.config.var_limit_pct = new_var_limit
                self.logger.info(f"VaR limit adjusted: {original_var_limit:.4f} -> {new_var_limit:.4f}")
            
            # 포지션 크기 조정 (미래 거래에 적용)
            if 'position_size_multiplier' in adjustments:
                self.state.active_adjustments['position_size_multiplier'] = adjustments['position_size_multiplier']
                
            # 슬리피지 배수 조정
            if 'slippage_multiplier' in adjustments:
                multiplier = adjustments['slippage_multiplier']
                original_slippage = self.original_parameters.get('slippage_rate', self.position_manager.slippage_rate)
                new_slippage = original_slippage * multiplier
                self.position_manager.slippage_rate = new_slippage
                self.logger.info(f"Slippage adjusted: {original_slippage:.4f} -> {new_slippage:.4f}")
                
        except Exception as e:
            self.logger.error(f"Failed to apply risk adjustments: {e}")
    
    def _activate_emergency_mode(self, market_conditions: Dict[str, str]):
        """긴급 모드 활성화"""
        if self.state.emergency_mode:
            return  # 이미 긴급 모드
        
        self.state.emergency_mode = True
        self.logger.critical("EMERGENCY MODE ACTIVATED")
        
        try:
            # 긴급 포지션 축소
            reduction_ratio = self.config.extreme_condition_emergency_reduction
            
            for symbol in list(self.position_manager.positions.keys()):
                try:
                    position = self.position_manager.positions[symbol]
                    if not position.is_flat:
                        reduce_size = abs(position.size) * reduction_ratio
                        self.position_manager.close_position(
                            symbol, 
                            size=reduce_size,
                            reason="Emergency market condition reduction"
                        )
                        self.logger.critical(f"Emergency reduction: {symbol} by {reduce_size}")
                except Exception as e:
                    self.logger.error(f"Emergency reduction failed for {symbol}: {e}")
            
            # VaR 한도 대폭 축소
            emergency_var_multiplier = 0.3  # 70% 축소
            original_var_limit = self.original_parameters.get('var_limit_pct', self.risk_manager.config.var_limit_pct)
            self.risk_manager.config.var_limit_pct = original_var_limit * emergency_var_multiplier
            
            # 긴급 알림
            alert_message = f"Emergency mode activated due to extreme market conditions: {market_conditions}"
            self.state.system_alerts.append(alert_message)
            
            # 긴급 콜백 실행
            for callback in self.emergency_callbacks:
                try:
                    callback("emergency_mode_activated", {'market_conditions': market_conditions})
                except Exception as e:
                    self.logger.error(f"Emergency callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Emergency mode activation failed: {e}")
    
    def _restore_normal_parameters(self):
        """정상 파라미터 복구"""
        try:
            # 긴급 모드 해제
            if self.state.emergency_mode:
                self.state.emergency_mode = False
                self.logger.info("Emergency mode deactivated")
            
            # 원래 파라미터 복구
            if 'slippage_rate' in self.original_parameters:
                self.position_manager.slippage_rate = self.original_parameters['slippage_rate']
            
            if 'commission_rate' in self.original_parameters:
                self.position_manager.commission_rate = self.original_parameters['commission_rate']
            
            if 'var_limit_pct' in self.original_parameters:
                self.risk_manager.config.var_limit_pct = self.original_parameters['var_limit_pct']
            
            # 활성 조정 초기화
            self.state.active_adjustments.clear()
            
            self.logger.info("Normal parameters restored")
            
        except Exception as e:
            self.logger.error(f"Failed to restore normal parameters: {e}")
    
    def _backup_current_parameters(self) -> Dict[str, Any]:
        """현재 파라미터 백업"""
        return {
            'slippage_rate': self.position_manager.slippage_rate,
            'commission_rate': self.position_manager.commission_rate,
            'var_limit_pct': self.risk_manager.config.var_limit_pct
        }
    
    def get_current_adjustments(self) -> Dict[str, Any]:
        """현재 적용중인 조정사항 반환"""
        return {
            'market_regime': self.state.current_market_regime,
            'emergency_mode': self.state.emergency_mode,
            'active_adjustments': self.state.active_adjustments.copy(),
            'last_calibration': self.state.last_calibration,
            'adjustment_history': self.state.adjustment_history[-10:],  # 최근 10개
            'system_alerts': self.state.system_alerts[-5:]  # 최근 5개
        }
    
    def force_full_calibration(self):
        """강제 전체 보정 실행"""
        self.logger.info("Force full calibration requested")
        self._run_full_calibration()
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            'running': self.running,
            'current_regime': self.state.current_market_regime,
            'emergency_mode': self.state.emergency_mode,
            'last_calibration': self.state.last_calibration,
            'last_market_check': self.state.last_market_condition_check,
            'active_adjustments_count': len(self.state.active_adjustments),
            'alerts_count': len(self.state.system_alerts),
            'threads_alive': {
                'calibration': self.calibration_thread.is_alive() if self.calibration_thread else False,
                'market_monitor': self.market_monitor_thread.is_alive() if self.market_monitor_thread else False
            }
        }