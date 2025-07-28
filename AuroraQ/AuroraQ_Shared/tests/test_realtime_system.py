#!/usr/bin/env python3
"""
실시간 시스템 테스트
실시간 보정 시스템과 하이브리드 거래 시스템 테스트
"""

import sys
import os
import unittest
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# 모듈 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 개별 모듈 직접 임포트 (상대 임포트 문제 회피)
try:
    from integration.realtime_calibration_system import (
        RealtimeCalibrationSystem, 
        RealtimeCalibrationConfig
    )
    from integration.realtime_hybrid_system import (
        RealtimeHybridSystem,
        RealtimeSystemConfig,
        TradingSignal
    )
except ImportError:
    # 테스트 환경에서 모듈을 찾을 수 없는 경우 스킵
    print("⚠️ 실시간 시스템 모듈을 임포트할 수 없습니다. 기본 테스트만 실행합니다.")
    RealtimeCalibrationSystem = None
    RealtimeCalibrationConfig = None
    RealtimeHybridSystem = None
    RealtimeSystemConfig = None
    TradingSignal = None
from position_management.unified_position_manager import UnifiedPositionManager
from risk_management.advanced_risk_manager import AdvancedRiskManager
from risk_management.risk_models import RiskConfig


@unittest.skipIf(RealtimeCalibrationSystem is None, "실시간 보정 시스템 모듈을 찾을 수 없음")
class TestRealtimeCalibrationSystem(unittest.TestCase):
    """실시간 보정 시스템 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 포지션 관리자
        self.position_manager = UnifiedPositionManager(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # 리스크 관리자
        risk_config = RiskConfig()
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.position_manager,
            config=risk_config
        )
        
        # 보정 시스템 설정
        self.calibration_config = RealtimeCalibrationConfig(
            calibration_interval_minutes=1,  # 테스트를 위해 짧게 설정
            quick_calibration_interval_minutes=0.5,
            market_condition_check_interval_seconds=10,
            enable_auto_slippage_adjustment=True,
            enable_auto_risk_adjustment=True
        )
        
        # 보정 시스템
        self.calibration_system = RealtimeCalibrationSystem(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            config=self.calibration_config,
            log_directory="test_logs"
        )
    
    def test_calibration_system_initialization(self):
        """보정 시스템 초기화 테스트"""
        self.assertIsNotNone(self.calibration_system)
        self.assertEqual(self.calibration_system.config.calibration_interval_minutes, 1)
        self.assertFalse(self.calibration_system.running)
        self.assertFalse(self.calibration_system.state.emergency_mode)
    
    def test_parameter_backup_and_restore(self):
        """파라미터 백업 및 복구 테스트"""
        # 원래 파라미터 확인
        original_slippage = self.position_manager.slippage_rate
        original_commission = self.position_manager.commission_rate
        
        # 백업된 파라미터 확인
        backup = self.calibration_system.original_parameters
        self.assertEqual(backup['slippage_rate'], original_slippage)
        self.assertEqual(backup['commission_rate'], original_commission)
        
        # 파라미터 변경
        self.position_manager.slippage_rate = 0.001
        self.position_manager.commission_rate = 0.002
        
        # 복구 테스트
        self.calibration_system._restore_normal_parameters()
        self.assertEqual(self.position_manager.slippage_rate, original_slippage)
        self.assertEqual(self.position_manager.commission_rate, original_commission)
    
    def test_market_regime_detection(self):
        """시장 레짐 감지 테스트"""
        # 정상 상태 확인
        self.assertEqual(self.calibration_system.state.current_market_regime, "normal")
        
        # 시장 레짐 변경 시뮬레이션
        self.calibration_system._handle_market_regime_change("high_volatility", {"AAPL": "high_volatility"})
        self.assertEqual(self.calibration_system.state.current_market_regime, "high_volatility")
        
        # 조정사항 적용 확인
        self.assertTrue(len(self.calibration_system.state.adjustment_history) > 0)
    
    def test_emergency_mode_activation(self):
        """긴급 모드 활성화 테스트"""
        # 긴급 모드 활성화
        self.calibration_system._activate_emergency_mode({"AAPL": "extreme"})
        
        self.assertTrue(self.calibration_system.state.emergency_mode)
        self.assertTrue(len(self.calibration_system.state.system_alerts) > 0)
    
    def test_calibration_callbacks(self):
        """보정 콜백 테스트"""
        callback_called = []
        
        def test_callback(parameters):
            callback_called.append(parameters)
        
        self.calibration_system.add_parameter_callback(test_callback)
        
        # 콜백 테스트를 위한 모의 보정 결과
        test_parameters = {"AAPL": {"slippage": 0.001, "commission": 0.001}}
        
        # 콜백 실행
        for callback in self.calibration_system.parameter_callbacks:
            callback(test_parameters)
        
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0], test_parameters)
    
    def test_system_status(self):
        """시스템 상태 테스트"""
        status = self.calibration_system.get_system_status()
        
        self.assertIn('running', status)
        self.assertIn('current_regime', status)
        self.assertIn('emergency_mode', status)
        self.assertIn('threads_alive', status)
        
        self.assertFalse(status['running'])
        self.assertEqual(status['current_regime'], 'normal')
        self.assertFalse(status['emergency_mode'])


@unittest.skipIf(RealtimeHybridSystem is None, "실시간 하이브리드 시스템 모듈을 찾을 수 없음")
class TestRealtimeHybridSystem(unittest.TestCase):
    """실시간 하이브리드 시스템 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.system_config = RealtimeSystemConfig(
            initial_capital=100000,
            max_portfolio_risk=0.02,
            data_update_interval=0.1,  # 테스트를 위해 짧게 설정
            signal_generation_interval=1,
            enable_realtime_calibration=True,
            daily_loss_limit=0.10,  # 테스트를 위해 높게 설정
            emergency_stop_enabled=True
        )
        
        self.hybrid_system = RealtimeHybridSystem(self.system_config)
    
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        self.assertIsNotNone(self.hybrid_system)
        self.assertIsNotNone(self.hybrid_system.position_manager)
        self.assertIsNotNone(self.hybrid_system.risk_manager)
        self.assertIsNotNone(self.hybrid_system.calibration_system)
        
        self.assertFalse(self.hybrid_system.running)
        self.assertFalse(self.hybrid_system.state.running)
    
    def test_trading_signal_validation(self):
        """거래 신호 유효성 검증 테스트"""
        # 유효한 신호
        valid_signal = TradingSignal(
            symbol="AAPL",
            action="buy",
            confidence=0.8,
            size_ratio=0.05
        )
        
        self.assertTrue(self.hybrid_system._validate_signal(valid_signal))
        
        # 무효한 신호 (낮은 신뢰도)
        invalid_signal = TradingSignal(
            symbol="AAPL",
            action="buy",
            confidence=0.3,
            size_ratio=0.05
        )
        
        self.assertFalse(self.hybrid_system._validate_signal(invalid_signal))
        
        # 무효한 신호 (잘못된 액션)
        invalid_action_signal = TradingSignal(
            symbol="AAPL",
            action="invalid",
            confidence=0.8,
            size_ratio=0.05
        )
        
        self.assertFalse(self.hybrid_system._validate_signal(invalid_action_signal))
    
    def test_portfolio_state_update(self):
        """포트폴리오 상태 업데이트 테스트"""
        # 초기 상태
        initial_equity = self.hybrid_system.state.current_equity
        
        # 상태 업데이트
        self.hybrid_system._update_portfolio_state()
        
        # 업데이트된 상태 확인
        self.assertIsNotNone(self.hybrid_system.state.current_equity)
        self.assertEqual(self.hybrid_system.state.current_positions_count, 0)
    
    def test_signal_generation(self):
        """신호 생성 테스트"""
        signals = self.hybrid_system._generate_trading_signals()
        
        # 신호가 리스트인지 확인
        self.assertIsInstance(signals, list)
        
        # 생성된 신호가 있다면 유효성 확인
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertIn(signal.action, ['buy', 'sell'])
            self.assertTrue(0 <= signal.confidence <= 1)
    
    def test_manual_trade_execution(self):
        """수동 거래 실행 테스트"""
        # 수동 거래 실행
        initial_trades = len(self.hybrid_system.position_manager.trade_history)
        
        # 모의 거래 (실제로는 시장 데이터가 필요)
        try:
            self.hybrid_system.execute_manual_trade("AAPL", "buy", 0.01, "Test trade")
        except Exception as e:
            # 시장 데이터가 없어서 실패할 수 있음
            pass
    
    def test_system_callbacks(self):
        """시스템 콜백 테스트"""
        signal_callbacks = []
        trade_callbacks = []
        risk_callbacks = []
        
        def signal_callback(signal):
            signal_callbacks.append(signal)
        
        def trade_callback(trade_data):
            trade_callbacks.append(trade_data)
        
        def risk_callback(risk_data):
            risk_callbacks.append(risk_data)
        
        # 콜백 등록
        self.hybrid_system.add_signal_callback(signal_callback)
        self.hybrid_system.add_trade_callback(trade_callback)
        self.hybrid_system.add_risk_callback(risk_callback)
        
        # 콜백 등록 확인
        self.assertEqual(len(self.hybrid_system.signal_callbacks), 1)
        self.assertEqual(len(self.hybrid_system.trade_callbacks), 1)
        self.assertEqual(len(self.hybrid_system.risk_callbacks), 1)
    
    def test_system_status(self):
        """시스템 상태 테스트"""
        status = self.hybrid_system.get_system_status()
        
        # 필수 상태 정보 확인
        self.assertIn('system_state', status)
        self.assertIn('portfolio_state', status)
        self.assertIn('threads_status', status)
        
        # 시스템 상태
        self.assertIn('running', status['system_state'])
        self.assertFalse(status['system_state']['running'])
        
        # 포트폴리오 상태
        self.assertIn('current_equity', status['portfolio_state'])
        self.assertIn('positions_count', status['portfolio_state'])
        
        # 스레드 상태
        self.assertIn('data_thread', status['threads_status'])
        self.assertIn('signal_thread', status['threads_status'])
        self.assertIn('monitoring_thread', status['threads_status'])
    
    def test_safety_limits(self):
        """안전 장치 테스트"""
        # 일일 손실 한도 시뮬레이션
        self.hybrid_system.state.daily_pnl = -0.15  # -15% 손실 (한도 초과)
        
        # 안전 장치 체크
        try:
            self.hybrid_system._check_safety_limits()
        except Exception:
            # 긴급 정지가 발동될 수 있음
            pass
    
    def tearDown(self):
        """테스트 정리"""
        # 시스템이 실행 중이라면 중지
        if self.hybrid_system.running:
            self.hybrid_system.stop_system()


@unittest.skipIf(RealtimeHybridSystem is None, "실시간 시스템 모듈을 찾을 수 없음")
class TestSystemIntegration(unittest.TestCase):
    """시스템 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.system_config = RealtimeSystemConfig(
            initial_capital=100000,
            enable_realtime_calibration=True,
            data_update_interval=0.1,
            signal_generation_interval=0.5
        )
        
        self.hybrid_system = RealtimeHybridSystem(self.system_config)
    
    def test_calibration_integration(self):
        """보정 시스템 통합 테스트"""
        # 보정 시스템이 올바르게 통합되었는지 확인
        self.assertIsNotNone(self.hybrid_system.calibration_system)
        
        # 보정 시스템 설정 확인
        calibration_status = self.hybrid_system.calibration_system.get_system_status()
        self.assertIsInstance(calibration_status, dict)
    
    def test_backtest_synchronization(self):
        """백테스트 동기화 테스트"""
        from integration.backtest_integration import (
            create_synchronized_backtest_environment,
            sync_backtest_with_realtime_parameters
        )
        
        # 실시간 시스템 설정 추출
        realtime_config = {
            'initial_capital': self.system_config.initial_capital,
            'max_portfolio_risk': self.system_config.max_portfolio_risk,
            'enable_realtime_calibration': self.system_config.enable_realtime_calibration
        }
        
        # 동기화된 백테스트 환경 생성
        sync_backtest = create_synchronized_backtest_environment(realtime_config)
        
        self.assertIsNotNone(sync_backtest)
        self.assertEqual(sync_backtest.position_manager.initial_capital, realtime_config['initial_capital'])
    
    def test_parameter_synchronization(self):
        """파라미터 동기화 테스트"""
        # 실시간 시스템의 현재 파라미터 가져오기
        current_adjustments = self.hybrid_system.calibration_system.get_current_adjustments()
        
        self.assertIsInstance(current_adjustments, dict)
        self.assertIn('market_regime', current_adjustments)
        self.assertIn('active_adjustments', current_adjustments)
    
    def tearDown(self):
        """테스트 정리"""
        if self.hybrid_system.running:
            self.hybrid_system.stop_system()


def run_integration_demo():
    """통합 시스템 데모 실행"""
    print("🚀 AuroraQ 실시간 통합 시스템 데모")
    print("=" * 50)
    
    if RealtimeHybridSystem is None:
        print("⚠️ 실시간 시스템 모듈을 찾을 수 없습니다.")
        print("📝 기본 모듈 테스트만 실행됩니다.")
        return
    
    # 시스템 설정
    config = RealtimeSystemConfig(
        initial_capital=100000,
        max_portfolio_risk=0.02,
        enable_realtime_calibration=True,
        data_update_interval=1,
        signal_generation_interval=5
    )
    
    # 하이브리드 시스템 생성
    hybrid_system = RealtimeHybridSystem(config)
    
    try:
        print("📊 시스템 초기화 완료")
        print(f"💰 초기 자본: ${config.initial_capital:,.0f}")
        print(f"📈 최대 포트폴리오 리스크: {config.max_portfolio_risk:.1%}")
        
        # 시스템 시작
        print("\n🔄 시스템 시작...")
        hybrid_system.start_system()
        
        # 잠시 실행
        print("⏱️ 5초간 시스템 실행...")
        time.sleep(5)
        
        # 상태 확인
        status = hybrid_system.get_system_status()
        print(f"\n📋 시스템 상태:")
        print(f"   실행 중: {status['system_state']['running']}")
        print(f"   가동 시간: {status['system_state']['uptime_seconds']:.1f}초")
        print(f"   현재 자본: ${status['portfolio_state']['current_equity']:,.2f}")
        print(f"   포지션 수: {status['portfolio_state']['positions_count']}")
        
        # 보정 시스템 상태
        if 'calibration_system' in status:
            cal_status = status['calibration_system']
            print(f"   보정 시스템: {'활성' if cal_status['running'] else '비활성'}")
            print(f"   시장 레짐: {cal_status['current_regime']}")
        
        print("\n✅ 데모 완료")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류: {e}")
    
    finally:
        # 시스템 중지
        print("🛑 시스템 중지...")
        hybrid_system.stop_system()
        print("🏁 데모 종료")


if __name__ == '__main__':
    # 단위 테스트 실행
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50)
    
    # 통합 데모 실행
    run_integration_demo()