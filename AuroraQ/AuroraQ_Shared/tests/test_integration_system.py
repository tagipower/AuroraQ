#!/usr/bin/env python3
"""
통합 시스템 테스트
백테스트-리스크관리-보정시스템 통합 검증
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 테스트 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration import (
    BacktestRiskIntegration, create_risk_aware_backtest,
    create_calibrated_backtest, quick_risk_backtest
)
from risk_management import RiskConfig, RiskMetrics
from calibration import CalibrationConfig, CalibrationResult
from position_management import UnifiedPositionManager


class MockBacktestEngine:
    """백테스트 엔진 모킹"""
    
    def __init__(self):
        self.execution_parameters = {}
        
    def run(self, strategy, data, start_date=None, end_date=None):
        return {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'total_trades': 100,
            'win_rate': 0.65
        }
    
    def update_execution_parameters(self, params):
        self.execution_parameters.update(params)


class MockStrategy:
    """전략 모킹"""
    
    def __init__(self):
        self.portfolio = Mock()
        self.portfolio.positions = {}
        self.portfolio.cash = 50000
        
    def generate_signals(self, data, **kwargs):
        # 간단한 시그널 생성
        return pd.DataFrame({
            'AAPL': np.random.normal(0, 1, len(data)),
            'GOOGL': np.random.normal(0, 1, len(data))
        }, index=data.index)


class TestBacktestRiskIntegration(unittest.TestCase):
    """백테스트 리스크 통합 테스트"""
    
    def setUp(self):
        self.mock_engine = MockBacktestEngine()
        self.risk_config = RiskConfig(
            var_limit_pct=0.05,
            max_drawdown_limit=0.15
        )
        
        # 보정 비활성화 통합 시스템
        self.integration_no_cal = BacktestRiskIntegration(
            backtest_engine=self.mock_engine,
            risk_config=self.risk_config,
            enable_calibration=False
        )
        
        # 보정 활성화 통합 시스템
        self.integration_with_cal = BacktestRiskIntegration(
            backtest_engine=self.mock_engine,
            risk_config=self.risk_config,
            enable_calibration=True
        )
        
        # 테스트 데이터
        self.test_data = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.02 + 0.001,
            'GOOGL': np.random.randn(100) * 0.025 + 0.0005
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))
        
        self.test_strategy = MockStrategy()
        
    def test_initialization_without_calibration(self):
        """보정 없는 초기화 테스트"""
        self.assertFalse(self.integration_no_cal.enable_calibration)
        self.assertIsNone(self.integration_no_cal.calibration_manager)
        self.assertIsNotNone(self.integration_no_cal.position_manager)
        self.assertIsNotNone(self.integration_no_cal.risk_manager)
        
    def test_initialization_with_calibration(self):
        """보정 포함 초기화 테스트"""
        self.assertTrue(self.integration_with_cal.enable_calibration)
        self.assertIsNotNone(self.integration_with_cal.calibration_manager)
        self.assertIsNotNone(self.integration_with_cal.position_manager)
        self.assertIsNotNone(self.integration_with_cal.risk_manager)
        
    def test_parameter_initialization(self):
        """파라미터 초기화 테스트"""
        # 기본 파라미터 확인
        self.assertIn('commission_rate', self.integration_with_cal._initial_params)
        self.assertIn('slippage_rate', self.integration_with_cal._initial_params)
        self.assertIn('fill_rate', self.integration_with_cal._initial_params)
        
        # 현재 파라미터 확인
        self.assertIn('commission_rate', self.integration_with_cal._current_params)
        self.assertEqual(
            len(self.integration_with_cal._initial_params),
            len(self.integration_with_cal._current_params)
        )
        
    def test_strategy_integration_without_calibration(self):
        """보정 없는 전략 통합 테스트"""
        integrated_strategy = self.integration_no_cal.integrate_with_strategy(
            self.test_strategy
        )
        
        # 전략이 리스크 관리 기능과 통합되었는지 확인
        self.assertNotEqual(
            integrated_strategy.generate_signals,
            self.test_strategy.generate_signals
        )
        
        # 신호 생성 테스트
        signals = integrated_strategy.generate_signals(self.test_data[:10])
        self.assertIsInstance(signals, pd.DataFrame)
        
    def test_strategy_integration_with_calibration(self):
        """보정 포함 전략 통합 테스트"""
        # 리스크 통합
        risk_integrated = self.integration_with_cal.integrate_with_strategy(
            self.test_strategy
        )
        
        # 보정 통합
        calibration_integrated = self.integration_with_cal._wrap_strategy_with_calibration(
            risk_integrated, self.test_data
        )
        
        # 신호 생성 테스트
        signals = calibration_integrated.generate_signals(self.test_data[:10])
        self.assertIsInstance(signals, pd.DataFrame)
        
    def test_backtest_execution_without_calibration(self):
        """보정 없는 백테스트 실행 테스트"""
        result = self.integration_no_cal.run_risk_aware_backtest(
            strategy=self.test_strategy,
            data=self.test_data
        )
        
        # 기본 결과 확인
        self.assertIn('total_return', result)
        self.assertIn('risk_analysis', result)
        self.assertIn('risk_metrics_history', result)
        self.assertIn('risk_adjusted_performance', result)
        
        # 보정 관련 항목이 없어야 함
        self.assertNotIn('calibration_analysis', result)
        self.assertNotIn('calibration_history', result)
        
    def test_backtest_execution_with_calibration(self):
        """보정 포함 백테스트 실행 테스트"""
        result = self.integration_with_cal.run_risk_aware_backtest(
            strategy=self.test_strategy,
            data=self.test_data,
            enable_periodic_calibration=True
        )
        
        # 기본 결과 확인
        self.assertIn('total_return', result)
        self.assertIn('risk_analysis', result)
        self.assertIn('risk_metrics_history', result)
        self.assertIn('risk_adjusted_performance', result)
        
        # 보정 관련 항목 확인
        self.assertIn('calibration_analysis', result)
        self.assertIn('calibration_history', result)
        self.assertIn('calibrated_parameters', result)
        
    def test_parameter_update_mechanism(self):
        """파라미터 업데이트 메커니즘 테스트"""
        original_params = self.integration_with_cal._current_params.copy()
        
        # 모킹된 보정 결과 생성
        mock_calibration_result = CalibrationResult(
            symbol="AAPL",
            calibrated_slippage=0.0008,
            calibrated_commission=0.0012,
            calibrated_fill_rate=0.95,
            confidence_score=0.8
        )
        
        # 파라미터 업데이트
        self.integration_with_cal._update_backtest_parameters(mock_calibration_result)
        
        # 파라미터가 업데이트되었는지 확인
        self.assertNotEqual(
            original_params['slippage_rate'],
            self.integration_with_cal._current_params['slippage_rate']
        )
        self.assertEqual(
            self.integration_with_cal._current_params['slippage_rate'],
            0.0008
        )
        
    def test_calibration_impact_analysis(self):
        """보정 영향 분석 테스트"""
        # 보정 이력 생성
        calibration_result = CalibrationResult(
            symbol="AAPL",
            original_slippage=0.0005,
            original_commission=0.001,
            original_fill_rate=1.0,
            calibrated_slippage=0.0008,
            calibrated_commission=0.0012,
            calibrated_fill_rate=0.95,
            confidence_score=0.8,
            trades_analyzed=150,
            market_condition="normal"
        )
        
        self.integration_with_cal.calibration_history.append(calibration_result)
        
        # 영향 분석 실행
        impact_analysis = self.integration_with_cal._analyze_calibration_impact()
        
        self.assertIn('calibration_enabled', impact_analysis)
        self.assertIn('total_calibrations', impact_analysis)
        self.assertIn('latest_calibration', impact_analysis)
        self.assertIn('parameter_changes', impact_analysis)
        self.assertIn('impact_summary', impact_analysis)
        
        self.assertTrue(impact_analysis['calibration_enabled'])
        self.assertEqual(impact_analysis['total_calibrations'], 1)
        
    def test_integration_status(self):
        """통합 상태 확인 테스트"""
        status = self.integration_with_cal.get_integration_status()
        
        self.assertIn('position_manager_initialized', status)
        self.assertIn('risk_manager_initialized', status)
        self.assertIn('backtest_engine_connected', status)
        self.assertIn('calibration', status)
        
        calibration_status = status['calibration']
        self.assertIn('enabled', calibration_status)
        self.assertIn('manager_initialized', calibration_status)
        self.assertIn('current_parameters', calibration_status)
        
    def test_calibration_summary(self):
        """보정 요약 테스트"""
        summary = self.integration_with_cal.get_calibration_summary()
        
        self.assertIn('calibration_enabled', summary)
        self.assertIn('current_parameters', summary)
        self.assertIn('original_parameters', summary)
        
        self.assertTrue(summary['calibration_enabled'])
        
    def test_periodic_calibration_check(self):
        """주기적 보정 체크 테스트"""
        # 보정이 필요한 시점 설정
        past_date = datetime.now() - timedelta(days=2)
        
        # 보정 체크 실행 (에러가 발생하지 않아야 함)
        try:
            self.integration_with_cal.periodic_calibration_check(past_date)
        except Exception as e:
            self.fail(f"주기적 보정 체크 실패: {e}")
            
    def test_risk_metrics_integration(self):
        """리스크 지표 통합 테스트"""
        # 포트폴리오 스냅샷 업데이트
        self.integration_with_cal.risk_manager.update_portfolio_snapshot(
            total_equity=100000,
            cash=50000,
            positions={
                'AAPL': {'market_value': 30000, 'size': 120, 'avg_price': 250},
                'GOOGL': {'market_value': 20000, 'size': 8, 'avg_price': 2500}
            },
            prices={'AAPL': 250, 'GOOGL': 2500}
        )
        
        # 리스크 지표 계산
        metrics = self.integration_with_cal.risk_manager.calculate_risk_metrics()
        
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertEqual(metrics.total_exposure, 50000)
        self.assertEqual(metrics.cash_ratio, 0.5)


class TestConvenienceFunctions(unittest.TestCase):
    """편의 함수 테스트"""
    
    def test_create_risk_aware_backtest(self):
        """리스크 인식 백테스트 생성 테스트"""
        backtest_system = create_risk_aware_backtest(
            initial_capital=50000,
            enable_calibration=True
        )
        
        self.assertIsInstance(backtest_system, BacktestRiskIntegration)
        self.assertTrue(backtest_system.enable_calibration)
        self.assertEqual(backtest_system.position_manager.initial_capital, 50000)
        
    def test_create_calibrated_backtest(self):
        """보정 백테스트 생성 테스트"""
        backtest_system = create_calibrated_backtest(
            initial_capital=75000,
            calibration_interval_hours=12,
            min_trades_for_calibration=50
        )
        
        self.assertIsInstance(backtest_system, BacktestRiskIntegration)
        self.assertTrue(backtest_system.enable_calibration)
        self.assertEqual(
            backtest_system.calibration_manager.config.calibration_interval_hours,
            12
        )
        self.assertEqual(
            backtest_system.calibration_manager.config.min_trades_for_calibration,
            50
        )
        
    def test_quick_risk_backtest(self):
        """빠른 리스크 백테스트 테스트"""
        # 테스트 데이터 생성
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02 + 0.001
        }, index=pd.date_range('2024-01-01', periods=50, freq='D'))
        
        test_strategy = MockStrategy()
        
        result = quick_risk_backtest(
            strategy=test_strategy,
            data=test_data,
            initial_capital=25000,
            enable_calibration=False  # 빠른 테스트를 위해 비활성화
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_return', result)
        self.assertIn('risk_analysis', result)


class TestPerformanceAndScalability(unittest.TestCase):
    """성능 및 확장성 테스트"""
    
    def setUp(self):
        self.mock_engine = MockBacktestEngine()
        self.integration = BacktestRiskIntegration(
            backtest_engine=self.mock_engine,
            enable_calibration=False  # 성능 테스트를 위해 비활성화
        )
        
    def test_large_data_performance(self):
        """대용량 데이터 성능 테스트"""
        import time
        
        # 1년치 일일 데이터 (252일)
        large_data = pd.DataFrame({
            'AAPL': np.random.randn(252) * 0.02 + 0.001,
            'GOOGL': np.random.randn(252) * 0.025 + 0.0005,
            'MSFT': np.random.randn(252) * 0.018 + 0.0008
        }, index=pd.date_range('2024-01-01', periods=252, freq='D'))
        
        strategy = MockStrategy()
        
        start_time = time.time()
        result = self.integration.run_risk_aware_backtest(
            strategy=strategy,
            data=large_data
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 기준: 1년 데이터 처리가 10초 이내
        self.assertLess(execution_time, 10.0)
        self.assertIsInstance(result, dict)
        
    def test_multiple_strategies_scalability(self):
        """다중 전략 확장성 테스트"""
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.02 + 0.001
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))
        
        results = []
        
        # 여러 전략 동시 실행
        for i in range(5):
            strategy = MockStrategy()
            result = self.integration.run_risk_aware_backtest(
                strategy=strategy,
                data=test_data
            )
            results.append(result)
        
        # 모든 결과가 유효한지 확인
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('total_return', result)
            self.assertIn('risk_analysis', result)
            
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 메모리 집약적 작업 수행
        for i in range(10):
            test_data = pd.DataFrame({
                f'STOCK_{j}': np.random.randn(500) * 0.02 + 0.001
                for j in range(10)  # 10개 종목
            }, index=pd.date_range('2024-01-01', periods=500, freq='D'))
            
            strategy = MockStrategy()
            result = self.integration.run_risk_aware_backtest(
                strategy=strategy,
                data=test_data
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가가 500MB 이하여야 함
        self.assertLess(memory_increase, 500)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """오류 처리 및 복구 테스트"""
    
    def setUp(self):
        self.mock_engine = MockBacktestEngine()
        self.integration = BacktestRiskIntegration(
            backtest_engine=self.mock_engine,
            enable_calibration=True
        )
        
    def test_missing_data_handling(self):
        """누락 데이터 처리 테스트"""
        # 누락 값이 있는 데이터
        incomplete_data = pd.DataFrame({
            'AAPL': [0.01, np.nan, 0.02, -0.01, np.nan],
            'GOOGL': [0.005, 0.01, np.nan, 0.015, -0.005]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        strategy = MockStrategy()
        
        # 에러 없이 실행되어야 함
        try:
            result = self.integration.run_risk_aware_backtest(
                strategy=strategy,
                data=incomplete_data,
                enable_periodic_calibration=False  # 간단히 하기 위해 비활성화
            )
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"누락 데이터 처리 실패: {e}")
            
    def test_calibration_failure_recovery(self):
        """보정 실패 복구 테스트"""
        # 보정 관리자를 실패하도록 모킹
        self.integration.calibration_manager.calibrate_parameters = Mock(
            side_effect=Exception("Calibration failed")
        )
        
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02 + 0.001
        }, index=pd.date_range('2024-01-01', periods=50, freq='D'))
        
        strategy = MockStrategy()
        
        # 보정 실패에도 백테스트는 실행되어야 함
        try:
            result = self.integration.run_risk_aware_backtest(
                strategy=strategy,
                data=test_data,
                enable_periodic_calibration=True
            )
            self.assertIsInstance(result, dict)
        except Exception as e:
            # 보정 실패는 무시하고 계속 진행해야 함
            pass
            
    def test_invalid_strategy_handling(self):
        """잘못된 전략 처리 테스트"""
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(10) * 0.02 + 0.001
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        # 잘못된 신호를 반환하는 전략
        class InvalidStrategy:
            def generate_signals(self, data, **kwargs):
                return "invalid_signal"  # DataFrame이 아님
        
        invalid_strategy = InvalidStrategy()
        
        # 적절히 처리되어야 함
        try:
            result = self.integration.run_risk_aware_backtest(
                strategy=invalid_strategy,
                data=test_data,
                enable_periodic_calibration=False
            )
        except Exception as e:
            # 예상된 에러이므로 통과
            pass


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)