#!/usr/bin/env python3
"""
리스크 관리 시스템 테스트
VaR, CVaR, MDD 계산 및 포지션 사이징 검증
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# 테스트 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management import (
    AdvancedRiskManager, VaRCalculator, RiskConfig, 
    RiskMetrics, PortfolioRiskAnalyzer
)
from position_management import UnifiedPositionManager, Position, PositionSide


class TestVaRCalculator(unittest.TestCase):
    """VaR 계산기 테스트"""
    
    def setUp(self):
        self.var_calculator = VaRCalculator()
        
        # 테스트용 수익률 데이터 생성
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)  # 일일 수익률
        
    def test_historical_var(self):
        """Historical VaR 계산 테스트"""
        result = self.var_calculator.calculate_var(
            self.returns, 
            method='historical', 
            confidence_level=0.95
        )
        
        self.assertIn('var_95', result)
        self.assertIn('cvar_95', result)
        self.assertIsInstance(result['var_95'], float)
        self.assertIsInstance(result['cvar_95'], float)
        
        # VaR은 음수여야 함 (손실을 나타냄)
        self.assertLess(result['var_95'], 0)
        
        # CVaR은 VaR보다 더 극단적이어야 함
        self.assertLess(result['cvar_95'], result['var_95'])
        
    def test_parametric_var(self):
        """Parametric VaR 계산 테스트"""
        result = self.var_calculator.calculate_var(
            self.returns,
            method='parametric',
            confidence_level=0.99
        )
        
        self.assertIn('var_99', result)
        self.assertIn('cvar_99', result)
        
        # 99% VaR은 95% VaR보다 극단적이어야 함
        var_95 = self.var_calculator.calculate_var(
            self.returns, method='parametric', confidence_level=0.95
        )['var_95']
        
        self.assertLess(result['var_99'], var_95)
        
    def test_monte_carlo_var(self):
        """Monte Carlo VaR 계산 테스트"""
        result = self.var_calculator.calculate_var(
            self.returns,
            method='monte_carlo',
            confidence_level=0.95,
            num_simulations=1000
        )
        
        self.assertIn('var_95', result)
        self.assertIn('cvar_95', result)
        self.assertIn('simulation_results', result)
        
        # 시뮬레이션 결과가 올바른 크기인지 확인
        self.assertEqual(len(result['simulation_results']), 1000)
        
    def test_cornish_fisher_var(self):
        """Cornish-Fisher VaR 계산 테스트"""
        result = self.var_calculator.calculate_var(
            self.returns,
            method='cornish_fisher',
            confidence_level=0.95
        )
        
        self.assertIn('var_95', result)
        self.assertIn('cvar_95', result)
        self.assertIn('skewness', result)
        self.assertIn('kurtosis', result)
        
    def test_portfolio_var(self):
        """포트폴리오 VaR 계산 테스트"""
        # 다중 자산 수익률 매트릭스
        returns_matrix = np.random.multivariate_normal(
            mean=[0.001, 0.0005, 0.0015],
            cov=[[0.0004, 0.0001, 0.0002],
                 [0.0001, 0.0003, 0.0001],
                 [0.0002, 0.0001, 0.0005]],
            size=252
        )
        
        weights = np.array([0.4, 0.3, 0.3])
        
        result = self.var_calculator.calculate_portfolio_var(
            returns_matrix,
            weights,
            confidence_level=0.95
        )
        
        self.assertIn('portfolio_var', result)
        self.assertIn('component_var', result)
        self.assertIn('marginal_var', result)
        
        # 컴포넌트 VaR의 합이 포트폴리오 VaR과 일치해야 함
        component_var_sum = sum(result['component_var'].values())
        self.assertAlmostEqual(
            component_var_sum, 
            result['portfolio_var'], 
            places=6
        )


class TestAdvancedRiskManager(unittest.TestCase):
    """고급 리스크 관리자 테스트"""
    
    def setUp(self):
        # 포지션 관리자 모킹
        self.position_manager = Mock(spec=UnifiedPositionManager)
        self.position_manager.get_equity.return_value = 100000
        self.position_manager.cash = 50000
        self.position_manager.positions = {}
        
        # 리스크 설정
        self.risk_config = RiskConfig(
            var_limit_pct=0.05,
            max_drawdown_limit=0.15,
            max_position_size_pct=0.20,
            max_sector_exposure_pct=0.30
        )
        
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.position_manager,
            config=self.risk_config
        )
        
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsInstance(self.risk_manager.var_calculator, VaRCalculator)
        self.assertEqual(self.risk_manager.config.var_limit_pct, 0.05)
        self.assertIsNotNone(self.risk_manager.portfolio_analyzer)
        
    def test_risk_metrics_calculation(self):
        """리스크 지표 계산 테스트"""
        # 포트폴리오 데이터 설정
        self.risk_manager.update_portfolio_snapshot(
            total_equity=100000,
            cash=50000,
            positions={
                'AAPL': {'market_value': 25000, 'size': 100, 'avg_price': 250},
                'GOOGL': {'market_value': 25000, 'size': 10, 'avg_price': 2500}
            },
            prices={'AAPL': 250, 'GOOGL': 2500}
        )
        
        metrics = self.risk_manager.calculate_risk_metrics()
        
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertEqual(metrics.total_exposure, 50000)
        self.assertEqual(metrics.cash_ratio, 0.5)
        self.assertGreaterEqual(metrics.overall_risk_score, 0)
        self.assertLessEqual(metrics.overall_risk_score, 100)
        
    def test_position_sizing_recommendation(self):
        """포지션 사이징 권고 테스트"""
        recommendation = self.risk_manager.get_position_sizing_recommendation(
            symbol='AAPL',
            current_price=250,
            signal_confidence=0.8
        )
        
        self.assertIn('recommended_size', recommendation)
        self.assertIn('max_allowed_size', recommendation)
        self.assertIn('risk_adjustment_factor', recommendation)
        self.assertIn('adjustments', recommendation)
        
        # 권고 크기는 최대 허용 크기를 초과하지 않아야 함
        self.assertLessEqual(
            recommendation['recommended_size'],
            recommendation['max_allowed_size']
        )
        
    def test_risk_limit_validation(self):
        """리스크 한도 검증 테스트"""
        # VaR 한도 초과 시나리오
        high_risk_metrics = RiskMetrics(
            var_95_pct=0.08,  # 한도(5%) 초과
            current_drawdown=0.10,
            total_exposure=80000,
            cash_ratio=0.2
        )
        
        alerts = self.risk_manager._validate_risk_limits(high_risk_metrics)
        
        # VaR 한도 초과 알림이 있어야 함
        var_alerts = [alert for alert in alerts if 'VaR' in alert.message]
        self.assertGreater(len(var_alerts), 0)
        
    def test_historical_returns_management(self):
        """과거 수익률 관리 테스트"""
        # 수익률 데이터 추가
        for i in range(10):
            date = datetime.now() - timedelta(days=i)
            returns = {'AAPL': 0.01, 'GOOGL': -0.005, 'PORTFOLIO': 0.002}
            self.risk_manager.add_returns_data(date, returns)
        
        # 데이터가 올바르게 저장되었는지 확인
        self.assertEqual(len(self.risk_manager.returns_history), 10)
        
        # 특정 종목의 수익률 조회
        aapl_returns = self.risk_manager.get_symbol_returns('AAPL', days=5)
        self.assertEqual(len(aapl_returns), 5)
        
    def test_risk_callback_system(self):
        """리스크 콜백 시스템 테스트"""
        callback_executed = []
        
        def test_callback(metrics, alerts):
            callback_executed.append((metrics, alerts))
        
        self.risk_manager.add_risk_callback(test_callback)
        
        # 리스크 지표 계산 (콜백 트리거)
        metrics = self.risk_manager.calculate_risk_metrics()
        
        # 콜백이 실행되었는지 확인
        self.assertEqual(len(callback_executed), 1)
        self.assertIsInstance(callback_executed[0][0], RiskMetrics)
        
    def test_stress_testing(self):
        """스트레스 테스트"""
        # 포트폴리오 설정
        self.risk_manager.update_portfolio_snapshot(
            total_equity=100000,
            cash=30000,
            positions={
                'AAPL': {'market_value': 35000, 'size': 140, 'avg_price': 250},
                'GOOGL': {'market_value': 35000, 'size': 14, 'avg_price': 2500}
            },
            prices={'AAPL': 250, 'GOOGL': 2500}
        )
        
        # 스트레스 시나리오 (20% 하락)
        stress_result = self.risk_manager.run_stress_test(
            scenarios={
                'market_crash': {'AAPL': -0.20, 'GOOGL': -0.15}
            }
        )
        
        self.assertIn('market_crash', stress_result)
        scenario_result = stress_result['market_crash']
        
        self.assertIn('portfolio_value_change', scenario_result)
        self.assertIn('new_portfolio_value', scenario_result)
        self.assertLess(scenario_result['portfolio_value_change'], 0)


class TestPortfolioRiskAnalyzer(unittest.TestCase):
    """포트폴리오 리스크 분석기 테스트"""
    
    def setUp(self):
        self.analyzer = PortfolioRiskAnalyzer()
        
    def test_correlation_analysis(self):
        """상관관계 분석 테스트"""
        # 테스트 수익률 데이터
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.0005, 0.025, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100)
        })
        
        correlation_analysis = self.analyzer.analyze_correlations(returns_data)
        
        self.assertIn('correlation_matrix', correlation_analysis)
        self.assertIn('high_correlation_pairs', correlation_analysis)
        self.assertIn('portfolio_diversification_ratio', correlation_analysis)
        
        # 상관관계 매트릭스 검증
        corr_matrix = correlation_analysis['correlation_matrix']
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # 대각선 요소는 1이어야 함
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix), 
            [1.0, 1.0, 1.0]
        )
        
    def test_sector_exposure_analysis(self):
        """섹터 노출 분석 테스트"""
        positions = {
            'AAPL': {'market_value': 25000, 'sector': 'Technology'},
            'GOOGL': {'market_value': 20000, 'sector': 'Technology'},
            'JPM': {'market_value': 15000, 'sector': 'Financial'},
            'JNJ': {'market_value': 10000, 'sector': 'Healthcare'}
        }
        
        sector_analysis = self.analyzer.analyze_sector_exposure(positions)
        
        self.assertIn('sector_weights', sector_analysis)
        self.assertIn('concentration_risk', sector_analysis)
        
        # 섹터 가중치 합계는 1이어야 함
        total_weight = sum(sector_analysis['sector_weights'].values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # Technology 섹터가 최대 노출이어야 함
        tech_weight = sector_analysis['sector_weights']['Technology']
        self.assertAlmostEqual(tech_weight, 45000/70000, places=6)
        
    def test_liquidity_analysis(self):
        """유동성 분석 테스트"""
        positions = {
            'AAPL': {
                'market_value': 25000, 
                'avg_daily_volume': 50000000,
                'size': 100
            },
            'SMALL_CAP': {
                'market_value': 15000,
                'avg_daily_volume': 100000,
                'size': 500
            }
        }
        
        liquidity_analysis = self.analyzer.analyze_liquidity_risk(positions)
        
        self.assertIn('liquidity_scores', liquidity_analysis)
        self.assertIn('illiquid_positions', liquidity_analysis)
        self.assertIn('portfolio_liquidity_score', liquidity_analysis)
        
        # AAPL은 SMALL_CAP보다 유동성이 높아야 함
        aapl_score = liquidity_analysis['liquidity_scores']['AAPL']
        small_cap_score = liquidity_analysis['liquidity_scores']['SMALL_CAP']
        self.assertGreater(aapl_score, small_cap_score)
        
    def test_drawdown_analysis(self):
        """낙폭 분석 테스트"""
        # 테스트 포트폴리오 가치 시계열
        portfolio_values = pd.Series([
            100000, 102000, 101000, 105000, 103000,
            98000, 95000, 97000, 99000, 101000
        ])
        
        drawdown_analysis = self.analyzer.analyze_drawdown(portfolio_values)
        
        self.assertIn('current_drawdown', drawdown_analysis)
        self.assertIn('max_drawdown', drawdown_analysis)
        self.assertIn('drawdown_duration', drawdown_analysis)
        
        # 최대 낙폭 계산 검증
        expected_max_dd = (95000 - 105000) / 105000
        self.assertAlmostEqual(
            drawdown_analysis['max_drawdown'],
            expected_max_dd,
            places=6
        )


class TestRiskIntegration(unittest.TestCase):
    """리스크 관리 통합 테스트"""
    
    def setUp(self):
        # 실제 포지션 관리자 생성
        self.position_manager = UnifiedPositionManager(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.position_manager,
            config=RiskConfig()
        )
        
    def test_end_to_end_risk_management(self):
        """엔드투엔드 리스크 관리 테스트"""
        # 1. 포지션 생성
        from position_management import OrderSignal, OrderSide
        
        signal = OrderSignal(
            symbol='AAPL',
            side=OrderSide.BUY,
            size=100,
            price=250,
            timestamp=datetime.now()
        )
        
        # 2. 리스크 기반 포지션 사이징
        sizing_rec = self.risk_manager.get_position_sizing_recommendation(
            symbol='AAPL',
            current_price=250,
            signal_confidence=0.8
        )
        
        # 3. 포지션 실행
        trade = self.position_manager.execute_trade(signal, 250, "TEST_STRATEGY")
        
        # 4. 리스크 지표 계산
        self.risk_manager.update_portfolio_snapshot(
            total_equity=self.position_manager.get_equity(),
            cash=self.position_manager.cash,
            positions={
                symbol: {
                    'market_value': pos.market_value,
                    'size': pos.size,
                    'avg_price': pos.avg_price
                } for symbol, pos in self.position_manager.positions.items()
            },
            prices={'AAPL': 250}
        )
        
        metrics = self.risk_manager.calculate_risk_metrics()
        
        # 5. 검증
        self.assertIsNotNone(trade)
        self.assertGreater(len(self.position_manager.positions), 0)
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertGreater(metrics.total_exposure, 0)
        
    def test_risk_limit_enforcement(self):
        """리스크 한도 시행 테스트"""
        # 큰 포지션으로 리스크 한도 테스트
        large_signal = OrderSignal(
            symbol='AAPL',
            side=OrderSide.BUY,
            size=500,  # 큰 사이즈
            price=250,
            timestamp=datetime.now()
        )
        
        # 포지션 사이징 권고 확인
        sizing_rec = self.risk_manager.get_position_sizing_recommendation(
            symbol='AAPL',
            current_price=250,
            signal_confidence=0.8
        )
        
        # 권고 크기가 원래 신호보다 작아야 함 (리스크 조정)
        self.assertLess(
            sizing_rec['recommended_size'],
            large_signal.size
        )
        
        # 조정 사유가 있어야 함
        self.assertGreater(
            len(sizing_rec['adjustments']['applied_limits']),
            0
        )


class TestRiskMetrics(unittest.TestCase):
    """리스크 지표 테스트"""
    
    def test_risk_metrics_creation(self):
        """리스크 지표 생성 테스트"""
        metrics = RiskMetrics(
            var_95_pct=0.045,
            var_99_pct=0.068,
            cvar_95_pct=0.055,
            current_drawdown=0.12,
            max_drawdown=0.18,
            total_exposure=75000,
            cash_ratio=0.25,
            position_count=5,
            largest_position_pct=0.18,
            sector_concentration=0.35,
            correlation_risk=0.42
        )
        
        # 기본 속성 검증
        self.assertEqual(metrics.var_95_pct, 0.045)
        self.assertEqual(metrics.total_exposure, 75000)
        
        # 전체 리스크 점수 계산 검증
        self.assertGreater(metrics.overall_risk_score, 0)
        self.assertLessEqual(metrics.overall_risk_score, 100)
        
    def test_risk_metrics_to_dict(self):
        """리스크 지표 딕셔너리 변환 테스트"""
        metrics = RiskMetrics()
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('var_95_pct', metrics_dict)
        self.assertIn('overall_risk_score', metrics_dict)
        self.assertIn('timestamp', metrics_dict)


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)