#!/usr/bin/env python3
"""
보정 시스템 테스트
실거래 데이터 분석 및 백테스트 파라미터 자동 보정 검증
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
import sys

# 테스트 환경 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import (
    CalibrationManager, CalibrationConfig, CalibrationResult,
    ExecutionAnalyzer, ExecutionMetrics, MarketConditionDetector,
    RealTradeMonitor, TradeRecord, MonitoringStats
)


class TestExecutionAnalyzer(unittest.TestCase):
    """실거래 체결 분석기 테스트"""
    
    def setUp(self):
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ExecutionAnalyzer(data_path=self.temp_dir)
        
        # 테스트용 실거래 데이터 생성
        self.sample_execution_data = self._create_sample_execution_data()
        
    def tearDown(self):
        # 임시 파일 정리
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def _create_sample_execution_data(self):
        """테스트용 실거래 데이터 생성"""
        np.random.seed(42)
        data = []
        
        for i in range(100):
            timestamp = datetime.now() - timedelta(days=i//5)
            side = np.random.choice(['buy', 'sell'])
            size = np.random.uniform(10, 1000)
            price = 150 + np.random.normal(0, 5)
            
            # 슬리피지 시뮬레이션
            slippage_factor = np.random.normal(0.0005, 0.0002)
            if side == 'buy':
                executed_price = price * (1 + abs(slippage_factor))
            else:
                executed_price = price * (1 - abs(slippage_factor))
            
            # 체결률 시뮬레이션
            fill_rate = np.random.choice([1.0, 0.95, 0.9], p=[0.8, 0.15, 0.05])
            executed_size = size * fill_rate
            
            commission = executed_size * executed_price * 0.001
            
            data.append({
                'timestamp': timestamp.isoformat(),
                'symbol': 'AAPL',
                'side': side,
                'size': size,
                'price': price,
                'executed_price': executed_price,
                'executed_size': executed_size,
                'commission': commission,
                'status': 'executed' if fill_rate == 1.0 else 'partial'
            })
        
        return data
    
    def _create_sample_log_file(self, filename):
        """샘플 로그 파일 생성"""
        log_path = os.path.join(self.temp_dir, filename)
        with open(log_path, 'w') as f:
            json.dump(self.sample_execution_data, f)
        return log_path
    
    def test_load_execution_data(self):
        """실거래 데이터 로드 테스트"""
        # 로그 파일 생성
        self._create_sample_log_file("execution_monitor_20240101.json")
        
        # 데이터 로드
        df = self.analyzer._load_execution_data("AAPL", None, None)
        
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('symbol', df.columns)
        self.assertIn('slippage', df.columns)  # 계산된 컬럼
        
    def test_slippage_analysis(self):
        """슬리피지 분석 테스트"""
        df = pd.DataFrame(self.sample_execution_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'])
        df['executed_price'] = pd.to_numeric(df['executed_price'])
        
        slippage_analysis = self.analyzer._analyze_slippage(df)
        
        self.assertIn('avg', slippage_analysis)
        self.assertIn('median', slippage_analysis)
        self.assertIn('std', slippage_analysis)
        self.assertIn('percentile_95', slippage_analysis)
        self.assertIn('by_size', slippage_analysis)
        self.assertIn('by_time', slippage_analysis)
        
        # 평균 슬리피지가 합리적인 범위 내에 있는지 확인
        self.assertGreater(slippage_analysis['avg'], -0.01)  # -1% 이상
        self.assertLess(slippage_analysis['avg'], 0.01)      # 1% 이하
        
    def test_fill_rate_analysis(self):
        """체결률 분석 테스트"""
        df = pd.DataFrame(self.sample_execution_data)
        df['executed_size'] = pd.to_numeric(df['executed_size'])
        df['size'] = pd.to_numeric(df['size'])
        
        fill_analysis = self.analyzer._analyze_fill_rates(df)
        
        self.assertIn('overall', fill_analysis)
        self.assertIn('partial', fill_analysis)
        self.assertIn('avg_time', fill_analysis)
        self.assertIn('by_size', fill_analysis)
        
        # 전체 체결률이 0과 1 사이에 있는지 확인
        self.assertGreaterEqual(fill_analysis['overall'], 0)
        self.assertLessEqual(fill_analysis['overall'], 1)
        
    def test_commission_analysis(self):
        """수수료 분석 테스트"""
        df = pd.DataFrame(self.sample_execution_data)
        df['commission'] = pd.to_numeric(df['commission'])
        df['executed_size'] = pd.to_numeric(df['executed_size'])
        df['executed_price'] = pd.to_numeric(df['executed_price'])
        
        commission_analysis = self.analyzer._analyze_commissions(df)
        
        self.assertIn('avg_rate', commission_analysis)
        self.assertIn('by_size', commission_analysis)
        self.assertIn('total', commission_analysis)
        
        # 수수료율이 합리적인 범위 내에 있는지 확인
        self.assertGreater(commission_analysis['avg_rate'], 0)
        self.assertLess(commission_analysis['avg_rate'], 0.01)  # 1% 미만
        
    def test_execution_quality_assessment(self):
        """체결 품질 평가 테스트"""
        df = pd.DataFrame(self.sample_execution_data)
        df['executed_size'] = pd.to_numeric(df['executed_size'])
        df['size'] = pd.to_numeric(df['size'])
        df['price'] = pd.to_numeric(df['price'])
        df['executed_price'] = pd.to_numeric(df['executed_price'])
        
        quality_analysis = self.analyzer._analyze_execution_quality(df)
        
        self.assertIn('score', quality_analysis)
        self.assertIn('latency', quality_analysis)
        
        # 품질 점수가 0과 1 사이에 있는지 확인
        self.assertGreaterEqual(quality_analysis['score'], 0)
        self.assertLessEqual(quality_analysis['score'], 1)
        
    def test_data_quality_assessment(self):
        """데이터 품질 평가 테스트"""
        df = pd.DataFrame(self.sample_execution_data)
        df['executed_size'] = pd.to_numeric(df['executed_size'])
        df['size'] = pd.to_numeric(df['size'])
        df['executed_price'] = pd.to_numeric(df['executed_price'])
        
        quality_score = self.analyzer._assess_data_quality(df)
        
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 1)
        
    def test_complete_analysis_workflow(self):
        """전체 분석 워크플로우 테스트"""
        # 로그 파일 생성
        self._create_sample_log_file("execution_monitor_20240101.json")
        
        # 전체 분석 실행
        metrics = self.analyzer.analyze_execution_logs("AAPL")
        
        self.assertIsInstance(metrics, ExecutionMetrics)
        self.assertEqual(metrics.symbol, "AAPL")
        self.assertGreater(metrics.total_trades, 0)
        self.assertGreater(metrics.data_quality_score, 0)


class TestMarketConditionDetector(unittest.TestCase):
    """시장 상황 감지기 테스트"""
    
    def setUp(self):
        self.detector = MarketConditionDetector()
        
    def test_market_condition_detection(self):
        """시장 상황 감지 테스트"""
        condition = self.detector.detect_current_condition("AAPL")
        
        expected_conditions = ['normal', 'high_volatility', 'low_liquidity', 
                             'market_stress', 'after_hours']
        self.assertIn(condition, expected_conditions)
        
    def test_detailed_condition_analysis(self):
        """상세 시장 상황 분석 테스트"""
        detailed_condition = self.detector.get_detailed_condition("AAPL")
        
        self.assertIsNotNone(detailed_condition.condition)
        self.assertGreaterEqual(detailed_condition.confidence, 0)
        self.assertLessEqual(detailed_condition.confidence, 1)
        self.assertIn(detailed_condition.volatility_level, 
                     ['low', 'normal', 'high', 'extreme'])
        self.assertIn(detailed_condition.liquidity_level,
                     ['high', 'normal', 'low', 'very_low'])
        
    def test_market_data_collection(self):
        """시장 데이터 수집 테스트"""
        market_data = self.detector._collect_market_data("AAPL")
        
        self.assertIn('symbol', market_data)
        self.assertIn('timestamp', market_data)
        self.assertIn('volatility', market_data)
        self.assertIn('volume', market_data)
        self.assertIn('liquidity', market_data)
        self.assertIn('trading_session', market_data)
        
    def test_trading_session_detection(self):
        """거래 세션 감지 테스트"""
        # 정규 시간 테스트
        regular_time = datetime.now().replace(hour=12)  # 12:00
        session = self.detector._get_trading_session(regular_time)
        self.assertEqual(session, 'regular')
        
        # 시간외 테스트
        after_hours_time = datetime.now().replace(hour=18)  # 18:00
        session = self.detector._get_trading_session(after_hours_time)
        self.assertEqual(session, 'after_hours')
        
    def test_condition_statistics(self):
        """시장 상황 통계 테스트"""
        stats = self.detector.get_condition_statistics("AAPL", lookback_days=7)
        
        if stats:  # 데이터가 있는 경우
            self.assertIn('condition_distribution', stats)
            self.assertIn('confidence_stats', stats)
            self.assertIn('most_common_condition', stats)
            self.assertIn('current_condition', stats)


class TestCalibrationManager(unittest.TestCase):
    """보정 관리자 테스트"""
    
    def setUp(self):
        # 모킹된 ExecutionAnalyzer
        self.mock_analyzer = Mock(spec=ExecutionAnalyzer)
        self.mock_analyzer.analyze_execution_logs.return_value = ExecutionMetrics(
            symbol="AAPL",
            total_trades=150,
            avg_slippage=0.0008,
            avg_commission_rate=0.0012,
            fill_rate=0.96,
            data_quality_score=0.85
        )
        
        self.config = CalibrationConfig(
            calibration_interval_hours=24,
            min_trades_for_calibration=100,
            slippage_weight=0.7,
            fill_rate_weight=0.8,
            commission_weight=1.0
        )
        
        self.calibration_manager = CalibrationManager(
            config=self.config,
            execution_analyzer=self.mock_analyzer
        )
        
    def test_calibration_config(self):
        """보정 설정 테스트"""
        self.assertEqual(self.config.calibration_interval_hours, 24)
        self.assertEqual(self.config.min_trades_for_calibration, 100)
        self.assertEqual(self.config.slippage_weight, 0.7)
        
    def test_should_calibrate_logic(self):
        """보정 필요성 판단 로직 테스트"""
        # 첫 보정 (항상 필요)
        should_calibrate = self.calibration_manager._should_calibrate("AAPL")
        self.assertTrue(should_calibrate)
        
        # 보정 실행 후
        result = self.calibration_manager.calibrate_parameters("AAPL")
        
        # 즉시 다시 체크 (불필요)
        should_calibrate = self.calibration_manager._should_calibrate("AAPL")
        self.assertFalse(should_calibrate)
        
    def test_slippage_calibration(self):
        """슬리피지 보정 테스트"""
        original_slippage = 0.0005
        metrics = ExecutionMetrics(avg_slippage=0.0012)
        
        calibrated = self.calibration_manager._calibrate_slippage(
            original_slippage, metrics, "normal"
        )
        
        # 보정된 값이 원본과 실거래 데이터 사이에 있어야 함
        self.assertGreater(calibrated, original_slippage)
        self.assertLess(calibrated, metrics.avg_slippage)
        
    def test_commission_calibration(self):
        """수수료 보정 테스트"""
        original_commission = 0.001
        metrics = ExecutionMetrics(avg_commission_rate=0.0015)
        
        calibrated = self.calibration_manager._calibrate_commission(
            original_commission, metrics
        )
        
        # 보정된 값이 합리적인 범위 내에 있어야 함
        self.assertGreater(calibrated, 0)
        self.assertLess(calibrated, self.config.max_commission_rate)
        
    def test_fill_rate_calibration(self):
        """체결률 보정 테스트"""
        original_fill_rate = 1.0
        metrics = ExecutionMetrics(fill_rate=0.95)
        
        calibrated = self.calibration_manager._calibrate_fill_rate(
            original_fill_rate, metrics, "normal"
        )
        
        # 보정된 값이 최소 체결률 이상이어야 함
        self.assertGreaterEqual(calibrated, self.config.min_fill_rate)
        self.assertLessEqual(calibrated, 1.0)
        
    def test_market_adjustment_factors(self):
        """시장 상황별 조정 팩터 테스트"""
        normal_factor = self.calibration_manager._get_market_adjustment_factor("normal")
        volatility_factor = self.calibration_manager._get_market_adjustment_factor("high_volatility")
        stress_factor = self.calibration_manager._get_market_adjustment_factor("market_stress")
        
        self.assertEqual(normal_factor, 1.0)
        self.assertGreater(volatility_factor, normal_factor)
        self.assertGreater(stress_factor, volatility_factor)
        
    def test_confidence_score_calculation(self):
        """신뢰도 점수 계산 테스트"""
        high_quality_metrics = ExecutionMetrics(
            total_trades=500,
            data_quality_score=0.95
        )
        
        low_quality_metrics = ExecutionMetrics(
            total_trades=50,
            data_quality_score=0.6
        )
        
        high_confidence = self.calibration_manager._calculate_confidence_score(
            high_quality_metrics, "normal"
        )
        
        low_confidence = self.calibration_manager._calculate_confidence_score(
            low_quality_metrics, "market_stress"
        )
        
        self.assertGreater(high_confidence, low_confidence)
        self.assertGreaterEqual(high_confidence, 0)
        self.assertLessEqual(high_confidence, 1)
        
    def test_calibration_callback_system(self):
        """보정 콜백 시스템 테스트"""
        callback_results = []
        
        def test_callback(result):
            callback_results.append(result)
        
        self.calibration_manager.add_calibration_callback(test_callback)
        
        # 보정 실행
        result = self.calibration_manager.calibrate_parameters("AAPL")
        
        # 콜백이 실행되었는지 확인
        self.assertEqual(len(callback_results), 1)
        self.assertIsInstance(callback_results[0], CalibrationResult)
        
    def test_calibration_result_structure(self):
        """보정 결과 구조 테스트"""
        result = self.calibration_manager.calibrate_parameters("AAPL")
        
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.symbol, "AAPL")
        self.assertGreater(result.trades_analyzed, 0)
        self.assertGreaterEqual(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1)
        
        # 보정된 값들이 설정된 범위 내에 있는지 확인
        self.assertGreater(result.calibrated_slippage, 0)
        self.assertGreater(result.calibrated_commission, 0)
        self.assertGreaterEqual(result.calibrated_fill_rate, self.config.min_fill_rate)
        
    def test_minimal_adjustment_fallback(self):
        """최소 보정 폴백 테스트"""
        # 데이터 부족 시나리오
        insufficient_metrics = ExecutionMetrics(
            symbol="TEST",
            total_trades=10,  # 최소 요구량 미만
            data_quality_score=0.5
        )
        
        self.mock_analyzer.analyze_execution_logs.return_value = insufficient_metrics
        
        result = self.calibration_manager._perform_calibration(
            "TEST", insufficient_metrics, "normal"
        )
        
        # 신뢰도가 낮아야 함
        self.assertLess(result.confidence_score, 0.5)
        
        # 조정 사유에 데이터 부족이 포함되어야 함
        self.assertIn("Insufficient trade data", result.adjustment_reason)


class TestRealTradeMonitor(unittest.TestCase):
    """실거래 모니터 테스트"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = RealTradeMonitor(log_directory=self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_trade_record_creation(self):
        """거래 기록 생성 테스트"""
        trade_record = self.monitor.record_order_execution(
            trade_id="TEST_001",
            symbol="AAPL",
            side="buy",
            order_size=100,
            order_price=250,
            executed_size=100,
            executed_price=250.5,
            commission=25.05
        )
        
        self.assertIsInstance(trade_record, TradeRecord)
        self.assertEqual(trade_record.trade_id, "TEST_001")
        self.assertEqual(trade_record.symbol, "AAPL")
        self.assertEqual(trade_record.executed_size, 100)
        
        # 슬리피지가 올바르게 계산되었는지 확인
        expected_slippage = (250.5 - 250) / 250
        self.assertAlmostEqual(trade_record.slippage, expected_slippage, places=6)
        
    def test_monitoring_stats_update(self):
        """모니터링 통계 업데이트 테스트"""
        # 여러 거래 기록
        for i in range(10):
            self.monitor.record_order_execution(
                trade_id=f"TEST_{i:03d}",
                symbol="AAPL",
                side="buy",
                order_size=100,
                order_price=250,
                executed_size=95 if i % 3 == 0 else 100,  # 일부 부분 체결
                executed_price=250 + np.random.normal(0, 0.1),
                commission=25
            )
        
        stats = self.monitor.get_monitoring_stats()
        
        self.assertIsInstance(stats, MonitoringStats)
        self.assertEqual(stats.total_trades, 10)
        self.assertGreaterEqual(stats.avg_fill_rate, 0)
        self.assertLessEqual(stats.avg_fill_rate, 1)
        
    def test_recent_trades_query(self):
        """최근 거래 조회 테스트"""
        # 거래 기록 생성
        for i in range(5):
            self.monitor.record_order_execution(
                trade_id=f"RECENT_{i:03d}",
                symbol="AAPL",
                side="buy",
                order_size=100,
                order_price=250,
                executed_size=100,
                executed_price=250,
                commission=25
            )
        
        recent_trades = self.monitor.get_recent_trades(symbol="AAPL", hours=1)
        
        self.assertEqual(len(recent_trades), 5)
        self.assertTrue(all(trade.symbol == "AAPL" for trade in recent_trades))
        
    def test_execution_summary(self):
        """체결 요약 통계 테스트"""
        # 다양한 거래 기록 생성
        for i in range(20):
            self.monitor.record_order_execution(
                trade_id=f"SUMMARY_{i:03d}",
                symbol="AAPL",
                side="buy" if i % 2 == 0 else "sell",
                order_size=100,
                order_price=250,
                executed_size=100 if i % 4 != 0 else 95,
                executed_price=250 + np.random.normal(0, 0.5),
                commission=25
            )
        
        summary = self.monitor.get_execution_summary(symbol="AAPL", hours=24)
        
        self.assertIn('total_trades', summary)
        self.assertIn('executed_trades', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('execution_stats', summary)
        
        self.assertEqual(summary['total_trades'], 20)
        self.assertGreater(summary['success_rate'], 0)


class TestCalibrationIntegration(unittest.TestCase):
    """보정 시스템 통합 테스트"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # 실제 컴포넌트들로 통합 테스트
        self.execution_analyzer = ExecutionAnalyzer(data_path=self.temp_dir)
        self.calibration_manager = CalibrationManager(
            execution_analyzer=self.execution_analyzer
        )
        self.real_trade_monitor = RealTradeMonitor(log_directory=self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_end_to_end_calibration_workflow(self):
        """엔드투엔드 보정 워크플로우 테스트"""
        # 1. 실거래 데이터 생성 및 모니터링
        trade_records = []
        for i in range(100):
            trade_record = self.real_trade_monitor.record_order_execution(
                trade_id=f"E2E_{i:03d}",
                symbol="AAPL",
                side="buy" if i % 2 == 0 else "sell",
                order_size=100 + np.random.uniform(-20, 20),
                order_price=250 + np.random.normal(0, 2),
                executed_size=100 if np.random.random() > 0.1 else 95,
                executed_price=250 + np.random.normal(0.1, 0.3),
                commission=25 + np.random.uniform(-2, 2)
            )
            trade_records.append(trade_record)
        
        # 2. 로그 파일 생성 (실제 환경 시뮬레이션)
        log_data = [record.to_dict() for record in trade_records]
        log_file = os.path.join(self.temp_dir, "execution_monitor_20240101.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f)
        
        # 3. 보정 실행
        calibration_result = self.calibration_manager.calibrate_parameters("AAPL")
        
        # 4. 결과 검증
        self.assertIsInstance(calibration_result, CalibrationResult)
        self.assertEqual(calibration_result.symbol, "AAPL")
        self.assertGreater(calibration_result.trades_analyzed, 90)  # 대부분의 거래 분석됨
        self.assertGreater(calibration_result.confidence_score, 0.5)  # 충분한 신뢰도
        
        # 보정 파라미터가 합리적인 범위 내에 있는지 확인
        self.assertGreater(calibration_result.calibrated_slippage, 0)
        self.assertLess(calibration_result.calibrated_slippage, 0.01)  # 1% 미만
        
        self.assertGreater(calibration_result.calibrated_commission, 0)
        self.assertLess(calibration_result.calibrated_commission, 0.01)  # 1% 미만
        
        self.assertGreaterEqual(calibration_result.calibrated_fill_rate, 0.8)
        self.assertLessEqual(calibration_result.calibrated_fill_rate, 1.0)
        
    def test_calibration_consistency(self):
        """보정 일관성 테스트"""
        # 동일한 데이터로 여러 번 보정 실행
        results = []
        
        # 테스트 데이터 생성
        for i in range(50):
            self.real_trade_monitor.record_order_execution(
                trade_id=f"CONSISTENCY_{i:03d}",
                symbol="AAPL",
                side="buy",
                order_size=100,
                order_price=250,
                executed_size=100,
                executed_price=250.1,  # 일정한 슬리피지
                commission=25
            )
        
        # 로그 파일 생성
        log_data = [record.to_dict() for record in self.real_trade_monitor.get_recent_trades()]
        log_file = os.path.join(self.temp_dir, "execution_monitor_consistency.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f)
        
        # 여러 번 보정 실행
        for _ in range(3):
            result = self.calibration_manager.calibrate_parameters("AAPL", force_calibration=True)
            results.append(result)
        
        # 결과 일관성 확인
        slippages = [r.calibrated_slippage for r in results]
        commissions = [r.calibrated_commission for r in results]
        
        # 표준편차가 작아야 함 (일관성)
        self.assertLess(np.std(slippages), 0.0001)
        self.assertLess(np.std(commissions), 0.0001)
        
    def test_calibration_performance(self):
        """보정 성능 테스트"""
        import time
        
        # 대량 데이터 생성
        for i in range(1000):
            self.real_trade_monitor.record_order_execution(
                trade_id=f"PERF_{i:04d}",
                symbol="AAPL",
                side="buy" if i % 2 == 0 else "sell",
                order_size=100,
                order_price=250,
                executed_size=100,
                executed_price=250 + np.random.normal(0, 0.1),
                commission=25
            )
        
        # 로그 파일 생성
        log_data = [record.to_dict() for record in self.real_trade_monitor.get_recent_trades()]
        log_file = os.path.join(self.temp_dir, "execution_monitor_performance.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f)
        
        # 성능 측정
        start_time = time.time()
        result = self.calibration_manager.calibrate_parameters("AAPL")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 기준: 1000건 처리가 5초 이내
        self.assertLess(execution_time, 5.0)
        self.assertGreater(result.trades_analyzed, 900)  # 대부분 처리됨


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)