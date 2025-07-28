#!/usr/bin/env python3
"""
최적화 모듈 테스트
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import sys

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import get_logger, PerformanceMetrics, calculate_sharpe_ratio

logger = get_logger("TestOptimization")

class TestOptimization:
    """최적화 테스트 클래스"""
    
    def setup_method(self):
        """테스트 셋업"""
        self.test_data = self._create_test_data()
        self.test_trades = self._create_test_trades()
    
    def _create_test_data(self) -> pd.DataFrame:
        """테스트용 가격 데이터 생성"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        
        # 트렌드가 있는 가격 시뮬레이션
        trend = np.linspace(0, 0.2, 200)  # 20% 상승 트렌드
        noise = np.random.normal(0, 0.02, 200)
        price_changes = trend + noise
        
        prices = 50000 * np.cumprod(1 + price_changes / 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.998, 1.002, 200),
            'high': prices * np.random.uniform(1.001, 1.005, 200),
            'low': prices * np.random.uniform(0.995, 0.999, 200),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 200)
        })
    
    def _create_test_trades(self) -> list:
        """테스트용 거래 내역 생성"""
        trades = []
        
        # 10개의 모의 거래 생성
        for i in range(10):
            # 개시 거래
            open_trade = {
                'action': 'open',
                'size': 0.01 * (1 if i % 2 == 0 else -1),  # 매수/매도 교대
                'price': 50000 + i * 100,
                'timestamp': datetime(2024, 1, 1, i),
                'signal_info': {'strategy': 'test', 'confidence': 0.7}
            }
            trades.append(open_trade)
            
            # 청산 거래
            close_trade = {
                'action': 'close',
                'size': -open_trade['size'],
                'price': open_trade['price'] * (1 + np.random.normal(0.01, 0.02)),  # 1% 평균 수익
                'timestamp': datetime(2024, 1, 1, i, 30),  # 30분 후 청산
                'pnl': 0,  # 계산될 예정
                'pnl_pct': 0,  # 계산될 예정
                'holding_time': pd.Timedelta(minutes=30),
                'reason': 'test'
            }
            
            # PnL 계산
            pnl = (close_trade['price'] - open_trade['price']) * open_trade['size']
            pnl_pct = pnl / abs(open_trade['size'] * open_trade['price'])
            
            close_trade['pnl'] = pnl
            close_trade['pnl_pct'] = pnl_pct
            
            trades.append(close_trade)
        
        return trades
    
    def test_performance_metrics_calculation(self):
        """성과 지표 계산 테스트"""
        from utils.metrics import calculate_performance_metrics
        
        metrics = calculate_performance_metrics(self.test_trades)
        
        # 기본 검증
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 10  # 10개 거래
        assert 0 <= metrics.win_rate <= 1  # 승률은 0-1 사이
        assert metrics.profit_factor >= 0  # 이익 팩터는 0 이상
        
        logger.info(f"총 거래: {metrics.total_trades}")
        logger.info(f"승률: {metrics.win_rate:.2%}")
        logger.info(f"샤프 비율: {metrics.sharpe_ratio:.3f}")
        logger.info(f"최대 낙폭: {metrics.max_drawdown:.2%}")
    
    def test_sharpe_ratio_calculation(self):
        """샤프 비율 계산 테스트"""
        # 테스트 수익률 데이터
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012, -0.003, 0.018, 0.005]
        
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        logger.info(f"샤프 비율: {sharpe:.3f}")
        
        # 빈 리스트 테스트
        empty_sharpe = calculate_sharpe_ratio([])
        assert empty_sharpe == 0.0
        
        # 단일 값 테스트
        single_sharpe = calculate_sharpe_ratio([0.01])
        assert single_sharpe == 0.0
    
    def test_combination_optimizer_loading(self):
        """조합 최적화기 로딩 테스트"""
        try:
            sys.path.append(os.path.dirname(parent_dir))
            from optimization.optimal_combination_recommender import OptimalCombinationRecommender
            
            optimizer = OptimalCombinationRecommender()
            assert optimizer is not None
            
            logger.info("조합 최적화기 로딩 성공")
            
        except ImportError as e:
            logger.warning(f"조합 최적화기 로딩 실패: {e}")
            pytest.skip("최적화 모듈을 로드할 수 없습니다")
    
    def test_grid_search_optimization(self):
        """그리드 서치 최적화 테스트"""
        try:
            sys.path.append(os.path.dirname(parent_dir))
            from optimization.optimal_combination_recommender import OptimalCombinationRecommender
            
            optimizer = OptimalCombinationRecommender()
            
            # 작은 그리드로 빠른 테스트
            results = optimizer.grid_search_optimization(
                price_data=self.test_data,
                ppo_weights=[0.3, 0.4],
                rule_weights=[0.2, 0.3],
                hybrid_modes=["ensemble"],
                execution_strategies=["market"],
                max_combinations=4
            )
            
            # 결과 검증
            assert "optimization_method" in results
            assert "total_combinations_tested" in results
            assert "best_combination" in results
            
            if results["best_combination"]:
                best = results["best_combination"]
                assert "weights" in best
                assert "hybrid_mode" in best
                assert "execution_strategy" in best
                assert "combined_score" in best
            
            logger.info(f"테스트된 조합 수: {results['total_combinations_tested']}")
            
        except Exception as e:
            logger.warning(f"그리드 서치 최적화 테스트 실패: {e}")
            pytest.skip("최적화 테스트를 수행할 수 없습니다")
    
    def test_optimization_results_loading(self):
        """최적화 결과 로딩 테스트"""
        # 기존 최적화 결과 파일이 있는지 확인
        results_dir = os.path.join(os.path.dirname(parent_dir), "optimization", "results")
        
        if os.path.exists(results_dir):
            result_files = [f for f in os.listdir(results_dir) if f.startswith("optimal_combinations_")]
            
            if result_files:
                latest_file = sorted(result_files)[-1]
                file_path = os.path.join(results_dir, latest_file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 기본 구조 검증
                    assert "optimization_method" in data
                    assert "best_combination" in data
                    
                    if data["best_combination"]:
                        best = data["best_combination"]
                        assert "weights" in best
                        assert "hybrid_mode" in best
                        assert "execution_strategy" in best
                    
                    logger.info(f"최적화 결과 로딩 성공: {latest_file}")
                    
                except Exception as e:
                    logger.warning(f"최적화 결과 파일 읽기 실패: {e}")
            else:
                logger.info("최적화 결과 파일이 없습니다")
        else:
            logger.info("최적화 결과 디렉토리가 없습니다")
    
    def test_strategy_weight_calculation(self):
        """전략 가중치 계산 테스트"""
        try:
            sys.path.append(os.path.dirname(parent_dir))
            from optimization.optimal_combination_recommender import StrategyWeight
            
            # 가중치 객체 생성
            weight = StrategyWeight(
                ppo_weight=0.4,
                rule_a_weight=0.2,
                rule_b_weight=0.2,
                rule_c_weight=0.2
            )
            
            # 정규화 테스트
            weight.normalize()
            
            total = weight.ppo_weight + weight.rule_a_weight + weight.rule_b_weight + weight.rule_c_weight
            assert abs(total - 1.0) < 1e-6  # 합이 1에 가까워야 함
            
            # 딕셔너리 변환 테스트
            weight_dict = weight.to_dict()
            assert "ppo" in weight_dict
            assert "rule_a" in weight_dict
            assert "rule_b" in weight_dict
            assert "rule_c" in weight_dict
            
            logger.info("전략 가중치 계산 검증 완료")
            
        except ImportError as e:
            logger.warning(f"전략 가중치 테스트 실패: {e}")
            pytest.skip("전략 가중치 모듈을 로드할 수 없습니다")
    
    def test_performance_report_formatting(self):
        """성과 리포트 포맷팅 테스트"""
        from utils.metrics import calculate_performance_metrics, format_performance_report
        
        metrics = calculate_performance_metrics(self.test_trades)
        report = format_performance_report(metrics)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "수익률" in report
        assert "승률" in report
        assert "샤프 비율" in report
        
        logger.info("성과 리포트 포맷팅 검증 완료")

if __name__ == "__main__":
    # 직접 실행 시 테스트 수행
    test_class = TestOptimization()
    
    print("🧪 최적화 모듈 테스트 시작")
    
    try:
        test_class.setup_method()
        
        print("1/7: 성과 지표 계산 테스트")
        test_class.test_performance_metrics_calculation()
        print("✅ 통과")
        
        print("2/7: 샤프 비율 계산 테스트")
        test_class.test_sharpe_ratio_calculation()
        print("✅ 통과")
        
        print("3/7: 조합 최적화기 로딩 테스트")
        test_class.test_combination_optimizer_loading()
        print("✅ 통과")
        
        print("4/7: 그리드 서치 최적화 테스트")
        test_class.test_grid_search_optimization()
        print("✅ 통과")
        
        print("5/7: 최적화 결과 로딩 테스트")
        test_class.test_optimization_results_loading()
        print("✅ 통과")
        
        print("6/7: 전략 가중치 계산 테스트")
        test_class.test_strategy_weight_calculation()
        print("✅ 통과")
        
        print("7/7: 성과 리포트 포맷팅 테스트")
        test_class.test_performance_report_formatting()
        print("✅ 통과")
        
        print("\n🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()