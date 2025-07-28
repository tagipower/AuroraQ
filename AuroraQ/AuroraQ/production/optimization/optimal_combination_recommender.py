#!/usr/bin/env python3
"""
최적 조합 추천 시스템
PPO와 Rule 전략들의 최적 가중치 조합을 학습하고 추천하는 시스템
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import logging

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import get_logger
from backtest.complete_hybrid_backtest import CompleteHybridBacktestSystem

logger = get_logger("OptimalCombinationRecommender")

@dataclass
class StrategyWeight:
    """전략 가중치"""
    ppo_weight: float
    rule_a_weight: float
    rule_b_weight: float
    rule_c_weight: float
    
    def normalize(self):
        """가중치 정규화"""
        total = self.ppo_weight + self.rule_a_weight + self.rule_b_weight + self.rule_c_weight
        if total > 0:
            self.ppo_weight /= total
            self.rule_a_weight /= total
            self.rule_b_weight /= total
            self.rule_c_weight /= total
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "ppo": self.ppo_weight,
            "rule_a": self.rule_a_weight,
            "rule_b": self.rule_b_weight,
            "rule_c": self.rule_c_weight
        }

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    execution_rate: float
    signal_quality: float
    
    def combined_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """종합 점수 계산"""
        if weights is None:
            weights = {
                "sharpe_ratio": 0.25,
                "win_rate": 0.20,
                "profit_factor": 0.15,
                "execution_rate": 0.15,
                "max_drawdown": -0.10,  # 음수 가중치 (낮을수록 좋음)
                "volatility": -0.05,    # 음수 가중치
                "signal_quality": 0.10
            }
        
        score = (
            self.sharpe_ratio * weights.get("sharpe_ratio", 0) +
            self.win_rate * weights.get("win_rate", 0) +
            self.profit_factor * weights.get("profit_factor", 0) +
            self.execution_rate * weights.get("execution_rate", 0) +
            self.max_drawdown * weights.get("max_drawdown", 0) +
            self.volatility * weights.get("volatility", 0) +
            self.signal_quality * weights.get("signal_quality", 0)
        )
        
        return score

class OptimalCombinationRecommender:
    """최적 조합 추천 시스템"""
    
    def __init__(self):
        self.tested_combinations = []
        self.best_combination = None
        self.optimization_history = []
        
    def grid_search_optimization(self, 
                                price_data: pd.DataFrame,
                                ppo_weights: List[float] = None,
                                rule_weights: List[float] = None,
                                hybrid_modes: List[str] = None,
                                execution_strategies: List[str] = None,
                                max_combinations: int = 50) -> Dict[str, Any]:
        """그리드 서치를 통한 최적 조합 탐색"""
        
        logger.info("=== 그리드 서치 최적화 시작 ===")
        
        # 기본값 설정
        if ppo_weights is None:
            ppo_weights = [0.2, 0.3, 0.4, 0.5, 0.6]
        if rule_weights is None:
            rule_weights = [0.1, 0.15, 0.2, 0.25]
        if hybrid_modes is None:
            hybrid_modes = ["ensemble", "consensus", "competition"]
        if execution_strategies is None:
            execution_strategies = ["market", "limit", "smart"]
        
        # 조합 생성
        combinations = []
        for ppo_w in ppo_weights:
            remaining = 1.0 - ppo_w
            for rule_a_w in rule_weights:
                for rule_b_w in rule_weights:
                    for rule_c_w in rule_weights:
                        total_rule = rule_a_w + rule_b_w + rule_c_w
                        if total_rule <= remaining and total_rule > 0:
                            # 정규화
                            norm_factor = remaining / total_rule
                            weight = StrategyWeight(
                                ppo_weight=ppo_w,
                                rule_a_weight=rule_a_w * norm_factor,
                                rule_b_weight=rule_b_w * norm_factor,
                                rule_c_weight=rule_c_w * norm_factor
                            )
                            
                            for hybrid_mode in hybrid_modes:
                                for exec_strategy in execution_strategies:
                                    combinations.append({
                                        "weights": weight,
                                        "hybrid_mode": hybrid_mode,
                                        "execution_strategy": exec_strategy
                                    })
        
        # 조합 수 제한
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        logger.info(f"총 {len(combinations)}개 조합 테스트 시작")
        
        results = []
        for i, combo in enumerate(combinations):
            try:
                logger.info(f"조합 테스트 {i+1}/{len(combinations)}: {combo['hybrid_mode']}/{combo['execution_strategy']}")
                
                # 백테스트 실행
                performance = self._test_combination(price_data, combo)
                
                result = {
                    "combination_id": i + 1,
                    "weights": combo["weights"].to_dict(),
                    "hybrid_mode": combo["hybrid_mode"],
                    "execution_strategy": combo["execution_strategy"],
                    "performance": performance,
                    "combined_score": performance.combined_score() if performance else 0.0
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"조합 {i+1} 테스트 실패: {e}")
                continue
        
        # 결과 정렬 (점수 높은 순)
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # 최적 조합 저장
        if results:
            self.best_combination = results[0]
        
        logger.info("=== 그리드 서치 최적화 완료 ===")
        
        return {
            "optimization_method": "grid_search",
            "total_combinations_tested": len(results),
            "best_combination": results[0] if results else None,
            "top_combinations": results[:10],  # 상위 10개
            "optimization_summary": self._generate_optimization_summary(results)
        }
    
    def _test_combination(self, 
                         price_data: pd.DataFrame, 
                         combination: Dict[str, Any]) -> Optional[PerformanceMetrics]:
        """개별 조합 테스트"""
        
        try:
            # 하이브리드 백테스트 시스템 생성
            system = CompleteHybridBacktestSystem(
                rule_strategies=["RuleStrategyA", "RuleStrategyB", "RuleStrategyC"],
                enable_ppo=True,
                hybrid_mode=combination["hybrid_mode"],
                execution_strategy=combination["execution_strategy"],
                risk_tolerance="moderate"
            )
            
            # 백테스트 실행 (작은 샘플로 빠른 테스트)
            result = system.run_complete_backtest(
                price_data, 
                start_index=50, 
                max_iterations=min(30, len(price_data) - 60)
            )
            
            # 성과 지표 계산
            return self._calculate_performance_metrics(result)
            
        except Exception as e:
            logger.error(f"조합 테스트 중 오류: {e}")
            return None
    
    def _calculate_performance_metrics(self, backtest_result: Dict[str, Any]) -> PerformanceMetrics:
        """백테스트 결과에서 성과 지표 계산"""
        
        # 기본값
        metrics = PerformanceMetrics(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            total_return=0.0,
            volatility=0.0,
            execution_rate=0.0,
            signal_quality=0.0
        )
        
        try:
            # 실행 통계에서 지표 추출
            exec_stats = backtest_result.get("execution_stats", {})
            performance_summary = backtest_result.get("performance_summary", {})
            
            # 실행률
            exec_analysis = performance_summary.get("execution_analysis", {})
            metrics.execution_rate = exec_analysis.get("execution_rate", 0.0)
            
            # 신호 품질 (신호율로 대체)
            signal_analysis = performance_summary.get("signal_analysis", {})
            metrics.signal_quality = signal_analysis.get("hybrid_signal_rate", 0.0)
            
            # 수익률 시뮬레이션 (간단한 근사)
            financial_perf = performance_summary.get("financial_performance", {})
            total_cost = financial_perf.get("total_cost", 0)
            
            # 체결된 거래 수에 따른 수익률 추정
            executed_trades = exec_analysis.get("executed_trades", 0)
            if executed_trades > 0:
                # 간단한 수익률 추정 (실제로는 더 복잡한 계산 필요)
                estimated_return = executed_trades * 0.01 - total_cost * 0.001  # 단순 추정
                metrics.total_return = estimated_return
                metrics.win_rate = min(0.8, executed_trades / 10.0)  # 간단한 승률 추정
                metrics.sharpe_ratio = estimated_return / max(0.1, abs(estimated_return * 0.5))
                metrics.volatility = abs(estimated_return * 0.3)
                metrics.max_drawdown = abs(estimated_return * 0.2)
                metrics.profit_factor = max(1.0, 1.0 + estimated_return)
            
        except Exception as e:
            logger.error(f"성과 지표 계산 오류: {e}")
        
        return metrics
    
    def _generate_optimization_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최적화 요약 생성"""
        
        if not results:
            return {"message": "테스트된 조합이 없습니다"}
        
        # 점수 분포
        scores = [r["combined_score"] for r in results]
        
        # 최고 성과 분석
        best = results[0]
        worst = results[-1]
        
        # 하이브리드 모드별 평균 성과
        mode_performance = {}
        for mode in ["ensemble", "consensus", "competition"]:
            mode_results = [r for r in results if r["hybrid_mode"] == mode]
            if mode_results:
                avg_score = np.mean([r["combined_score"] for r in mode_results])
                mode_performance[mode] = {
                    "average_score": avg_score,
                    "best_score": max(r["combined_score"] for r in mode_results),
                    "count": len(mode_results)
                }
        
        # 실행 전략별 평균 성과
        exec_performance = {}
        for strategy in ["market", "limit", "smart"]:
            strategy_results = [r for r in results if r["execution_strategy"] == strategy]
            if strategy_results:
                avg_score = np.mean([r["combined_score"] for r in strategy_results])
                exec_performance[strategy] = {
                    "average_score": avg_score,
                    "best_score": max(r["combined_score"] for r in strategy_results),
                    "count": len(strategy_results)
                }
        
        return {
            "total_combinations": len(results),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores),
                "median": np.median(scores)
            },
            "best_combination": {
                "score": best["combined_score"],
                "weights": best["weights"],
                "hybrid_mode": best["hybrid_mode"],
                "execution_strategy": best["execution_strategy"]
            },
            "worst_combination": {
                "score": worst["combined_score"],
                "weights": worst["weights"],
                "hybrid_mode": worst["hybrid_mode"],
                "execution_strategy": worst["execution_strategy"]
            },
            "hybrid_mode_performance": mode_performance,
            "execution_strategy_performance": exec_performance,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """추천사항 생성"""
        
        recommendations = []
        
        if not results:
            return ["테스트 결과가 없어 추천을 생성할 수 없습니다"]
        
        # 상위 조합들 분석
        top_5 = results[:5]
        
        # 하이브리드 모드 추천
        mode_counts = {}
        for result in top_5:
            mode = result["hybrid_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        best_mode = max(mode_counts, key=mode_counts.get)
        recommendations.append(f"하이브리드 모드: '{best_mode}' 모드가 상위 5개 조합 중 {mode_counts[best_mode]}개에서 사용됨")
        
        # 실행 전략 추천
        exec_counts = {}
        for result in top_5:
            strategy = result["execution_strategy"]
            exec_counts[strategy] = exec_counts.get(strategy, 0) + 1
        
        best_exec = max(exec_counts, key=exec_counts.get)
        recommendations.append(f"실행 전략: '{best_exec}' 전략이 상위 5개 조합 중 {exec_counts[best_exec]}개에서 사용됨")
        
        # 가중치 패턴 분석
        best_weights = results[0]["weights"]
        ppo_weight = best_weights["ppo"]
        
        if ppo_weight > 0.5:
            recommendations.append(f"PPO 가중치: {ppo_weight:.2f}로 높음 - PPO 중심 전략 권장")
        elif ppo_weight < 0.3:
            recommendations.append(f"PPO 가중치: {ppo_weight:.2f}로 낮음 - Rule 기반 전략 권장")
        else:
            recommendations.append(f"PPO 가중치: {ppo_weight:.2f}로 균형적 - 하이브리드 전략 권장")
        
        # 성과 개선 여지
        best_score = results[0]["combined_score"]
        if best_score < 0.5:
            recommendations.append("전반적인 성과가 낮음 - 전략 매개변수 튜닝 필요")
        elif best_score > 0.8:
            recommendations.append("우수한 성과 달성 - 현재 설정 유지 권장")
        
        return recommendations
    
    def save_optimization_results(self, 
                                results: Dict[str, Any], 
                                output_path: str = None) -> str:
        """최적화 결과 저장"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"optimization/results/optimal_combinations_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"최적화 결과 저장: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    
    # 테스트 데이터 로드
    try:
        data_path = "data/price/test_backtest_data.csv"
        if os.path.exists(data_path):
            price_data = pd.read_csv(data_path)
            logger.info(f"가격 데이터 로드: {len(price_data)}건")
        else:
            # 테스트 데이터 생성
            np.random.seed(42)
            price_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=200, freq='h'),
                'close': 50000 + np.cumsum(np.random.randn(200) * 100),
                'open': 50000 + np.cumsum(np.random.randn(200) * 100),
                'high': 50000 + np.cumsum(np.random.randn(200) * 100),
                'low': 50000 + np.cumsum(np.random.randn(200) * 100),
                'volume': np.random.randint(100000, 1000000, 200)
            })
            logger.info("테스트 데이터 생성")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    # 최적화 실행
    recommender = OptimalCombinationRecommender()
    
    logger.info("=== 최적 조합 추천 시스템 시작 ===")
    
    # 빠른 테스트를 위한 축소된 그리드
    results = recommender.grid_search_optimization(
        price_data=price_data,
        ppo_weights=[0.3, 0.4, 0.5],
        rule_weights=[0.15, 0.2],
        hybrid_modes=["ensemble", "consensus"],
        execution_strategies=["market", "smart"],
        max_combinations=12  # 빠른 테스트
    )
    
    # 결과 저장
    output_path = recommender.save_optimization_results(results)
    
    # 요약 출력
    if results["best_combination"]:
        best = results["best_combination"]
        logger.info("=== 최적 조합 결과 ===")
        logger.info(f"최고 점수: {best['combined_score']:.4f}")
        logger.info(f"하이브리드 모드: {best['hybrid_mode']}")
        logger.info(f"실행 전략: {best['execution_strategy']}")
        logger.info(f"가중치: {best['weights']}")
        
        recommendations = results["optimization_summary"]["recommendations"]
        logger.info("=== 추천사항 ===")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
    
    logger.info(f"상세 결과: {output_path}")
    logger.info("=== 최적 조합 추천 시스템 완료 ===")

if __name__ == "__main__":
    main()