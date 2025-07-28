"""
Strategy Scores JSON Generator
백테스트 결과를 기반으로 전략별 성과 점수를 생성하고 저장
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("StrategyScoresGenerator")


@dataclass
class StrategyScore:
    """전략 점수 데이터 클래스"""
    strategy_name: str
    total_return: float
    roi: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_win: float
    avg_loss: float
    consistency_score: float
    composite_score: float
    confidence_level: float
    
    # 추가 메트릭
    volatility: float
    calmar_ratio: float
    expectancy: float
    avg_holding_time_hours: float
    
    # 분류 점수
    risk_score: float        # 0-100 (낮을수록 안전)
    reward_score: float      # 0-100 (높을수록 수익성 좋음)
    stability_score: float   # 0-100 (높을수록 안정적)
    overall_grade: str       # A+, A, B+, B, C+, C, D, F


class StrategyScoresGenerator:
    """전략 점수 생성기"""
    
    def __init__(self, output_dir: str = "reports/strategy_scores"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 등급 기준
        self.grade_thresholds = {
            'A+': 90, 'A': 80, 'B+': 70, 'B': 60, 
            'C+': 50, 'C': 40, 'D': 30, 'F': 0
        }
    
    def generate_scores(self, backtest_results: Dict[str, Any]) -> Dict[str, StrategyScore]:
        """백테스트 결과로부터 전략 점수 생성"""
        strategy_scores = {}
        
        if 'metrics' not in backtest_results or 'all_metrics' not in backtest_results['metrics']:
            logger.error("백테스트 결과에 메트릭 정보가 없습니다")
            return strategy_scores
        
        all_metrics = backtest_results['metrics']['all_metrics']
        
        for strategy_name, metrics in all_metrics.items():
            try:
                score = self._calculate_strategy_score(strategy_name, metrics)
                strategy_scores[strategy_name] = score
                logger.info(f"{strategy_name} 점수 계산 완료: {score.overall_grade} ({score.composite_score:.2f})")
                
            except Exception as e:
                logger.error(f"{strategy_name} 점수 계산 실패: {e}")
                continue
        
        return strategy_scores
    
    def _calculate_strategy_score(self, strategy_name: str, metrics) -> StrategyScore:
        """개별 전략 점수 계산"""
        
        # 기본 메트릭 추출
        total_return = float(metrics.total_return)
        roi = float(metrics.roi)
        sharpe_ratio = float(metrics.sharpe_ratio)
        max_drawdown = float(metrics.max_drawdown)
        win_rate = float(metrics.win_rate)
        profit_factor = float(metrics.profit_factor)
        total_trades = int(metrics.total_trades)
        
        # 안전한 값 추출
        avg_win = float(metrics.avg_win) if metrics.avg_win > 0 else 0.0
        avg_loss = float(metrics.avg_loss) if metrics.avg_loss > 0 else 0.0
        volatility = float(metrics.volatility) if hasattr(metrics, 'volatility') else 0.0
        expectancy = float(metrics.expectancy) if hasattr(metrics, 'expectancy') else 0.0
        
        # 분류 점수 계산
        risk_score = self._calculate_risk_score(max_drawdown, volatility, sharpe_ratio)
        reward_score = self._calculate_reward_score(roi, profit_factor, win_rate)
        stability_score = self._calculate_stability_score(
            metrics.consistency_score if hasattr(metrics, 'consistency_score') else 0,
            total_trades,
            max_drawdown
        )
        
        # 종합 점수 계산
        composite_score = self._calculate_composite_score(risk_score, reward_score, stability_score)
        overall_grade = self._assign_grade(composite_score)
        
        return StrategyScore(
            strategy_name=strategy_name,
            total_return=total_return,
            roi=roi,
            annualized_return=float(metrics.annualized_return),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=float(metrics.sortino_ratio),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            consistency_score=float(metrics.consistency_score) if hasattr(metrics, 'consistency_score') else 0,
            composite_score=float(metrics.composite_score) if hasattr(metrics, 'composite_score') else composite_score,
            confidence_level=float(metrics.confidence_level) if hasattr(metrics, 'confidence_level') else 0,
            volatility=volatility,
            calmar_ratio=float(metrics.calmar_ratio) if hasattr(metrics, 'calmar_ratio') else 0,
            expectancy=expectancy,
            avg_holding_time_hours=float(metrics.avg_holding_time_hours) if hasattr(metrics, 'avg_holding_time_hours') else 0,
            risk_score=risk_score,
            reward_score=reward_score,
            stability_score=stability_score,
            overall_grade=overall_grade
        )
    
    def _calculate_risk_score(self, max_drawdown: float, volatility: float, sharpe_ratio: float) -> float:
        """리스크 점수 계산 (0-100, 낮을수록 안전)"""
        # Max Drawdown 점수 (50% 가중치)
        dd_score = min(max_drawdown * 1000, 100)  # 10% 드로우다운 = 100점
        
        # Volatility 점수 (30% 가중치)
        vol_score = min(volatility * 500, 100)  # 20% 변동성 = 100점
        
        # Sharpe ratio 점수 (20% 가중치, 역방향)
        sharpe_score = max(0, 100 - (sharpe_ratio + 2) * 25)  # 샤프 비율이 높을수록 리스크 점수 낮음
        
        return dd_score * 0.5 + vol_score * 0.3 + sharpe_score * 0.2
    
    def _calculate_reward_score(self, roi: float, profit_factor: float, win_rate: float) -> float:
        """수익 점수 계산 (0-100, 높을수록 수익성 좋음)"""
        # ROI 점수 (40% 가중치)
        roi_score = min(max(roi * 200, 0), 100)  # 50% ROI = 100점
        
        # Profit Factor 점수 (35% 가중치)
        pf_score = min(profit_factor * 50, 100)  # PF 2.0 = 100점
        
        # Win Rate 점수 (25% 가중치)
        wr_score = win_rate * 100  # 직접 퍼센트로 변환
        
        return roi_score * 0.4 + pf_score * 0.35 + wr_score * 0.25
    
    def _calculate_stability_score(self, consistency_score: float, total_trades: int, max_drawdown: float) -> float:
        """안정성 점수 계산 (0-100, 높을수록 안정적)"""
        # Consistency 점수 (50% 가중치)
        cons_score = consistency_score * 100
        
        # 거래 수 점수 (30% 가중치)
        trade_score = min(total_trades * 2, 100)  # 50거래 = 100점
        
        # 드로우다운 점수 (20% 가중치, 역방향)
        dd_score = max(0, 100 - max_drawdown * 1000)  # 드로우다운이 낮을수록 높은 점수
        
        return cons_score * 0.5 + trade_score * 0.3 + dd_score * 0.2
    
    def _calculate_composite_score(self, risk_score: float, reward_score: float, stability_score: float) -> float:
        """종합 점수 계산"""
        # 리스크는 역방향 점수 (100 - risk_score)
        risk_adjusted = 100 - risk_score
        
        # 가중 평균: 수익성 40%, 안정성 35%, 리스크 25%
        composite = reward_score * 0.4 + stability_score * 0.35 + risk_adjusted * 0.25
        
        return min(max(composite, 0), 100)
    
    def _assign_grade(self, composite_score: float) -> str:
        """종합 점수를 등급으로 변환"""
        for grade, threshold in self.grade_thresholds.items():
            if composite_score >= threshold:
                return grade
        return 'F'
    
    def save_scores(self, strategy_scores: Dict[str, StrategyScore], 
                   timestamp: Optional[datetime] = None) -> str:
        """전략 점수를 JSON 파일로 저장"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # 파일명 생성
        filename = f"strategy_scores_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # JSON 직렬화 가능한 형태로 변환
        scores_dict = {}
        for strategy_name, score in strategy_scores.items():
            scores_dict[strategy_name] = {
                'strategy_name': score.strategy_name,
                'total_return': score.total_return,
                'roi': score.roi,
                'annualized_return': score.annualized_return,
                'sharpe_ratio': score.sharpe_ratio,
                'sortino_ratio': score.sortino_ratio,
                'max_drawdown': score.max_drawdown,
                'win_rate': score.win_rate,
                'profit_factor': score.profit_factor,
                'total_trades': score.total_trades,
                'avg_win': score.avg_win,
                'avg_loss': score.avg_loss,
                'consistency_score': score.consistency_score,
                'composite_score': score.composite_score,
                'confidence_level': score.confidence_level,
                'volatility': score.volatility,
                'calmar_ratio': score.calmar_ratio,
                'expectancy': score.expectancy,
                'avg_holding_time_hours': score.avg_holding_time_hours,
                'risk_score': score.risk_score,
                'reward_score': score.reward_score,
                'stability_score': score.stability_score,
                'overall_grade': score.overall_grade
            }
        
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'total_strategies': len(strategy_scores),
                'generator_version': '1.0.0'
            },
            'scores': scores_dict,
            'summary': self._generate_summary(strategy_scores)
        }
        
        # JSON 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"전략 점수 파일 저장 완료: {filepath}")
        return filepath
    
    def _generate_summary(self, strategy_scores: Dict[str, StrategyScore]) -> Dict[str, Any]:
        """전략 점수 요약 생성"""
        if not strategy_scores:
            return {}
        
        scores_list = list(strategy_scores.values())
        
        # 등급별 분포
        grade_distribution = {}
        for score in scores_list:
            grade = score.overall_grade
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        # 최고/최저 전략
        best_strategy = max(scores_list, key=lambda x: x.composite_score)
        worst_strategy = min(scores_list, key=lambda x: x.composite_score)
        
        # 평균 메트릭
        avg_roi = np.mean([s.roi for s in scores_list])
        avg_sharpe = np.mean([s.sharpe_ratio for s in scores_list])
        avg_max_dd = np.mean([s.max_drawdown for s in scores_list])
        avg_win_rate = np.mean([s.win_rate for s in scores_list])
        
        return {
            'grade_distribution': grade_distribution,
            'best_strategy': {
                'name': best_strategy.strategy_name,
                'grade': best_strategy.overall_grade,
                'composite_score': best_strategy.composite_score,
                'roi': best_strategy.roi
            },
            'worst_strategy': {
                'name': worst_strategy.strategy_name,
                'grade': worst_strategy.overall_grade,
                'composite_score': worst_strategy.composite_score,
                'roi': worst_strategy.roi
            },
            'averages': {
                'roi': avg_roi,
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': avg_max_dd,
                'win_rate': avg_win_rate
            }
        }


def generate_strategy_scores_from_backtest(backtest_results: Dict[str, Any], 
                                         output_dir: str = "reports/strategy_scores") -> str:
    """백테스트 결과로부터 전략 점수 생성 (편의 함수)"""
    generator = StrategyScoresGenerator(output_dir)
    scores = generator.generate_scores(backtest_results)
    return generator.save_scores(scores)


if __name__ == "__main__":
    # 테스트용 더미 데이터
    dummy_results = {
        'metrics': {
            'all_metrics': {
                'TestStrategy': type('obj', (object,), {
                    'total_return': 0.15,
                    'roi': 0.15,
                    'annualized_return': 0.12,
                    'sharpe_ratio': 1.2,
                    'sortino_ratio': 1.5,
                    'max_drawdown': 0.08,
                    'win_rate': 0.55,
                    'profit_factor': 1.8,
                    'total_trades': 25,
                    'avg_win': 100.0,
                    'avg_loss': 80.0,
                    'consistency_score': 0.7,
                    'composite_score': 0.75,
                    'confidence_level': 0.8,
                    'volatility': 0.15,
                    'calmar_ratio': 1.5,
                    'expectancy': 15.0,
                    'avg_holding_time_hours': 24.0
                })()
            }
        }
    }
    
    filepath = generate_strategy_scores_from_backtest(dummy_results)
    print(f"테스트 파일 생성: {filepath}")