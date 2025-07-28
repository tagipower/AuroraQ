"""
포트폴리오 최적화 엔진 - 리스크 패리티 기반
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import cvxpy as cp
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """포트폴리오 최적화 제약조건"""
    min_weight: float = 0.0           # 최소 비중
    max_weight: float = 0.4           # 최대 비중 (40%)
    max_turnover: float = 0.5         # 최대 회전율 (50%)
    target_volatility: float = 0.15   # 목표 변동성 (연 15%)
    min_assets: int = 3               # 최소 자산 수
    
    
class PortfolioOptimizer:
    """
    효율적 포트폴리오 최적화
    - 리스크 패리티
    - 평균-분산 최적화
    - 거래비용 고려
    """
    
    def __init__(self, constraints: OptimizationConstraints = None):
        self.constraints = constraints or OptimizationConstraints()
        self.last_weights = None
        
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        optimization_method: str = 'risk_parity'
    ) -> pd.Series:
        """
        포트폴리오 최적화 실행
        
        Args:
            expected_returns: 예상 수익률
            covariance_matrix: 공분산 행렬
            current_weights: 현재 비중 (리밸런싱용)
            optimization_method: 'risk_parity', 'mean_variance', 'min_variance'
            
        Returns:
            최적 비중
        """
        n_assets = len(expected_returns)
        
        if optimization_method == 'risk_parity':
            weights = self._risk_parity_optimization(covariance_matrix)
        elif optimization_method == 'mean_variance':
            weights = self._mean_variance_optimization(
                expected_returns, covariance_matrix
            )
        elif optimization_method == 'min_variance':
            weights = self._min_variance_optimization(covariance_matrix)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
            
        # 회전율 제약 적용
        if current_weights is not None:
            weights = self._apply_turnover_constraint(weights, current_weights)
            
        # 최소 자산 수 제약
        weights = self._apply_min_assets_constraint(weights)
        
        self.last_weights = weights
        return weights
    
    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """리스크 패리티 최적화"""
        n_assets = len(cov_matrix)
        
        # 초기 추정치 (동일 비중)
        x0 = np.ones(n_assets) / n_assets
        
        # 목적함수: 리스크 기여도의 분산 최소화
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # 리스크 기여도가 모두 같아지도록
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 합 = 1
            {'type': 'ineq', 'fun': lambda x: x - self.constraints.min_weight},  # 최소 비중
            {'type': 'ineq', 'fun': lambda x: self.constraints.max_weight - x}   # 최대 비중
        ]
        
        # 최적화 실행
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            # Fallback to equal weight
            return pd.Series(x0, index=cov_matrix.index)
            
        weights = pd.Series(result.x, index=cov_matrix.index)
        return weights / weights.sum()  # 정규화
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """평균-분산 최적화 (cvxpy 사용)"""
        n_assets = len(expected_returns)
        
        # 변수
        weights = cp.Variable(n_assets)
        
        # 목적함수: 샤프 비율 최대화 (리스크 조정 수익)
        portfolio_return = expected_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        # 목표 변동성 제약을 위한 리스크 예산
        risk_budget = self.constraints.target_volatility ** 2
        
        # 목적함수
        objective = cp.Maximize(
            portfolio_return - 0.5 * portfolio_variance  # 리스크 회피 계수 0.5
        )
        
        # 제약조건
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.constraints.min_weight,
            weights <= self.constraints.max_weight,
            portfolio_variance <= risk_budget
        ]
        
        # 문제 정의 및 해결
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status != cp.OPTIMAL:
                logger.warning(f"Mean-variance optimization not optimal: {problem.status}")
                # Fallback
                return self._equal_weight_portfolio(expected_returns.index)
                
            return pd.Series(weights.value, index=expected_returns.index)
            
        except Exception as e:
            logger.error(f"Mean-variance optimization error: {e}")
            return self._equal_weight_portfolio(expected_returns.index)
    
    def _min_variance_optimization(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """최소 분산 포트폴리오"""
        n_assets = len(cov_matrix)
        
        # cvxpy 변수
        weights = cp.Variable(n_assets)
        
        # 목적함수: 분산 최소화
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # 제약조건
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.constraints.min_weight,
            weights <= self.constraints.max_weight
        ]
        
        # 문제 해결
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status != cp.OPTIMAL:
                logger.warning("Min variance optimization not optimal")
                return self._equal_weight_portfolio(cov_matrix.index)
                
            return pd.Series(weights.value, index=cov_matrix.index)
            
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}")
            return self._equal_weight_portfolio(cov_matrix.index)
    
    def _apply_turnover_constraint(
        self,
        new_weights: pd.Series,
        current_weights: pd.Series
    ) -> pd.Series:
        """회전율 제약 적용"""
        turnover = np.abs(new_weights - current_weights).sum()
        
        if turnover > self.constraints.max_turnover:
            # 회전율 제한
            scale = self.constraints.max_turnover / turnover
            adjusted_weights = current_weights + scale * (new_weights - current_weights)
            
            # 재정규화
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            
            logger.info(f"Turnover reduced from {turnover:.2%} to {self.constraints.max_turnover:.2%}")
            return adjusted_weights
            
        return new_weights
    
    def _apply_min_assets_constraint(self, weights: pd.Series) -> pd.Series:
        """최소 자산 수 제약"""
        if (weights > 0.01).sum() >= self.constraints.min_assets:
            return weights
            
        # 상위 N개 자산만 선택
        top_assets = weights.nlargest(self.constraints.min_assets).index
        
        # 나머지는 0으로
        adjusted_weights = weights.copy()
        adjusted_weights[~adjusted_weights.index.isin(top_assets)] = 0
        
        # 재정규화
        return adjusted_weights / adjusted_weights.sum()
    
    def _equal_weight_portfolio(self, assets: pd.Index) -> pd.Series:
        """동일 비중 포트폴리오 (fallback)"""
        n_assets = min(len(assets), self.constraints.min_assets)
        weights = pd.Series(0.0, index=assets)
        weights.iloc[:n_assets] = 1.0 / n_assets
        return weights
    
    def calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        # 포트폴리오 수익률
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # 연환산 지표
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 샤프 비율
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 최대 낙폭
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'effective_assets': (weights > 0.01).sum()
        }