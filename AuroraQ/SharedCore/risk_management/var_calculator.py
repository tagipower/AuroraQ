#!/usr/bin/env python3
"""
VaR (Value at Risk) 계산기
다양한 방법론을 통한 VaR 및 CVaR 계산
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class VaRCalculator:
    """VaR 및 CVaR 계산기"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.methods = [
            'historical',
            'parametric', 
            'monte_carlo',
            'cornish_fisher'
        ]
    
    def calculate_var(self, 
                     returns: np.ndarray,
                     method: str = 'historical',
                     confidence_level: float = 0.95,
                     portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        VaR 계산
        
        Args:
            returns: 수익률 배열
            method: 계산 방법 ('historical', 'parametric', 'monte_carlo', 'cornish_fisher')
            confidence_level: 신뢰수준
            portfolio_value: 포트폴리오 가치
            
        Returns:
            VaR 계산 결과
        """
        if len(returns) == 0:
            return {'var': 0.0, 'var_pct': 0.0, 'cvar': 0.0, 'cvar_pct': 0.0}
        
        # 결측값 제거
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {'var': 0.0, 'var_pct': 0.0, 'cvar': 0.0, 'cvar_pct': 0.0}
        
        if method == 'historical':
            var_pct, cvar_pct = self._historical_var(returns, confidence_level)
        elif method == 'parametric':
            var_pct, cvar_pct = self._parametric_var(returns, confidence_level)
        elif method == 'monte_carlo':
            var_pct, cvar_pct = self._monte_carlo_var(returns, confidence_level)
        elif method == 'cornish_fisher':
            var_pct, cvar_pct = self._cornish_fisher_var(returns, confidence_level)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 절대값으로 변환
        var_abs = abs(var_pct) * portfolio_value
        cvar_abs = abs(cvar_pct) * portfolio_value
        
        return {
            'var': var_abs,
            'var_pct': abs(var_pct),
            'cvar': cvar_abs,
            'cvar_pct': abs(cvar_pct),
            'method': method,
            'confidence_level': confidence_level,
            'observations': len(returns)
        }
    
    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """히스토리컬 VaR 계산"""
        # VaR: 분위수
        var_pct = np.percentile(returns, (1 - confidence_level) * 100)
        
        # CVaR: VaR 이하의 평균
        tail_returns = returns[returns <= var_pct]
        cvar_pct = np.mean(tail_returns) if len(tail_returns) > 0 else var_pct
        
        return var_pct, cvar_pct
    
    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """파라메트릭 VaR 계산 (정규분포 가정)"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 정규분포의 분위수
        z_score = stats.norm.ppf(1 - confidence_level)
        var_pct = mean_return + z_score * std_return
        
        # CVaR 계산 (정규분포)
        # CVaR = μ + σ * φ(z) / (1-α), where φ는 표준정규분포의 PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_pct = mean_return - std_return * phi_z / (1 - confidence_level)
        
        return var_pct, cvar_pct
    
    def _monte_carlo_var(self, 
                        returns: np.ndarray, 
                        confidence_level: float,
                        n_simulations: int = 10000) -> Tuple[float, float]:
        """몬테카를로 시뮬레이션 VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 정규분포 시뮬레이션 (기본)
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # VaR 및 CVaR 계산
        var_pct = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        tail_returns = simulated_returns[simulated_returns <= var_pct]
        cvar_pct = np.mean(tail_returns) if len(tail_returns) > 0 else var_pct
        
        return var_pct, cvar_pct
    
    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Cornish-Fisher 확장을 사용한 VaR (비정규분포 고려)"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 왜도와 첨도 계산
        skewness = stats.skew(returns)
        kurtosis_excess = stats.kurtosis(returns, fisher=True)  # 초과 첨도
        
        # 정규분포 분위수
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher 수정
        cf_z = (z + 
                (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis_excess / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)
        
        var_pct = mean_return + cf_z * std_return
        
        # CVaR는 히스토리컬 방법으로 근사
        tail_returns = returns[returns <= var_pct]
        cvar_pct = np.mean(tail_returns) if len(tail_returns) > 0 else var_pct
        
        return var_pct, cvar_pct
    
    def calculate_component_var(self,
                               returns_matrix: np.ndarray,
                               weights: np.ndarray,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """컴포넌트 VaR 계산 (포트폴리오 내 각 자산의 VaR 기여도)"""
        
        if returns_matrix.shape[0] == 0 or returns_matrix.shape[1] != len(weights):
            return {}
        
        # 포트폴리오 수익률
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # 포트폴리오 VaR
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level=confidence_level)
        
        # 각 자산의 marginal VaR 계산
        marginal_vars = []
        epsilon = 1e-6
        
        for i in range(len(weights)):
            # 가중치를 미소 증가
            weights_up = weights.copy()
            weights_up[i] += epsilon
            weights_up = weights_up / np.sum(weights_up)  # 정규화
            
            portfolio_returns_up = np.dot(returns_matrix, weights_up)
            var_up = self.calculate_var(portfolio_returns_up, confidence_level=confidence_level)['var_pct']
            
            # Marginal VaR
            marginal_var = (var_up - portfolio_var['var_pct']) / epsilon
            marginal_vars.append(marginal_var)
        
        # Component VaR = Weight * Marginal VaR
        component_vars = weights * np.array(marginal_vars)
        
        return {
            'portfolio_var': portfolio_var,
            'marginal_vars': marginal_vars,
            'component_vars': component_vars.tolist(),
            'component_var_pct': component_vars / np.sum(np.abs(component_vars)) if np.sum(np.abs(component_vars)) > 0 else component_vars
        }
    
    def calculate_incremental_var(self,
                                 base_returns: np.ndarray,
                                 new_asset_returns: np.ndarray,
                                 base_weight: float = 0.95,
                                 confidence_level: float = 0.95) -> Dict[str, float]:
        """증분 VaR 계산 (새 자산 추가 시 VaR 변화)"""
        
        # 기존 포트폴리오 VaR
        base_var = self.calculate_var(base_returns, confidence_level=confidence_level)
        
        # 새 포트폴리오 수익률 (가중평균)
        new_weight = 1 - base_weight
        combined_returns = base_weight * base_returns + new_weight * new_asset_returns
        
        # 새 포트폴리오 VaR
        new_var = self.calculate_var(combined_returns, confidence_level=confidence_level)
        
        # 증분 VaR
        incremental_var = new_var['var_pct'] - base_var['var_pct']
        incremental_cvar = new_var['cvar_pct'] - base_var['cvar_pct']
        
        return {
            'base_var': base_var['var_pct'],
            'new_var': new_var['var_pct'],
            'incremental_var': incremental_var,
            'incremental_cvar': incremental_cvar,
            'var_change_pct': incremental_var / abs(base_var['var_pct']) if base_var['var_pct'] != 0 else 0
        }
    
    def backtest_var(self,
                    returns: np.ndarray,
                    var_estimates: np.ndarray,
                    confidence_level: float = 0.95) -> Dict[str, Any]:
        """VaR 백테스트 (Kupiec test)"""
        
        if len(returns) != len(var_estimates):
            raise ValueError("Returns and VaR estimates must have same length")
        
        # VaR 위반 횟수
        violations = np.sum(returns < -var_estimates)
        total_observations = len(returns)
        
        # 예상 위반 횟수
        expected_violations = total_observations * (1 - confidence_level)
        
        # 실제 위반률
        violation_rate = violations / total_observations
        
        # Kupiec 통계량 (우도비 검정)
        if violations == 0 or violations == total_observations:
            kupiec_stat = 0
            p_value = 1.0
        else:
            lr = 2 * (violations * np.log(violation_rate / (1 - confidence_level)) +
                     (total_observations - violations) * np.log((1 - violation_rate) / confidence_level))
            kupiec_stat = lr
            p_value = 1 - stats.chi2.cdf(lr, df=1)
        
        # 판정 (p-value > 0.05이면 VaR 모델이 적절)
        is_adequate = p_value > 0.05
        
        return {
            'violations': violations,
            'total_observations': total_observations,
            'violation_rate': violation_rate,
            'expected_violation_rate': 1 - confidence_level,
            'kupiec_statistic': kupiec_stat,
            'p_value': p_value,
            'is_adequate': is_adequate,
            'excess_violations': violations - expected_violations
        }
    
    def calculate_rolling_var(self,
                             returns: np.ndarray,
                             window_size: int = 252,
                             confidence_level: float = 0.95,
                             method: str = 'historical') -> pd.Series:
        """롤링 VaR 계산"""
        
        if len(returns) < window_size:
            return pd.Series([])
        
        rolling_vars = []
        
        for i in range(window_size, len(returns) + 1):
            window_returns = returns[i-window_size:i]
            var_result = self.calculate_var(window_returns, method=method, confidence_level=confidence_level)
            rolling_vars.append(var_result['var_pct'])
        
        return pd.Series(rolling_vars)
    
    def stress_test_var(self,
                       returns: np.ndarray,
                       stress_scenarios: List[float],
                       confidence_level: float = 0.95) -> Dict[str, Any]:
        """스트레스 테스트"""
        
        base_var = self.calculate_var(returns, confidence_level=confidence_level)
        stress_results = {}
        
        for scenario_pct in stress_scenarios:
            # 스트레스 시나리오 적용 (모든 수익률에 충격 추가)
            stressed_returns = returns + scenario_pct
            stressed_var = self.calculate_var(stressed_returns, confidence_level=confidence_level)
            
            stress_results[f"{scenario_pct:.1%}"] = {
                'stressed_var_pct': stressed_var['var_pct'],
                'var_change': stressed_var['var_pct'] - base_var['var_pct'],
                'var_multiplier': stressed_var['var_pct'] / base_var['var_pct'] if base_var['var_pct'] != 0 else float('inf')
            }
        
        return {
            'base_var': base_var,
            'stress_results': stress_results
        }
    
    def calculate_diversification_benefit(self,
                                        individual_vars: List[float],
                                        portfolio_var: float) -> Dict[str, float]:
        """다각화 효과 계산"""
        
        # 개별 VaR의 합
        sum_individual_vars = sum(individual_vars)
        
        # 다각화 효과
        diversification_benefit = sum_individual_vars - portfolio_var
        diversification_ratio = diversification_benefit / sum_individual_vars if sum_individual_vars > 0 else 0
        
        return {
            'sum_individual_vars': sum_individual_vars,
            'portfolio_var': portfolio_var,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': diversification_ratio,
            'diversification_pct': diversification_ratio * 100
        }
    
    def optimal_var_allocation(self,
                              returns_matrix: np.ndarray,
                              target_var: float,
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """목표 VaR에 맞는 최적 자산 배분"""
        
        n_assets = returns_matrix.shape[1]
        
        def objective(weights):
            """목적함수: 목표 VaR와의 차이"""
            portfolio_returns = np.dot(returns_matrix, weights)
            portfolio_var = self.calculate_var(portfolio_returns, confidence_level=confidence_level)['var_pct']
            return (portfolio_var - target_var) ** 2
        
        # 제약조건: 가중치 합 = 1, 가중치 >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # 초기 추정치 (균등 배분)
        initial_weights = np.ones(n_assets) / n_assets
        
        # 최적화 실행
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_returns = np.dot(returns_matrix, optimal_weights)
                achieved_var = self.calculate_var(portfolio_returns, confidence_level=confidence_level)
                
                return {
                    'success': True,
                    'optimal_weights': optimal_weights.tolist(),
                    'target_var': target_var,
                    'achieved_var': achieved_var['var_pct'],
                    'optimization_error': abs(achieved_var['var_pct'] - target_var)
                }
            else:
                return {'success': False, 'message': 'Optimization failed'}
                
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def calculate_multiple_var(self,
                              returns: np.ndarray,
                              portfolio_value: float = 1.0) -> Dict[str, Any]:
        """모든 방법론으로 VaR 계산"""
        
        results = {}
        
        for method in self.methods:
            for confidence_level in self.confidence_levels:
                try:
                    var_result = self.calculate_var(
                        returns, method=method, 
                        confidence_level=confidence_level,
                        portfolio_value=portfolio_value
                    )
                    
                    key = f"{method}_{int(confidence_level*100)}"
                    results[key] = var_result
                    
                except Exception as e:
                    results[f"{method}_{int(confidence_level*100)}"] = {
                        'error': str(e),
                        'var': 0.0,
                        'var_pct': 0.0,
                        'cvar': 0.0, 
                        'cvar_pct': 0.0
                    }
        
        # 방법론별 결과 비교
        comparison = self._compare_var_methods(results)
        results['comparison'] = comparison
        
        return results
    
    def _compare_var_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """VaR 방법론 비교"""
        
        var_95_values = []
        var_99_values = []
        
        for key, result in results.items():
            if '95' in key and 'error' not in result:
                var_95_values.append(result['var_pct'])
            elif '99' in key and 'error' not in result:
                var_99_values.append(result['var_pct'])
        
        comparison = {}
        
        if var_95_values:
            comparison['var_95'] = {
                'min': min(var_95_values),
                'max': max(var_95_values),
                'mean': np.mean(var_95_values),
                'std': np.std(var_95_values),
                'range': max(var_95_values) - min(var_95_values)
            }
        
        if var_99_values:
            comparison['var_99'] = {
                'min': min(var_99_values),
                'max': max(var_99_values),
                'mean': np.mean(var_99_values),
                'std': np.std(var_99_values),
                'range': max(var_99_values) - min(var_99_values)
            }
        
        return comparison