#!/usr/bin/env python3
"""
포트폴리오 리스크 분석기
포트폴리오 수준에서의 종합적인 리스크 분석 및 최적화
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
import logging

from .var_calculator import VaRCalculator
from .risk_models import RiskMetrics, RiskConfig, PortfolioSnapshot, RiskScenario

warnings.filterwarnings('ignore')


class PortfolioRiskAnalyzer:
    """포트폴리오 리스크 분석기"""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.var_calculator = VaRCalculator(confidence_levels=self.config.var_confidence_levels)
        self.logger = logging.getLogger(__name__)
        
        # 시나리오 관리
        self.scenarios: List[RiskScenario] = []
        self._initialize_default_scenarios()
    
    def _initialize_default_scenarios(self):
        """기본 리스크 시나리오 초기화"""
        
        # 시장 충격 시나리오
        market_crash = RiskScenario(
            name="Market Crash",
            description="전체 시장 20% 하락",
            probability=0.05
        )
        market_crash.add_market_shock("ALL", -0.20)
        market_crash.add_volatility_change("ALL", 2.0)
        self.scenarios.append(market_crash)
        
        # 중기 조정 시나리오
        market_correction = RiskScenario(
            name="Market Correction",
            description="시장 10% 조정",
            probability=0.15
        )
        market_correction.add_market_shock("ALL", -0.10)
        market_correction.add_volatility_change("ALL", 1.5)
        self.scenarios.append(market_correction)
        
        # 고변동성 시나리오
        high_volatility = RiskScenario(
            name="High Volatility",
            description="변동성 급등",
            probability=0.20
        )
        high_volatility.add_volatility_change("ALL", 3.0)
        self.scenarios.append(high_volatility)
    
    def analyze_portfolio_risk(self, 
                             portfolio_snapshot: PortfolioSnapshot,
                             price_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """포트폴리오 종합 리스크 분석"""
        
        analysis_results = {
            'timestamp': datetime.now(),
            'portfolio_summary': self._get_portfolio_summary(portfolio_snapshot),
            'var_analysis': {},
            'concentration_analysis': {},
            'correlation_analysis': {},
            'scenario_analysis': {},
            'optimization_recommendations': {},
            'risk_attribution': {}
        }
        
        # 1. VaR 분석
        if len(portfolio_snapshot.returns) >= 30:
            analysis_results['var_analysis'] = self._analyze_var_comprehensive(
                portfolio_snapshot, price_history
            )
        
        # 2. 집중도 분석
        analysis_results['concentration_analysis'] = self._analyze_concentration(portfolio_snapshot)
        
        # 3. 상관관계 분석
        if price_history is not None and len(portfolio_snapshot.positions) > 1:
            analysis_results['correlation_analysis'] = self._analyze_correlations(
                portfolio_snapshot, price_history
            )
        
        # 4. 시나리오 분석
        analysis_results['scenario_analysis'] = self._analyze_scenarios(portfolio_snapshot)
        
        # 5. 포트폴리오 최적화 권고
        analysis_results['optimization_recommendations'] = self._generate_optimization_recommendations(
            portfolio_snapshot, analysis_results
        )
        
        # 6. 리스크 기여도 분석
        if len(portfolio_snapshot.returns) >= 30:
            analysis_results['risk_attribution'] = self._analyze_risk_attribution(
                portfolio_snapshot, price_history
            )
        
        return analysis_results
    
    def _get_portfolio_summary(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """포트폴리오 요약 정보"""
        
        weights = portfolio_snapshot.get_position_weights()
        
        summary = {
            'total_equity': portfolio_snapshot.total_equity,
            'cash_ratio': portfolio_snapshot.cash / portfolio_snapshot.total_equity if portfolio_snapshot.total_equity > 0 else 0,
            'position_count': len(portfolio_snapshot.positions),
            'largest_position': max(weights.values()) if weights else 0,
            'position_weights': weights,
            'total_market_value': sum(
                pos.get('market_value', 0) for pos in portfolio_snapshot.positions.values()
            )
        }
        
        if len(portfolio_snapshot.returns) > 0:
            returns = np.array(portfolio_snapshot.returns)
            summary.update({
                'total_return': (portfolio_snapshot.total_equity / 100000 - 1) if portfolio_snapshot.total_equity > 0 else 0,  # 가정: 초기 자본 100K
                'volatility': np.std(returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                'max_return': np.max(returns),
                'min_return': np.min(returns),
                'positive_days': np.sum(returns > 0) / len(returns)
            })
        
        return summary
    
    def _analyze_var_comprehensive(self, 
                                 portfolio_snapshot: PortfolioSnapshot,
                                 price_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """종합적인 VaR 분석"""
        
        returns = portfolio_snapshot.get_portfolio_returns(self.config.var_lookback_period)
        
        if len(returns) < 30:
            return {}
        
        # 모든 방법론으로 VaR 계산
        var_results = self.var_calculator.calculate_multiple_var(
            returns, portfolio_snapshot.total_equity
        )
        
        # 백테스트 (과거 VaR 예측 성능)
        if len(returns) >= 60:
            backtest_results = self._backtest_var_performance(returns)
            var_results['backtest'] = backtest_results
        
        # 롤링 VaR
        if len(returns) >= 100:
            rolling_var = self.var_calculator.calculate_rolling_var(
                returns, window_size=60
            )
            var_results['rolling_var'] = {
                'current': rolling_var.iloc[-1] if len(rolling_var) > 0 else 0,
                'trend': rolling_var.tolist()[-30:],  # 최근 30일 트렌드
                'volatility': rolling_var.std()
            }
        
        # 컴포넌트 VaR (포지션별 기여도)
        if price_history is not None and len(portfolio_snapshot.positions) > 1:
            component_var = self._calculate_component_var(portfolio_snapshot, price_history, returns)
            var_results['component_var'] = component_var
        
        return var_results
    
    def _backtest_var_performance(self, returns: np.ndarray) -> Dict[str, Any]:
        """VaR 모델 백테스트"""
        
        window_size = 60
        var_estimates = []
        actual_returns = []
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            var_95 = self.var_calculator.calculate_var(window_returns, confidence_level=0.95)['var_pct']
            var_estimates.append(var_95)
            actual_returns.append(returns[i])
        
        if len(var_estimates) < 10:
            return {}
        
        var_estimates = np.array(var_estimates)
        actual_returns = np.array(actual_returns)
        
        # 백테스트 수행
        backtest_result = self.var_calculator.backtest_var(
            actual_returns, var_estimates, confidence_level=0.95
        )
        
        return backtest_result
    
    def _calculate_component_var(self, 
                               portfolio_snapshot: PortfolioSnapshot,
                               price_history: pd.DataFrame,
                               portfolio_returns: np.ndarray) -> Dict[str, Any]:
        """컴포넌트 VaR 계산"""
        
        symbols = list(portfolio_snapshot.positions.keys())
        weights = portfolio_snapshot.get_position_weights()
        
        # 각 종목의 수익률 매트릭스 구성
        returns_matrix = []
        for symbol in symbols:
            if symbol in price_history.columns:
                symbol_prices = price_history[symbol].dropna()
                symbol_returns = symbol_prices.pct_change().dropna()
                
                # 포트폴리오 수익률과 같은 길이로 맞춤
                min_length = min(len(symbol_returns), len(portfolio_returns))
                returns_matrix.append(symbol_returns.iloc[-min_length:].values)
        
        if len(returns_matrix) < 2:
            return {}
        
        returns_matrix = np.array(returns_matrix).T
        weights_array = np.array([weights.get(symbol, 0) for symbol in symbols])
        
        # 컴포넌트 VaR 계산
        component_var_result = self.var_calculator.calculate_component_var(
            returns_matrix, weights_array
        )
        
        if component_var_result:
            # 결과를 심볼과 매핑
            component_var_by_symbol = {}
            for i, symbol in enumerate(symbols):
                if i < len(component_var_result.get('component_vars', [])):
                    component_var_by_symbol[symbol] = {
                        'component_var': component_var_result['component_vars'][i],
                        'marginal_var': component_var_result['marginal_vars'][i],
                        'weight': weights_array[i]
                    }
            
            component_var_result['by_symbol'] = component_var_by_symbol
        
        return component_var_result
    
    def _analyze_concentration(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """집중도 분석"""
        
        weights = portfolio_snapshot.get_position_weights()
        
        if not weights:
            return {}
        
        # 허핀달 지수
        herfindahl_index = sum(w**2 for w in weights.values())
        
        # 집중도 지표들
        sorted_weights = sorted(weights.values(), reverse=True)
        
        concentration_metrics = {
            'herfindahl_index': herfindahl_index,
            'top_1_concentration': sorted_weights[0] if len(sorted_weights) > 0 else 0,
            'top_3_concentration': sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights),
            'top_5_concentration': sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights),
            'effective_positions': 1 / herfindahl_index if herfindahl_index > 0 else 0,
            'position_weights': weights
        }
        
        # 집중도 위험 평가
        risk_level = "Low"
        if herfindahl_index > 0.4:
            risk_level = "High"
        elif herfindahl_index > 0.2:
            risk_level = "Medium"
        
        concentration_metrics['risk_level'] = risk_level
        
        # 권고사항
        recommendations = []
        if sorted_weights[0] > 0.3:
            recommendations.append(f"가장 큰 포지션({sorted_weights[0]:.1%})의 비중을 줄이는 것을 고려하세요")
        
        if len(weights) < 5:
            recommendations.append("포트폴리오 다각화를 위해 추가 종목을 고려하세요")
        
        concentration_metrics['recommendations'] = recommendations
        
        return concentration_metrics
    
    def _analyze_correlations(self, 
                            portfolio_snapshot: PortfolioSnapshot,
                            price_history: pd.DataFrame) -> Dict[str, Any]:
        """상관관계 분석"""
        
        symbols = list(portfolio_snapshot.positions.keys())
        
        if len(symbols) < 2:
            return {}
        
        # 각 종목의 수익률 계산
        returns_data = {}
        for symbol in symbols:
            if symbol in price_history.columns:
                prices = price_history[symbol].dropna()
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return {}
        
        # 수익률 DataFrame 생성
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 30:
            return {}
        
        # 상관관계 매트릭스
        correlation_matrix = returns_df.corr()
        
        # 상관관계 통계
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                correlations.append({
                    'asset1': correlation_matrix.columns[i],
                    'asset2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
        
        correlation_values = [c['correlation'] for c in correlations]
        
        analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'avg_correlation': np.mean(correlation_values) if correlation_values else 0,
            'max_correlation': np.max(correlation_values) if correlation_values else 0,
            'min_correlation': np.min(correlation_values) if correlation_values else 0,
            'high_correlation_pairs': [
                c for c in correlations if c['correlation'] > 0.7
            ],
            'negative_correlation_pairs': [
                c for c in correlations if c['correlation'] < -0.3
            ]
        }
        
        # 다각화 효과 분석
        weights = portfolio_snapshot.get_position_weights()
        portfolio_variance = self._calculate_portfolio_variance(returns_df, weights)
        individual_variances = [returns_df[symbol].var() for symbol in symbols]
        weighted_individual_variance = sum(
            weights.get(symbol, 0)**2 * var for symbol, var in zip(symbols, individual_variances)
        )
        
        diversification_ratio = 1 - (portfolio_variance / weighted_individual_variance) if weighted_individual_variance > 0 else 0
        
        analysis['diversification_analysis'] = {
            'portfolio_variance': portfolio_variance,
            'diversification_ratio': diversification_ratio,
            'diversification_benefit': diversification_ratio * 100
        }
        
        return analysis
    
    def _calculate_portfolio_variance(self, returns_df: pd.DataFrame, weights: Dict[str, float]) -> float:
        """포트폴리오 분산 계산"""
        
        symbols = list(returns_df.columns)
        weight_vector = np.array([weights.get(symbol, 0) for symbol in symbols])
        
        cov_matrix = returns_df.cov().values
        portfolio_variance = np.dot(weight_vector, np.dot(cov_matrix, weight_vector))
        
        return portfolio_variance
    
    def _analyze_scenarios(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """시나리오 분석"""
        
        scenario_results = {}
        
        for scenario in self.scenarios:
            try:
                # 시나리오를 포트폴리오에 적용
                scenario_impact = scenario.apply_to_portfolio(portfolio_snapshot)
                
                # 총 영향 계산
                total_impact = sum(scenario_impact.values())
                impact_percentage = total_impact / portfolio_snapshot.total_equity if portfolio_snapshot.total_equity > 0 else 0
                
                scenario_results[scenario.name] = {
                    'description': scenario.description,
                    'probability': scenario.probability,
                    'total_impact': total_impact,
                    'impact_percentage': impact_percentage,
                    'by_position': scenario_impact,
                    'expected_impact': total_impact * scenario.probability
                }
                
            except Exception as e:
                self.logger.error(f"Scenario analysis failed for {scenario.name}: {e}")
                continue
        
        # 시나리오 요약
        if scenario_results:
            expected_losses = [result['expected_impact'] for result in scenario_results.values()]
            worst_case_impact = min([result['impact_percentage'] for result in scenario_results.values()])
            
            scenario_summary = {
                'expected_loss': sum(expected_losses),
                'worst_case_scenario': worst_case_impact,
                'scenarios_count': len(scenario_results),
                'detailed_results': scenario_results
            }
        else:
            scenario_summary = {}
        
        return scenario_summary
    
    def _analyze_risk_attribution(self, 
                                portfolio_snapshot: PortfolioSnapshot,
                                price_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """리스크 기여도 분석"""
        
        weights = portfolio_snapshot.get_position_weights()
        returns = portfolio_snapshot.get_portfolio_returns()
        
        if len(returns) < 30 or len(weights) < 2:
            return {}
        
        attribution = {}
        
        # 1. 변동성 기여도
        if price_history is not None:
            attribution['volatility_attribution'] = self._calculate_volatility_attribution(
                portfolio_snapshot, price_history
            )
        
        # 2. VaR 기여도 (이미 component_var에서 계산됨)
        
        # 3. 섹터별 기여도 (간단한 예시)
        sector_attribution = {}
        for symbol, weight in weights.items():
            sector = self._get_sector_from_symbol(symbol)
            sector_attribution[sector] = sector_attribution.get(sector, 0) + weight
        
        attribution['sector_attribution'] = sector_attribution
        
        # 4. 성과 기여도
        if len(returns) > 0:
            portfolio_return = returns[-1] if len(returns) > 0 else 0
            
            # 간단한 성과 기여도 (실제로는 개별 종목 수익률 필요)
            performance_attribution = {}
            for symbol, weight in weights.items():
                # 예시: 각 포지션이 포트폴리오 수익률에 기여한다고 가정
                contribution = weight * portfolio_return
                performance_attribution[symbol] = contribution
            
            attribution['performance_attribution'] = performance_attribution
        
        return attribution
    
    def _calculate_volatility_attribution(self, 
                                        portfolio_snapshot: PortfolioSnapshot,
                                        price_history: pd.DataFrame) -> Dict[str, Any]:
        """변동성 기여도 계산"""
        
        symbols = list(portfolio_snapshot.positions.keys())
        weights = portfolio_snapshot.get_position_weights()
        
        volatility_attribution = {}
        
        for symbol in symbols:
            if symbol in price_history.columns:
                prices = price_history[symbol].dropna()
                returns = prices.pct_change().dropna()
                
                if len(returns) >= 30:
                    volatility = returns.std() * np.sqrt(252)
                    weight = weights.get(symbol, 0)
                    
                    volatility_attribution[symbol] = {
                        'individual_volatility': volatility,
                        'weight': weight,
                        'weighted_volatility': volatility * weight,
                        'contribution_pct': (volatility * weight) * 100
                    }
        
        return volatility_attribution
    
    def _get_sector_from_symbol(self, symbol: str) -> str:
        """심볼에서 섹터 추정 (간단한 예시)"""
        # 실제로는 외부 데이터 소스에서 섹터 정보를 가져와야 함
        if symbol.upper().startswith(('AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA')):
            return 'Technology'
        elif symbol.upper().startswith(('JPM', 'BAC', 'WFC', 'GS')):
            return 'Financial'
        elif symbol.upper().startswith(('JNJ', 'PFE', 'UNH', 'ABBV')):
            return 'Healthcare'
        elif symbol.upper().startswith(('XOM', 'CVX', 'COP')):
            return 'Energy'
        else:
            return 'Other'
    
    def _generate_optimization_recommendations(self, 
                                             portfolio_snapshot: PortfolioSnapshot,
                                             analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """포트폴리오 최적화 권고사항 생성"""
        
        recommendations = []
        
        # 1. 집중도 기반 권고
        concentration = analysis_results.get('concentration_analysis', {})
        if concentration.get('top_1_concentration', 0) > 0.3:
            recommendations.append({
                'type': 'concentration',
                'priority': 'high',
                'title': '과도한 포지션 집중도',
                'description': f"가장 큰 포지션이 {concentration['top_1_concentration']:.1%}를 차지합니다.",
                'action': '포지션 분산을 위해 비중을 줄이거나 다른 종목을 추가하세요.',
                'impact': 'risk_reduction'
            })
        
        # 2. 상관관계 기반 권고
        correlation = analysis_results.get('correlation_analysis', {})
        high_corr_pairs = correlation.get('high_correlation_pairs', [])
        if len(high_corr_pairs) > 0:
            recommendations.append({
                'type': 'correlation',
                'priority': 'medium',
                'title': '높은 상관관계 감지',
                'description': f"{len(high_corr_pairs)}개의 높은 상관관계 쌍이 발견되었습니다.",
                'action': '상관관계가 낮은 자산을 추가하여 다각화를 개선하세요.',
                'impact': 'diversification'
            })
        
        # 3. VaR 기반 권고
        var_analysis = analysis_results.get('var_analysis', {})
        if 'historical_95' in var_analysis:
            var_95_pct = var_analysis['historical_95'].get('var_pct', 0)
            if var_95_pct > 0.05:  # 5% 초과
                recommendations.append({
                    'type': 'var',
                    'priority': 'high',
                    'title': 'VaR 한도 초과',
                    'description': f"95% VaR이 {var_95_pct:.2%}로 권장 수준을 초과합니다.",
                    'action': '포지션 크기를 줄이거나 헷지 전략을 고려하세요.',
                    'impact': 'risk_reduction'
                })
        
        # 4. 시나리오 기반 권고
        scenario_analysis = analysis_results.get('scenario_analysis', {})
        worst_case = scenario_analysis.get('worst_case_scenario', 0)
        if worst_case < -0.25:  # 25% 이상 손실
            recommendations.append({
                'type': 'scenario',
                'priority': 'high',
                'title': '극단적 시나리오 위험',
                'description': f"최악의 시나리오에서 {abs(worst_case):.1%} 손실이 예상됩니다.",
                'action': '방어적 자산 추가나 헷지 전략을 고려하세요.',
                'impact': 'downside_protection'
            })
        
        # 5. 다각화 권고
        if len(portfolio_snapshot.positions) < 5:
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'title': '포지션 수 부족',
                'description': f"현재 {len(portfolio_snapshot.positions)}개 포지션으로 다각화가 부족합니다.",
                'action': '5-10개 종목으로 포지션을 확대하여 다각화하세요.',
                'impact': 'risk_reduction'
            })
        
        # 우선순위별 정렬
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def calculate_efficient_frontier(self, 
                                   price_history: pd.DataFrame,
                                   symbols: List[str],
                                   num_portfolios: int = 100) -> Dict[str, Any]:
        """효율적 투자선 계산"""
        
        # 수익률 계산
        returns_data = {}
        for symbol in symbols:
            if symbol in price_history.columns:
                prices = price_history[symbol].dropna()
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return {}
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 30:
            return {}
        
        # 평균 수익률과 공분산 행렬
        mean_returns = returns_df.mean() * 252  # 연환산
        cov_matrix = returns_df.cov() * 252
        
        # 효율적 투자선 계산
        num_assets = len(symbols)
        results = np.zeros((4, num_portfolios))
        
        np.random.seed(42)
        
        for i in range(num_portfolios):
            # 랜덤 가중치 생성
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            
            # 포트폴리오 수익률과 리스크
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 샤프 비율 (리스크 프리 레이트 0 가정)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            results[3, i] = i  # 포트폴리오 인덱스
        
        # 최적 포트폴리오 찾기
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        
        efficient_frontier = {
            'returns': results[0].tolist(),
            'volatilities': results[1].tolist(),
            'sharpe_ratios': results[2].tolist(),
            'max_sharpe_portfolio': {
                'return': results[0, max_sharpe_idx],
                'volatility': results[1, max_sharpe_idx],
                'sharpe_ratio': results[2, max_sharpe_idx]
            },
            'min_volatility_portfolio': {
                'return': results[0, min_vol_idx],
                'volatility': results[1, min_vol_idx],
                'sharpe_ratio': results[2, min_vol_idx]
            },
            'symbols': symbols,
            'mean_returns': mean_returns.to_dict(),
            'correlation_matrix': returns_df.corr().to_dict()
        }
        
        return efficient_frontier
    
    def add_custom_scenario(self, scenario: RiskScenario):
        """커스텀 시나리오 추가"""
        self.scenarios.append(scenario)
    
    def get_risk_budget_allocation(self, 
                                 portfolio_snapshot: PortfolioSnapshot,
                                 target_risk_budget: float = 0.02) -> Dict[str, Any]:
        """리스크 예산 배분 계산"""
        
        weights = portfolio_snapshot.get_position_weights()
        
        if not weights:
            return {}
        
        # 현재 리스크 기여도 계산 (간단한 예시)
        risk_contributions = {}
        total_risk = 0
        
        for symbol, weight in weights.items():
            # 예시: 가중치에 비례한 리스크 기여도
            risk_contribution = weight * target_risk_budget
            risk_contributions[symbol] = risk_contribution
            total_risk += risk_contribution
        
        # 리스크 예산 배분 권고
        recommendations = {}
        for symbol, contribution in risk_contributions.items():
            budget_utilization = contribution / target_risk_budget
            
            if budget_utilization > 0.3:  # 30% 이상
                recommendations[symbol] = {
                    'action': 'reduce',
                    'current_contribution': contribution,
                    'recommended_contribution': target_risk_budget * 0.25,
                    'reason': '과도한 리스크 기여도'
                }
            elif budget_utilization < 0.05:  # 5% 미만
                recommendations[symbol] = {
                    'action': 'increase',
                    'current_contribution': contribution,
                    'recommended_contribution': target_risk_budget * 0.1,
                    'reason': '리스크 예산 활용도 부족'
                }
        
        return {
            'target_risk_budget': target_risk_budget,
            'current_risk_contributions': risk_contributions,
            'total_risk_utilization': total_risk / target_risk_budget,
            'recommendations': recommendations
        }