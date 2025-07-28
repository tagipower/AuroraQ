#!/usr/bin/env python3
"""
성과 지표 계산 - 포괄적인 백테스트 성과 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """백테스트 성과 분석기"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self,
                         equity_curve: pd.Series,
                         returns: pd.Series,
                         trades: pd.DataFrame,
                         risk_free_rate: Optional[float] = None) -> Dict:
        """
        포괄적인 성과 지표 계산
        
        Args:
            equity_curve: 자산가치 시계열
            returns: 수익률 시계열
            trades: 거래 내역 DataFrame
            risk_free_rate: 무위험 수익률
            
        Returns:
            성과 지표 딕셔너리
        """
        rf_rate = risk_free_rate or self.risk_free_rate
        
        metrics = {}
        
        # 기본 수익률 지표
        metrics.update(self._calculate_return_metrics(equity_curve, returns))
        
        # 리스크 지표
        metrics.update(self._calculate_risk_metrics(equity_curve, returns, rf_rate))
        
        # 거래 통계
        metrics.update(self._calculate_trade_metrics(trades))
        
        # 시간별 분석
        metrics.update(self._calculate_time_based_metrics(returns))
        
        # 고급 지표
        metrics.update(self._calculate_advanced_metrics(equity_curve, returns, rf_rate))
        
        return metrics
    
    def _calculate_return_metrics(self, equity_curve: pd.Series, returns: pd.Series) -> Dict:
        """수익률 지표 계산"""
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'cumulative_return': 0.0
            }
        
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        # 총 수익률
        total_return = (final_value - initial_value) / initial_value
        
        # 연환산 수익률
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days / 365.25, 1/252)  # 최소 1일
        annualized_return = (final_value / initial_value) ** (1/years) - 1
        
        # 누적 수익률 시계열
        cumulative_returns = (equity_curve / initial_value - 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_returns.iloc[-1],
            'cumulative_returns_series': cumulative_returns
        }
    
    def _calculate_risk_metrics(self, equity_curve: pd.Series, returns: pd.Series, rf_rate: float) -> Dict:
        """리스크 지표 계산"""
        if len(returns) == 0:
            return {
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        # 변동성 (연환산)
        volatility = returns.std() * np.sqrt(252)
        
        # 최대 낙폭 (MDD)
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # 샤프 비율
        excess_returns = returns.mean() * 252 - rf_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
        
        # 칼마 비율
        annualized_return = returns.mean() * 252
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'drawdown_series': drawdowns
        }
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """거래 통계 계산"""
        if trades.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # 거래쌍 분석을 위해 매수/매도 매칭
        trade_pairs = self._match_trade_pairs(trades)
        
        if not trade_pairs:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # 손익 계산
        pnl_list = [pair['pnl'] for pair in trade_pairs]
        winning_trades = [pnl for pnl in pnl_list if pnl > 0]
        losing_trades = [pnl for pnl in pnl_list if pnl <= 0]
        
        # 통계 계산
        total_trades = len(trade_pairs)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_time_based_metrics(self, returns: pd.Series) -> Dict:
        """시간별 성과 분석"""
        if len(returns) == 0:
            return {
                'monthly_returns': pd.Series(),
                'yearly_returns': pd.Series(),
                'best_month': 0.0,
                'worst_month': 0.0,
                'positive_months': 0,
                'negative_months': 0
            }
        
        # 월별 수익률
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 연도별 수익률
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # 월별 통계
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns <= 0).sum()
        
        return {
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
            'best_month': best_month,
            'worst_month': worst_month,
            'positive_months': positive_months,
            'negative_months': negative_months,
            'monthly_win_rate': positive_months / len(monthly_returns) if len(monthly_returns) > 0 else 0
        }
    
    def _calculate_advanced_metrics(self, equity_curve: pd.Series, returns: pd.Series, rf_rate: float) -> Dict:
        """고급 성과 지표"""
        if len(returns) < 30:
            return {
                'rolling_sharpe': pd.Series(),
                'rolling_volatility': pd.Series(),
                'var_95': 0.0,
                'cvar_95': 0.0,
                'tail_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        # 롤링 샤프 비율 (30일 윈도우)
        rolling_returns = returns.rolling(30)
        rolling_sharpe = rolling_returns.apply(
            lambda x: (x.mean() * 252 - rf_rate) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        # 롤링 변동성
        rolling_volatility = returns.rolling(30).std() * np.sqrt(252)
        
        # VaR 및 CVaR (95% 신뢰구간)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # 테일 비율 (상위 5% / 하위 5%)
        upper_tail = returns.quantile(0.95)
        lower_tail = returns.quantile(0.05)
        tail_ratio = abs(upper_tail / lower_tail) if lower_tail != 0 else 0
        
        # 왜도와 첨도
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _match_trade_pairs(self, trades: pd.DataFrame) -> List[Dict]:
        """매수/매도 거래를 매칭하여 손익 계산"""
        if trades.empty:
            return []
        
        trade_pairs = []
        position = 0.0
        avg_price = 0.0
        
        for _, trade in trades.iterrows():
            if trade['side'] == 'buy':
                if position == 0:
                    # 새 포지션 시작
                    position = trade['size']
                    avg_price = trade['price']
                else:
                    # 기존 포지션에 추가
                    total_value = (position * avg_price) + (trade['size'] * trade['price'])
                    total_size = position + trade['size']
                    avg_price = total_value / total_size
                    position = total_size
            
            elif trade['side'] == 'sell' and position > 0:
                # 매도 - 손익 실현
                sell_size = min(trade['size'], position)
                pnl = (trade['price'] - avg_price) * sell_size
                
                trade_pairs.append({
                    'entry_price': avg_price,
                    'exit_price': trade['price'],
                    'size': sell_size,
                    'pnl': pnl,
                    'return': pnl / (avg_price * sell_size),
                    'exit_time': trade['timestamp']
                })
                
                position -= sell_size
                
                if position <= 0:
                    position = 0.0
                    avg_price = 0.0
        
        return trade_pairs
    
    def calculate_sector_analysis(self, returns: pd.Series) -> Dict:
        """섹터별/기간별 상세 분석"""
        if len(returns) == 0:
            return {}
        
        analysis = {}
        
        # 요일별 성과
        weekday_returns = returns.groupby(returns.index.weekday).mean()
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        analysis['weekday_performance'] = dict(zip(weekday_names, weekday_returns))
        
        # 시간대별 성과 (시간 정보가 있는 경우)
        if hasattr(returns.index, 'hour'):
            hourly_returns = returns.groupby(returns.index.hour).mean()
            analysis['hourly_performance'] = hourly_returns.to_dict()
        
        # 월별 성과
        monthly_returns = returns.groupby(returns.index.month).mean()
        month_names = ['1월', '2월', '3월', '4월', '5월', '6월',
                      '7월', '8월', '9월', '10월', '11월', '12월']
        analysis['monthly_performance'] = dict(zip(month_names, monthly_returns))
        
        return analysis
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """성과 리포트 생성"""
        report = f"""
=== 백테스트 성과 리포트 ===

📊 수익률 지표
- 총 수익률: {metrics.get('total_return', 0):.2%}
- 연환산 수익률: {metrics.get('annualized_return', 0):.2%}

⚠️ 리스크 지표
- 최대 낙폭: {metrics.get('max_drawdown', 0):.2%}
- 변동성: {metrics.get('volatility', 0):.2%}
- 샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}
- 소르티노 비율: {metrics.get('sortino_ratio', 0):.2f}

💹 거래 통계
- 총 거래: {metrics.get('total_trades', 0)}
- 승률: {metrics.get('win_rate', 0):.1%}
- 평균 수익: {metrics.get('avg_win', 0):.2%}
- 평균 손실: {metrics.get('avg_loss', 0):.2%}
- 손익비: {metrics.get('profit_factor', 0):.2f}

📈 고급 지표
- VaR (95%): {metrics.get('var_95', 0):.2%}
- CVaR (95%): {metrics.get('cvar_95', 0):.2%}
- 왜도: {metrics.get('skewness', 0):.2f}
- 첨도: {metrics.get('kurtosis', 0):.2f}

========================
"""
        return report