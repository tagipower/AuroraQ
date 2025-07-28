#!/usr/bin/env python3
"""
AuroraQ Shared Performance Metrics
=================================

Unified performance metrics calculation for Production, Backtest, and Shared components.
Consolidates performance analysis functionality across all AuroraQ components.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """통합 성과 지표"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    avg_holding_period: float  # 평균 보유 기간 (분)
    
    # 추가 리스크 지표
    var_95: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_holding_period': self.avg_holding_period,
            'var_95': self.var_95,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio
        }

def calculate_returns_from_trades(trades: List[Dict[str, Any]]) -> List[float]:
    """거래 내역에서 수익률 계산"""
    returns = []
    
    closed_trades = [t for t in trades if t.get('action') == 'close']
    
    for trade in closed_trades:
        pnl_pct = trade.get('pnl_pct', 0.0)
        returns.append(pnl_pct)
    
    return returns

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """샤프 비율 계산"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """소르티노 비율 계산 (하향 변동성만 고려)"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # 하향 편차 계산
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_deviation

def calculate_max_drawdown(returns: List[float]) -> float:
    """최대 낙폭 계산"""
    if not returns:
        return 0.0
    
    # 누적 수익률 계산
    cumulative_returns = np.cumprod(1 + np.array(returns))
    
    # 최고점 추적
    peak = np.maximum.accumulate(cumulative_returns)
    
    # 낙폭 계산
    drawdowns = (cumulative_returns - peak) / peak
    
    return abs(np.min(drawdowns))

def calculate_win_rate(returns: List[float]) -> float:
    """승률 계산"""
    if not returns:
        return 0.0
    
    winning_trades = sum(1 for r in returns if r > 0)
    return winning_trades / len(returns)

def calculate_profit_factor(returns: List[float]) -> float:
    """이익 팩터 계산 (총 이익 / 총 손실)"""
    if not returns:
        return 1.0
    
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 1.0
    
    return gross_profit / gross_loss

def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
    """변동성 계산"""
    if not returns or len(returns) < 2:
        return 0.0
    
    volatility = np.std(returns)
    
    if annualize:
        # 연간화 (일일 거래 가정 시 252일, 시간별 거래 시 365*24)
        volatility *= np.sqrt(252)  # 일일 거래 기준
    
    return volatility

def calculate_var(returns: List[float], confidence_level: float = 0.05) -> float:
    """Value at Risk 계산"""
    if not returns or len(returns) < 20:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)

def calculate_performance_metrics(trades: Union[List[Dict[str, Any]], pd.DataFrame], 
                                initial_capital: float = 100000) -> PerformanceMetrics:
    """종합 성과 지표 계산 - 거래 리스트 또는 DataFrame 지원"""
    
    # DataFrame을 리스트로 변환
    if isinstance(trades, pd.DataFrame):
        if trades.empty:
            return _get_empty_metrics()
        trades = trades.to_dict('records')
    
    if not trades:
        return _get_empty_metrics()
    
    # 거래 데이터 필터링
    closed_trades = [t for t in trades if t.get('action') == 'close']
    
    if not closed_trades:
        return _get_empty_metrics()
    
    # 수익률 추출
    returns = [t.get('pnl_pct', 0.0) for t in closed_trades]
    pnl_amounts = [t.get('pnl', 0.0) for t in closed_trades]
    
    # 기본 통계
    total_trades = len(closed_trades)
    winning_trades = sum(1 for r in returns if r > 0)
    losing_trades = sum(1 for r in returns if r < 0)
    
    # 수익률 지표
    total_return = sum(returns)
    avg_trade_return = np.mean(returns) if returns else 0.0
    
    # 연간화 수익률 (거래 기간 기반)
    if len(closed_trades) >= 2:
        try:
            first_trade = closed_trades[0]['timestamp']
            last_trade = closed_trades[-1]['timestamp']
            
            if isinstance(first_trade, str):
                first_trade = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
            if isinstance(last_trade, str):
                last_trade = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
            
            days_elapsed = (last_trade - first_trade).days + 1
            if days_elapsed > 0:
                annualized_return = ((1 + total_return) ** (365 / days_elapsed)) - 1
            else:
                annualized_return = total_return
        except (KeyError, ValueError, TypeError):
            annualized_return = total_return
    else:
        annualized_return = total_return
    
    # 위험 지표
    volatility = calculate_volatility(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    sortino_ratio = calculate_sortino_ratio(returns)
    max_drawdown = calculate_max_drawdown(returns)
    var_95 = calculate_var(returns, 0.05)
    
    # 거래 성과 지표
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 극값
    largest_win = max(returns) if returns else 0.0
    largest_loss = min(returns) if returns else 0.0
    
    # 평균 보유 기간
    holding_periods = []
    for trade in closed_trades:
        if 'holding_time' in trade:
            holding_time = trade['holding_time']
            if isinstance(holding_time, timedelta):
                holding_periods.append(holding_time.total_seconds() / 60)  # 분 단위
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_holding_period=avg_holding_period,
        var_95=var_95,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio
    )

def _get_empty_metrics() -> PerformanceMetrics:
    """빈 메트릭스 반환"""
    return PerformanceMetrics(
        total_return=0.0, annualized_return=0.0, volatility=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
        profit_factor=1.0, avg_trade_return=0.0, total_trades=0,
        winning_trades=0, losing_trades=0, largest_win=0.0,
        largest_loss=0.0, avg_holding_period=0.0
    )

def calculate_portfolio_metrics(positions: Dict[str, float], 
                              prices: Dict[str, float]) -> Dict[str, Any]:
    """포트폴리오 지표 계산"""
    
    if not positions or not prices:
        return {'total_value': 0.0, 'asset_allocation': {}}
    
    # 총 포트폴리오 가치
    total_value = sum(positions[asset] * prices.get(asset, 0) 
                     for asset in positions)
    
    # 자산 배분
    asset_allocation = {}
    for asset in positions:
        if total_value > 0:
            allocation = (positions[asset] * prices.get(asset, 0)) / total_value
            asset_allocation[asset] = allocation
    
    return {
        'total_value': total_value,
        'asset_allocation': asset_allocation,
        'num_positions': len([p for p in positions.values() if p != 0]),
        'max_position_weight': max(asset_allocation.values()) if asset_allocation else 0.0
    }

def calculate_risk_metrics(trades: Union[List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, float]:
    """리스크 관련 지표 계산"""
    
    # DataFrame을 리스트로 변환
    if isinstance(trades, pd.DataFrame):
        if trades.empty:
            return {}
        trades = trades.to_dict('records')
    
    if not trades:
        return {}
    
    # 기본 통계
    returns = [t.get('pnl_pct', 0.0) for t in trades if t.get('action') == 'close']
    
    if not returns:
        return {}
    
    returns = np.array(returns)
    
    # Sharpe Ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Maximum Drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown)
    
    # Value at Risk (VaR) - 95% 신뢰수준
    var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0
    
    # Calmar Ratio
    annual_return = mean_return * 252
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = mean_return / downside_deviation if downside_deviation > 0 else 0
    
    return {
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_drawdown, 3),
        "var_95": round(var_95, 4),
        "calmar_ratio": round(calmar, 3),
        "avg_loss": round(np.mean(returns[returns < 0]), 4) if any(returns < 0) else 0,
        "avg_win": round(np.mean(returns[returns > 0]), 4) if any(returns > 0) else 0,
        "risk_reward_ratio": abs(np.mean(returns[returns > 0]) / np.mean(returns[returns < 0])) 
                            if any(returns < 0) and any(returns > 0) else 0
    }

def format_performance_report(metrics: PerformanceMetrics) -> str:
    """성과 지표를 읽기 쉬운 리포트로 포맷팅"""
    
    report = f"""
=== AuroraQ 거래 성과 리포트 ===

📊 수익 지표:
   총 수익률: {metrics.total_return:.2%}
   연간 수익률: {metrics.annualized_return:.2%}
   평균 거래 수익률: {metrics.avg_trade_return:.2%}

📈 위험 지표:
   변동성: {metrics.volatility:.2%}
   샤프 비율: {metrics.sharpe_ratio:.3f}
   소르티노 비율: {metrics.sortino_ratio:.3f}
   최대 낙폭: {metrics.max_drawdown:.2%}
   칼마 비율: {metrics.calmar_ratio:.3f}
   VaR (95%): {metrics.var_95:.2%}

🎯 거래 통계:
   총 거래 수: {metrics.total_trades}
   승률: {metrics.win_rate:.1%}
   이익 팩터: {metrics.profit_factor:.2f}
   승리/패배: {metrics.winning_trades}/{metrics.losing_trades}

💰 극값:
   최대 수익: {metrics.largest_win:.2%}
   최대 손실: {metrics.largest_loss:.2%}
   평균 보유시간: {metrics.avg_holding_period:.1f}분

===============================
"""
    
    return report

def compare_strategies(metrics_dict: Dict[str, PerformanceMetrics]) -> pd.DataFrame:
    """전략별 성과 비교"""
    
    comparison_data = []
    
    for strategy_name, metrics in metrics_dict.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{metrics.total_return:.2%}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
            'Max Drawdown': f"{metrics.max_drawdown:.2%}",
            'Win Rate': f"{metrics.win_rate:.1%}",
            'Total Trades': metrics.total_trades,
            'Profit Factor': f"{metrics.profit_factor:.2f}"
        })
    
    return pd.DataFrame(comparison_data)

# 백테스트 호환성을 위한 알리아스
def calculate_backtest_metrics(trades_df: pd.DataFrame, **kwargs) -> PerformanceMetrics:
    """백테스트 메트릭스 계산 (호환성 함수)"""
    return calculate_performance_metrics(trades_df, **kwargs)