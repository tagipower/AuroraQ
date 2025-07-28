#!/usr/bin/env python3
"""
성과 지표 계산
샤프 비율, 최대 낙폭, 승률 등 거래 성과 지표
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """성과 지표"""
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
            'avg_holding_period': self.avg_holding_period
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

def calculate_performance_metrics(trades: List[Dict[str, Any]], 
                                initial_capital: float = 100000) -> PerformanceMetrics:
    """종합 성과 지표 계산"""
    
    if not trades:
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
            profit_factor=1.0, avg_trade_return=0.0, total_trades=0,
            winning_trades=0, losing_trades=0, largest_win=0.0,
            largest_loss=0.0, avg_holding_period=0.0
        )
    
    # 거래 데이터 필터링
    closed_trades = [t for t in trades if t.get('action') == 'close']
    
    if not closed_trades:
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
            profit_factor=1.0, avg_trade_return=0.0, total_trades=0,
            winning_trades=0, losing_trades=0, largest_win=0.0,
            largest_loss=0.0, avg_holding_period=0.0
        )
    
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
        first_trade = closed_trades[0]['timestamp']
        last_trade = closed_trades[-1]['timestamp']
        
        if isinstance(first_trade, str):
            first_trade = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
        if isinstance(last_trade, str):
            last_trade = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
        
        days_elapsed = (last_trade - first_trade).days + 1
        annualized_return = ((1 + total_return) ** (365 / days_elapsed)) - 1
    else:
        annualized_return = total_return
    
    # 위험 지표
    volatility = calculate_volatility(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(returns)
    
    # 거래 성과 지표
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
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
        avg_holding_period=avg_holding_period
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

def format_performance_report(metrics: PerformanceMetrics) -> str:
    """성과 지표를 읽기 쉬운 리포트로 포맷팅"""
    
    report = f"""
=== 거래 성과 리포트 ===

📊 수익 지표:
   총 수익률: {metrics.total_return:.2%}
   연간 수익률: {metrics.annualized_return:.2%}
   평균 거래 수익률: {metrics.avg_trade_return:.2%}

📈 위험 지표:
   변동성: {metrics.volatility:.2%}
   샤프 비율: {metrics.sharpe_ratio:.3f}
   최대 낙폭: {metrics.max_drawdown:.2%}

🎯 거래 통계:
   총 거래 수: {metrics.total_trades}
   승률: {metrics.win_rate:.1%}
   이익 팩터: {metrics.profit_factor:.2f}
   승리/패배: {metrics.winning_trades}/{metrics.losing_trades}

💰 극값:
   최대 수익: {metrics.largest_win:.2%}
   최대 손실: {metrics.largest_loss:.2%}
   평균 보유시간: {metrics.avg_holding_period:.1f}분

========================
"""
    
    return report