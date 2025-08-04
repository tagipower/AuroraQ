#!/usr/bin/env python3
"""
포트폴리오 관리 - 자본, 포지션, 수익률 추적
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    commission: float
    trade_id: str
    
    @property
    def value(self) -> float:
        """거래 금액"""
        return self.size * self.price
    
    @property
    def total_cost(self) -> float:
        """수수료 포함 총 비용"""
        return self.value + self.commission


@dataclass
class Position:
    """포지션 정보"""
    size: float = 0.0  # 보유 수량
    avg_price: float = 0.0  # 평균 매입가
    unrealized_pnl: float = 0.0  # 미실현 손익
    realized_pnl: float = 0.0  # 실현 손익
    
    @property
    def is_long(self) -> bool:
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        return self.size < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.size) < 1e-8
    
    @property
    def market_value(self) -> float:
        """현재 시장가치"""
        return abs(self.size) * self.avg_price


class Portfolio:
    """포트폴리오 관리자"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 max_position_size: float = 0.95):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission
        self.max_position_size = max_position_size
        
        # 포지션 관리
        self.position = Position()
        self.trades: List[Trade] = []
        
        # 성과 추적
        self.equity_history: List[Tuple[datetime, float]] = []
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
        
        # 통계
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def get_equity(self) -> float:
        """총 자산가치 (현금 + 포지션 가치)"""
        return self.cash + self.position.market_value + self.position.unrealized_pnl
    
    def get_position_value(self) -> float:
        """포지션 가치"""
        return self.position.market_value + self.position.unrealized_pnl
    
    def get_available_capital(self) -> float:
        """사용 가능한 자본"""
        equity = self.get_equity()
        position_value = abs(self.get_position_value())
        return max(0, equity * self.max_position_size - position_value)
    
    def can_buy(self, size: float, price: float) -> Tuple[bool, str]:
        """매수 가능 여부 확인"""
        required_capital = size * price * (1 + self.commission_rate)
        available = self.get_available_capital()
        
        if required_capital > available:
            return False, f"자본 부족: 필요 ${required_capital:.2f}, 가용 ${available:.2f}"
        
        return True, "OK"
    
    def can_sell(self, size: float) -> Tuple[bool, str]:
        """매도 가능 여부 확인"""
        if self.position.size < size:
            return False, f"보유량 부족: 보유 {self.position.size:.6f}, 매도 {size:.6f}"
        
        return True, "OK"
    
    def execute_buy(self, size: float, price: float, timestamp: datetime, trade_id: str) -> Trade:
        """매수 실행"""
        # 수수료 계산
        commission = size * price * self.commission_rate
        total_cost = size * price + commission
        
        # 자본 확인
        can_buy, reason = self.can_buy(size, price)
        if not can_buy:
            raise ValueError(f"매수 불가: {reason}")
        
        # 포지션 업데이트
        if self.position.is_flat:
            # 새 포지션
            self.position.size = size
            self.position.avg_price = price
        else:
            # 기존 포지션에 추가
            total_size = self.position.size + size
            total_value = (self.position.size * self.position.avg_price) + (size * price)
            self.position.avg_price = total_value / total_size
            self.position.size = total_size
        
        # 현금 업데이트
        self.cash -= total_cost
        self.total_commission_paid += commission
        
        # 거래 기록
        trade = Trade(
            timestamp=timestamp,
            side='buy',
            size=size,
            price=price,
            commission=commission,
            trade_id=trade_id
        )
        self.trades.append(trade)
        self.total_trades += 1
        
        return trade
    
    def execute_sell(self, size: float, price: float, timestamp: datetime, trade_id: str) -> Trade:
        """매도 실행"""
        # 보유량 확인
        can_sell, reason = self.can_sell(size)
        if not can_sell:
            raise ValueError(f"매도 불가: {reason}")
        
        # 수수료 계산
        commission = size * price * self.commission_rate
        proceeds = size * price - commission
        
        # 실현 손익 계산
        cost_basis = size * self.position.avg_price
        realized_pnl = proceeds - cost_basis
        self.position.realized_pnl += realized_pnl
        
        # 포지션 업데이트
        self.position.size -= size
        if abs(self.position.size) < 1e-8:
            self.position.size = 0.0
            self.position.avg_price = 0.0
        
        # 현금 업데이트
        self.cash += proceeds
        self.total_commission_paid += commission
        
        # 거래 통계 업데이트
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # 거래 기록
        trade = Trade(
            timestamp=timestamp,
            side='sell',
            size=size,
            price=price,
            commission=commission,
            trade_id=trade_id
        )
        self.trades.append(trade)
        self.total_trades += 1
        
        return trade
    
    def update(self, current_price: float, timestamp: Optional[datetime] = None):
        """포트폴리오 업데이트"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # 미실현 손익 업데이트
        if not self.position.is_flat:
            market_value = self.position.size * current_price
            cost_basis = self.position.size * self.position.avg_price
            self.position.unrealized_pnl = market_value - cost_basis
        else:
            self.position.unrealized_pnl = 0.0
        
        # 총 자산가치
        current_equity = self.get_equity()
        
        # 최고점 및 낙폭 업데이트
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
        
        current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 자산가치 기록
        self.equity_history.append((timestamp, current_equity))
    
    def get_returns(self) -> pd.Series:
        """수익률 시계열 반환"""
        if len(self.equity_history) < 2:
            return pd.Series()
        
        timestamps, equity_values = zip(*self.equity_history)
        equity_series = pd.Series(equity_values, index=timestamps)
        return equity_series.pct_change().dropna()
    
    def get_total_return(self) -> float:
        """총 수익률"""
        if not self.equity_history:
            return 0.0
        
        final_equity = self.equity_history[-1][1]
        return (final_equity - self.initial_capital) / self.initial_capital
    
    def get_position_info(self) -> Dict:
        """포지션 정보 딕셔너리"""
        return {
            'size': self.position.size,
            'avg_price': self.position.avg_price,
            'market_value': self.position.market_value,
            'unrealized_pnl': self.position.unrealized_pnl,
            'realized_pnl': self.position.realized_pnl,
            'is_long': self.position.is_long,
            'is_short': self.position.is_short,
            'is_flat': self.position.is_flat
        }
    
    def get_performance_summary(self) -> Dict:
        """성과 요약"""
        returns = self.get_returns()
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_equity': self.get_equity(),
            'total_return': self.get_total_return(),
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_commission': self.total_commission_paid,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl
        }
        
        if len(returns) > 0:
            summary.update({
                'volatility': returns.std() * np.sqrt(252),  # 연환산
                'sharpe_ratio': self._calculate_sharpe(returns),
                'avg_return': returns.mean(),
                'max_daily_return': returns.max(),
                'min_daily_return': returns.min()
            })
        
        return summary
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # 연환산
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility
    
    def to_dataframe(self) -> pd.DataFrame:
        """거래 내역을 DataFrame으로 변환"""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'timestamp': trade.timestamp,
                'side': trade.side,
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'total_cost': trade.total_cost,
                'trade_id': trade.trade_id
            })
        
        return pd.DataFrame(data)
    
    def reset(self):
        """포트폴리오 초기화"""
        self.cash = self.initial_capital
        self.position = Position()
        self.trades.clear()
        self.equity_history.clear()
        self.high_water_mark = self.initial_capital
        self.max_drawdown = 0.0
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0