#!/usr/bin/env python3
"""
포지션 및 거래 데이터 모델
백테스트와 실시간 거래에서 공통으로 사용하는 데이터 구조
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """거래 상태"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class Trade:
    """거래 기록"""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    size: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def value(self) -> float:
        """거래 금액 (수수료 제외)"""
        return self.size * self.price
    
    @property
    def total_cost(self) -> float:
        """총 비용 (수수료 + 슬리피지 포함)"""
        base_cost = self.value
        if self.side == OrderSide.BUY:
            return base_cost + self.commission + (self.slippage * base_cost)
        else:
            return base_cost - self.commission - (self.slippage * base_cost)
    
    @property
    def net_value(self) -> float:
        """순 거래가치"""
        multiplier = 1 if self.side == OrderSide.BUY else -1
        return self.total_cost * multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'value': self.value,
            'commission': self.commission,
            'slippage': self.slippage,
            'total_cost': self.total_cost,
            'status': self.status.value,
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }


@dataclass
class PositionState:
    """포지션 상태"""
    symbol: str = ""
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    entry_timestamp: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """현재 시장가치"""
        if self.side == PositionSide.FLAT:
            return 0.0
        return abs(self.size) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """취득원가"""
        if self.side == PositionSide.FLAT:
            return 0.0
        return abs(self.size) * self.avg_entry_price
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """손익률"""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis
    
    @property
    def is_long(self) -> bool:
        """롱 포지션 여부"""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """숏 포지션 여부"""
        return self.side == PositionSide.SHORT
    
    @property
    def is_flat(self) -> bool:
        """빈 포지션 여부"""
        return self.side == PositionSide.FLAT or abs(self.size) < 1e-8
    
    def update_price(self, new_price: float):
        """가격 업데이트 및 미실현 손익 계산"""
        self.current_price = new_price
        self.last_update = datetime.now()
        
        if not self.is_flat:
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = (new_price - self.avg_entry_price) * self.size
            else:  # SHORT
                self.unrealized_pnl = (self.avg_entry_price - new_price) * abs(self.size)
    
    def add_position(self, trade: Trade):
        """포지션 추가"""
        if self.is_flat:
            # 새 포지션 시작
            self.symbol = trade.symbol
            self.side = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
            self.size = trade.size if trade.side == OrderSide.BUY else -trade.size
            self.avg_entry_price = trade.price
            self.entry_timestamp = trade.timestamp
        else:
            # 기존 포지션에 추가
            current_value = self.size * self.avg_entry_price
            trade_value = trade.size * trade.price
            
            if trade.side == OrderSide.BUY:
                new_size = self.size + trade.size
                new_avg_price = (current_value + trade_value) / new_size if new_size != 0 else 0
                self.size = new_size
            else:  # SELL
                new_size = self.size - trade.size
                if new_size > 0:
                    # 부분 청산
                    self.size = new_size
                elif new_size == 0:
                    # 완전 청산
                    self._close_position()
                    return
                else:
                    # 반대 포지션
                    self.side = PositionSide.SHORT
                    self.size = -new_size
                    self.avg_entry_price = trade.price
                    self.entry_timestamp = trade.timestamp
                    return
            
            self.avg_entry_price = new_avg_price
        
        # 수수료 누적
        self.total_commission += trade.commission
    
    def close_position(self, trade: Trade) -> float:
        """포지션 청산 및 실현 손익 반환"""
        if self.is_flat:
            return 0.0
        
        # 실현 손익 계산
        if self.side == PositionSide.LONG:
            realized_pnl = (trade.price - self.avg_entry_price) * min(trade.size, self.size)
        else:
            realized_pnl = (self.avg_entry_price - trade.price) * min(trade.size, abs(self.size))
        
        # 수수료 차감
        realized_pnl -= trade.commission
        
        # 포지션 크기 조정
        if trade.size >= abs(self.size):
            # 완전 청산
            self.realized_pnl += realized_pnl
            self._close_position()
        else:
            # 부분 청산
            remaining_ratio = (abs(self.size) - trade.size) / abs(self.size)
            self.size *= remaining_ratio
            self.realized_pnl += realized_pnl
        
        return realized_pnl
    
    def _close_position(self):
        """포지션 완전 청산"""
        self.side = PositionSide.FLAT
        self.size = 0.0
        self.avg_entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.entry_timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'avg_entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'pnl_percentage': self.pnl_percentage,
            'total_commission': self.total_commission,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'last_update': self.last_update.isoformat()
        }


@dataclass
class OrderSignal:
    """거래 신호"""
    action: str  # 'buy', 'sell', 'hold'
    symbol: str = ""
    size: float = 0.0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    confidence: float = 0.5
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_buy(self) -> bool:
        return self.action.lower() == 'buy'
    
    @property
    def is_sell(self) -> bool:
        return self.action.lower() == 'sell'
    
    @property
    def is_hold(self) -> bool:
        return self.action.lower() == 'hold'
    
    def validate(self) -> Tuple[bool, str]:
        """신호 유효성 검증"""
        if self.action not in ['buy', 'sell', 'hold']:
            return False, f"Invalid action: {self.action}"
        
        if not self.is_hold and self.size <= 0:
            return False, f"Invalid size: {self.size}"
        
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            return False, "Limit price required for limit orders"
        
        if self.order_type == OrderType.STOP and self.stop_price is None:
            return False, "Stop price required for stop orders"
        
        if self.confidence < 0 or self.confidence > 1:
            return False, f"Invalid confidence: {self.confidence}"
        
        return True, "Valid"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'action': self.action,
            'symbol': self.symbol,
            'size': self.size,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class Position:
    """확장된 포지션 클래스"""
    
    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.state = PositionState(symbol=symbol)
        self.trades: List[Trade] = []
        self.entry_trades: List[Trade] = []
        self.exit_trades: List[Trade] = []
        
    @property
    def size(self) -> float:
        return self.state.size
    
    @property
    def avg_price(self) -> float:
        return self.state.avg_entry_price
    
    @property
    def is_long(self) -> bool:
        return self.state.is_long
    
    @property
    def is_short(self) -> bool:
        return self.state.is_short
    
    @property
    def is_flat(self) -> bool:
        return self.state.is_flat
    
    @property
    def unrealized_pnl(self) -> float:
        return self.state.unrealized_pnl
    
    @property
    def realized_pnl(self) -> float:
        return self.state.realized_pnl
    
    def update_price(self, price: float):
        """가격 업데이트"""
        self.state.update_price(price)
    
    def add_trade(self, trade: Trade):
        """거래 추가"""
        self.trades.append(trade)
        
        if trade.side == OrderSide.BUY:
            self.entry_trades.append(trade)
        else:
            self.exit_trades.append(trade)
        
        # 포지션 상태 업데이트
        if trade.side == OrderSide.BUY:
            self.state.add_position(trade)
        else:
            self.state.close_position(trade)
    
    def get_position_info(self) -> Dict[str, Any]:
        """포지션 정보 반환"""
        info = self.state.to_dict()
        info.update({
            'total_trades': len(self.trades),
            'entry_trades': len(self.entry_trades),
            'exit_trades': len(self.exit_trades),
            'avg_trade_size': np.mean([t.size for t in self.trades]) if self.trades else 0,
            'total_volume': sum(t.value for t in self.trades),
            'holding_period': (datetime.now() - self.state.entry_timestamp).days if self.state.entry_timestamp else 0
        })
        return info
    
    def get_trade_history(self) -> pd.DataFrame:
        """거래 내역 DataFrame 반환"""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(trade_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').sort_index()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """포지션 성과 지표 계산"""
        if not self.trades:
            return {}
        
        df = self.get_trade_history()
        
        metrics = {
            'total_trades': len(self.trades),
            'total_volume': df['value'].sum(),
            'total_commission': df['commission'].sum(),
            'avg_trade_size': df['size'].mean(),
            'largest_trade': df['value'].max(),
            'smallest_trade': df['value'].min(),
        }
        
        # 수익성 분석
        if self.exit_trades:
            returns = []
            for exit_trade in self.exit_trades:
                # 해당 청산 거래의 수익률 계산
                entry_price = self.state.avg_entry_price  # 단순화
                exit_price = exit_trade.price
                trade_return = (exit_price - entry_price) / entry_price
                returns.append(trade_return)
            
            if returns:
                metrics.update({
                    'avg_return': np.mean(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'best_trade': max(returns),
                    'worst_trade': min(returns),
                    'volatility': np.std(returns)
                })
        
        return metrics