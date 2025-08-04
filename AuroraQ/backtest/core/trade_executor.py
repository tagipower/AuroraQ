#!/usr/bin/env python3
"""
거래 실행기 - 주문 처리 및 체결 시뮬레이션
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from .portfolio import Portfolio, Trade


@dataclass
class OrderSignal:
    """주문 신호"""
    action: str  # 'buy', 'sell', 'hold'
    size: float  # 주문 크기 (0-1: 자본 비율, >1: 절대값)
    order_type: str = 'market'  # 'market', 'limit'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    confidence: float = 0.5  # 신호 신뢰도
    reason: str = ""  # 신호 발생 이유


@dataclass
class MarketData:
    """시장 데이터"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def ohlc(self) -> Tuple[float, float, float, float]:
        return self.open, self.high, self.low, self.close


class TradeExecutor:
    """거래 실행기"""
    
    def __init__(self,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 min_order_size: float = 0.0001,
                 max_order_size: float = 1.0):
        self.commission = commission
        self.slippage = slippage
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        
        # 주문 추적
        self.pending_orders: List[Dict] = []
        self.executed_trades: List[Trade] = []
        
    def execute(self,
               signal: OrderSignal,
               market_data: MarketData,
               portfolio: Portfolio) -> Optional[Trade]:
        """
        주문 신호 실행
        
        Args:
            signal: 주문 신호
            market_data: 현재 시장 데이터
            portfolio: 포트폴리오
            
        Returns:
            실행된 거래 또는 None
        """
        if signal.action == 'hold':
            return None
        
        # 주문 크기 계산
        order_size = self._calculate_order_size(signal, portfolio, market_data)
        
        if order_size < self.min_order_size:
            return None
        
        # 주문 타입별 처리
        if signal.order_type == 'market':
            return self._execute_market_order(
                signal, order_size, market_data, portfolio
            )
        elif signal.order_type == 'limit':
            return self._execute_limit_order(
                signal, order_size, market_data, portfolio
            )
        else:
            raise ValueError(f"지원하지 않는 주문 타입: {signal.order_type}")
    
    def _calculate_order_size(self,
                            signal: OrderSignal,
                            portfolio: Portfolio,
                            market_data: MarketData) -> float:
        """주문 크기 계산"""
        
        if signal.size <= 1.0:
            # 비율로 지정된 경우
            if signal.action == 'buy':
                available_capital = portfolio.get_available_capital()
                order_value = available_capital * signal.size
                order_size = order_value / market_data.close
            else:  # sell
                current_position = portfolio.position.size
                order_size = current_position * signal.size
        else:
            # 절대값으로 지정된 경우
            order_size = signal.size
        
        # 최대 주문 크기 제한
        if signal.action == 'buy':
            max_buy_value = portfolio.get_available_capital() * self.max_order_size
            max_order_size = max_buy_value / market_data.close
            order_size = min(order_size, max_order_size)
        else:  # sell
            max_sell_size = portfolio.position.size * self.max_order_size
            order_size = min(order_size, max_sell_size)
        
        return max(0, order_size)
    
    def _execute_market_order(self,
                            signal: OrderSignal,
                            order_size: float,
                            market_data: MarketData,
                            portfolio: Portfolio) -> Optional[Trade]:
        """시장가 주문 실행"""
        
        # 슬리피지 적용
        execution_price = self._apply_slippage(
            market_data.close, signal.action, market_data
        )
        
        # 거래 실행
        trade_id = str(uuid.uuid4())[:8]
        
        try:
            if signal.action == 'buy':
                trade = portfolio.execute_buy(
                    size=order_size,
                    price=execution_price,
                    timestamp=market_data.timestamp,
                    trade_id=trade_id
                )
            else:  # sell
                trade = portfolio.execute_sell(
                    size=order_size,
                    price=execution_price,
                    timestamp=market_data.timestamp,
                    trade_id=trade_id
                )
            
            self.executed_trades.append(trade)
            return trade
            
        except ValueError as e:
            # 주문 실행 실패 (자본 부족, 보유량 부족 등)
            return None
    
    def _execute_limit_order(self,
                           signal: OrderSignal,
                           order_size: float,
                           market_data: MarketData,
                           portfolio: Portfolio) -> Optional[Trade]:
        """지정가 주문 실행"""
        
        if signal.limit_price is None:
            raise ValueError("지정가 주문에는 limit_price가 필요합니다")
        
        # 체결 조건 확인
        can_fill = False
        
        if signal.action == 'buy' and market_data.low <= signal.limit_price:
            can_fill = True
            execution_price = min(signal.limit_price, market_data.close)
        elif signal.action == 'sell' and market_data.high >= signal.limit_price:
            can_fill = True
            execution_price = max(signal.limit_price, market_data.close)
        
        if not can_fill:
            # 미체결 주문을 대기열에 추가
            self._add_pending_order(signal, order_size, market_data)
            return None
        
        # 체결 실행
        trade_id = str(uuid.uuid4())[:8]
        
        try:
            if signal.action == 'buy':
                trade = portfolio.execute_buy(
                    size=order_size,
                    price=execution_price,
                    timestamp=market_data.timestamp,
                    trade_id=trade_id
                )
            else:  # sell
                trade = portfolio.execute_sell(
                    size=order_size,
                    price=execution_price,
                    timestamp=market_data.timestamp,
                    trade_id=trade_id
                )
            
            self.executed_trades.append(trade)
            return trade
            
        except ValueError as e:
            return None
    
    def _apply_slippage(self,
                       price: float,
                       action: str,
                       market_data: MarketData) -> float:
        """슬리피지 적용"""
        
        # 기본 슬리피지
        base_slippage = self.slippage
        
        # 거래량 기반 슬리피지 조정
        if market_data.volume > 0:
            # 거래량이 적을수록 슬리피지 증가
            volume_factor = min(2.0, 1000000 / max(market_data.volume, 100000))
            adjusted_slippage = base_slippage * volume_factor
        else:
            adjusted_slippage = base_slippage * 2.0
        
        # 변동성 기반 슬리피지 조정
        if market_data.high > market_data.low:
            spread = (market_data.high - market_data.low) / market_data.close
            volatility_factor = 1 + spread * 2
            adjusted_slippage *= volatility_factor
        
        # 슬리피지 적용
        if action == 'buy':
            # 매수시 더 높은 가격으로 체결
            return price * (1 + adjusted_slippage)
        else:
            # 매도시 더 낮은 가격으로 체결
            return price * (1 - adjusted_slippage)
    
    def _add_pending_order(self,
                          signal: OrderSignal,
                          order_size: float,
                          market_data: MarketData):
        """미체결 주문 추가"""
        pending_order = {
            'signal': signal,
            'order_size': order_size,
            'created_at': market_data.timestamp,
            'order_id': str(uuid.uuid4())[:8]
        }
        self.pending_orders.append(pending_order)
    
    def process_pending_orders(self,
                             market_data: MarketData,
                             portfolio: Portfolio) -> List[Trade]:
        """미체결 주문 처리"""
        executed_trades = []
        orders_to_remove = []
        
        for i, order in enumerate(self.pending_orders):
            signal = order['signal']
            order_size = order['order_size']
            
            # 체결 조건 확인
            can_fill = False
            
            if signal.action == 'buy' and market_data.low <= signal.limit_price:
                can_fill = True
                execution_price = min(signal.limit_price, market_data.close)
            elif signal.action == 'sell' and market_data.high >= signal.limit_price:
                can_fill = True
                execution_price = max(signal.limit_price, market_data.close)
            
            if can_fill:
                # 체결 실행
                trade_id = order['order_id']
                
                try:
                    if signal.action == 'buy':
                        trade = portfolio.execute_buy(
                            size=order_size,
                            price=execution_price,
                            timestamp=market_data.timestamp,
                            trade_id=trade_id
                        )
                    else:  # sell
                        trade = portfolio.execute_sell(
                            size=order_size,
                            price=execution_price,
                            timestamp=market_data.timestamp,
                            trade_id=trade_id
                        )
                    
                    executed_trades.append(trade)
                    self.executed_trades.append(trade)
                    orders_to_remove.append(i)
                    
                except ValueError:
                    # 실행 실패시 주문 취소
                    orders_to_remove.append(i)
        
        # 처리된 주문 제거
        for i in reversed(orders_to_remove):
            self.pending_orders.pop(i)
        
        return executed_trades
    
    def cancel_all_pending_orders(self):
        """모든 미체결 주문 취소"""
        cancelled_count = len(self.pending_orders)
        self.pending_orders.clear()
        return cancelled_count
    
    def get_execution_stats(self) -> Dict:
        """실행 통계"""
        if not self.executed_trades:
            return {
                'total_trades': 0,
                'total_volume': 0.0,
                'total_commission': 0.0,
                'avg_trade_size': 0.0,
                'avg_slippage': 0.0
            }
        
        total_volume = sum(trade.value for trade in self.executed_trades)
        total_commission = sum(trade.commission for trade in self.executed_trades)
        avg_trade_size = total_volume / len(self.executed_trades)
        
        return {
            'total_trades': len(self.executed_trades),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'avg_trade_size': avg_trade_size,
            'avg_commission_rate': total_commission / max(total_volume, 1),
            'pending_orders': len(self.pending_orders)
        }
    
    def reset(self):
        """실행기 초기화"""
        self.pending_orders.clear()
        self.executed_trades.clear()