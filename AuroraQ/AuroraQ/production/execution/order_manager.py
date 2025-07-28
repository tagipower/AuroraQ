#!/usr/bin/env python3
"""
주문 관리자
주문 생성, 추적, 체결 관리
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("OrderManager")

class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """주문 객체"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def remaining_quantity(self) -> float:
        """미체결 수량"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """완전 체결 여부"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """활성 주문 여부"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]

class OrderManager:
    """주문 관리자"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.commission_rate = 0.001  # 0.1% 기본 수수료
    
    def create_order(self, 
                    symbol: str,
                    side: OrderSide, 
                    order_type: OrderType,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """주문 생성"""
        
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # 주문 검증
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.error(f"주문 검증 실패: {order_id}")
            return order
        
        self.orders[order_id] = order
        logger.info(f"주문 생성: {order_id} {side.value} {quantity} {symbol} @ {price}")
        
        return order
    
    def submit_order(self, order_id: str) -> bool:
        """주문 제출"""
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"주문을 찾을 수 없음: {order_id}")
            return False
        
        if order.status != OrderStatus.PENDING:
            logger.error(f"주문 상태 오류: {order_id} - {order.status}")
            return False
        
        order.status = OrderStatus.SUBMITTED
        logger.info(f"주문 제출: {order_id}")
        
        return True
    
    def fill_order(self, 
                  order_id: str, 
                  fill_quantity: float, 
                  fill_price: float) -> bool:
        """주문 체결 처리"""
        
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"주문을 찾을 수 없음: {order_id}")
            return False
        
        if not order.is_active:
            logger.error(f"비활성 주문: {order_id} - {order.status}")
            return False
        
        if fill_quantity > order.remaining_quantity:
            logger.error(f"체결 수량 초과: {order_id}")
            return False
        
        # 평균 체결 가격 계산
        total_filled_value = order.filled_quantity * order.avg_fill_price
        new_filled_value = fill_quantity * fill_price
        total_quantity = order.filled_quantity + fill_quantity
        
        order.avg_fill_price = (total_filled_value + new_filled_value) / total_quantity
        order.filled_quantity += fill_quantity
        
        # 수수료 계산
        commission = fill_quantity * fill_price * self.commission_rate
        order.commission += commission
        
        # 상태 업데이트
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            logger.info(f"주문 완전 체결: {order_id}")
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(f"주문 부분 체결: {order_id} - {fill_quantity}/{order.quantity}")
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        order = self.orders.get(order_id)
        if not order:
            logger.error(f"주문을 찾을 수 없음: {order_id}")
            return False
        
        if not order.is_active:
            logger.error(f"취소 불가능한 주문: {order_id} - {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"주문 취소: {order_id}")
        
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """주문 조회"""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """활성 주문 목록 조회"""
        active_orders = [order for order in self.orders.values() if order.is_active]
        
        if symbol:
            active_orders = [order for order in active_orders if order.symbol == symbol]
        
        return active_orders
    
    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """체결된 주문 목록 조회"""
        filled_orders = [order for order in self.orders.values() if order.is_filled]
        
        if symbol:
            filled_orders = [order for order in filled_orders if order.symbol == symbol]
        
        return filled_orders
    
    def _validate_order(self, order: Order) -> bool:
        """주문 검증"""
        
        # 기본 검증
        if order.quantity <= 0:
            logger.error("주문 수량은 0보다 커야 합니다")
            return False
        
        # 가격 검증
        if order.order_type == OrderType.LIMIT and (not order.price or order.price <= 0):
            logger.error("지정가 주문은 유효한 가격이 필요합니다")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                logger.error("스톱 주문은 유효한 스톱 가격이 필요합니다")
                return False
        
        return True
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """주문 통계 조회"""
        total_orders = len(self.orders)
        active_orders = len(self.get_active_orders())
        filled_orders = len(self.get_filled_orders())
        
        total_volume = sum(order.filled_quantity * order.avg_fill_price 
                          for order in self.orders.values() if order.is_filled)
        
        total_commission = sum(order.commission for order in self.orders.values())
        
        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]),
            "rejected_orders": len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED]),
            "total_volume": total_volume,
            "total_commission": total_commission,
            "avg_fill_rate": filled_orders / total_orders if total_orders > 0 else 0
        }