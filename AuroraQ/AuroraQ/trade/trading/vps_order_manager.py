#!/usr/bin/env python3
"""
VPS 주문 관리자 (통합 로깅 연동)
AuroraQ Production 주문 관리 시스템을 VPS 환경에 최적화
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uuid
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

# VPS 통합 로깅 시스템
from vps_logging import get_vps_log_integrator, LogCategory, LogLevel

class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"         # 대기 중
    SUBMITTED = "submitted"     # 제출됨
    PARTIALLY_FILLED = "partially_filled"  # 부분 체결
    FILLED = "filled"          # 완전 체결
    CANCELLED = "cancelled"    # 취소됨
    REJECTED = "rejected"      # 거부됨
    EXPIRED = "expired"        # 만료됨
    FAILED = "failed"          # 실패

class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"

class OrderPriority(Enum):
    """주문 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class VPSOrder:
    """VPS 최적화 주문 객체"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    # VPS 최적화 필드
    strategy: str = "hybrid"
    priority: OrderPriority = OrderPriority.NORMAL
    timeout_seconds: int = 300  # 5분 기본 타임아웃
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 실행 추적
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
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
    
    @property
    def is_expired(self) -> bool:
        """만료 여부 체크"""
        if self.timeout_seconds <= 0:
            return False
        
        check_time = self.submitted_at or self.created_at
        return datetime.now() > check_time + timedelta(seconds=self.timeout_seconds)
    
    @property
    def fill_percentage(self) -> float:
        """체결률"""
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'strategy': self.strategy,
            'priority': self.priority.value,
            'remaining_quantity': self.remaining_quantity,
            'fill_percentage': self.fill_percentage,
            'metadata': self.metadata
        }

class VPSOrderManager:
    """VPS 최적화 주문 관리자"""
    
    def __init__(self, enable_logging: bool = True):
        """
        VPS 주문 관리자 초기화
        
        Args:
            enable_logging: 통합 로깅 활성화
        """
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("vps_order_manager")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 주문 관리
        self.orders: Dict[str, VPSOrder] = {}
        self.order_history: List[VPSOrder] = []
        
        # 주문 큐 (우선순위별)
        self.order_queues = {
            OrderPriority.URGENT: [],
            OrderPriority.HIGH: [],
            OrderPriority.NORMAL: [],
            OrderPriority.LOW: []
        }
        
        # VPS 최적화 설정
        self.commission_rate = 0.001  # 0.1% 기본 수수료
        self.max_concurrent_orders = 10
        self.order_processing_delay = 0.1  # 100ms 처리 지연
        
        # 성능 통계
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "failed_orders": 0,
            "avg_fill_time": 0.0,
            "total_commission": 0.0,
            "success_rate": 0.0
        }
        
        # 실행 상태
        self.is_processing = False
        self.processing_task = None
    
    async def create_order(self,
                          symbol: str,
                          side: OrderSide,
                          order_type: OrderType,
                          quantity: float,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          strategy: str = "hybrid",
                          priority: OrderPriority = OrderPriority.NORMAL,
                          timeout_seconds: int = 300,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        주문 생성
        
        Args:
            symbol: 거래 심볼
            side: 주문 방향
            order_type: 주문 타입
            quantity: 수량
            price: 가격 (지정가용)
            stop_price: 스톱 가격
            strategy: 전략명
            priority: 우선순위
            timeout_seconds: 타임아웃 (초)
            metadata: 추가 메타데이터
            
        Returns:
            str: 주문 ID
        """
        try:
            # 주문 ID 생성
            order_id = f"vps_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            # 주문 객체 생성
            order = VPSOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                strategy=strategy,
                priority=priority,
                timeout_seconds=timeout_seconds,
                metadata=metadata or {}
            )
            
            # 주문 검증
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                order.status = OrderStatus.REJECTED
                
                if self.logger:
                    self.logger.error(f"Order validation failed: {validation_result['reason']}")
                
                # 거부 로깅
                if self.log_integrator:
                    await self.log_integrator.log_security_event(
                        event_type="order_rejected",
                        severity="medium",
                        description=f"Order rejected: {validation_result['reason']}",
                        order_id=order_id,
                        symbol=symbol,
                        reason=validation_result['reason']
                    )
                
                self.stats["rejected_orders"] += 1
                return order_id
            
            # 주문 저장
            self.orders[order_id] = order
            
            # 우선순위 큐에 추가
            self.order_queues[priority].append(order_id)
            
            # 통계 업데이트
            self.stats["total_orders"] += 1
            
            # 로깅
            if self.logger:
                self.logger.info(
                    f"Order created: {order_id} {symbol} {side.value} {quantity} @ {price or 'MARKET'}",
                    strategy=strategy,
                    priority=priority.name
                )
            
            # 주문 생성 로깅 (Raw 범주)
            if self.log_integrator:
                await self.log_integrator.log_api_request(
                    method="CREATE",
                    path="/orders",
                    status_code=200,
                    response_time=0.01,
                    order_id=order_id,
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    strategy=strategy
                )
            
            # 주문 처리 시작
            await self._start_order_processing()
            
            return order_id
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Order creation error: {e}")
            
            # 에러 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_creation_failed",
                    severity="high",
                    description=f"Failed to create order: {str(e)}",
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    error_details=str(e)
                )
            
            raise
    
    async def _validate_order(self, order: VPSOrder) -> Dict[str, Any]:
        """주문 검증"""
        try:
            # 기본 검증
            if order.quantity <= 0:
                return {"valid": False, "reason": "수량은 0보다 커야 합니다"}
            
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
                return {"valid": False, "reason": "지정가 주문은 가격이 필요합니다"}
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                return {"valid": False, "reason": "스톱 주문은 스톱 가격이 필요합니다"}
            
            # 동시 주문 수 제한
            active_orders = [o for o in self.orders.values() if o.is_active]
            if len(active_orders) >= self.max_concurrent_orders:
                return {"valid": False, "reason": f"최대 동시 주문 수 초과: {len(active_orders)}/{self.max_concurrent_orders}"}
            
            # 심볼별 중복 주문 체크 (같은 전략)
            same_strategy_orders = [
                o for o in active_orders 
                if o.symbol == order.symbol and o.strategy == order.strategy and o.side == order.side
            ]
            
            if len(same_strategy_orders) > 0:
                return {"valid": False, "reason": f"동일 전략 중복 주문: {order.symbol} {order.strategy}"}
            
            return {"valid": True, "reason": "검증 통과"}
            
        except Exception as e:
            return {"valid": False, "reason": f"검증 오류: {str(e)}"}
    
    async def _start_order_processing(self):
        """주문 처리 시작"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._process_orders())
    
    async def _process_orders(self):
        """주문 처리 루프"""
        try:
            while self.is_processing:
                # 우선순위별로 주문 처리
                processed = False
                
                for priority in [OrderPriority.URGENT, OrderPriority.HIGH, OrderPriority.NORMAL, OrderPriority.LOW]:
                    queue = self.order_queues[priority]
                    
                    if queue:
                        order_id = queue.pop(0)
                        if order_id in self.orders:
                            await self._execute_order(order_id)
                            processed = True
                            break
                
                # 만료된 주문 확인
                await self._check_expired_orders()
                
                # 처리할 주문이 없으면 잠시 대기
                if not processed:
                    await asyncio.sleep(self.order_processing_delay)
                
                # 모든 큐가 비어있으면 처리 중단
                if all(len(queue) == 0 for queue in self.order_queues.values()):
                    self.is_processing = False
                    break
                    
        except Exception as e:
            self.is_processing = False
            if self.logger:
                self.logger.error(f"Order processing error: {e}")
    
    async def _execute_order(self, order_id: str):
        """주문 실행"""
        try:
            if order_id not in self.orders:
                return
            
            order = self.orders[order_id]
            
            # 이미 처리된 주문은 건너뛰기
            if not order.is_active:
                return
            
            # 만료 체크
            if order.is_expired:
                await self._expire_order(order_id)
                return
            
            # 주문 제출
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # 실제 거래소 실행 시뮬레이션
            execution_result = await self._simulate_order_execution(order)
            
            if execution_result['success']:
                # 체결 성공
                await self._fill_order(order_id, execution_result)
            else:
                # 체결 실패
                await self._fail_order(order_id, execution_result['error'])
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Order execution error for {order_id}: {e}")
            
            # 실행 실패 처리
            await self._fail_order(order_id, str(e))
    
    async def _simulate_order_execution(self, order: VPSOrder) -> Dict[str, Any]:
        """주문 실행 시뮬레이션 (실제 환경에서는 거래소 API 호출)"""
        try:
            # 시뮬레이션 지연
            await asyncio.sleep(0.1)
            
            # 성공률 시뮬레이션 (95% 성공)
            import random
            if random.random() < 0.95:
                # 성공적인 체결
                slippage = random.uniform(-0.001, 0.001)  # ±0.1% 슬리피지
                
                if order.order_type == OrderType.MARKET:
                    # 시장가는 현재가 기준
                    execution_price = (order.price or 50000) * (1 + slippage)
                else:
                    # 지정가는 지정한 가격
                    execution_price = order.price
                
                return {
                    'success': True,
                    'filled_quantity': order.quantity,
                    'avg_fill_price': execution_price,
                    'commission': order.quantity * execution_price * self.commission_rate,
                    'execution_time': datetime.now()
                }
            else:
                # 실행 실패
                return {
                    'success': False,
                    'error': 'Insufficient liquidity'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _fill_order(self, order_id: str, execution_result: Dict[str, Any]):
        """주문 체결 처리"""
        try:
            order = self.orders[order_id]
            
            # 체결 정보 업데이트
            order.filled_quantity = execution_result['filled_quantity']
            order.avg_fill_price = execution_result['avg_fill_price']
            order.commission = execution_result['commission']
            order.status = OrderStatus.FILLED
            order.filled_at = execution_result['execution_time']
            
            # 통계 업데이트
            self.stats["filled_orders"] += 1
            self.stats["total_commission"] += order.commission
            
            # 평균 체결 시간 업데이트
            fill_time = (order.filled_at - order.created_at).total_seconds()
            total_filled = self.stats["filled_orders"]
            current_avg = self.stats["avg_fill_time"]
            self.stats["avg_fill_time"] = ((current_avg * (total_filled - 1)) + fill_time) / total_filled
            
            # 성공률 업데이트
            total_orders = self.stats["total_orders"]
            self.stats["success_rate"] = self.stats["filled_orders"] / total_orders if total_orders > 0 else 0
            
            # 히스토리에 추가
            self.order_history.append(order)
            
            # 로깅
            if self.logger:
                self.logger.info(
                    f"Order filled: {order_id} {order.symbol} {order.filled_quantity} @ {order.avg_fill_price:.4f}",
                    commission=order.commission,
                    fill_time=fill_time
                )
            
            # 체결 로깅 (Tagged 범주 - 중요 이벤트)
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_filled",
                    severity="medium",
                    description=f"Order filled: {order.symbol} {order.filled_quantity} @ {order.avg_fill_price:.4f}",
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    filled_quantity=order.filled_quantity,
                    avg_fill_price=order.avg_fill_price,
                    commission=order.commission,
                    strategy=order.strategy,
                    fill_time_seconds=fill_time
                )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fill order error for {order_id}: {e}")
    
    async def _fail_order(self, order_id: str, error_reason: str):
        """주문 실패 처리"""
        try:
            order = self.orders[order_id]
            
            # 재시도 가능한지 확인
            if order.retry_count < order.max_retries:
                order.retry_count += 1
                order.status = OrderStatus.PENDING
                
                # 우선순위 큐에 다시 추가 (높은 우선순위로)
                retry_priority = OrderPriority.HIGH if order.priority == OrderPriority.NORMAL else order.priority
                self.order_queues[retry_priority].append(order_id)
                
                if self.logger:
                    self.logger.warning(f"Order retry {order.retry_count}/{order.max_retries}: {order_id}")
                
                return
            
            # 최대 재시도 초과 시 실패 처리
            order.status = OrderStatus.FAILED
            
            # 통계 업데이트
            self.stats["failed_orders"] += 1
            
            # 로깅
            if self.logger:
                self.logger.error(f"Order failed: {order_id} - {error_reason}")
            
            # 실패 로깅 (Tagged 범주)
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_failed",
                    severity="high",
                    description=f"Order failed after {order.retry_count} retries: {error_reason}",
                    order_id=order_id,
                    symbol=order.symbol,
                    error_reason=error_reason,
                    retry_count=order.retry_count
                )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fail order error for {order_id}: {e}")
    
    async def _expire_order(self, order_id: str):
        """주문 만료 처리"""
        try:
            order = self.orders[order_id]
            order.status = OrderStatus.EXPIRED
            
            # 로깅
            if self.logger:
                self.logger.warning(f"Order expired: {order_id}")
            
            # 만료 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_expired",
                    severity="medium",
                    description=f"Order expired: {order_id}",
                    order_id=order_id,
                    symbol=order.symbol,
                    timeout_seconds=order.timeout_seconds
                )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Expire order error for {order_id}: {e}")
    
    async def _check_expired_orders(self):
        """만료된 주문 확인"""
        try:
            current_time = datetime.now()
            expired_orders = []
            
            for order_id, order in self.orders.items():
                if order.is_active and order.is_expired:
                    expired_orders.append(order_id)
            
            for order_id in expired_orders:
                await self._expire_order(order_id)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Check expired orders error: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소
        
        Args:
            order_id: 주문 ID
            
        Returns:
            bool: 취소 성공 여부
        """
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            
            # 취소 가능한 상태인지 확인
            if not order.is_active:
                return False
            
            # 주문 상태 업데이트
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()
            
            # 큐에서 제거
            for queue in self.order_queues.values():
                if order_id in queue:
                    queue.remove(order_id)
                    break
            
            # 통계 업데이트
            self.stats["cancelled_orders"] += 1
            
            # 로깅
            if self.logger:
                self.logger.info(f"Order cancelled: {order_id}")
            
            # 취소 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_cancelled",
                    severity="low",
                    description=f"Order manually cancelled: {order_id}",
                    order_id=order_id,
                    symbol=order.symbol
                )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cancel order error for {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        모든 주문 취소 (또는 특정 심볼)
        
        Args:
            symbol: 특정 심볼 (None이면 모든 주문)
            
        Returns:
            int: 취소된 주문 수
        """
        try:
            cancelled_count = 0
            orders_to_cancel = []
            
            # 취소할 주문 목록 생성
            for order_id, order in self.orders.items():
                if order.is_active:
                    if symbol is None or order.symbol == symbol:
                        orders_to_cancel.append(order_id)
            
            # 주문 취소 실행
            for order_id in orders_to_cancel:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            
            # 전체 취소 로깅
            if cancelled_count > 0 and self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="bulk_order_cancellation",
                    severity="medium",
                    description=f"Bulk cancellation: {cancelled_count} orders cancelled",
                    symbol=symbol,
                    cancelled_count=cancelled_count
                )
            
            return cancelled_count
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cancel all orders error: {e}")
            return 0
    
    def get_order(self, order_id: str) -> Optional[VPSOrder]:
        """주문 정보 조회"""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[VPSOrder]:
        """활성 주문 목록 조회"""
        active_orders = [order for order in self.orders.values() if order.is_active]
        
        if symbol:
            active_orders = [order for order in active_orders if order.symbol == symbol]
        
        return active_orders
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[VPSOrder]:
        """주문 히스토리 조회"""
        history = self.order_history.copy()
        
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        
        # 최신순 정렬
        history.sort(key=lambda x: x.created_at, reverse=True)
        
        return history[:limit]
    
    def get_order_stats(self) -> Dict[str, Any]:
        """주문 통계 조회"""
        stats = self.stats.copy()
        
        # 추가 통계 계산
        stats.update({
            "active_orders": len([o for o in self.orders.values() if o.is_active]),
            "pending_orders": len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            "queue_sizes": {
                priority.name: len(queue) 
                for priority, queue in self.order_queues.items()
            },
            "processing_status": self.is_processing
        })
        
        return stats
    
    async def get_order_summary(self) -> Dict[str, Any]:
        """주문 요약 정보"""
        try:
            active_orders = self.get_active_orders()
            
            summary = {
                "total_orders": self.stats["total_orders"],
                "active_orders": len(active_orders),
                "filled_orders": self.stats["filled_orders"],
                "cancelled_orders": self.stats["cancelled_orders"],
                "failed_orders": self.stats["failed_orders"],
                "success_rate": self.stats["success_rate"],
                "avg_fill_time": self.stats["avg_fill_time"],
                "total_commission": self.stats["total_commission"],
                "processing_status": self.is_processing,
                "order_details": []
            }
            
            # 활성 주문 상세 정보
            for order in active_orders:
                summary["order_details"].append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "created_at": order.created_at.isoformat(),
                    "strategy": order.strategy,
                    "priority": order.priority.name
                })
            
            return summary
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Order summary error: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """주문 관리자 종료"""
        try:
            # 처리 중단
            self.is_processing = False
            
            # 처리 태스크 취소
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # 모든 활성 주문 취소
            cancelled_count = await self.cancel_all_orders()
            
            if self.logger:
                self.logger.info(f"Order manager shutdown: {cancelled_count} orders cancelled")
            
            # 종료 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="order_manager_shutdown",
                    severity="medium",
                    description="Order manager shutdown completed",
                    cancelled_orders=cancelled_count,
                    final_stats=self.stats
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Order manager shutdown error: {e}")

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_order_manager() -> VPSOrderManager:
    """VPS 최적화된 주문 관리자 생성"""
    return VPSOrderManager(enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_order_manager():
        manager = create_vps_order_manager()
        
        # 주문 생성 테스트
        order_id = await manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            strategy="test",
            priority=OrderPriority.HIGH
        )
        
        print(f"Order created: {order_id}")
        
        # 주문 처리 대기
        await asyncio.sleep(2)
        
        # 주문 상태 확인
        order = manager.get_order(order_id)
        if order:
            print(f"Order status: {order.status.value}")
            print(f"Order details: {order.to_dict()}")
        
        # 통계 확인
        stats = manager.get_order_stats()
        print("Order stats:", json.dumps(stats, indent=2))
        
        # 요약 정보
        summary = await manager.get_order_summary()
        print("Order summary:", json.dumps(summary, indent=2, default=str))
        
        # 종료
        await manager.shutdown()
    
    asyncio.run(test_order_manager())