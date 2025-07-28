import logging
from typing import Optional, Dict, Any
from order.binance_order_client import BinanceOrderClient
from config.trade_config_loader import get_trade_config

# 포지션 파일 기반 공유 모듈 추가 (안전한 import)
try:
    from core.position_tracker_file import set_current_position
    POSITION_TRACKING_ENABLED = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("position_tracker_file module not found - position tracking disabled")
    POSITION_TRACKING_ENABLED = False
    
    def set_current_position(position: Optional[str]) -> bool:
        """Dummy function when module is not available"""
        return False

logger = logging.getLogger(__name__)

class OrderSignal:
    def __init__(
        self, 
        signal_type: str, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: Optional[float] = None
    ):
        """
        주문 신호 생성 클래스
        
        Args:
            signal_type: 'entry' 또는 'exit'
            side: 'BUY' 또는 'SELL'
            symbol: 거래 심볼 (예: 'BTCUSDT')
            quantity: 주문 수량
            price: 지정가 주문 시 가격 (시장가는 None)
        """
        self.signal_type = signal_type
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price

        self.config = get_trade_config()
        self.client = BinanceOrderClient()

    def execute(self, timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        주문 실행
        
        Args:
            timestamp: 주문 타임스탬프 (옵션)
            
        Returns:
            Dict: 주문 응답 또는 None (실패 시)
        """
        try:
            # 신호 타입에 따른 주문 유형 결정
            if self.signal_type == 'entry':
                order_type = self.config.get('order', {}).get('entry_type', 'MARKET')
            elif self.signal_type == 'exit':
                order_type = self.config.get('order', {}).get('exit_type', 'MARKET')
            else:
                raise ValueError(f"Unknown signal type: {self.signal_type}")

            # 주문 실행
            response = self.client.place_order(
                symbol=self.symbol,
                side=self.side,
                quantity=self.quantity,
                price=self.price,
                order_type=order_type
            )

            logger.info(f"[OrderSignal] Order executed successfully: {response}")

            # 체결 성공 시 포지션 상태 파일에 기록
            if POSITION_TRACKING_ENABLED and response:
                try:
                    if self.signal_type == 'entry':
                        position = "long" if self.side == "BUY" else "short"
                        set_current_position(position)
                        logger.debug(f"Position set to: {position}")
                    elif self.signal_type == 'exit':
                        set_current_position(None)
                        logger.debug("Position cleared")
                except Exception as e:
                    logger.warning(f"Failed to update position tracking: {e}")

            return response

        except ValueError as e:
            logger.error(f"[OrderSignal] Invalid parameters: {e}")
            return None
        except Exception as e:
            logger.error(f"[OrderSignal] Order execution failed: {e}", exc_info=True)
            return None
