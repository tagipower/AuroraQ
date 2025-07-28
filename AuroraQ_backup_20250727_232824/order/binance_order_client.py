import os
import time
import yaml
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

# 환경 설정 로딩
with open('./config/trade_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ORDER_TYPE = config['order']['default_order_type']
SLIPPAGE = config['order'].get('slippage', 0.001)
FEE_RATES = config['order'].get('fee', {'market': 0.002, 'limit': 0.001})

logger = logging.getLogger("BinanceOrderClient")

class BinanceOrderClient:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)

    def execute_order(self, symbol: str, side: str, quantity: float, price: float = None, order_type: str = ORDER_TYPE):
        """
        바이낸스에 주문을 실행하는 함수입니다.
        :param symbol: BTCUSDT
        :param side: BUY or SELL
        :param quantity: 주문 수량
        :param price: 지정가 주문 시 가격
        :param order_type: 'market' or 'limit'
        :return: 체결 정보
        """
        try:
            if order_type == 'market':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            elif order_type == 'limit':
                assert price is not None, "Limit 주문 시 price는 필수입니다."
                adjusted_price = price * (1 + SLIPPAGE if side == 'BUY' else 1 - SLIPPAGE)
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=round(adjusted_price, 2)
                )
            else:
                raise ValueError(f"지원되지 않는 주문 유형: {order_type}")

            # 체결 정보 처리
            executed_price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else None
            commission = sum(float(fill.get('commission', 0)) for fill in order.get('fills', []))
            fee_rate = FEE_RATES.get(order_type, 0.002)
            
            return {
                'symbol': symbol,
                'side': side,
                'executed_price': executed_price,
                'quantity': quantity,
                'fee': commission if commission else quantity * executed_price * fee_rate,
                'status': order.get('status', 'UNKNOWN'),
                'order_id': order.get('orderId')
            }

        except (BinanceAPIException, BinanceOrderException, Exception) as e:
            logger.error(f"❌ 주문 실패 - {side} {symbol}: {e}")
            return {
                'symbol': symbol,
                'side': side,
                'executed_price': None,
                'quantity': quantity,
                'fee': 0,
                'status': 'FAILED',
                'error': str(e)
            }
