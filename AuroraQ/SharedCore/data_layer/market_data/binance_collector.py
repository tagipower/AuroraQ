"""
Binance API 실제 데이터 수집기
실시간 암호화폐 데이터 및 거래 실행
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import hmac
import hashlib
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


@dataclass
class OrderRequest:
    """주문 요청"""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"


@dataclass
class OrderResponse:
    """주문 응답"""
    order_id: int
    symbol: str
    status: str
    filled_qty: float
    avg_price: float
    commission: float
    timestamp: datetime


class BinanceCollector:
    """
    Binance API 실제 연결 및 데이터 수집
    - 실시간 시장 데이터
    - 계정 정보 조회
    - 주문 실행 (실제/테스트넷)
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API 엔드포인트 설정
        if base_url:
            self.base_url = base_url
        elif testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
            
        self.session = None
        self.server_time_offset = 0
        
        # API 제한 관리
        self.request_count = 0
        self.weight_count = 0
        self.last_reset_time = time.time()
        
    async def connect(self):
        """API 연결 초기화"""
        self.session = aiohttp.ClientSession()
        
        try:
            # 서버 시간 동기화
            await self._sync_server_time()
            
            # 계정 정보 확인
            account_info = await self.get_account_info()
            
            logger.info(f"Connected to Binance {'Testnet' if self.testnet else 'Mainnet'}")
            logger.info(f"Account status: {account_info.get('accountType', 'UNKNOWN')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            if self.session:
                await self.session.close()
            raise
    
    async def _sync_server_time(self):
        """서버 시간 동기화"""
        try:
            url = f"{self.base_url}/api/v3/time"
            async with self.session.get(url) as response:
                data = await response.json()
                server_time = data['serverTime']
                local_time = int(time.time() * 1000)
                self.server_time_offset = server_time - local_time
                
                logger.debug(f"Server time offset: {self.server_time_offset}ms")
                
        except Exception as e:
            logger.warning(f"Failed to sync server time: {e}")
            self.server_time_offset = 0
    
    def _get_timestamp(self) -> int:
        """동기화된 타임스탬프 반환"""
        return int(time.time() * 1000) + self.server_time_offset
    
    def _create_signature(self, query_string: str) -> str:
        """API 서명 생성"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """API 요청 실행"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
            
        # 서명 필요한 요청
        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._create_signature(query_string)
        
        # API 사용량 체크
        await self._check_rate_limits()
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
                    
        except Exception as e:
            logger.error(f"API request failed: {method} {endpoint} - {e}")
            raise
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """응답 처리"""
        # 사용량 헤더 업데이트
        if 'X-MBX-USED-WEIGHT-1M' in response.headers:
            self.weight_count = int(response.headers['X-MBX-USED-WEIGHT-1M'])
        
        if response.status == 200:
            return await response.json()
        else:
            error_data = await response.json()
            error_msg = error_data.get('msg', f"HTTP {response.status}")
            logger.error(f"API error: {error_msg}")
            raise Exception(f"Binance API error: {error_msg}")
    
    async def _check_rate_limits(self):
        """API 사용량 제한 체크"""
        current_time = time.time()
        
        # 1분마다 리셋
        if current_time - self.last_reset_time > 60:
            self.request_count = 0
            self.weight_count = 0
            self.last_reset_time = current_time
        
        # 요청 제한 체크 (1200/분)
        if self.request_count >= 1200:
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Weight 제한 체크 (6000/분)
        if self.weight_count >= 5500:  # 여유분 확보
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.warning(f"Weight limit approaching, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.request_count += 1
    
    async def get_market_data(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        K선 데이터 조회
        
        Args:
            symbol: 거래쌍 (예: BTCUSDT)
            interval: 시간간격 (1m, 5m, 15m, 1h, 4h, 1d)
            limit: 최대 개수 (최대 1000)
            start_time: 시작 시간
            end_time: 종료 시간
        """
        params = {
            'symbol': symbol.replace('/', ''),  # BTC/USDT -> BTCUSDT
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        data = await self._request('GET', '/api/v3/klines', params)
        
        # DataFrame 변환
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # 데이터 타입 변환
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')[numeric_columns]
        
        logger.debug(f"Retrieved {len(df)} {interval} candles for {symbol}")
        return df
    
    async def get_ticker_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        params = {'symbol': symbol.replace('/', '')}
        data = await self._request('GET', '/api/v3/ticker/price', params)
        return float(data['price'])
    
    async def get_account_info(self) -> Dict[str, Any]:
        """계정 정보 조회"""
        return await self._request('GET', '/api/v3/account', signed=True)
    
    async def get_balance(self, asset: str) -> Tuple[float, float]:
        """잔고 조회 (free, locked)"""
        account = await self.get_account_info()
        
        for balance in account['balances']:
            if balance['asset'] == asset:
                return float(balance['free']), float(balance['locked'])
        
        return 0.0, 0.0
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """주문 실행"""
        params = {
            'symbol': order.symbol.replace('/', ''),
            'side': order.side.value,
            'type': order.type.value,
            'quantity': str(order.quantity),
            'timeInForce': order.time_in_force
        }
        
        if order.price is not None:
            params['price'] = str(order.price)
        if order.stop_price is not None:
            params['stopPrice'] = str(order.stop_price)
        
        data = await self._request('POST', '/api/v3/order', params, signed=True)
        
        return OrderResponse(
            order_id=data['orderId'],
            symbol=data['symbol'],
            status=data['status'],
            filled_qty=float(data['executedQty']),
            avg_price=float(data.get('price', 0)) if data.get('price') else 0,
            commission=0.0,  # 별도 조회 필요
            timestamp=datetime.fromtimestamp(data['transactTime'] / 1000)
        )
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 취소"""
        params = {
            'symbol': symbol.replace('/', ''),
            'orderId': order_id
        }
        return await self._request('DELETE', '/api/v3/order', params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.replace('/', '')
        
        return await self._request('GET', '/api/v3/openOrders', params, signed=True)
    
    async def get_order_history(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """주문 내역 조회"""
        params = {
            'symbol': symbol.replace('/', ''),
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        
        return await self._request('GET', '/api/v3/allOrders', params, signed=True)
    
    async def get_trade_history(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """거래 내역 조회"""
        params = {
            'symbol': symbol.replace('/', ''),
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        
        return await self._request('GET', '/api/v3/myTrades', params, signed=True)
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """거래소 정보 조회"""
        return await self._request('GET', '/api/v3/exchangeInfo')
    
    async def get_24hr_stats(self, symbol: str) -> Dict[str, Any]:
        """24시간 통계"""
        params = {'symbol': symbol.replace('/', '')}
        return await self._request('GET', '/api/v3/ticker/24hr', params)
    
    async def close(self):
        """연결 종료"""
        if self.session:
            await self.session.close()
            logger.info("Binance connection closed")


# 팩토리 함수
def create_binance_collector(
    api_key: str,
    api_secret: str,
    testnet: bool = True
) -> BinanceCollector:
    """Binance 수집기 생성"""
    return BinanceCollector(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )