"""
통합 데이터 제공자 - AuroraQ와 MacroQ가 읽기 전용으로 참조
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import aioredis
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logger import get_logger

logger = get_logger("UnifiedDataProvider")


@dataclass
class MarketData:
    """시장 데이터 표준 형식"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = None


class DataCollector(ABC):
    """데이터 수집기 기본 클래스"""
    
    @abstractmethod
    async def fetch(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        pass


class UnifiedDataProvider:
    """
    통합 데이터 레이어 v2.0 - 모드별 선택적 데이터 로딩
    
    모든 데이터는 캐싱되며, AuroraQ와 MacroQ는 이 인터페이스를 통해서만 데이터에 접근
    """
    
    def __init__(
        self, 
        redis_host: str = "localhost", 
        redis_port: int = 6379,
        use_crypto: bool = True,
        use_macro: bool = False
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.cache = None
        
        # 모드별 활성화 플래그
        self.use_crypto = use_crypto
        self.use_macro = use_macro
        
        # 수집기 및 로드된 모듈 추적
        self.collectors = {}
        self.loaded_data_types = set()
        
        # TTL 설정 (모드별 최적화)
        self.crypto_ttl = 60 if use_crypto else 0     # 1분 (실시간)
        self.macro_ttl = 3600 if use_macro else 0     # 1시간 (느린 업데이트)
        self.sentiment_ttl = 1800 if use_crypto else 0  # 30분 (감정분석)
        
        self._initialize_collectors()
        
    def _initialize_collectors(self):
        """모드별 선택적 데이터 수집기 초기화"""
        if self.use_crypto:
            # Binance 암호화폐 데이터 수집기 초기화
            logger.info("Initializing Binance crypto data collector...")
            self.loaded_data_types.add("crypto")
            
            # Binance 수집기 초기화
            try:
                import os
                from .market_data.binance_collector import create_binance_collector
                
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_API_SECRET') 
                testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
                
                if api_key and api_secret:
                    self.binance_collector = create_binance_collector(
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=testnet
                    )
                    logger.info(f"Binance collector initialized ({'testnet' if testnet else 'mainnet'})")
                else:
                    logger.warning("Binance API keys not found, using dummy data")
                    self.binance_collector = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize Binance collector: {e}")
                self.binance_collector = None
            
        if self.use_macro:
            # 거시경제 데이터 수집기 초기화
            logger.info("Initializing macro data collectors...")
            self.loaded_data_types.add("stocks")
            self.loaded_data_types.add("etf")
            self.loaded_data_types.add("macro_events")
            # 추후 구현: AlphaVantageCollector, FREDCollector
            
        logger.info(f"Data collectors initialized for: {', '.join(self.loaded_data_types)}")
        
    async def connect(self):
        """Redis 및 데이터 수집기 연결"""
        try:
            self.cache = await aioredis.create_redis_pool(
                f'redis://{self.redis_host}:{self.redis_port}'
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.cache = {}  # Fallback to dict
        
        # Binance 수집기 연결
        if hasattr(self, 'binance_collector') and self.binance_collector:
            try:
                await self.binance_collector.connect()
                logger.info("Binance collector connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect Binance collector: {e}")
                self.binance_collector = None
    
    async def get_market_data(
        self,
        asset_type: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        시장 데이터 조회 (읽기 전용) - 모드별 필터링 적용
        
        Args:
            asset_type: 'crypto', 'stock', 'etf', 'forex', 'bond'
            symbol: 심볼 (예: 'BTC/USDT', 'SPY', 'EUR/USD')
            timeframe: '1m', '5m', '1h', '1d' 등
            start_time: 시작 시간
            end_time: 종료 시간
            use_cache: 캐시 사용 여부
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        # 모드별 데이터 타입 검증
        if asset_type == "crypto" and not self.use_crypto:
            raise ValueError(f"Crypto data not available in current mode. Loaded types: {self.loaded_data_types}")
        
        if asset_type in ["stock", "etf", "bond"] and not self.use_macro:
            raise ValueError(f"Macro data not available in current mode. Loaded types: {self.loaded_data_types}")
        
        cache_key = f"{asset_type}:{symbol}:{timeframe}"
        
        # 캐시 확인
        if use_cache and self.cache:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
        
        # 실제 데이터 수집
        logger.info(f"Fetching {asset_type} data for {symbol}")
        
        # 암호화폐 데이터는 Binance에서 수집
        if asset_type == "crypto" and hasattr(self, 'binance_collector') and self.binance_collector:
            try:
                # Binance 심볼 변환 (BTC/USDT -> BTCUSDT)
                binance_symbol = symbol.replace('/', '')
                
                # 타임프레임 매핑
                timeframe_map = {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '1h', '4h': '4h', '1d': '1d'
                }
                binance_interval = timeframe_map.get(timeframe, '1h')
                
                # 제한 설정 (API 제한 고려)
                limit = min(500, 1000)  # 최대 1000개
                
                # Binance API 호출
                real_data = await self.binance_collector.get_market_data(
                    symbol=binance_symbol,
                    interval=binance_interval,
                    limit=limit,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # 모드별 TTL 적용
                ttl = self.crypto_ttl
                
                # 캐시 저장
                if self.cache and ttl > 0:
                    # DataFrame을 dict로 변환하여 저장
                    cache_data = {
                        'data': real_data.to_dict('records'),
                        'index': real_data.index.tolist(),
                        'columns': real_data.columns.tolist()
                    }
                    await self._save_to_cache(cache_key, cache_data, ttl=ttl)
                
                logger.info(f"Retrieved {len(real_data)} real data points for {symbol}")
                return real_data
                
            except Exception as e:
                logger.error(f"Failed to fetch real data from Binance: {e}")
                # 실패시 더미 데이터로 fallback
        
        # 더미 데이터 생성 (Binance 실패시 또는 다른 자산 타입)
        dummy_data = self._generate_dummy_data(symbol, timeframe, start_time, end_time)
        
        # 모드별 TTL 적용
        ttl = self.crypto_ttl if asset_type == "crypto" else self.macro_ttl
        
        # 캐시 저장
        if self.cache and ttl > 0:
            await self._save_to_cache(cache_key, dummy_data, ttl=ttl)
            
        return dummy_data
    
    async def get_sentiment_score(
        self,
        asset: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        감정 점수 조회 (읽기 전용) - AuroraQ 모드에서만 사용
        
        Returns:
            Dict: {'overall': 0.7, 'news': 0.8, 'social': 0.6}
        """
        # AuroraQ 모드에서만 감정분석 제공
        if not self.use_crypto:
            raise ValueError("Sentiment analysis not available in current mode")
            
        if timestamp is None:
            timestamp = datetime.now()
            
        cache_key = f"sentiment:{asset}:{timestamp.strftime('%Y%m%d%H')}"
        
        # 캐시 확인
        if self.cache:
            cached_sentiment = await self._get_from_cache(cache_key)
            if cached_sentiment:
                return cached_sentiment
        
        # 감정 분석 엔진에서 가져오기 (추후 구현)
        sentiment = {
            'overall': 0.5 + np.random.randn() * 0.1,  # 임시
            'news': 0.5 + np.random.randn() * 0.1,
            'social': 0.5 + np.random.randn() * 0.1
        }
        
        # 캐시 저장 (모드별 TTL)
        if self.cache and self.sentiment_ttl > 0:
            await self._save_to_cache(cache_key, sentiment, ttl=self.sentiment_ttl)
            
        return sentiment
    
    async def get_macro_events(
        self,
        start_date: datetime,
        end_date: datetime,
        importance: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        거시경제 이벤트 조회 (읽기 전용) - MacroQ 모드에서만 사용
        
        Args:
            start_date: 시작일
            end_date: 종료일
            importance: 'high', 'medium', 'low', 'all'
            
        Returns:
            List[Dict]: 이벤트 목록
        """
        # MacroQ 모드에서만 거시경제 이벤트 제공
        if not self.use_macro:
            raise ValueError("Macro events not available in current mode")
            
        cache_key = f"events:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}:{importance}"
        
        # 캐시 확인
        if self.cache:
            cached_events = await self._get_from_cache(cache_key)
            if cached_events:
                return cached_events
        
        # 이벤트 캘린더에서 가져오기 (추후 구현)
        events = [
            {
                'date': '2025-07-30',
                'time': '14:00',
                'event': 'FOMC Meeting',
                'importance': 'high',
                'forecast': None,
                'previous': None
            },
            {
                'date': '2025-08-05',
                'time': '08:30',
                'event': 'CPI Release',
                'importance': 'high',
                'forecast': 3.2,
                'previous': 3.1
            }
        ]
        
        # 중요도 필터링
        if importance != 'all':
            events = [e for e in events if e['importance'] == importance]
        
        # 캐시 저장 (모드별 TTL)
        if self.cache and self.macro_ttl > 0:
            await self._save_to_cache(cache_key, events, ttl=86400)  # 24시간 TTL
            
        return events
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        if isinstance(self.cache, dict):
            return self.cache.get(key)
        else:
            data = await self.cache.get(key)
            if data:
                return json.loads(data)
        return None
    
    async def _save_to_cache(self, key: str, data: Any, ttl: int = 300):
        """캐시에 데이터 저장"""
        if isinstance(self.cache, dict):
            self.cache[key] = data
        else:
            await self.cache.setex(key, ttl, json.dumps(data, default=str))
    
    def _generate_dummy_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> pd.DataFrame:
        """임시 더미 데이터 생성"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()
            
        # 타임프레임별 주기
        freq_map = {
            '1m': 'T',
            '5m': '5T',
            '15m': '15T',
            '1h': 'H',
            '4h': '4H',
            '1d': 'D'
        }
        
        freq = freq_map.get(timeframe, 'H')
        timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 랜덤 OHLCV 생성
        n = len(timestamps)
        base_price = 50000 if 'BTC' in symbol else 100
        returns = np.random.randn(n) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.randn(n) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
            'close': prices,
            'volume': np.random.randint(100, 10000, n)
        })
        
        return df
    
    async def close(self):
        """연결 종료"""
        # Binance 수집기 종료
        if hasattr(self, 'binance_collector') and self.binance_collector:
            await self.binance_collector.close()
            
        # Redis 연결 종료
        if self.cache and not isinstance(self.cache, dict):
            self.cache.close()
            await self.cache.wait_closed()
            logger.info("Data provider connections closed")