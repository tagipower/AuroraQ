#!/usr/bin/env python3
"""
VPS 시장 데이터 제공자 (통합 로깅 연동)
실시간 시장 데이터 수집 및 캐싱을 VPS 환경에 최적화
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from enum import Enum

# VPS 통합 로깅 시스템
from vps_logging import get_vps_log_integrator, LogCategory, LogLevel

class DataSource(Enum):
    """데이터 소스"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    SIMULATION = "simulation"

class DataType(Enum):
    """데이터 타입"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINES = "klines"

@dataclass
class VPSMarketDataPoint:
    """VPS 최적화 시장 데이터 포인트"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    
    # VPS 최적화 필드
    source: DataSource = DataSource.SIMULATION
    data_type: DataType = DataType.TICKER
    latency_ms: float = 0.0
    quality_score: float = 1.0  # 0-1, 데이터 품질
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def spread(self) -> float:
        """매수-매도 스프레드"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """중간 가격"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'source': self.source.value,
            'data_type': self.data_type.value,
            'latency_ms': self.latency_ms,
            'quality_score': self.quality_score,
            'metadata': self.metadata
        }

class VPSMarketDataProvider:
    """VPS 최적화 시장 데이터 제공자"""
    
    def __init__(self, 
                 exchange: str = "binance",
                 mode: str = "simulation",
                 enable_logging: bool = True):
        """
        VPS 시장 데이터 제공자 초기화
        
        Args:
            exchange: 거래소 이름
            mode: 모드 (simulation, live)
            enable_logging: 통합 로깅 활성화
        """
        self.exchange = exchange
        self.mode = mode
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("vps_market_data")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 데이터 소스 매핑
        self.data_source = self._get_data_source(exchange)
        
        # VPS 최적화 설정
        self.cache_ttl_seconds = 5  # 5초 캐시
        self.max_cache_size = 1000
        self.connection_timeout = 10
        self.request_timeout = 5
        
        # 데이터 캐시
        self.price_cache: Dict[str, VPSMarketDataPoint] = {}
        self.historical_data: Dict[str, deque] = {}
        
        # 성능 통계
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0.0,
            "data_quality_avg": 0.0,
            "last_update": None
        }
        
        # HTTP 세션
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API 엔드포인트
        self.api_endpoints = self._setup_api_endpoints()
        
        # 실시간 연결 상태
        self.connected = False
        self.last_heartbeat = None
    
    def _get_data_source(self, exchange: str) -> DataSource:
        """거래소명을 데이터 소스로 변환"""
        mapping = {
            "binance": DataSource.BINANCE,
            "coinbase": DataSource.COINBASE,
            "kraken": DataSource.KRAKEN,
            "simulation": DataSource.SIMULATION
        }
        return mapping.get(exchange, DataSource.SIMULATION)
    
    def _setup_api_endpoints(self) -> Dict[str, str]:
        """API 엔드포인트 설정"""
        if self.data_source == DataSource.BINANCE:
            return {
                "ticker": "https://api.binance.com/api/v3/ticker/24hr",
                "price": "https://api.binance.com/api/v3/ticker/price",
                "orderbook": "https://api.binance.com/api/v3/depth",
                "klines": "https://api.binance.com/api/v3/klines"
            }
        elif self.data_source == DataSource.COINBASE:
            return {
                "ticker": "https://api.coinbase.com/v2/exchange-rates",
                "price": "https://api.coinbase.com/v2/prices/{symbol}/spot",
                "orderbook": "https://api.coinbase.com/v2/products/{symbol}/book"
            }
        else:
            # 시뮬레이션 또는 기타
            return {}
    
    async def initialize(self):
        """데이터 제공자 초기화"""
        try:
            # HTTP 세션 생성
            timeout = aiohttp.ClientTimeout(
                total=self.connection_timeout,
                connect=self.request_timeout
            )
            
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # 연결 테스트
            if self.mode == "live":
                await self._test_connection()
            
            self.connected = True
            
            if self.logger:
                self.logger.info(f"Market data provider initialized: {self.exchange} ({self.mode})")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Market data provider initialization failed: {e}")
            
            # 초기화 실패 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="market_data_init_failed",
                    severity="high",
                    description=f"Market data provider initialization failed: {str(e)}",
                    exchange=self.exchange,
                    mode=self.mode,
                    error_details=str(e)
                )
            
            return False
    
    async def _test_connection(self):
        """연결 테스트"""
        try:
            if self.data_source == DataSource.BINANCE:
                # Binance 서버 시간 확인
                async with self.session.get("https://api.binance.com/api/v3/time") as response:
                    if response.status == 200:
                        data = await response.json()
                        server_time = data.get('serverTime', 0)
                        
                        if self.logger:
                            self.logger.info(f"Binance connection test successful, server time: {server_time}")
                    else:
                        raise Exception(f"Connection test failed: HTTP {response.status}")
            
        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")
    
    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        현재 가격 조회
        
        Args:
            symbol: 거래 심볼 (예: BTCUSDT)
            
        Returns:
            Optional[Dict[str, Any]]: 가격 정보
        """
        try:
            start_time = time.time()
            self.stats["total_requests"] += 1
            
            # 캐시 확인
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                self.stats["cache_hits"] += 1
                
                # 캐시 히트 로깅
                if self.log_integrator:
                    await self.log_integrator.log_system_metrics(
                        component="market_data_cache",
                        metrics={
                            "cache_hit": 1,
                            "symbol": symbol,
                            "latency_ms": (time.time() - start_time) * 1000
                        }
                    )
                
                return cached_data.to_dict()
            
            # 캐시 미스
            self.stats["cache_misses"] += 1
            
            # 실제 데이터 조회 (Fallback 체인 적용)
            market_data = await self._fetch_with_fallback_chain(symbol)
            
            if market_data:
                # 캐시에 저장
                self._cache_data(symbol, market_data)
                
                # 히스토리에 추가
                self._add_to_history(symbol, market_data)
                
                # 성능 통계 업데이트
                latency = (time.time() - start_time) * 1000
                await self._update_performance_stats(latency, market_data.quality_score)
                
                return market_data.to_dict()
            else:
                self.stats["failed_requests"] += 1
                return None
                
        except Exception as e:
            self.stats["failed_requests"] += 1
            
            if self.logger:
                self.logger.error(f"Get current price error for {symbol}: {e}")
            
            # 에러 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="market_data_fetch_failed",
                    severity="medium",
                    description=f"Failed to fetch market data: {str(e)}",
                    symbol=symbol,
                    exchange=self.exchange,
                    error_details=str(e)
                )
            
            return None
    
    def _get_cached_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """캐시된 데이터 조회"""
        if symbol in self.price_cache:
            cached_data = self.price_cache[symbol]
            
            # TTL 확인
            age_seconds = (datetime.now() - cached_data.timestamp).total_seconds()
            if age_seconds <= self.cache_ttl_seconds:
                return cached_data
            else:
                # 만료된 캐시 제거
                del self.price_cache[symbol]
        
        return None
    
    def _cache_data(self, symbol: str, data: VPSMarketDataPoint):
        """데이터 캐시 저장"""
        # 캐시 크기 제한
        if len(self.price_cache) >= self.max_cache_size:
            # 가장 오래된 데이터 제거
            oldest_symbol = min(self.price_cache.keys(), 
                               key=lambda k: self.price_cache[k].timestamp)
            del self.price_cache[oldest_symbol]
        
        self.price_cache[symbol] = data
    
    def _add_to_history(self, symbol: str, data: VPSMarketDataPoint):
        """히스토리 데이터 추가"""
        if symbol not in self.historical_data:
            self.historical_data[symbol] = deque(maxlen=1000)  # 최대 1000개 보관
        
        self.historical_data[symbol].append(data)
    
    async def _simulate_market_data(self, symbol: str) -> VPSMarketDataPoint:
        """시장 데이터 시뮬레이션"""
        try:
            # 간단한 가격 시뮬레이션
            base_price = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 1.0
            
            # 랜덤 변동 (±2%)
            import random
            price_change = random.uniform(-0.02, 0.02)
            current_price = base_price * (1 + price_change)
            
            # 스프레드 시뮬레이션 (0.01-0.1%)
            spread_pct = random.uniform(0.0001, 0.001)
            spread = current_price * spread_pct
            
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2
            
            # 볼륨 시뮬레이션
            volume = random.uniform(1000, 10000)
            
            # 24시간 데이터 시뮬레이션
            high_24h = current_price * random.uniform(1.0, 1.05)
            low_24h = current_price * random.uniform(0.95, 1.0)
            change_24h = random.uniform(-0.1, 0.1)
            
            return VPSMarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=volume,
                bid=bid_price,
                ask=ask_price,
                high_24h=high_24h,
                low_24h=low_24h,
                change_24h=change_24h,
                source=DataSource.SIMULATION,
                data_type=DataType.TICKER,
                latency_ms=random.uniform(10, 50),
                quality_score=random.uniform(0.9, 1.0)
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Market data simulation error: {e}")
            raise
    
    async def _fetch_live_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """실제 시장 데이터 조회"""
        try:
            if not self.session:
                raise Exception("HTTP session not initialized")
            
            start_time = time.time()
            
            if self.data_source == DataSource.BINANCE:
                # Binance API 호출
                url = self.api_endpoints["ticker"]
                params = {"symbol": symbol}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Binance 응답 파싱
                        return VPSMarketDataPoint(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            price=float(data.get('lastPrice', 0)),
                            volume=float(data.get('volume', 0)),
                            bid=float(data.get('bidPrice', 0)),
                            ask=float(data.get('askPrice', 0)),
                            high_24h=float(data.get('highPrice', 0)),
                            low_24h=float(data.get('lowPrice', 0)),
                            change_24h=float(data.get('priceChangePercent', 0)) / 100,
                            source=DataSource.BINANCE,
                            data_type=DataType.TICKER,
                            latency_ms=(time.time() - start_time) * 1000,
                            quality_score=1.0,
                            metadata={"count": data.get('count', 0)}
                        )
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            else:
                # 기타 거래소 구현
                raise Exception(f"Unsupported exchange: {self.data_source}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Live data fetch error for {symbol}: {e}")
            raise
    
    async def _fetch_with_fallback_chain(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """Fallback 체인을 통한 데이터 조회"""
        fallback_methods = [
            ("primary", self._fetch_primary_data),
            ("backup_api", self._fetch_backup_api_data),
            ("cached_fallback", self._fetch_cached_fallback_data),
            ("simulation", self._fetch_simulation_fallback_data)
        ]
        
        for method_name, method in fallback_methods:
            try:
                if self.logger:
                    self.logger.debug(f"Trying {method_name} for {symbol}")
                
                result = await method(symbol)
                if result:
                    if method_name != "primary":
                        self.stats[f"fallback_{method_name}_used"] = self.stats.get(f"fallback_{method_name}_used", 0) + 1
                        if self.logger:
                            self.logger.warning(f"Using fallback method: {method_name} for {symbol}")
                    return result
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Fallback method {method_name} failed for {symbol}: {e}")
                continue
        
        # 모든 fallback 실패
        if self.logger:
            self.logger.error(f"All fallback methods failed for {symbol}")
        return None
    
    async def _fetch_primary_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """기본 데이터 조회 방법"""
        if self.mode == "simulation":
            return await self._simulate_market_data(symbol)
        else:
            return await self._fetch_live_data(symbol)
    
    async def _fetch_backup_api_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """백업 API를 통한 데이터 조회"""
        try:
            if self.mode == "simulation":
                return None  # 시뮬레이션 모드에서는 백업 API 사용 안 함
            
            # 백업 API 엔드포인트 (예: CoinGecko, CryptoCompare)
            backup_url = f"https://api.coingecko.com/api/v3/simple/price"
            
            # 심볼 변환 (BTCUSDT -> bitcoin)
            symbol_map = {
                "BTCUSDT": "bitcoin",
                "ETHUSDT": "ethereum",
                "BNBUSDT": "binancecoin"
            }
            
            coingecko_id = symbol_map.get(symbol, symbol.lower().replace("usdt", ""))
            
            if not self.session:
                return None
            
            params = {
                "ids": coingecko_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            
            async with self.session.get(backup_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if coingecko_id in data:
                        price_data = data[coingecko_id]
                        price = price_data.get("usd", 0)
                        change_24h = price_data.get("usd_24h_change", 0) / 100
                        
                        return VPSMarketDataPoint(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            price=float(price),
                            volume=0.0,  # CoinGecko 기본 API에서는 볼륨 정보 없음
                            bid=float(price) * 0.999,  # 추정값
                            ask=float(price) * 1.001,  # 추정값
                            high_24h=float(price) * (1 + abs(change_24h) * 0.5),  # 추정값
                            low_24h=float(price) * (1 - abs(change_24h) * 0.5),   # 추정값
                            change_24h=change_24h,
                            source=DataSource.BINANCE,  # 원래 소스 유지
                            data_type=DataType.TICKER,
                            latency_ms=0,
                            quality_score=0.7,  # 백업 데이터이므로 품질 점수 낮춤
                            metadata={"backup_source": "coingecko", "estimated": True}
                        )
            
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Backup API fallback failed for {symbol}: {e}")
            return None
    
    async def _fetch_cached_fallback_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """만료된 캐시 데이터를 fallback으로 사용"""
        try:
            if symbol in self.price_cache:
                cached_data = self.price_cache[symbol]
                age_minutes = (datetime.now() - cached_data.timestamp).total_seconds() / 60
                
                # 30분 이내의 만료된 캐시는 fallback으로 사용 가능
                if age_minutes <= 30:
                    # 품질 점수를 시간에 따라 감소
                    quality_reduction = min(0.5, age_minutes / 60)  # 최대 50% 감소
                    
                    fallback_data = VPSMarketDataPoint(
                        symbol=cached_data.symbol,
                        timestamp=datetime.now(),  # 현재 시간으로 업데이트
                        price=cached_data.price,
                        volume=cached_data.volume,
                        bid=cached_data.bid,
                        ask=cached_data.ask,
                        high_24h=cached_data.high_24h,
                        low_24h=cached_data.low_24h,
                        change_24h=cached_data.change_24h,
                        source=cached_data.source,
                        data_type=cached_data.data_type,
                        latency_ms=0,
                        quality_score=max(0.3, cached_data.quality_score - quality_reduction),
                        metadata={
                            **cached_data.metadata,
                            "fallback_source": "expired_cache",
                            "cache_age_minutes": round(age_minutes, 1)
                        }
                    )
                    
                    if self.logger:
                        self.logger.info(f"Using expired cache as fallback for {symbol} (age: {age_minutes:.1f}min)")
                    
                    return fallback_data
            
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Cached fallback failed for {symbol}: {e}")
            return None
    
    async def _fetch_simulation_fallback_data(self, symbol: str) -> Optional[VPSMarketDataPoint]:
        """시뮬레이션 데이터를 최후의 fallback으로 사용"""
        try:
            # 기존 시뮬레이션 데이터에 fallback 표시 추가
            sim_data = await self._simulate_market_data(symbol)
            if sim_data:
                sim_data.metadata.update({
                    "fallback_source": "simulation",
                    "fallback_reason": "all_methods_failed"
                })
                sim_data.quality_score = 0.2  # 최소 품질 점수
                
                if self.logger:
                    self.logger.warning(f"Using simulation fallback for {symbol}")
            
            return sim_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Simulation fallback failed for {symbol}: {e}")
            return None
    
    async def _update_performance_stats(self, latency_ms: float, quality_score: float):
        """성능 통계 업데이트"""
        try:
            # 평균 지연시간 업데이트
            total_requests = self.stats["total_requests"]
            current_avg = self.stats["avg_latency_ms"]
            
            if total_requests == 1:
                self.stats["avg_latency_ms"] = latency_ms
            else:
                # 지수 이동 평균
                alpha = 0.1
                self.stats["avg_latency_ms"] = alpha * latency_ms + (1 - alpha) * current_avg
            
            # 평균 데이터 품질 업데이트
            current_quality_avg = self.stats["data_quality_avg"]
            if total_requests == 1:
                self.stats["data_quality_avg"] = quality_score
            else:
                self.stats["data_quality_avg"] = alpha * quality_score + (1 - alpha) * current_quality_avg
            
            self.stats["last_update"] = datetime.now()
            
            # 성능 메트릭 로깅 (주기적으로)
            if total_requests % 100 == 0 and self.log_integrator:
                await self.log_integrator.log_system_metrics(
                    component="market_data_provider",
                    metrics={
                        "total_requests": total_requests,
                        "cache_hit_rate": self.stats["cache_hits"] / total_requests if total_requests > 0 else 0,
                        "avg_latency_ms": self.stats["avg_latency_ms"],
                        "data_quality_avg": self.stats["data_quality_avg"],
                        "failed_requests": self.stats["failed_requests"]
                    }
                )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Performance stats update error: {e}")
    
    async def get_historical_data(self, 
                                symbol: str, 
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        히스토리 데이터 조회
        
        Args:
            symbol: 거래 심볼
            limit: 조회할 데이터 수
            
        Returns:
            List[Dict[str, Any]]: 히스토리 데이터
        """
        try:
            if symbol not in self.historical_data:
                return []
            
            history = list(self.historical_data[symbol])
            
            # 최신 데이터부터 반환
            history.reverse()
            
            # 제한 수만큼 반환
            limited_history = history[:limit]
            
            return [data.to_dict() for data in limited_history]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Get historical data error for {symbol}: {e}")
            return []
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        여러 심볼의 가격 동시 조회
        
        Args:
            symbols: 심볼 목록
            
        Returns:
            Dict[str, Optional[Dict[str, Any]]]: 심볼별 가격 정보
        """
        try:
            tasks = [self.get_current_price(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            price_data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    if self.logger:
                        self.logger.warning(f"Failed to get price for {symbol}: {result}")
                    price_data[symbol] = None
                else:
                    price_data[symbol] = result
            
            return price_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Get multiple prices error: {e}")
            return {symbol: None for symbol in symbols}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.stats["total_requests"]
        
        return {
            "cache_size": len(self.price_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": self.stats["cache_hits"] / total_requests if total_requests > 0 else 0,
            "cache_miss_rate": self.stats["cache_misses"] / total_requests if total_requests > 0 else 0,
            "cached_symbols": list(self.price_cache.keys()),
            "historical_symbols": list(self.historical_data.keys()),
            "total_historical_points": sum(len(deque_data) for deque_data in self.historical_data.values())
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """제공자 통계 조회"""
        stats = self.stats.copy()
        
        # 추가 통계
        total_requests = stats["total_requests"]
        stats.update({
            "exchange": self.exchange,
            "mode": self.mode,
            "connected": self.connected,
            "cache_hit_rate": stats["cache_hits"] / total_requests if total_requests > 0 else 0,
            "success_rate": (total_requests - stats["failed_requests"]) / total_requests if total_requests > 0 else 0,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "cache_stats": self.get_cache_stats()
        })
        
        if stats["last_update"]:
            stats["last_update"] = stats["last_update"].isoformat()
        
        return stats
    
    async def clear_cache(self, symbol: Optional[str] = None):
        """
        캐시 정리
        
        Args:
            symbol: 특정 심볼 (None이면 전체 정리)
        """
        try:
            if symbol:
                if symbol in self.price_cache:
                    del self.price_cache[symbol]
                if symbol in self.historical_data:
                    del self.historical_data[symbol]
            else:
                self.price_cache.clear()
                self.historical_data.clear()
            
            if self.logger:
                self.logger.info(f"Cache cleared for: {symbol or 'all symbols'}")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache clear error: {e}")
    
    async def shutdown(self):
        """데이터 제공자 종료"""
        try:
            self.connected = False
            
            # HTTP 세션 종료
            if self.session:
                await self.session.close()
            
            # 캐시 정리
            await self.clear_cache()
            
            if self.logger:
                self.logger.info("Market data provider shutdown")
            
            # 종료 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="market_data_provider_shutdown",
                    severity="medium",
                    description="Market data provider shutdown completed",
                    final_stats=self.stats
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Market data provider shutdown error: {e}")

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_market_data_provider(exchange: str = "binance", 
                                   mode: str = "simulation") -> VPSMarketDataProvider:
    """VPS 최적화된 시장 데이터 제공자 생성"""
    return VPSMarketDataProvider(exchange=exchange, mode=mode, enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_market_data_provider():
        provider = create_vps_market_data_provider("binance", "simulation")
        
        # 초기화
        success = await provider.initialize()
        print(f"Initialization: {success}")
        
        if success:
            # 단일 가격 조회
            price_data = await provider.get_current_price("BTCUSDT")
            print("Price data:", json.dumps(price_data, indent=2, default=str))
            
            # 다중 가격 조회
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            multiple_prices = await provider.get_multiple_prices(symbols)
            print("Multiple prices:", json.dumps(multiple_prices, indent=2, default=str))
            
            # 히스토리 데이터 (몇 번 더 호출해서 히스토리 생성)
            for _ in range(5):
                await provider.get_current_price("BTCUSDT")
                await asyncio.sleep(0.1)
            
            history = await provider.get_historical_data("BTCUSDT", limit=5)
            print(f"Historical data points: {len(history)}")
            
            # 통계 확인
            stats = provider.get_provider_stats()
            print("Provider stats:", json.dumps(stats, indent=2, default=str))
            
            # 종료
            await provider.shutdown()
    
    asyncio.run(test_market_data_provider())