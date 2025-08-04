#!/usr/bin/env python3
"""
Option Expiry Monitor for AuroraQ Sentiment Service
옵션/선물 만기 모니터 - Deribit, Binance, CME 기반
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class InstrumentType(Enum):
    """금융상품 유형"""
    OPTION = "option"
    FUTURE = "future"
    PERPETUAL = "perpetual"

class ExpiryUrgency(Enum):
    """만기 긴급도"""
    IMMEDIATE = "immediate"  # 24시간 이내
    HIGH = "high"           # 3일 이내
    MEDIUM = "medium"       # 1주일 이내
    LOW = "low"             # 1개월 이내

@dataclass
class ExpiryEvent:
    """만기 이벤트"""
    instrument_name: str
    instrument_type: InstrumentType
    expiry_timestamp: datetime
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    open_interest: float = 0.0
    volume_24h: float = 0.0
    underlying_asset: str = "BTC"
    exchange: str = "deribit"
    
    # 계산된 필드
    time_to_expiry_hours: float = 0.0
    urgency: ExpiryUrgency = ExpiryUrgency.LOW
    impact_score: float = 0.0
    
    def __post_init__(self):
        """후처리"""
        self.time_to_expiry_hours = (self.expiry_timestamp - datetime.now(timezone.utc)).total_seconds() / 3600
        self._calculate_urgency()
        self._calculate_impact_score()
    
    def _calculate_urgency(self):
        """긴급도 계산"""
        if self.time_to_expiry_hours <= 24:
            self.urgency = ExpiryUrgency.IMMEDIATE
        elif self.time_to_expiry_hours <= 72:  # 3일
            self.urgency = ExpiryUrgency.HIGH
        elif self.time_to_expiry_hours <= 168:  # 1주일
            self.urgency = ExpiryUrgency.MEDIUM
        else:
            self.urgency = ExpiryUrgency.LOW
    
    def _calculate_impact_score(self):
        """영향도 점수 계산"""
        # 기본 점수 (미결제약정 기반)
        oi_score = min(1.0, self.open_interest / 10000)  # 10000 BTC 기준
        
        # 거래량 점수
        volume_score = min(1.0, self.volume_24h / 1000)  # 1000 BTC 기준
        
        # 시간 가중치 (만료가 가까울수록 높음)
        if self.time_to_expiry_hours > 0:
            time_weight = max(0.1, min(2.0, 168 / self.time_to_expiry_hours))  # 1주일 기준 역산
        else:
            time_weight = 2.0
        
        # 옵션 타입별 가중치 (ATM 옵션이 더 중요)
        type_weight = 1.0
        if self.instrument_type == InstrumentType.OPTION and self.strike_price:
            # 현재가 대비 ATM 정도 계산 (간단 구현)
            type_weight = 1.2  # 옵션은 기본적으로 높은 가중치
        
        self.impact_score = (oi_score * 0.4 + volume_score * 0.2) * time_weight * type_weight
        self.impact_score = min(2.0, self.impact_score)

@dataclass 
class ExpiryCluster:
    """만기 클러스터 (같은 날짜의 여러 상품들)"""
    expiry_date: datetime
    events: List[ExpiryEvent] = field(default_factory=list)
    total_open_interest: float = 0.0
    total_volume: float = 0.0
    cluster_impact_score: float = 0.0
    dominant_strikes: List[float] = field(default_factory=list)
    
    def update_cluster_metrics(self):
        """클러스터 메트릭 업데이트"""
        self.total_open_interest = sum(event.open_interest for event in self.events)
        self.total_volume = sum(event.volume_24h for event in self.events)
        self.cluster_impact_score = sum(event.impact_score for event in self.events)
        
        # 주요 행사가격 계산 (OI 기준 상위 3개)
        strikes_oi = defaultdict(float)
        for event in self.events:
            if event.strike_price:
                strikes_oi[event.strike_price] += event.open_interest
        
        sorted_strikes = sorted(strikes_oi.items(), key=lambda x: x[1], reverse=True)
        self.dominant_strikes = [strike for strike, _ in sorted_strikes[:3]]

class OptionExpiryMonitor:
    """옵션 만기 모니터"""
    
    def __init__(self, 
                 update_interval: int = 3600,  # 1시간
                 max_days_ahead: int = 90):
        """
        초기화
        
        Args:
            update_interval: 업데이트 간격 (초)
            max_days_ahead: 모니터링할 최대 일수
        """
        self.update_interval = update_interval
        self.max_days_ahead = max_days_ahead
        
        # 거래소별 API 엔드포인트
        self.api_endpoints = {
            "deribit": {
                "base_url": "https://www.deribit.com/api/v2",
                "instruments": "/public/get_instruments",
                "ticker": "/public/ticker"
            },
            "binance": {
                "base_url": "https://dapi.binance.com",  # Delivery API
                "exchange_info": "/dapi/v1/exchangeInfo",
                "ticker": "/dapi/v1/ticker/24hr"
            }
        }
        
        # 데이터 저장소
        self.expiry_events: Dict[str, ExpiryEvent] = {}  # instrument_name -> event
        self.expiry_clusters: Dict[str, ExpiryCluster] = {}  # date_string -> cluster
        self.historical_expiries: deque = deque(maxlen=100)  # 최근 만료된 상품들
        
        # 실행 제어
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 통계
        self.stats = {
            "total_instruments_tracked": 0,
            "active_options": 0,
            "active_futures": 0,
            "expiries_this_week": 0,
            "last_update": None,
            "update_errors": 0,
            "api_calls_made": 0
        }
    
    async def start(self):
        """모니터 시작"""
        if self.is_running:
            logger.warning("Option expiry monitor already running")
            return
        
        # HTTP 세션 생성
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'AuroraQ-Option-Monitor/1.0'}
        )
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Option expiry monitor started (interval: {self.update_interval}s)")
        
        # 초기 데이터 수집
        await self._update_all_expiries()
    
    async def stop(self):
        """모니터 중지"""
        self.is_running = False
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.25)  # 연결 정리 대기
        
        logger.info("Option expiry monitor stopped")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                await self._update_all_expiries()
                await self._cleanup_expired_events()
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.stats["update_errors"] += 1
                await asyncio.sleep(300)  # 오류 시 5분 대기
    
    async def _update_all_expiries(self):
        """모든 만기 정보 업데이트"""
        logger.debug("Updating option expiry data...")
        
        try:
            # 거래소별 병렬 수집
            tasks = [
                self._update_deribit_expiries(),
                self._update_binance_expiries()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for exchange, result in zip(["deribit", "binance"], results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to update {exchange} expiries: {result}")
                    self.stats["update_errors"] += 1
            
            # 클러스터 업데이트
            self._update_expiry_clusters()
            
            # 통계 업데이트
            self._update_stats()
            
            logger.debug(f"Expiry data updated: {len(self.expiry_events)} instruments tracked")
            
        except Exception as e:
            logger.error(f"Failed to update expiry data: {e}")
    
    async def _update_deribit_expiries(self):
        """Deribit 만기 정보 업데이트"""
        try:
            base_url = self.api_endpoints["deribit"]["base_url"]
            
            # BTC와 ETH 상품 정보 가져오기
            for currency in ["BTC", "ETH"]:
                instruments_url = f"{base_url}{self.api_endpoints['deribit']['instruments']}"
                params = {
                    "currency": currency,
                    "kind": "option",  # 먼저 옵션
                    "expired": "false"
                }
                
                async with self.session.get(instruments_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_deribit_instruments(data["result"], currency)
                    else:
                        logger.warning(f"Deribit API error for {currency} options: {response.status}")
                
                self.stats["api_calls_made"] += 1
                
                # 선물도 가져오기
                params["kind"] = "future"
                async with self.session.get(instruments_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_deribit_instruments(data["result"], currency)
                
                self.stats["api_calls_made"] += 1
            
        except Exception as e:
            logger.error(f"Deribit update failed: {e}")
    
    async def _process_deribit_instruments(self, instruments: List[Dict], currency: str):
        """Deribit 상품 정보 처리"""
        cutoff_date = datetime.now(timezone.utc) + timedelta(days=self.max_days_ahead)
        
        for instrument in instruments:
            try:
                instrument_name = instrument["instrument_name"]
                expiry_timestamp = datetime.fromtimestamp(
                    instrument["expiration_timestamp"] / 1000, tz=timezone.utc
                )
                
                # 너무 먼 만료일은 제외
                if expiry_timestamp > cutoff_date:
                    continue
                
                # 이미 만료된 것은 제외
                if expiry_timestamp < datetime.now(timezone.utc):
                    continue
                
                # 상품 유형 결정
                if instrument["kind"] == "option":
                    instrument_type = InstrumentType.OPTION
                    strike_price = instrument.get("strike")
                    option_type = instrument.get("option_type")
                else:
                    instrument_type = InstrumentType.FUTURE
                    strike_price = None
                    option_type = None
                
                # 티커 정보 가져오기 (OI, Volume)
                ticker_data = await self._get_deribit_ticker(instrument_name)
                
                # ExpiryEvent 생성
                event = ExpiryEvent(
                    instrument_name=instrument_name,
                    instrument_type=instrument_type,
                    expiry_timestamp=expiry_timestamp,
                    strike_price=strike_price,
                    option_type=option_type,
                    open_interest=ticker_data.get("open_interest", 0.0),
                    volume_24h=ticker_data.get("stats", {}).get("volume", 0.0),
                    underlying_asset=currency,
                    exchange="deribit"
                )
                
                self.expiry_events[instrument_name] = event
                
            except Exception as e:
                logger.debug(f"Failed to process Deribit instrument: {e}")
    
    async def _get_deribit_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """Deribit 티커 정보 조회"""
        try:
            base_url = self.api_endpoints["deribit"]["base_url"]
            ticker_url = f"{base_url}{self.api_endpoints['deribit']['ticker']}"
            
            params = {"instrument_name": instrument_name}
            
            async with self.session.get(ticker_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["result"]
                else:
                    return {}
            
        except Exception as e:
            logger.debug(f"Failed to get Deribit ticker for {instrument_name}: {e}")
            return {}
    
    async def _update_binance_expiries(self):
        """Binance 선물 만기 정보 업데이트"""
        try:
            base_url = self.api_endpoints["binance"]["base_url"]
            exchange_info_url = f"{base_url}{self.api_endpoints['binance']['exchange_info']}"
            
            # 거래소 정보 가져오기
            async with self.session.get(exchange_info_url) as response:
                if response.status == 200:
                    data = await response.json()
                    await self._process_binance_instruments(data["symbols"])
                else:
                    logger.warning(f"Binance API error: {response.status}")
            
            self.stats["api_calls_made"] += 1
            
        except Exception as e:
            logger.error(f"Binance update failed: {e}")
    
    async def _process_binance_instruments(self, symbols: List[Dict]):
        """Binance 상품 정보 처리"""
        cutoff_date = datetime.now(timezone.utc) + timedelta(days=self.max_days_ahead)
        
        for symbol_info in symbols:
            try:
                symbol = symbol_info["symbol"]
                
                # BTC, ETH 관련 상품만
                if not (symbol.startswith("BTC") or symbol.startswith("ETH")):
                    continue
                
                # 만료일이 있는 상품만 (영구선물 제외)
                if symbol_info.get("contractType") == "PERPETUAL":
                    continue
                
                # 만료일 파싱
                delivery_date = symbol_info.get("deliveryDate")
                if not delivery_date:
                    continue
                
                expiry_timestamp = datetime.fromtimestamp(
                    delivery_date / 1000, tz=timezone.utc
                )
                
                # 필터링
                if expiry_timestamp > cutoff_date or expiry_timestamp < datetime.now(timezone.utc):
                    continue
                
                # 기초자산 결정
                underlying = "BTC" if symbol.startswith("BTC") else "ETH"
                
                # 24시간 통계 가져오기
                ticker_data = await self._get_binance_ticker(symbol)
                
                # ExpiryEvent 생성
                event = ExpiryEvent(
                    instrument_name=symbol,
                    instrument_type=InstrumentType.FUTURE,
                    expiry_timestamp=expiry_timestamp,
                    strike_price=None,
                    option_type=None,
                    open_interest=float(ticker_data.get("openInterest", 0)),
                    volume_24h=float(ticker_data.get("volume", 0)),
                    underlying_asset=underlying,
                    exchange="binance"
                )
                
                self.expiry_events[symbol] = event
                
            except Exception as e:
                logger.debug(f"Failed to process Binance instrument: {e}")
    
    async def _get_binance_ticker(self, symbol: str) -> Dict[str, Any]:
        """Binance 티커 정보 조회"""
        try:
            base_url = self.api_endpoints["binance"]["base_url"]
            ticker_url = f"{base_url}{self.api_endpoints['binance']['ticker']}"
            
            params = {"symbol": symbol}
            
            async with self.session.get(ticker_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and data:
                        return data[0]
                    return {}
                else:
                    return {}
            
        except Exception as e:
            logger.debug(f"Failed to get Binance ticker for {symbol}: {e}")
            return {}
    
    def _update_expiry_clusters(self):
        """만기 클러스터 업데이트"""
        self.expiry_clusters.clear()
        
        # 날짜별로 그룹화
        date_groups = defaultdict(list)
        for event in self.expiry_events.values():
            date_key = event.expiry_timestamp.strftime("%Y-%m-%d")
            date_groups[date_key].append(event)
        
        # 클러스터 생성
        for date_key, events in date_groups.items():
            if events:
                cluster = ExpiryCluster(
                    expiry_date=events[0].expiry_timestamp.replace(hour=8, minute=0, second=0),  # 표준화
                    events=events
                )
                cluster.update_cluster_metrics()
                self.expiry_clusters[date_key] = cluster
    
    async def _cleanup_expired_events(self):
        """만료된 이벤트 정리"""
        current_time = datetime.now(timezone.utc)
        expired_events = []
        
        for instrument_name, event in list(self.expiry_events.items()):
            if event.expiry_timestamp < current_time:
                expired_events.append(event)
                del self.expiry_events[instrument_name]
        
        # 히스토리에 추가
        self.historical_expiries.extend(expired_events)
        
        if expired_events:
            logger.debug(f"Cleaned up {len(expired_events)} expired events")
    
    def _update_stats(self):
        """통계 업데이트"""
        self.stats["total_instruments_tracked"] = len(self.expiry_events)
        self.stats["active_options"] = sum(
            1 for event in self.expiry_events.values()
            if event.instrument_type == InstrumentType.OPTION
        )
        self.stats["active_futures"] = sum(
            1 for event in self.expiry_events.values()
            if event.instrument_type == InstrumentType.FUTURE
        )
        
        # 이번 주 만료
        week_ahead = datetime.now(timezone.utc) + timedelta(days=7)
        self.stats["expiries_this_week"] = sum(
            1 for event in self.expiry_events.values()
            if event.expiry_timestamp <= week_ahead
        )
        
        self.stats["last_update"] = datetime.now().isoformat()
    
    def get_urgent_expiries(self, urgency: ExpiryUrgency = ExpiryUrgency.HIGH) -> List[ExpiryEvent]:
        """긴급 만료 이벤트 조회"""
        urgent_values = {
            ExpiryUrgency.IMMEDIATE: [ExpiryUrgency.IMMEDIATE],
            ExpiryUrgency.HIGH: [ExpiryUrgency.IMMEDIATE, ExpiryUrgency.HIGH],
            ExpiryUrgency.MEDIUM: [ExpiryUrgency.IMMEDIATE, ExpiryUrgency.HIGH, ExpiryUrgency.MEDIUM],
            ExpiryUrgency.LOW: list(ExpiryUrgency)
        }
        
        target_urgencies = set(urgent_values[urgency])
        
        urgent_events = [
            event for event in self.expiry_events.values()
            if event.urgency in target_urgencies
        ]
        
        # 영향도순 정렬
        return sorted(urgent_events, key=lambda x: x.impact_score, reverse=True)
    
    def get_expiry_clusters_by_impact(self, limit: int = 10) -> List[ExpiryCluster]:
        """영향도순 만기 클러스터"""
        clusters = list(self.expiry_clusters.values())
        sorted_clusters = sorted(clusters, key=lambda x: x.cluster_impact_score, reverse=True)
        return sorted_clusters[:limit]
    
    def get_expiry_calendar(self, days_ahead: int = 30) -> Dict[str, Any]:
        """만기 캘린더"""
        cutoff_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        calendar_data = {}
        for date_key, cluster in self.expiry_clusters.items():
            if cluster.expiry_date <= cutoff_date:
                calendar_data[date_key] = {
                    "date": cluster.expiry_date.isoformat(),
                    "instruments_count": len(cluster.events),
                    "total_open_interest": cluster.total_open_interest,
                    "total_volume": cluster.total_volume,
                    "impact_score": cluster.cluster_impact_score,
                    "dominant_strikes": cluster.dominant_strikes,
                    "exchanges": list(set(event.exchange for event in cluster.events))
                }
        
        return calendar_data
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """모니터 통계"""
        return {
            "is_running": self.is_running,
            "update_interval": self.update_interval,
            "max_days_ahead": self.max_days_ahead,
            **self.stats,
            "clusters_count": len(self.expiry_clusters),
            "historical_expiries_count": len(self.historical_expiries)
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_option_monitor():
        """옵션 만기 모니터 테스트"""
        
        print("=== Option Expiry Monitor 테스트 ===")
        
        monitor = OptionExpiryMonitor(update_interval=300)  # 5분 간격 테스트
        
        try:
            await monitor.start()
            
            # 30초 대기 후 결과 확인
            await asyncio.sleep(30)
            
            print(f"\n1. 추적 중인 상품 수: {len(monitor.expiry_events)}")
            
            # 긴급 만료 이벤트
            urgent_events = monitor.get_urgent_expiries(ExpiryUrgency.HIGH)
            print(f"\n2. 긴급 만료 이벤트 ({len(urgent_events)}개):")
            for i, event in enumerate(urgent_events[:5], 1):
                print(f"  {i}. {event.instrument_name} ({event.exchange})")
                print(f"     만료: {event.expiry_timestamp.strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"     긴급도: {event.urgency.value}")
                print(f"     영향도: {event.impact_score:.2f}")
                print(f"     OI: {event.open_interest:.0f}, Volume: {event.volume_24h:.0f}")
            
            # 만기 클러스터
            clusters = monitor.get_expiry_clusters_by_impact(5)
            print(f"\n3. 상위 영향도 만기 클러스터:")
            for i, cluster in enumerate(clusters, 1):
                print(f"  {i}. {cluster.expiry_date.strftime('%Y-%m-%d')}")
                print(f"     상품 수: {len(cluster.events)}")
                print(f"     총 OI: {cluster.total_open_interest:.0f}")
                print(f"     영향도: {cluster.cluster_impact_score:.2f}")
                print(f"     주요 행사가: {cluster.dominant_strikes[:3]}")
            
            # 만기 캘린더
            print(f"\n4. 만기 캘린더 (7일간):")
            calendar = monitor.get_expiry_calendar(7)
            for date_key in sorted(calendar.keys()):
                data = calendar[date_key]
                print(f"  {date_key}: {data['instruments_count']}개 상품, "
                      f"영향도 {data['impact_score']:.1f}")
            
            # 통계
            print(f"\n5. 모니터 통계:")
            stats = monitor.get_monitor_stats()
            for key, value in stats.items():
                if key not in ["last_update"]:
                    print(f"  {key}: {value}")
            
        finally:
            await monitor.stop()
        
        print(f"\n=== 테스트 완료 ===")
    
    # 테스트 실행
    asyncio.run(test_option_monitor())