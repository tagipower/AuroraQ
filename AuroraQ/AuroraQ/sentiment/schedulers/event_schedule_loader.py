#!/usr/bin/env python3
"""
Event Schedule Loader for AuroraQ Sentiment Service
경제일정 로더 - FOMC, CPI, PPI 등 경제 이벤트 스케줄 관리
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
import re
from bs4 import BeautifulSoup
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not available, RSS parsing will be limited")
    
    # 간단한 feedparser 폴백
    class MockFeedParser:
        @staticmethod
        def parse(url):
            return {'entries': [], 'status': 404}
    
    feedparser = MockFeedParser()

from collections import defaultdict

logger = logging.getLogger(__name__)

class EventImportance(Enum):
    """이벤트 중요도"""
    CRITICAL = "critical"    # FOMC, NFP 등
    HIGH = "high"           # CPI, PPI, GDP 등
    MEDIUM = "medium"       # 기타 주요 지표
    LOW = "low"             # 소규모 지표

class EventCategory(Enum):
    """이벤트 카테고리"""
    INTEREST_RATE = "interest_rate"     # 금리 관련
    INFLATION = "inflation"             # 인플레이션
    EMPLOYMENT = "employment"           # 고용
    GDP = "gdp"                        # GDP
    MANUFACTURING = "manufacturing"     # 제조업
    CONSUMER = "consumer"              # 소비자 관련
    HOUSING = "housing"                # 부동산
    TRADE = "trade"                    # 무역
    FED_SPEECH = "fed_speech"          # 연준 발언
    OTHER = "other"                    # 기타

@dataclass
class EconomicEvent:
    """경제 이벤트"""
    event_id: str
    title: str
    category: EventCategory
    importance: EventImportance
    scheduled_time: datetime
    country: str = "US"
    currency: str = "USD"
    
    # 예측/실제 값
    forecast_value: Optional[str] = None
    previous_value: Optional[str] = None
    actual_value: Optional[str] = None
    
    # 메타데이터
    description: str = ""
    source: str = ""
    impact_assets: List[str] = field(default_factory=lambda: ["USD", "BTC", "ETH"])
    
    # 계산된 필드
    time_to_event_hours: float = 0.0
    market_impact_score: float = 0.0
    volatility_expected: bool = False
    
    def __post_init__(self):
        """후처리"""
        self._calculate_time_to_event()
        self._calculate_market_impact()
    
    def _calculate_time_to_event(self):
        """이벤트까지 시간 계산"""
        self.time_to_event_hours = (
            self.scheduled_time - datetime.now(timezone.utc)
        ).total_seconds() / 3600
    
    def _calculate_market_impact(self):
        """시장 영향도 계산"""
        # 기본 중요도 점수
        importance_scores = {
            EventImportance.CRITICAL: 1.0,
            EventImportance.HIGH: 0.8,
            EventImportance.MEDIUM: 0.5,
            EventImportance.LOW: 0.2
        }
        
        base_score = importance_scores[self.importance]
        
        # 시간 가중치 (24시간 이내는 가중치 증가)
        if 0 <= self.time_to_event_hours <= 24:
            time_multiplier = 1.5
        elif 0 <= self.time_to_event_hours <= 72:  # 3일
            time_multiplier = 1.2
        elif self.time_to_event_hours <= 168:  # 1주일
            time_multiplier = 1.0
        else:
            time_multiplier = 0.8
        
        # 카테고리별 가중치
        category_multipliers = {
            EventCategory.INTEREST_RATE: 1.5,
            EventCategory.INFLATION: 1.3,
            EventCategory.EMPLOYMENT: 1.2,
            EventCategory.GDP: 1.1,
            EventCategory.FED_SPEECH: 1.0,
            EventCategory.MANUFACTURING: 0.9,
            EventCategory.CONSUMER: 0.8,
            EventCategory.HOUSING: 0.7,
            EventCategory.TRADE: 0.6,
            EventCategory.OTHER: 0.5
        }
        
        category_multiplier = category_multipliers.get(self.category, 0.5)
        
        self.market_impact_score = base_score * time_multiplier * category_multiplier
        self.volatility_expected = self.market_impact_score > 0.8

@dataclass
class EventCluster:
    """이벤트 클러스터 (같은 시간대 여러 이벤트)"""
    date: datetime
    events: List[EconomicEvent] = field(default_factory=list)
    total_impact_score: float = 0.0
    volatility_likelihood: float = 0.0
    dominant_categories: List[EventCategory] = field(default_factory=list)
    
    def update_cluster_metrics(self):
        """클러스터 메트릭 업데이트"""
        if not self.events:
            return
        
        self.total_impact_score = sum(event.market_impact_score for event in self.events)
        
        # 변동성 가능성 (최대 영향도 이벤트 기준)
        max_impact = max(event.market_impact_score for event in self.events)
        event_count_multiplier = min(1.5, 1.0 + (len(self.events) - 1) * 0.1)
        self.volatility_likelihood = min(1.0, max_impact * event_count_multiplier)
        
        # 주요 카테고리 계산
        category_counts = defaultdict(int)
        for event in self.events:
            category_counts[event.category] += 1
        
        sorted_categories = sorted(
            category_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        self.dominant_categories = [cat for cat, _ in sorted_categories[:3]]

class EventScheduleLoader:
    """경제일정 로더"""
    
    def __init__(self, update_interval: int = 21600):  # 6시간
        """
        초기화
        
        Args:
            update_interval: 업데이트 간격 (초)
        """
        self.update_interval = update_interval
        
        # 데이터 소스 설정
        self.data_sources = {
            "investing": {
                "url": "https://www.investing.com/economic-calendar/",
                "rss": "https://www.investing.com/rss/news_1.rss"
            },
            "forexfactory": {
                "url": "https://www.forexfactory.com/calendar"
            },
            "fred": {
                "base_url": "https://fred.stlouisfed.org"
            }
        }
        
        # 정적 이벤트 템플릿 (정규 스케줄)
        self.static_events = self._load_static_event_templates()
        
        # 데이터 저장소
        self.scheduled_events: Dict[str, EconomicEvent] = {}
        self.event_clusters: Dict[str, EventCluster] = {}
        self.historical_events: List[EconomicEvent] = []
        
        # 실행 제어
        self.is_running = False
        self.loader_task: Optional[asyncio.Task] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 통계
        self.stats = {
            "total_events_loaded": 0,
            "events_this_week": 0,
            "critical_events_pending": 0,
            "last_update": None,
            "load_errors": 0,
            "sources_used": 0
        }
    
    def _load_static_event_templates(self) -> Dict[str, Dict]:
        """정적 이벤트 템플릿 로드"""
        return {
            # FOMC 관련
            "fomc_meeting": {
                "title": "FOMC Meeting",
                "category": EventCategory.INTEREST_RATE,
                "importance": EventImportance.CRITICAL,
                "description": "Federal Open Market Committee Meeting",
                "impact_assets": ["USD", "BTC", "ETH", "GOLD"]
            },
            "fomc_minutes": {
                "title": "FOMC Meeting Minutes",
                "category": EventCategory.INTEREST_RATE,
                "importance": EventImportance.HIGH,
                "description": "FOMC Meeting Minutes Release"
            },
            
            # 인플레이션 지표
            "cpi": {
                "title": "Consumer Price Index (CPI)",
                "category": EventCategory.INFLATION,
                "importance": EventImportance.HIGH,
                "description": "Monthly inflation measure"
            },
            "ppi": {
                "title": "Producer Price Index (PPI)",
                "category": EventCategory.INFLATION,
                "importance": EventImportance.MEDIUM,
                "description": "Wholesale price inflation measure"
            },
            "pce": {
                "title": "Personal Consumption Expenditure (PCE)",
                "category": EventCategory.INFLATION,
                "importance": EventImportance.HIGH,
                "description": "Fed's preferred inflation measure"
            },
            
            # 고용 지표
            "nfp": {
                "title": "Non-Farm Payrolls (NFP)",
                "category": EventCategory.EMPLOYMENT,
                "importance": EventImportance.CRITICAL,
                "description": "Monthly employment report"
            },
            "unemployment_rate": {
                "title": "Unemployment Rate",
                "category": EventCategory.EMPLOYMENT,
                "importance": EventImportance.HIGH,
                "description": "Monthly unemployment rate"
            },
            "initial_claims": {
                "title": "Initial Jobless Claims",
                "category": EventCategory.EMPLOYMENT,
                "importance": EventImportance.MEDIUM,
                "description": "Weekly unemployment claims"
            },
            
            # GDP
            "gdp_preliminary": {
                "title": "GDP (Preliminary)",
                "category": EventCategory.GDP,
                "importance": EventImportance.HIGH,
                "description": "Quarterly GDP preliminary reading"
            },
            "gdp_final": {
                "title": "GDP (Final)",
                "category": EventCategory.GDP,
                "importance": EventImportance.MEDIUM,
                "description": "Quarterly GDP final reading"
            },
            
            # 제조업
            "ism_manufacturing": {
                "title": "ISM Manufacturing PMI",
                "category": EventCategory.MANUFACTURING,
                "importance": EventImportance.MEDIUM,
                "description": "Manufacturing sector activity index"
            },
            "ism_services": {
                "title": "ISM Services PMI",
                "category": EventCategory.MANUFACTURING,
                "importance": EventImportance.MEDIUM,
                "description": "Services sector activity index"
            },
            
            # 소비자 관련
            "retail_sales": {
                "title": "Retail Sales",
                "category": EventCategory.CONSUMER,
                "importance": EventImportance.MEDIUM,
                "description": "Monthly retail sales data"
            },
            "consumer_confidence": {
                "title": "Consumer Confidence Index",
                "category": EventCategory.CONSUMER,
                "importance": EventImportance.MEDIUM,
                "description": "Consumer sentiment measure"
            }
        }
    
    async def start(self):
        """로더 시작"""
        if self.is_running:
            logger.warning("Event schedule loader already running")
            return
        
        # HTTP 세션 생성
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'AuroraQ-EventScheduler/1.0'}
        )
        
        self.is_running = True
        self.loader_task = asyncio.create_task(self._loader_loop())
        logger.info(f"Event schedule loader started (interval: {self.update_interval}s)")
        
        # 초기 데이터 로드
        await self._load_all_events()
    
    async def stop(self):
        """로더 중지"""
        self.is_running = False
        
        if self.loader_task and not self.loader_task.done():
            self.loader_task.cancel()
            try:
                await self.loader_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.25)
        
        logger.info("Event schedule loader stopped")
    
    async def _loader_loop(self):
        """로더 루프"""
        while self.is_running:
            try:
                await self._load_all_events()
                await self._cleanup_past_events()
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loader loop error: {e}")
                self.stats["load_errors"] += 1
                await asyncio.sleep(1800)  # 오류 시 30분 대기
    
    async def _load_all_events(self):
        """모든 이벤트 로드"""
        logger.debug("Loading economic events...")
        
        try:
            # 여러 소스에서 병렬 로딩
            tasks = [
                self._load_from_investing_rss(),
                self._generate_recurring_events(),
                self._load_manual_events()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            total_loaded = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Event loading task {i} failed: {result}")
                    self.stats["load_errors"] += 1
                elif isinstance(result, int):
                    total_loaded += result
            
            # 클러스터 업데이트
            self._update_event_clusters()
            
            # 통계 업데이트
            self._update_stats()
            
            logger.debug(f"Economic events loaded: {total_loaded} new events")
            
        except Exception as e:
            logger.error(f"Failed to load economic events: {e}")
    
    async def _load_from_investing_rss(self) -> int:
        """Investing.com RSS에서 이벤트 로드"""
        try:
            rss_url = self.data_sources["investing"]["rss"]
            
            async with self.session.get(rss_url) as response:
                if response.status != 200:
                    return 0
                
                rss_content = await response.text()
            
            # RSS 파싱
            feed = feedparser.parse(rss_content)
            events_loaded = 0
            
            for entry in feed.entries[:20]:  # 상위 20개만
                try:
                    event = self._parse_rss_entry(entry)
                    if event:
                        self.scheduled_events[event.event_id] = event
                        events_loaded += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to parse RSS entry: {e}")
            
            self.stats["sources_used"] += 1
            return events_loaded
            
        except Exception as e:
            logger.error(f"Failed to load from Investing RSS: {e}")
            return 0
    
    def _parse_rss_entry(self, entry) -> Optional[EconomicEvent]:
        """RSS 엔트리 파싱"""
        try:
            title = entry.title
            description = entry.get('description', '')
            pub_date = entry.get('published_parsed')
            
            if not pub_date:
                return None
            
            # 날짜 변환
            scheduled_time = datetime(*pub_date[:6], tzinfo=timezone.utc)
            
            # 미래 이벤트만
            if scheduled_time < datetime.now(timezone.utc):
                return None
            
            # 카테고리 및 중요도 추론
            category, importance = self._infer_event_category_importance(title, description)
            
            event_id = f"rss_{hash(title + str(scheduled_time))}"
            
            return EconomicEvent(
                event_id=event_id,
                title=title,
                category=category,
                importance=importance,
                scheduled_time=scheduled_time,
                description=description,
                source="investing_rss"
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse RSS entry: {e}")
            return None
    
    def _infer_event_category_importance(self, title: str, description: str) -> Tuple[EventCategory, EventImportance]:
        """제목과 설명으로부터 카테고리와 중요도 추론"""
        text = (title + " " + description).lower()
        
        # 중요도 키워드 매칭
        critical_keywords = ["fomc", "fed meeting", "nfp", "non-farm payroll", "interest rate"]
        high_keywords = ["cpi", "ppi", "pce", "gdp", "unemployment", "inflation"]
        medium_keywords = ["ism", "retail sales", "housing", "consumer confidence"]
        
        if any(keyword in text for keyword in critical_keywords):
            importance = EventImportance.CRITICAL
        elif any(keyword in text for keyword in high_keywords):
            importance = EventImportance.HIGH
        elif any(keyword in text for keyword in medium_keywords):
            importance = EventImportance.MEDIUM
        else:
            importance = EventImportance.LOW
        
        # 카테고리 키워드 매칭
        category_keywords = {
            EventCategory.INTEREST_RATE: ["fomc", "fed", "interest rate", "monetary policy"],
            EventCategory.INFLATION: ["cpi", "ppi", "pce", "inflation"],
            EventCategory.EMPLOYMENT: ["nfp", "unemployment", "jobless", "employment"],
            EventCategory.GDP: ["gdp", "gross domestic"],
            EventCategory.MANUFACTURING: ["ism", "manufacturing", "pmi"],
            EventCategory.CONSUMER: ["retail sales", "consumer confidence", "consumer spending"],
            EventCategory.HOUSING: ["housing", "home sales", "mortgage"],
            EventCategory.FED_SPEECH: ["fed speech", "powell", "yellen", "fed chair"]
        }
        
        category = EventCategory.OTHER
        for cat, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                category = cat
                break
        
        return category, importance
    
    async def _generate_recurring_events(self) -> int:
        """정기적 이벤트 생성 (예정된 스케줄 기반)"""
        events_generated = 0
        
        try:
            # 다음 30일간의 정기 이벤트 생성
            current_date = datetime.now(timezone.utc).date()
            
            # 매월 첫 번째 금요일 - NFP
            for month_offset in range(3):  # 3개월
                target_date = current_date.replace(day=1) + timedelta(days=32 * month_offset)
                target_date = target_date.replace(day=1)
                
                # 첫 번째 금요일 찾기
                while target_date.weekday() != 4:  # 4 = Friday
                    target_date += timedelta(days=1)
                
                nfp_event = self._create_event_from_template(
                    "nfp",
                    datetime.combine(target_date, datetime.min.time().replace(hour=13, minute=30, tzinfo=timezone.utc))  # 1:30 PM UTC
                )
                
                if nfp_event:
                    self.scheduled_events[nfp_event.event_id] = nfp_event
                    events_generated += 1
            
            # 매월 중순 - CPI (보통 10-15일)
            for month_offset in range(3):
                target_date = current_date.replace(day=15) + timedelta(days=32 * month_offset)
                target_date = target_date.replace(day=15)
                
                cpi_event = self._create_event_from_template(
                    "cpi",
                    datetime.combine(target_date, datetime.min.time().replace(hour=13, minute=30, tzinfo=timezone.utc))
                )
                
                if cpi_event:
                    self.scheduled_events[cpi_event.event_id] = cpi_event
                    events_generated += 1
            
            return events_generated
            
        except Exception as e:
            logger.error(f"Failed to generate recurring events: {e}")
            return 0
    
    def _create_event_from_template(self, template_key: str, scheduled_time: datetime) -> Optional[EconomicEvent]:
        """템플릿으로부터 이벤트 생성"""
        template = self.static_events.get(template_key)
        if not template:
            return None
        
        event_id = f"static_{template_key}_{scheduled_time.strftime('%Y%m%d')}"
        
        # 이미 존재하는 이벤트는 건너뛰기
        if event_id in self.scheduled_events:
            return None
        
        return EconomicEvent(
            event_id=event_id,
            title=template["title"],
            category=template["category"],
            importance=template["importance"],
            scheduled_time=scheduled_time,
            description=template["description"],
            impact_assets=template.get("impact_assets", ["USD", "BTC", "ETH"]),
            source="static_template"
        )
    
    async def _load_manual_events(self) -> int:
        """수동 이벤트 로드 (설정 파일 등)"""
        # 추후 구현: JSON 파일이나 데이터베이스에서 수동 이벤트 로드
        return 0
    
    def _update_event_clusters(self):
        """이벤트 클러스터 업데이트"""
        self.event_clusters.clear()
        
        # 날짜별로 그룹화 (같은 날의 이벤트들)
        date_groups = defaultdict(list)
        for event in self.scheduled_events.values():
            date_key = event.scheduled_time.strftime("%Y-%m-%d")
            date_groups[date_key].append(event)
        
        # 클러스터 생성
        for date_key, events in date_groups.items():
            if events:
                cluster = EventCluster(
                    date=events[0].scheduled_time.replace(hour=0, minute=0, second=0),
                    events=events
                )
                cluster.update_cluster_metrics()
                self.event_clusters[date_key] = cluster
    
    async def _cleanup_past_events(self):
        """과거 이벤트 정리"""
        current_time = datetime.now(timezone.utc)
        past_events = []
        
        for event_id, event in list(self.scheduled_events.items()):
            if event.scheduled_time < current_time:
                past_events.append(event)
                del self.scheduled_events[event_id]
        
        # 히스토리에 추가
        self.historical_events.extend(past_events)
        
        # 히스토리 크기 제한 (최근 100개)
        if len(self.historical_events) > 100:
            self.historical_events = self.historical_events[-100:]
        
        if past_events:
            logger.debug(f"Cleaned up {len(past_events)} past events")
    
    def _update_stats(self):
        """통계 업데이트"""
        self.stats["total_events_loaded"] = len(self.scheduled_events)
        
        # 이번 주 이벤트
        week_ahead = datetime.now(timezone.utc) + timedelta(days=7)
        self.stats["events_this_week"] = sum(
            1 for event in self.scheduled_events.values()
            if event.scheduled_time <= week_ahead
        )
        
        # 중요 이벤트
        self.stats["critical_events_pending"] = sum(
            1 for event in self.scheduled_events.values()
            if event.importance == EventImportance.CRITICAL
        )
        
        self.stats["last_update"] = datetime.now().isoformat()
    
    def get_events_by_timeframe(self, hours_ahead: int = 168) -> List[EconomicEvent]:
        """시간 범위별 이벤트 조회"""
        cutoff_time = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
        
        events = [
            event for event in self.scheduled_events.values()
            if event.scheduled_time <= cutoff_time
        ]
        
        return sorted(events, key=lambda x: x.scheduled_time)
    
    def get_high_impact_events(self, days_ahead: int = 30) -> List[EconomicEvent]:
        """고영향 이벤트 조회"""
        cutoff_time = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        high_impact_events = [
            event for event in self.scheduled_events.values()
            if (event.scheduled_time <= cutoff_time and 
                event.importance in [EventImportance.CRITICAL, EventImportance.HIGH])
        ]
        
        return sorted(high_impact_events, key=lambda x: x.market_impact_score, reverse=True)
    
    def get_event_calendar(self, days_ahead: int = 14) -> Dict[str, Any]:
        """이벤트 캘린더"""
        cutoff_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        calendar_data = {}
        for date_key, cluster in self.event_clusters.items():
            if cluster.date <= cutoff_date:
                calendar_data[date_key] = {
                    "date": cluster.date.isoformat(),
                    "events_count": len(cluster.events),
                    "total_impact_score": cluster.total_impact_score,
                    "volatility_likelihood": cluster.volatility_likelihood,
                    "dominant_categories": [cat.value for cat in cluster.dominant_categories],
                    "critical_events": [
                        event.title for event in cluster.events
                        if event.importance == EventImportance.CRITICAL
                    ],
                    "high_events": [
                        event.title for event in cluster.events  
                        if event.importance == EventImportance.HIGH
                    ]
                }
        
        return calendar_data
    
    def get_next_major_event(self) -> Optional[EconomicEvent]:
        """다음 주요 이벤트"""
        major_events = [
            event for event in self.scheduled_events.values()
            if event.importance in [EventImportance.CRITICAL, EventImportance.HIGH]
        ]
        
        if not major_events:
            return None
        
        return min(major_events, key=lambda x: x.scheduled_time)
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """로더 통계"""
        return {
            "is_running": self.is_running,
            "update_interval": self.update_interval,
            **self.stats,
            "clusters_count": len(self.event_clusters),
            "historical_events_count": len(self.historical_events),
            "static_templates_count": len(self.static_events)
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_event_loader():
        """이벤트 스케줄 로더 테스트"""
        
        print("=== Event Schedule Loader 테스트 ===")
        
        loader = EventScheduleLoader(update_interval=600)  # 10분 간격 테스트
        
        try:
            await loader.start()
            
            # 30초 대기 후 결과 확인
            await asyncio.sleep(30)
            
            print(f"\n1. 로드된 이벤트 수: {len(loader.scheduled_events)}")
            
            # 다음 주요 이벤트
            next_major = loader.get_next_major_event()
            if next_major:
                print(f"\n2. 다음 주요 이벤트:")
                print(f"   제목: {next_major.title}")
                print(f"   일시: {next_major.scheduled_time.strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"   중요도: {next_major.importance.value}")
                print(f"   영향도: {next_major.market_impact_score:.2f}")
                print(f"   시간까지: {next_major.time_to_event_hours:.1f}시간")
            
            # 고영향 이벤트
            high_impact = loader.get_high_impact_events(7)
            print(f"\n3. 고영향 이벤트 (7일간, {len(high_impact)}개):")
            for i, event in enumerate(high_impact[:5], 1):
                print(f"   {i}. {event.title}")
                print(f"      {event.scheduled_time.strftime('%m-%d %H:%M')} UTC, "
                      f"영향도: {event.market_impact_score:.2f}")
            
            # 이벤트 캘린더
            print(f"\n4. 이벤트 캘린더 (2주간):")
            calendar = loader.get_event_calendar(14)
            for date_key in sorted(calendar.keys())[:7]:  # 상위 7일만
                data = calendar[date_key]
                print(f"   {date_key}: {data['events_count']}개 이벤트, "
                      f"변동성 가능성 {data['volatility_likelihood']:.1%}")
                if data['critical_events']:
                    print(f"      중요: {', '.join(data['critical_events'])}")
            
            # 통계
            print(f"\n5. 로더 통계:")
            stats = loader.get_loader_stats()
            for key, value in stats.items():
                if key not in ["last_update"]:
                    print(f"   {key}: {value}")
            
        finally:
            await loader.stop()
        
        print(f"\n=== 테스트 완료 ===")
    
    # 테스트 실행
    asyncio.run(test_event_loader())