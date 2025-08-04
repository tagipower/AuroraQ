#!/usr/bin/env python3
"""
Enhanced News Collector V3 for AuroraQ Sentiment Service
완성도 높은 뉴스 수집 시스템 - 중요도 점수화, 사전 필터링, 유사도 기반 중복 제거 통합
"""

import asyncio
import aiohttp
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import json
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

# 백오프 라이브러리 폴백
try:
    import backoff
    HAS_BACKOFF = True
except ImportError:
    HAS_BACKOFF = False
    def backoff_decorator(*args, **kwargs):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                for attempt in range(3):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)
            return wrapper
        return decorator
    
    class backoff:
        expo = None
        @staticmethod
        def on_exception(*args, **kwargs):
            return backoff_decorator(*args, **kwargs)

# 새로운 시스템들 임포트
try:
    from ..utils.news_importance_scorer import get_news_importance_scorer, ImportanceFeatures
    from ..utils.news_deduplicator import get_news_deduplicator, DeduplicationResult
    from ..utils.news_prefilter import get_news_prefilter, FilterResult, FilterDecision
    from ..utils.news_topic_classifier import NewsTopicClassifier, TopicClassification, NewsTopicCategory
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    # 폴백 구현
    ADVANCED_MODULES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Advanced news processing modules not available, using basic implementation")
    
    @dataclass
    class ImportanceFeatures:
        total_score: float = 0.5
    
    @dataclass 
    class DeduplicationResult:
        deduplicated_count: int = 0
        removed_news_ids: Set[str] = field(default_factory=set)
    
    @dataclass
    class FilterResult:
        decision: str = "approve_normal"
        importance_score: float = 0.5
        quality_score: float = 0.5
    
    def get_news_importance_scorer():
        return None
    
    def get_news_deduplicator():
        return None
        
    def get_news_prefilter():
        return None
    
    class NewsTopicCategory(Enum):
        MACRO = "macro"
        REGULATION = "regulation"
        TECHNOLOGY = "technology"
        MARKET = "market"
        CORPORATE = "corporate"
        SECURITY = "security"
        ADOPTION = "adoption"
        ANALYSIS = "analysis"
        EVENT = "event"
        OTHER = "other"
    
    @dataclass
    class TopicClassification:
        primary_topic: "NewsTopicCategory" = NewsTopicCategory.OTHER
        secondary_topics: List["NewsTopicCategory"] = field(default_factory=list)
        confidence_scores: Dict["NewsTopicCategory", float] = field(default_factory=dict)
        keywords_found: Dict[str, List[str]] = field(default_factory=dict)
        processing_time: float = 0.0
        
        @property
        def is_high_confidence(self) -> bool:
            return self.confidence_scores.get(self.primary_topic, 0) >= 0.7
        
        @property
        def is_multi_topic(self) -> bool:
            return len(self.secondary_topics) > 0
    
    class NewsTopicClassifier:
        def classify(self, title: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            return TopicClassification(
                primary_topic=NewsTopicCategory.OTHER,
                confidence_scores={NewsTopicCategory.OTHER: 0.5}
            )

from asyncio import Semaphore

logger = logging.getLogger(__name__)

class NewsSource(Enum):
    """뉴스 소스 종류"""
    GOOGLE_NEWS = "google_news"
    YAHOO_FINANCE = "yahoo_finance"  
    NEWSAPI = "newsapi"
    FINNHUB = "finnhub"
    COINDESK = "coindesk"
    REUTERS = "reuters"

class ProcessingStage(Enum):
    """처리 단계"""
    COLLECTION = "collection"           # 뉴스 수집
    DEDUPLICATION = "deduplication"     # 중복 제거
    IMPORTANCE_SCORING = "importance_scoring"  # 중요도 점수화
    TOPIC_CLASSIFICATION = "topic_classification"  # 토픽 분류
    PRE_FILTERING = "pre_filtering"     # 사전 필터링
    FINAL_PROCESSING = "final_processing"  # 최종 처리

@dataclass
class EnhancedNewsItem:
    """강화된 뉴스 아이템 데이터 클래스"""
    title: str
    content: str
    url: str
    source: NewsSource
    published_at: datetime
    category: str = "general"
    symbol: Optional[str] = None
    
    # 기본 속성
    relevance_score: float = 0.5
    entities: List[str] = field(default_factory=list)
    hash_id: str = field(init=False)
    
    # V3 추가 속성
    importance_features: Optional[ImportanceFeatures] = None
    filter_result: Optional[FilterResult] = None
    topic_classification: Optional[TopicClassification] = None
    processing_stage: ProcessingStage = ProcessingStage.COLLECTION
    
    # 메타데이터
    collection_timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """해시 ID 생성 및 초기화"""
        content_for_hash = f"{self.title}{self.url}{self.published_at.date()}"
        self.hash_id = hashlib.md5(content_for_hash.encode()).hexdigest()
    
    def __eq__(self, other):
        """중복 제거를 위한 비교"""
        if isinstance(other, EnhancedNewsItem):
            return self.hash_id == other.hash_id
        return False
    
    def __hash__(self):
        """해시 가능하게 만들기"""
        return hash(self.hash_id)
    
    @property
    def final_importance_score(self) -> float:
        """최종 중요도 점수 (여러 소스 통합)"""
        if self.importance_features:
            return self.importance_features.total_score
        return self.relevance_score
    
    @property
    def should_analyze_with_finbert(self) -> bool:
        """FinBERT 분석 필요 여부"""
        if self.filter_result:
            return self.filter_result.decision in ["approve_high_priority", "approve_normal"]
        return self.final_importance_score >= 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (API 응답용)"""
        return {
            "hash_id": self.hash_id,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "url": self.url,
            "source": self.source.value,
            "published_at": self.published_at.isoformat(),
            "category": self.category,
            "symbol": self.symbol,
            "relevance_score": self.relevance_score,
            "importance_score": self.final_importance_score,
            "should_analyze_finbert": self.should_analyze_with_finbert,
            "entities": self.entities,
            "processing_stage": self.processing_stage.value,
            "collection_timestamp": self.collection_timestamp.isoformat(),
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }

@dataclass
class CollectionResult:
    """수집 결과 종합"""
    original_count: int
    deduplicated_count: int
    high_priority_count: int
    normal_priority_count: int
    low_priority_count: int
    rejected_count: int
    
    total_processing_time: float
    stage_timings: Dict[ProcessingStage, float]
    
    news_items: List[EnhancedNewsItem]
    topic_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def finbert_analysis_count(self) -> int:
        """FinBERT 분석 대상 개수"""
        return sum(1 for item in self.news_items if item.should_analyze_with_finbert)
    
    @property
    def processing_efficiency(self) -> float:
        """처리 효율성 (고품질 뉴스 비율)"""
        if self.deduplicated_count == 0:
            return 0.0
        return (self.high_priority_count + self.normal_priority_count) / self.deduplicated_count

class EnhancedNewsCollectorV3:
    """강화된 뉴스 수집기 V3 - 완성도 높은 뉴스 수집 시스템"""
    
    def __init__(self, 
                 api_keys: Dict[str, str],
                 max_concurrent_requests: int = 3,
                 request_timeout: int = 10,
                 max_retries: int = 3,
                 enable_importance_scoring: bool = True,
                 enable_prefiltering: bool = True,
                 enable_advanced_deduplication: bool = True,
                 enable_topic_classification: bool = True):
        """
        초기화
        
        Args:
            api_keys: API 키 딕셔너리
            max_concurrent_requests: 최대 동시 요청 수
            request_timeout: 요청 타임아웃
            max_retries: 최대 재시도 횟수
            enable_importance_scoring: 중요도 점수화 활성화
            enable_prefiltering: 사전 필터링 활성화
            enable_advanced_deduplication: 고급 중복 제거 활성화
            enable_topic_classification: 토픽 자동 분류 활성화
        """
        self.api_keys = api_keys
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        
        # V3 기능 활성화 플래그
        self.enable_importance_scoring = enable_importance_scoring
        self.enable_prefiltering = enable_prefiltering
        self.enable_advanced_deduplication = enable_advanced_deduplication
        self.enable_topic_classification = enable_topic_classification
        
        # 고급 처리 시스템 초기화
        self.importance_scorer = get_news_importance_scorer() if enable_importance_scoring else None
        self.deduplicator = get_news_deduplicator() if enable_advanced_deduplication else None
        self.prefilter = get_news_prefilter() if enable_prefiltering else None
        self.topic_classifier = NewsTopicClassifier() if enable_topic_classification else None
        
        # HTTP 설정
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = Semaphore(max_concurrent_requests)
        self.collected_hashes: Set[str] = set()
        
        # Rate limiting
        self.rate_limits = {
            NewsSource.NEWSAPI: {"requests": 0, "reset_time": time.time() + 3600, "limit": 100},
            NewsSource.FINNHUB: {"requests": 0, "reset_time": time.time() + 3600, "limit": 60},
            NewsSource.GOOGLE_NEWS: {"requests": 0, "reset_time": time.time() + 60, "limit": 50},
            NewsSource.YAHOO_FINANCE: {"requests": 0, "reset_time": time.time() + 60, "limit": 100},
        }
        
        # 심볼별 키워드 매핑 (확장)
        self.symbol_keywords = {
            "BTC": ["bitcoin", "btc", "cryptocurrency", "crypto", "digital asset"],
            "ETH": ["ethereum", "eth", "smart contract", "defi", "web3"],
            "CRYPTO": ["cryptocurrency", "crypto", "blockchain", "digital asset", "altcoin"],
            "STOCK": ["stock market", "equity", "shares", "trading", "nasdaq", "s&p"],
            "FOREX": ["forex", "currency", "dollar", "euro", "exchange rate"],
            "MACRO": ["federal reserve", "fed", "interest rate", "inflation", "gdp", "economic"]
        }
        
        # V3 통계 (확장)
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_items_collected": 0,
            "duplicates_filtered": 0,
            "importance_scored": 0,
            "prefiltered": 0,
            "finbert_candidates": 0,
            "processing_times": {},
            "errors_by_source": {},
            "last_collection": None
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        connector = aiohttp.TCPConnector(
            limit=100,
            ttl_dns_cache=300,
            limit_per_host=30
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout,
            connect=5,
            sock_read=self.request_timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'AuroraQ-Sentiment-Service/3.0 (Enhanced-VPS-Optimized)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.25)
    
    async def collect_comprehensive_news(self,
                                       symbol: str = "crypto", 
                                       hours_back: int = 24,
                                       max_per_source: int = 20) -> CollectionResult:
        """
        종합적인 뉴스 수집 (V3 메인 함수)
        
        Args:
            symbol: 대상 심볼
            hours_back: 수집 시간 범위 (시간)
            max_per_source: 소스별 최대 수집 개수
            
        Returns:
            CollectionResult: 종합 수집 결과
        """
        overall_start_time = time.time()
        stage_timings = {}
        
        logger.info(f"Starting comprehensive news collection for {symbol} (V3 Enhanced)")
        
        try:
            # 1단계: 뉴스 수집
            stage_start = time.time()
            raw_news_items = await self._collect_from_all_sources(symbol, hours_back, max_per_source)
            stage_timings[ProcessingStage.COLLECTION] = time.time() - stage_start
            
            if not raw_news_items:
                logger.warning("No news items collected")
                return self._create_empty_result(stage_timings, overall_start_time)
            
            logger.info(f"Raw collection: {len(raw_news_items)} items")
            
            # 2단계: 중복 제거 (고급)
            stage_start = time.time()
            deduplicated_items = await self._advanced_deduplication(raw_news_items)
            stage_timings[ProcessingStage.DEDUPLICATION] = time.time() - stage_start
            
            logger.info(f"After deduplication: {len(deduplicated_items)} items")
            
            # 3단계: 중요도 점수화
            stage_start = time.time()
            scored_items = await self._calculate_importance_scores(deduplicated_items)
            stage_timings[ProcessingStage.IMPORTANCE_SCORING] = time.time() - stage_start
            
            logger.info(f"Importance scoring completed for {len(scored_items)} items")
            
            # 4단계: 토픽 분류
            stage_start = time.time()
            classified_items = await self._classify_topics(scored_items)
            stage_timings[ProcessingStage.TOPIC_CLASSIFICATION] = time.time() - stage_start
            
            logger.info(f"Topic classification completed for {len(classified_items)} items")
            
            # 5단계: 사전 필터링
            stage_start = time.time()
            filtered_items = await self._apply_prefiltering(classified_items)
            stage_timings[ProcessingStage.PRE_FILTERING] = time.time() - stage_start
            
            logger.info(f"After pre-filtering: {len(filtered_items)} items")
            
            # 6단계: 최종 처리 및 우선순위 정렬
            stage_start = time.time()
            final_items = await self._final_processing(filtered_items)
            stage_timings[ProcessingStage.FINAL_PROCESSING] = time.time() - stage_start
            
            # 결과 생성
            result = self._create_collection_result(
                original_count=len(raw_news_items),
                final_items=final_items,
                stage_timings=stage_timings,
                total_time=time.time() - overall_start_time
            )
            
            # 통계 업데이트
            self._update_comprehensive_stats(result)
            
            logger.info(f"Comprehensive collection completed: "
                       f"{result.original_count} -> {result.deduplicated_count} "
                       f"(efficiency: {result.processing_efficiency:.1%}, "
                       f"FinBERT candidates: {result.finbert_analysis_count})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive collection: {e}")
            return self._create_empty_result(stage_timings, overall_start_time)
    
    async def _collect_from_all_sources(self, symbol: str, hours_back: int, max_per_source: int) -> List[EnhancedNewsItem]:
        """모든 소스에서 뉴스 수집"""
        tasks = [
            self._collect_google_news_enhanced(symbol, hours_back, max_per_source),
            self._collect_yahoo_finance_enhanced(symbol, hours_back, max_per_source),
            self._collect_newsapi_enhanced(symbol, hours_back, max_per_source),
            self._collect_finnhub_enhanced(symbol, hours_back, max_per_source)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection task {i} failed: {result}")
                self.stats["failed_requests"] += 1
            elif isinstance(result, list):
                all_items.extend(result)
                self.stats["successful_requests"] += 1
        
        return all_items
    
    async def _advanced_deduplication(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """고급 중복 제거"""
        if not self.deduplicator or not self.enable_advanced_deduplication:
            # 기본 해시 기반 중복 제거
            return self._basic_deduplication(news_items)
        
        try:
            # EnhancedNewsItem을 딕셔너리로 변환
            items_dict = [item.to_dict() for item in news_items]
            
            # 고급 중복 제거 실행
            dedup_result = self.deduplicator.deduplicate_news_batch(items_dict)
            
            # 중복이 제거된 아이템들만 반환
            deduplicated_items = []
            for item in news_items:
                if item.hash_id not in dedup_result.removed_news_ids:
                    item.processing_stage = ProcessingStage.DEDUPLICATION
                    deduplicated_items.append(item)
            
            self.stats["duplicates_filtered"] += (len(news_items) - len(deduplicated_items))
            
            return deduplicated_items
            
        except Exception as e:
            logger.error(f"Advanced deduplication failed: {e}")
            return self._basic_deduplication(news_items)
    
    def _basic_deduplication(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """기본 해시 기반 중복 제거"""
        seen_hashes = set()
        deduplicated = []
        
        for item in news_items:
            if item.hash_id not in seen_hashes:
                seen_hashes.add(item.hash_id)
                item.processing_stage = ProcessingStage.DEDUPLICATION
                deduplicated.append(item)
        
        self.stats["duplicates_filtered"] += (len(news_items) - len(deduplicated))
        return deduplicated
    
    async def _calculate_importance_scores(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """중요도 점수 계산"""
        if not self.importance_scorer or not self.enable_importance_scoring:
            # 기본 relevance_score 사용
            for item in news_items:
                item.processing_stage = ProcessingStage.IMPORTANCE_SCORING
            return news_items
        
        try:
            for item in news_items:
                # 중요도 점수 계산
                importance_features = self.importance_scorer.calculate_importance_score(
                    title=item.title,
                    content=item.content,
                    source_url=item.url,
                    published_at=item.published_at,
                    metadata=item.metadata
                )
                
                item.importance_features = importance_features
                item.processing_stage = ProcessingStage.IMPORTANCE_SCORING
                
                # 메타데이터 업데이트
                item.metadata.update({
                    "importance_breakdown": {
                        "source_score": importance_features.source_score,
                        "policy_keyword_score": importance_features.policy_keyword_score,
                        "content_quality_score": importance_features.content_quality_score,
                        "timing_score": importance_features.timing_score
                    }
                })
            
            self.stats["importance_scored"] += len(news_items)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Importance scoring failed: {e}")
            return news_items
    
    async def _classify_topics(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """토픽 분류 단계"""
        if not self.enable_topic_classification or not self.topic_classifier:
            logger.debug("Topic classification disabled")
            return news_items
        
        logger.info(f"Classifying topics for {len(news_items)} items")
        
        try:
            for item in news_items:
                # 토픽 분류 수행
                metadata = {
                    'source': item.source.value,
                    'category': item.category,
                    'published_at': item.published_at,
                    'importance_score': item.importance_features.total_score if item.importance_features else 0.5
                }
                
                topic_classification = self.topic_classifier.classify(
                    title=item.title,
                    content=item.content,
                    metadata=metadata
                )
                
                item.topic_classification = topic_classification
                item.processing_stage = ProcessingStage.TOPIC_CLASSIFICATION
                
                # 메타데이터 업데이트
                item.metadata.update({
                    "topic": {
                        "primary": topic_classification.primary_topic.value,
                        "secondary": [t.value for t in topic_classification.secondary_topics],
                        "confidence": topic_classification.confidence_scores.get(
                            topic_classification.primary_topic, 0
                        ),
                        "is_multi_topic": topic_classification.is_multi_topic,
                        "keywords_found": topic_classification.keywords_found
                    }
                })
                
                # 토픽별 통계 업데이트
                topic_name = topic_classification.primary_topic.value
                if topic_name not in self.stats:
                    self.stats[topic_name] = 0
                self.stats[topic_name] += 1
            
            self.stats["topics_classified"] = self.stats.get("topics_classified", 0) + len(news_items)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            return news_items
    
    async def _apply_prefiltering(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """사전 필터링 적용"""
        if not self.prefilter or not self.enable_prefiltering:
            # 기본 필터링 (중요도 기반)
            filtered = []
            for item in news_items:
                if item.final_importance_score >= 0.2:  # 기본 임계값
                    item.processing_stage = ProcessingStage.PRE_FILTERING
                    filtered.append(item)
            return filtered
        
        try:
            # 뉴스 아이템을 딕셔너리로 변환
            items_dict = [item.to_dict() for item in news_items]
            importance_scores = [item.final_importance_score for item in news_items]
            
            # 사전 필터링 실행
            filter_results = self.prefilter.filter_news_batch(items_dict, importance_scores)
            
            # 필터링 결과 적용
            filtered_items = []
            for item, filter_result in zip(news_items, filter_results):
                item.filter_result = filter_result
                item.processing_stage = ProcessingStage.PRE_FILTERING
                
                # 승인된 뉴스만 포함
                if filter_result.decision in ["approve_high_priority", "approve_normal", "approve_low_priority"]:
                    filtered_items.append(item)
            
            self.stats["prefiltered"] += len(filtered_items)
            
            return filtered_items
            
        except Exception as e:
            logger.error(f"Pre-filtering failed: {e}")
            # 기본 필터링으로 폴백
            return news_items
    
    async def _final_processing(self, news_items: List[EnhancedNewsItem]) -> List[EnhancedNewsItem]:
        """최종 처리 및 정렬"""
        for item in news_items:
            item.processing_stage = ProcessingStage.FINAL_PROCESSING
            
            # FinBERT 분석 필요 여부 최종 결정
            if item.should_analyze_with_finbert:
                self.stats["finbert_candidates"] += 1
        
        # 우선순위로 정렬 (중요도 점수 + 품질 점수)
        news_items.sort(key=lambda x: (
            -x.final_importance_score,
            -(x.filter_result.quality_score if x.filter_result else 0.5),
            x.published_at
        ), reverse=True)
        
        return news_items
    
    # 개별 소스 수집 함수들 (Enhanced 버전)
    async def _collect_google_news_enhanced(self, symbol: str, hours_back: int, max_results: int) -> List[EnhancedNewsItem]:
        """Google News 수집 (Enhanced)"""
        if not self._check_rate_limit(NewsSource.GOOGLE_NEWS):
            return []
        
        try:
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            search_query = quote_plus(" OR ".join(keywords))
            rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            
            content = await self._fetch_with_retry(rss_url)
            items = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            root = ET.fromstring(content)
            for item_elem in root.findall('.//item')[:max_results]:
                try:
                    news_item = self._parse_google_news_item_enhanced(item_elem, cutoff_time, symbol, keywords)
                    if news_item and news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                except Exception as e:
                    logger.debug(f"Error parsing Google News item: {e}")
                    continue
            
            self.stats["total_items_collected"] += len(items)
            return items
            
        except Exception as e:
            logger.error(f"Google News enhanced collection failed: {e}")
            self._record_error(NewsSource.GOOGLE_NEWS, str(e))
            return []
    
    def _parse_google_news_item_enhanced(self, item: ET.Element, cutoff_time: datetime, 
                                       symbol: str, keywords: List[str]) -> Optional[EnhancedNewsItem]:
        """Google News 아이템 파싱 (Enhanced)"""
        title_elem = item.find('title')
        if title_elem is None or not title_elem.text:
            return None
        
        title = title_elem.text.strip()
        
        link_elem = item.find('link')
        url = link_elem.text if link_elem is not None else ""
        
        pub_date_elem = item.find('pubDate')
        if pub_date_elem is not None and pub_date_elem.text:
            try:
                pub_datetime = datetime.strptime(
                    pub_date_elem.text, '%a, %d %b %Y %H:%M:%S GMT'
                )
            except ValueError:
                pub_datetime = datetime.now()
        else:
            pub_datetime = datetime.now()
        
        if pub_datetime < cutoff_time:
            return None
        
        description_elem = item.find('description')
        content = description_elem.text if description_elem is not None else ""
        
        if content:
            content = re.sub('<[^<]+?>', '', content)
        
        # Enhanced 속성들
        entities = self._extract_entities_enhanced(f"{title} {content}")
        relevance_score = self._calculate_relevance_score_enhanced(title, keywords)
        
        return EnhancedNewsItem(
            title=title,
            content=content,
            url=url,
            source=NewsSource.GOOGLE_NEWS,
            published_at=pub_datetime,
            category="crypto" if symbol.upper() in ["BTC", "ETH", "CRYPTO"] else "finance",
            symbol=symbol.upper(),
            entities=entities,
            relevance_score=relevance_score,
            metadata={
                "keywords_matched": [kw for kw in keywords if kw.lower() in title.lower() or kw.lower() in content.lower()],
                "source_quality": "tier_3"  # Google News는 Tier 3
            }
        )
    
    async def _collect_yahoo_finance_enhanced(self, symbol: str, hours_back: int, max_results: int) -> List[EnhancedNewsItem]:
        """Yahoo Finance 수집 (Enhanced) - 간소화된 구현"""
        if not self._check_rate_limit(NewsSource.YAHOO_FINANCE):
            return []
        
        try:
            rss_urls = {
                "BTC": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
                "ETH": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US", 
                "DEFAULT": "https://feeds.finance.yahoo.com/rss/2.0/headline"
            }
            
            rss_url = rss_urls.get(symbol.upper(), rss_urls["DEFAULT"])
            content = await self._fetch_with_retry(rss_url)
            
            items = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            root = ET.fromstring(content)
            for item_elem in root.findall('.//item')[:max_results]:
                try:
                    news_item = self._parse_yahoo_finance_item_enhanced(item_elem, cutoff_time, symbol)
                    if news_item and news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                except Exception as e:
                    logger.debug(f"Error parsing Yahoo Finance item: {e}")
                    continue
            
            self.stats["total_items_collected"] += len(items)
            return items
            
        except Exception as e:
            logger.error(f"Yahoo Finance enhanced collection failed: {e}")
            self._record_error(NewsSource.YAHOO_FINANCE, str(e))
            return []
    
    def _parse_yahoo_finance_item_enhanced(self, item: ET.Element, cutoff_time: datetime, 
                                         symbol: str) -> Optional[EnhancedNewsItem]:
        """Yahoo Finance 아이템 파싱 (Enhanced)"""
        title_elem = item.find('title')
        if title_elem is None or not title_elem.text:
            return None
        
        title = title_elem.text.strip()
        
        link_elem = item.find('link')
        url = link_elem.text if link_elem is not None else ""
        
        pub_date_elem = item.find('pubDate') 
        if pub_date_elem is not None and pub_date_elem.text:
            try:
                pub_datetime = datetime.strptime(
                    pub_date_elem.text, '%a, %d %b %Y %H:%M:%S %z'
                ).replace(tzinfo=None)
            except ValueError:
                pub_datetime = datetime.now()
        else:
            pub_datetime = datetime.now()
        
        if pub_datetime < cutoff_time:
            return None
            
        description_elem = item.find('description')
        content = description_elem.text if description_elem is not None else ""
        
        if content:
            content = re.sub('<[^<]+?>', '', content).strip()
        
        keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
        
        return EnhancedNewsItem(
            title=title,
            content=content,
            url=url,
            source=NewsSource.YAHOO_FINANCE,
            published_at=pub_datetime,
            category="finance",
            symbol=symbol.upper(),
            entities=self._extract_entities_enhanced(f"{title} {content}"),
            relevance_score=self._calculate_relevance_score_enhanced(title, keywords),
            metadata={
                "keywords_matched": [kw for kw in keywords if kw.lower() in title.lower() or kw.lower() in content.lower()],
                "source_quality": "tier_3"
            }
        )
    
    async def _collect_newsapi_enhanced(self, symbol: str, hours_back: int, max_results: int) -> List[EnhancedNewsItem]:
        """NewsAPI 수집 (Enhanced) - 간소화된 구현"""
        if not self.api_keys.get("newsapi_key") or not self._check_rate_limit(NewsSource.NEWSAPI):
            return []
        
        try:
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            query = " OR ".join(keywords)
            domains = "reuters.com,bloomberg.com,cnbc.com,marketwatch.com,wsj.com"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "domains": domains,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(max_results, 100),
                "from": (datetime.now() - timedelta(hours=hours_back)).isoformat(),
                "apiKey": self.api_keys["newsapi_key"]
            }
            
            response_text = await self._fetch_with_retry(url, params)
            data = json.loads(response_text)
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            items = []
            for article in data.get("articles", []):
                try:
                    news_item = self._parse_newsapi_article_enhanced(article, symbol, keywords)
                    if news_item and news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                except Exception as e:
                    logger.debug(f"Error parsing NewsAPI article: {e}")
                    continue
            
            self.stats["total_items_collected"] += len(items)
            return items
            
        except Exception as e:
            logger.error(f"NewsAPI enhanced collection failed: {e}")
            self._record_error(NewsSource.NEWSAPI, str(e))
            return []
    
    def _parse_newsapi_article_enhanced(self, article: Dict, symbol: str, keywords: List[str]) -> Optional[EnhancedNewsItem]:
        """NewsAPI 기사 파싱 (Enhanced)"""
        title = article.get("title", "")
        if "[Removed]" in title or not title.strip():
            return None
        
        content = article.get("description", "") or article.get("content", "")
        url = article.get("url", "")
        
        published_at_str = article.get("publishedAt", "")
        if published_at_str:
            try:
                published_at = datetime.fromisoformat(
                    published_at_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except ValueError:
                published_at = datetime.now()
        else:
            published_at = datetime.now()
        
        return EnhancedNewsItem(
            title=title,
            content=content,
            url=url,
            source=NewsSource.NEWSAPI,
            published_at=published_at,
            category="crypto" if symbol.upper() in ["BTC", "ETH", "CRYPTO"] else "finance",
            symbol=symbol.upper(),
            entities=self._extract_entities_enhanced(f"{title} {content}"),
            relevance_score=self._calculate_relevance_score_enhanced(title, keywords),
            metadata={
                "keywords_matched": [kw for kw in keywords if kw.lower() in title.lower() or kw.lower() in content.lower()],
                "source_quality": "tier_1"  # NewsAPI는 Tier 1 소스들
            }
        )
    
    async def _collect_finnhub_enhanced(self, symbol: str, hours_back: int, max_results: int) -> List[EnhancedNewsItem]:
        """Finnhub 수집 (Enhanced) - 간소화된 구현"""
        if not self.api_keys.get("finnhub_key") or not self._check_rate_limit(NewsSource.FINNHUB):
            return []
        
        try:
            category = "crypto" if symbol.upper() in ["BTC", "ETH", "CRYPTO"] else "general"
            
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": category,
                "token": self.api_keys["finnhub_key"]
            }
            
            response_text = await self._fetch_with_retry(url, params)
            articles = json.loads(response_text)
            
            if not isinstance(articles, list):
                logger.error("Finnhub returned invalid response format")
                return []
            
            items = []
            cutoff_timestamp = (datetime.now() - timedelta(hours=hours_back)).timestamp()
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            
            for article in articles[:max_results]:
                try:
                    if article.get("datetime", 0) < cutoff_timestamp:
                        continue
                    
                    news_item = self._parse_finnhub_article_enhanced(article, symbol, keywords, category)
                    if news_item and news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                except Exception as e:
                    logger.debug(f"Error parsing Finnhub article: {e}")
                    continue
            
            self.stats["total_items_collected"] += len(items)
            return items
            
        except Exception as e:
            logger.error(f"Finnhub enhanced collection failed: {e}")
            self._record_error(NewsSource.FINNHUB, str(e))
            return []
    
    def _parse_finnhub_article_enhanced(self, article: Dict, symbol: str, keywords: List[str], category: str) -> Optional[EnhancedNewsItem]:
        """Finnhub 기사 파싱 (Enhanced)"""
        title = article.get("headline", "")
        if not title.strip():
            return None
        
        content = article.get("summary", "")
        url = article.get("url", "")
        published_at = datetime.fromtimestamp(article.get("datetime", time.time()))
        
        return EnhancedNewsItem(
            title=title,
            content=content,
            url=url,
            source=NewsSource.FINNHUB,
            published_at=published_at,
            category=category,
            symbol=symbol.upper(),
            entities=self._extract_entities_enhanced(f"{title} {content}"),
            relevance_score=self._calculate_relevance_score_enhanced(title, keywords),
            metadata={
                "keywords_matched": [kw for kw in keywords if kw.lower() in title.lower() or kw.lower() in content.lower()],
                "source_quality": "tier_2"
            }
        )
    
    # 헬퍼 함수들
    def _extract_entities_enhanced(self, text: str, max_entities: int = 15) -> List[str]:
        """Enhanced 엔티티 추출"""
        if not text:
            return []
        
        # 확장된 엔티티 패턴
        important_terms = {
            "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto",
            "blockchain", "defi", "nft", "sec", "fed", "fomc", "cpi", "ppi",
            "etf", "regulation", "tesla", "apple", "microsoft", "nvidia",
            "powell", "yellen", "biden", "trump", "blackrock", "grayscale",
            "coinbase", "binance", "ftx", "tether", "usdt", "usdc"
        }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        entities = []
        seen = set()
        
        for word in words:
            if len(word) >= 3 and word in important_terms and word not in seen:
                entities.append(word)
                seen.add(word)
                
                if len(entities) >= max_entities:
                    break
        
        return entities
    
    def _calculate_relevance_score_enhanced(self, text: str, keywords: List[str]) -> float:
        """Enhanced 관련성 점수 계산"""
        if not keywords or not text:
            return 0.5
        
        text_lower = text.lower()
        
        # 키워드 매칭 점수
        exact_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        partial_matches = sum(1 for keyword in keywords 
                            for word in keyword.lower().split() 
                            if word in text_lower)
        
        # 가중 점수 계산
        exact_score = min(0.8, exact_matches / len(keywords) * 0.8)
        partial_score = min(0.4, partial_matches / (len(keywords) * 2) * 0.4)
        
        return min(1.0, 0.1 + exact_score + partial_score)
    
    def _check_rate_limit(self, source: NewsSource) -> bool:
        """Rate limit 확인"""
        now = time.time()
        rate_info = self.rate_limits.get(source)
        
        if not rate_info:
            return True
        
        if now >= rate_info["reset_time"]:
            rate_info["requests"] = 0
            if source in [NewsSource.NEWSAPI, NewsSource.FINNHUB]:
                rate_info["reset_time"] = now + 3600
            else:
                rate_info["reset_time"] = now + 60
        
        if rate_info["requests"] >= rate_info["limit"]:
            remaining_time = rate_info["reset_time"] - now
            logger.warning(f"Rate limit exceeded for {source.value}. Reset in {remaining_time:.0f}s")
            return False
        
        rate_info["requests"] += 1
        return True
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _fetch_with_retry(self, url: str, params: Optional[Dict] = None) -> str:
        """재시도 로직이 포함된 HTTP 요청"""
        async with self.semaphore:
            self.stats["total_requests"] += 1
            
            try:
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    content = await response.text()
                    self.stats["successful_requests"] += 1
                    return content
            except Exception as e:
                self.stats["failed_requests"] += 1
                logger.error(f"Request failed for {url}: {e}")
                raise
    
    def _record_error(self, source: NewsSource, error_msg: str):
        """에러 기록"""
        source_name = source.value
        if source_name not in self.stats["errors_by_source"]:
            self.stats["errors_by_source"][source_name] = []
        
        self.stats["errors_by_source"][source_name].append({
            "timestamp": datetime.now().isoformat(),
            "error": error_msg[:200]
        })
        
        # 최근 10개 에러만 유지
        if len(self.stats["errors_by_source"][source_name]) > 10:
            self.stats["errors_by_source"][source_name] = \
                self.stats["errors_by_source"][source_name][-10:]
    
    # 결과 생성 함수들
    def _create_collection_result(self, original_count: int, final_items: List[EnhancedNewsItem],
                                stage_timings: Dict[ProcessingStage, float], total_time: float) -> CollectionResult:
        """수집 결과 생성"""
        # 우선순위별 카운트
        high_priority = sum(1 for item in final_items 
                          if item.filter_result and item.filter_result.decision == "approve_high_priority")
        normal_priority = sum(1 for item in final_items
                            if item.filter_result and item.filter_result.decision == "approve_normal")
        low_priority = sum(1 for item in final_items
                         if item.filter_result and item.filter_result.decision == "approve_low_priority")
        rejected = original_count - len(final_items)
        
        # 토픽 분포 계산
        topic_distribution = {}
        for item in final_items:
            if item.topic_classification:
                topic = item.topic_classification.primary_topic.value
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        return CollectionResult(
            original_count=original_count,
            deduplicated_count=len(final_items),
            high_priority_count=high_priority,
            normal_priority_count=normal_priority,
            low_priority_count=low_priority,
            rejected_count=rejected,
            total_processing_time=total_time,
            stage_timings=stage_timings,
            news_items=final_items,
            topic_distribution=topic_distribution
        )
    
    def _create_empty_result(self, stage_timings: Dict[ProcessingStage, float], total_time: float) -> CollectionResult:
        """빈 결과 생성"""
        return CollectionResult(
            original_count=0,
            deduplicated_count=0,
            high_priority_count=0,
            normal_priority_count=0,
            low_priority_count=0,
            rejected_count=0,
            total_processing_time=total_time,
            stage_timings=stage_timings,
            news_items=[]
        )
    
    def _update_comprehensive_stats(self, result: CollectionResult):
        """종합 통계 업데이트"""
        self.stats["last_collection"] = datetime.now().isoformat()
        self.stats["processing_times"].update({
            stage.value: timing for stage, timing in result.stage_timings.items()
        })
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Enhanced 통계 반환"""
        return {
            "v3_features": {
                "importance_scoring_enabled": self.enable_importance_scoring,
                "prefiltering_enabled": self.enable_prefiltering,
                "advanced_deduplication_enabled": self.enable_advanced_deduplication,
                "topic_classification_enabled": self.enable_topic_classification
            },
            "collection_stats": self.stats,
            "system_status": {
                "importance_scorer_available": self.importance_scorer is not None,
                "deduplicator_available": self.deduplicator is not None,
                "prefilter_available": self.prefilter is not None,
                "topic_classifier_available": self.topic_classifier is not None
            }
        }

# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_collector_v3():
        """Enhanced News Collector V3 테스트"""
        api_keys = {
            "newsapi_key": "your_newsapi_key_here",
            "finnhub_key": "your_finnhub_key_here"
        }
        
        async with EnhancedNewsCollectorV3(api_keys) as collector:
            print("=== Enhanced News Collector V3 Test ===\n")
            
            # 종합 뉴스 수집 테스트
            result = await collector.collect_comprehensive_news(
                symbol="BTC", 
                hours_back=6, 
                max_per_source=5
            )
            
            print(f"Collection Results:")
            print(f"  Original: {result.original_count}")
            print(f"  Deduplicated: {result.deduplicated_count}")
            print(f"  High Priority: {result.high_priority_count}")
            print(f"  Normal Priority: {result.normal_priority_count}")
            print(f"  Low Priority: {result.low_priority_count}")
            print(f"  Rejected: {result.rejected_count}")
            print(f"  FinBERT Candidates: {result.finbert_analysis_count}")
            print(f"  Processing Efficiency: {result.processing_efficiency:.1%}")
            print(f"  Total Time: {result.total_processing_time:.3f}s")
            
            print(f"\nStage Timings:")
            for stage, timing in result.stage_timings.items():
                print(f"  {stage.value}: {timing:.3f}s")
            
            print(f"\nTop 3 News Items:")
            for i, item in enumerate(result.news_items[:3], 1):
                print(f"\n{i}. [{item.source.value}] {item.title[:60]}...")
                print(f"   Importance: {item.final_importance_score:.3f}")
                print(f"   FinBERT Analysis: {item.should_analyze_with_finbert}")
                if item.filter_result:
                    print(f"   Filter Decision: {item.filter_result.decision}")
                if item.topic_classification:
                    print(f"   Topic: {item.topic_classification.primary_topic.value} "
                          f"(confidence: {item.topic_classification.confidence_scores.get(item.topic_classification.primary_topic, 0):.2f})")
            
            # 토픽 분포 출력
            if result.topic_distribution:
                print(f"\nTopic Distribution:")
                for topic, count in sorted(result.topic_distribution.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / result.deduplicated_count) * 100 if result.deduplicated_count > 0 else 0
                    print(f"  {topic}: {count} ({percentage:.1f}%)")
            
            # 통계 출력
            stats = collector.get_enhanced_stats()
            print(f"\n=== Enhanced Statistics ===")
            print(json.dumps(stats, indent=2, default=str))
    
    # 테스트 실행
    asyncio.run(test_enhanced_collector_v3())