#!/usr/bin/env python3
"""
Base News Collector Interface
모든 뉴스 수집기의 기본 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import logging


class NewsCategory(Enum):
    """뉴스 카테고리"""
    HEADLINE = "headline"  # 주요 헤드라인
    CRYPTO = "crypto"  # 암호화폐
    FINANCE = "finance"  # 금융/경제
    MACRO = "macro"  # 거시경제
    BREAKING = "breaking"  # 속보
    PERSON = "person"  # 주요 인물
    COMMUNITY = "community"  # 커뮤니티/소셜


class SentimentScore(Enum):
    """감정 점수"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsArticle:
    """뉴스 기사 데이터"""
    id: str
    title: str
    content: str
    summary: str
    url: str
    source: str
    author: Optional[str]
    published_date: datetime
    collected_date: datetime
    category: NewsCategory
    keywords: List[str]
    entities: List[str]  # 언급된 주요 인물/조직
    sentiment_score: Optional[float] = None
    
    @property
    def published(self) -> datetime:
        """호환성을 위한 published 속성"""
        return self.published_date
    sentiment_label: Optional[SentimentScore] = None
    relevance_score: Optional[float] = None  # 0-1 관련성 점수
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "author": self.author,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "collected_date": self.collected_date.isoformat() if self.collected_date else None,
            "category": self.category.value,
            "keywords": self.keywords,
            "entities": self.entities,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label.value if self.sentiment_label else None,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata or {}
        }


@dataclass
class CollectorConfig:
    """수집기 설정"""
    api_key: Optional[str] = None
    rate_limit: int = 100  # 시간당 요청 수
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: int = 300  # 5분
    max_results: int = 100


class BaseNewsCollector(ABC):
    """뉴스 수집기 기본 클래스"""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}
        self._rate_limiter = []
        self.stats = {
            "requests_made": 0,
            "articles_collected": 0,
            "errors": 0,
            "cache_hits": 0
        }
    
    @abstractmethod
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """주요 헤드라인 수집"""
        pass
    
    @abstractmethod
    async def search_news(self, keywords: List[str], 
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         count: int = 20,
                         category: Optional[NewsCategory] = None,
                         **kwargs) -> List[NewsArticle]:
        """키워드 기반 뉴스 검색"""
        pass
    
    @abstractmethod
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """속보 수집"""
        pass
    
    async def collect_by_category(self, category: NewsCategory, 
                                 count: int = 20) -> List[NewsArticle]:
        """카테고리별 뉴스 수집"""
        # 기본 구현 - 하위 클래스에서 오버라이드 가능
        category_keywords = {
            NewsCategory.CRYPTO: ["bitcoin", "ethereum", "crypto", "blockchain"],
            NewsCategory.FINANCE: ["stock", "market", "economy", "finance"],
            NewsCategory.MACRO: ["FOMC", "CPI", "GDP", "inflation", "central bank"],
            NewsCategory.PERSON: ["CEO", "president", "minister", "chairman"]
        }
        
        keywords = category_keywords.get(category, [])
        if keywords:
            return await self.search_news(keywords, count=count)
        return []
    
    async def analyze_sentiment(self, article: NewsArticle) -> NewsArticle:
        """기본 감정 분석 (하위 클래스에서 구현)"""
        # 간단한 키워드 기반 분석 (실제로는 더 정교한 분석 필요)
        positive_words = ["gain", "rise", "success", "profit", "growth", "bullish", "upgrade"]
        negative_words = ["loss", "fall", "fail", "crash", "bearish", "downgrade", "crisis"]
        
        text = (article.title + " " + article.summary).lower()
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            article.sentiment_score = 0.5 + (positive_count / (positive_count + negative_count)) * 0.5
            article.sentiment_label = SentimentScore.POSITIVE
        elif negative_count > positive_count:
            article.sentiment_score = -0.5 - (negative_count / (positive_count + negative_count)) * 0.5
            article.sentiment_label = SentimentScore.NEGATIVE
        else:
            article.sentiment_score = 0.0
            article.sentiment_label = SentimentScore.NEUTRAL
        
        return article
    
    def extract_entities(self, text: str) -> List[str]:
        """주요 엔티티 추출 (간단한 구현)"""
        # 실제로는 NER (Named Entity Recognition) 사용
        entities = []
        
        # 주요 인물/조직 패턴
        patterns = [
            "CEO", "President", "Chairman", "Minister",
            "Federal Reserve", "SEC", "ECB", "Bank of",
            "Bitcoin", "Ethereum", "BTC", "ETH"
        ]
        
        for pattern in patterns:
            if pattern.lower() in text.lower():
                entities.append(pattern)
        
        return list(set(entities))
    
    def calculate_relevance(self, article: NewsArticle, 
                          target_keywords: List[str]) -> float:
        """관련성 점수 계산"""
        if not target_keywords:
            return 1.0
        
        text = (article.title + " " + article.summary + " " + article.content).lower()
        matched = sum(1 for keyword in target_keywords if keyword.lower() in text)
        
        return min(1.0, matched / len(target_keywords))
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        return {
            "status": "healthy",
            "collector": self.__class__.__name__,
            "stats": self.stats.copy(),
            "config": {
                "rate_limit": self.config.rate_limit,
                "cache_ttl": self.config.cache_ttl
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return self.stats.copy()
    
    async def close(self):
        """리소스 정리"""
        self._cache.clear()
        self.logger.info(f"{self.__class__.__name__} closed")


class NewsAggregator:
    """여러 수집기를 통합 관리하는 클래스"""
    
    def __init__(self):
        self.collectors: Dict[str, BaseNewsCollector] = {}
        self.logger = logging.getLogger("NewsAggregator")
    
    def register_collector(self, name: str, collector: BaseNewsCollector):
        """수집기 등록"""
        self.collectors[name] = collector
        self.logger.info(f"Registered collector: {name}")
    
    async def collect_all_headlines(self, count_per_source: int = 10) -> List[NewsArticle]:
        """모든 소스에서 헤드라인 수집"""
        all_articles = []
        
        for name, collector in self.collectors.items():
            try:
                articles = await collector.collect_headlines(count_per_source)
                all_articles.extend(articles)
                self.logger.info(f"Collected {len(articles)} headlines from {name}")
            except Exception as e:
                self.logger.error(f"Error collecting from {name}: {e}")
        
        # 중복 제거 (URL 기준)
        unique_articles = {}
        for article in all_articles:
            if article.url not in unique_articles:
                unique_articles[article.url] = article
        
        # 발행일 기준 정렬
        sorted_articles = sorted(
            unique_articles.values(), 
            key=lambda x: x.published_date, 
            reverse=True
        )
        
        return sorted_articles
    
    async def search_all_sources(self, keywords: List[str], 
                                since: Optional[datetime] = None,
                                count_per_source: int = 10) -> List[NewsArticle]:
        """모든 소스에서 키워드 검색"""
        all_articles = []
        
        for name, collector in self.collectors.items():
            try:
                articles = await collector.search_news(
                    keywords, since=since, count=count_per_source
                )
                all_articles.extend(articles)
                self.logger.info(f"Found {len(articles)} articles from {name}")
            except Exception as e:
                self.logger.error(f"Error searching {name}: {e}")
        
        return self._deduplicate_and_sort(all_articles)
    
    async def get_breaking_news_all(self, minutes: int = 30) -> List[NewsArticle]:
        """모든 소스에서 속보 수집"""
        all_breaking = []
        
        for name, collector in self.collectors.items():
            try:
                breaking = await collector.get_breaking_news(minutes)
                all_breaking.extend(breaking)
                self.logger.info(f"Found {len(breaking)} breaking news from {name}")
            except Exception as e:
                self.logger.error(f"Error getting breaking news from {name}: {e}")
        
        return self._deduplicate_and_sort(all_breaking)
    
    def _deduplicate_and_sort(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """중복 제거 및 정렬"""
        # URL 기준 중복 제거
        unique_articles = {}
        for article in articles:
            if article.url not in unique_articles:
                unique_articles[article.url] = article
        
        # 발행일 기준 정렬
        return sorted(
            unique_articles.values(),
            key=lambda x: x.published_date,
            reverse=True
        )
    
    async def analyze_market_sentiment(self, hours: int = 6) -> Dict[str, Any]:
        """시장 전체 감정 분석"""
        since = datetime.now() - timedelta(hours=hours)
        
        # 주요 키워드로 뉴스 수집
        crypto_news = await self.search_all_sources(
            ["bitcoin", "ethereum", "crypto"], 
            since=since
        )
        
        # 감정 분석
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        
        for article in crypto_news:
            if article.sentiment_label:
                if article.sentiment_label.value > 0:
                    sentiments["positive"] += 1
                elif article.sentiment_label.value < 0:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1
        
        total = sum(sentiments.values())
        
        return {
            "period_hours": hours,
            "total_articles": total,
            "sentiment_distribution": sentiments,
            "overall_sentiment": (sentiments["positive"] - sentiments["negative"]) / total if total > 0 else 0,
            "top_sources": self._get_top_sources(crypto_news),
            "top_keywords": self._extract_top_keywords(crypto_news)
        }
    
    def _get_top_sources(self, articles: List[NewsArticle], top_n: int = 5) -> List[Dict[str, int]]:
        """상위 소스 추출"""
        source_counts = {}
        for article in articles:
            source_counts[article.source] = source_counts.get(article.source, 0) + 1
        
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"source": source, "count": count} for source, count in sorted_sources[:top_n]]
    
    def _extract_top_keywords(self, articles: List[NewsArticle], top_n: int = 10) -> List[Dict[str, int]]:
        """상위 키워드 추출"""
        keyword_counts = {}
        for article in articles:
            for keyword in article.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": kw, "count": count} for kw, count in sorted_keywords[:top_n]]
    
    async def close_all(self):
        """모든 수집기 종료"""
        for name, collector in self.collectors.items():
            try:
                await collector.close()
                self.logger.info(f"Closed collector: {name}")
            except Exception as e:
                self.logger.error(f"Error closing {name}: {e}")