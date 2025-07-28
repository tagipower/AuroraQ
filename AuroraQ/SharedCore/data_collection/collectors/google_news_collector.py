#!/usr/bin/env python3
"""
Google News RSS Collector
무료로 사용 가능한 Google News RSS 피드 수집기
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
from urllib.parse import quote_plus
import logging

from ..base_collector import (
    BaseNewsCollector, NewsArticle, NewsCategory, 
    CollectorConfig, SentimentScore
)


class GoogleNewsCollector(BaseNewsCollector):
    """Google News RSS 피드 수집기"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        if not config:
            config = CollectorConfig(
                rate_limit=1000,  # RSS는 제한이 느슨함
                timeout=30.0,
                cache_ttl=600  # 10분 캐시
            )
        super().__init__(config)
        
        self.base_url = "https://news.google.com/rss"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 지원 언어 및 지역
        self.language = "en"
        self.country = "US"
        
        # 카테고리 매핑
        self.category_mapping = {
            NewsCategory.HEADLINE: "NATION",
            NewsCategory.FINANCE: "BUSINESS", 
            NewsCategory.CRYPTO: "TECHNOLOGY",
            NewsCategory.BREAKING: "NATION"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _fetch_rss(self, url: str) -> Optional[str]:
        """RSS 피드 가져오기"""
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.error(f"RSS fetch failed: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching RSS: {e}")
            self.stats["errors"] += 1
            return None
    
    def _parse_google_news_item(self, item: Dict[str, Any]) -> Optional[NewsArticle]:
        """Google News RSS 아이템 파싱"""
        try:
            # 제목에서 소스 분리
            title_parts = item.get('title', '').split(' - ')
            if len(title_parts) >= 2:
                title = ' - '.join(title_parts[:-1])
                source = title_parts[-1]
            else:
                title = item.get('title', '')
                source = 'Google News'
            
            # 발행 시간 파싱
            pub_date_str = item.get('published', '')
            try:
                pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
            except:
                pub_date = datetime.now()
            
            # ID 생성 (URL 해시)
            url = item.get('link', '')
            article_id = hashlib.md5(url.encode()).hexdigest()
            
            # 요약 정리
            summary = item.get('description', '')
            if summary.startswith('<'):
                # HTML 태그 제거 (간단한 방법)
                import re
                summary = re.sub('<[^<]+?>', '', summary)
            
            # 키워드 추출 (제목과 요약에서)
            keywords = self._extract_keywords(title + " " + summary)
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=summary,  # RSS는 전체 내용을 제공하지 않음
                summary=summary[:500] if summary else "",
                url=url,
                source=source,
                author=None,
                published_date=pub_date,
                collected_date=datetime.now(),
                category=NewsCategory.HEADLINE,
                keywords=keywords,
                entities=self.extract_entities(title + " " + summary),
                metadata={"rss_source": "google_news"}
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Google News item: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출"""
        # 중요 단어들
        important_words = [
            "bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft",
            "stock", "market", "economy", "inflation", "recession",
            "fed", "fomc", "cpi", "gdp", "rate", "dollar",
            "breaking", "urgent", "alert", "crash", "surge"
        ]
        
        text_lower = text.lower()
        keywords = []
        
        for word in important_words:
            if word in text_lower:
                keywords.append(word)
        
        return keywords
    
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """주요 헤드라인 수집"""
        url = f"{self.base_url}?hl={self.language}-{self.country}&gl={self.country}&ceid={self.country}:{self.language}"
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_google_news_item(item)
            if article:
                # 감정 분석
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def search_news(self, keywords: List[str], 
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         count: int = 20) -> List[NewsArticle]:
        """키워드 기반 뉴스 검색"""
        # Google News RSS 검색 URL 생성
        query = " ".join(keywords)
        encoded_query = quote_plus(query)
        
        url = f"{self.base_url}/search?q={encoded_query}&hl={self.language}-{self.country}&gl={self.country}&ceid={self.country}:{self.language}"
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_google_news_item(item)
            if article:
                # 시간 필터링
                if since and article.published_date < since:
                    continue
                if until and article.published_date > until:
                    continue
                
                # 관련성 점수 계산
                article.relevance_score = self.calculate_relevance(article, keywords)
                
                # 감정 분석
                article = await self.analyze_sentiment(article)
                
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        # 관련성 점수로 정렬
        articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return articles
    
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """속보 수집"""
        # Google News는 별도의 속보 피드가 없으므로 최신 뉴스에서 필터링
        headlines = await self.collect_headlines(count=50)
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        breaking_news = []
        
        for article in headlines:
            if article.published_date >= cutoff_time:
                # 속보 키워드 확인
                breaking_keywords = ["breaking", "urgent", "alert", "just in", "update"]
                title_lower = article.title.lower()
                
                if any(kw in title_lower for kw in breaking_keywords):
                    article.category = NewsCategory.BREAKING
                    breaking_news.append(article)
        
        return breaking_news
    
    async def collect_by_category(self, category: NewsCategory, 
                                 count: int = 20) -> List[NewsArticle]:
        """카테고리별 뉴스 수집"""
        # Google News 카테고리 매핑
        google_category = self.category_mapping.get(category, "NATION")
        
        url = f"{self.base_url}/headlines/section/topic/{google_category}?hl={self.language}-{self.country}&gl={self.country}&ceid={self.country}:{self.language}"
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_google_news_item(item)
            if article:
                article.category = category
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def collect_crypto_news(self, count: int = 20) -> List[NewsArticle]:
        """암호화폐 관련 뉴스 특화 수집"""
        crypto_keywords = [
            "bitcoin", "ethereum", "cryptocurrency", "blockchain",
            "defi", "nft", "crypto market", "btc", "eth"
        ]
        
        articles = await self.search_news(crypto_keywords, count=count)
        
        # 카테고리 설정
        for article in articles:
            article.category = NewsCategory.CRYPTO
        
        return articles
    
    async def collect_macro_news(self, count: int = 20) -> List[NewsArticle]:
        """거시경제 뉴스 특화 수집"""
        macro_keywords = [
            "federal reserve", "fomc", "inflation", "cpi", "gdp",
            "interest rate", "central bank", "economic data", "job report"
        ]
        
        articles = await self.search_news(macro_keywords, count=count)
        
        # 카테고리 설정
        for article in articles:
            article.category = NewsCategory.MACRO
        
        return articles
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()


# 사용 예제
async def main():
    """Google News Collector 테스트"""
    collector = GoogleNewsCollector()
    
    try:
        # 헤드라인 수집
        print("📰 Collecting headlines...")
        headlines = await collector.collect_headlines(count=5)
        print(f"Found {len(headlines)} headlines")
        
        for article in headlines[:3]:
            print(f"\n- {article.title}")
            print(f"  Source: {article.source}")
            print(f"  Published: {article.published_date}")
            print(f"  Sentiment: {article.sentiment_label.name if article.sentiment_label else 'N/A'}")
        
        # 암호화폐 뉴스 검색
        print("\n\n🔍 Searching crypto news...")
        crypto_news = await collector.collect_crypto_news(count=5)
        print(f"Found {len(crypto_news)} crypto articles")
        
        for article in crypto_news[:3]:
            print(f"\n- {article.title}")
            print(f"  Relevance: {article.relevance_score:.2f}")
            print(f"  Keywords: {', '.join(article.keywords[:5])}")
        
        # 속보 확인
        print("\n\n🚨 Checking breaking news...")
        breaking = await collector.get_breaking_news(minutes=60)
        print(f"Found {len(breaking)} breaking news in last hour")
        
        # 통계
        stats = collector.get_stats()
        print(f"\n📊 Stats: {stats}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())