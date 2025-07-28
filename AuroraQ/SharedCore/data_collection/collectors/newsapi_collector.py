#!/usr/bin/env python3
"""
NewsAPI Collector
NewsAPI.org를 통한 글로벌 뉴스 수집기
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
from urllib.parse import urlencode

from ..base_collector import (
    BaseNewsCollector, NewsArticle, NewsCategory,
    CollectorConfig, SentimentScore
)


class NewsAPICollector(BaseNewsCollector):
    """NewsAPI.org 뉴스 수집기"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        # API 키 확인
        api_key = config.api_key if config else os.getenv("NEWSAPI_KEY")
        
        if not api_key:
            self.logger.warning("NEWSAPI_KEY not found. Get free key at https://newsapi.org/")
        
        if not config:
            config = CollectorConfig(
                api_key=api_key,
                rate_limit=100,  # 무료 티어는 시간당 100 요청
                timeout=30.0,
                cache_ttl=900  # 15분 캐시 (더 길게)
            )
        else:
            config.api_key = api_key
            
        super().__init__(config)
        
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 주요 뉴스 소스들
        self.crypto_sources = [
            "coindesk", "cointelegraph", "crypto-coins-news"
        ]
        
        self.finance_sources = [
            "bloomberg", "reuters", "cnbc", "financial-times",
            "the-wall-street-journal", "marketwatch"
        ]
        
        self.general_sources = [
            "bbc-news", "cnn", "reuters", "associated-press",
            "the-guardian-uk", "usa-today"
        ]
        
        # 카테고리별 키워드
        self.category_keywords = {
            NewsCategory.CRYPTO: [
                "bitcoin", "ethereum", "cryptocurrency", "blockchain",
                "defi", "nft", "crypto market", "digital currency"
            ],
            NewsCategory.FINANCE: [
                "stock market", "wall street", "nasdaq", "dow jones",
                "s&p 500", "earnings", "ipo", "merger"
            ],
            NewsCategory.MACRO: [
                "federal reserve", "inflation", "gdp", "unemployment",
                "interest rates", "economic policy", "recession"
            ],
            NewsCategory.PERSON: [
                "ceo", "president", "chairman", "executive",
                "jerome powell", "elon musk", "warren buffett"
            ]
        }
        
        # 언어 및 국가 설정
        self.language = "en"
        self.country = "us"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                'X-API-Key': self.config.api_key,
                'User-Agent': 'AuroraQ/1.0'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """NewsAPI 요청"""
        if not self.config.api_key:
            self.logger.error("API key required for NewsAPI")
            return None
        
        try:
            session = await self._get_session()
            url = f"{self.base_url}{endpoint}"
            
            async with session.get(url, params=params) as response:
                self.stats["requests_made"] += 1
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    self.logger.error("Rate limit exceeded")
                    self.stats["errors"] += 1
                    return None
                elif response.status == 426:
                    self.logger.error("Upgrade required - free tier limitations")
                    self.stats["errors"] += 1
                    return None
                else:
                    self.logger.error(f"API request failed: {response.status}")
                    text = await response.text()
                    self.logger.error(f"Response: {text}")
                    self.stats["errors"] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error making request: {e}")
            self.stats["errors"] += 1
            return None
    
    def _parse_newsapi_article(self, item: Dict[str, Any], 
                              category: NewsCategory = NewsCategory.HEADLINE) -> Optional[NewsArticle]:
        """NewsAPI 기사 파싱"""
        try:
            # 기본 정보
            title = item.get('title', '')
            description = item.get('description', '')
            content = item.get('content', description)  # content는 보통 잘림
            url = item.get('url', '')
            
            # ID 생성
            article_id = hashlib.md5(url.encode()).hexdigest()
            
            # 시간 파싱
            published_at = item.get('publishedAt', '')
            try:
                published_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            except:
                published_date = datetime.now()
            
            # 소스 정보
            source_info = item.get('source', {})
            source_name = source_info.get('name', 'Unknown')
            
            # 작성자
            author = item.get('author', 'Unknown')
            
            # 키워드 추출
            text_content = f"{title} {description} {content}"
            keywords = self._extract_newsapi_keywords(text_content)
            
            # 카테고리 자동 결정 (키워드 기반)
            if category == NewsCategory.HEADLINE:
                category = self._determine_category_from_keywords(keywords, text_content)
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=content or description,
                summary=description[:500] if description else "",
                url=url,
                source=source_name,
                author=author,
                published_date=published_date,
                collected_date=datetime.now(),
                category=category,
                keywords=keywords,
                entities=self.extract_entities(text_content),
                metadata={
                    "source_id": source_info.get('id', ''),
                    "url_to_image": item.get('urlToImage', ''),
                    "newsapi_category": category.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing NewsAPI article: {e}")
            return None
    
    def _extract_newsapi_keywords(self, text: str) -> List[str]:
        """NewsAPI 기사에서 키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        # 모든 카테고리의 키워드 확인
        for category, category_keywords in self.category_keywords.items():
            for keyword in category_keywords:
                if keyword.lower() in text_lower:
                    keywords.append(keyword)
        
        # 추가 금융/경제 키워드
        additional_keywords = [
            "market", "trading", "investor", "price", "value",
            "revenue", "profit", "loss", "growth", "decline",
            "announcement", "report", "data", "forecast"
        ]
        
        for keyword in additional_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def _determine_category_from_keywords(self, keywords: List[str], text: str) -> NewsCategory:
        """키워드 기반 카테고리 결정"""
        text_lower = text.lower()
        
        # 우선순위 기반 분류
        if any(kw in keywords for kw in self.category_keywords[NewsCategory.CRYPTO]):
            return NewsCategory.CRYPTO
        elif any(kw in keywords for kw in self.category_keywords[NewsCategory.MACRO]):
            return NewsCategory.MACRO
        elif any(kw in keywords for kw in self.category_keywords[NewsCategory.FINANCE]):
            return NewsCategory.FINANCE
        elif any(kw in keywords for kw in self.category_keywords[NewsCategory.PERSON]):
            return NewsCategory.PERSON
        else:
            # 속보 키워드 확인
            breaking_keywords = ["breaking", "urgent", "alert", "just in"]
            if any(kw in text_lower for kw in breaking_keywords):
                return NewsCategory.BREAKING
            return NewsCategory.HEADLINE
    
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """주요 헤드라인 수집"""
        params = {
            "country": self.country,
            "pageSize": min(count, 100),  # API 제한
            "page": 1
        }
        
        data = await self._make_request("/top-headlines", params)
        
        if not data or 'articles' not in data:
            return []
        
        articles = []
        for item in data['articles'][:count]:
            article = self._parse_newsapi_article(item, NewsCategory.HEADLINE)
            if article:
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def search_news(self, keywords: List[str],
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         count: int = 20) -> List[NewsArticle]:
        """키워드 기반 뉴스 검색"""
        # 검색 쿼리 구성
        query = " OR ".join(f'"{keyword}"' for keyword in keywords[:5])  # 최대 5개
        
        params = {
            "q": query,
            "language": self.language,
            "sortBy": "relevancy",
            "pageSize": min(count, 100),
            "page": 1
        }
        
        # 시간 범위 설정
        if since:
            params["from"] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            # 기본적으로 지난 30일
            params["from"] = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if until:
            params["to"] = until.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        data = await self._make_request("/everything", params)
        
        if not data or 'articles' not in data:
            return []
        
        articles = []
        for item in data['articles'][:count]:
            article = self._parse_newsapi_article(item)
            if article:
                article.relevance_score = self.calculate_relevance(article, keywords)
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        # 관련성 점수로 정렬
        articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return articles
    
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """속보 수집"""
        # 최신 헤드라인에서 속보 키워드 필터링
        params = {
            "country": self.country,
            "pageSize": 50,
            "page": 1
        }
        
        data = await self._make_request("/top-headlines", params)
        
        if not data or 'articles' not in data:
            return []
        
        breaking_news = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        for item in data['articles']:
            article = self._parse_newsapi_article(item, NewsCategory.BREAKING)
            if article and article.published_date >= cutoff_time:
                # 속보 키워드 확인
                breaking_keywords = ["breaking", "urgent", "alert", "just in", "update"]
                title_lower = article.title.lower()
                
                if any(kw in title_lower for kw in breaking_keywords):
                    article = await self.analyze_sentiment(article)
                    breaking_news.append(article)
        
        return breaking_news
    
    async def collect_crypto_news(self, count: int = 20) -> List[NewsArticle]:
        """암호화폐 뉴스 특화 수집"""
        crypto_keywords = ["bitcoin", "ethereum", "cryptocurrency", "blockchain"]
        
        articles = await self.search_news(
            keywords=crypto_keywords,
            since=datetime.now() - timedelta(days=7),
            count=count
        )
        
        # 카테고리 설정
        for article in articles:
            article.category = NewsCategory.CRYPTO
        
        return articles
    
    async def collect_finance_news(self, count: int = 20) -> List[NewsArticle]:
        """금융 뉴스 수집"""
        finance_keywords = ["stock market", "wall street", "nasdaq", "s&p 500"]
        
        articles = await self.search_news(
            keywords=finance_keywords,
            since=datetime.now() - timedelta(days=3),
            count=count
        )
        
        for article in articles:
            article.category = NewsCategory.FINANCE
        
        return articles
    
    async def collect_macro_news(self, count: int = 20) -> List[NewsArticle]:
        """거시경제 뉴스 수집"""
        macro_keywords = ["federal reserve", "inflation", "gdp", "unemployment"]
        
        articles = await self.search_news(
            keywords=macro_keywords,
            since=datetime.now() - timedelta(days=7),
            count=count
        )
        
        for article in articles:
            article.category = NewsCategory.MACRO
        
        return articles
    
    async def collect_person_news(self, person: str, count: int = 10) -> List[NewsArticle]:
        """특정 인물 관련 뉴스"""
        articles = await self.search_news(
            keywords=[person],
            since=datetime.now() - timedelta(days=30),
            count=count
        )
        
        for article in articles:
            article.category = NewsCategory.PERSON
        
        return articles
    
    async def collect_by_source(self, sources: List[str], count: int = 20) -> List[NewsArticle]:
        """특정 소스들에서 뉴스 수집"""
        source_string = ",".join(sources[:20])  # API 제한
        
        params = {
            "sources": source_string,
            "pageSize": min(count, 100),
            "page": 1
        }
        
        data = await self._make_request("/top-headlines", params)
        
        if not data or 'articles' not in data:
            return []
        
        articles = []
        for item in data['articles'][:count]:
            article = self._parse_newsapi_article(item)
            if article:
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def get_tech_giants_news(self, count: int = 15) -> List[NewsArticle]:
        """빅테크 관련 뉴스"""
        tech_keywords = [
            "Apple", "Google", "Microsoft", "Amazon", "Tesla",
            "Meta", "Netflix", "NVIDIA"
        ]
        
        articles = await self.search_news(
            keywords=tech_keywords,
            since=datetime.now() - timedelta(days=7),
            count=count
        )
        
        return articles
    
    async def get_fed_news(self, count: int = 10) -> List[NewsArticle]:
        """연준 관련 뉴스"""
        fed_keywords = ["Federal Reserve", "Jerome Powell", "FOMC", "Fed meeting"]
        
        articles = await self.search_news(
            keywords=fed_keywords,
            since=datetime.now() - timedelta(days=14),
            count=count
        )
        
        for article in articles:
            article.category = NewsCategory.MACRO
        
        return articles
    
    async def analyze_news_sentiment_by_category(self, hours: int = 24) -> Dict[str, Any]:
        """카테고리별 뉴스 감정 분석"""
        since = datetime.now() - timedelta(hours=hours)
        
        categories_data = {}
        
        # 주요 카테고리별로 뉴스 수집 및 분석
        category_methods = {
            "crypto": self.collect_crypto_news,
            "finance": self.collect_finance_news,
            "macro": self.collect_macro_news
        }
        
        for category_name, method in category_methods.items():
            try:
                articles = await method(count=20)
                
                # 감정 분석
                sentiments = {"positive": 0, "negative": 0, "neutral": 0}
                total_score = 0
                
                for article in articles:
                    if article.sentiment_label:
                        if article.sentiment_label.value > 0:
                            sentiments["positive"] += 1
                        elif article.sentiment_label.value < 0:
                            sentiments["negative"] += 1
                        else:
                            sentiments["neutral"] += 1
                    
                    if article.sentiment_score:
                        total_score += article.sentiment_score
                
                article_count = len(articles)
                avg_sentiment = total_score / article_count if article_count > 0 else 0
                
                categories_data[category_name] = {
                    "article_count": article_count,
                    "sentiment_distribution": sentiments,
                    "average_sentiment": avg_sentiment,
                    "top_sources": self._get_top_sources(articles)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {category_name}: {e}")
                categories_data[category_name] = {"error": str(e)}
        
        return {
            "analysis_period_hours": hours,
            "categories": categories_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_top_sources(self, articles: List[NewsArticle], top_n: int = 5) -> List[Dict[str, int]]:
        """상위 소스 추출"""
        source_counts = {}
        for article in articles:
            source_counts[article.source] = source_counts.get(article.source, 0) + 1
        
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"source": source, "count": count} for source, count in sorted_sources[:top_n]]
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()


# 사용 예제
async def main():
    """NewsAPI Collector 테스트"""
    if not os.getenv("NEWSAPI_KEY"):
        print("❌ NEWSAPI_KEY environment variable required")
        print("Get free API key at: https://newsapi.org/register")
        return
    
    collector = NewsAPICollector()
    
    try:
        # 헤드라인 수집
        print("📰 Collecting headlines...")
        headlines = await collector.collect_headlines(count=5)
        print(f"Found {len(headlines)} headlines")
        
        for article in headlines[:3]:
            print(f"\n- {article.title}")
            print(f"  Source: {article.source}")
            print(f"  Category: {article.category.value}")
            print(f"  Sentiment: {article.sentiment_label.name if article.sentiment_label else 'N/A'}")
        
        # 암호화폐 뉴스
        print("\n\n🔍 Searching crypto news...")
        crypto_news = await collector.collect_crypto_news(count=5)
        print(f"Found {len(crypto_news)} crypto articles")
        
        for article in crypto_news[:3]:
            print(f"\n- {article.title}")
            print(f"  Relevance: {article.relevance_score:.2f}")
        
        # 연준 뉴스
        print("\n\n🏛️ Fed news...")
        fed_news = await collector.get_fed_news(count=3)
        print(f"Found {len(fed_news)} Fed articles")
        
        # 카테고리별 감정 분석
        print("\n\n📊 Category sentiment analysis...")
        sentiment_analysis = await collector.analyze_news_sentiment_by_category(hours=48)
        
        for category, data in sentiment_analysis["categories"].items():
            if "error" not in data:
                print(f"\n{category.upper()}:")
                print(f"  Articles: {data['article_count']}")
                print(f"  Avg Sentiment: {data['average_sentiment']:.2f}")
                print(f"  Distribution: {data['sentiment_distribution']}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())