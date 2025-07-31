#!/usr/bin/env python3
"""
Enhanced News Collector for AuroraQ Sentiment Service
Google News, Yahoo Finance, NewsAPI, Finnhub 통합 수집기
"""

import asyncio
import aiohttp
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class NewsSource(Enum):
    """뉴스 소스 종류"""
    GOOGLE_NEWS = "google_news"
    YAHOO_FINANCE = "yahoo_finance"
    NEWSAPI = "newsapi"
    FINNHUB = "finnhub"
    COINDESK = "coindesk"
    REUTERS = "reuters"

@dataclass
class NewsItem:
    """뉴스 아이템 데이터 클래스"""
    title: str
    content: str
    url: str
    source: NewsSource
    published_at: datetime
    category: str = "general"
    symbol: Optional[str] = None
    relevance_score: float = 0.5
    entities: List[str] = field(default_factory=list)
    hash_id: str = field(init=False)
    
    def __post_init__(self):
        """해시 ID 생성 (중복 제거용)"""
        content_for_hash = f"{self.title}{self.url}{self.published_at.date()}"
        self.hash_id = hashlib.md5(content_for_hash.encode()).hexdigest()

class EnhancedNewsCollector:
    """강화된 뉴스 수집기"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        초기화
        
        Args:
            api_keys: API 키 딕셔너리
                - newsapi_key: NewsAPI 키
                - finnhub_key: Finnhub 키
        """
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        self.collected_hashes: Set[str] = set()
        self.rate_limits = {
            NewsSource.NEWSAPI: {"requests": 0, "reset_time": time.time() + 3600},
            NewsSource.FINNHUB: {"requests": 0, "reset_time": time.time() + 3600},
            NewsSource.GOOGLE_NEWS: {"requests": 0, "reset_time": time.time() + 60},
        }
        
        # 키워드 매핑 (심볼별)
        self.symbol_keywords = {
            "BTC": ["bitcoin", "btc", "cryptocurrency", "crypto"],
            "ETH": ["ethereum", "eth", "smart contract", "defi"],
            "CRYPTO": ["cryptocurrency", "crypto", "blockchain", "digital asset"],
            "STOCK": ["stock market", "equity", "shares", "trading"],
            "FOREX": ["forex", "currency", "dollar", "euro"]
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AuroraQ-Sentiment-Service/1.0 (https://github.com/auroraQ)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, source: NewsSource, limit: int = 100) -> bool:
        """Rate limit 확인"""
        now = time.time()
        rate_info = self.rate_limits.get(source)
        
        if not rate_info:
            return True
        
        # 시간 리셋
        if now >= rate_info["reset_time"]:
            rate_info["requests"] = 0
            rate_info["reset_time"] = now + 3600  # 1시간 후 리셋
        
        # 요청 수 확인
        if rate_info["requests"] >= limit:
            logger.warning(f"Rate limit exceeded for {source.value}")
            return False
        
        rate_info["requests"] += 1
        return True
    
    async def collect_google_news(self, 
                                symbol: str = "crypto", 
                                hours_back: int = 24,
                                max_results: int = 20) -> List[NewsItem]:
        """Google News RSS 수집"""
        
        if not self._check_rate_limit(NewsSource.GOOGLE_NEWS, 50):
            return []
        
        logger.info(f"Collecting Google News for {symbol}...")
        
        try:
            # 키워드 생성
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            search_query = " OR ".join(keywords)
            
            # Google News RSS URL
            rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            
            async with self.session.get(rss_url) as response:
                response.raise_for_status()
                content = await response.text()
            
            # XML 파싱
            root = ET.fromstring(content)
            items = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in root.findall('.//item')[:max_results]:
                try:
                    title = item.find('title')
                    title_text = title.text if title is not None else ""
                    
                    link = item.find('link')
                    url = link.text if link is not None else ""
                    
                    pub_date = item.find('pubDate')
                    if pub_date is not None:
                        # RFC 2822 형식 파싱
                        pub_datetime = datetime.strptime(
                            pub_date.text, '%a, %d %b %Y %H:%M:%S GMT'
                        )
                    else:
                        pub_datetime = datetime.now()
                    
                    if pub_datetime < cutoff_time:
                        continue
                    
                    description = item.find('description')
                    content_text = description.text if description is not None else ""
                    
                    # 엔티티 추출 (간단한 키워드 매칭)
                    entities = self._extract_entities(f"{title_text} {content_text}")
                    
                    news_item = NewsItem(
                        title=title_text,
                        content=content_text,
                        url=url,
                        source=NewsSource.GOOGLE_NEWS,
                        published_at=pub_datetime,
                        category="crypto" if symbol.upper() in ["BTC", "ETH", "CRYPTO"] else "finance",
                        symbol=symbol.upper(),
                        entities=entities,
                        relevance_score=self._calculate_relevance_score(title_text, keywords)
                    )
                    
                    if news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing Google News item: {e}")
                    continue
            
            logger.info(f"Collected {len(items)} Google News items")
            return items
            
        except Exception as e:
            logger.error(f"Google News collection failed: {e}")
            return []
    
    async def collect_yahoo_finance(self,
                                  symbol: str = "crypto",
                                  hours_back: int = 24,
                                  max_results: int = 15) -> List[NewsItem]:
        """Yahoo Finance RSS 수집"""
        
        logger.info(f"Collecting Yahoo Finance news for {symbol}...")
        
        try:
            # Yahoo Finance RSS URLs
            if symbol.upper() in ["BTC", "CRYPTO"]:
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US"
            elif symbol.upper() == "ETH":
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US"
            else:
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
            
            async with self.session.get(rss_url) as response:
                response.raise_for_status()
                content = await response.text()
            
            # XML 파싱
            root = ET.fromstring(content)
            items = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in root.findall('.//item')[:max_results]:
                try:
                    title = item.find('title')
                    title_text = title.text if title is not None else ""
                    
                    link = item.find('link')
                    url = link.text if link is not None else ""
                    
                    pub_date = item.find('pubDate')
                    if pub_date is not None:
                        pub_datetime = datetime.strptime(
                            pub_date.text, '%a, %d %b %Y %H:%M:%S %z'
                        ).replace(tzinfo=None)
                    else:
                        pub_datetime = datetime.now()
                    
                    if pub_datetime < cutoff_time:
                        continue
                    
                    description = item.find('description')
                    content_text = description.text if description is not None else ""
                    
                    # HTML 태그 제거
                    content_text = BeautifulSoup(content_text, 'html.parser').get_text()
                    
                    entities = self._extract_entities(f"{title_text} {content_text}")
                    keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
                    
                    news_item = NewsItem(
                        title=title_text,
                        content=content_text,
                        url=url,
                        source=NewsSource.YAHOO_FINANCE,
                        published_at=pub_datetime,
                        category="finance",
                        symbol=symbol.upper(),
                        entities=entities,
                        relevance_score=self._calculate_relevance_score(title_text, keywords)
                    )
                    
                    if news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing Yahoo Finance item: {e}")
                    continue
            
            logger.info(f"Collected {len(items)} Yahoo Finance items")
            return items
            
        except Exception as e:
            logger.error(f"Yahoo Finance collection failed: {e}")
            return []
    
    async def collect_newsapi(self,
                            symbol: str = "crypto",
                            hours_back: int = 24,
                            max_results: int = 20) -> List[NewsItem]:
        """NewsAPI 수집"""
        
        if not self.api_keys.get("newsapi_key"):
            logger.warning("NewsAPI key not provided, skipping...")
            return []
        
        if not self._check_rate_limit(NewsSource.NEWSAPI, 100):
            return []
        
        logger.info(f"Collecting NewsAPI articles for {symbol}...")
        
        try:
            # 키워드와 도메인 설정
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            query = " OR ".join(keywords)
            
            # 신뢰할 수 있는 도메인
            domains = "coindesk.com,cointelegraph.com,reuters.com,bloomberg.com,cnbc.com"
            
            # API 요청
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "domains": domains,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "from": (datetime.now() - timedelta(hours=hours_back)).isoformat(),
                "apiKey": self.api_keys["newsapi_key"]
            }
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
            
            if data["status"] != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            items = []
            for article in data.get("articles", []):
                try:
                    # 데이터 추출
                    title = article.get("title", "")
                    content = article.get("description", "") or article.get("content", "")
                    url = article.get("url", "")
                    source_name = article.get("source", {}).get("name", "Unknown")
                    
                    published_at_str = article.get("publishedAt", "")
                    if published_at_str:
                        published_at = datetime.fromisoformat(
                            published_at_str.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    else:
                        published_at = datetime.now()
                    
                    # [Removed] 태그가 있는 기사 스킵
                    if "[Removed]" in title or not title.strip():
                        continue
                    
                    entities = self._extract_entities(f"{title} {content}")
                    
                    news_item = NewsItem(
                        title=title,
                        content=content,
                        url=url,
                        source=NewsSource.NEWSAPI,
                        published_at=published_at,
                        category="crypto" if symbol.upper() in ["BTC", "ETH", "CRYPTO"] else "finance",
                        symbol=symbol.upper(),
                        entities=entities,
                        relevance_score=self._calculate_relevance_score(title, keywords)
                    )
                    
                    if news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing NewsAPI article: {e}")
                    continue
            
            logger.info(f"Collected {len(items)} NewsAPI articles")
            return items
            
        except Exception as e:
            logger.error(f"NewsAPI collection failed: {e}")
            return []
    
    async def collect_finnhub(self,
                            symbol: str = "crypto",
                            hours_back: int = 24,
                            max_results: int = 15) -> List[NewsItem]:
        """Finnhub 뉴스 수집"""
        
        if not self.api_keys.get("finnhub_key"):
            logger.warning("Finnhub key not provided, skipping...")
            return []
        
        if not self._check_rate_limit(NewsSource.FINNHUB, 60):
            return []
        
        logger.info(f"Collecting Finnhub news for {symbol}...")
        
        try:
            # 심볼 매핑
            if symbol.upper() in ["BTC", "CRYPTO"]:
                category = "crypto"
            else:
                category = "general"
            
            # API 요청
            url = "https://finnhub.io/api/v1/news"
            params = {
                "category": category,
                "token": self.api_keys["finnhub_key"]
            }
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                articles = await response.json()
            
            if not isinstance(articles, list):
                logger.error("Finnhub returned invalid response format")
                return []
            
            items = []
            cutoff_timestamp = (datetime.now() - timedelta(hours=hours_back)).timestamp()
            keywords = self.symbol_keywords.get(symbol.upper(), [symbol])
            
            for article in articles[:max_results]:
                try:
                    # 시간 필터링
                    if article.get("datetime", 0) < cutoff_timestamp:
                        continue
                    
                    title = article.get("headline", "")
                    content = article.get("summary", "")
                    url = article.get("url", "")
                    source_name = article.get("source", "Finnhub")
                    
                    published_at = datetime.fromtimestamp(article.get("datetime", time.time()))
                    
                    if not title.strip():
                        continue
                    
                    entities = self._extract_entities(f"{title} {content}")
                    
                    news_item = NewsItem(
                        title=title,
                        content=content,
                        url=url,
                        source=NewsSource.FINNHUB,
                        published_at=published_at,
                        category=category,
                        symbol=symbol.upper(),
                        entities=entities,
                        relevance_score=self._calculate_relevance_score(title, keywords)
                    )
                    
                    if news_item.hash_id not in self.collected_hashes:
                        items.append(news_item)
                        self.collected_hashes.add(news_item.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing Finnhub article: {e}")
                    continue
            
            logger.info(f"Collected {len(items)} Finnhub articles")
            return items
            
        except Exception as e:
            logger.error(f"Finnhub collection failed: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[str]:
        """간단한 엔티티 추출 (키워드 기반)"""
        
        # 주요 엔티티 패턴
        entity_patterns = {
            "crypto": r'\b(bitcoin|btc|ethereum|eth|cryptocurrency|crypto|blockchain|defi|nft)\b',
            "person": r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
            "organization": r'\b(SEC|FDA|Fed|Tesla|Apple|Microsoft|JPMorgan|Goldman Sachs)\b',
            "event": r'\b(FOMC|CPI|ETF|IPO|earnings|meeting|approval|regulation)\b'
        }
        
        entities = []
        text_lower = text.lower()
        
        for category, pattern in entity_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # 중복 제거 및 정리
        return list(set([entity.strip() for entity in entities if len(entity.strip()) > 2]))
    
    def _calculate_relevance_score(self, text: str, keywords: List[str]) -> float:
        """관련성 점수 계산"""
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # 0.1 (최소) ~ 1.0 (완전 매칭) 범위
        score = 0.1 + (matches / len(keywords)) * 0.9
        return min(1.0, score)
    
    async def collect_all_sources(self,
                                symbol: str = "crypto",
                                hours_back: int = 24,
                                max_per_source: int = 15) -> List[NewsItem]:
        """모든 소스에서 뉴스 수집"""
        
        logger.info(f"Starting comprehensive news collection for {symbol}...")
        start_time = time.time()
        
        # 병렬 수집
        tasks = [
            self.collect_google_news(symbol, hours_back, max_per_source),
            self.collect_yahoo_finance(symbol, hours_back, max_per_source),
            self.collect_newsapi(symbol, hours_back, max_per_source),
            self.collect_finnhub(symbol, hours_back, max_per_source)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection task {i} failed: {result}")
            elif isinstance(result, list):
                all_items.extend(result)
        
        # 중복 제거 (해시 기반)
        unique_items = []
        seen_hashes = set()
        
        for item in all_items:
            if item.hash_id not in seen_hashes:
                unique_items.append(item)
                seen_hashes.add(item.hash_id)
        
        # 관련성 점수로 정렬
        unique_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        collection_time = time.time() - start_time
        
        logger.info(f"Comprehensive collection completed: "
                   f"{len(unique_items)} unique items from {len(all_items)} total "
                   f"in {collection_time:.2f}s")
        
        return unique_items
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        return {
            "total_collected_hashes": len(self.collected_hashes),
            "rate_limits": {
                source.value: {
                    "requests": info["requests"],
                    "remaining_time": max(0, info["reset_time"] - time.time())
                }
                for source, info in self.rate_limits.items()
            },
            "supported_sources": [source.value for source in NewsSource],
            "symbol_keywords": self.symbol_keywords
        }


# 테스트 코드
if __name__ == "__main__":
    import json
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_collector():
        """수집기 테스트"""
        
        # API 키 설정 (실제 키로 교체 필요)
        api_keys = {
            "newsapi_key": "your_newsapi_key_here",
            "finnhub_key": "your_finnhub_key_here"
        }
        
        async with EnhancedNewsCollector(api_keys) as collector:
            # 암호화폐 뉴스 수집 테스트
            print("=== 암호화폐 뉴스 수집 테스트 ===")
            crypto_news = await collector.collect_all_sources("BTC", hours_back=6, max_per_source=5)
            
            print(f"\n수집된 뉴스: {len(crypto_news)}개")
            
            for i, news in enumerate(crypto_news[:3], 1):
                print(f"\n{i}. [{news.source.value}] {news.title}")
                print(f"   URL: {news.url}")
                print(f"   발행시간: {news.published_at}")
                print(f"   관련성: {news.relevance_score:.2f}")
                print(f"   엔티티: {news.entities[:3]}")
            
            # 통계 출력
            stats = collector.get_collection_stats()
            print(f"\n=== 수집 통계 ===")
            print(json.dumps(stats, indent=2, default=str))
    
    # 테스트 실행
    asyncio.run(test_collector())