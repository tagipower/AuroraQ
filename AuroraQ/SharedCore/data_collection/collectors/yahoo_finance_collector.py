#!/usr/bin/env python3
"""
Yahoo Finance News Collector
무료 Yahoo Finance RSS 피드를 통한 금융 뉴스 수집기
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import json
import re
from urllib.parse import quote_plus

from ..base_collector import (
    BaseNewsCollector, NewsArticle, NewsCategory,
    CollectorConfig, SentimentScore
)


class YahooFinanceCollector(BaseNewsCollector):
    """Yahoo Finance 뉴스 수집기"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        if not config:
            config = CollectorConfig(
                rate_limit=500,  # Yahoo Finance RSS는 관대함
                timeout=30.0,
                cache_ttl=300  # 5분 캐시
            )
        super().__init__(config)
        
        self.base_rss_url = "https://finance.yahoo.com/rss"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 주요 RSS 피드들
        self.feeds = {
            "headlines": "https://finance.yahoo.com/news/rssindex",
            "crypto": "https://finance.yahoo.com/news/rss/category-cryptocurrency",
            "markets": "https://finance.yahoo.com/news/rss/category-stock-market-news",
            "economy": "https://finance.yahoo.com/news/rss/category-economy",
            "currencies": "https://finance.yahoo.com/news/rss/category-currencies",
            "commodities": "https://finance.yahoo.com/news/rss/category-commodities"
        }
        
        # 티커 심볼 매핑
        self.crypto_symbols = {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "BNB-USD": "Binance Coin",
            "SOL-USD": "Solana",
            "ADA-USD": "Cardano"
        }
        
        self.market_symbols = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "GC=F": "Gold",
            "CL=F": "Oil"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
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
    
    def _parse_yahoo_item(self, item: Dict[str, Any], 
                         category: NewsCategory = NewsCategory.FINANCE) -> Optional[NewsArticle]:
        """Yahoo Finance RSS 아이템 파싱"""
        try:
            # 아이템이 딕셔너리인지 확인
            if not isinstance(item, dict):
                self.logger.warning(f"Yahoo RSS item is not a dict: {type(item)}")
                return None
            
            title = item.get('title', '')
            url = item.get('link', '')
            
            if not title or not url:
                return None
            
            # ID 생성
            article_id = hashlib.md5(url.encode()).hexdigest()
            
            # 발행 시간 파싱
            pub_date_str = item.get('published', '')
            try:
                pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
                # timezone 제거 (naive datetime으로 변환)
                pub_date = pub_date.replace(tzinfo=None)
            except:
                pub_date = datetime.now()
            
            # 요약 추출
            description = item.get('description', '')
            summary = self._clean_html(description)[:500] if description else ''
            
            # 소스 추출 (안전하게)
            source = 'Yahoo Finance'
            if 'source' in item:
                source_data = item['source']
                if isinstance(source_data, dict):
                    source = source_data.get('content', 'Yahoo Finance')
                elif isinstance(source_data, str):
                    source = source_data
            
            # 키워드 및 엔티티 추출
            text_content = title + " " + summary
            keywords = self._extract_finance_keywords(text_content)
            entities = self._extract_tickers(text_content)
            
            # GUID 안전하게 추출
            guid = ''
            if 'guid' in item:
                guid_data = item['guid']
                if isinstance(guid_data, dict):
                    guid = guid_data.get('content', '')
                elif isinstance(guid_data, str):
                    guid = guid_data
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=summary,  # RSS는 전체 내용 미제공
                summary=summary,
                url=url,
                source=source,
                author=item.get('author'),
                published_date=pub_date,
                collected_date=datetime.now(),
                category=category,
                keywords=keywords,
                entities=entities,
                metadata={
                    "rss_source": "yahoo_finance",
                    "guid": guid
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Yahoo item: {e}")
            return None
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        # 간단한 HTML 태그 제거
        text = re.sub('<[^<]+?>', '', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_finance_keywords(self, text: str) -> List[str]:
        """금융 관련 키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        # 금융 키워드
        finance_terms = [
            "stock", "market", "trading", "investor", "earnings",
            "revenue", "profit", "loss", "growth", "recession",
            "inflation", "rate", "bond", "yield", "volatility",
            "bull", "bear", "rally", "crash", "correction"
        ]
        
        # 암호화폐 키워드
        crypto_terms = [
            "bitcoin", "ethereum", "crypto", "blockchain", "defi",
            "nft", "altcoin", "mining", "wallet", "exchange"
        ]
        
        # 경제 지표
        economic_terms = [
            "gdp", "cpi", "unemployment", "fomc", "fed", "ecb",
            "inflation", "deflation", "stimulus", "tapering"
        ]
        
        all_terms = finance_terms + crypto_terms + economic_terms
        
        for term in all_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))
    
    def _extract_tickers(self, text: str) -> List[str]:
        """주식/암호화폐 티커 추출"""
        tickers = []
        
        # 일반적인 티커 패턴 (대문자 2-5자)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # 알려진 티커 확인
        known_tickers = list(self.crypto_symbols.keys()) + list(self.market_symbols.keys())
        
        for ticker in potential_tickers:
            # 일반적인 단어 제외
            common_words = ["CEO", "CFO", "IPO", "ETF", "NYSE", "NASDAQ", "USD", "EUR"]
            if ticker not in common_words and len(ticker) >= 2:
                tickers.append(ticker)
        
        # 암호화폐 이름도 추가
        text_lower = text.lower()
        for symbol, name in self.crypto_symbols.items():
            if name.lower() in text_lower:
                tickers.append(symbol)
        
        return list(set(tickers))
    
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """주요 금융 헤드라인 수집"""
        url = self.feeds["headlines"]
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_yahoo_item(item, NewsCategory.HEADLINE)
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
        # Yahoo Finance는 RSS 검색을 직접 지원하지 않음
        # 대신 여러 피드에서 수집 후 필터링
        all_articles = []
        
        # 모든 피드에서 수집
        for feed_name, feed_url in self.feeds.items():
            rss_content = await self._fetch_rss(feed_url)
            if not rss_content:
                continue
            
            feed = feedparser.parse(rss_content)
            
            for item in feed.entries:
                article = self._parse_yahoo_item(item)
                if article:
                    # 키워드 매칭
                    text = (article.title + " " + article.summary).lower()
                    if any(kw.lower() in text for kw in keywords):
                        # 시간 필터링
                        if since and article.published_date < since:
                            continue
                        if until and article.published_date > until:
                            continue
                        
                        article.relevance_score = self.calculate_relevance(article, keywords)
                        all_articles.append(article)
        
        # 중복 제거 (URL 기준)
        unique_articles = {}
        for article in all_articles:
            if article.url not in unique_articles:
                unique_articles[article.url] = article
        
        # 관련성 점수로 정렬
        sorted_articles = sorted(
            unique_articles.values(),
            key=lambda x: x.relevance_score or 0,
            reverse=True
        )[:count]
        
        # 감정 분석
        for article in sorted_articles:
            article = await self.analyze_sentiment(article)
            self.stats["articles_collected"] += 1
        
        return sorted_articles
    
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """금융 속보 수집"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        breaking_news = []
        
        # 주요 피드에서 최신 뉴스 확인
        for feed_name in ["headlines", "markets", "crypto"]:
            feed_url = self.feeds[feed_name]
            rss_content = await self._fetch_rss(feed_url)
            
            if not rss_content:
                continue
            
            feed = feedparser.parse(rss_content)
            
            for item in feed.entries[:10]:  # 각 피드에서 최신 10개만
                article = self._parse_yahoo_item(item, NewsCategory.BREAKING)
                
                if article and article.published_date >= cutoff_time:
                    # 속보 키워드 확인
                    breaking_keywords = [
                        "breaking", "alert", "urgent", "flash",
                        "plunge", "surge", "crash", "spike", "halt"
                    ]
                    
                    text_lower = article.title.lower()
                    if any(kw in text_lower for kw in breaking_keywords):
                        article = await self.analyze_sentiment(article)
                        breaking_news.append(article)
        
        # 시간순 정렬
        breaking_news.sort(key=lambda x: x.published_date, reverse=True)
        
        return breaking_news
    
    async def collect_crypto_news(self, count: int = 20) -> List[NewsArticle]:
        """암호화폐 뉴스 특화 수집"""
        url = self.feeds["crypto"]
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_yahoo_item(item, NewsCategory.CRYPTO)
            if article:
                # 암호화폐 특화 감정 분석
                article = await self.analyze_crypto_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def collect_market_news(self, count: int = 20) -> List[NewsArticle]:
        """주식 시장 뉴스 수집"""
        url = self.feeds["markets"]
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_yahoo_item(item, NewsCategory.FINANCE)
            if article:
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def collect_economic_news(self, count: int = 20) -> List[NewsArticle]:
        """경제 뉴스 수집"""
        url = self.feeds["economy"]
        
        rss_content = await self._fetch_rss(url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_yahoo_item(item, NewsCategory.MACRO)
            if article:
                article = await self.analyze_sentiment(article)
                articles.append(article)
                self.stats["articles_collected"] += 1
        
        return articles
    
    async def analyze_crypto_sentiment(self, article: NewsArticle) -> NewsArticle:
        """암호화폐 특화 감정 분석"""
        # 암호화폐 특화 감정 키워드
        crypto_positive = [
            "adoption", "institutional", "bullish", "ath", "moon",
            "accumulation", "breakout", "upgrade", "halving", "defi growth"
        ]
        
        crypto_negative = [
            "ban", "regulation", "bearish", "crash", "hack", "scam",
            "rug pull", "sec lawsuit", "crackdown", "bubble"
        ]
        
        text = (article.title + " " + article.summary).lower()
        
        positive_count = sum(1 for word in crypto_positive if word in text)
        negative_count = sum(1 for word in crypto_negative if word in text)
        
        # 가격 변동 패턴 확인
        price_patterns = {
            "surge": 0.3, "soar": 0.3, "rally": 0.2, "jump": 0.2,
            "plunge": -0.3, "crash": -0.4, "tumble": -0.3, "drop": -0.2
        }
        
        sentiment_adjustment = 0
        for pattern, weight in price_patterns.items():
            if pattern in text:
                sentiment_adjustment += weight
        
        # 최종 감정 점수 계산
        base_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
        final_score = max(-1, min(1, base_score + sentiment_adjustment))
        
        article.sentiment_score = final_score
        
        if final_score > 0.3:
            article.sentiment_label = SentimentScore.POSITIVE
        elif final_score < -0.3:
            article.sentiment_label = SentimentScore.NEGATIVE
        else:
            article.sentiment_label = SentimentScore.NEUTRAL
        
        return article
    
    async def get_ticker_news(self, ticker: str, count: int = 10) -> List[NewsArticle]:
        """특정 티커 관련 뉴스"""
        # Yahoo Finance는 티커별 RSS도 제공
        ticker_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        rss_content = await self._fetch_rss(ticker_url)
        if not rss_content:
            return []
        
        feed = feedparser.parse(rss_content)
        articles = []
        
        for item in feed.entries[:count]:
            article = self._parse_yahoo_item(item)
            if article:
                article.metadata["ticker"] = ticker
                article = await self.analyze_sentiment(article)
                articles.append(article)
        
        return articles
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()


# 사용 예제
async def main():
    """Yahoo Finance Collector 테스트"""
    collector = YahooFinanceCollector()
    
    try:
        # 금융 헤드라인
        print("📈 Collecting financial headlines...")
        headlines = await collector.collect_headlines(count=5)
        print(f"Found {len(headlines)} headlines")
        
        for article in headlines[:3]:
            print(f"\n- {article.title}")
            print(f"  Source: {article.source}")
            print(f"  Keywords: {', '.join(article.keywords[:5])}")
        
        # 암호화폐 뉴스
        print("\n\n🪙 Collecting crypto news...")
        crypto_news = await collector.collect_crypto_news(count=5)
        print(f"Found {len(crypto_news)} crypto articles")
        
        for article in crypto_news[:3]:
            print(f"\n- {article.title}")
            print(f"  Sentiment: {article.sentiment_label.name if article.sentiment_label else 'N/A'}")
            print(f"  Score: {article.sentiment_score:.2f}" if article.sentiment_score else "")
        
        # BTC 뉴스
        print("\n\n₿ Getting Bitcoin news...")
        btc_news = await collector.get_ticker_news("BTC-USD", count=3)
        print(f"Found {len(btc_news)} BTC articles")
        
        for article in btc_news:
            print(f"\n- {article.title}")
            print(f"  Published: {article.published_date}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())