#!/usr/bin/env python3
"""
Finnhub News and Economic Calendar Collector
무료 Finnhub API를 통한 금융 뉴스 및 경제 이벤트 수집기
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import logging

from ..base_collector import (
    BaseNewsCollector, NewsArticle, NewsCategory,
    CollectorConfig, SentimentScore
)


class FinnhubCollector(BaseNewsCollector):
    """Finnhub API 수집기 - 뉴스, 경제 캘린더, 센티멘트"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        # API 키 확인
        api_key = config.api_key if config else os.getenv("FINNHUB_API_KEY")
        
        if not api_key:
            logging.warning("FINNHUB_API_KEY not found. Limited functionality.")
        
        if not config:
            config = CollectorConfig(
                api_key=api_key,
                rate_limit=60,  # 무료 티어는 분당 60 요청
                timeout=30.0,
                cache_ttl=300  # 5분 캐시
            )
        else:
            config.api_key = api_key
            
        super().__init__(config)
        
        self.base_url = "https://finnhub.io/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 주요 심볼
        self.crypto_symbols = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT"]
        self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.forex_symbols = ["OANDA:EUR_USD", "OANDA:GBP_USD", "OANDA:USD_JPY"]
        
        # 경제 이벤트 중요도
        self.event_importance = {
            "high": ["FOMC", "NFP", "CPI", "GDP", "Interest Rate"],
            "medium": ["PMI", "Retail Sales", "Consumer Confidence", "PPI"],
            "low": ["Housing Starts", "Building Permits", "Trade Balance"]
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                'X-Finnhub-Token': self.config.api_key,
                'Content-Type': 'application/json'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """API 요청"""
        if not self.config.api_key:
            self.logger.error("API key required for Finnhub")
            return None
        
        try:
            session = await self._get_session()
            url = f"{self.base_url}{endpoint}"
            
            # API 키를 파라미터로도 전달
            if params is None:
                params = {}
            params['token'] = self.config.api_key
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    self.stats["requests_made"] += 1
                    return await response.json()
                elif response.status == 429:
                    self.logger.error("Rate limit exceeded")
                    self.stats["errors"] += 1
                    return None
                else:
                    self.logger.error(f"API request failed: {response.status}")
                    self.stats["errors"] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error making request: {e}")
            self.stats["errors"] += 1
            return None
    
    def _parse_finnhub_news(self, item: Dict[str, Any]) -> Optional[NewsArticle]:
        """Finnhub 뉴스 아이템 파싱"""
        try:
            # ID 생성
            news_id = str(item.get('id', ''))
            if not news_id:
                # URL 기반 ID 생성
                url = item.get('url', '')
                news_id = hashlib.md5(url.encode()).hexdigest()
            
            # 시간 변환
            timestamp = item.get('datetime', 0)
            published_date = datetime.fromtimestamp(timestamp)
            
            # 제목과 요약
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            
            # 카테고리 결정
            category_str = item.get('category', '').lower()
            if 'crypto' in category_str:
                category = NewsCategory.CRYPTO
            elif 'forex' in category_str or 'company' in category_str:
                category = NewsCategory.FINANCE
            else:
                category = NewsCategory.HEADLINE
            
            # 관련 심볼 추출
            related_symbols = item.get('related', '').split(',') if item.get('related') else []
            
            return NewsArticle(
                id=news_id,
                title=headline,
                content=summary,
                summary=summary[:500] if summary else "",
                url=item.get('url', ''),
                source=item.get('source', 'Finnhub'),
                author=None,
                published_date=published_date,
                collected_date=datetime.now(),
                category=category,
                keywords=self._extract_keywords(headline + " " + summary),
                entities=related_symbols,
                metadata={
                    "image": item.get('image', ''),
                    "related_symbols": related_symbols,
                    "finnhub_category": item.get('category', '')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Finnhub news: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        # 금융 키워드
        finance_keywords = [
            "earnings", "revenue", "profit", "loss", "growth",
            "merger", "acquisition", "ipo", "dividend", "guidance"
        ]
        
        # 암호화폐 키워드
        crypto_keywords = [
            "bitcoin", "ethereum", "crypto", "blockchain", "defi",
            "mining", "halving", "regulation", "adoption"
        ]
        
        # 경제 키워드
        economic_keywords = [
            "inflation", "recession", "gdp", "unemployment", "rate",
            "fed", "ecb", "stimulus", "policy", "trade"
        ]
        
        all_keywords = finance_keywords + crypto_keywords + economic_keywords
        
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """일반 뉴스 헤드라인 수집"""
        # Finnhub 일반 뉴스
        data = await self._make_request("/news", params={"category": "general"})
        
        if not data:
            return []
        
        articles = []
        for item in data[:count]:
            article = self._parse_finnhub_news(item)
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
        # Finnhub은 직접적인 키워드 검색을 지원하지 않음
        # 대신 관련 심볼의 뉴스를 수집 후 필터링
        all_articles = []
        
        # 키워드와 관련된 심볼 찾기
        relevant_symbols = self._get_relevant_symbols(keywords)
        
        for symbol in relevant_symbols[:5]:  # 최대 5개 심볼
            articles = await self.get_company_news(symbol, days_back=7)
            
            # 키워드 필터링
            for article in articles:
                text = (article.title + " " + article.summary).lower()
                if any(kw.lower() in text for kw in keywords):
                    # 시간 필터링
                    if since and article.published_date < since:
                        continue
                    if until and article.published_date > until:
                        continue
                    
                    article.relevance_score = self.calculate_relevance(article, keywords)
                    all_articles.append(article)
        
        # 관련성 점수로 정렬
        all_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return all_articles[:count]
    
    def _get_relevant_symbols(self, keywords: List[str]) -> List[str]:
        """키워드와 관련된 심볼 찾기"""
        symbols = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        # 암호화폐
        if any(kw in keywords_lower for kw in ["bitcoin", "btc", "crypto"]):
            symbols.extend(self.crypto_symbols)
        
        # 주식
        stock_mapping = {
            "apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT",
            "tesla": "TSLA", "amazon": "AMZN"
        }
        
        for keyword, symbol in stock_mapping.items():
            if keyword in keywords_lower:
                symbols.append(symbol)
        
        # 기본값
        if not symbols:
            symbols = ["AAPL", "BINANCE:BTCUSDT"]
        
        return symbols
    
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """최신 속보 수집"""
        # 최근 뉴스 수집
        data = await self._make_request("/news", params={"category": "general"})
        
        if not data:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        breaking_news = []
        
        for item in data:
            article = self._parse_finnhub_news(item)
            if article and article.published_date >= cutoff_time:
                article.category = NewsCategory.BREAKING
                article = await self.analyze_sentiment(article)
                breaking_news.append(article)
        
        return breaking_news
    
    async def get_company_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """특정 심볼의 뉴스"""
        # 날짜 범위
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        params = {
            "symbol": symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d")
        }
        
        data = await self._make_request("/company-news", params=params)
        
        if not data:
            return []
        
        articles = []
        for item in data:
            article = self._parse_finnhub_news(item)
            if article:
                article.metadata["symbol"] = symbol
                article = await self.analyze_sentiment(article)
                articles.append(article)
        
        return articles
    
    async def get_crypto_news(self, symbol: str = "BINANCE:BTCUSDT") -> List[NewsArticle]:
        """암호화폐 뉴스"""
        # Finnhub의 암호화폐 뉴스
        articles = await self.get_company_news(symbol, days_back=3)
        
        # 카테고리 설정
        for article in articles:
            article.category = NewsCategory.CRYPTO
        
        return articles
    
    async def get_economic_calendar(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """경제 캘린더 이벤트"""
        # Finnhub 경제 캘린더
        data = await self._make_request("/calendar/economic")
        
        if not data or 'economicCalendar' not in data:
            return []
        
        events = []
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        for event in data['economicCalendar']:
            try:
                event_date = datetime.strptime(event.get('time', ''), '%Y-%m-%d %H:%M:%S')
                
                if event_date <= cutoff_date:
                    # 중요도 결정
                    event_name = event.get('event', '')
                    importance = self._get_event_importance(event_name)
                    
                    events.append({
                        "event": event_name,
                        "country": event.get('country', ''),
                        "date": event_date.isoformat(),
                        "actual": event.get('actual'),
                        "estimate": event.get('estimate'),
                        "previous": event.get('prev'),
                        "importance": importance,
                        "unit": event.get('unit', '')
                    })
                    
            except Exception as e:
                self.logger.error(f"Error parsing economic event: {e}")
                continue
        
        # 날짜순 정렬
        events.sort(key=lambda x: x['date'])
        
        return events
    
    def _get_event_importance(self, event_name: str) -> str:
        """경제 이벤트 중요도 판단"""
        event_lower = event_name.lower()
        
        for importance, keywords in self.event_importance.items():
            if any(kw.lower() in event_lower for kw in keywords):
                return importance
        
        return "medium"  # 기본값
    
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """시장 센티멘트 (뉴스 기반)"""
        # 심볼의 최근 뉴스 수집
        articles = await self.get_company_news(symbol, days_back=3)
        
        if not articles:
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0,
                "article_count": 0
            }
        
        # 감정 집계
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
        avg_score = total_score / article_count if article_count > 0 else 0
        
        # 전체 감정 결정
        if avg_score > 0.2:
            overall_sentiment = "bullish"
        elif avg_score < -0.2:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "score": avg_score,
            "article_count": article_count,
            "distribution": sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_fomc_events(self) -> List[Dict[str, Any]]:
        """FOMC 이벤트 필터링"""
        events = await self.get_economic_calendar(days_ahead=60)  # 60일
        
        fomc_events = []
        for event in events:
            if "fomc" in event['event'].lower() or "federal reserve" in event['event'].lower():
                fomc_events.append(event)
        
        return fomc_events
    
    async def get_major_economic_events(self) -> List[Dict[str, Any]]:
        """주요 경제 이벤트만 필터링"""
        events = await self.get_economic_calendar(days_ahead=30)
        
        # 고중요도 이벤트만
        major_events = [e for e in events if e.get('importance') == 'high']
        
        return major_events
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()


# 사용 예제
async def main():
    """Finnhub Collector 테스트"""
    # API 키 필요
    if not os.getenv("FINNHUB_API_KEY"):
        print("❌ FINNHUB_API_KEY environment variable required")
        print("Get free API key at: https://finnhub.io/register")
        return
    
    collector = FinnhubCollector()
    
    try:
        # 일반 뉴스
        print("📰 Collecting general news...")
        news = await collector.collect_headlines(count=5)
        print(f"Found {len(news)} articles")
        
        for article in news[:3]:
            print(f"\n- {article.title}")
            print(f"  Published: {article.published_date}")
            print(f"  Sentiment: {article.sentiment_label.name if article.sentiment_label else 'N/A'}")
        
        # 비트코인 뉴스
        print("\n\n₿ Getting Bitcoin news...")
        btc_news = await collector.get_crypto_news("BINANCE:BTCUSDT")
        print(f"Found {len(btc_news)} BTC articles")
        
        # 경제 캘린더
        print("\n\n📅 Economic Calendar...")
        events = await collector.get_major_economic_events()
        print(f"Found {len(events)} major events")
        
        for event in events[:5]:
            print(f"\n- {event['event']} ({event['country']})")
            print(f"  Date: {event['date']}")
            print(f"  Importance: {event['importance']}")
        
        # 시장 센티멘트
        print("\n\n💭 Market Sentiment...")
        sentiment = await collector.get_market_sentiment("AAPL")
        print(f"AAPL Sentiment: {sentiment['sentiment']} (score: {sentiment['score']:.2f})")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())