#!/usr/bin/env python3
"""
News Collector - 기존 인터페이스 호환성
기존 sentiment_engine에서 사용하던 인터페이스와 호환되는 새로운 뉴스 수집기
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# 새로운 데이터 수집 시스템 임포트
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_collection.news_aggregation_system import AuroraQNewsAggregator
from data_collection.base_collector import NewsArticle, NewsCategory


# 하위호환을 위한 데이터 클래스 (기존 형태 유지)
@dataclass
class LegacyNewsArticle:
    """기존 인터페이스용 뉴스 기사 (하위호환)"""
    id: str
    title: str
    summary: str
    content: str
    url: str
    published: datetime
    source: str
    author: Optional[str]
    keywords: List[str]
    engagement: Dict[str, int]
    
    @classmethod
    def from_new_article(cls, article: NewsArticle) -> 'LegacyNewsArticle':
        """새로운 NewsArticle을 기존 형태로 변환"""
        return cls(
            id=article.id,
            title=article.title,
            summary=article.summary,
            content=article.content,
            url=article.url,
            published=article.published_date,
            source=article.source,
            author=article.author,
            keywords=article.keywords,
            engagement={'count': article.metadata.get('score', 0) if article.metadata else 0}
        )


class NewsCollector:
    """
    하위 호환성을 위한 뉴스 수집기
    기존 코드에서 사용하던 인터페이스를 유지하면서 새로운 시스템 활용
    """
    
    def __init__(self, access_token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # 새로운 수집 시스템 초기화
        self.aggregator = AuroraQNewsAggregator()
        
        # 기존 인터페이스 호환을 위한 설정
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'binance', 'coinbase'
        }
        
        # 감정 분석을 위한 키워드
        self.sentiment_keywords = {
            'positive': {
                'bull', 'bullish', 'moon', 'pump', 'surge', 'rally',
                'breakthrough', 'adoption', 'partnership', 'launch',
                'upgrade', 'milestone', 'success', 'growth', 'rise'
            },
            'negative': {
                'bear', 'bearish', 'crash', 'dump', 'fall', 'drop',
                'hack', 'scam', 'regulation', 'ban', 'concern',
                'risk', 'decline', 'loss', 'sell-off', 'correction'
            }
        }
        
        self.logger.info("NewsCollector initialized with new aggregation system")
    
    async def connect(self):
        """연결 확인 (하위호환용)"""
        try:
            health = await self.aggregator.get_system_health()
            if health.get('status') == 'healthy':
                self.logger.info(f"Connected to news aggregation system")
                self.logger.info(f"Active collectors: {health.get('active_collectors', 0)}")
            else:
                self.logger.warning(f"System status: {health.get('status')}")
        except Exception as e:
            self.logger.warning(f"Connection check failed: {e}")
    
    async def get_latest_crypto_news(self, hours_back: int = 24, max_articles: int = 200) -> List[LegacyNewsArticle]:
        """최신 암호화폐 뉴스 수집 (기존 인터페이스 호환)"""
        self.logger.info(f"Collecting crypto news from last {hours_back} hours...")
        
        try:
            # 새로운 시스템으로 암호화폐 뉴스 수집
            news_data = await self.aggregator.collect_comprehensive_news(
                categories=[NewsCategory.CRYPTO],
                hours_back=hours_back,
                articles_per_category=max_articles
            )
            
            crypto_articles = news_data.get('crypto', [])
            
            # 기존 형태로 변환
            legacy_articles = []
            for article in crypto_articles[:max_articles]:
                legacy_article = LegacyNewsArticle.from_new_article(article)
                legacy_articles.append(legacy_article)
            
            self.logger.info(f"Collected {len(legacy_articles)} crypto articles")
            return legacy_articles
            
        except Exception as e:
            self.logger.error(f"Error collecting crypto news: {e}")
            return []
    
    async def search_crypto_news(self, keywords: List[str], hours: int = 24) -> List[LegacyNewsArticle]:
        """암호화폐 뉴스 검색 (기존 인터페이스 호환)"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            # 새로운 시스템으로 검색
            articles = await self.aggregator.search_all_sources(
                keywords=keywords,
                since=since,
                count_per_source=50
            )
            
            # 암호화폐 관련 필터링
            crypto_articles = []
            for article in articles:
                if self._is_crypto_relevant_new(article):
                    crypto_articles.append(article)
            
            # 기존 형태로 변환
            legacy_articles = []
            for article in crypto_articles:
                legacy_article = LegacyNewsArticle.from_new_article(article)
                legacy_articles.append(legacy_article)
            
            return legacy_articles
            
        except Exception as e:
            self.logger.error(f"Error searching crypto news: {e}")
            return []
    
    def _is_crypto_relevant_new(self, article: NewsArticle) -> bool:
        """새로운 NewsArticle이 암호화폐 관련인지 확인"""
        # 카테고리 확인
        if article.category == NewsCategory.CRYPTO:
            return True
        
        # 키워드 확인
        text = (article.title + ' ' + article.summary).lower()
        for keyword in self.crypto_keywords:
            if keyword in text:
                return True
        
        return False
    
    async def search_btc_news(self, hours: int = 24) -> List[LegacyNewsArticle]:
        """BTC 관련 뉴스 검색"""
        return await self.search_crypto_news(["bitcoin", "btc"], hours)
    
    async def search_eth_news(self, hours: int = 24) -> List[LegacyNewsArticle]:
        """ETH 관련 뉴스 검색"""
        return await self.search_crypto_news(["ethereum", "eth"], hours)
    
    async def search_defi_news(self, hours: int = 24) -> List[LegacyNewsArticle]:
        """DeFi 관련 뉴스 검색"""
        return await self.search_crypto_news(["defi", "decentralized finance"], hours)
    
    async def search_regulation_news(self, hours: int = 48) -> List[LegacyNewsArticle]:
        """규제 관련 뉴스 검색"""
        return await self.search_crypto_news(["crypto regulation", "sec crypto", "bitcoin etf"], hours)
    
    async def search_altcoin_news(self, symbols: List[str], hours: int = 24) -> List[LegacyNewsArticle]:
        """알트코인 뉴스 검색"""
        keywords = symbols + [f"{symbol} price" for symbol in symbols]
        return await self.search_crypto_news(keywords, hours)
    
    async def get_trending_crypto_news(self, count: int = 20) -> List[LegacyNewsArticle]:
        """트렌딩 암호화폐 뉴스"""
        try:
            # 속보에서 암호화폐 관련 필터링
            breaking_news = await self.aggregator.get_breaking_news_all(minutes=180)  # 3시간
            
            crypto_breaking = []
            for article in breaking_news:
                if self._is_crypto_relevant_new(article):
                    crypto_breaking.append(article)
            
            # 기존 형태로 변환
            legacy_articles = []
            for article in crypto_breaking[:count]:
                legacy_article = LegacyNewsArticle.from_new_article(article)
                legacy_articles.append(legacy_article)
            
            return legacy_articles
            
        except Exception as e:
            self.logger.error(f"Error getting trending news: {e}")
            return []
    
    async def get_market_sentiment_articles(self, hours: int = 6) -> Dict[str, List[LegacyNewsArticle]]:
        """시장 감정별로 분류된 기사들 (기존 인터페이스 호환)"""
        try:
            # 최근 암호화폐 뉴스 수집
            crypto_articles = await self.get_latest_crypto_news(hours_back=hours, max_articles=100)
            
            # 감정별 분류
            sentiment_articles = {
                'positive': [],
                'negative': [],
                'neutral': []
            }
            
            for article in crypto_articles:
                sentiment = self._classify_sentiment_legacy(article)
                sentiment_articles[sentiment].append(article)
            
            return sentiment_articles
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {'positive': [], 'negative': [], 'neutral': []}
    
    def _classify_sentiment_legacy(self, article: LegacyNewsArticle) -> str:
        """기존 방식의 감정 분류"""
        text = (article.title + " " + article.summary).lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    async def get_symbol_specific_news(self, symbol: str, hours: int = 24) -> List[LegacyNewsArticle]:
        """특정 심볼에 대한 뉴스"""
        keywords = [symbol, f"{symbol} price", f"{symbol} news"]
        return await self.search_crypto_news(keywords, hours)
    
    async def get_breaking_news(self, minutes: int = 30) -> List[LegacyNewsArticle]:
        """최근 속보"""
        try:
            breaking_articles = await self.aggregator.get_breaking_news_all(minutes=minutes)
            
            # 암호화폐 관련만 필터링
            crypto_breaking = []
            for article in breaking_articles:
                if self._is_crypto_relevant_new(article):
                    crypto_breaking.append(article)
            
            # 기존 형태로 변환
            legacy_articles = []
            for article in crypto_breaking:
                legacy_article = LegacyNewsArticle.from_new_article(article)
                legacy_articles.append(legacy_article)
            
            return legacy_articles
            
        except Exception as e:
            self.logger.error(f"Error getting breaking news: {e}")
            return []
    
    def calculate_basic_sentiment(self, article: LegacyNewsArticle) -> Dict[str, float]:
        """기본적인 감정 점수 계산 (기존 인터페이스 호환)"""
        text = (article.title + ' ' + article.summary).lower()
        
        positive_count = 0
        negative_count = 0
        
        # 긍정 키워드 카운트
        for keyword in self.sentiment_keywords['positive']:
            positive_count += text.count(keyword)
        
        # 부정 키워드 카운트
        for keyword in self.sentiment_keywords['negative']:
            negative_count += text.count(keyword)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            sentiment_score = 0.5  # 중립
        else:
            sentiment_score = positive_count / total_count
        
        # 참여도 가중치 적용
        engagement_weight = min(1.0, article.engagement.get('count', 0) / 100)
        
        return {
            'sentiment': sentiment_score,
            'confidence': min(0.8, total_count * 0.1 + engagement_weight),
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'engagement_weight': engagement_weight
        }
    
    async def get_sentiment_summary(self, articles: List[LegacyNewsArticle]) -> Dict[str, Any]:
        """기사들의 감정 요약 (기존 인터페이스 호환)"""
        if not articles:
            return {
                'overall_sentiment': 0.5,
                'confidence': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        confidences = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            sentiment_data = self.calculate_basic_sentiment(article)
            sentiment = sentiment_data['sentiment']
            confidence = sentiment_data['confidence']
            
            sentiments.append(sentiment)
            confidences.append(confidence)
            
            if sentiment > 0.6:
                positive_count += 1
            elif sentiment < 0.4:
                negative_count += 1
            else:
                neutral_count += 1
        
        # 신뢰도 가중 평균
        if sum(confidences) > 0:
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / sum(confidences)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            weighted_sentiment = 0.5
            avg_confidence = 0.0
        
        return {
            'overall_sentiment': weighted_sentiment,
            'confidence': avg_confidence,
            'article_count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': {
                'positive': positive_count / len(articles),
                'negative': negative_count / len(articles),
                'neutral': neutral_count / len(articles)
            }
        }
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 (기존 인터페이스 호환)"""
        try:
            health = await self.aggregator.get_system_health()
            
            return {
                "status": health.get("status", "unknown"),
                "client_available": health.get("active_collectors", 0) > 0,
                "api_stats": {
                    "total_articles": health.get("collection_stats", {}).get("total_articles", 0)
                },
                "rate_limit": {},
                "cache_size": 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "client_available": False
            }
    
    async def close(self):
        """리소스 정리"""
        try:
            await self.aggregator.close_all()
            self.logger.info("NewsCollector closed")
        except Exception as e:
            self.logger.error(f"Error closing NewsCollector: {e}")


# 팩토리 함수 (기존 호환성)
def create_news_collector(access_token: Optional[str] = None) -> NewsCollector:
    """뉴스 수집기 생성 (기존 인터페이스 호환)"""
    return NewsCollector(access_token=access_token)


# 사용 예제
async def main():
    """호환성 테스트"""
    collector = NewsCollector()
    
    try:
        # 연결 테스트
        await collector.connect()
        
        # 기존 방식으로 뉴스 수집
        print("📰 Collecting latest crypto news...")
        articles = await collector.get_latest_crypto_news(hours_back=6, max_articles=10)
        print(f"Found {len(articles)} articles")
        
        for article in articles[:3]:
            print(f"\n- {article.title}")
            print(f"  Source: {article.source}")
            print(f"  Published: {article.published}")
        
        # 감정 분석
        print("\n💭 Analyzing sentiment...")
        sentiment_summary = await collector.get_sentiment_summary(articles)
        print(f"Overall sentiment: {sentiment_summary['overall_sentiment']:.2f}")
        print(f"Article distribution: {sentiment_summary['sentiment_distribution']}")
        
        # 속보 확인
        print("\n🚨 Breaking news...")
        breaking = await collector.get_breaking_news(minutes=60)
        print(f"Found {len(breaking)} breaking news")
        
        # 통계
        stats = await collector.get_collection_stats()
        print(f"\n📊 Stats: {stats}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())