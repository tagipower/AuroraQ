#!/usr/bin/env python3
"""
Reddit News and Sentiment Collector
Reddit API를 통한 커뮤니티 감정 및 트렌드 수집기
"""

import asyncio
import aiohttp
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
from urllib.parse import urlencode

from ..base_collector import (
    BaseNewsCollector, NewsArticle, NewsCategory,
    CollectorConfig, SentimentScore
)


class RedditCollector(BaseNewsCollector):
    """Reddit 커뮤니티 감정 수집기"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        if not config:
            config = CollectorConfig(
                rate_limit=60,  # Reddit API는 분당 60 요청
                timeout=30.0,
                cache_ttl=300  # 5분 캐시
            )
        super().__init__(config)
        
        # Reddit API (읽기 전용은 인증 없이 가능)
        self.base_url = "https://www.reddit.com"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 주요 서브레딧
        self.crypto_subreddits = [
            "cryptocurrency", "bitcoin", "ethereum", "cryptomarkets",
            "altcoin", "defi", "cryptomoonshots", "satoshistreetbets"
        ]
        
        self.finance_subreddits = [
            "wallstreetbets", "stocks", "investing", "stockmarket",
            "options", "daytrading", "pennystocks", "forex"
        ]
        
        self.news_subreddits = [
            "worldnews", "news", "economics", "finance",
            "business", "technology"
        ]
        
        # 감정 지표 키워드
        self.bullish_terms = [
            "moon", "bullish", "buy", "hold", "diamond hands", "to the moon",
            "gains", "profit", "green", "pump", "rocket", "🚀", "💎", "🙌"
        ]
        
        self.bearish_terms = [
            "bear", "bearish", "sell", "dump", "crash", "red", "loss",
            "rekt", "bag holder", "paper hands", "bubble", "📉", "🐻"
        ]
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 관리"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                'User-Agent': 'AuroraQ/1.0 (News Collector Bot)'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _fetch_reddit_data(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Reddit 데이터 가져오기 - API 차단 시 빈 데이터 반환"""
        try:
            # Reddit API가 차단된 경우 즉시 빈 데이터 반환
            self.logger.warning("Reddit API is currently blocked - returning empty data")
            return {"data": {"children": []}}
            
            # 아래는 원본 코드 (현재 사용 안함)
            # session = await self._get_session()
            # url = f"{self.base_url}{endpoint}.json"
            # 
            # if params:
            #     url += "?" + urlencode(params)
            # 
            # async with session.get(url) as response:
            #     if response.status == 200:
            #         return await response.json()
            #     else:
            #         self.logger.error(f"Reddit fetch failed: {response.status}")
            #         return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching Reddit data: {e}")
            self.stats["errors"] += 1
            return {"data": {"children": []}}
    
    def _parse_reddit_post(self, post_data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Reddit 포스트를 NewsArticle로 변환"""
        try:
            data = post_data.get('data', {})
            
            # 기본 정보 추출
            post_id = data.get('id', '')
            title = data.get('title', '')
            selftext = data.get('selftext', '')
            url = f"https://reddit.com{data.get('permalink', '')}"
            author = data.get('author', 'deleted')
            subreddit = data.get('subreddit', '')
            
            # 시간 정보
            created_utc = data.get('created_utc', 0)
            published_date = datetime.fromtimestamp(created_utc)
            
            # 포스트 내용
            content = selftext if selftext else title
            summary = content[:500] if len(content) > 500 else content
            
            # 메타데이터
            metadata = {
                "subreddit": subreddit,
                "score": data.get('score', 0),
                "num_comments": data.get('num_comments', 0),
                "upvote_ratio": data.get('upvote_ratio', 0),
                "is_video": data.get('is_video', False),
                "link_flair_text": data.get('link_flair_text', ''),
                "total_awards": data.get('total_awards_received', 0)
            }
            
            # 키워드 추출
            keywords = self._extract_reddit_keywords(title + " " + selftext)
            
            # 카테고리 결정
            category = self._determine_category(subreddit, keywords)
            
            return NewsArticle(
                id=post_id,
                title=title,
                content=content,
                summary=summary,
                url=url,
                source=f"r/{subreddit}",
                author=author,
                published_date=published_date,
                collected_date=datetime.now(),
                category=category,
                keywords=keywords,
                entities=self.extract_entities(title + " " + content),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Reddit post: {e}")
            return None
    
    def _extract_reddit_keywords(self, text: str) -> List[str]:
        """Reddit 특화 키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        # 암호화폐 티커
        crypto_tickers = [
            "btc", "eth", "bnb", "ada", "sol", "dot", "link",
            "matic", "avax", "atom", "algo", "xrp", "doge", "shib"
        ]
        
        # 금융 용어
        finance_terms = [
            "squeeze", "shorts", "calls", "puts", "yolo", "fomo",
            "dd", "tendies", "ape", "hodl", "dca", "ath", "dip"
        ]
        
        # 일반 키워드
        general_terms = [
            "news", "breaking", "update", "analysis", "prediction",
            "rumor", "leak", "announcement", "report"
        ]
        
        all_terms = crypto_tickers + finance_terms + general_terms
        
        for term in all_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))
    
    def _determine_category(self, subreddit: str, keywords: List[str]) -> NewsCategory:
        """서브레딧과 키워드로 카테고리 결정"""
        subreddit_lower = subreddit.lower()
        
        if subreddit_lower in self.crypto_subreddits:
            return NewsCategory.CRYPTO
        elif subreddit_lower in self.finance_subreddits:
            return NewsCategory.FINANCE
        elif subreddit_lower in self.news_subreddits:
            # 키워드로 세분화
            crypto_keywords = ["bitcoin", "crypto", "eth", "btc"]
            if any(kw in keywords for kw in crypto_keywords):
                return NewsCategory.CRYPTO
            return NewsCategory.HEADLINE
        else:
            return NewsCategory.COMMUNITY
    
    async def analyze_reddit_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Reddit 특화 감정 분석"""
        text = (article.title + " " + article.content).lower()
        metadata = article.metadata or {}
        
        # 텍스트 기반 감정 분석
        bullish_count = sum(1 for term in self.bullish_terms if term in text)
        bearish_count = sum(1 for term in self.bearish_terms if term in text)
        
        # Reddit 메트릭 기반 가중치
        score = metadata.get('score', 0)
        upvote_ratio = metadata.get('upvote_ratio', 0.5)
        num_comments = metadata.get('num_comments', 0)
        
        # 인기도 가중치 (0-1)
        popularity_weight = min(1.0, (score / 1000) + (num_comments / 100))
        
        # 감정 점수 계산
        if bullish_count + bearish_count > 0:
            text_sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
        else:
            text_sentiment = 0
        
        # Upvote ratio 기반 커뮤니티 감정
        community_sentiment = (upvote_ratio - 0.5) * 2  # -1 to 1
        
        # 최종 감정 점수 (텍스트 40%, 커뮤니티 40%, 인기도 20%)
        final_score = (text_sentiment * 0.4 + 
                      community_sentiment * 0.4 + 
                      popularity_weight * 0.2)
        
        article.sentiment_score = max(-1, min(1, final_score))
        
        # 감정 라벨
        if final_score > 0.3:
            article.sentiment_label = SentimentScore.POSITIVE
        elif final_score < -0.3:
            article.sentiment_label = SentimentScore.NEGATIVE
        else:
            article.sentiment_label = SentimentScore.NEUTRAL
        
        return article
    
    async def collect_headlines(self, count: int = 20) -> List[NewsArticle]:
        """Reddit 핫 포스트 수집 - API 차단으로 인해 비활성화됨"""
        self.logger.warning("Reddit collector is disabled due to API block")
        return []  # 빈 리스트 반환
    
    async def search_news(self, keywords: List[str],
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         count: int = 20) -> List[NewsArticle]:
        """Reddit 검색 - API 차단으로 인해 비활성화됨"""
        self.logger.warning("Reddit search is disabled due to API block")
        return []  # 빈 리스트 반환
    
    def _get_relevant_subreddits(self, keywords: List[str]) -> List[str]:
        """키워드에 따른 관련 서브레딧 선택"""
        keywords_lower = [kw.lower() for kw in keywords]
        
        # 암호화폐 관련
        if any(kw in keywords_lower for kw in ["bitcoin", "crypto", "btc", "eth"]):
            return self.crypto_subreddits
        
        # 금융 관련
        if any(kw in keywords_lower for kw in ["stock", "market", "trading", "finance"]):
            return self.finance_subreddits
        
        # 기본값
        return self.news_subreddits
    
    async def get_breaking_news(self, minutes: int = 30) -> List[NewsArticle]:
        """Reddit 실시간 트렌딩 - API 차단으로 인해 비활성화됨"""
        self.logger.warning("Reddit breaking news is disabled due to API block")
        return []  # 빈 리스트 반환
    
    async def get_trending_sentiment(self, subreddit: str = "cryptocurrency") -> Dict[str, Any]:
        """특정 서브레딧의 트렌딩 감정 분석 - API 차단으로 인해 비활성화됨"""
        self.logger.warning("Reddit sentiment analysis is disabled due to API block")
        return {
            "subreddit": subreddit,
            "total_posts_analyzed": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "overall_sentiment": 0,
            "average_score": 0,
            "average_comments": 0,
            "trending_keywords": [],
            "timestamp": datetime.now().isoformat(),
            "status": "disabled_due_to_api_block"
        }
    
    async def get_wallstreetbets_sentiment(self) -> Dict[str, Any]:
        """WSB 특별 감정 분석"""
        return await self.get_trending_sentiment("wallstreetbets")
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()


# 사용 예제
async def main():
    """Reddit Collector 테스트"""
    collector = RedditCollector()
    
    try:
        # 핫 포스트 수집
        print("🔥 Collecting hot posts...")
        hot_posts = await collector.collect_headlines(count=10)
        print(f"Found {len(hot_posts)} hot posts")
        
        for post in hot_posts[:3]:
            print(f"\n- {post.title}")
            print(f"  Subreddit: {post.source}")
            print(f"  Score: {post.metadata.get('score', 0)}")
            print(f"  Sentiment: {post.sentiment_label.name if post.sentiment_label else 'N/A'}")
        
        # 암호화폐 검색
        print("\n\n🔍 Searching crypto posts...")
        crypto_posts = await collector.search_news(["bitcoin", "ethereum"], count=5)
        print(f"Found {len(crypto_posts)} crypto posts")
        
        # 트렌딩 감정 분석
        print("\n\n📊 Analyzing r/cryptocurrency sentiment...")
        sentiment = await collector.get_trending_sentiment("cryptocurrency")
        print(f"Overall sentiment: {sentiment.get('overall_sentiment', 0):.2f}")
        print(f"Distribution: {sentiment.get('sentiment_distribution', {})}")
        
        if sentiment.get('trending_keywords'):
            print("\nTrending keywords:")
            for kw in sentiment['trending_keywords'][:5]:
                print(f"  - {kw['keyword']}: {kw['count']}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())