"""
Feedly API 뉴스 수집기
실시간 암호화폐 관련 뉴스 및 감정분석을 위한 데이터 수집
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """뉴스 기사 데이터"""
    id: str
    title: str
    summary: str
    content: str
    url: str
    published: datetime
    source: str
    author: Optional[str]
    keywords: List[str]
    engagement: Dict[str, int]  # shares, comments, likes


@dataclass
class FeedlyStream:
    """Feedly 스트림 정보"""
    id: str
    title: str
    description: str
    website: str
    subscribers: int
    language: str


class FeedlyCollector:
    """
    Feedly API를 통한 암호화폐 뉴스 수집
    - RSS 피드 기반 뉴스 수집
    - 키워드 필터링
    - 실시간 업데이트
    """
    
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token
        self.base_url = "https://cloud.feedly.com/v3"
        self.session = None
        
        # 암호화폐 관련 키워드
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'binance', 'coinbase',
            'dogecoin', 'ripple', 'cardano', 'polkadot', 'chainlink',
            'solana', 'avalanche', 'polygon', 'uniswap', 'web3'
        }
        
        # 추적할 소스들 (주요 암호화폐 뉴스 사이트)
        self.crypto_sources = [
            'CoinDesk', 'Cointelegraph', 'CryptoSlate', 'The Block',
            'Decrypt', 'CoinGecko', 'CryptoNews', 'Bitcoin Magazine',
            'Blockworks', 'NewsBTC', 'CoinJournal', 'U.Today'
        ]
        
        # 감정 분석을 위한 긍정/부정 키워드
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
        
    async def connect(self):
        """Feedly API 연결"""
        self.session = aiohttp.ClientSession()
        
        if self.access_token:
            # 토큰 유효성 검증
            try:
                profile = await self._request('GET', '/profile')
                logger.info(f"Connected to Feedly as {profile.get('fullName', 'User')}")
            except Exception as e:
                logger.warning(f"Feedly token validation failed: {e}")
        else:
            logger.info("Connected to Feedly (public access)")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """API 요청"""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                headers['Content-Type'] = 'application/json'
                async with self.session.post(url, json=data, headers=headers) as response:
                    return await self._handle_response(response)
                    
        except Exception as e:
            logger.error(f"Feedly API request failed: {method} {endpoint} - {e}")
            raise
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """응답 처리"""
        if response.status == 200:
            return await response.json()
        elif response.status == 429:  # Rate limit
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limit hit, waiting {retry_after}s")
            await asyncio.sleep(retry_after)
            raise Exception("Rate limit exceeded")
        else:
            error_text = await response.text()
            logger.error(f"Feedly API error {response.status}: {error_text}")
            raise Exception(f"Feedly API error: {response.status}")
    
    async def search_feeds(self, query: str, count: int = 20) -> List[FeedlyStream]:
        """피드 검색"""
        params = {
            'query': query,
            'count': count
        }
        
        data = await self._request('GET', '/search/feeds', params)
        
        feeds = []
        for result in data.get('results', []):
            feeds.append(FeedlyStream(
                id=result['feedId'],
                title=result['title'],
                description=result.get('description', ''),
                website=result.get('website', ''),
                subscribers=result.get('subscribers', 0),
                language=result.get('language', 'en')
            ))
        
        return feeds
    
    async def get_crypto_feeds(self) -> List[FeedlyStream]:
        """암호화폐 관련 피드 검색"""
        all_feeds = []
        
        # 주요 키워드로 검색
        search_terms = ['cryptocurrency', 'bitcoin', 'blockchain', 'crypto news']
        
        for term in search_terms:
            try:
                feeds = await self.search_feeds(term, 10)
                all_feeds.extend(feeds)
                await asyncio.sleep(1)  # Rate limit 방지
            except Exception as e:
                logger.error(f"Failed to search feeds for '{term}': {e}")
        
        # 중복 제거
        unique_feeds = {}
        for feed in all_feeds:
            if feed.id not in unique_feeds:
                unique_feeds[feed.id] = feed
        
        # 구독자 수 기준 정렬
        sorted_feeds = sorted(unique_feeds.values(), key=lambda x: x.subscribers, reverse=True)
        
        logger.info(f"Found {len(sorted_feeds)} crypto feeds")
        return sorted_feeds[:50]  # 상위 50개만
    
    async def get_stream_content(
        self,
        stream_id: str,
        count: int = 100,
        newer_than: Optional[datetime] = None
    ) -> List[NewsArticle]:
        """스트림 콘텐츠 조회"""
        params = {
            'count': min(count, 1000)
        }
        
        if newer_than:
            params['newerThan'] = int(newer_than.timestamp() * 1000)
        
        try:
            data = await self._request('GET', f'/streams/{quote(stream_id, safe="")}/contents', params)
            
            articles = []
            for item in data.get('items', []):
                article = await self._parse_article(item)
                if article and self._is_crypto_relevant(article):
                    articles.append(article)
            
            logger.debug(f"Retrieved {len(articles)} crypto articles from {stream_id}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to get stream content for {stream_id}: {e}")
            return []
    
    async def _parse_article(self, item: Dict[str, Any]) -> Optional[NewsArticle]:
        """Feedly 아이템을 NewsArticle로 변환"""
        try:
            # 기본 정보 추출
            article_id = item.get('id', '')
            title = item.get('title', '')
            
            # 요약 및 콘텐츠
            summary = ''
            content = ''
            
            if 'summary' in item:
                summary = item['summary'].get('content', '')
            if 'content' in item:
                content = item['content'].get('content', '')
            
            # URL 추출
            url = ''
            if 'canonicalUrl' in item:
                url = item['canonicalUrl']
            elif 'alternate' in item and item['alternate']:
                url = item['alternate'][0].get('href', '')
            
            # 발행시간
            published = datetime.fromtimestamp(item.get('published', 0) / 1000)
            
            # 소스 정보
            source = ''
            if 'origin' in item:
                source = item['origin'].get('title', '')
            
            # 작성자
            author = item.get('author')
            
            # 키워드 추출
            keywords = self._extract_keywords(title + ' ' + summary)
            
            # 참여도 정보
            engagement = {
                'shares': item.get('engagement', {}).get('shares', 0),
                'comments': item.get('engagement', {}).get('comments', 0),
                'likes': item.get('engagement', {}).get('likes', 0)
            }
            
            return NewsArticle(
                id=article_id,
                title=title,
                summary=summary,
                content=content,
                url=url,
                published=published,
                source=source,
                author=author,
                keywords=keywords,
                engagement=engagement
            )
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 암호화폐 키워드 추출"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _is_crypto_relevant(self, article: NewsArticle) -> bool:
        """암호화폐 관련 기사인지 확인"""
        # 키워드가 있으면 관련
        if article.keywords:
            return True
        
        # 제목이나 요약에서 키워드 검색
        text = (article.title + ' ' + article.summary).lower()
        
        for keyword in self.crypto_keywords:
            if keyword in text:
                return True
        
        return False
    
    def calculate_basic_sentiment(self, article: NewsArticle) -> Dict[str, float]:
        """기본적인 감정 점수 계산"""
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
        engagement_weight = min(1.0, (
            article.engagement['shares'] * 0.1 +
            article.engagement['comments'] * 0.05 +
            article.engagement['likes'] * 0.01
        ) / 100)
        
        return {
            'sentiment': sentiment_score,
            'confidence': min(0.8, total_count * 0.1 + engagement_weight),
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'engagement_weight': engagement_weight
        }
    
    async def get_latest_crypto_news(
        self,
        hours_back: int = 24,
        max_articles: int = 200
    ) -> List[NewsArticle]:
        """최신 암호화폐 뉴스 수집"""
        logger.info(f"Collecting crypto news from last {hours_back} hours...")
        
        # 시간 범위 설정
        since = datetime.now() - timedelta(hours=hours_back)
        
        # 암호화폐 피드 가져오기
        crypto_feeds = await self.get_crypto_feeds()
        
        all_articles = []
        
        # 각 피드에서 기사 수집
        for feed in crypto_feeds[:10]:  # 상위 10개 피드만
            try:
                articles = await self.get_stream_content(
                    feed.id,
                    count=20,
                    newer_than=since
                )
                all_articles.extend(articles)
                
                # Rate limit 방지
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect from feed {feed.title}: {e}")
                continue
        
        # 중복 제거 (URL 기준)
        unique_articles = {}
        for article in all_articles:
            if article.url and article.url not in unique_articles:
                unique_articles[article.url] = article
        
        # 발행시간 기준 정렬 (최신순)
        sorted_articles = sorted(
            unique_articles.values(),
            key=lambda x: x.published,
            reverse=True
        )
        
        result = sorted_articles[:max_articles]
        logger.info(f"Collected {len(result)} unique crypto articles")
        
        return result
    
    async def get_sentiment_summary(
        self,
        articles: List[NewsArticle]
    ) -> Dict[str, Any]:
        """기사들의 감정 요약"""
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
    
    async def close(self):
        """연결 종료"""
        if self.session:
            await self.session.close()
            logger.info("Feedly connection closed")


# 팩토리 함수
def create_feedly_collector(access_token: Optional[str] = None) -> FeedlyCollector:
    """Feedly 수집기 생성"""
    return FeedlyCollector(access_token=access_token)