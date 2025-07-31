#!/usr/bin/env python3
"""
Reddit Collector for AuroraQ Sentiment Service
Reddit API를 통한 암호화폐/금융 관련 소셜 데이터 수집
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import logging
import re
import hashlib
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

@dataclass
class RedditPost:
    """Reddit 포스트 데이터 클래스"""
    title: str
    content: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: datetime
    author: str
    post_id: str
    upvote_ratio: float
    category: str = "social"
    symbol: Optional[str] = None
    sentiment_keywords: List[str] = field(default_factory=list)
    hash_id: str = field(init=False)
    
    def __post_init__(self):
        """해시 ID 생성"""
        content_for_hash = f"{self.title}{self.post_id}{self.subreddit}"
        self.hash_id = hashlib.md5(content_for_hash.encode()).hexdigest()

class RedditCollector:
    """Reddit 데이터 수집기"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        초기화
        
        Args:
            client_id: Reddit API 클라이언트 ID
            client_secret: Reddit API 클라이언트 시크릿
            user_agent: User Agent 문자열
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0
        self.collected_hashes: Set[str] = set()
        self.last_request_time: float = 0
        self.min_request_interval = 1.0  # Rate limiting: 1초 간격
        
        # 암호화폐 관련 서브레딧
        self.crypto_subreddits = [
            "cryptocurrency", "bitcoin", "ethereum", "cryptomarkets",
            "btc", "ethtrader", "coinbase", "binance", "defi",
            "cryptotechnology", "altcoin", "cryptonews"
        ]
        
        # 금융 관련 서브레딧
        self.finance_subreddits = [
            "investing", "stocks", "wallstreetbets", "SecurityAnalysis",
            "financialindependence", "economics", "trading", "options"
        ]
        
        # 감정 키워드 패턴
        self.sentiment_patterns = {
            "bullish": r'\b(bull|bullish|moon|pump|surge|rally|breakout|buy|hold|hodl)\b',
            "bearish": r'\b(bear|bearish|dump|crash|drop|sell|panic|fear|correction)\b',
            "neutral": r'\b(sideways|consolidation|stable|wait|watch|analysis)\b'
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': self.user_agent}
        )
        await self._authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_wait(self):
        """Rate limiting 대기"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _authenticate(self) -> bool:
        """Reddit API 인증"""
        
        # 토큰이 유효한지 확인
        if self.access_token and time.time() < self.token_expires_at:
            return True
        
        logger.info("Authenticating with Reddit API...")
        
        try:
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            
            async with self.session.post(
                'https://www.reddit.com/api/v1/access_token',
                data=auth_data,
                auth=auth
            ) as response:
                response.raise_for_status()
                token_data = await response.json()
            
            self.access_token = token_data['access_token']
            self.token_expires_at = time.time() + token_data['expires_in'] - 60  # 1분 여유
            
            # 세션 헤더 업데이트
            self.session.headers['Authorization'] = f'Bearer {self.access_token}'
            
            logger.info("Reddit API authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Reddit API authentication failed: {e}")
            return False
    
    async def _make_request(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """안전한 API 요청"""
        
        await self._rate_limit_wait()
        
        # 토큰 재인증 확인
        if time.time() >= self.token_expires_at:
            if not await self._authenticate():
                return None
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 401:
                    # 토큰 만료, 재인증 시도
                    logger.warning("Token expired, re-authenticating...")
                    if await self._authenticate():
                        # 재시도
                        async with self.session.get(url, params=params) as retry_response:
                            retry_response.raise_for_status()
                            return await retry_response.json()
                    return None
                
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"Reddit API request failed: {e}")
            return None
    
    def _extract_sentiment_keywords(self, text: str) -> List[str]:
        """텍스트에서 감정 키워드 추출"""
        keywords = []
        text_lower = text.lower()
        
        for sentiment, pattern in self.sentiment_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                keywords.extend([f"{sentiment}:{match}" for match in matches])
        
        return keywords
    
    def _calculate_engagement_score(self, score: int, num_comments: int, upvote_ratio: float) -> float:
        """참여도 점수 계산"""
        
        # 정규화된 점수 계산
        normalized_score = max(0, min(100, score)) / 100.0
        normalized_comments = max(0, min(500, num_comments)) / 500.0
        normalized_ratio = max(0.5, min(1.0, upvote_ratio))
        
        # 가중 평균
        engagement = (
            normalized_score * 0.5 +
            normalized_comments * 0.3 +
            normalized_ratio * 0.2
        )
        
        return round(engagement, 3)
    
    async def collect_subreddit_posts(self,
                                    subreddit: str,
                                    symbol: str = "crypto",
                                    time_filter: str = "day",
                                    sort_by: str = "hot",
                                    limit: int = 25) -> List[RedditPost]:
        """서브레딧에서 포스트 수집"""
        
        if not self.access_token:
            logger.error("Not authenticated with Reddit API")
            return []
        
        logger.info(f"Collecting posts from r/{subreddit} ({sort_by}, {time_filter})...")
        
        try:
            # API 엔드포인트
            if sort_by == "top":
                url = f"https://oauth.reddit.com/r/{subreddit}/top"
                params = {"t": time_filter, "limit": limit}
            else:
                url = f"https://oauth.reddit.com/r/{subreddit}/{sort_by}"
                params = {"limit": limit}
            
            data = await self._make_request(url, params)
            
            if not data or 'data' not in data or 'children' not in data['data']:
                logger.warning(f"No data received from r/{subreddit}")
                return []
            
            posts = []
            cutoff_time = datetime.now() - timedelta(hours=24)  # 24시간 내 포스트만
            
            for child in data['data']['children']:
                try:
                    post_data = child['data']
                    
                    # 기본 정보 추출
                    title = post_data.get('title', '')
                    content = post_data.get('selftext', '') or post_data.get('url', '')
                    url = f"https://reddit.com{post_data.get('permalink', '')}"
                    score = post_data.get('score', 0)
                    num_comments = post_data.get('num_comments', 0)
                    created_utc = datetime.fromtimestamp(post_data.get('created_utc', 0))
                    author = post_data.get('author', '[deleted]')
                    post_id = post_data.get('id', '')
                    upvote_ratio = post_data.get('upvote_ratio', 0.5)
                    
                    # 시간 필터링
                    if created_utc < cutoff_time:
                        continue
                    
                    # 삭제된 포스트나 봇 포스트 필터링
                    if author in ['[deleted]', 'AutoModerator'] or not title.strip():
                        continue
                    
                    # 최소 참여도 필터링 (스코어 또는 댓글 수)
                    if score < 5 and num_comments < 3:
                        continue
                    
                    # 감정 키워드 추출
                    sentiment_keywords = self._extract_sentiment_keywords(f"{title} {content}")
                    
                    reddit_post = RedditPost(
                        title=title,
                        content=content[:1000],  # 내용 길이 제한
                        url=url,
                        subreddit=subreddit,
                        score=score,
                        num_comments=num_comments,
                        created_utc=created_utc,
                        author=author,
                        post_id=post_id,
                        upvote_ratio=upvote_ratio,
                        symbol=symbol.upper(),
                        sentiment_keywords=sentiment_keywords
                    )
                    
                    # 중복 체크
                    if reddit_post.hash_id not in self.collected_hashes:
                        posts.append(reddit_post)
                        self.collected_hashes.add(reddit_post.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing Reddit post: {e}")
                    continue
            
            logger.info(f"Collected {len(posts)} posts from r/{subreddit}")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to collect from r/{subreddit}: {e}")
            return []
    
    async def collect_crypto_sentiment(self,
                                     symbol: str = "crypto",
                                     hours_back: int = 24,
                                     max_posts_per_sub: int = 10) -> List[RedditPost]:
        """암호화폐 관련 서브레딧에서 감정 데이터 수집"""
        
        logger.info(f"Collecting crypto sentiment for {symbol}...")
        
        # 심볼별 서브레딧 선택
        if symbol.upper() == "BTC":
            subreddits = ["bitcoin", "btc", "cryptocurrency", "cryptomarkets"]
        elif symbol.upper() == "ETH":
            subreddits = ["ethereum", "ethtrader", "cryptocurrency", "defi"]
        else:
            subreddits = self.crypto_subreddits[:6]  # 상위 6개 서브레딧
        
        # 병렬 수집
        tasks = []
        for subreddit in subreddits:
            tasks.append(
                self.collect_subreddit_posts(
                    subreddit, symbol, "day", "hot", max_posts_per_sub
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_posts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection from {subreddits[i]} failed: {result}")
            elif isinstance(result, list):
                all_posts.extend(result)
        
        # 참여도 점수로 정렬
        all_posts.sort(
            key=lambda p: self._calculate_engagement_score(p.score, p.num_comments, p.upvote_ratio),
            reverse=True
        )
        
        logger.info(f"Total crypto sentiment posts collected: {len(all_posts)}")
        return all_posts
    
    async def collect_finance_sentiment(self,
                                      symbol: str = "stock",
                                      hours_back: int = 24,
                                      max_posts_per_sub: int = 10) -> List[RedditPost]:
        """금융 관련 서브레딧에서 감정 데이터 수집"""
        
        logger.info(f"Collecting finance sentiment for {symbol}...")
        
        # 주요 금융 서브레딧
        subreddits = ["investing", "stocks", "SecurityAnalysis", "economics"]
        
        # 병렬 수집
        tasks = []
        for subreddit in subreddits:
            tasks.append(
                self.collect_subreddit_posts(
                    subreddit, symbol, "day", "hot", max_posts_per_sub
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_posts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection from {subreddits[i]} failed: {result}")
            elif isinstance(result, list):
                all_posts.extend(result)
        
        # 참여도 점수로 정렬
        all_posts.sort(
            key=lambda p: self._calculate_engagement_score(p.score, p.num_comments, p.upvote_ratio),
            reverse=True
        )
        
        logger.info(f"Total finance sentiment posts collected: {len(all_posts)}")
        return all_posts
    
    async def get_trending_topics(self, subreddit: str = "cryptocurrency", limit: int = 20) -> List[Dict[str, Any]]:
        """트렌딩 토픽 추출"""
        
        logger.info(f"Getting trending topics from r/{subreddit}...")
        
        try:
            posts = await self.collect_subreddit_posts(subreddit, "crypto", "day", "hot", limit)
            
            # 키워드 빈도 분석
            keyword_counts = {}
            
            for post in posts:
                # 제목에서 키워드 추출
                words = re.findall(r'\b[A-Za-z]{3,}\b', post.title.lower())
                
                for word in words:
                    if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all']:
                        keyword_counts[word] = keyword_counts.get(word, 0) + post.score
            
            # 상위 토픽 반환
            trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return [
                {
                    "topic": topic,
                    "score": score,
                    "mentions": sum(1 for p in posts if topic in p.title.lower())
                }
                for topic, score in trending
            ]
            
        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        return {
            "total_collected_posts": len(self.collected_hashes),
            "authenticated": bool(self.access_token),
            "token_expires_in": max(0, self.token_expires_at - time.time()),
            "supported_crypto_subreddits": len(self.crypto_subreddits),
            "supported_finance_subreddits": len(self.finance_subreddits),
            "rate_limit_interval": self.min_request_interval,
            "sentiment_patterns": list(self.sentiment_patterns.keys())
        }


# 테스트 코드
if __name__ == "__main__":
    import json
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_reddit_collector():
        """Reddit 수집기 테스트"""
        
        # Reddit API 자격증명 (실제 값으로 교체 필요)
        client_id = "your_reddit_client_id"
        client_secret = "your_reddit_client_secret"
        user_agent = "AuroraQ-Sentiment:v1.0 (by /u/your_username)"
        
        if client_id == "your_reddit_client_id":
            print("❌ Reddit API 자격증명을 설정해주세요!")
            return
        
        async with RedditCollector(client_id, client_secret, user_agent) as collector:
            print("=== Reddit 암호화폐 감정 수집 테스트 ===")
            
            # 암호화폐 감정 수집
            crypto_posts = await collector.collect_crypto_sentiment("BTC", max_posts_per_sub=5)
            
            print(f"\n수집된 암호화폐 포스트: {len(crypto_posts)}개")
            
            for i, post in enumerate(crypto_posts[:3], 1):
                print(f"\n{i}. [r/{post.subreddit}] {post.title[:60]}...")
                print(f"   점수: {post.score}, 댓글: {post.num_comments}")
                print(f"   비율: {post.upvote_ratio:.2f}")
                print(f"   감정 키워드: {post.sentiment_keywords[:3]}")
                print(f"   작성자: {post.author}")
                print(f"   시간: {post.created_utc}")
            
            # 트렌딩 토픽
            print(f"\n=== 트렌딩 토픽 ===")
            trending = await collector.get_trending_topics("cryptocurrency", 10)
            
            for i, topic in enumerate(trending[:5], 1):
                print(f"{i}. {topic['topic']} (점수: {topic['score']}, 언급: {topic['mentions']})")
            
            # 통계 출력
            stats = collector.get_collection_stats()
            print(f"\n=== 수집 통계 ===")
            print(json.dumps(stats, indent=2, default=str))
    
    # 테스트 실행
    asyncio.run(test_reddit_collector())