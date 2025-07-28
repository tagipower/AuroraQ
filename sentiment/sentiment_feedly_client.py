# sentiment_feedly_client.py - 개선된 버전

import requests
import logging
import time
import yaml
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from urllib.parse import quote
import backoff
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class FeedlyArticle:
    """Feedly 기사 데이터 모델"""
    id: str
    title: str
    summary: str
    published: datetime
    origin_id: str
    canonical_url: Optional[str] = None
    author: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    engagement: int = 0
    engagement_rate: float = 0.0
    
    @property
    def snippet(self) -> str:
        """summary의 alias (하위 호환성)"""
        return self.summary[:500] if self.summary else ""

class FeedlyClient:
    """개선된 Feedly API 클라이언트"""
    
    BASE_URL = "https://cloud.feedly.com/v3"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    def __init__(self, 
                 access_token: str,
                 stream_config_path: Union[str, Path] = "config/stream_ids.yaml",
                 timeout: int = DEFAULT_TIMEOUT,
                 max_workers: int = 4,
                 rate_limit_per_second: float = 10.0):
        """
        Args:
            access_token: Feedly API 액세스 토큰
            stream_config_path: Stream ID 설정 파일 경로
            timeout: 요청 타임아웃 (초)
            max_workers: 병렬 처리 워커 수
            rate_limit_per_second: 초당 요청 제한
        """
        if not access_token:
            raise ValueError("Access token is required")
            
        self.access_token = access_token
        self.headers = {
            "Authorization": f"OAuth {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "SentimentAnalyzer/1.0"
        }
        
        # 설정
        self.timeout = timeout
        self.max_workers = max_workers
        self.rate_limit_delay = 1.0 / rate_limit_per_second
        self.last_request_time = 0.0
        
        # Stream ID 로드
        self.stream_config_path = Path(stream_config_path)
        self.stream_ids = self._load_stream_ids()
        
        # HTTP 세션 설정
        self.session = self._create_session()
        
        # 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_articles': 0
        }
        
        logger.info(f"FeedlyClient initialized with {len(self.stream_ids)} streams")
    
    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션 생성"""
        session = requests.Session()
        
        # 재시도 전략
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update(self.headers)
        
        return session
    
    def _load_stream_ids(self) -> List[str]:
        """Stream ID 목록 로드 (개선된 버전)"""
        if not self.stream_config_path.exists():
            logger.warning(f"Stream config not found: {self.stream_config_path}")
            return []
        
        try:
            with open(self.stream_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 다양한 키 이름 지원
            stream_ids = (
                config.get("stream_ids") or 
                config.get("streams") or 
                config.get("feed_ids") or 
                []
            )
            
            # 유효성 검증
            valid_streams = []
            for stream_id in stream_ids:
                if self._validate_stream_id(stream_id):
                    valid_streams.append(stream_id)
                else:
                    logger.warning(f"Invalid stream ID: {stream_id}")
            
            logger.info(f"Loaded {len(valid_streams)} valid stream IDs")
            return valid_streams
            
        except Exception as e:
            logger.error(f"Failed to load stream IDs: {e}")
            return []
    
    def _validate_stream_id(self, stream_id: str) -> bool:
        """Stream ID 유효성 검증"""
        if not stream_id or not isinstance(stream_id, str):
            return False
        
        # Feedly stream ID 패턴
        valid_prefixes = ['feed/', 'user/', 'topic/', 'category/', 'enterprise/']
        return any(stream_id.startswith(prefix) for prefix in valid_prefixes)
    
    def _rate_limit(self):
        """요청 속도 제한"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=60
    )
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """API 요청 실행 (재시도 로직 포함)"""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            self.stats['total_requests'] += 1
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            self.stats['successful_requests'] += 1
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limit 도달
                retry_after = int(e.response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit reached. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                raise
            elif e.response.status_code == 401:
                logger.error("Authentication failed. Check your access token.")
                raise
            else:
                logger.error(f"HTTP error: {e}")
                self.stats['failed_requests'] += 1
                raise
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.stats['failed_requests'] += 1
            raise
    
    def fetch_articles(self, 
                      stream_id: str,
                      count: int = 20,
                      newer_than_minutes: Optional[int] = 60,
                      continuation: Optional[str] = None,
                      ranked: str = "newest",
                      unread_only: bool = False) -> List[FeedlyArticle]:
        """
        단일 스트림에서 기사 가져오기 (개선된 버전)
        
        Args:
            stream_id: Feedly stream ID
            count: 가져올 기사 수
            newer_than_minutes: N분 이내 기사만 가져오기
            continuation: 페이지네이션 토큰
            ranked: 정렬 방식 (newest, oldest, engagement)
            unread_only: 읽지 않은 기사만
            
        Returns:
            FeedlyArticle 객체 리스트
        """
        if not stream_id:
            logger.warning("Stream ID not provided")
            return []
        
        # 파라미터 구성
        params = {
            "streamId": stream_id,
            "count": min(count, 1000),  # API 제한
            "ranked": ranked
        }
        
        if newer_than_minutes:
            newer_than = int((datetime.utcnow() - timedelta(minutes=newer_than_minutes)).timestamp() * 1000)
            params["newerThan"] = newer_than
        
        if continuation:
            params["continuation"] = continuation
            
        if unread_only:
            params["unreadOnly"] = "true"
        
        try:
            logger.info(f"Fetching articles from {stream_id}")
            
            data = self._make_request("streams/contents", params)
            
            raw_articles = data.get("items", [])
            articles = [self._parse_article(item) for item in raw_articles]
            articles = [a for a in articles if a is not None]
            
            self.stats['total_articles'] += len(articles)
            
            logger.info(f"Fetched {len(articles)} articles from {stream_id}")
            
            # Continuation 토큰 저장 (페이지네이션용)
            self._last_continuation = data.get("continuation")
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch articles from {stream_id}: {e}")
            return []
    
    def _parse_article(self, item: Dict[str, Any]) -> Optional[FeedlyArticle]:
        """원시 기사 데이터를 FeedlyArticle 객체로 파싱"""
        try:
            # 필수 필드 확인
            if not item.get("id") or not item.get("title"):
                return None
            
            # 발행 시간 파싱
            published_ms = item.get("published", item.get("crawled", 0))
            published = datetime.fromtimestamp(published_ms / 1000) if published_ms else datetime.utcnow()
            
            # 요약 추출
            summary = ""
            if "summary" in item:
                summary = item["summary"].get("content", "")
            elif "content" in item:
                summary = item["content"].get("content", "")
            
            # 기사 객체 생성
            article = FeedlyArticle(
                id=item["id"],
                title=item.get("title", "").strip(),
                summary=self._clean_html(summary)[:1000],
                published=published,
                origin_id=item.get("originId", ""),
                canonical_url=item.get("canonicalUrl") or item.get("alternate", [{}])[0].get("href"),
                author=item.get("author"),
                categories=[cat.get("label", "") for cat in item.get("categories", [])],
                keywords=item.get("keywords", []),
                engagement=item.get("engagement", 0),
                engagement_rate=item.get("engagementRate", 0.0)
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    @staticmethod
    def _clean_html(text: str) -> str:
        """HTML 태그 제거"""
        import re
        # 간단한 HTML 제거 (BeautifulSoup 대신)
        text = re.sub('<[^<]+?>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fetch_all_articles(self, 
                          count_per_stream: int = 20,
                          newer_than_minutes: int = 60,
                          parallel: bool = True) -> List[FeedlyArticle]:
        """
        모든 스트림에서 기사 가져오기
        
        Args:
            count_per_stream: 스트림당 기사 수
            newer_than_minutes: N분 이내 기사만
            parallel: 병렬 처리 여부
            
        Returns:
            전체 기사 리스트
        """
        if not self.stream_ids:
            logger.warning("No stream IDs configured")
            return []
        
        if parallel:
            return self._fetch_all_parallel(count_per_stream, newer_than_minutes)
        else:
            return self._fetch_all_sequential(count_per_stream, newer_than_minutes)
    
    def _fetch_all_sequential(self, count_per_stream: int, newer_than_minutes: int) -> List[FeedlyArticle]:
        """순차적으로 모든 스트림에서 기사 가져오기"""
        all_articles = []
        
        for stream_id in self.stream_ids:
            articles = self.fetch_articles(
                stream_id=stream_id,
                count=count_per_stream,
                newer_than_minutes=newer_than_minutes
            )
            all_articles.extend(articles)
        
        return all_articles
    
    def _fetch_all_parallel(self, count_per_stream: int, newer_than_minutes: int) -> List[FeedlyArticle]:
        """병렬로 모든 스트림에서 기사 가져오기"""
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stream = {
                executor.submit(
                    self.fetch_articles,
                    stream_id,
                    count_per_stream,
                    newer_than_minutes
                ): stream_id
                for stream_id in self.stream_ids
            }
            
            for future in as_completed(future_to_stream):
                stream_id = future_to_stream[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Failed to fetch from {stream_id}: {e}")
        
        return all_articles
    
    async def fetch_articles_async(self, 
                                  stream_id: str,
                                  count: int = 20,
                                  newer_than_minutes: Optional[int] = 60) -> List[FeedlyArticle]:
        """비동기로 기사 가져오기"""
        params = {
            "streamId": stream_id,
            "count": count
        }
        
        if newer_than_minutes:
            newer_than = int((datetime.utcnow() - timedelta(minutes=newer_than_minutes)).timestamp() * 1000)
            params["newerThan"] = newer_than
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.BASE_URL}/streams/contents",
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    raw_articles = data.get("items", [])
                    articles = [self._parse_article(item) for item in raw_articles]
                    return [a for a in articles if a is not None]
                    
            except Exception as e:
                logger.error(f"Async fetch failed for {stream_id}: {e}")
                return []
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """스트림 정보 가져오기"""
        try:
            endpoint = f"streams/{quote(stream_id, safe='')}"
            return self._make_request(endpoint)
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None
    
    def search_feeds(self, query: str, count: int = 20) -> List[Dict[str, Any]]:
        """피드 검색"""
        try:
            params = {
                "query": query,
                "count": count
            }
            
            data = self._make_request("search/feeds", params)
            return data.get("results", [])
            
        except Exception as e:
            logger.error(f"Feed search failed: {e}")
            return []
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """사용자 카테고리 목록 가져오기"""
        try:
            data = self._make_request("categories")
            return data
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def mark_as_read(self, entry_ids: List[str]):
        """기사를 읽음으로 표시"""
        try:
            endpoint = "markers"
            data = {
                "action": "markAsRead",
                "type": "entries",
                "entryIds": entry_ids
            }
            
            response = self.session.post(
                f"{self.BASE_URL}/{endpoint}",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Marked {len(entry_ids)} articles as read")
            
        except Exception as e:
            logger.error(f"Failed to mark as read: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """클라이언트 통계 반환"""
        return {
            **self.stats,
            'stream_count': len(self.stream_ids),
            'average_articles_per_stream': (
                self.stats['total_articles'] / len(self.stream_ids) 
                if self.stream_ids else 0
            )
        }
    
    def close(self):
        """세션 종료"""
        self.session.close()
        logger.info("FeedlyClient session closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 테스트 코드
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 환경변수 로드
    load_dotenv()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트용 stream_ids.yaml 생성
    test_streams = {
        "streams": [
            "feed/http://feeds.reuters.com/reuters/businessNews",
            "feed/http://feeds.bloomberg.com/markets/news",
            "topic/crypto"
        ]
    }
    
    test_config_path = Path("test_stream_ids.yaml")
    with open(test_config_path, 'w') as f:
        yaml.dump(test_streams, f)
    
    try:
        # 클라이언트 테스트
        access_token = os.getenv("FEEDLY_ACCESS_TOKEN", "test_token")
        
        if access_token == "test_token":
            logger.warning("Using test token - API calls will fail")
        
        with FeedlyClient(
            access_token=access_token,
            stream_config_path=test_config_path
        ) as client:
            
            print("=== Feedly Client Test ===")
            print(f"Loaded streams: {client.stream_ids}")
            
            # 단일 스트림 테스트
            if client.stream_ids:
                print(f"\nFetching from first stream: {client.stream_ids[0]}")
                articles = client.fetch_articles(
                    client.stream_ids[0],
                    count=5,
                    newer_than_minutes=120
                )
                
                for article in articles[:3]:
                    print(f"\n📰 {article.title}")
                    print(f"   Published: {article.published}")
                    print(f"   URL: {article.canonical_url}")
                    print(f"   Engagement: {article.engagement}")
            
            # 통계
            print(f"\nStatistics: {client.get_statistics()}")
            
    finally:
        # 테스트 파일 정리
        if test_config_path.exists():
            os.unlink(test_config_path)