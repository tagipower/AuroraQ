# sentiment_collector_feedly.py - 개선된 버전

import os
import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import asyncio
import aiofiles
from dotenv import load_dotenv
import yaml
import hashlib
from concurrent.futures import ThreadPoolExecutor

from sentiment_analyzer import SentimentAnalyzer
from sentiment_feedly_client import FeedlyClient

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class Article:
    """기사 데이터 모델"""
    id: str
    title: str
    snippet: str
    source: str
    published: Optional[datetime] = None
    url: Optional[str] = None
    
    @property
    def content_hash(self) -> str:
        """컨텐츠 해시 생성 (중복 체크용)"""
        content = f"{self.title}{self.snippet}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class SentimentData:
    """감정 분석 결과 데이터"""
    timestamp: datetime
    source: str
    title: str
    snippet: str
    sentiment_score: float
    label: str = ""
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    scenario_tag: str = ""
    article_id: str = ""
    url: str = ""

class FeedlySentimentCollector:
    """개선된 Feedly 감정 수집기"""
    
    DEFAULT_CONFIG = {
        'output_dir': "data/sentiment",
        'stream_id_yaml': "config/stream_ids.yaml",
        'batch_size': 10,
        'max_articles_per_stream': 50,
        'sentiment_threshold': 0.2,
        'cache_duration_hours': 24,
        'save_format': 'csv',  # csv, json, both
        'enable_deduplication': True,
        'enable_async': True
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        # 환경변수 로드
        load_dotenv()
        
        # 설정 병합
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # API 토큰 확인
        self.api_token = os.getenv("FEEDLY_API_TOKEN")
        if not self.api_token:
            raise ValueError("FEEDLY_API_TOKEN not found in environment variables")
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 출력 디렉토리 생성
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 디렉토리
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 중복 체크용 세트
        self.seen_hashes: Set[str] = set()
        self._load_seen_hashes()
        
        # 통계 추적
        self.stats = defaultdict(int)
        
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.feedly = FeedlyClient(self.api_token)
            self.analyzer = SentimentAnalyzer()
            
            # Stream ID 로드
            self._load_stream_ids()
            
            # 비동기 실행자
            if self.config['enable_async']:
                self.executor = ThreadPoolExecutor(max_workers=4)
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_stream_ids(self):
        """Stream ID 목록 로드"""
        stream_yaml_path = Path(self.config['stream_id_yaml'])
        
        if not stream_yaml_path.exists():
            logger.warning(f"Stream ID file not found: {stream_yaml_path}")
            self.stream_ids = []
            return
        
        try:
            with open(stream_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.stream_ids = data.get('streams', [])
                
            logger.info(f"Loaded {len(self.stream_ids)} stream IDs")
            
            # Stream ID 검증
            self.stream_ids = [sid for sid in self.stream_ids if self._validate_stream_id(sid)]
            
        except Exception as e:
            logger.error(f"Failed to load stream IDs: {e}")
            self.stream_ids = []
    
    def _validate_stream_id(self, stream_id: str) -> bool:
        """Stream ID 유효성 검사"""
        if not stream_id or not isinstance(stream_id, str):
            return False
        
        # Feedly stream ID 패턴 검증
        if not (stream_id.startswith('feed/') or 
                stream_id.startswith('user/') or
                stream_id.startswith('topic/')):
            logger.warning(f"Invalid stream ID format: {stream_id}")
            return False
            
        return True
    
    def _load_seen_hashes(self):
        """이전에 처리한 기사 해시 로드"""
        seen_file = self.cache_dir / "seen_hashes.json"
        
        if seen_file.exists():
            try:
                with open(seen_file, 'r') as f:
                    data = json.load(f)
                    # 24시간 이내 해시만 유지
                    cutoff = datetime.utcnow() - timedelta(hours=self.config['cache_duration_hours'])
                    
                    for hash_val, timestamp_str in data.items():
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp > cutoff:
                            self.seen_hashes.add(hash_val)
                            
                logger.info(f"Loaded {len(self.seen_hashes)} seen hashes")
                
            except Exception as e:
                logger.error(f"Failed to load seen hashes: {e}")
    
    def _save_seen_hashes(self):
        """처리한 기사 해시 저장"""
        seen_file = self.cache_dir / "seen_hashes.json"
        
        # 현재 시간
        now = datetime.utcnow()
        
        # 해시와 타임스탬프 저장
        data = {
            hash_val: now.isoformat() 
            for hash_val in self.seen_hashes
        }
        
        try:
            with open(seen_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save seen hashes: {e}")
    
    async def collect_async(self) -> List[SentimentData]:
        """비동기 수집 실행"""
        logger.info(f"Starting async collection for {len(self.stream_ids)} streams")
        
        # 비동기 태스크 생성
        tasks = []
        for stream_id in self.stream_ids:
            task = self._collect_stream_async(stream_id)
            tasks.append(task)
        
        # 동시 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Stream collection failed: {result}")
            elif isinstance(result, list):
                all_data.extend(result)
        
        return all_data
    
    async def _collect_stream_async(self, stream_id: str) -> List[SentimentData]:
        """단일 스트림 비동기 수집"""
        loop = asyncio.get_event_loop()
        
        try:
            # Feedly API 호출 (동기 -> 비동기 변환)
            articles = await loop.run_in_executor(
                self.executor,
                self.feedly.get_feed_articles,
                stream_id,
                self.config['max_articles_per_stream']
            )
            
            # 기사 처리
            processed_data = []
            
            for article_data in articles:
                article = self._parse_article(article_data, stream_id)
                
                if not article:
                    continue
                
                # 중복 체크
                if self.config['enable_deduplication']:
                    if article.content_hash in self.seen_hashes:
                        self.stats['duplicates'] += 1
                        continue
                
                # 감정 분석 (비동기)
                sentiment_result = await loop.run_in_executor(
                    self.executor,
                    self.analyzer.analyze_detailed,
                    {'title': article.title, 'snippet': article.snippet}
                )
                
                # 임계값 필터링
                if abs(sentiment_result.sentiment_score) < self.config['sentiment_threshold']:
                    self.stats['filtered'] += 1
                    continue
                
                # 데이터 생성
                sentiment_data = SentimentData(
                    timestamp=datetime.utcnow(),
                    source=stream_id,
                    title=article.title,
                    snippet=article.snippet[:500],  # 길이 제한
                    sentiment_score=sentiment_result.sentiment_score,
                    label=sentiment_result.label.value,
                    confidence=sentiment_result.confidence,
                    keywords=sentiment_result.keywords,
                    scenario_tag=sentiment_result.scenario_tag,
                    article_id=article.id,
                    url=article.url or ""
                )
                
                processed_data.append(sentiment_data)
                
                # 해시 추가
                self.seen_hashes.add(article.content_hash)
                self.stats['processed'] += 1
            
            logger.info(f"Collected {len(processed_data)} articles from {stream_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to collect from {stream_id}: {e}")
            return []
    
    def collect_sync(self) -> List[SentimentData]:
        """동기 수집 실행"""
        logger.info(f"Starting sync collection for {len(self.stream_ids)} streams")
        
        all_data = []
        
        for stream_id in self.stream_ids:
            try:
                data = self._collect_stream_sync(stream_id)
                all_data.extend(data)
                
            except Exception as e:
                logger.error(f"Failed to collect from {stream_id}: {e}")
                continue
        
        return all_data
    
    def _collect_stream_sync(self, stream_id: str) -> List[SentimentData]:
        """단일 스트림 동기 수집"""
        logger.info(f"Collecting from: {stream_id}")
        
        try:
            articles = self.feedly.get_feed_articles(
                stream_id=stream_id,
                count=self.config['max_articles_per_stream']
            )
            
            processed_data = []
            
            for article_data in articles:
                article = self._parse_article(article_data, stream_id)
                
                if not article:
                    continue
                
                # 중복 체크
                if self.config['enable_deduplication']:
                    if article.content_hash in self.seen_hashes:
                        self.stats['duplicates'] += 1
                        continue
                
                # 감정 분석
                sentiment_result = self.analyzer.analyze_detailed({
                    'title': article.title,
                    'snippet': article.snippet
                })
                
                # 임계값 필터링
                if abs(sentiment_result.sentiment_score) < self.config['sentiment_threshold']:
                    self.stats['filtered'] += 1
                    continue
                
                # 데이터 생성
                sentiment_data = SentimentData(
                    timestamp=datetime.utcnow(),
                    source=stream_id,
                    title=article.title,
                    snippet=article.snippet[:500],
                    sentiment_score=sentiment_result.sentiment_score,
                    label=sentiment_result.label.value,
                    confidence=sentiment_result.confidence,
                    keywords=sentiment_result.keywords,
                    scenario_tag=sentiment_result.scenario_tag,
                    article_id=article.id,
                    url=article.url or ""
                )
                
                processed_data.append(sentiment_data)
                self.seen_hashes.add(article.content_hash)
                self.stats['processed'] += 1
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Stream collection error: {e}")
            return []
    
    def _parse_article(self, article_data: Dict, stream_id: str) -> Optional[Article]:
        """기사 데이터 파싱"""
        try:
            # 필수 필드 확인
            title = article_data.get('title', '').strip()
            if not title:
                return None
            
            # Article 객체 생성
            article = Article(
                id=article_data.get('id', ''),
                title=title,
                snippet=article_data.get('summary', {}).get('content', '')[:1000],
                source=stream_id,
                published=self._parse_timestamp(article_data.get('published')),
                url=article_data.get('canonicalUrl', '')
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_ms: Optional[int]) -> Optional[datetime]:
        """타임스탬프 파싱"""
        if not timestamp_ms:
            return None
            
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000)
        except Exception:
            return None
    
    def save_results(self, data: List[SentimentData]):
        """결과 저장"""
        if not data:
            logger.warning("No data to save")
            return
        
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # CSV 저장
        if self.config['save_format'] in ['csv', 'both']:
            csv_file = self.output_dir / f"sentiment_feedly_{timestamp_str}.csv"
            self._save_csv(data, csv_file)
        
        # JSON 저장
        if self.config['save_format'] in ['json', 'both']:
            json_file = self.output_dir / f"sentiment_feedly_{timestamp_str}.json"
            self._save_json(data, json_file)
        
        # 해시 저장
        self._save_seen_hashes()
        
        # 통계 출력
        self._print_stats()
    
    def _save_csv(self, data: List[SentimentData], file_path: Path):
        """CSV 형식으로 저장"""
        try:
            # 필드명 정의
            fieldnames = [
                'timestamp', 'source', 'title', 'snippet', 
                'sentiment_score', 'label', 'confidence',
                'keywords', 'scenario_tag', 'article_id', 'url'
            ]
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in data:
                    row = asdict(item)
                    # 리스트를 문자열로 변환
                    row['keywords'] = ', '.join(row['keywords'])
                    # 타임스탬프 포맷
                    row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow(row)
            
            logger.info(f"Saved {len(data)} records to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    
    def _save_json(self, data: List[SentimentData], file_path: Path):
        """JSON 형식으로 저장"""
        try:
            # 직렬화 가능한 형태로 변환
            json_data = []
            for item in data:
                item_dict = asdict(item)
                item_dict['timestamp'] = item_dict['timestamp'].isoformat()
                json_data.append(item_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data)} records to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    
    def _print_stats(self):
        """통계 출력"""
        logger.info("=== Collection Statistics ===")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Duplicates: {self.stats['duplicates']}")
        logger.info(f"Filtered: {self.stats['filtered']}")
        logger.info(f"Errors: {self.stats['errors']}")
    
    def run(self):
        """수집 실행 (자동 모드 선택)"""
        if self.config['enable_async']:
            # 비동기 실행
            data = asyncio.run(self.collect_async())
        else:
            # 동기 실행
            data = self.collect_sync()
        
        # 결과 저장
        self.save_results(data)
        
        return data


# 메인 실행
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/sentiment_collector.log')
        ]
    )
    
    # 설정 로드
    config_path = Path("config/sentiment_config.yaml")
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f).get('sentiment', {})
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            config = {}
    else:
        logger.warning("Config file not found, using defaults")
        config = {}
    
    # 수집기 실행
    try:
        collector = FeedlySentimentCollector(config)
        collector.run()
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        raise