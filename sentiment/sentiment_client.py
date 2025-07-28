
# sentiment_client.py - 개선된 버전

import logging
from datetime import datetime
from typing import Optional, Dict, Union
from pathlib import Path
import os
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 가정: 다른 모듈들이 존재한다고 가정
from sentiment.sentiment_router import SentimentScoreRouter
from sentiment.sentiment_event_manager import SentimentEventManager
from sentiment.sentiment_fusion_manager import SentimentFusionManager

logger = logging.getLogger(__name__)

class SentimentMode(Enum):
    """감정 분석 모드"""
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class SentimentConfig:
    """감정 분석 설정"""
    mode: SentimentMode
    csv_path: Optional[Path] = None
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5분
    fusion_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.mode == SentimentMode.BACKTEST and not self.csv_path:
            # 환경변수에서 기본 경로 설정
            default_path = os.getenv("SENTIMENT_CSV_PATH", "data/sentiment/historical.csv")
            self.csv_path = Path(default_path)
            
        if self.csv_path and not self.csv_path.exists():
            logger.warning(f"CSV path does not exist: {self.csv_path}")

class SentimentClient:
    """개선된 감정 분석 클라이언트"""
    
    def __init__(self, config: Union[SentimentConfig, str] = "live", 
                 csv_path: Optional[str] = None):
        """
        Args:
            config: SentimentConfig 객체 또는 모드 문자열
            csv_path: 백테스트용 CSV 경로 (config가 문자열인 경우)
        """
        # 설정 초기화
        if isinstance(config, str):
            mode = SentimentMode(config)
            self.config = SentimentConfig(
                mode=mode,
                csv_path=Path(csv_path) if csv_path else None
            )
        else:
            self.config = config
            
        # 컴포넌트 초기화
        try:
            self._initialize_components()
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
            
        # 캐시 초기화
        self._cache = {}
        self._cache_timestamps = {}
        
        # 비동기 실행을 위한 executor
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # 상태 추적
        self._last_score = 0.5  # 중립 점수로 초기화
        self._score_history = []
        self._max_history = 100
        
    def _initialize_components(self):
        """컴포넌트 초기화"""
        self.router = SentimentScoreRouter(
            mode=self.config.mode.value,
            csv_path=str(self.config.csv_path) if self.config.csv_path else None
        )
        
        self.manager = SentimentEventManager(
            for_backtest=(self.config.mode == SentimentMode.BACKTEST)
        )
        
        self.fusion = SentimentFusionManager(
            weights=self.config.fusion_weights
        )
        
        logger.info(f"Initialized SentimentClient in {self.config.mode.value} mode")

    def get_score(self, timestamp: Optional[datetime] = None, 
                  news_text: Optional[str] = None,
                  use_cache: bool = True) -> float:
        """
        감정 점수 통합 추출 (개선된 버전)
        
        Args:
            timestamp: 백테스트용 타임스탬프
            news_text: 실시간 분석용 뉴스 텍스트
            use_cache: 캐시 사용 여부
            
        Returns:
            정규화된 감정 점수 (0.0 ~ 1.0)
        """
        # 입력 검증
        if self.config.mode == SentimentMode.BACKTEST and not timestamp:
            raise ValueError("Timestamp required for backtest mode")
            
        if self.config.mode == SentimentMode.LIVE and not news_text and not timestamp:
            logger.warning("No input provided, returning cached score")
            return self._last_score
            
        # 캐시 확인
        cache_key = self._get_cache_key(timestamp, news_text)
        if use_cache and self.config.cache_enabled:
            cached_score = self._get_cached_score(cache_key)
            if cached_score is not None:
                logger.debug(f"Using cached score: {cached_score}")
                return cached_score
        
        try:
            # 소스별 점수 수집
            source_scores = self._collect_scores(timestamp, news_text)
            
            if not source_scores:
                logger.warning("No source scores available")
                return self._last_score
            
            # 점수 통합
            fused_score = self.fusion.fuse(source_scores)
            
            # 점수 정규화 (0.0 ~ 1.0)
            normalized_score = self._normalize_score(fused_score)
            
            # 상태 업데이트
            self._update_state(source_scores, normalized_score)
            
            # 캐시 저장
            if use_cache and self.config.cache_enabled:
                self._cache_score(cache_key, normalized_score)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error getting sentiment score: {e}", exc_info=True)
            return self._last_score
    
    async def get_score_async(self, timestamp: Optional[datetime] = None,
                             news_text: Optional[str] = None) -> float:
        """비동기 감정 점수 획득"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_score,
            timestamp,
            news_text
        )
    
    def _collect_scores(self, timestamp: Optional[datetime],
                       news_text: Optional[str]) -> Dict[str, float]:
        """소스별 점수 수집"""
        source_scores = {}
        
        # 타임스탬프 기반 점수
        if timestamp:
            try:
                historical_score = self.router.get_score(timestamp=timestamp)
                if historical_score is not None:
                    source_scores["historical"] = historical_score
            except Exception as e:
                logger.error(f"Error getting historical score: {e}")
        
        # 뉴스 텍스트 기반 점수
        if news_text:
            try:
                news_score = self.router.get_score(news_text=news_text)
                if news_score is not None:
                    source_scores["news"] = news_score
            except Exception as e:
                logger.error(f"Error getting news score: {e}")
        
        # 추가 소스 (확장 가능)
        # source_scores["social"] = self._get_social_sentiment()
        # source_scores["technical"] = self._get_technical_sentiment()
        
        return source_scores
    
    def _normalize_score(self, score: float) -> float:
        """점수를 0.0 ~ 1.0 범위로 정규화"""
        # 입력이 -1 ~ 1 범위라고 가정
        normalized = (score + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))
    
    def _update_state(self, source_scores: Dict[str, float], 
                     fused_score: float):
        """상태 업데이트"""
        # 매니저 업데이트
        self.manager.update_sentiment(source_scores)
        
        # 내부 상태 업데이트
        self._last_score = fused_score
        
        # 히스토리 업데이트
        self._score_history.append({
            'timestamp': datetime.utcnow(),
            'score': fused_score,
            'sources': source_scores.copy()
        })
        
        # 히스토리 크기 제한
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)
    
    def get_latest_delta(self) -> float:
        """직전 감정 점수와의 변화량 반환"""
        delta = self.manager.get_sentiment_delta()
        
        # 정규화된 점수 기준으로 변환
        if len(self._score_history) >= 2:
            current = self._score_history[-1]['score']
            previous = self._score_history[-2]['score']
            return current - previous
            
        return delta
    
    def get_cached_score(self) -> float:
        """최근 감정 점수 반환"""
        manager_score = self.manager.get_sentiment_score()
        
        if manager_score is not None:
            return self._normalize_score(manager_score)
            
        return self._last_score
    
    def get_score_history(self, minutes: int = 60) -> list:
        """지정된 시간 동안의 점수 히스토리 반환"""
        cutoff_time = datetime.utcnow() - datetime.timedelta(minutes=minutes)
        
        return [
            entry for entry in self._score_history
            if entry['timestamp'] >= cutoff_time
        ]
    
    def get_volatility(self, window: int = 20) -> float:
        """감정 점수의 변동성 계산"""
        if len(self._score_history) < window:
            return 0.0
            
        recent_scores = [entry['score'] for entry in self._score_history[-window:]]
        
        # 표준편차 계산
        mean = sum(recent_scores) / len(recent_scores)
        variance = sum((x - mean) ** 2 for x in recent_scores) / len(recent_scores)
        
        return variance ** 0.5
    
    # 캐시 관련 메서드
    def _get_cache_key(self, timestamp: Optional[datetime], 
                      news_text: Optional[str]) -> str:
        """캐시 키 생성"""
        if timestamp:
            return f"ts_{timestamp.isoformat()}"
        elif news_text:
            # 텍스트의 해시값 사용
            import hashlib
            text_hash = hashlib.md5(news_text.encode()).hexdigest()[:8]
            return f"text_{text_hash}"
        else:
            return "default"
    
    def _get_cached_score(self, cache_key: str) -> Optional[float]:
        """캐시에서 점수 조회"""
        if cache_key not in self._cache:
            return None
            
        cached_time = self._cache_timestamps.get(cache_key)
        if not cached_time:
            return None
            
        # TTL 확인
        if (datetime.utcnow() - cached_time).seconds > self.config.cache_ttl:
            # 캐시 만료
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
            
        return self._cache[cache_key]
    
    def _cache_score(self, cache_key: str, score: float):
        """점수 캐싱"""
        self._cache[cache_key] = score
        self._cache_timestamps[cache_key] = datetime.utcnow()
        
        # 캐시 크기 제한
        if len(self._cache) > 1000:
            # 가장 오래된 항목 제거
            oldest_key = min(self._cache_timestamps, 
                           key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
    
    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def close(self):
        """리소스 정리"""
        self._executor.shutdown(wait=True)
        logger.info("SentimentClient closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 동기 테스트
    print("=== 동기 테스트 ===")
    
    # 라이브 모드
    with SentimentClient(mode="live") as client:
        score = client.get_score(news_text="Bitcoin hits new high after ETF approval")
        print(f"감정 점수: {score:.4f}")
        print(f"감정 변화량 Δ: {client.get_latest_delta():.4f}")
        print(f"변동성: {client.get_volatility():.4f}")
    
    # 백테스트 모드
    config = SentimentConfig(
        mode=SentimentMode.BACKTEST,
        csv_path=Path("data/sentiment_history.csv")
    )
    
    with SentimentClient(config) as client:
        score = client.get_score(timestamp=datetime.utcnow())
        print(f"\n백테스트 점수: {score:.4f}")
    
    # 비동기 테스트
    async def async_test():
        print("\n=== 비동기 테스트 ===")
        client = SentimentClient(mode="live")
        
        tasks = [
            client.get_score_async(news_text="Market crashes on Fed announcement"),
            client.get_score_async(news_text="Tech stocks rally on earnings beat"),
            client.get_score_async(news_text="Oil prices surge amid supply concerns")
        ]
        
        scores = await asyncio.gather(*tasks)
        
        for i, score in enumerate(scores, 1):
            print(f"비동기 점수 {i}: {score:.4f}")
        
        client.close()
    
    # 비동기 실행
    asyncio.run(async_test())