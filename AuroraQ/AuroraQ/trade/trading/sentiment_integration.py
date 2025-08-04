#!/usr/bin/env python3
"""
Trading <-> Sentiment Service Integration
트레이딩 시스템과 감정 분석 서비스 간 통합 모듈
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import sys
from pathlib import Path

# 프로젝트 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

# 환경 설정
try:
    from config.env_loader import get_vps_env_config
    env_config = get_vps_env_config()
except ImportError:
    # Fallback 설정 (감정 분석 서비스 없이 동작)
    class MockEnvConfig:
        sentiment_service_url = "http://localhost:8000"
        sentiment_confidence_threshold = 0.6
        enable_sentiment_analysis = False  # 테스트 환경에서는 비활성화
    env_config = MockEnvConfig()

@dataclass
class SentimentScore:
    """감정 분석 점수"""
    value: float = 0.0  # -1.0 (매우 부정) ~ 1.0 (매우 긍정)
    confidence: float = 0.0  # 0.0 ~ 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "unknown"
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """유효한 점수인지 확인"""
        return self.confidence >= env_config.sentiment_confidence_threshold
    
    @property
    def normalized_score(self) -> float:
        """정규화된 점수 (0.0 ~ 1.0)"""
        return (self.value + 1.0) / 2.0
    
    @property
    def weighted_score(self) -> float:
        """신뢰도로 가중된 점수"""
        return self.value * self.confidence if self.is_valid else 0.0

@dataclass 
class MarketSentiment:
    """시장 전체 감정 상태"""
    overall_score: float = 0.0
    fear_greed_index: float = 0.5  # 0.0 (극도의 공포) ~ 1.0 (극도의 탐욕)
    volatility_sentiment: float = 0.0
    trend_sentiment: float = 0.0
    social_sentiment: float = 0.0
    news_sentiment: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_feature_vector(self) -> List[float]:
        """ML 모델용 특성 벡터로 변환"""
        return [
            self.overall_score,
            self.fear_greed_index,
            self.volatility_sentiment,
            self.trend_sentiment,
            self.social_sentiment,
            self.news_sentiment
        ]

class SentimentCache:
    """감정 분석 결과 캐시"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, SentimentScore] = {}
        self.timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.history = deque(maxlen=100)  # 최근 100개 기록
        
    def get(self, key: str) -> Optional[SentimentScore]:
        """캐시에서 감정 점수 조회"""
        if key not in self.cache:
            return None
            
        # TTL 확인
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None
            
        return self.cache[key]
    
    def set(self, key: str, value: SentimentScore):
        """캐시에 감정 점수 저장"""
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.history.append(value)
    
    def get_recent_average(self, count: int = 10) -> float:
        """최근 N개 감정 점수 평균"""
        if not self.history:
            return 0.0
        
        recent_scores = list(self.history)[-count:]
        valid_scores = [s.weighted_score for s in recent_scores if s.is_valid]
        
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

class SentimentServiceClient:
    """Sentiment Service API 클라이언트"""
    
    def __init__(self, 
                 base_url: str = None,
                 timeout: int = 10,
                 retry_attempts: int = 3):
        self.base_url = base_url or env_config.sentiment_service_url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.cache = SentimentCache()
        self.logger = logging.getLogger(__name__)
        
        # 연결 상태
        self.is_connected = False
        self.last_error = None
        self.stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0
        }
    
    async def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/health") as response:
                    self.is_connected = response.status == 200
                    return self.is_connected
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            return False
    
    async def get_sentiment_score(self, 
                                 text: str = "",
                                 symbol: str = "BTCUSDT",
                                 use_cache: bool = True) -> SentimentScore:
        """감정 분석 점수 조회"""
        # 감정 분석이 비활성화된 경우 즉시 fallback 반환
        if not env_config.enable_sentiment_analysis:
            return SentimentScore(
                value=0.0,
                confidence=0.3,  # 낮은 신뢰도
                timestamp=datetime.now().isoformat(),
                source="disabled",
                raw_data={"status": "sentiment_analysis_disabled"}
            )
        
        # 캐시 확인
        cache_key = f"{symbol}:{hash(text)}" if text else symbol
        if use_cache:
            cached_score = self.cache.get(cache_key)
            if cached_score:
                self.stats['cache_hits'] += 1
                return cached_score
        
        self.stats['requests_total'] += 1
        start_time = time.time()
        
        try:
            # API 호출
            payload = {
                "text": text or f"Market sentiment for {symbol}",
                "symbol": symbol,
                "use_onnx": True
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}/metrics/sentiment/analyze",
                    json=payload
                ) as response:
                    response_time = time.time() - start_time
                    self._update_response_time(response_time)
                    
                    if response.status == 200:
                        data = await response.json()
                        score = self._parse_sentiment_response(data, symbol)
                        
                        # 캐시 저장
                        if use_cache:
                            self.cache.set(cache_key, score)
                        
                        self.stats['requests_success'] += 1
                        return score
                    else:
                        raise Exception(f"API error: {response.status}")
                        
        except Exception as e:
            self.stats['requests_failed'] += 1
            self.last_error = str(e)
            self.logger.warning(f"Sentiment API failed: {e}")
            
            # Fallback 점수 반환
            return SentimentScore(
                value=0.0,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                source="fallback_error",
                raw_data={"error": str(e)}
            )
    
    async def get_market_sentiment(self, symbol: str = "BTCUSDT") -> MarketSentiment:
        """시장 전체 감정 상태 조회"""
        try:
            # 병렬로 여러 감정 지표 수집
            tasks = [
                self.get_sentiment_score("", symbol),  # 기본 감정
                self.get_sentiment_score("market fear greed", symbol),  # 공포/탐욕
                self.get_sentiment_score("volatility market", symbol),  # 변동성
                self.get_sentiment_score("trend analysis", symbol),  # 트렌드
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 파싱
            scores = []
            for result in results:
                if isinstance(result, SentimentScore) and result.is_valid:
                    scores.append(result.weighted_score)
                else:
                    scores.append(0.0)  # 기본값
            
            # MarketSentiment 구성
            overall_score = sum(scores) / len(scores) if scores else 0.0
            fear_greed = (scores[1] + 1.0) / 2.0 if len(scores) > 1 else 0.5
            
            return MarketSentiment(
                overall_score=overall_score,
                fear_greed_index=fear_greed,
                volatility_sentiment=scores[2] if len(scores) > 2 else 0.0,
                trend_sentiment=scores[3] if len(scores) > 3 else 0.0,
                social_sentiment=self.cache.get_recent_average(20),
                news_sentiment=self.cache.get_recent_average(10),
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get market sentiment: {e}")
            return MarketSentiment()  # 기본값 반환
    
    def _parse_sentiment_response(self, data: Dict[str, Any], symbol: str) -> SentimentScore:
        """API 응답을 SentimentScore로 변환"""
        try:
            # 다양한 응답 형식 지원
            if 'sentiment' in data:
                value = float(data['sentiment'])
                confidence = float(data.get('confidence', 0.5))
            elif 'score' in data:
                value = float(data['score'])
                confidence = float(data.get('confidence', 0.5))
            else:
                # 기본값
                value = 0.0
                confidence = 0.0
            
            return SentimentScore(
                value=max(-1.0, min(1.0, value)),  # -1.0 ~ 1.0 범위 제한
                confidence=max(0.0, min(1.0, confidence)),  # 0.0 ~ 1.0 범위 제한
                timestamp=datetime.now().isoformat(),
                source="sentiment_service",
                raw_data=data
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse sentiment response: {e}")
            return SentimentScore(
                value=0.0,
                confidence=0.0,
                source="parse_error",
                raw_data=data
            )
    
    def _update_response_time(self, response_time: float):
        """응답 시간 통계 업데이트"""
        current_avg = self.stats['avg_response_time']
        total_requests = self.stats['requests_total']
        
        self.stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """클라이언트 통계 조회"""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(1, self.stats['requests_total']) * 100
        )
        
        success_rate = (
            self.stats['requests_success'] / max(1, self.stats['requests_total']) * 100
        )
        
        return {
            **self.stats,
            'cache_hit_rate_percent': cache_hit_rate,
            'success_rate_percent': success_rate,
            'is_connected': self.is_connected,
            'last_error': self.last_error,
            'cache_size': len(self.cache.cache)
        }

# 전역 클라이언트 인스턴스
_global_sentiment_client: Optional[SentimentServiceClient] = None

def get_sentiment_client() -> SentimentServiceClient:
    """전역 감정 분석 클라이언트 가져오기 (싱글톤)"""
    global _global_sentiment_client
    
    if _global_sentiment_client is None:
        _global_sentiment_client = SentimentServiceClient()
    
    return _global_sentiment_client

async def get_current_sentiment(symbol: str = "BTCUSDT") -> SentimentScore:
    """현재 감정 점수 조회 (편의 함수)"""
    client = get_sentiment_client()
    return await client.get_sentiment_score(symbol=symbol)

async def get_market_sentiment_state(symbol: str = "BTCUSDT") -> MarketSentiment:
    """시장 감정 상태 조회 (편의 함수)"""
    client = get_sentiment_client()
    return await client.get_market_sentiment(symbol)

# 테스트 및 데모
async def demo_sentiment_integration():
    """감정 분석 통합 데모"""
    print("🧠 Sentiment Integration Demo")
    print("=" * 50)
    
    client = get_sentiment_client()
    
    # 서비스 상태 확인
    print("1. Health Check...")
    is_healthy = await client.health_check()
    print(f"   Service Status: {'✅ Connected' if is_healthy else '❌ Disconnected'}")
    
    # 감정 점수 조회
    print("\n2. Getting Sentiment Score...")
    sentiment = await client.get_sentiment_score(
        text="Bitcoin price is showing strong upward momentum",
        symbol="BTCUSDT"
    )
    print(f"   Score: {sentiment.value:.3f}")
    print(f"   Confidence: {sentiment.confidence:.3f}")
    print(f"   Valid: {sentiment.is_valid}")
    print(f"   Weighted: {sentiment.weighted_score:.3f}")
    
    # 시장 감정 상태
    print("\n3. Getting Market Sentiment...")
    market_sentiment = await client.get_market_sentiment("BTCUSDT")
    print(f"   Overall Score: {market_sentiment.overall_score:.3f}")
    print(f"   Fear/Greed Index: {market_sentiment.fear_greed_index:.3f}")
    print(f"   Feature Vector: {market_sentiment.to_feature_vector()}")
    
    # 통계
    print("\n4. Client Statistics...")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_sentiment_integration())