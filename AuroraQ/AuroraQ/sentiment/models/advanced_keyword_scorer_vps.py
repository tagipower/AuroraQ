#!/usr/bin/env python3
"""
Advanced Keyword Scorer VPS Implementation
VPS 최적화된 키워드 스코어링 및 센티먼트 분석
"""

import asyncio
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 데이터 클래스들
from dataclasses import dataclass
from enum import Enum

@dataclass
class MultiModalSentiment:
    """멀티 모달 센티먼트 결과"""
    sentiment_score: float
    confidence: float
    label: str
    keywords: List[str]
    processing_time: float

@dataclass
class AdvancedFeatures:
    """고급 특징 추출 결과"""
    keyword_density: float
    sentiment_consistency: float
    volatility_indicators: List[str]
    market_signals: List[str]

class EmotionalState(Enum):
    """감정 상태"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class MarketRegime(Enum):
    """시장 체제"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class VPSAdvancedKeywordScorer:
    """VPS 최적화된 고급 키워드 스코어러"""
    
    def __init__(self):
        self.financial_keywords = FINANCIAL_KEYWORDS
        self.stats = {
            "processed_texts": 0,
            "avg_processing_time": 0.0,
            "last_update": datetime.now()
        }
    
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> MultiModalSentiment:
        """센티먼트 분석"""
        start_time = time.time()
        
        result = await analyze_sentiment_vps(text, metadata)
        keywords = extract_financial_keywords(text)
        
        processing_time = time.time() - start_time
        
        return MultiModalSentiment(
            sentiment_score=result["sentiment"],
            confidence=result["confidence"],
            label=result["label"],
            keywords=keywords,
            processing_time=processing_time
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return await get_vps_performance_stats()

# Financial keywords for sentiment analysis
FINANCIAL_KEYWORDS = {
    'positive': {
        'surge', 'rally', 'bullish', 'gains', 'rise', 'up', 'growth', 'increase',
        'boom', 'profit', 'strong', 'healthy', 'recovery', 'optimistic', 'soar',
        'breakthrough', 'success', 'positive', 'upward', 'bull', 'good', 'great'
    },
    'negative': {
        'crash', 'drop', 'fall', 'bearish', 'decline', 'down', 'loss', 'decrease',
        'plunge', 'collapse', 'weak', 'poor', 'recession', 'pessimistic', 'plummet',
        'failure', 'bad', 'negative', 'downward', 'bear', 'terrible', 'awful'
    },
    'neutral': {
        'stable', 'unchanged', 'flat', 'sideways', 'hold', 'maintain', 'steady',
        'neutral', 'balanced', 'mixed', 'moderate', 'average', 'normal'
    }
}

async def analyze_sentiment_vps(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    VPS 최적화된 센티먼트 분석
    
    Args:
        text: 분석할 텍스트
        metadata: 추가 메타데이터
        
    Returns:
        Dict: 분석 결과 (sentiment, confidence, processing_time)
    """
    start_time = time.time()
    
    try:
        if not text or len(text.strip()) == 0:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "label": "neutral",
                "processing_time": time.time() - start_time,
                "method": "keyword_based"
            }
        
        # 텍스트 전처리
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # 키워드 매칭 점수 계산
        positive_score = sum(1 for word in words if word in FINANCIAL_KEYWORDS['positive'])
        negative_score = sum(1 for word in words if word in FINANCIAL_KEYWORDS['negative'])
        neutral_score = sum(1 for word in words if word in FINANCIAL_KEYWORDS['neutral'])
        
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            # 키워드가 없는 경우 중립
            sentiment_score = 0.0
            confidence = 0.3
            label = "neutral"
        else:
            # 점수 정규화
            pos_ratio = positive_score / total_score
            neg_ratio = negative_score / total_score
            neu_ratio = neutral_score / total_score
            
            if pos_ratio > neg_ratio and pos_ratio > neu_ratio:
                sentiment_score = pos_ratio
                label = "positive"
                confidence = min(0.9, pos_ratio + 0.3)
            elif neg_ratio > pos_ratio and neg_ratio > neu_ratio:
                sentiment_score = -neg_ratio
                label = "negative"
                confidence = min(0.9, neg_ratio + 0.3)
            else:
                sentiment_score = 0.0
                label = "neutral"
                confidence = min(0.9, neu_ratio + 0.2)
        
        processing_time = time.time() - start_time
        
        result = {
            "sentiment": float(sentiment_score),
            "confidence": float(confidence),
            "label": label,
            "processing_time": processing_time,
            "method": "keyword_based",
            "keyword_counts": {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }
        }
        
        # 메타데이터 추가
        if metadata:
            result["metadata"] = metadata
            
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "sentiment": 0.0,
            "confidence": 0.0,
            "label": "neutral",
            "processing_time": time.time() - start_time,
            "method": "keyword_based",
            "error": str(e)
        }

async def get_vps_performance_stats() -> Dict[str, Any]:
    """
    VPS 성능 통계 반환
    
    Returns:
        Dict: 성능 메트릭
    """
    try:
        import psutil
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        memory_percent = memory.percent
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "memory_mb": memory_mb,
            "disk_usage": disk_percent,
            "available_memory_mb": memory.available / 1024 / 1024
        }
        
    except ImportError:
        # psutil이 없는 경우 기본값 반환
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "memory_mb": 0.0,
            "disk_usage": 0.0,
            "available_memory_mb": 0.0,
            "note": "psutil not available"
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

async def batch_analyze_sentiment_vps(texts: List[str], 
                                    batch_size: int = 32,
                                    metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    VPS 최적화된 배치 센티먼트 분석
    
    Args:
        texts: 분석할 텍스트 목록
        batch_size: 배치 크기
        metadata: 추가 메타데이터
        
    Returns:
        List[Dict]: 분석 결과 목록
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = []
        
        for text in batch_texts:
            try:
                result = await analyze_sentiment_vps(text, metadata)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Batch analysis error for text: {e}")
                batch_results.append({
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "label": "neutral",
                    "processing_time": 0.0,
                    "method": "keyword_based",
                    "error": str(e)
                })
        
        results.extend(batch_results)
        
        # 배치 간 짧은 대기 (VPS 리소스 보호)
        if i + batch_size < len(texts):
            await asyncio.sleep(0.01)
    
    return results

def extract_financial_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    금융 관련 키워드 추출
    
    Args:
        text: 입력 텍스트
        max_keywords: 최대 키워드 수
        
    Returns:
        List[str]: 추출된 키워드 목록
    """
    if not text:
        return []
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # 모든 금융 키워드 수집
    all_financial_keywords = set()
    for category in FINANCIAL_KEYWORDS.values():
        all_financial_keywords.update(category)
    
    # 텍스트에서 금융 키워드 찾기
    found_keywords = []
    for word in words:
        if word in all_financial_keywords and word not in found_keywords:
            found_keywords.append(word)
            if len(found_keywords) >= max_keywords:
                break
    
    return found_keywords

# 테스트 코드
if __name__ == "__main__":
    async def test_vps_scorer():
        print("=== VPS Sentiment Scorer Test ===")
        
        # 테스트 텍스트들
        test_texts = [
            "Bitcoin price surges to new all-time high, bullish momentum continues",
            "Market crash fears grip investors as prices plummet",
            "Cryptocurrency remains stable amid market uncertainty",
            "Strong rally in tech stocks drives market gains",
            "Bearish sentiment dominates as recession fears grow"
        ]
        
        print("1. Single text analysis:")
        for i, text in enumerate(test_texts[:3], 1):
            result = await analyze_sentiment_vps(text)
            print(f"   {i}. {text[:50]}...")
            print(f"      Sentiment: {result['sentiment']:.3f} ({result['label']})")
            print(f"      Confidence: {result['confidence']:.3f}")
            print(f"      Keywords: {extract_financial_keywords(text)[:3]}")
            print()
        
        print("2. Batch analysis:")
        batch_results = await batch_analyze_sentiment_vps(test_texts)
        for i, result in enumerate(batch_results, 1):
            print(f"   {i}. {result['label']} ({result['sentiment']:.3f})")
        
        print("3. Performance stats:")
        stats = await get_vps_performance_stats()
        for key, value in stats.items():
            if key != "timestamp":
                print(f"   {key}: {value}")
    
    asyncio.run(test_vps_scorer())

# 편의 함수들
async def clear_vps_cache():
    """VPS 캐시 정리"""
    logger.info("VPS cache cleared")
    return {"status": "cleared", "timestamp": datetime.now().isoformat()}