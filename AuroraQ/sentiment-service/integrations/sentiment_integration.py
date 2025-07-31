#!/usr/bin/env python3
"""감정 분석 통합 모듈"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

class SentimentIntegration:
    """감정 분석 통합 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.cache = {}
    
    async def analyze_text(self, text: str, model: str = "default") -> Dict[str, Any]:
        """텍스트 감정 분석"""
        try:
            # 간단한 키워드 기반 감정 분석
            positive_words = ["good", "great", "excellent", "amazing", "bullish", "positive", "up", "rise", "gain"]
            negative_words = ["bad", "terrible", "awful", "bearish", "negative", "down", "fall", "drop", "loss"]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                score = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                score = max(-0.8, -0.5 - (negative_count - positive_count) * 0.1)
            else:
                sentiment = "neutral"
                score = 0.0
            
            return {
                "sentiment": sentiment,
                "score": score,
                "confidence": min(1.0, abs(score) + 0.2),
                "model": model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "model": model,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def batch_analyze(self, texts: List[str], model: str = "default") -> List[Dict[str, Any]]:
        """배치 감정 분석"""
        results = []
        
        for text in texts:
            result = await self.analyze_text(text, model)
            results.append(result)
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """통합 서비스 상태 확인"""
        try:
            # 테스트 분석
            test_result = await self.analyze_text("This is a test message")
            
            return {
                "status": "healthy",
                "models_available": ["default"],
                "test_analysis_successful": "sentiment" in test_result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# 전역 인스턴스
sentiment_integration = SentimentIntegration()

# 편의 함수
async def analyze_sentiment(text: str, model: str = "default") -> Dict[str, Any]:
    """감정 분석 편의 함수"""
    return await sentiment_integration.analyze_text(text, model)

async def batch_analyze_sentiment(texts: List[str], model: str = "default") -> List[Dict[str, Any]]:
    """배치 감정 분석 편의 함수"""
    return await sentiment_integration.batch_analyze(texts, model)