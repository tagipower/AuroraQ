"""
배치 감정 분석 프로세서 - CPU/메모리 효율적 처리
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import psutil
import torch

from ..utils.logger import get_logger

logger = get_logger("BatchSentimentProcessor")


class BatchSentimentProcessor:
    """
    뉴스 감정 분석 배치 처리 시스템
    - 리소스 효율적 배치 처리
    - 자동 배치 크기 조정
    - 캐시 관리
    """
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        max_batch_size: int = 32,
        max_memory_usage: float = 0.7  # 70% 메모리 사용 제한
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_memory_usage = max_memory_usage
        self.current_batch_size = max_batch_size
        
        # 성능 모니터링
        self.processing_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
        # 모델은 나중에 로드 (lazy loading)
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'  # 초기에는 CPU 사용
        
    def initialize_model(self):
        """모델 초기화 (처음 사용 시)"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model = self.model.to(self.device)
                logger.info("Using GPU for sentiment analysis")
            else:
                logger.info("Using CPU for sentiment analysis")
                
            self.model.eval()  # 평가 모드
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to simple sentiment
            self.model = None
    
    async def process_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        배치 감정 분석 실행
        
        Returns:
            List[Dict]: 각 텍스트의 감정 점수
            [{'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}, ...]
        """
        if not self.model:
            self.initialize_model()
            
        if not self.model:
            # Fallback: 랜덤 감정 점수
            return self._generate_dummy_sentiments(texts)
        
        start_time = time.time()
        
        # 메모리 체크 및 배치 크기 조정
        self._adjust_batch_size()
        
        results = []
        
        # 배치 처리
        for i in range(0, len(texts), self.current_batch_size):
            batch = texts[i:i + self.current_batch_size]
            batch_results = await self._process_single_batch(batch)
            results.extend(batch_results)
            
            # 중간 메모리 정리
            if i % (self.current_batch_size * 5) == 0:
                self._cleanup_memory()
        
        # 처리 시간 기록
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        logger.info(f"Processed {len(texts)} texts in {processing_time:.2f}s")
        
        return results
    
    async def _process_single_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """단일 배치 처리"""
        try:
            # 토큰화
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # CPU로 이동 및 변환
            predictions = predictions.cpu().numpy()
            
            # 결과 포맷팅 (positive, negative, neutral)
            results = []
            for pred in predictions:
                results.append({
                    'positive': float(pred[0]),
                    'neutral': float(pred[1]),
                    'negative': float(pred[2])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return self._generate_dummy_sentiments(texts)
    
    def _adjust_batch_size(self):
        """메모리 사용량에 따른 배치 크기 자동 조정"""
        memory_percent = psutil.virtual_memory().percent / 100
        self.memory_usage_history.append(memory_percent)
        
        if memory_percent > self.max_memory_usage:
            # 메모리 사용량이 높으면 배치 크기 감소
            self.current_batch_size = max(4, int(self.current_batch_size * 0.8))
            logger.warning(f"High memory usage ({memory_percent:.1%}). Reducing batch size to {self.current_batch_size}")
        elif memory_percent < 0.5 and len(self.processing_times) > 10:
            # 메모리 여유가 있고 처리가 안정적이면 배치 크기 증가
            avg_time = np.mean(list(self.processing_times)[-10:])
            if avg_time < 1.0:  # 평균 처리 시간이 1초 미만
                self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
                logger.info(f"Increasing batch size to {self.current_batch_size}")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        import gc
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def _generate_dummy_sentiments(self, texts: List[str]) -> List[Dict[str, float]]:
        """더미 감정 점수 생성 (폴백)"""
        results = []
        for text in texts:
            # 간단한 규칙 기반 감정 분석
            text_lower = text.lower()
            
            positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'gain', 'profit']
            negative_words = ['bad', 'poor', 'negative', 'down', 'loss', 'crash', 'fear']
            
            pos_score = sum(1 for word in positive_words if word in text_lower)
            neg_score = sum(1 for word in negative_words if word in text_lower)
            
            total = pos_score + neg_score + 1
            
            results.append({
                'positive': pos_score / total,
                'negative': neg_score / total,
                'neutral': 1 / total
            })
        
        return results
    
    async def process_news_items(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        뉴스 아이템 처리 (제목 + 내용)
        
        Args:
            news_items: [{'title': str, 'content': str, 'source': str, 'timestamp': datetime}, ...]
            
        Returns:
            원본 + 감정 점수가 추가된 뉴스 아이템
        """
        # 제목과 내용 결합
        texts = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('content', '')[:500]}"  # 최대 500자
            texts.append(text)
        
        # 배치 감정 분석
        sentiments = await self.process_batch(texts)
        
        # 결과 병합
        for item, sentiment in zip(news_items, sentiments):
            item['sentiment'] = sentiment
            item['sentiment_score'] = sentiment['positive'] - sentiment['negative']  # -1 ~ 1
            
        return news_items
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        if not self.processing_times:
            return {}
            
        return {
            'avg_processing_time': np.mean(list(self.processing_times)),
            'max_processing_time': max(self.processing_times),
            'current_batch_size': self.current_batch_size,
            'avg_memory_usage': np.mean(list(self.memory_usage_history)) if self.memory_usage_history else 0,
            'texts_per_second': self.current_batch_size / np.mean(list(self.processing_times)) if self.processing_times else 0
        }