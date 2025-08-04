#!/usr/bin/env python3
"""
FinBERT Batch Processor V2 for AuroraQ Sentiment Service
VPS 최적화 버전 - 메모리 효율성, 동적 배치 크기, CPU 최적화
"""

import asyncio
import time
import torch
import gc
import psutil
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import numpy as np

# Transformers 라이브러리
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)

logger = logging.getLogger(__name__)

# VPS 환경 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 병렬처리 경고 방지
torch.set_num_threads(2)  # CPU 스레드 제한

class ProcessingStatus(Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class BatchItem:
    """배치 처리 아이템"""
    content_hash: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbol: Optional[str] = None
    category: str = "general"
    priority: int = 1  # 1=high, 2=medium, 3=low
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class BatchResult:
    """배치 처리 결과"""
    content_hash: str
    sentiment_score: float
    confidence: float
    label: str
    keywords: List[str]
    processing_time: float
    model_version: str
    batch_id: str
    processed_at: datetime

class ResourceMonitor:
    """시스템 리소스 모니터"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.history = deque(maxlen=100)
        
    def get_current_usage(self) -> Dict[str, float]:
        """현재 리소스 사용량"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        usage = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'memory_percent': self.process.memory_percent()
        }
        
        self.history.append({
            'timestamp': time.time(),
            **usage
        })
        
        return usage
    
    def get_average_usage(self, seconds: int = 60) -> Dict[str, float]:
        """평균 리소스 사용량"""
        if not self.history:
            return {'cpu_percent': 0, 'memory_mb': 0}
        
        cutoff_time = time.time() - seconds
        recent_history = [h for h in self.history if h['timestamp'] > cutoff_time]
        
        if not recent_history:
            return {'cpu_percent': 0, 'memory_mb': 0}
        
        return {
            'cpu_percent': sum(h['cpu_percent'] for h in recent_history) / len(recent_history),
            'memory_mb': sum(h['memory_mb'] for h in recent_history) / len(recent_history)
        }
    
    def should_reduce_batch_size(self) -> bool:
        """배치 크기를 줄여야 하는지 확인"""
        current = self.get_current_usage()
        
        # 메모리 2GB 이상 또는 CPU 80% 이상
        return current['memory_mb'] > 2048 or current['cpu_percent'] > 80

class FinBERTBatchProcessorV2:
    """FinBERT 배치 프로세서 V2 - VPS 최적화"""
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 cache_manager: Optional[Any] = None,
                 initial_batch_size: int = 8,
                 max_batch_size: int = 16,
                 min_batch_size: int = 2,
                 max_sequence_length: int = 256):  # 512에서 256으로 감소
        """
        초기화
        
        Args:
            model_name: FinBERT 모델 이름
            cache_manager: 캐시 매니저
            initial_batch_size: 초기 배치 크기
            max_batch_size: 최대 배치 크기
            min_batch_size: 최소 배치 크기
            max_sequence_length: 최대 시퀀스 길이
        """
        self.model_name = model_name
        self.cache_manager = cache_manager
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_sequence_length = max_sequence_length
        
        # 모델 관련
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_version: str = "finbert_v2.0"
        self.device = "cpu"  # VPS에서 CPU 사용
        self.model_lock = threading.Lock()
        
        # 리소스 모니터
        self.resource_monitor = ResourceMonitor()
        
        # 배치 관리
        self.batch_queue: deque = deque()
        self.processing_results: deque = deque(maxlen=1000)
        self.failed_items: deque = deque(maxlen=100)
        
        # 스케줄링
        self.batch_interval = 900  # 15분 (900초)
        self.last_batch_time = 0
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # 성능 통계
        self.stats = {
            "total_processed": 0,
            "total_batches": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "model_load_time": 0.0,
            "last_batch_time": None,
            "items_per_batch_avg": 0.0,
            "batch_size_adjustments": 0,
            "memory_peaks": []
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=1)  # 단일 워커
        
        # 키워드 캐시
        self.keyword_cache = {}
        self.cache_size_limit = 1000
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize_model()
        await self.start_processor()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop_processor()
        self.thread_pool.shutdown(wait=True)
        
        # 모델 메모리 해제
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # 가비지 컬렉션
        gc.collect()
    
    async def initialize_model(self) -> bool:
        """모델 초기화 (메모리 효율적)"""
        
        if self.model is not None:
            logger.info("FinBERT model already loaded")
            return True
        
        logger.info(f"Loading FinBERT model: {self.model_name}")
        start_time = time.time()
        
        try:
            def load_model():
                """스레드에서 모델 로드"""
                # 토크나이저 로드
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    model_max_length=self.max_sequence_length
                )
                
                # 모델 로드 (메모리 효율적 설정)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True  # 메모리 효율적 로딩
                )
                
                # CPU로 이동 및 최적화
                model = model.to(self.device)
                model.eval()  # 평가 모드
                
                # 모델 최적화
                for param in model.parameters():
                    param.requires_grad = False  # 그래디언트 비활성화
                
                return tokenizer, model
            
            # 스레드 풀에서 모델 로드
            future = self.thread_pool.submit(load_model)
            self.tokenizer, self.model = await asyncio.wrap_future(future)
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            # 초기 메모리 사용량 기록
            usage = self.resource_monitor.get_current_usage()
            logger.info(f"FinBERT model loaded in {load_time:.2f}s. "
                       f"Memory: {usage['memory_mb']:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            return False
    
    def _extract_keywords_cached(self, text: str, top_k: int = 5) -> List[str]:
        """캐시된 키워드 추출"""
        # 텍스트 해시를 캐시 키로 사용
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        
        if cache_key in self.keyword_cache:
            return self.keyword_cache[cache_key]
        
        keywords = self._extract_keywords(text, top_k)
        
        # 캐시 크기 제한
        if len(self.keyword_cache) >= self.cache_size_limit:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(self.keyword_cache))
            del self.keyword_cache[oldest_key]
        
        self.keyword_cache[cache_key] = keywords
        return keywords
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """간단한 키워드 추출 (메모리 효율적)"""
        
        # 금융 관련 중요 키워드 (축소된 목록)
        important_keywords = {
            "bitcoin", "ethereum", "crypto", "stock", "market", 
            "trading", "price", "surge", "drop", "rally", "crash",
            "bullish", "bearish", "regulation", "policy"
        }
        
        # 텍스트 전처리
        words = text.lower().split()[:100]  # 최대 100단어만 처리
        word_freq = {}
        
        for word in words:
            # 특수문자 제거
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if len(clean_word) >= 3 and len(clean_word) <= 20:  # 길이 제한
                # 중요 키워드에 가중치 부여
                weight = 2 if clean_word in important_keywords else 1
                word_freq[clean_word] = word_freq.get(clean_word, 0) + weight
        
        # 빈도순 정렬 후 상위 키워드 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k] if freq > 1 or word in important_keywords]
    
    def _process_single_text(self, text: str) -> Tuple[float, float, str]:
        """단일 텍스트 처리 (최적화)"""
        
        with self.model_lock:
            try:
                if not self.model or not self.tokenizer:
                    raise ValueError("Model not initialized")
                
                # 텍스트 길이 제한
                if len(text) > self.max_sequence_length * 4:
                    text = text[:self.max_sequence_length * 4]
                
                # 토크나이징
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_sequence_length,
                    padding=True
                )
                
                # 추론 (그래디언트 비활성화)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # 레이블과 점수 추출
                    confidence, predicted = torch.max(predictions, dim=-1)
                    confidence = float(confidence.item())
                    label_id = predicted.item()
                    
                    # 레이블 매핑 (FinBERT 특정)
                    label_map = {0: "positive", 1: "negative", 2: "neutral"}
                    label = label_map.get(label_id, "neutral")
                    
                    # 감정 점수 계산
                    if label == "positive":
                        sentiment_score = confidence
                    elif label == "negative":
                        sentiment_score = -confidence
                    else:  # neutral
                        sentiment_score = 0.0
                
                return sentiment_score, confidence, label
                
            except Exception as e:
                logger.error(f"FinBERT processing failed: {e}")
                return 0.0, 0.0, "neutral"
    
    async def process_batch(self, batch_items: List[BatchItem]) -> List[BatchResult]:
        """배치 처리 (최적화)"""
        
        if not batch_items:
            return []
        
        batch_id = hashlib.md5(
            f"{datetime.now().isoformat()}{len(batch_items)}".encode()
        ).hexdigest()[:8]
        
        logger.info(f"Processing batch {batch_id} with {len(batch_items)} items "
                   f"(batch size: {self.current_batch_size})")
        
        start_time = time.time()
        results = []
        
        # 리소스 사용량 확인
        initial_usage = self.resource_monitor.get_current_usage()
        
        # 배치 아이템들을 처리
        for i, item in enumerate(batch_items):
            item_start = time.time()
            item.status = ProcessingStatus.PROCESSING
            
            try:
                # 리소스 체크 (매 5개 항목마다)
                if i > 0 and i % 5 == 0:
                    if self.resource_monitor.should_reduce_batch_size():
                        logger.warning("High resource usage detected, stopping batch early")
                        # 나머지 항목들을 큐에 다시 추가
                        for remaining_item in batch_items[i:]:
                            remaining_item.status = ProcessingStatus.PENDING
                            self.batch_queue.appendleft(remaining_item)
                        break
                
                # 텍스트 준비
                full_text = f"{item.title}. {item.content}"
                
                # FinBERT 분석 (스레드 풀에서 실행)
                future = self.thread_pool.submit(self._process_single_text, full_text)
                sentiment_score, confidence, label = await asyncio.wrap_future(future)
                
                # 키워드 추출 (캐시 사용)
                keywords = self._extract_keywords_cached(full_text)
                
                processing_time = time.time() - item_start
                
                # 결과 생성
                result = BatchResult(
                    content_hash=item.content_hash,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    label=label,
                    keywords=keywords,
                    processing_time=processing_time,
                    model_version=self.model_version,
                    batch_id=batch_id,
                    processed_at=datetime.now()
                )
                
                results.append(result)
                item.status = ProcessingStatus.COMPLETED
                item.processing_time = processing_time
                
                # 캐시 매니저에 결과 저장
                if self.cache_manager:
                    try:
                        await self.cache_manager.process_and_preserve(
                            content_hash=item.content_hash,
                            sentiment_score=sentiment_score,
                            confidence=confidence,
                            keywords=keywords
                        )
                    except Exception as e:
                        logger.error(f"Cache manager error: {e}")
                
                logger.debug(f"Processed {item.content_hash}: "
                           f"{label} ({sentiment_score:.3f}, {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to process item {item.content_hash}: {e}")
                item.status = ProcessingStatus.FAILED
                item.error_message = str(e)
                item.retry_count += 1
                
                # 재시도 가능한 아이템은 큐에 다시 추가
                if item.retry_count < 3:
                    self.batch_queue.append(item)
                else:
                    self.failed_items.append(item)
        
        batch_time = time.time() - start_time
        
        # 리소스 사용량 기록
        final_usage = self.resource_monitor.get_current_usage()
        peak_memory = max(initial_usage['memory_mb'], final_usage['memory_mb'])
        self.stats["memory_peaks"].append(peak_memory)
        
        # 최근 10개 피크만 유지
        if len(self.stats["memory_peaks"]) > 10:
            self.stats["memory_peaks"] = self.stats["memory_peaks"][-10:]
        
        # 동적 배치 크기 조정
        self._adjust_batch_size(batch_time, len(results), peak_memory)
        
        # 통계 업데이트
        self._update_stats(results, batch_time)
        
        # 처리 결과 저장
        self.processing_results.extend(results)
        
        logger.info(f"Batch {batch_id} completed: "
                   f"{len(results)} processed, {len(batch_items) - len(results)} failed "
                   f"in {batch_time:.2f}s. Memory peak: {peak_memory:.1f}MB")
        
        # 메모리 정리
        if peak_memory > 1500:  # 1.5GB 이상 사용 시
            gc.collect()
            logger.info("Garbage collection triggered due to high memory usage")
        
        return results
    
    def _adjust_batch_size(self, batch_time: float, items_processed: int, peak_memory: float):
        """동적 배치 크기 조정"""
        
        old_size = self.current_batch_size
        
        # 메모리 기반 조정
        if peak_memory > 2000:  # 2GB 이상
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 2)
        elif peak_memory < 1000 and batch_time < 60:  # 1GB 미만이고 처리 시간 짧음
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
        
        # 처리 시간 기반 조정
        if items_processed > 0:
            avg_time_per_item = batch_time / items_processed
            if avg_time_per_item > 5:  # 항목당 5초 이상
                self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
        
        if old_size != self.current_batch_size:
            self.stats["batch_size_adjustments"] += 1
            logger.info(f"Batch size adjusted: {old_size} -> {self.current_batch_size}")
    
    def _update_stats(self, results: List[BatchResult], batch_time: float):
        """통계 업데이트"""
        self.stats["total_processed"] += len(results)
        self.stats["total_batches"] += 1
        
        if results:
            avg_time = sum(r.processing_time for r in results) / len(results)
            current_avg = self.stats["avg_processing_time"]
            total_processed = self.stats["total_processed"]
            
            # 이동 평균 업데이트
            self.stats["avg_processing_time"] = (
                (current_avg * (total_processed - len(results)) + avg_time * len(results))
                / total_processed
            )
        
        self.stats["last_batch_time"] = datetime.now().isoformat()
        self.stats["items_per_batch_avg"] = (
            self.stats["total_processed"] / self.stats["total_batches"]
        )
    
    async def add_to_queue(self, 
                          content_hash: str,
                          title: str,
                          content: str,
                          source: str,
                          url: str,
                          published_at: datetime,
                          symbol: Optional[str] = None,
                          category: str = "general",
                          priority: int = 1) -> bool:
        """큐에 아이템 추가 (중복 체크 개선)"""
        
        # 중복 확인 (해시 세트 사용)
        existing_hashes = {item.content_hash for item in self.batch_queue}
        if content_hash in existing_hashes:
            logger.debug(f"Item already in queue: {content_hash}")
            return False
        
        batch_item = BatchItem(
            content_hash=content_hash,
            title=title,
            content=content,
            source=source,
            url=url,
            published_at=published_at,
            symbol=symbol,
            category=category,
            priority=priority
        )
        
        # 우선순위에 따라 큐 위치 결정
        if priority == 1:
            self.batch_queue.appendleft(batch_item)  # 높은 우선순위는 앞에
        else:
            self.batch_queue.append(batch_item)
        
        logger.debug(f"Added to batch queue: {content_hash} (queue size: {len(self.batch_queue)})")
        
        return True
    
    def _prepare_batch(self) -> List[BatchItem]:
        """다음 배치 준비 (최적화)"""
        
        if not self.batch_queue:
            return []
        
        # 현재 배치 크기만큼 아이템 추출
        batch_items = []
        for _ in range(min(len(self.batch_queue), self.current_batch_size)):
            batch_items.append(self.batch_queue.popleft())
        
        return batch_items
    
    async def run_batch_cycle(self) -> bool:
        """배치 사이클 실행"""
        
        try:
            # 배치 준비
            batch_items = self._prepare_batch()
            
            if not batch_items:
                logger.debug("No items in batch queue")
                return False
            
            # 배치 처리
            results = await self.process_batch(batch_items)
            
            if results:
                logger.info(f"Batch cycle completed: {len(results)} items processed")
                return True
            else:
                logger.warning("Batch cycle completed with no results")
                return False
                
        except Exception as e:
            logger.error(f"Batch cycle failed: {e}")
            self.stats["total_errors"] += 1
            return False
    
    async def start_processor(self):
        """배치 프로세서 시작"""
        
        if self.is_running:
            logger.warning("Batch processor already running")
            return
        
        if not self.model:
            if not await self.initialize_model():
                raise RuntimeError("Failed to initialize FinBERT model")
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._processor_loop())
        
        logger.info(f"FinBERT batch processor started (interval: {self.batch_interval}s)")
    
    async def stop_processor(self):
        """배치 프로세서 중지"""
        
        self.is_running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # 남은 항목 처리
        if self.batch_queue:
            logger.info(f"Processing remaining {len(self.batch_queue)} items before shutdown...")
            await self.run_batch_cycle()
        
        logger.info("FinBERT batch processor stopped")
    
    async def _processor_loop(self):
        """프로세서 메인 루프"""
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 15분 간격 체크 또는 큐가 가득 찬 경우
                should_process = (
                    (current_time - self.last_batch_time) >= self.batch_interval or
                    len(self.batch_queue) >= self.current_batch_size * 2  # 동적 임계값
                )
                
                if should_process:
                    success = await self.run_batch_cycle()
                    if success:
                        self.last_batch_time = current_time
                
                # 다음 체크까지 대기 (10초 간격)
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processor loop error: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """프로세서 통계 반환 (확장)"""
        
        current_usage = self.resource_monitor.get_current_usage()
        avg_usage = self.resource_monitor.get_average_usage()
        
        return {
            "is_running": self.is_running,
            "model_loaded": self.model is not None,
            "queue_size": len(self.batch_queue),
            "completed_items": len(self.processing_results),
            "failed_items": len(self.failed_items),
            "current_batch_size": self.current_batch_size,
            "batch_interval_seconds": self.batch_interval,
            "last_batch_time": self.last_batch_time,
            "next_batch_in_seconds": max(0, self.batch_interval - (time.time() - self.last_batch_time)),
            "resource_usage": {
                "current": current_usage,
                "average_60s": avg_usage,
                "memory_peaks": self.stats.get("memory_peaks", [])
            },
            "keyword_cache_size": len(self.keyword_cache),
            **self.stats
        }
    
    async def force_batch_run(self) -> bool:
        """강제 배치 실행 (테스트/디버깅용)"""
        
        logger.info("Forcing batch run...")
        return await self.run_batch_cycle()
    
    def clear_cache(self):
        """캐시 정리"""
        self.keyword_cache.clear()
        gc.collect()
        logger.info("Cache cleared")


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_finbert_processor():
        """FinBERT 배치 프로세서 테스트"""
        
        # 캐시 매니저 없이 테스트
        async with FinBERTBatchProcessorV2() as processor:
            print("=== FinBERT 배치 프로세서 V2 테스트 ===")
            
            # 테스트 아이템들 추가
            test_items = [
                {
                    "content_hash": "test1",
                    "title": "Bitcoin Surges to New All-Time High",
                    "content": "Bitcoin reached a new record as institutional adoption grows",
                    "source": "reuters",
                    "url": "https://reuters.com/test1",
                    "published_at": datetime.now(),
                    "symbol": "BTC"
                },
                {
                    "content_hash": "test2", 
                    "title": "Market Crash Fears Grip Investors",
                    "content": "Economic uncertainty leads to massive selloff across markets",
                    "source": "bloomberg",
                    "url": "https://bloomberg.com/test2",
                    "published_at": datetime.now(),
                    "symbol": "MARKET"
                }
            ]
            
            print(f"\n1. 아이템 큐에 추가...")
            for item in test_items:
                await processor.add_to_queue(**item)
            
            print(f"큐 크기: {len(processor.batch_queue)}")
            
            # 강제 배치 실행
            print(f"\n2. 배치 처리 실행...")
            success = await processor.force_batch_run()
            print(f"배치 처리 성공: {success}")
            
            # 결과 확인
            print(f"\n3. 처리 결과:")
            for result in list(processor.processing_results)[:5]:
                print(f"  {result.content_hash}: {result.label} "
                      f"({result.sentiment_score:.3f}, {result.confidence:.3f})")
                print(f"    키워드: {result.keywords}")
                print(f"    처리시간: {result.processing_time:.3f}s")
            
            # 통계
            print(f"\n4. 프로세서 통계:")
            stats = processor.get_processor_stats()
            for key, value in stats.items():
                if key != "resource_usage":
                    print(f"  {key}: {value}")
            
            print(f"\n5. 리소스 사용량:")
            resource_usage = stats.get("resource_usage", {})
            current = resource_usage.get("current", {})
            print(f"  CPU: {current.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {current.get('memory_mb', 0):.1f}MB")
    
    # 테스트 실행
    asyncio.run(test_finbert_processor())