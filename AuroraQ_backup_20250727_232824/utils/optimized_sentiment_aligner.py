import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import bisect
from functools import lru_cache
import threading


class OptimizedSentimentAligner:
    """
    최적화된 감정 점수 정렬 시스템
    - 이진 탐색 기반 빠른 시간 매칭
    - 효율적인 보간 및 캐싱
    - 메모리 효율적인 슬라이딩 윈도우
    """
    
    def __init__(self, window_size: int = 10000, cache_size: int = 1000):
        self.window_size = window_size
        self.cache_size = cache_size
        
        # 정렬된 타임스탬프와 감정 점수 저장
        self.timestamps = []
        self.sentiment_scores = []
        
        # 캐시
        self.alignment_cache = {}
        self.interpolation_cache = {}
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 통계
        self.stats = {
            "alignments": 0,
            "cache_hits": 0,
            "interpolations": 0
        }
    
    def update_sentiment_data(self, sentiment_df: pd.DataFrame):
        """감정 데이터 업데이트 (효율적인 메모리 관리)"""
        with self.lock:
            # 타임스탬프 순으로 정렬
            sorted_df = sentiment_df.sort_values('timestamp')
            
            # 윈도우 크기 제한
            if len(sorted_df) > self.window_size:
                sorted_df = sorted_df.tail(self.window_size)
            
            # numpy 배열로 저장 (메모리 효율성)
            self.timestamps = sorted_df['timestamp'].values.astype('datetime64[ns]')
            self.sentiment_scores = sorted_df['sentiment_score'].values.astype('float32')
            
            # 캐시 초기화
            self.alignment_cache.clear()
            self.interpolation_cache.clear()
    
    def align_single(self, timestamp: pd.Timestamp) -> float:
        """단일 타임스탬프에 대한 감정 점수 조회"""
        with self.lock:
            if len(self.timestamps) == 0:
                return 0.5
            
            # 캐시 확인
            cache_key = timestamp.value // 10**9  # 초 단위로 캐싱
            if cache_key in self.alignment_cache:
                self.stats["cache_hits"] += 1
                return self.alignment_cache[cache_key]
            
            self.stats["alignments"] += 1
            
            # numpy datetime64로 변환
            target_time = np.datetime64(timestamp)
            
            # 이진 탐색으로 가장 가까운 인덱스 찾기
            idx = self._binary_search_nearest(target_time)
            
            # 보간이 필요한 경우
            if idx > 0 and idx < len(self.timestamps) - 1:
                score = self._interpolate(idx, target_time)
            else:
                # 가장 가까운 값 사용
                score = float(self.sentiment_scores[idx])
            
            # 캐시에 저장
            self.alignment_cache[cache_key] = score
            
            # 캐시 크기 관리
            if len(self.alignment_cache) > self.cache_size:
                # 가장 오래된 항목 제거
                oldest_key = min(self.alignment_cache.keys())
                del self.alignment_cache[oldest_key]
            
            return score
    
    def align_batch(self, timestamps: pd.Series) -> pd.Series:
        """배치 타임스탬프 정렬 (벡터화 최적화)"""
        if len(self.timestamps) == 0:
            return pd.Series(0.5, index=timestamps.index)
        
        with self.lock:
            # numpy 배열로 변환
            target_times = pd.to_datetime(timestamps).values.astype('datetime64[ns]')
            
            # 벡터화된 검색 및 보간
            scores = np.empty(len(target_times), dtype='float32')
            
            for i, target_time in enumerate(target_times):
                scores[i] = self.align_single(pd.Timestamp(target_time))
            
            return pd.Series(scores, index=timestamps.index)
    
    def _binary_search_nearest(self, target: np.datetime64) -> int:
        """이진 탐색으로 가장 가까운 인덱스 찾기"""
        # 타임스탬프를 정수로 변환 (나노초)
        target_val = target.astype('int64')
        timestamp_vals = self.timestamps.astype('int64')
        
        # bisect를 사용한 이진 탐색
        idx = bisect.bisect_left(timestamp_vals, target_val)
        
        # 경계 처리
        if idx == 0:
            return 0
        if idx == len(self.timestamps):
            return len(self.timestamps) - 1
        
        # 더 가까운 인덱스 선택
        before_diff = abs(timestamp_vals[idx - 1] - target_val)
        after_diff = abs(timestamp_vals[idx] - target_val)
        
        return idx - 1 if before_diff < after_diff else idx
    
    def _interpolate(self, idx: int, target: np.datetime64) -> float:
        """선형 보간"""
        self.stats["interpolations"] += 1
        
        # 인접한 두 점 사이에서 보간
        t1 = self.timestamps[idx].astype('int64')
        t2 = self.timestamps[idx + 1].astype('int64')
        target_val = target.astype('int64')
        
        # 시간 비율 계산
        ratio = (target_val - t1) / (t2 - t1) if t2 != t1 else 0
        
        # 선형 보간
        s1 = self.sentiment_scores[idx]
        s2 = self.sentiment_scores[idx + 1]
        
        return float(s1 + ratio * (s2 - s1))
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        with self.lock:
            total_requests = self.stats["alignments"]
            cache_hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "data_points": len(self.timestamps),
                "cache_size": len(self.alignment_cache),
                "total_alignments": self.stats["alignments"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": cache_hit_rate,
                "interpolations": self.stats["interpolations"]
            }
    
    def optimize_for_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """특정 시간 범위에 대해 최적화 (프리페치)"""
        with self.lock:
            # 해당 범위의 데이터만 메모리에 유지
            start_idx = self._binary_search_nearest(np.datetime64(start_time))
            end_idx = self._binary_search_nearest(np.datetime64(end_time))
            
            # 안전한 마진 추가
            start_idx = max(0, start_idx - 100)
            end_idx = min(len(self.timestamps) - 1, end_idx + 100)
            
            # 범위 내 데이터만 유지
            self.timestamps = self.timestamps[start_idx:end_idx + 1]
            self.sentiment_scores = self.sentiment_scores[start_idx:end_idx + 1]
            
            # 캐시 초기화
            self.alignment_cache.clear()


# 전역 인스턴스
_sentiment_aligner = OptimizedSentimentAligner()


def get_sentiment_aligner() -> OptimizedSentimentAligner:
    """전역 감정 점수 정렬기 반환"""
    return _sentiment_aligner