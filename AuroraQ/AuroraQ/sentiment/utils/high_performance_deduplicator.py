#!/usr/bin/env python3
"""
High Performance News Deduplicator - MinHash + LSH 기반 고성능 중복 제거
대량 뉴스 처리를 위한 확장 가능한 중복 제거 시스템
"""

import hashlib
import time
import re
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class MinHashSignature:
    """MinHash 시그니처"""
    signature: np.ndarray
    hash_functions: int
    shingle_size: int
    
    def jaccard_similarity(self, other: 'MinHashSignature') -> float:
        """MinHash 기반 Jaccard 유사도 추정"""
        if len(self.signature) != len(other.signature):
            return 0.0
        return np.mean(self.signature == other.signature)

@dataclass
class LSHBucket:
    """LSH 버킷"""
    bucket_id: str
    news_ids: Set[str] = field(default_factory=set)
    signatures: Dict[str, MinHashSignature] = field(default_factory=dict)
    
    def add_item(self, news_id: str, signature: MinHashSignature):
        """아이템을 버킷에 추가"""
        self.news_ids.add(news_id)
        self.signatures[news_id] = signature

@dataclass
class DuplicationCandidate:
    """중복 후보"""
    news_id_1: str
    news_id_2: str
    similarity_score: float
    comparison_method: str
    
    def __eq__(self, other):
        if not isinstance(other, DuplicationCandidate):
            return False
        return (self.news_id_1 == other.news_id_1 and self.news_id_2 == other.news_id_2) or \
               (self.news_id_1 == other.news_id_2 and self.news_id_2 == other.news_id_1)
    
    def __hash__(self):
        # 순서에 상관없이 같은 해시 생성
        ids = tuple(sorted([self.news_id_1, self.news_id_2]))
        return hash(ids)

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    total_items: int = 0
    processing_time: float = 0.0
    minhash_time: float = 0.0
    lsh_time: float = 0.0
    similarity_time: float = 0.0
    clustering_time: float = 0.0
    
    candidates_generated: int = 0
    comparisons_made: int = 0
    duplicates_found: int = 0
    
    @property
    def items_per_second(self) -> float:
        return self.total_items / self.processing_time if self.processing_time > 0 else 0
    
    @property
    def comparison_efficiency(self) -> float:
        """비교 효율성 (전체 가능한 비교 대비 실제 비교)"""
        total_possible = (self.total_items * (self.total_items - 1)) // 2
        return self.comparisons_made / total_possible if total_possible > 0 else 0

class HighPerformanceDeduplicator:
    """고성능 뉴스 중복 제거 시스템"""
    
    def __init__(self,
                 similarity_threshold: float = 0.8,
                 minhash_num_perm: int = 128,
                 shingle_size: int = 3,
                 lsh_bands: int = 16,
                 lsh_rows: int = 8,
                 title_clustering_enabled: bool = True,
                 max_cache_size: int = 10000):
        """
        초기화
        
        Args:
            similarity_threshold: 중복 판정 임계값
            minhash_num_perm: MinHash 순열 개수 (높을수록 정확, 느림)
            shingle_size: Shingle 크기 (3-5 권장)
            lsh_bands: LSH 밴드 수
            lsh_rows: LSH 밴드당 행 수
            title_clustering_enabled: 제목 기반 사전 클러스터링 활성화
            max_cache_size: 최대 캐시 크기
        """
        self.similarity_threshold = similarity_threshold
        self.minhash_num_perm = minhash_num_perm
        self.shingle_size = shingle_size
        self.lsh_bands = lsh_bands
        self.lsh_rows = lsh_rows
        self.title_clustering_enabled = title_clustering_enabled
        self.max_cache_size = max_cache_size
        
        # LSH 파라미터 검증
        if lsh_bands * lsh_rows != minhash_num_perm:
            raise ValueError(f"lsh_bands * lsh_rows ({lsh_bands * lsh_rows}) must equal minhash_num_perm ({minhash_num_perm})")
        
        # MinHash 해시 함수들 (미리 생성)
        self.hash_functions = self._generate_hash_functions()
        
        # LSH 버킷들
        self.lsh_buckets: Dict[str, LSHBucket] = {}
        
        # 캐시
        self.signature_cache: Dict[str, MinHashSignature] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # 성능 통계
        self.performance_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "last_cleanup": datetime.now()
        }
    
    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """MinHash용 해시 함수들 생성"""
        # 큰 소수
        large_prime = 2**31 - 1
        
        hash_functions = []
        np.random.seed(42)  # 재현 가능한 결과를 위해
        
        for _ in range(self.minhash_num_perm):
            a = np.random.randint(1, large_prime)
            b = np.random.randint(0, large_prime)
            hash_functions.append((a, b))
        
        return hash_functions
    
    def deduplicate_news_batch_optimized(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        최적화된 배치 중복 제거
        
        Args:
            news_items: 뉴스 아이템 리스트
            
        Returns:
            Dict: 중복 제거 결과 및 성능 메트릭
        """
        start_time = time.time()
        metrics = PerformanceMetrics(total_items=len(news_items))
        
        logger.info(f"Starting optimized deduplication for {len(news_items)} items")
        
        try:
            # 1단계: 제목 기반 사전 클러스터링 (옵션)
            if self.title_clustering_enabled:
                clustered_items = self._pre_cluster_by_title(news_items)
                logger.debug(f"Pre-clustering created {len(clustered_items)} clusters")
            else:
                clustered_items = [news_items]  # 모든 아이템을 하나의 클러스터로
            
            # 2단계: 각 클러스터별로 MinHash 기반 중복 제거
            all_unique_items = []
            all_duplicates = []
            
            for cluster in clustered_items:
                if len(cluster) <= 1:
                    all_unique_items.extend(cluster)
                    continue
                
                unique_items, duplicates, cluster_metrics = self._process_cluster_optimized(cluster)
                all_unique_items.extend(unique_items)
                all_duplicates.extend(duplicates)
                
                # 메트릭 누적
                metrics.minhash_time += cluster_metrics.minhash_time
                metrics.lsh_time += cluster_metrics.lsh_time
                metrics.similarity_time += cluster_metrics.similarity_time
                metrics.candidates_generated += cluster_metrics.candidates_generated
                metrics.comparisons_made += cluster_metrics.comparisons_made
                metrics.duplicates_found += cluster_metrics.duplicates_found
            
            metrics.processing_time = time.time() - start_time
            
            # 결과 생성
            result = {
                "original_count": len(news_items),
                "deduplicated_count": len(all_unique_items),
                "duplicates_found": len(all_duplicates),
                "deduplication_rate": len(all_duplicates) / len(news_items) if news_items else 0,
                "unique_items": all_unique_items,
                "duplicate_pairs": all_duplicates,
                "performance_metrics": {
                    "total_items": metrics.total_items,
                    "processing_time": metrics.processing_time,
                    "minhash_time": metrics.minhash_time,
                    "lsh_time": metrics.lsh_time,
                    "similarity_time": metrics.similarity_time,
                    "clustering_time": metrics.clustering_time,
                    "candidates_generated": metrics.candidates_generated,
                    "comparisons_made": metrics.comparisons_made,
                    "duplicates_found": metrics.duplicates_found,
                    "items_per_second": metrics.items_per_second,
                    "comparison_efficiency": metrics.comparison_efficiency
                },
                "processing_method": "minhash_lsh_optimized"
            }
            
            # 성능 통계 업데이트
            self._update_performance_stats(metrics)
            
            deduplication_rate = len(all_duplicates) / len(news_items) if news_items else 0
            logger.info(f"Optimized deduplication completed: {len(news_items)} -> {len(all_unique_items)} "
                       f"({deduplication_rate:.1%} duplicates removed) in {metrics.processing_time:.3f}s "
                       f"({metrics.items_per_second:.1f} items/sec)")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized deduplication failed: {e}")
            return {
                "original_count": len(news_items),
                "deduplicated_count": len(news_items),
                "duplicates_found": 0,
                "deduplication_rate": 0.0,
                "unique_items": news_items,
                "duplicate_pairs": [],
                "performance_metrics": {
                    "total_items": len(news_items),
                    "processing_time": time.time() - start_time,
                    "minhash_time": 0.0,
                    "lsh_time": 0.0,
                    "similarity_time": 0.0,
                    "clustering_time": 0.0,
                    "candidates_generated": 0,
                    "comparisons_made": 0,
                    "duplicates_found": 0,
                    "items_per_second": 0.0,
                    "comparison_efficiency": 0.0
                },
                "error": str(e)
            }
    
    def _pre_cluster_by_title(self, news_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """제목 기반 사전 클러스터링"""
        clusters = defaultdict(list)
        
        for item in news_items:
            title = item.get('title', '')
            if not title:
                clusters['__no_title__'].append(item)
                continue
            
            # 제목 정규화 및 해시
            normalized_title = self._normalize_title(title)
            title_hash = self._title_hash(normalized_title)
            
            clusters[title_hash].append(item)
        
        # 클러스터를 리스트로 변환
        return list(clusters.values())
    
    def _normalize_title(self, title: str) -> str:
        """제목 정규화"""
        # 소문자 변환
        title = title.lower()
        
        # 특수문자 및 숫자 제거 (핵심 단어만 유지)
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\d+', '', title)
        
        # 불용어 제거 (간단한 버전)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words = [word for word in title.split() if word not in stop_words and len(word) > 2]
        
        # 알파벳 순으로 정렬 (순서 무관하게)
        words.sort()
        
        return ' '.join(words[:10])  # 최대 10개 단어만 사용
    
    def _title_hash(self, normalized_title: str) -> str:
        """제목 해시 생성"""
        if not normalized_title.strip():
            return "__empty_title__"
        
        # 3-gram 기반 해시 (더 유연한 매칭)
        words = normalized_title.split()
        if len(words) <= 3:
            return normalized_title
        
        # 3-gram 생성
        trigrams = []
        for i in range(len(words) - 2):
            trigram = ' '.join(words[i:i+3])
            trigrams.append(trigram)
        
        # 가장 빈번한 3-gram 사용
        if trigrams:
            trigram_counts = Counter(trigrams)
            most_common = trigram_counts.most_common(1)[0][0]
            return hashlib.md5(most_common.encode()).hexdigest()[:8]
        
        return hashlib.md5(normalized_title.encode()).hexdigest()[:8]
    
    def _process_cluster_optimized(self, cluster_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[DuplicationCandidate], PerformanceMetrics]:
        """클러스터 최적화 처리"""
        metrics = PerformanceMetrics(total_items=len(cluster_items))
        
        # 1. MinHash 시그니처 생성
        minhash_start = time.time()
        signatures = {}
        
        for item in cluster_items:
            item_id = item.get('hash_id', item.get('url', ''))
            if not item_id:
                continue
            
            # 캐시 확인
            if item_id in self.signature_cache:
                signatures[item_id] = self.signature_cache[item_id]
                self.performance_stats["cache_hits"] += 1
            else:
                signature = self._create_minhash_signature(item)
                signatures[item_id] = signature
                
                # 캐시에 저장 (크기 제한)
                if len(self.signature_cache) < self.max_cache_size:
                    self.signature_cache[item_id] = signature
                
                self.performance_stats["cache_misses"] += 1
        
        metrics.minhash_time = time.time() - minhash_start
        
        # 2. LSH를 사용한 후보 쌍 찾기
        lsh_start = time.time()
        candidate_pairs = self._find_candidate_pairs_lsh(signatures)
        metrics.lsh_time = time.time() - lsh_start
        metrics.candidates_generated = len(candidate_pairs)
        
        # 3. 정확한 유사도 계산
        similarity_start = time.time()
        duplicate_candidates = []
        
        for item_id_1, item_id_2 in candidate_pairs:
            # 캐시 확인
            cache_key = tuple(sorted([item_id_1, item_id_2]))
            
            if cache_key in self.similarity_cache:
                similarity = self.similarity_cache[cache_key]
                self.performance_stats["cache_hits"] += 1
            else:
                similarity = signatures[item_id_1].jaccard_similarity(signatures[item_id_2])
                
                # 캐시에 저장
                if len(self.similarity_cache) < self.max_cache_size:
                    self.similarity_cache[cache_key] = similarity
                
                self.performance_stats["cache_misses"] += 1
            
            metrics.comparisons_made += 1
            
            if similarity >= self.similarity_threshold:
                duplicate_candidates.append(DuplicationCandidate(
                    news_id_1=item_id_1,
                    news_id_2=item_id_2,
                    similarity_score=similarity,
                    comparison_method="minhash"
                ))
        
        metrics.similarity_time = time.time() - similarity_start
        metrics.duplicates_found = len(duplicate_candidates)
        
        # 4. 중복 제거된 아이템 선택
        unique_items = self._select_unique_items(cluster_items, duplicate_candidates)
        
        return unique_items, duplicate_candidates, metrics
    
    def _create_minhash_signature(self, news_item: Dict[str, Any]) -> MinHashSignature:
        """MinHash 시그니처 생성"""
        # 텍스트 추출 및 정규화
        title = news_item.get('title', '')
        content = news_item.get('content', '')
        text = f"{title} {content}".lower()
        
        # Shingle 생성
        shingles = self._create_shingles(text, self.shingle_size)
        if not shingles:
            # 빈 텍스트인 경우 기본 시그니처
            return MinHashSignature(
                signature=np.full(self.minhash_num_perm, np.inf),
                hash_functions=self.minhash_num_perm,
                shingle_size=self.shingle_size
            )
        
        # Shingle을 해시값으로 변환
        shingle_hashes = {self._hash_shingle(shingle) for shingle in shingles}
        
        # MinHash 계산
        signature = np.full(self.minhash_num_perm, np.inf)
        
        for i, (a, b) in enumerate(self.hash_functions):
            min_hash_value = min(
                (a * shingle_hash + b) % (2**31 - 1)
                for shingle_hash in shingle_hashes
            )
            signature[i] = min_hash_value
        
        return MinHashSignature(
            signature=signature,
            hash_functions=self.minhash_num_perm,
            shingle_size=self.shingle_size
        )
    
    def _create_shingles(self, text: str, k: int) -> Set[str]:
        """k-shingle 생성"""
        if not text or len(text) < k:
            return {text} if text else set()
        
        # 문자 기반 shingle
        shingles = set()
        for i in range(len(text) - k + 1):
            shingle = text[i:i+k]
            shingles.add(shingle)
        
        return shingles
    
    def _hash_shingle(self, shingle: str) -> int:
        """Shingle을 해시값으로 변환"""
        return hash(shingle) % (2**31 - 1)
    
    def _find_candidate_pairs_lsh(self, signatures: Dict[str, MinHashSignature]) -> Set[Tuple[str, str]]:
        """LSH를 사용한 후보 쌍 찾기"""
        candidate_pairs = set()
        
        # LSH 버킷 초기화
        lsh_buckets = defaultdict(set)
        
        # 각 아이템을 LSH 버킷에 할당
        for item_id, signature in signatures.items():
            # 각 밴드에 대해 해시값 계산
            for band in range(self.lsh_bands):
                start_idx = band * self.lsh_rows
                end_idx = start_idx + self.lsh_rows
                
                # 밴드의 시그니처 부분을 해시
                band_signature = tuple(signature.signature[start_idx:end_idx])
                bucket_key = f"band_{band}_{hash(band_signature)}"
                
                lsh_buckets[bucket_key].add(item_id)
        
        # 같은 버킷에 있는 아이템들을 후보 쌍으로 추가
        for bucket_items in lsh_buckets.values():
            if len(bucket_items) > 1:
                bucket_list = list(bucket_items)
                for i in range(len(bucket_list)):
                    for j in range(i + 1, len(bucket_list)):
                        pair = tuple(sorted([bucket_list[i], bucket_list[j]]))
                        candidate_pairs.add(pair)
        
        return candidate_pairs
    
    def _select_unique_items(self, cluster_items: List[Dict[str, Any]], 
                           duplicate_candidates: List[DuplicationCandidate]) -> List[Dict[str, Any]]:
        """중복 제거된 고유 아이템 선택"""
        # 아이템 ID -> 아이템 매핑
        item_map = {}
        for item in cluster_items:
            item_id = item.get('hash_id', item.get('url', ''))
            if item_id:
                item_map[item_id] = item
        
        # 중복 그래프 구성
        duplicate_graph = defaultdict(set)
        for candidate in duplicate_candidates:
            duplicate_graph[candidate.news_id_1].add(candidate.news_id_2)
            duplicate_graph[candidate.news_id_2].add(candidate.news_id_1)
        
        # 중복 그룹 찾기 (연결된 컴포넌트)
        visited = set()
        unique_items = []
        
        for item_id in item_map.keys():
            if item_id in visited:
                continue
            
            # DFS로 연결된 중복 그룹 찾기
            duplicate_group = self._find_duplicate_group(item_id, duplicate_graph, visited)
            
            if duplicate_group:
                # 그룹에서 가장 좋은 아이템 선택
                representative = self._select_best_item(duplicate_group, item_map)
                if representative in item_map:
                    unique_items.append(item_map[representative])
        
        return unique_items
    
    def _find_duplicate_group(self, start_id: str, duplicate_graph: Dict[str, Set[str]], 
                             visited: Set[str]) -> Set[str]:
        """DFS로 중복 그룹 찾기"""
        if start_id in visited:
            return set()
        
        group = set()
        stack = [start_id]
        
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            
            visited.add(current_id)
            group.add(current_id)
            
            # 연결된 노드들 추가
            for neighbor in duplicate_graph.get(current_id, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return group
    
    def _select_best_item(self, duplicate_group: Set[str], item_map: Dict[str, Dict[str, Any]]) -> str:
        """중복 그룹에서 가장 좋은 아이템 선택"""
        if len(duplicate_group) == 1:
            return list(duplicate_group)[0]
        
        # 선택 기준: 1) 내용 길이, 2) 소스 품질, 3) 발행 시간
        best_item_id = None
        best_score = -1
        
        for item_id in duplicate_group:
            if item_id not in item_map:
                continue
            
            item = item_map[item_id]
            
            # 점수 계산
            score = 0
            
            # 내용 길이 (30%)
            content_length = len(item.get('content', ''))
            score += min(1.0, content_length / 1000) * 0.3
            
            # 소스 품질 (40%)
            source_url = item.get('url', '').lower()
            if 'reuters.com' in source_url or 'bloomberg.com' in source_url:
                score += 0.4
            elif 'cnbc.com' in source_url or 'wsj.com' in source_url:
                score += 0.3
            elif 'yahoo.com' in source_url or 'finnhub.io' in source_url:
                score += 0.2
            else:
                score += 0.1
            
            # 제목 품질 (30%)
            title_length = len(item.get('title', ''))
            if 30 <= title_length <= 100:  # 적절한 길이
                score += 0.3
            elif title_length > 0:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_item_id = item_id
        
        return best_item_id or list(duplicate_group)[0]
    
    def _update_performance_stats(self, metrics: PerformanceMetrics):
        """성능 통계 업데이트"""
        self.performance_stats["total_processed"] += metrics.total_items
        
        # 평균 처리 시간 업데이트 (지수 이동 평균)
        alpha = 0.1
        if self.performance_stats["avg_processing_time"] == 0:
            self.performance_stats["avg_processing_time"] = metrics.processing_time
        else:
            self.performance_stats["avg_processing_time"] = (
                alpha * metrics.processing_time + 
                (1 - alpha) * self.performance_stats["avg_processing_time"]
            )
    
    def cleanup_cache(self, max_age_minutes: int = 60):
        """캐시 정리"""
        # 간단한 LRU: 캐시 크기가 한계를 넘으면 절반 정리
        if len(self.signature_cache) > self.max_cache_size:
            # 오래된 항목부터 제거 (간단한 구현)
            items_to_remove = len(self.signature_cache) - (self.max_cache_size // 2)
            keys_to_remove = list(self.signature_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.signature_cache[key]
        
        if len(self.similarity_cache) > self.max_cache_size:
            items_to_remove = len(self.similarity_cache) - (self.max_cache_size // 2)
            keys_to_remove = list(self.similarity_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        self.performance_stats["last_cleanup"] = datetime.now()
        logger.debug(f"Cache cleanup completed. "
                    f"Signature cache: {len(self.signature_cache)}, "
                    f"Similarity cache: {len(self.similarity_cache)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        cache_hit_rate = 0.0
        total_cache_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.performance_stats["cache_hits"] / total_cache_requests
        
        return {
            **self.performance_stats,
            "cache_hit_rate": cache_hit_rate,
            "signature_cache_size": len(self.signature_cache),
            "similarity_cache_size": len(self.similarity_cache),
            "configuration": {
                "similarity_threshold": self.similarity_threshold,
                "minhash_num_perm": self.minhash_num_perm,
                "shingle_size": self.shingle_size,
                "lsh_bands": self.lsh_bands,
                "lsh_rows": self.lsh_rows,
                "title_clustering_enabled": self.title_clustering_enabled
            }
        }

# 글로벌 인스턴스 (싱글톤 패턴)
_global_hp_deduplicator: Optional[HighPerformanceDeduplicator] = None

def get_high_performance_deduplicator() -> HighPerformanceDeduplicator:
    """고성능 중복 제거 시스템 인스턴스 반환"""
    global _global_hp_deduplicator
    if _global_hp_deduplicator is None:
        _global_hp_deduplicator = HighPerformanceDeduplicator()
    return _global_hp_deduplicator

def deduplicate_news_optimized(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """뉴스 아이템 최적화 중복 제거 (간편 함수)"""
    deduplicator = get_high_performance_deduplicator()
    result = deduplicator.deduplicate_news_batch_optimized(news_items)
    return result.get("unique_items", news_items)

if __name__ == "__main__":
    # 성능 테스트 및 데모
    import random
    import string
    
    def generate_test_news(count: int, duplicate_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """테스트용 뉴스 생성"""
        news_items = []
        base_titles = [
            "Federal Reserve Raises Interest Rates",
            "Bitcoin Price Surges Above $30,000",
            "SEC Approves Bitcoin ETF",
            "Tesla Reports Strong Quarterly Earnings",
            "Apple Announces New iPhone"
        ]
        
        # 기본 뉴스 생성
        for i in range(int(count * (1 - duplicate_ratio))):
            title = random.choice(base_titles)
            content = ''.join(random.choices(string.ascii_letters + ' ', k=500))
            
            news_items.append({
                "title": f"{title} - {i}",
                "content": content,
                "url": f"https://example.com/news/{i}",
                "hash_id": f"news_{i}"
            })
        
        # 중복 뉴스 생성 (기존 뉴스의 변형)
        duplicate_count = int(count * duplicate_ratio)
        for i in range(duplicate_count):
            base_news = random.choice(news_items)
            # 제목 약간 변경
            modified_title = base_news["title"].replace("-", ":")
            # 내용 일부 변경
            modified_content = base_news["content"][:400] + " additional content"
            
            news_items.append({
                "title": modified_title,
                "content": modified_content,
                "url": f"https://example.com/news/dup_{i}",
                "hash_id": f"news_dup_{i}"
            })
        
        return news_items
    
    # 성능 테스트
    print("=== High Performance Deduplicator Test ===\n")
    
    # 다양한 크기로 테스트
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"Testing with {size} items...")
        
        # 테스트 데이터 생성
        test_data = generate_test_news(size, duplicate_ratio=0.2)
        
        # 고성능 중복 제거 테스트
        hp_deduplicator = HighPerformanceDeduplicator()
        
        start_time = time.time()
        result = hp_deduplicator.deduplicate_news_batch_optimized(test_data)
        processing_time = time.time() - start_time
        
        print(f"  Results: {result['original_count']} -> {result['deduplicated_count']} "
              f"({result['deduplication_rate']:.1%} duplicates)")
        print(f"  Processing time: {processing_time:.3f}s ({len(test_data)/processing_time:.1f} items/sec)")
        
        metrics = result['performance_metrics']
        print(f"  Efficiency: {metrics['comparison_efficiency']:.1%} comparison efficiency")
        print(f"  Candidates: {metrics['candidates_generated']} candidates generated")
        print(f"  Comparisons: {metrics['comparisons_made']} actual comparisons")
        print()
    
    # 성능 통계 출력
    stats = hp_deduplicator.get_performance_stats()
    print("=== Performance Statistics ===")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"Configuration: {stats['configuration']}")
    print(f"Total processed: {stats['total_processed']} items")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")