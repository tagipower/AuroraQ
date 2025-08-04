#!/usr/bin/env python3
"""
News Deduplicator - 유사도 기반 뉴스 중복 제거 시스템
Hash ID 기반 + 유사도 분석 병합 구조
"""

import re
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# 텍스트 유사도 계산을 위한 간단한 구현
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class NewsCluster:
    """유사한 뉴스들의 클러스터"""
    representative_id: str              # 대표 뉴스 ID
    member_ids: Set[str] = field(default_factory=set)  # 클러스터 멤버들
    similarity_threshold: float = 0.8   # 유사도 임계값
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_member(self, news_id: str):
        """클러스터에 멤버 추가"""
        self.member_ids.add(news_id)
    
    def get_size(self) -> int:
        """클러스터 크기 반환"""
        return len(self.member_ids)

@dataclass
class DeduplicationResult:
    """중복 제거 결과"""
    original_count: int
    deduplicated_count: int
    duplicate_clusters: Dict[str, NewsCluster]
    removed_news_ids: Set[str]
    processing_time: float
    deduplicated_items: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def deduplication_rate(self) -> float:
        """중복 제거율"""
        if self.original_count == 0:
            return 0.0
        return (self.original_count - self.deduplicated_count) / self.original_count

class NewsDeduplicator:
    """뉴스 중복 제거 시스템"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 min_content_length: int = 50,
                 max_clusters: int = 1000):
        """
        초기화
        
        Args:
            similarity_threshold: 유사도 임계값 (0.0-1.0)
            min_content_length: 최소 내용 길이 (짧은 내용 필터링)
            max_clusters: 최대 클러스터 수 (메모리 관리)
        """
        self.similarity_threshold = similarity_threshold
        self.min_content_length = min_content_length
        self.max_clusters = max_clusters
        
        # 클러스터 저장소
        self.news_clusters: Dict[str, NewsCluster] = {}
        self.news_to_cluster: Dict[str, str] = {}  # 뉴스 ID -> 클러스터 ID 매핑
        
        # 캐시된 뉴스 특징 (메모리 효율성을 위해 제한적으로 저장)
        self.news_features: Dict[str, Dict[str, Any]] = {}
        self.feature_cache_limit = 5000
        
        # 통계
        self.stats = {
            "total_processed": 0,
            "total_duplicates_found": 0,
            "total_clusters_created": 0,
            "avg_cluster_size": 0.0,
            "processing_time_ms": 0.0,
            "last_cleanup": datetime.now()
        }
    
    def deduplicate_news_batch(self, news_items: List[Dict[str, Any]]) -> DeduplicationResult:
        """
        뉴스 배치 중복 제거
        
        Args:
            news_items: 뉴스 아이템 리스트 (title, content, url, hash_id 포함)
            
        Returns:
            DeduplicationResult: 중복 제거 결과
        """
        start_time = time.time()
        original_count = len(news_items)
        
        logger.info(f"Starting deduplication for {original_count} news items")
        
        try:
            # 1단계: Hash ID 기반 중복 제거
            hash_deduplicated = self._deduplicate_by_hash(news_items)
            
            # 2단계: 유사도 기반 중복 제거
            similarity_deduplicated, clusters, removed_ids = self._deduplicate_by_similarity(hash_deduplicated)
            
            processing_time = time.time() - start_time
            
            # 결과 생성
            result = DeduplicationResult(
                original_count=original_count,
                deduplicated_count=len(similarity_deduplicated),
                duplicate_clusters=clusters,
                removed_news_ids=removed_ids,
                processing_time=processing_time,
                deduplicated_items=similarity_deduplicated
            )
            
            # 통계 업데이트
            self._update_stats(result)
            
            logger.info(f"Deduplication completed: {original_count} -> {len(similarity_deduplicated)} "
                       f"({result.deduplication_rate:.1%} removed) in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during deduplication: {e}")
            # 에러 시 원본 반환
            return DeduplicationResult(
                original_count=original_count,
                deduplicated_count=original_count,
                duplicate_clusters={},
                removed_news_ids=set(),
                processing_time=time.time() - start_time,
                deduplicated_items=news_items
            )
    
    def _deduplicate_by_hash(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hash ID 기반 중복 제거 (1단계)"""
        seen_hashes = set()
        deduplicated = []
        
        for item in news_items:
            hash_id = item.get('hash_id')
            if not hash_id:
                # hash_id가 없으면 생성
                content_for_hash = f"{item.get('title', '')}{item.get('url', '')}"
                hash_id = hashlib.md5(content_for_hash.encode()).hexdigest()
                item['hash_id'] = hash_id
            
            if hash_id not in seen_hashes:
                seen_hashes.add(hash_id)
                deduplicated.append(item)
        
        logger.debug(f"Hash deduplication: {len(news_items)} -> {len(deduplicated)}")
        return deduplicated
    
    def _deduplicate_by_similarity(self, news_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, NewsCluster], Set[str]]:
        """유사도 기반 중복 제거 (2단계)"""
        if len(news_items) <= 1:
            return news_items, {}, set()
        
        # 뉴스 특징 추출
        news_features = {}
        for item in news_items:
            news_id = item.get('hash_id', item.get('url', ''))
            if news_id:
                features = self._extract_news_features(item)
                news_features[news_id] = {
                    'features': features,
                    'item': item
                }
        
        # 유사도 기반 클러스터링
        clusters = {}
        news_to_cluster = {}
        removed_ids = set()
        
        news_ids = list(news_features.keys())
        
        for i, news_id_1 in enumerate(news_ids):
            if news_id_1 in removed_ids:
                continue
            
            # 이미 클러스터에 속해있는지 확인
            if news_id_1 in news_to_cluster:
                continue
            
            # 새 클러스터 생성
            cluster_id = f"cluster_{len(clusters)}"
            cluster = NewsCluster(
                representative_id=news_id_1,
                similarity_threshold=self.similarity_threshold
            )
            cluster.add_member(news_id_1)
            
            # 유사한 뉴스 찾기
            features_1 = news_features[news_id_1]['features']
            
            for j in range(i + 1, len(news_ids)):
                news_id_2 = news_ids[j]
                if news_id_2 in removed_ids or news_id_2 in news_to_cluster:
                    continue
                
                features_2 = news_features[news_id_2]['features']
                similarity = self._calculate_similarity(features_1, features_2)
                
                if similarity >= self.similarity_threshold:
                    cluster.add_member(news_id_2)
                    news_to_cluster[news_id_2] = cluster_id
                    removed_ids.add(news_id_2)
            
            # 클러스터 저장 (크기가 1보다 큰 경우만)
            if cluster.get_size() > 1:
                clusters[cluster_id] = cluster
                news_to_cluster[news_id_1] = cluster_id
        
        # 중복되지 않은 뉴스만 반환
        deduplicated_items = []
        for news_id, data in news_features.items():
            if news_id not in removed_ids:
                deduplicated_items.append(data['item'])
        
        logger.debug(f"Similarity deduplication: {len(news_items)} -> {len(deduplicated_items)}, "
                    f"{len(clusters)} clusters created")
        
        return deduplicated_items, clusters, removed_ids
    
    def _extract_news_features(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """뉴스 특징 추출"""
        title = news_item.get('title', '')
        content = news_item.get('content', '')
        
        # 텍스트 정규화
        normalized_title = self._normalize_text(title)
        normalized_content = self._normalize_text(content)
        
        # 특징 추출
        features = {
            'title_words': set(normalized_title.split()),
            'content_words': set(normalized_content.split()),
            'title_length': len(title),
            'content_length': len(content),
            'title_hash': hashlib.md5(normalized_title.encode()).hexdigest(),
            'key_phrases': self._extract_key_phrases(f"{title} {content}"),
            'numbers': self._extract_numbers(f"{title} {content}"),
            'entities': self._extract_simple_entities(f"{title} {content}")
        }
        
        return features
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        if not text:
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # 특수문자 제거 (일부 유지)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        return text.strip()
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> Set[str]:
        """주요 구문 추출"""
        if not text:
            return set()
        
        # 간단한 n-gram 추출 (2-3 단어 구문)
        words = self._normalize_text(text).split()
        phrases = set()
        
        # 2-gram
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 6:  # 너무 짧은 구문 제외
                phrases.add(phrase)
        
        # 3-gram (중요한 구문들)
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) > 10:  # 충분히 긴 구문만
                phrases.add(phrase)
        
        # 빈도 기반 필터링 (간단한 구현)
        return set(list(phrases)[:max_phrases])
    
    def _extract_numbers(self, text: str) -> Set[str]:
        """숫자 패턴 추출"""
        if not text:
            return set()
        
        # 다양한 숫자 패턴
        patterns = [
            r'\d+\.\d+%',      # 퍼센트
            r'\$\d+(?:,\d{3})*(?:\.\d+)?',  # 금액
            r'\d+(?:,\d{3})+',  # 큰 숫자
            r'\d{4}-\d{2}-\d{2}',  # 날짜
        ]
        
        numbers = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.update(matches)
        
        return numbers
    
    def _extract_simple_entities(self, text: str) -> Set[str]:
        """간단한 엔티티 추출"""
        if not text:
            return set()
        
        # 주요 금융/암호화폐 엔티티
        entities = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'fed', 'sec', 'fomc',
            'powell', 'yellen', 'biden', 'trump', 'tesla', 'apple',
            'microsoft', 'amazon', 'google', 'meta', 'nvidia'
        }
        
        text_lower = text.lower()
        found_entities = set()
        
        for entity in entities:
            if entity in text_lower:
                found_entities.add(entity)
        
        return found_entities
    
    def _calculate_similarity(self, features_1: Dict[str, Any], features_2: Dict[str, Any]) -> float:
        """두 뉴스 간 유사도 계산"""
        try:
            # 1. 제목 유사도 (가중치: 0.4)
            title_similarity = self._calculate_jaccard_similarity(
                features_1['title_words'], 
                features_2['title_words']
            )
            
            # 2. 내용 유사도 (가중치: 0.3)
            content_similarity = self._calculate_jaccard_similarity(
                features_1['content_words'][:100],  # 처음 100단어만 사용 (성능 최적화)
                features_2['content_words'][:100]
            )
            
            # 3. 주요 구문 유사도 (가중치: 0.2)
            phrase_similarity = self._calculate_jaccard_similarity(
                features_1['key_phrases'],
                features_2['key_phrases']
            )
            
            # 4. 숫자 패턴 유사도 (가중치: 0.1)
            number_similarity = self._calculate_jaccard_similarity(
                features_1['numbers'],
                features_2['numbers']
            )
            
            # 가중 평균 계산
            total_similarity = (
                title_similarity * 0.4 +
                content_similarity * 0.3 +
                phrase_similarity * 0.2 +
                number_similarity * 0.1
            )
            
            # 엔티티 부스트 (같은 주요 엔티티가 있으면 유사도 증가)
            if features_1['entities'] & features_2['entities']:
                entity_boost = len(features_1['entities'] & features_2['entities']) * 0.05
                total_similarity = min(1.0, total_similarity + entity_boost)
            
            return total_similarity
            
        except Exception as e:
            logger.debug(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Jaccard 유사도 계산"""
        if not set1 and not set2:
            return 1.0
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _update_stats(self, result: DeduplicationResult):
        """통계 업데이트"""
        self.stats["total_processed"] += result.original_count
        self.stats["total_duplicates_found"] += (result.original_count - result.deduplicated_count)
        self.stats["total_clusters_created"] += len(result.duplicate_clusters)
        self.stats["processing_time_ms"] = result.processing_time * 1000
        
        # 평균 클러스터 크기 계산
        if result.duplicate_clusters:
            cluster_sizes = [cluster.get_size() for cluster in result.duplicate_clusters.values()]
            self.stats["avg_cluster_size"] = sum(cluster_sizes) / len(cluster_sizes)
    
    def cleanup_old_features(self, max_age_hours: int = 24):
        """오래된 특징 캐시 정리"""
        if len(self.news_features) <= self.feature_cache_limit:
            return
        
        # 간단한 LRU: 절반 정도 제거
        items_to_remove = len(self.news_features) - (self.feature_cache_limit // 2)
        keys_to_remove = list(self.news_features.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.news_features[key]
        
        self.stats["last_cleanup"] = datetime.now()
        logger.debug(f"Cleaned up {items_to_remove} cached features")
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """중복 제거 통계 반환"""
        total_processed = self.stats["total_processed"]
        if total_processed == 0:
            return self.stats
        
        return {
            **self.stats,
            "deduplication_rate": self.stats["total_duplicates_found"] / total_processed,
            "efficiency_score": (
                (total_processed - self.stats["total_duplicates_found"]) / total_processed
                if total_processed > 0 else 0
            ),
            "cached_features_count": len(self.news_features),
            "active_clusters_count": len(self.news_clusters)
        }

# 글로벌 인스턴스
_global_deduplicator: Optional[NewsDeduplicator] = None

def get_news_deduplicator() -> NewsDeduplicator:
    """뉴스 중복 제거 시스템 인스턴스 반환"""
    global _global_deduplicator
    if _global_deduplicator is None:
        _global_deduplicator = NewsDeduplicator()
    return _global_deduplicator

def deduplicate_news_items(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """뉴스 아이템 중복 제거 (간편 함수)"""
    deduplicator = get_news_deduplicator()
    result = deduplicator.deduplicate_news_batch(news_items)
    return [item for item in news_items if item.get('hash_id') not in result.removed_news_ids]

if __name__ == "__main__":
    # 테스트 코드
    deduplicator = NewsDeduplicator(similarity_threshold=0.7)
    
    # 테스트 데이터
    test_news = [
        {
            "title": "Fed Raises Interest Rates by 0.75%",
            "content": "The Federal Reserve raised interest rates by 0.75 percentage points in an aggressive move to combat inflation.",
            "url": "https://reuters.com/fed-rates-1",
            "hash_id": "news_1"
        },
        {
            "title": "Federal Reserve Hikes Rates by 75 Basis Points",
            "content": "In a significant policy shift, the Fed increased rates by 0.75% to address rising inflation concerns.",
            "url": "https://bloomberg.com/fed-rates-2",
            "hash_id": "news_2"
        },
        {
            "title": "Bitcoin Price Surges Above $50,000",
            "content": "Bitcoin reached a new monthly high, trading above $50,000 amid institutional buying interest.",
            "url": "https://coindesk.com/btc-surge",
            "hash_id": "news_3"
        },
        {
            "title": "Fed Raises Interest Rates by 0.75%",  # 완전 중복
            "content": "The Federal Reserve raised interest rates by 0.75 percentage points in an aggressive move to combat inflation.",
            "url": "https://reuters.com/fed-rates-1",
            "hash_id": "news_1"
        }
    ]
    
    print("=== News Deduplication Test ===\n")
    print(f"Original news count: {len(test_news)}")
    
    result = deduplicator.deduplicate_news_batch(test_news)
    
    print(f"Deduplicated news count: {result.deduplicated_count}")
    print(f"Deduplication rate: {result.deduplication_rate:.1%}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Clusters found: {len(result.duplicate_clusters)}")
    
    for cluster_id, cluster in result.duplicate_clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster.get_size()}")
        print(f"  Representative: {cluster.representative_id}")
        print(f"  Members: {cluster.member_ids}")
    
    # 통계 출력
    stats = deduplicator.get_deduplication_stats()
    print(f"\n=== Deduplication Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")