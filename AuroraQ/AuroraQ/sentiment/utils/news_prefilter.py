#!/usr/bin/env python3
"""
News Pre-Filter System - FinBERT 분석 전 사전 필터링 시스템
불필요한 기사 제거 및 분석 우선순위 결정
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FilterReason(Enum):
    """필터링 이유"""
    CONTENT_TOO_SHORT = "content_too_short"           # 내용 부족
    CONTENT_TOO_LONG = "content_too_long"             # 내용 과다
    INVALID_LANGUAGE = "invalid_language"             # 언어 부적합
    SPAM_DETECTED = "spam_detected"                   # 스팸 감지
    LOW_RELEVANCE = "low_relevance"                   # 관련성 낮음
    DUPLICATE_CONTENT = "duplicate_content"           # 중복 내용
    OUTDATED_NEWS = "outdated_news"                   # 오래된 뉴스
    LOW_IMPORTANCE = "low_importance"                 # 중요도 낮음
    TECHNICAL_NOISE = "technical_noise"               # 기술적 노이즈
    ADVERTISEMENT = "advertisement"                   # 광고성 내용

class FilterDecision(Enum):
    """필터링 결정"""
    APPROVE_HIGH_PRIORITY = "approve_high_priority"   # 고우선순위 승인
    APPROVE_NORMAL = "approve_normal"                 # 일반 승인
    APPROVE_LOW_PRIORITY = "approve_low_priority"     # 저우선순위 승인
    REJECT = "reject"                                 # 거부

@dataclass
class FilterResult:
    """필터링 결과"""
    decision: FilterDecision
    confidence: float                    # 결정 신뢰도 (0.0-1.0)
    reasons: List[FilterReason]         # 필터링 이유들
    importance_score: float = 0.0       # 중요도 점수
    quality_score: float = 0.0          # 품질 점수
    processing_time: float = 0.0        # 처리 시간
    metadata: Dict[str, Any] = None     # 추가 메타데이터
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class NewsPreFilter:
    """뉴스 사전 필터링 시스템"""
    
    def __init__(self, 
                 min_content_length: int = 50,
                 max_content_length: int = 5000,
                 min_importance_threshold: float = 0.2,
                 spam_threshold: float = 0.7):
        """
        초기화
        
        Args:
            min_content_length: 최소 내용 길이
            max_content_length: 최대 내용 길이  
            min_importance_threshold: 최소 중요도 임계값
            spam_threshold: 스팸 감지 임계값
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.min_importance_threshold = min_importance_threshold
        self.spam_threshold = spam_threshold
        
        # 스팸 키워드 패턴
        self.spam_patterns = {
            # 광고성 키워드
            "advertisement": {
                r"click here", r"buy now", r"limited time", r"special offer",
                r"discount", r"sale", r"promo", r"advertisement", r"sponsored",
                r"affiliate", r"partnership", r"collaboration"
            },
            
            # 클릭베이트 패턴
            "clickbait": {
                r"you won't believe", r"shocking", r"amazing", r"incredible",
                r"this will", r"must see", r"viral", r"trending now",
                r"everyone is talking", r"goes viral"
            },
            
            # 저품질 컨텐츠
            "low_quality": {
                r"lorem ipsum", r"placeholder", r"test content",
                r"under construction", r"coming soon", r"more details",
                r"stay tuned", r"update soon"
            }
        }
        
        # 금융 관련성 키워드
        self.relevance_keywords = {
            # 고관련성 (가중치 1.0)
            "high_relevance": {
                "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "crypto",
                "blockchain", "fed", "federal reserve", "interest rate", "inflation",
                "sec", "regulation", "etf", "institutional", "trading", "market",
                "investment", "finance", "economic", "monetary", "fiscal"
            },
            
            # 중관련성 (가중치 0.6)
            "medium_relevance": {
                "technology", "innovation", "digital", "fintech", "payment",
                "currency", "dollar", "euro", "yen", "gold", "commodity",
                "stock", "equity", "bond", "derivative", "futures"
            },
            
            # 저관련성 (가중치 0.3)
            "low_relevance": {
                "business", "company", "corporate", "industry", "sector",
                "economy", "growth", "development", "strategy", "policy"
            }
        }
        
        # 언어 감지 패턴 (간단한 구현)
        self.language_patterns = {
            "english": {
                r"\b(?:the|and|or|but|in|on|at|to|for|of|with|by|from|is|are|was|were|be|been|have|has|had|do|does|did|will|would|could|should|may|might|can|shall)\b"
            },
            "non_english": {
                r"[가-힣]",  # 한글
                r"[а-я]",    # 러시아어
                r"[α-ω]",    # 그리스어
                r"[\u4e00-\u9fff]"  # 중국어
            }
        }
        
        # 통계
        self.stats = {
            "total_processed": 0,
            "approved_high_priority": 0,
            "approved_normal": 0,
            "approved_low_priority": 0,
            "rejected": 0,
            "avg_processing_time": 0.0,
            "filter_reasons": {},
            "last_update": datetime.now()
        }
    
    def filter_news_item(self, 
                        news_item: Dict[str, Any],
                        importance_score: Optional[float] = None) -> FilterResult:
        """
        개별 뉴스 아이템 필터링
        
        Args:
            news_item: 뉴스 아이템 (title, content, url, published_at 등)
            importance_score: 사전 계산된 중요도 점수
            
        Returns:
            FilterResult: 필터링 결과
        """
        start_time = time.time()
        
        try:
            title = news_item.get('title', '')
            content = news_item.get('content', '')
            url = news_item.get('url', '')
            published_at = news_item.get('published_at', datetime.now())
            
            reasons = []
            quality_scores = []
            
            # 1. 기본 유효성 검사
            basic_check, basic_reasons, basic_quality = self._check_basic_validity(title, content, url)
            reasons.extend(basic_reasons)
            quality_scores.append(basic_quality)
            
            if not basic_check:
                return self._create_result(FilterDecision.REJECT, 0.9, reasons, 
                                         0.0, basic_quality, time.time() - start_time)
            
            # 2. 스팸 감지
            spam_check, spam_reasons, spam_score = self._detect_spam(title, content)
            reasons.extend(spam_reasons)
            quality_scores.append(1.0 - spam_score)  # 스팸 점수를 품질 점수로 변환
            
            if not spam_check:
                return self._create_result(FilterDecision.REJECT, 0.8, reasons,
                                         0.0, min(quality_scores), time.time() - start_time)
            
            # 3. 관련성 검사
            relevance_score = self._calculate_relevance_score(title, content)
            if relevance_score < 0.3:  # 관련성이 너무 낮음
                reasons.append(FilterReason.LOW_RELEVANCE)
                if relevance_score < 0.1:
                    return self._create_result(FilterDecision.REJECT, 0.7, reasons,
                                             0.0, min(quality_scores), time.time() - start_time)
            
            # 4. 시간 유효성 검사
            time_check, time_reasons = self._check_time_validity(published_at)
            reasons.extend(time_reasons)
            
            # 5. 언어 검사
            language_check, language_reasons = self._check_language(title, content)
            reasons.extend(language_reasons)
            
            if not language_check:
                return self._create_result(FilterDecision.REJECT, 0.6, reasons,
                                         0.0, min(quality_scores), time.time() - start_time)
            
            # 6. 중요도 기반 우선순위 결정
            final_importance = importance_score if importance_score is not None else relevance_score
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            
            decision, confidence = self._make_final_decision(
                final_importance, avg_quality, reasons, time_check
            )
            
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self._update_stats(decision, reasons, processing_time)
            
            return self._create_result(decision, confidence, reasons,
                                     final_importance, avg_quality, processing_time)
            
        except Exception as e:
            logger.error(f"Error filtering news item: {e}")
            return self._create_result(FilterDecision.REJECT, 1.0, 
                                     [FilterReason.TECHNICAL_NOISE], 
                                     0.0, 0.0, time.time() - start_time)
    
    def filter_news_batch(self, news_items: List[Dict[str, Any]], 
                         importance_scores: Optional[List[float]] = None) -> List[FilterResult]:
        """
        뉴스 배치 필터링
        
        Args:
            news_items: 뉴스 아이템 리스트
            importance_scores: 중요도 점수 리스트 (선택사항)
            
        Returns:
            List[FilterResult]: 필터링 결과 리스트
        """
        results = []
        
        for i, news_item in enumerate(news_items):
            importance_score = None
            if importance_scores and i < len(importance_scores):
                importance_score = importance_scores[i]
            
            result = self.filter_news_item(news_item, importance_score)
            results.append(result)
        
        logger.info(f"Batch filtering completed: {len(news_items)} items processed")
        return results
    
    def _check_basic_validity(self, title: str, content: str, url: str) -> Tuple[bool, List[FilterReason], float]:
        """기본 유효성 검사"""
        reasons = []
        quality_score = 1.0
        
        # 제목 검사
        if not title or len(title.strip()) < 5:
            reasons.append(FilterReason.CONTENT_TOO_SHORT)
            quality_score -= 0.3
        
        # 내용 길이 검사
        content_length = len(content) if content else 0
        
        if content_length < self.min_content_length:
            reasons.append(FilterReason.CONTENT_TOO_SHORT)
            quality_score -= 0.4
        elif content_length > self.max_content_length:
            reasons.append(FilterReason.CONTENT_TOO_LONG)
            quality_score -= 0.2
        
        # URL 유효성 (기본적인 검사)
        if not url or not url.startswith(('http://', 'https://')):
            quality_score -= 0.1
        
        # 제목과 내용의 유사성 검사 (너무 유사하면 품질 낮음)
        if title and content and len(title) > 20:
            title_in_content_ratio = content.lower().count(title.lower()[:50]) / max(len(content), 1)
            if title_in_content_ratio > 0.1:  # 제목이 내용에 너무 많이 반복
                quality_score -= 0.1
        
        is_valid = len(reasons) == 0 or (content_length >= self.min_content_length and title)
        return is_valid, reasons, max(0.0, quality_score)
    
    def _detect_spam(self, title: str, content: str) -> Tuple[bool, List[FilterReason], float]:
        """스팸 감지"""
        text = f"{title} {content}".lower()
        spam_score = 0.0
        reasons = []
        
        # 각 스팸 카테고리별 점수 계산
        for category, patterns in self.spam_patterns.items():
            category_score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
            
            if matches > 0:
                if category == "advertisement":
                    category_score = min(0.4, matches * 0.1)
                elif category == "clickbait":
                    category_score = min(0.3, matches * 0.08)
                elif category == "low_quality":
                    category_score = min(0.2, matches * 0.05)
                
                spam_score += category_score
        
        # 반복 패턴 검사
        if text:
            words = text.split()
            if len(words) > 10:
                unique_words = set(words)
                repetition_ratio = 1 - (len(unique_words) / len(words))
                if repetition_ratio > 0.5:  # 50% 이상 반복
                    spam_score += 0.2
        
        # 대문자 과다 사용 검사
        if title:
            upper_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
            if upper_ratio > 0.5:  # 50% 이상 대문자
                spam_score += 0.1
        
        spam_score = min(1.0, spam_score)
        
        if spam_score >= self.spam_threshold:
            reasons.append(FilterReason.SPAM_DETECTED)
        
        is_not_spam = spam_score < self.spam_threshold
        return is_not_spam, reasons, spam_score
    
    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """관련성 점수 계산"""
        text = f"{title} {content}".lower()
        if not text.strip():
            return 0.0
        
        relevance_score = 0.0
        
        # 각 관련성 레벨별 점수 계산
        for level, keywords in self.relevance_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            
            if matches > 0:
                if level == "high_relevance":
                    relevance_score += min(0.8, matches * 0.15)
                elif level == "medium_relevance":
                    relevance_score += min(0.4, matches * 0.08)
                elif level == "low_relevance":
                    relevance_score += min(0.2, matches * 0.04)
        
        return min(1.0, relevance_score)
    
    def _check_time_validity(self, published_at: datetime) -> Tuple[bool, List[FilterReason]]:
        """시간 유효성 검사"""
        reasons = []
        now = datetime.now()
        
        # 미래 날짜 검사
        if published_at > now + timedelta(hours=1):
            reasons.append(FilterReason.TECHNICAL_NOISE)
            return False, reasons
        
        # 너무 오래된 뉴스 검사 (7일 이상)
        if published_at < now - timedelta(days=7):
            reasons.append(FilterReason.OUTDATED_NEWS)
            return False, reasons
        
        return True, reasons
    
    def _check_language(self, title: str, content: str) -> Tuple[bool, List[FilterReason]]:
        """언어 검사 (영어 우선)"""
        text = f"{title} {content}".lower()
        reasons = []
        
        if not text.strip():
            return True, reasons  # 빈 텍스트는 통과
        
        # 영어 패턴 매칭
        english_matches = 0
        for pattern in self.language_patterns["english"]:
            english_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # 비영어 패턴 매칭
        non_english_matches = 0
        for pattern in self.language_patterns["non_english"]:
            non_english_matches += len(re.findall(pattern, text))
        
        # 단어 수 기준 언어 비율 계산
        words = text.split()
        total_words = len(words)
        
        if total_words < 5:  # 너무 짧으면 통과
            return True, reasons
        
        # 영어 비율이 낮으면 거부
        english_ratio = english_matches / max(total_words, 1)
        non_english_ratio = non_english_matches / max(total_words, 1)
        
        if english_ratio < 0.1 and non_english_ratio > 0.3:
            reasons.append(FilterReason.INVALID_LANGUAGE)
            return False, reasons
        
        return True, reasons
    
    def _make_final_decision(self, importance_score: float, quality_score: float, 
                           reasons: List[FilterReason], time_valid: bool) -> Tuple[FilterDecision, float]:
        """최종 결정"""
        
        # 거부 조건들
        if FilterReason.SPAM_DETECTED in reasons:
            return FilterDecision.REJECT, 0.9
        
        if FilterReason.INVALID_LANGUAGE in reasons:
            return FilterDecision.REJECT, 0.8
        
        if not time_valid:
            return FilterDecision.REJECT, 0.7
        
        if importance_score < self.min_importance_threshold:
            return FilterDecision.REJECT, 0.6
        
        # 승인 우선순위 결정
        combined_score = (importance_score * 0.7 + quality_score * 0.3)
        
        if combined_score >= 0.8:
            return FilterDecision.APPROVE_HIGH_PRIORITY, 0.9
        elif combined_score >= 0.6:
            return FilterDecision.APPROVE_NORMAL, 0.8
        elif combined_score >= 0.4:
            return FilterDecision.APPROVE_LOW_PRIORITY, 0.7
        else:
            return FilterDecision.REJECT, 0.6
    
    def _create_result(self, decision: FilterDecision, confidence: float, 
                      reasons: List[FilterReason], importance_score: float,
                      quality_score: float, processing_time: float) -> FilterResult:
        """필터링 결과 생성"""
        return FilterResult(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            importance_score=importance_score,
            quality_score=quality_score,
            processing_time=processing_time,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "filter_version": "v2.0"
            }
        )
    
    def _update_stats(self, decision: FilterDecision, reasons: List[FilterReason], 
                     processing_time: float):
        """통계 업데이트"""
        self.stats["total_processed"] += 1
        
        # 결정별 카운트
        if decision == FilterDecision.APPROVE_HIGH_PRIORITY:
            self.stats["approved_high_priority"] += 1
        elif decision == FilterDecision.APPROVE_NORMAL:
            self.stats["approved_normal"] += 1
        elif decision == FilterDecision.APPROVE_LOW_PRIORITY:
            self.stats["approved_low_priority"] += 1
        else:
            self.stats["rejected"] += 1
        
        # 필터링 이유 통계
        for reason in reasons:
            reason_key = reason.value
            if reason_key not in self.stats["filter_reasons"]:
                self.stats["filter_reasons"][reason_key] = 0
            self.stats["filter_reasons"][reason_key] += 1
        
        # 평균 처리 시간 업데이트
        alpha = 0.1
        if self.stats["total_processed"] == 1:
            self.stats["avg_processing_time"] = processing_time
        else:
            self.stats["avg_processing_time"] = (
                alpha * processing_time + (1 - alpha) * self.stats["avg_processing_time"]
            )
        
        self.stats["last_update"] = datetime.now()
    
    def get_approved_news(self, filter_results: List[FilterResult], 
                         news_items: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], FilterResult]]:
        """승인된 뉴스만 반환"""
        approved = []
        
        for i, (news_item, result) in enumerate(zip(news_items, filter_results)):
            if result.decision in [
                FilterDecision.APPROVE_HIGH_PRIORITY,
                FilterDecision.APPROVE_NORMAL,
                FilterDecision.APPROVE_LOW_PRIORITY
            ]:
                approved.append((news_item, result))
        
        # 우선순위로 정렬
        priority_order = {
            FilterDecision.APPROVE_HIGH_PRIORITY: 0,
            FilterDecision.APPROVE_NORMAL: 1,
            FilterDecision.APPROVE_LOW_PRIORITY: 2
        }
        
        approved.sort(key=lambda x: (
            priority_order[x[1].decision],
            -x[1].importance_score,
            -x[1].quality_score
        ))
        
        return approved
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """필터링 통계 반환"""
        total = self.stats["total_processed"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "approval_rate": (
                (self.stats["approved_high_priority"] + 
                 self.stats["approved_normal"] + 
                 self.stats["approved_low_priority"]) / total
            ),
            "rejection_rate": self.stats["rejected"] / total,
            "high_priority_rate": self.stats["approved_high_priority"] / total,
            "efficiency_score": self.stats["approved_high_priority"] / max(total, 1)
        }

# 글로벌 인스턴스
_global_prefilter: Optional[NewsPreFilter] = None

def get_news_prefilter() -> NewsPreFilter:
    """뉴스 사전 필터링 시스템 인스턴스 반환"""
    global _global_prefilter
    if _global_prefilter is None:
        _global_prefilter = NewsPreFilter()
    return _global_prefilter

if __name__ == "__main__":
    # 테스트 코드
    prefilter = NewsPreFilter()
    
    # 테스트 데이터
    test_news = [
        {
            "title": "Fed Raises Interest Rates by 0.75% to Combat Inflation",
            "content": "The Federal Reserve announced a 0.75 percentage point increase in interest rates, the largest hike since 1994. The decision comes as inflation continues to surge, reaching 8.6% year-over-year. Fed Chair Jerome Powell emphasized the central bank's commitment to bringing inflation back to the 2% target.",
            "url": "https://reuters.com/markets/fed-decision",
            "published_at": datetime.now() - timedelta(hours=2)
        },
        {
            "title": "CLICK HERE NOW! AMAZING CRYPTO GAINS!!!",
            "content": "Buy now! Limited time offer! You won't believe these gains! Click here for amazing profits! Special discount today only!",
            "url": "https://spam-site.com/crypto-gains",
            "published_at": datetime.now() - timedelta(hours=1)
        },
        {
            "title": "Recipe for Chocolate Cake",
            "content": "Mix flour, sugar, and cocoa powder. Add eggs and milk. Bake for 30 minutes at 350°F.",
            "url": "https://cooking.com/chocolate-cake",
            "published_at": datetime.now() - timedelta(hours=3)
        }
    ]
    
    print("=== News Pre-Filtering Test ===\n")
    
    results = prefilter.filter_news_batch(test_news)
    
    for i, (news, result) in enumerate(zip(test_news, results)):
        print(f"News {i+1}:")
        print(f"  Title: {news['title'][:50]}...")
        print(f"  Decision: {result.decision.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Importance: {result.importance_score:.3f}")
        print(f"  Quality: {result.quality_score:.3f}")
        print(f"  Reasons: {[r.value for r in result.reasons]}")
        print(f"  Processing Time: {result.processing_time:.3f}s")
        print()
    
    # 승인된 뉴스만 추출
    approved = prefilter.get_approved_news(results, test_news)
    print(f"Approved news: {len(approved)}/{len(test_news)}")
    
    # 통계 출력
    stats = prefilter.get_filter_stats()
    print(f"\n=== Filter Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        elif key != "filter_reasons":
            print(f"{key}: {value}")
    
    if stats.get("filter_reasons"):
        print(f"Filter reasons breakdown:")
        for reason, count in stats["filter_reasons"].items():
            print(f"  {reason}: {count}")