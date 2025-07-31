#!/usr/bin/env python3
"""
Data Quality Validator for AuroraQ Sentiment Service
데이터 품질 검증, 중복 제거, 신뢰도 기반 필터링
"""

import re
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from difflib import SequenceMatcher
import statistics

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """품질 등급"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"

class ContentType(Enum):
    """컨텐츠 유형"""
    NEWS_ARTICLE = "news_article"
    SOCIAL_POST = "social_post"
    SEARCH_RESULT = "search_result"
    RESEARCH_REPORT = "research_report"

@dataclass
class QualityScore:
    """품질 점수"""
    overall_score: float  # 0.0 ~ 1.0
    quality_level: QualityLevel
    content_quality: float
    source_credibility: float
    market_relevance: float
    temporal_relevance: float
    uniqueness_score: float
    flags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def is_acceptable(self, min_threshold: float = 0.6) -> bool:
        """수용 가능한 품질인지 확인"""
        return self.overall_score >= min_threshold and self.quality_level != QualityLevel.REJECTED

@dataclass
class ContentItem:
    """검증할 컨텐츠 아이템"""
    content_id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    content_type: ContentType
    category: str = "general"
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataQualityValidator:
    """데이터 품질 검증기"""
    
    def __init__(self):
        """초기화"""
        self.processed_hashes: Set[str] = set()
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        
        # 신뢰할 수 있는 소스 가중치
        self.source_credibility = {
            "reuters.com": 1.0,
            "bloomberg.com": 1.0,
            "wsj.com": 0.95,
            "ft.com": 0.95,
            "cnbc.com": 0.9,
            "marketwatch.com": 0.85,
            "yahoo.com": 0.8,
            "coindesk.com": 0.9,
            "google_news": 0.75,
            "newsapi": 0.7,
            "reddit": 0.6,
            "social": 0.5
        }
        
        # 컨텐츠 품질 패턴
        self.quality_patterns = {
            "spam_indicators": [
                r'\b(click here|buy now|limited time|act now|free money)\b',
                r'[!]{3,}',  # 과도한 느낌표
                r'[A-Z]{10,}',  # 과도한 대문자
                r'\$+[0-9,]+',  # 가격 강조
            ],
            "low_quality_indicators": [
                r'\b(breaking|urgent|alert|must read)\b',
                r'[\?\!]{2,}',  # 연속된 물음표/느낌표
                r'\b(100%|guaranteed|secret|insider)\b',
            ],
            "high_quality_indicators": [
                r'\b(analysis|research|study|report|according to)\b',
                r'\b(expert|analyst|economist|professor)\b',
                r'\b(data shows|statistics|survey|poll)\b',
            ]
        }
        
        # 시장 관련 키워드 (관련성 계산용)
        self.market_keywords = {
            "crypto": ["bitcoin", "ethereum", "cryptocurrency", "blockchain", "defi", "nft"],
            "stocks": ["stock", "equity", "share", "market", "trading", "investment"],
            "economy": ["economic", "gdp", "inflation", "interest", "federal", "policy"],
            "finance": ["financial", "banking", "credit", "loan", "mortgage", "fintech"]
        }
        
        # 최소 품질 기준
        self.min_title_length = 10
        self.min_content_length = 50
        self.max_title_length = 200
        self.max_content_length = 10000
        self.similarity_threshold = 0.85  # 중복 임계값
        
    def _calculate_content_hash(self, content: str, title: str) -> str:
        """컨텐츠 해시 계산"""
        hash_input = f"{title.lower().strip()}{content[:200].lower().strip()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """텍스트 특성 추출"""
        if not text:
            return {}
        
        # 기본 통계
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # 특수 문자 비율
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        special_ratio = special_chars / char_count if char_count > 0 else 0
        
        # 대문자 비율
        uppercase_chars = len(re.findall(r'[A-Z]', text))
        uppercase_ratio = uppercase_chars / char_count if char_count > 0 else 0
        
        # 숫자 비율
        digit_chars = len(re.findall(r'[0-9]', text))
        digit_ratio = digit_chars / char_count if char_count > 0 else 0
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": char_count / word_count if word_count > 0 else 0,
            "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0,
            "special_ratio": special_ratio,
            "uppercase_ratio": uppercase_ratio,
            "digit_ratio": digit_ratio
        }
    
    def _calculate_content_quality(self, item: ContentItem) -> Tuple[float, List[str]]:
        """컨텐츠 품질 점수 계산"""
        flags = []
        score = 0.5  # 기본 점수
        
        # 1. 길이 검증
        title_len = len(item.title.strip())
        content_len = len(item.content.strip())
        
        if title_len < self.min_title_length:
            flags.append("title_too_short")
            score -= 0.2
        elif title_len > self.max_title_length:
            flags.append("title_too_long")
            score -= 0.1
        
        if content_len < self.min_content_length:
            flags.append("content_too_short")
            score -= 0.3
        elif content_len > self.max_content_length:
            flags.append("content_too_long")
            score -= 0.1
        
        # 2. 스팸 지표 확인
        full_text = f"{item.title} {item.content}".lower()
        
        spam_matches = 0
        for pattern in self.quality_patterns["spam_indicators"]:
            if re.search(pattern, full_text, re.IGNORECASE):
                spam_matches += 1
                flags.append(f"spam_pattern_{spam_matches}")
        
        if spam_matches > 0:
            score -= spam_matches * 0.15
        
        # 3. 저품질 지표 확인
        low_quality_matches = 0
        for pattern in self.quality_patterns["low_quality_indicators"]:
            if re.search(pattern, full_text, re.IGNORECASE):
                low_quality_matches += 1
        
        if low_quality_matches > 0:
            score -= low_quality_matches * 0.1
            flags.append(f"low_quality_indicators_{low_quality_matches}")
        
        # 4. 고품질 지표 확인
        high_quality_matches = 0
        for pattern in self.quality_patterns["high_quality_indicators"]:
            if re.search(pattern, full_text, re.IGNORECASE):
                high_quality_matches += 1
        
        if high_quality_matches > 0:
            score += high_quality_matches * 0.1
            flags.append(f"high_quality_indicators_{high_quality_matches}")
        
        # 5. 텍스트 특성 분석
        features = self._extract_text_features(full_text)
        
        # 과도한 특수문자나 대문자 사용 패널티
        if features.get("special_ratio", 0) > 0.15:
            score -= 0.1
            flags.append("excessive_special_chars")
        
        if features.get("uppercase_ratio", 0) > 0.3:
            score -= 0.1
            flags.append("excessive_uppercase")
        
        # 6. URL 품질 확인
        if not item.url or "example.com" in item.url:
            score -= 0.1
            flags.append("invalid_url")
        
        return max(0.0, min(1.0, score)), flags
    
    def _calculate_source_credibility(self, item: ContentItem) -> float:
        """소스 신뢰도 계산"""
        
        # URL 기반 신뢰도
        url_credibility = 0.5
        if item.url:
            for domain, credibility in self.source_credibility.items():
                if domain in item.url.lower():
                    url_credibility = credibility
                    break
        
        # 소스명 기반 신뢰도
        source_credibility = self.source_credibility.get(item.source.lower(), 0.5)
        
        # 컨텐츠 유형별 가중치
        type_weights = {
            ContentType.NEWS_ARTICLE: 1.0,
            ContentType.RESEARCH_REPORT: 1.0,
            ContentType.SEARCH_RESULT: 0.8,
            ContentType.SOCIAL_POST: 0.6
        }
        
        type_weight = type_weights.get(item.content_type, 0.7)
        
        # 최종 신뢰도 (가중 평균)
        final_credibility = (url_credibility * 0.6 + source_credibility * 0.4) * type_weight
        
        return max(0.0, min(1.0, final_credibility))
    
    def _calculate_market_relevance(self, item: ContentItem) -> float:
        """시장 관련성 계산"""
        
        full_text = f"{item.title} {item.content}".lower()
        total_score = 0.0
        
        # 카테고리별 키워드 매칭
        for category, keywords in self.market_keywords.items():
            category_matches = sum(1 for keyword in keywords if keyword in full_text)
            if category_matches > 0:
                category_score = min(1.0, category_matches / len(keywords))
                total_score += category_score
        
        # 심볼 특화 관련성
        if item.symbol:
            symbol_lower = item.symbol.lower()
            if symbol_lower in full_text:
                total_score += 0.3
        
        # 기본 금융 용어 확인
        finance_terms = ["price", "trading", "market", "investment", "economic", "financial"]
        finance_matches = sum(1 for term in finance_terms if term in full_text)
        finance_score = min(0.5, finance_matches / len(finance_terms))
        total_score += finance_score
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_temporal_relevance(self, item: ContentItem) -> float:
        """시간적 관련성 계산"""
        
        if not item.published_at:
            return 0.5
        
        now = datetime.now()
        time_diff = now - item.published_at
        
        # 시간에 따른 관련성 감소 (지수적)
        hours_diff = time_diff.total_seconds() / 3600
        
        if hours_diff <= 1:
            return 1.0
        elif hours_diff <= 6:
            return 0.9
        elif hours_diff <= 24:
            return 0.8
        elif hours_diff <= 72:
            return 0.6
        elif hours_diff <= 168:  # 1주
            return 0.4
        else:
            return 0.2
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        
        # 간단한 전처리
        clean1 = re.sub(r'[^\w\s]', '', text1.lower()).strip()
        clean2 = re.sub(r'[^\w\s]', '', text2.lower()).strip()
        
        if not clean1 or not clean2:
            return 0.0
        
        # SequenceMatcher를 사용한 유사도 계산
        return SequenceMatcher(None, clean1, clean2).ratio()
    
    def _calculate_uniqueness_score(self, item: ContentItem, existing_items: List[ContentItem]) -> float:
        """고유성 점수 계산 (중복 검출)"""
        
        if not existing_items:
            return 1.0
        
        max_similarity = 0.0
        item_text = f"{item.title} {item.content[:500]}"  # 처음 500자만 비교
        
        for existing in existing_items:
            existing_text = f"{existing.title} {existing.content[:500]}"
            
            # 캐시된 유사도 확인
            cache_key = f"{item.content_id}_{existing.content_id}"
            reverse_key = f"{existing.content_id}_{item.content_id}"
            
            if cache_key in self.similarity_cache:
                similarity = self.similarity_cache[cache_key]
            elif reverse_key in self.similarity_cache:
                similarity = self.similarity_cache[reverse_key]
            else:
                similarity = self._calculate_text_similarity(item_text, existing_text)
                self.similarity_cache[cache_key] = similarity
            
            max_similarity = max(max_similarity, similarity)
            
            # 조기 종료 (이미 중복으로 판단되는 경우)
            if max_similarity >= self.similarity_threshold:
                break
        
        # 유사도를 고유성 점수로 변환
        uniqueness = 1.0 - max_similarity
        return max(0.0, uniqueness)
    
    def validate_single_item(self, 
                           item: ContentItem,
                           existing_items: Optional[List[ContentItem]] = None) -> QualityScore:
        """단일 아이템 품질 검증"""
        
        if existing_items is None:
            existing_items = []
        
        # 각 차원별 점수 계산
        content_quality, flags = self._calculate_content_quality(item)
        source_credibility = self._calculate_source_credibility(item)
        market_relevance = self._calculate_market_relevance(item)
        temporal_relevance = self._calculate_temporal_relevance(item)
        uniqueness_score = self._calculate_uniqueness_score(item, existing_items)
        
        # 가중 평균으로 종합 점수 계산
        weights = {
            "content_quality": 0.25,
            "source_credibility": 0.20,
            "market_relevance": 0.25,
            "temporal_relevance": 0.10,
            "uniqueness_score": 0.20
        }
        
        overall_score = (
            content_quality * weights["content_quality"] +
            source_credibility * weights["source_credibility"] +
            market_relevance * weights["market_relevance"] +
            temporal_relevance * weights["temporal_relevance"] +
            uniqueness_score * weights["uniqueness_score"]
        )
        
        # 품질 등급 결정
        if overall_score >= 0.8:
            quality_level = QualityLevel.HIGH
        elif overall_score >= 0.6:
            quality_level = QualityLevel.MEDIUM
        elif overall_score >= 0.4:
            quality_level = QualityLevel.LOW
        else:
            quality_level = QualityLevel.REJECTED
        
        # 중복 확인
        if uniqueness_score < (1.0 - self.similarity_threshold):
            flags.append("duplicate_content")
            quality_level = QualityLevel.REJECTED
        
        # 신뢰도 계산 (품질과 소스 신뢰도의 조합)
        confidence = (overall_score * 0.7 + source_credibility * 0.3)
        
        return QualityScore(
            overall_score=round(overall_score, 3),
            quality_level=quality_level,
            content_quality=round(content_quality, 3),
            source_credibility=round(source_credibility, 3),
            market_relevance=round(market_relevance, 3),
            temporal_relevance=round(temporal_relevance, 3),
            uniqueness_score=round(uniqueness_score, 3),
            flags=flags,
            confidence=round(confidence, 3)
        )
    
    def validate_batch(self, 
                      items: List[ContentItem],
                      min_threshold: float = 0.6) -> Tuple[List[ContentItem], List[QualityScore]]:
        """배치 검증 및 필터링"""
        
        logger.info(f"Validating batch of {len(items)} items...")
        
        validated_items = []
        quality_scores = []
        processed_items = []  # 중복 검출용
        
        for item in items:
            # 품질 검증
            quality_score = self.validate_single_item(item, processed_items)
            quality_scores.append(quality_score)
            
            # 수용 기준 확인
            if quality_score.is_acceptable(min_threshold):
                validated_items.append(item)
                processed_items.append(item)
            else:
                logger.debug(f"Item rejected: {item.title[:50]}... "
                           f"(score: {quality_score.overall_score:.2f}, "
                           f"level: {quality_score.quality_level.value})")
        
        # 통계 로깅
        accepted_count = len(validated_items)
        rejected_count = len(items) - accepted_count
        
        if quality_scores:
            avg_score = statistics.mean(score.overall_score for score in quality_scores)
            logger.info(f"Batch validation completed: "
                       f"{accepted_count} accepted, {rejected_count} rejected "
                       f"(avg score: {avg_score:.3f})")
        
        return validated_items, quality_scores
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        return {
            "processed_hashes": len(self.processed_hashes),
            "similarity_cache_size": len(self.similarity_cache),
            "similarity_threshold": self.similarity_threshold,
            "min_quality_thresholds": {
                "title_length": self.min_title_length,
                "content_length": self.min_content_length
            },
            "source_credibility_levels": len(self.source_credibility),
            "quality_patterns": {
                pattern_type: len(patterns) 
                for pattern_type, patterns in self.quality_patterns.items()
            }
        }


# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test_data_quality_validator():
        """데이터 품질 검증기 테스트"""
        
        validator = DataQualityValidator()
        
        print("=== 데이터 품질 검증 테스트 ===")
        
        # 테스트 아이템들
        test_items = [
            ContentItem(
                content_id="test1",
                title="Bitcoin Surges to New All-Time High Amid Institutional Adoption",
                content="Bitcoin reached a new all-time high today as major institutional investors continue to embrace the cryptocurrency. Analysis shows strong market fundamentals and growing corporate adoption.",
                url="https://reuters.com/bitcoin-news",
                source="reuters.com",
                published_at=datetime.now() - timedelta(hours=2),
                content_type=ContentType.NEWS_ARTICLE,
                category="crypto",
                symbol="BTC"
            ),
            ContentItem(
                content_id="test2",
                title="URGENT!!! BUY NOW!!! Bitcoin to the MOON!!!",
                content="Click here for guaranteed profits! Don't miss out! Limited time offer! FREE MONEY!",
                url="https://spam-site.com/scam",
                source="unknown",
                published_at=datetime.now() - timedelta(hours=1),
                content_type=ContentType.SOCIAL_POST,
                category="crypto",
                symbol="BTC"
            ),
            ContentItem(
                content_id="test3",
                title="Bitcoin Hits Record High as Institutions Invest",
                content="Bitcoin has reached record levels today with institutional investors showing increased interest. Market analysis indicates strong fundamentals supporting the price increase.",
                url="https://bloomberg.com/bitcoin",
                source="bloomberg.com",
                published_at=datetime.now() - timedelta(hours=1),
                content_type=ContentType.NEWS_ARTICLE,
                category="crypto",
                symbol="BTC"
            )
        ]
        
        # 개별 검증
        print("\n1. 개별 아이템 검증:")
        for i, item in enumerate(test_items, 1):
            quality_score = validator.validate_single_item(item)
            
            print(f"\n아이템 {i}: {item.title[:50]}...")
            print(f"  종합 점수: {quality_score.overall_score:.3f}")
            print(f"  품질 등급: {quality_score.quality_level.value}")
            print(f"  컨텐츠 품질: {quality_score.content_quality:.3f}")
            print(f"  소스 신뢰도: {quality_score.source_credibility:.3f}")
            print(f"  시장 관련성: {quality_score.market_relevance:.3f}")
            print(f"  시간적 관련성: {quality_score.temporal_relevance:.3f}")
            print(f"  고유성: {quality_score.uniqueness_score:.3f}")
            print(f"  신뢰도: {quality_score.confidence:.3f}")
            print(f"  플래그: {quality_score.flags}")
            print(f"  수용 가능: {quality_score.is_acceptable()}")
        
        # 배치 검증
        print(f"\n2. 배치 검증 (임계값: 0.6):")
        validated_items, quality_scores = validator.validate_batch(test_items, min_threshold=0.6)
        
        print(f"원본 아이템: {len(test_items)}개")
        print(f"검증 통과: {len(validated_items)}개")
        print(f"거부된 아이템: {len(test_items) - len(validated_items)}개")
        
        # 통계
        print(f"\n3. 검증 통계:")
        stats = validator.get_validation_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 테스트 실행
    test_data_quality_validator()