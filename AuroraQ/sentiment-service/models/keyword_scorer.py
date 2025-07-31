#!/usr/bin/env python3
"""
실시간 키워드 기반 감정 스코어러
0.5초 내 응답으로 거래 신호용 감정 점수 제공
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class SentimentDirection(Enum):
    """감정 방향"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class KeywordScore:
    """키워드 점수 결과"""
    score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    direction: SentimentDirection
    matched_keywords: List[str]
    category_scores: Dict[str, float]
    processing_time: float

class KeywordScorer:
    """실시간 키워드 기반 감정 분석기"""
    
    def __init__(self):
        """초기화"""
        self.start_time = time.time()
        self._load_keyword_dictionary()
        logger.info("KeywordScorer initialized successfully")
    
    def _load_keyword_dictionary(self):
        """키워드 사전 로드"""
        
        # 가격 관련 키워드 (가중치: 0.4)
        self.price_keywords = {
            # 강세 키워드
            "surge": 0.8, "rally": 0.7, "bull": 0.6, "pump": 0.9,
            "moon": 0.8, "breakout": 0.7, "resistance": 0.5, "support": 0.4,
            "higher": 0.4, "rise": 0.5, "gain": 0.5, "profit": 0.6,
            "up": 0.3, "increase": 0.4, "climb": 0.5, "soar": 0.8,
            
            # 약세 키워드  
            "crash": -0.9, "dump": -0.8, "bear": -0.6, "collapse": -0.9,
            "drop": -0.5, "fall": -0.5, "decline": -0.5, "plunge": -0.8,
            "down": -0.3, "loss": -0.6, "correction": -0.4, "dip": -0.3,
            "selloff": -0.7, "liquidation": -0.8, "capitulation": -0.9
        }
        
        # 기관/규제 관련 키워드 (가중치: 0.3)
        self.institutional_keywords = {
            # 긍정적
            "approval": 0.8, "adopt": 0.6, "accept": 0.5, "embrace": 0.7,
            "invest": 0.6, "fund": 0.5, "partnership": 0.6, "collaboration": 0.6,
            "institutional": 0.5, "whale": 0.4, "backing": 0.7, "support": 0.5,
            "etf": 0.7, "regulation": 0.3, "compliance": 0.4, "legal": 0.3,
            
            # 부정적
            "ban": -0.9, "reject": -0.8, "prohibit": -0.8, "restriction": -0.6,
            "investigation": -0.6, "lawsuit": -0.7, "violation": -0.7, "penalty": -0.6,
            "crack": -0.5, "crackdown": -0.8, "sanction": -0.7, "fine": -0.5
        }
        
        # 시장 심리 키워드 (가중치: 0.2)
        self.sentiment_keywords = {
            # 긍정적 심리
            "optimistic": 0.6, "confident": 0.6, "positive": 0.5, "bullish": 0.7,
            "excitement": 0.6, "enthusiasm": 0.6, "hope": 0.4, "faith": 0.5,
            "momentum": 0.5, "strength": 0.4, "recovery": 0.6, "rebound": 0.6,
            
            # 부정적 심리
            "fear": -0.7, "panic": -0.8, "anxiety": -0.6, "concern": -0.4,
            "worry": -0.5, "doubt": -0.4, "uncertainty": -0.5, "skeptical": -0.5,
            "bearish": -0.7, "pessimistic": -0.6, "negative": -0.5, "weak": -0.4
        }
        
        # 기술적 지표 키워드 (가중치: 0.1)
        self.technical_keywords = {
            # 긍정적 신호
            "golden": 0.7, "cross": 0.4, "oversold": 0.5, "bounce": 0.5,
            "trend": 0.3, "uptrend": 0.6, "bullish": 0.6, "breakthrough": 0.7,
            
            # 부정적 신호
            "death": -0.8, "overbought": -0.4, "downtrend": -0.6, "breakdown": -0.7,
            "resistance": -0.3, "rejection": -0.5, "failure": -0.6
        }
        
        # 카테고리별 가중치
        self.category_weights = {
            "price": 0.4,
            "institutional": 0.3,
            "sentiment": 0.2,
            "technical": 0.1
        }
    
    @lru_cache(maxsize=1000)
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (캐싱 적용)"""
        if not text:
            return ""
        
        # 소문자 변환 및 정규화
        text = text.lower().strip()
        
        # 특수문자 제거 (단어 경계 유지)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """카테고리별 키워드 추출"""
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        matched_keywords = {
            "price": [],
            "institutional": [],
            "sentiment": [], 
            "technical": []
        }
        
        # 각 단어에 대해 키워드 매칭
        for word in words:
            if word in self.price_keywords:
                matched_keywords["price"].append(word)
            if word in self.institutional_keywords:
                matched_keywords["institutional"].append(word)
            if word in self.sentiment_keywords:
                matched_keywords["sentiment"].append(word)
            if word in self.technical_keywords:
                matched_keywords["technical"].append(word)
        
        return matched_keywords
    
    def _calculate_category_score(self, keywords: List[str], 
                                keyword_dict: Dict[str, float]) -> float:
        """카테고리별 점수 계산"""
        if not keywords:
            return 0.0
        
        # 키워드별 점수 합계
        total_score = sum(keyword_dict.get(keyword, 0.0) for keyword in keywords)
        
        # 키워드 개수로 정규화 (과도한 점수 방지)
        normalized_score = total_score / max(1, len(keywords) ** 0.5)
        
        # -1.0 ~ 1.0 범위로 클리핑
        return max(-1.0, min(1.0, normalized_score))
    
    def _calculate_confidence(self, category_scores: Dict[str, float], 
                            matched_keywords: Dict[str, List[str]]) -> float:
        """신뢰도 계산"""
        
        # 1. 매칭된 키워드 총 개수
        total_keywords = sum(len(keywords) for keywords in matched_keywords.values())
        keyword_factor = min(1.0, total_keywords / 5.0)  # 5개 이상이면 최대
        
        # 2. 카테고리 다양성 (여러 카테고리에서 매칭되면 신뢰도 증가)
        active_categories = sum(1 for keywords in matched_keywords.values() if keywords)
        diversity_factor = active_categories / len(self.category_weights)
        
        # 3. 점수 일관성 (모든 카테고리가 같은 방향이면 신뢰도 증가)
        non_zero_scores = [score for score in category_scores.values() if abs(score) > 0.1]
        if len(non_zero_scores) > 1:
            # 모든 점수가 같은 부호인지 확인
            same_direction = all(score > 0 for score in non_zero_scores) or \
                           all(score < 0 for score in non_zero_scores)
            consistency_factor = 1.0 if same_direction else 0.7
        else:
            consistency_factor = 0.8
        
        # 종합 신뢰도 계산
        confidence = (keyword_factor * 0.4 + 
                     diversity_factor * 0.3 + 
                     consistency_factor * 0.3)
        
        return max(0.1, min(1.0, confidence))
    
    def _determine_direction(self, final_score: float) -> SentimentDirection:
        """감정 방향 결정"""
        if final_score > 0.1:
            return SentimentDirection.BULLISH
        elif final_score < -0.1:
            return SentimentDirection.BEARISH
        else:
            return SentimentDirection.NEUTRAL
    
    def analyze(self, text: str) -> KeywordScore:
        """실시간 키워드 기반 감정 분석"""
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                return KeywordScore(
                    score=0.0,
                    confidence=0.0,
                    direction=SentimentDirection.NEUTRAL,
                    matched_keywords=[],
                    category_scores={},
                    processing_time=time.time() - start_time
                )
            
            # 1. 키워드 추출
            matched_keywords = self._extract_keywords(text)
            
            # 2. 카테고리별 점수 계산
            category_scores = {
                "price": self._calculate_category_score(
                    matched_keywords["price"], self.price_keywords
                ),
                "institutional": self._calculate_category_score(
                    matched_keywords["institutional"], self.institutional_keywords
                ),
                "sentiment": self._calculate_category_score(
                    matched_keywords["sentiment"], self.sentiment_keywords
                ),
                "technical": self._calculate_category_score(
                    matched_keywords["technical"], self.technical_keywords
                )
            }
            
            # 3. 가중 평균으로 최종 점수 계산
            final_score = sum(
                score * self.category_weights[category]
                for category, score in category_scores.items()
            )
            
            # 4. 신뢰도 계산
            confidence = self._calculate_confidence(category_scores, matched_keywords)
            
            # 5. 방향 결정
            direction = self._determine_direction(final_score)
            
            # 6. 매칭된 모든 키워드 수집
            all_matched_keywords = []
            for keywords in matched_keywords.values():
                all_matched_keywords.extend(keywords)
            
            processing_time = time.time() - start_time
            
            result = KeywordScore(
                score=round(final_score, 4),
                confidence=round(confidence, 4),
                direction=direction,
                matched_keywords=list(set(all_matched_keywords)),  # 중복 제거
                category_scores={k: round(v, 4) for k, v in category_scores.items()},
                processing_time=round(processing_time * 1000, 2)  # ms 단위
            )
            
            logger.debug(f"Keyword analysis: {direction.value} "
                        f"(score: {final_score:.3f}, confidence: {confidence:.3f}, "
                        f"time: {processing_time*1000:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}", exc_info=True)
            return KeywordScore(
                score=0.0,
                confidence=0.0,
                direction=SentimentDirection.NEUTRAL,
                matched_keywords=[],
                category_scores={},
                processing_time=time.time() - start_time
            )
    
    def analyze_batch(self, texts: List[str]) -> List[KeywordScore]:
        """배치 키워드 분석"""
        start_time = time.time()
        
        results = []
        for text in texts:
            results.append(self.analyze(text))
        
        total_time = time.time() - start_time
        logger.info(f"Batch keyword analysis completed: {len(texts)} texts "
                   f"in {total_time*1000:.1f}ms "
                   f"(avg: {total_time*1000/len(texts):.1f}ms per text)")
        
        return results
    
    def get_statistics(self) -> Dict[str, int]:
        """키워드 사전 통계"""
        return {
            "total_keywords": (len(self.price_keywords) + 
                             len(self.institutional_keywords) + 
                             len(self.sentiment_keywords) + 
                             len(self.technical_keywords)),
            "price_keywords": len(self.price_keywords),
            "institutional_keywords": len(self.institutional_keywords),
            "sentiment_keywords": len(self.sentiment_keywords),
            "technical_keywords": len(self.technical_keywords),
            "uptime_seconds": int(time.time() - self.start_time)
        }


# 테스트 코드
if __name__ == "__main__":
    import json
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 스코어러 초기화
    scorer = KeywordScorer()
    
    # 테스트 케이스
    test_cases = [
        "Bitcoin surges as institutional adoption grows",
        "Market crash amid regulatory crackdown fears", 
        "Bullish breakout above resistance level",
        "Fed approval for crypto ETF boosts sentiment",
        "Bearish trend continues with heavy selling pressure",
        "Neutral market conditions with mixed signals"
    ]
    
    print("\n=== 키워드 기반 감정 분석 테스트 ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"테스트 {i}: {text}")
        result = scorer.analyze(text)
        
        print(f"  점수: {result.score} ({result.direction.value})")
        print(f"  신뢰도: {result.confidence}")
        print(f"  처리시간: {result.processing_time}ms")
        print(f"  매칭 키워드: {result.matched_keywords}")
        print(f"  카테고리별: {result.category_scores}")
        print("-" * 60)
    
    # 배치 테스트
    print("\n=== 배치 분석 테스트 ===")
    batch_results = scorer.analyze_batch(test_cases)
    avg_time = sum(r.processing_time for r in batch_results) / len(batch_results)
    print(f"평균 처리시간: {avg_time:.2f}ms")
    
    # 통계
    print(f"\n=== 키워드 사전 통계 ===")
    stats = scorer.get_statistics()
    print(json.dumps(stats, indent=2))