# sentiment/sentiment_score_refiner.py

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from utils.logger import get_logger

logger = get_logger("SentimentScoreRefiner")

class SentimentScoreRefiner:
    """
    뉴스 텍스트에서 감정 점수를 추출하고 정제하는 클래스
    실시간 감정 분석을 위한 핵심 모듈
    """
    
    def __init__(self):
                    # 긍정/부정 키워드 사전
        self.positive_keywords = {
            # 가격 관련
            "surge": 3, "soar": 3, "rally": 2.5, "climb": 2, "gain": 2,
            "rise": 1.5, "up": 1, "increase": 1.5, "jump": 2.5,
            "bullish": 3, "moon": 3, "pump": 2.5, "breakout": 2.5,
            
            # 채택/승인 관련
            "approval": 3, "approved": 3, "approve": 3, "approves": 3,
            "adopt": 2.5, "adoption": 2.5, "accept": 2, "embrace": 2.5, 
            "integrate": 2, "implement": 2, "etf": 2,
            
            # 기술/발전 관련
            "breakthrough": 3, "innovation": 2.5, "upgrade": 2,
            "improvement": 2, "enhance": 2, "advance": 2,
            
            # 시장 감정
            "optimistic": 2.5, "confident": 2, "positive": 2,
            "strong": 2, "robust": 2, "healthy": 2, "stable": 1.5,
            "steady": 1.5, "resilient": 2,
            
            # 기관/정부 지원
            "support": 2, "backing": 2.5, "endorse": 2.5,
            "partnership": 2, "collaboration": 2, "alliance": 2
        }
        
        self.negative_keywords = {
            # 가격 관련
            "crash": -3, "plunge": -3, "collapse": -3, "dump": -2.5,
            "drop": -2, "fall": -2, "decline": -2, "down": -1,
            "bearish": -3, "slump": -2.5, "tumble": -2.5,
            
            # 규제/금지 관련
            "ban": -3, "prohibit": -3, "restrict": -2.5, "regulation": -2,
            "crackdown": -3, "illegal": -3, "sue": -2.5, "lawsuit": -2.5,
            
            # 보안/사고 관련
            "hack": -3, "hacked": -3, "breach": -3, "exploit": -3, "scam": -3,
            "fraud": -3, "theft": -3, "stolen": -3, "attack": -2.5, "vulnerability": -2.5,
            
            # 시장 감정
            "fear": -2.5, "panic": -3, "concern": -2, "worry": -2,
            "uncertain": -2, "volatile": -1.5, "risk": -1.5,
            
            # 실패/문제
            "fail": -2.5, "failure": -2.5, "problem": -2, "issue": -1.5,
            "delay": -2, "postpone": -2, "cancel": -2.5, "abandon": -3
        }
        
        # 강화 표현
        self.intensifiers = {
            "very": 1.3, "extremely": 1.5, "highly": 1.4,
            "significantly": 1.4, "massively": 1.5, "hugely": 1.5
        }
        
        # 약화 표현
        self.diminishers = {
            "slightly": 0.7, "somewhat": 0.8, "relatively": 0.8,
            "fairly": 0.9, "moderately": 0.85
        }
        
        # 감정 점수 변환 범위
        self.min_raw_score = -10
        self.max_raw_score = 10
        
    def analyze_text(self, text: str) -> float:
        """
        텍스트를 분석하여 0~1 사이의 감정 점수를 반환
        
        :param text: 분석할 뉴스 텍스트
        :return: 0~1 사이의 감정 점수 (0: 매우 부정, 0.5: 중립, 1: 매우 긍정)
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning("유효하지 않은 텍스트 입력")
                return 0.5
            
            # 전처리
            processed_text = self._preprocess_text(text)
            
            # 문장 단위 분석
            sentences = self._split_sentences(processed_text)
            sentence_scores = []
            
            for sentence in sentences:
                score = self._analyze_sentence(sentence)
                if score != 0:  # 중립 문장은 제외
                    sentence_scores.append(score)
            
            # 전체 점수 계산
            if not sentence_scores:
                return 0.5
            
            # 가중 평균 (최근 문장에 더 높은 가중치)
            weights = np.linspace(0.8, 1.2, len(sentence_scores))
            weighted_score = np.average(sentence_scores, weights=weights)
            
            # 0~1 범위로 정규화
            normalized_score = self._normalize_score(weighted_score)
            
            logger.info(f"감정 분석 완료 - Raw: {weighted_score:.2f}, Normalized: {normalized_score:.3f}")
            return round(normalized_score, 4)
            
        except Exception as e:
            logger.error(f"텍스트 분석 중 오류: {e}")
            return 0.5
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 소문자 변환
        text = text.lower()
        
        # 특수문자 제거 (구두점 유지)
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # 다중 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 간단한 문장 분리 (정교한 분리가 필요한 경우 NLTK 사용)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence(self, sentence: str) -> float:
        """단일 문장 분석"""
        words = sentence.split()
        score = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # 강화/약화 표현 확인
            modifier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                modifier = self.intensifiers[words[i-1]]
            elif i > 0 and words[i-1] in self.diminishers:
                modifier = self.diminishers[words[i-1]]
            
            # 긍정/부정 키워드 점수 계산
            if word in self.positive_keywords:
                score += self.positive_keywords[word] * modifier
            elif word in self.negative_keywords:
                score += self.negative_keywords[word] * modifier
            
            # 부정 표현 처리 (not, no, never 등)
            if word in ['not', 'no', 'never', 'none'] and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in self.positive_keywords:
                    score -= self.positive_keywords[next_word] * 0.8
                elif next_word in self.negative_keywords:
                    score -= self.negative_keywords[next_word] * 0.8  # 이중 부정
            
            i += 1
        
        return score
    
    def _normalize_score(self, raw_score: float) -> float:
        """원시 점수를 0~1 범위로 정규화"""
        # 시그모이드 함수 사용하여 부드러운 변환
        # raw_score가 0일 때 0.5가 되도록 조정
        normalized = 1 / (1 + np.exp(-raw_score / 3))
        
        # 추가 조정: 극단값 완화
        if normalized > 0.9:
            normalized = 0.9 + (normalized - 0.9) * 0.5
        elif normalized < 0.1:
            normalized = 0.1 - (0.1 - normalized) * 0.5
        
        return max(0.0, min(1.0, normalized))
    
    def get_sentiment_score(self, text: str) -> float:
        """
        외부 인터페이스: 텍스트에서 감정 점수 추출
        sentiment_router.py에서 호출하는 메인 함수
        """
        return self.analyze_text(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        여러 텍스트를 일괄 분석
        
        :param texts: 분석할 텍스트 리스트
        :return: (텍스트, 점수) 튜플 리스트
        """
        results = []
        for text in texts:
            score = self.analyze_text(text)
            results.append((text, score))
        
        return results
    
    def get_keyword_impact(self, text: str) -> Dict[str, float]:
        """
        텍스트에서 발견된 키워드와 그 영향도 반환 (디버깅용)
        
        :param text: 분석할 텍스트
        :return: {키워드: 영향도} 딕셔너리
        """
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        impact = {}
        for word in words:
            if word in self.positive_keywords:
                impact[word] = self.positive_keywords[word]
            elif word in self.negative_keywords:
                impact[word] = self.negative_keywords[word]
        
        return impact


# 전역 인스턴스 생성
_refiner_instance = SentimentScoreRefiner()

def get_sentiment_score(text: str) -> float:
    """
    텍스트의 감정 점수를 반환하는 전역 함수
    sentiment_router.py에서 import하여 사용
    
    :param text: 분석할 뉴스 텍스트
    :return: 0~1 사이의 감정 점수
    """
    return _refiner_instance.get_sentiment_score(text)


# 테스트 코드
if __name__ == "__main__":
    # 테스트 예제
    test_texts = [
        "Bitcoin surges to new all-time high as institutional adoption accelerates",
        "Cryptocurrency market crashes amid regulatory crackdown fears",
        "SEC approves Bitcoin ETF after years of deliberation",
        "Major exchange hacked, millions in crypto stolen",
        "Slight increase in trading volume observed",
        "Market remains relatively stable despite global uncertainty"
    ]
    
    refiner = SentimentScoreRefiner()
    
    print("=== 감정 점수 분석 결과 ===")
    for text in test_texts:
        score = refiner.analyze_text(text)
        sentiment = "긍정" if score > 0.6 else "부정" if score < 0.4 else "중립"
        print(f"\n텍스트: {text}")
        print(f"점수: {score:.4f} ({sentiment})")
        
        # 키워드 영향도 출력
        impacts = refiner.get_keyword_impact(text)
        if impacts:
            print("주요 키워드:", impacts)