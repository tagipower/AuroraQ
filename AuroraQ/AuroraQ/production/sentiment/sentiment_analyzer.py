# sentiment/sentiment_analyzer.py - 개선된 버전

import logging
from transformers import pipeline, AutoTokenizer
import re
from typing import List, Dict, Optional, Union
from functools import lru_cache
import torch
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """감정 레이블 열거형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """감정 분석 결과 데이터클래스"""
    sentiment_score: float
    label: SentimentLabel
    confidence: float
    keywords: List[str]
    scenario_tag: str

class SentimentAnalyzer:
    """개선된 감정 분석기"""
    
    # 클래스 변수로 블랙리스트 정의
    KEYWORD_BLACKLIST = {
        "markets", "price", "today", "investor", "report", "news",
        "stock", "share", "trade", "market", "trading", "financial"
    }
    
    # 시나리오 매핑
    SCENARIO_MAPPINGS = {
        "positive": {
            "keywords": {
                ("approval", "growth"): "낙관 기대감",
                ("inflation",): "인플레이션 완화",
                ("rally", "surge"): "강세 랠리",
                ("recovery",): "경기 회복"
            },
            "default": "시장 낙관"
        },
        "negative": {
            "keywords": {
                ("crash", "fear"): "공포 상승",
                ("rate", "inflation"): "금리/물가 우려",
                ("recession",): "경기 침체 우려",
                ("default", "bankruptcy"): "신용 위기"
            },
            "default": "시장 불안"
        }
    }
    
    def __init__(self, model_name: str = "ProsusAI/finbert", 
                 device: Optional[str] = None,
                 max_length: int = 512):
        """
        Args:
            model_name: 사용할 모델 이름
            device: 연산 장치 (None이면 자동 선택)
            max_length: 최대 토큰 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            logger.info(f"Loading model: {model_name} on {self.device}")
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=max_length
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Model loaded successfully ✅")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.analyzer = None
            self.tokenizer = None
            raise

    def normalize_score(self, label: str, score: float) -> float:
        """
        감정 점수 정규화 (개선된 버전)
        
        Returns:
            -1.0 ~ 1.0 범위의 정규화된 점수
        """
        # 더 세밀한 정규화 로직
        if label == "positive":
            if score >= 0.8:
                return min(1.0, (score - 0.5) * 2)
            elif score >= 0.6:
                return (score - 0.5) * 1.5
            else:
                return max(0.0, (score - 0.5))
        elif label == "negative":
            if score >= 0.8:
                return max(-1.0, (0.5 - score) * 2)
            elif score >= 0.6:
                return (0.5 - score) * 1.5
            else:
                return min(0.0, (0.5 - score))
        else:  # neutral
            return 0.0

    @lru_cache(maxsize=128)
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        키워드 추출 (캐싱 적용)
        
        Args:
            text: 분석할 텍스트
            top_n: 추출할 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        # 소문자 변환 및 정규화
        text_lower = text.lower()
        
        # 단어 추출 (숫자 포함, 특수문자 제거)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        
        # 블랙리스트 필터링 및 중복 제거
        filtered_words = []
        seen = set()
        
        for word in words:
            if word not in self.KEYWORD_BLACKLIST and word not in seen:
                filtered_words.append(word)
                seen.add(word)
                
        # 빈도 기반 정렬 (간단한 구현)
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # 상위 N개 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]

    def tag_scenario(self, label: str, keywords: List[str]) -> str:
        """
        시나리오 태깅 (개선된 버전)
        """
        if label not in self.SCENARIO_MAPPINGS:
            return "중립 흐름"
            
        scenario_map = self.SCENARIO_MAPPINGS[label]
        
        # 키워드 매칭
        for keyword_tuple, scenario in scenario_map["keywords"].items():
            if any(kw in keywords for kw in keyword_tuple):
                return scenario
                
        return scenario_map["default"]

    def _validate_input(self, data: Union[dict, str]) -> str:
        """입력 데이터 검증 및 텍스트 추출"""
        if isinstance(data, str):
            text = data.strip()
        elif isinstance(data, dict):
            title = data.get("title", "").strip()
            snippet = data.get("snippet", "").strip()
            text = f"{title} {snippet}".strip()
        else:
            raise ValueError(f"Invalid input type: {type(data)}")
            
        if not text:
            raise ValueError("Empty text provided")
            
        return text

    def analyze(self, article: Union[dict, str]) -> float:
        """
        감정 점수 분석 (기본)
        
        Args:
            article: 분석할 기사 (dict 또는 str)
            
        Returns:
            정규화된 감정 점수 (-1.0 ~ 1.0)
        """
        try:
            text = self._validate_input(article)
            
            if not self.analyzer:
                logger.warning("Analyzer not initialized, returning neutral score")
                return 0.0
                
            # 토큰 수 체크
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
                if len(tokens) > self.max_length * 0.9:
                    logger.warning(f"Text truncated: {len(tokens)} tokens")
            
            result = self.analyzer(text)[0]
            label = result["label"].lower()
            score = float(result["score"])
            
            normalized_score = self.normalize_score(label, score)
            
            logger.debug(f"Analysis result: {label} ({score:.4f}) → {normalized_score:.4f}")
            return normalized_score
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return 0.0

    def analyze_text(self, text: str) -> float:
        """텍스트 감정 분석 (analyze 메서드 래퍼)"""
        return self.analyze(text)

    def analyze_detailed(self, article: Union[dict, str]) -> SentimentResult:
        """
        상세 감정 분석
        
        Returns:
            SentimentResult 객체
        """
        try:
            text = self._validate_input(article)
            
            if not self.analyzer:
                logger.warning("Analyzer not initialized")
                return self._get_default_result()
            
            result = self.analyzer(text)[0]
            label = result["label"].lower()
            confidence = float(result["score"])
            
            # 정규화된 점수 계산
            sentiment_score = self.normalize_score(label, confidence)
            
            # 키워드 추출
            keywords = self.extract_keywords(text)
            
            # 시나리오 태깅
            scenario_tag = self.tag_scenario(label, keywords)
            
            # Enum 변환
            label_enum = SentimentLabel(label) if label in [e.value for e in SentimentLabel] else SentimentLabel.NEUTRAL
            
            return SentimentResult(
                sentiment_score=sentiment_score,
                label=label_enum,
                confidence=round(confidence, 4),
                keywords=keywords,
                scenario_tag=scenario_tag
            )
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}", exc_info=True)
            return self._get_default_result()
    
    def _get_default_result(self) -> SentimentResult:
        """기본 결과 반환"""
        return SentimentResult(
            sentiment_score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            keywords=[],
            scenario_tag="분석 실패"
        )

    def analyze_batch(self, articles: List[Union[dict, str]], 
                     batch_size: int = 16) -> List[float]:
        """
        배치 분석 (성능 최적화)
        
        Args:
            articles: 분석할 기사 리스트
            batch_size: 배치 크기
            
        Returns:
            감정 점수 리스트
        """
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # 텍스트 추출
            texts = []
            for article in batch:
                try:
                    text = self._validate_input(article)
                    texts.append(text)
                except Exception as e:
                    logger.error(f"Invalid article in batch: {e}")
                    texts.append("")
            
            # 배치 분석
            if self.analyzer and texts:
                try:
                    batch_results = self.analyzer(texts)
                    for j, result in enumerate(batch_results):
                        if texts[j]:  # 유효한 텍스트만 처리
                            label = result["label"].lower()
                            score = float(result["score"])
                            normalized = self.normalize_score(label, score)
                            results.append(normalized)
                        else:
                            results.append(0.0)
                except Exception as e:
                    logger.error(f"Batch analysis failed: {e}")
                    results.extend([0.0] * len(texts))
            else:
                results.extend([0.0] * len(texts))
                
        return results


# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 분석기 초기화
    analyzer = SentimentAnalyzer()
    
    # 테스트 케이스
    test_cases = [
        {
            "title": "Bitcoin surges as ETF approval nears",
            "snippet": "Investors cheer the regulatory development as markets react positively."
        },
        "Markets are extremely volatile due to economic uncertainty.",
        {
            "title": "Fed raises interest rates amid inflation concerns",
            "snippet": "Central bank takes aggressive stance to combat rising prices."
        }
    ]
    
    print("\n=== 감정 분석 테스트 ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 {i}:")
        print(f"입력: {test_case}")
        
        # 기본 분석
        score = analyzer.analyze(test_case)
        print(f"감정 점수: {score:.4f}")
        
        # 상세 분석
        detailed = analyzer.analyze_detailed(test_case)
        print(f"상세 결과: {detailed}")
        print("-" * 50)
    
    # 배치 분석 테스트
    print("\n=== 배치 분석 테스트 ===")
    batch_scores = analyzer.analyze_batch(test_cases)
    print(f"배치 결과: {batch_scores}")