# SharedCore/sentiment_engine/analyzers/finbert_analyzer.py

import logging
from transformers import pipeline, AutoTokenizer
import re
from typing import List, Dict, Optional, Union
from functools import lru_cache
import torch
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import numpy as np

# Import news article structure
from ...data_collection.base_collector import NewsArticle

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
    timestamp: datetime = None

class FinBERTAnalyzer:
    """FinBERT 기반 고급 감정 분석기"""
    
    # 키워드 블랙리스트
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
                ("recovery",): "경기 회복",
                ("etf", "approval"): "ETF 승인 기대",
                ("adoption", "institutional"): "기관 채택"
            },
            "default": "시장 낙관"
        },
        "negative": {
            "keywords": {
                ("crash", "fear"): "공포 상승",
                ("rate", "inflation"): "금리/물가 우려",
                ("recession",): "경기 침체 우려",
                ("default", "bankruptcy"): "신용 위기",
                ("hack", "security"): "보안 우려",
                ("regulation", "ban"): "규제 리스크"
            },
            "default": "시장 불안"
        }
    }
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
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
        
        # 모델 로딩 지연 초기화
        self.analyzer = None
        self.tokenizer = None
        self._initialized = False
        
        logger.info(f"FinBERT Analyzer configured: {model_name} on {self.device}")
    
    async def initialize(self):
        """비동기 모델 초기화"""
        if self._initialized:
            return
            
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            # CPU에서 모델 로딩 (메모리 절약)
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=self.max_length
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self._initialized = True
            logger.info("FinBERT model loaded successfully ✅")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            # 폴백: 기본 감정 분석 모드
            self.analyzer = None
            self.tokenizer = None
            raise
    
    def normalize_score(self, label: str, score: float) -> float:
        """
        감정 점수를 0-1 범위로 정규화
        
        Returns:
            0.0 ~ 1.0 범위의 정규화된 점수
        """
        if label.lower() == "positive":
            # Positive: 0.5 ~ 1.0
            if score >= 0.8:
                return min(1.0, 0.5 + (score - 0.5))
            elif score >= 0.6:
                return 0.5 + (score - 0.5) * 0.75
            else:
                return max(0.5, 0.5 + (score - 0.5) * 0.5)
                
        elif label.lower() == "negative":
            # Negative: 0.0 ~ 0.5  
            if score >= 0.8:
                return max(0.0, 0.5 - (score - 0.5))
            elif score >= 0.6:
                return 0.5 - (score - 0.5) * 0.75
            else:
                return min(0.5, 0.5 - (score - 0.5) * 0.5)
        else:
            # Neutral: around 0.5
            return 0.5
    
    @lru_cache(maxsize=256)
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
        
        # 단어 추출 (3글자 이상)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        
        # 블랙리스트 필터링 및 중복 제거
        filtered_words = []
        seen = set()
        
        for word in words:
            if word not in self.KEYWORD_BLACKLIST and word not in seen:
                filtered_words.append(word)
                seen.add(word)
        
        # 빈도 기반 정렬
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 상위 N개 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    
    def tag_scenario(self, label: str, keywords: List[str]) -> str:
        """
        시나리오 태깅
        """
        if label not in self.SCENARIO_MAPPINGS:
            return "중립 흐름"
        
        scenario_map = self.SCENARIO_MAPPINGS[label]
        
        # 키워드 매칭
        for keyword_tuple, scenario in scenario_map["keywords"].items():
            if any(kw in keywords for kw in keyword_tuple):
                return scenario
        
        return scenario_map["default"]
    
    def _validate_input(self, data: Union[dict, str, NewsArticle]) -> str:
        """입력 데이터 검증 및 텍스트 추출"""
        if isinstance(data, str):
            text = data.strip()
        elif isinstance(data, NewsArticle):
            text = f"{data.title} {data.snippet}".strip()
        elif isinstance(data, dict):
            title = data.get("title", "").strip()
            snippet = data.get("snippet", "").strip()
            text = f"{title} {snippet}".strip()
        else:
            raise ValueError(f"Invalid input type: {type(data)}")
        
        if not text:
            raise ValueError("Empty text provided")
        
        return text
    
    async def analyze(self, article: Union[dict, str, NewsArticle]) -> float:
        """
        기본 감정 점수 분석
        
        Args:
            article: 분석할 기사 (dict, str 또는 NewsArticle)
            
        Returns:
            정규화된 감정 점수 (0.0 ~ 1.0)
        """
        try:
            # 초기화 확인
            if not self._initialized:
                await self.initialize()
            
            if not self.analyzer:
                logger.warning("Analyzer not initialized, returning neutral score")
                return 0.5
            
            text = self._validate_input(article)
            
            # 토큰 수 체크
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
                if len(tokens) > self.max_length * 0.9:
                    logger.warning(f"Text truncated: {len(tokens)} tokens")
            
            # 비동기 실행을 위해 executor 사용
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.analyzer, text)
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            label = result["label"].lower()
            score = float(result["score"])
            
            normalized_score = self.normalize_score(label, score)
            
            logger.debug(f"Analysis result: {label} ({score:.4f}) → {normalized_score:.4f}")
            return normalized_score
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return 0.5
    
    async def analyze_detailed(self, article: Union[dict, str, NewsArticle]) -> SentimentResult:
        """
        상세 감정 분석
        
        Returns:
            SentimentResult 객체
        """
        try:
            # 초기화 확인
            if not self._initialized:
                await self.initialize()
            
            if not self.analyzer:
                logger.warning("Analyzer not initialized")
                return self._get_default_result()
            
            text = self._validate_input(article)
            
            # 비동기 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.analyzer, text)
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
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
                scenario_tag=scenario_tag,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}", exc_info=True)
            return self._get_default_result()
    
    def _get_default_result(self) -> SentimentResult:
        """기본 결과 반환"""
        return SentimentResult(
            sentiment_score=0.5,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            keywords=[],
            scenario_tag="분석 실패",
            timestamp=datetime.utcnow()
        )
    
    async def analyze_batch(self, articles: List[Union[dict, str, NewsArticle]], 
                           batch_size: int = 16) -> List[float]:
        """
        배치 분석 (성능 최적화)
        
        Args:
            articles: 분석할 기사 리스트
            batch_size: 배치 크기
            
        Returns:
            감정 점수 리스트
        """
        try:
            # 초기화 확인
            if not self._initialized:
                await self.initialize()
            
            if not self.analyzer:
                return [0.5] * len(articles)
            
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
                if texts:
                    try:
                        # 비동기 실행
                        loop = asyncio.get_event_loop()
                        batch_results = await loop.run_in_executor(None, self.analyzer, texts)
                        
                        for j, result in enumerate(batch_results):
                            if texts[j]:  # 유효한 텍스트만 처리
                                label = result["label"].lower()
                                score = float(result["score"])
                                normalized = self.normalize_score(label, score)
                                results.append(normalized)
                            else:
                                results.append(0.5)
                    except Exception as e:
                        logger.error(f"Batch analysis failed: {e}")
                        results.extend([0.5] * len(texts))
                else:
                    results.extend([0.5] * len(texts))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return [0.5] * len(articles)
    
    async def close(self):
        """리소스 정리"""
        if hasattr(self, 'analyzer') and self.analyzer:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("FinBERT Analyzer resources cleaned up")


# 전역 인스턴스 관리 (싱글톤 패턴)
_global_analyzer: Optional[FinBERTAnalyzer] = None

async def get_finbert_analyzer() -> FinBERTAnalyzer:
    """전역 FinBERT 분석기 인스턴스 반환"""
    global _global_analyzer
    
    if _global_analyzer is None:
        _global_analyzer = FinBERTAnalyzer()
        await _global_analyzer.initialize()
    
    return _global_analyzer


# 기존 sentiment_score_refiner와의 호환성 함수
async def get_sentiment_score(text: str) -> float:
    """
    기존 sentiment_score_refiner의 get_sentiment_score 함수와 호환
    """
    analyzer = await get_finbert_analyzer()
    return await analyzer.analyze(text)


if __name__ == "__main__":
    import asyncio
    
    async def test_analyzer():
        """테스트 실행"""
        analyzer = FinBERTAnalyzer()
        await analyzer.initialize()
        
        test_cases = [
            "Bitcoin surges as ETF approval nears, investors cheer regulatory development",
            "Markets extremely volatile due to economic uncertainty and recession fears", 
            "Cryptocurrency exchange reports security breach, funds at risk",
            "Federal Reserve maintains neutral stance on digital assets"
        ]
        
        print("=== FinBERT Analyzer Test ===\n")
        
        for i, text in enumerate(test_cases, 1):
            print(f"Test {i}: {text[:50]}...")
            
            # 기본 분석
            score = await analyzer.analyze(text)
            print(f"  Score: {score:.4f}")
            
            # 상세 분석
            detailed = await analyzer.analyze_detailed(text)
            print(f"  Label: {detailed.label.value}")
            print(f"  Confidence: {detailed.confidence:.4f}")
            print(f"  Keywords: {detailed.keywords}")
            print(f"  Scenario: {detailed.scenario_tag}")
            print("-" * 50)
        
        # 배치 분석 테스트
        print("\n=== Batch Analysis ===")
        batch_scores = await analyzer.analyze_batch(test_cases)
        for i, score in enumerate(batch_scores):
            print(f"Article {i+1}: {score:.4f}")
        
        await analyzer.close()
    
    # 테스트 실행
    asyncio.run(test_analyzer())