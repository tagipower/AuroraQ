#!/usr/bin/env python3
"""
센티멘트 점수화 시스템
뉴스 센티멘트를 거래 신호로 변환
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .news_collector import NewsItem
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from ..utils.logger import get_logger

logger = get_logger("SentimentScorer")

class MarketSentiment(Enum):
    """시장 센티멘트 분류"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"

@dataclass
class MarketSentimentScore:
    """시장 센티멘트 점수"""
    overall_score: float  # -1.0 ~ 1.0
    sentiment_label: MarketSentiment
    confidence: float     # 0.0 ~ 1.0
    news_count: int
    bullish_ratio: float
    bearish_ratio: float
    sentiment_velocity: float  # 센티멘트 변화 속도
    key_themes: List[str]
    timestamp: datetime

class SentimentScorer:
    """센티멘트 점수화 시스템"""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.sentiment_history = []  # 과거 센티멘트 저장
        self.score_weights = {
            'recency': 0.4,      # 최신성 가중치
            'volume': 0.3,       # 뉴스 볼륨 가중치
            'credibility': 0.2,  # 신뢰도 가중치
            'relevance': 0.1     # 관련성 가중치
        }
    
    def calculate_market_sentiment(self, news_items: List[NewsItem], 
                                 asset_type: str = "crypto") -> MarketSentimentScore:
        """시장 센티멘트 계산"""
        
        if not news_items:
            return self._get_neutral_sentiment()
        
        # 1. 뉴스별 센티멘트 분석
        sentiment_results = []
        for news in news_items:
            try:
                result = self.analyzer.analyze_sentiment(
                    text=f"{news.title} {news.content}",
                    context=asset_type
                )
                sentiment_results.append((news, result))
            except Exception as e:
                logger.error(f"센티멘트 분석 실패: {e}")
                continue
        
        if not sentiment_results:
            return self._get_neutral_sentiment()
        
        # 2. 가중 평균 점수 계산
        weighted_scores = []
        total_weight = 0
        
        for news, sentiment in sentiment_results:
            weight = self._calculate_news_weight(news, sentiment)
            weighted_scores.append(sentiment.sentiment_score * weight)
            total_weight += weight
        
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        # 3. 센티멘트 분포 분석
        bullish_count = sum(1 for _, s in sentiment_results if s.sentiment_score > 0.1)
        bearish_count = sum(1 for _, s in sentiment_results if s.sentiment_score < -0.1)
        
        bullish_ratio = bullish_count / len(sentiment_results)
        bearish_ratio = bearish_count / len(sentiment_results)
        
        # 4. 센티멘트 레이블 결정
        sentiment_label = self._determine_sentiment_label(overall_score, bullish_ratio, bearish_ratio)
        
        # 5. 신뢰도 계산
        confidence = self._calculate_confidence(sentiment_results, overall_score)
        
        # 6. 센티멘트 변화 속도 계산
        velocity = self._calculate_sentiment_velocity(overall_score)
        
        # 7. 주요 테마 추출
        key_themes = self._extract_key_themes(sentiment_results)
        
        sentiment_score = MarketSentimentScore(
            overall_score=overall_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            news_count=len(sentiment_results),
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio,
            sentiment_velocity=velocity,
            key_themes=key_themes,
            timestamp=datetime.now()
        )
        
        # 히스토리 저장
        self.sentiment_history.append(sentiment_score)
        self._cleanup_history()
        
        logger.info(f"시장 센티멘트 계산 완료: {sentiment_label.value} "
                   f"(점수: {overall_score:.3f}, 신뢰도: {confidence:.3f})")
        
        return sentiment_score
    
    def _calculate_news_weight(self, news: NewsItem, sentiment: SentimentResult) -> float:
        """뉴스별 가중치 계산"""
        
        # 1. 최신성 가중치 (24시간 이내 = 1.0, 그 이후 지수 감소)
        time_diff = datetime.now() - news.timestamp
        hours_diff = time_diff.total_seconds() / 3600
        recency_weight = np.exp(-hours_diff / 24.0)  # 24시간 반감기
        
        # 2. 소스 신뢰도 가중치
        source_weights = {
            "Reuters": 1.0,
            "Yahoo Finance": 0.9,
            "CoinDesk": 0.9,
            "CoinTelegraph": 0.8,
            "Bloomberg": 1.0,
            "Wall Street Journal": 1.0
        }
        credibility_weight = source_weights.get(news.source, 0.5)
        
        # 3. 관련성 가중치 (키워드 기반)
        crypto_keywords = {"bitcoin", "crypto", "blockchain", "ethereum", "defi"}
        finance_keywords = {"stock", "market", "economy", "fed", "inflation"}
        
        relevance_weight = 0.5  # 기본값
        news_text = f"{news.title} {news.content}".lower()
        
        if any(keyword in news_text for keyword in crypto_keywords):
            relevance_weight = 1.0
        elif any(keyword in news_text for keyword in finance_keywords):
            relevance_weight = 0.8
        
        # 4. 센티멘트 신뢰도 가중치
        confidence_weight = sentiment.confidence
        
        # 종합 가중치 계산
        total_weight = (
            recency_weight * self.score_weights['recency'] +
            credibility_weight * self.score_weights['credibility'] +
            relevance_weight * self.score_weights['relevance'] +
            confidence_weight * 0.1  # 추가 보정
        )
        
        return max(0.1, min(2.0, total_weight))  # 0.1 ~ 2.0 범위로 제한
    
    def _determine_sentiment_label(self, score: float, bullish_ratio: float, bearish_ratio: float) -> MarketSentiment:
        """센티멘트 레이블 결정"""
        
        if score > 0.6 and bullish_ratio > 0.7:
            return MarketSentiment.VERY_BULLISH
        elif score > 0.2 and bullish_ratio > 0.5:
            return MarketSentiment.BULLISH
        elif score < -0.6 and bearish_ratio > 0.7:
            return MarketSentiment.VERY_BEARISH
        elif score < -0.2 and bearish_ratio > 0.5:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.NEUTRAL
    
    def _calculate_confidence(self, sentiment_results: List[tuple], overall_score: float) -> float:
        """신뢰도 계산"""
        
        if not sentiment_results:
            return 0.0
        
        # 1. 센티멘트 일관성 (표준편차 역수)
        scores = [result.sentiment_score for _, result in sentiment_results]
        consistency = 1.0 / (1.0 + np.std(scores))
        
        # 2. 뉴스 개수 (더 많은 뉴스 = 더 높은 신뢰도)
        volume_factor = min(1.0, len(sentiment_results) / 10.0)
        
        # 3. 개별 센티멘트 신뢰도 평균
        avg_confidence = np.mean([result.confidence for _, result in sentiment_results])
        
        # 4. 극단값 보정 (너무 극단적인 점수는 신뢰도 감소)
        extremity_penalty = 1.0 - min(0.3, abs(overall_score) * 0.3)
        
        total_confidence = consistency * 0.3 + volume_factor * 0.3 + avg_confidence * 0.3 + extremity_penalty * 0.1
        
        return max(0.1, min(1.0, total_confidence))
    
    def _calculate_sentiment_velocity(self, current_score: float) -> float:
        """센티멘트 변화 속도 계산"""
        
        if len(self.sentiment_history) < 2:
            return 0.0
        
        # 최근 3개 점수의 변화율 계산
        recent_scores = [s.overall_score for s in self.sentiment_history[-3:]]
        recent_scores.append(current_score)
        
        velocity = 0.0
        for i in range(1, len(recent_scores)):
            velocity += recent_scores[i] - recent_scores[i-1]
        
        return velocity / (len(recent_scores) - 1)
    
    def _extract_key_themes(self, sentiment_results: List[tuple]) -> List[str]:
        """주요 테마 추출"""
        
        theme_keywords = {}
        
        for news, sentiment in sentiment_results:
            for keyword in sentiment.keywords:
                if keyword not in theme_keywords:
                    theme_keywords[keyword] = 0
                theme_keywords[keyword] += abs(sentiment.sentiment_score)
        
        # 상위 5개 테마 반환
        sorted_themes = sorted(theme_keywords.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:5]]
    
    def _get_neutral_sentiment(self) -> MarketSentimentScore:
        """중립 센티멘트 반환"""
        return MarketSentimentScore(
            overall_score=0.0,
            sentiment_label=MarketSentiment.NEUTRAL,
            confidence=0.5,
            news_count=0,
            bullish_ratio=0.0,
            bearish_ratio=0.0,
            sentiment_velocity=0.0,
            key_themes=[],
            timestamp=datetime.now()
        )
    
    def _cleanup_history(self, days_to_keep: int = 7):
        """센티멘트 히스토리 정리"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        self.sentiment_history = [
            s for s in self.sentiment_history 
            if s.timestamp > cutoff_time
        ]
    
    def get_trading_signal(self, sentiment_score: MarketSentimentScore) -> Dict[str, Any]:
        """센티멘트 기반 거래 신호 생성"""
        
        signal_strength = abs(sentiment_score.overall_score) * sentiment_score.confidence
        
        if sentiment_score.sentiment_label == MarketSentiment.VERY_BULLISH:
            signal = "STRONG_BUY"
            confidence = min(0.9, signal_strength)
        elif sentiment_score.sentiment_label == MarketSentiment.BULLISH:
            signal = "BUY"
            confidence = min(0.7, signal_strength)
        elif sentiment_score.sentiment_label == MarketSentiment.VERY_BEARISH:
            signal = "STRONG_SELL"
            confidence = min(0.9, signal_strength)
        elif sentiment_score.sentiment_label == MarketSentiment.BEARISH:
            signal = "SELL"
            confidence = min(0.7, signal_strength)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "sentiment_score": sentiment_score.overall_score,
            "news_volume": sentiment_score.news_count,
            "key_themes": sentiment_score.key_themes,
            "timestamp": sentiment_score.timestamp
        }