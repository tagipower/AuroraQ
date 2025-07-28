#!/usr/bin/env python3
"""
Bitcoin Futures Market News Filter
비트코인 선물시장 반응 예측을 위한 특화 뉴스 필터링 시스템
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .base_collector import NewsArticle, NewsCategory


class MarketImpactLevel(Enum):
    """시장 영향도 레벨"""
    CRITICAL = "critical"      # 즉시 시장 반응 예상
    HIGH = "high"             # 높은 확률로 시장 영향
    MEDIUM = "medium"         # 중간 정도 영향
    LOW = "low"               # 제한적 영향
    MINIMAL = "minimal"       # 최소 영향


class EventType(Enum):
    """이벤트 유형"""
    REGULATORY = "regulatory"           # 규제 관련
    MONETARY_POLICY = "monetary_policy" # 통화정책
    INSTITUTIONAL = "institutional"     # 기관 투자
    TECHNICAL = "technical"            # 기술적 이슈
    MACRO_ECONOMIC = "macro_economic"  # 거시경제
    CELEBRITY = "celebrity"            # 유명인 발언
    MARKET_STRUCTURE = "market_structure" # 시장 구조 변화
    GEOPOLITICAL = "geopolitical"      # 지정학적 위험


@dataclass
class MarketRelevanceScore:
    """시장 관련성 점수"""
    impact_level: MarketImpactLevel
    event_type: EventType
    urgency_score: float  # 0-1 (긴급성)
    influence_score: float  # 0-1 (영향력)
    sentiment_multiplier: float  # -2 to +2 (감정 승수)
    confidence: float  # 0-1 (신뢰도)
    time_sensitivity: int  # minutes (시간 민감성)
    
    @property
    def composite_score(self) -> float:
        """종합 점수 계산"""
        base_score = (self.urgency_score + self.influence_score) / 2
        sentiment_adjusted = base_score * (1 + self.sentiment_multiplier * 0.3)
        return min(1.0, max(0.0, sentiment_adjusted * self.confidence))


class BitcoinFuturesNewsFilter:
    """비트코인 선물시장 특화 뉴스 필터"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 고영향 키워드 매트릭스
        self.impact_keywords = self._build_impact_keywords()
        
        # 주요 인물 데이터베이스
        self.key_figures = self._build_key_figures_db()
        
        # 기관 투자자 데이터베이스  
        self.institutions = self._build_institutions_db()
        
        # 규제 기관 및 정부 기관
        self.regulatory_bodies = self._build_regulatory_db()
        
        # 경제 지표 및 이벤트
        self.economic_indicators = self._build_economic_indicators()
        
    def _build_impact_keywords(self) -> Dict[MarketImpactLevel, Dict[str, List[str]]]:
        """영향도별 키워드 매트릭스 구축"""
        return {
            MarketImpactLevel.CRITICAL: {
                "regulatory": [
                    "bitcoin ban", "crypto ban", "emergency regulation", 
                    "trading halt", "exchange shutdown", "sec lawsuit",
                    "criminal charges", "ponzi scheme", "fraud investigation"
                ],
                "monetary": [
                    "fed rate hike", "emergency rate cut", "quantitative easing",
                    "dollar collapse", "currency crisis", "hyperinflation"
                ],
                "technical": [
                    "bitcoin hack", "blockchain attack", "51% attack",
                    "exchange hack", "wallet breach", "smart contract exploit"
                ],
                "institutional": [
                    "tesla sells bitcoin", "microstrategy sells", "grayscale liquidation",
                    "etf rejection", "etf approval", "institutional exodus"
                ]
            },
            MarketImpactLevel.HIGH: {
                "regulatory": [
                    "sec approval", "cftc guidance", "treasury statement",
                    "central bank announcement", "g20 decision", "fatf recommendation"
                ],
                "monetary": [
                    "interest rate decision", "inflation report", "unemployment data",
                    "gdp report", "fomc minutes", "jackson hole"
                ],
                "institutional": [
                    "institutional adoption", "corporate treasury", "pension fund",
                    "sovereign wealth fund", "central bank purchase", "etf inflow"
                ],
                "geopolitical": [
                    "china policy", "us sanctions", "war impact", "trade war",
                    "energy crisis", "political instability"
                ]
            },
            MarketImpactLevel.MEDIUM: {
                "market": [
                    "futures expiry", "options expiry", "cme volume",
                    "funding rate", "whale movement", "institutional flow"
                ],
                "technical": [
                    "network upgrade", "mining difficulty", "hash rate",
                    "developer activity", "protocol change"
                ],
                "adoption": [
                    "payment adoption", "merchant acceptance", "country adoption",
                    "banking integration", "fintech partnership"
                ]
            }
        }
    
    def _build_key_figures_db(self) -> Dict[str, Dict[str, Any]]:
        """주요 인물 데이터베이스"""
        return {
            # Fed 관련
            "jerome powell": {
                "role": "Fed Chairman", 
                "impact": MarketImpactLevel.CRITICAL,
                "keywords": ["jerome powell", "fed chairman", "federal reserve chair"]
            },
            "janet yellen": {
                "role": "Treasury Secretary",
                "impact": MarketImpactLevel.HIGH,
                "keywords": ["janet yellen", "treasury secretary"]
            },
            
            # 정치인
            "donald trump": {
                "role": "US President",
                "impact": MarketImpactLevel.HIGH,
                "keywords": ["president trump", "donald trump", "trump administration", "white house"]
            },
            
            # 비트코인 관련 CEO들
            "elon musk": {
                "role": "Tesla/SpaceX CEO",
                "impact": MarketImpactLevel.HIGH,
                "keywords": ["elon musk", "tesla ceo", "spacex"]
            },
            "michael saylor": {
                "role": "MicroStrategy Chairman",
                "impact": MarketImpactLevel.HIGH,
                "keywords": ["michael saylor", "microstrategy", "saylor"]
            },
            "larry fink": {
                "role": "BlackRock CEO",
                "impact": MarketImpactLevel.HIGH,
                "keywords": ["larry fink", "blackrock ceo"]
            },
            
            # 거래소 CEO들
            "brian armstrong": {
                "role": "Coinbase CEO",
                "impact": MarketImpactLevel.MEDIUM,
                "keywords": ["brian armstrong", "coinbase ceo"]
            },
            "changpeng zhao": {
                "role": "Binance CEO",
                "impact": MarketImpactLevel.MEDIUM,
                "keywords": ["changpeng zhao", "cz", "binance ceo"]
            },
            
            # 투자 거물들
            "warren buffett": {
                "role": "Berkshire Hathaway CEO",
                "impact": MarketImpactLevel.MEDIUM,
                "keywords": ["warren buffett", "berkshire hathaway"]
            },
            "ray dalio": {
                "role": "Bridgewater Associates",
                "impact": MarketImpactLevel.MEDIUM,
                "keywords": ["ray dalio", "bridgewater"]
            }
        }
    
    def _build_institutions_db(self) -> Dict[str, Dict[str, Any]]:
        """기관 투자자 데이터베이스"""
        return {
            "microstrategy": {"impact": MarketImpactLevel.CRITICAL, "type": "corporate"},
            "tesla": {"impact": MarketImpactLevel.CRITICAL, "type": "corporate"},
            "blackrock": {"impact": MarketImpactLevel.HIGH, "type": "asset_manager"},
            "grayscale": {"impact": MarketImpactLevel.HIGH, "type": "crypto_fund"},
            "ark invest": {"impact": MarketImpactLevel.MEDIUM, "type": "asset_manager"},
            "coinbase": {"impact": MarketImpactLevel.HIGH, "type": "exchange"},
            "binance": {"impact": MarketImpactLevel.HIGH, "type": "exchange"},
            "goldman sachs": {"impact": MarketImpactLevel.HIGH, "type": "investment_bank"},
            "jpmorgan": {"impact": MarketImpactLevel.HIGH, "type": "investment_bank"},
            "morgan stanley": {"impact": MarketImpactLevel.MEDIUM, "type": "investment_bank"}
        }
    
    def _build_regulatory_db(self) -> Dict[str, Dict[str, Any]]:
        """규제 기관 데이터베이스"""
        return {
            "sec": {"impact": MarketImpactLevel.CRITICAL, "region": "US"},
            "cftc": {"impact": MarketImpactLevel.HIGH, "region": "US"},
            "federal reserve": {"impact": MarketImpactLevel.CRITICAL, "region": "US"},
            "treasury": {"impact": MarketImpactLevel.HIGH, "region": "US"},
            "occ": {"impact": MarketImpactLevel.MEDIUM, "region": "US"},
            "ecb": {"impact": MarketImpactLevel.HIGH, "region": "EU"},
            "boe": {"impact": MarketImpactLevel.MEDIUM, "region": "UK"},
            "pboc": {"impact": MarketImpactLevel.HIGH, "region": "China"},
            "bis": {"impact": MarketImpactLevel.HIGH, "region": "Global"}
        }
    
    def _build_economic_indicators(self) -> Dict[str, Dict[str, Any]]:
        """경제 지표 데이터베이스"""
        return {
            "cpi": {
                "name": "Consumer Price Index",
                "impact": MarketImpactLevel.HIGH,
                "frequency": "monthly",
                "keywords": ["cpi", "inflation rate", "consumer price"]
            },
            "ppi": {
                "name": "Producer Price Index", 
                "impact": MarketImpactLevel.MEDIUM,
                "frequency": "monthly",
                "keywords": ["ppi", "producer price"]
            },
            "nonfarm payrolls": {
                "name": "Non-Farm Payrolls",
                "impact": MarketImpactLevel.HIGH,
                "frequency": "monthly", 
                "keywords": ["nonfarm payrolls", "unemployment rate", "jobs report"]
            },
            "gdp": {
                "name": "Gross Domestic Product",
                "impact": MarketImpactLevel.HIGH,
                "frequency": "quarterly",
                "keywords": ["gdp", "economic growth", "gross domestic"]
            },
            "fomc": {
                "name": "Federal Open Market Committee",
                "impact": MarketImpactLevel.CRITICAL,
                "frequency": "irregular",
                "keywords": ["fomc", "fed meeting", "interest rate decision"]
            }
        }
    
    def analyze_article_relevance(self, article: NewsArticle) -> MarketRelevanceScore:
        """기사의 시장 관련성 분석"""
        
        # 텍스트 준비
        full_text = f"{article.title} {article.summary} {article.content}".lower()
        
        # 각 요소별 점수 계산
        impact_level = self._calculate_impact_level(full_text)
        event_type = self._identify_event_type(full_text)
        urgency = self._calculate_urgency_score(article, full_text)
        influence = self._calculate_influence_score(full_text)
        sentiment_mult = self._calculate_sentiment_multiplier(article)
        confidence = self._calculate_confidence_score(article, full_text)
        time_sensitivity = self._calculate_time_sensitivity(event_type, impact_level)
        
        return MarketRelevanceScore(
            impact_level=impact_level,
            event_type=event_type,
            urgency_score=urgency,
            influence_score=influence,
            sentiment_multiplier=sentiment_mult,
            confidence=confidence,
            time_sensitivity=time_sensitivity
        )
    
    def _calculate_impact_level(self, text: str) -> MarketImpactLevel:
        """영향도 레벨 계산"""
        for level, categories in self.impact_keywords.items():
            for category, keywords in categories.items():
                if any(keyword in text for keyword in keywords):
                    return level
        return MarketImpactLevel.LOW
    
    def _identify_event_type(self, text: str) -> EventType:
        """이벤트 유형 식별"""
        type_keywords = {
            EventType.REGULATORY: ["regulation", "sec", "cftc", "ban", "legal", "compliance"],
            EventType.MONETARY_POLICY: ["fed", "interest rate", "monetary policy", "inflation"],
            EventType.INSTITUTIONAL: ["institutional", "corporate", "fund", "investment"],
            EventType.TECHNICAL: ["blockchain", "network", "protocol", "upgrade", "hack"],
            EventType.MACRO_ECONOMIC: ["gdp", "unemployment", "economic", "recession"],
            EventType.CELEBRITY: ["musk", "powell", "buffett", "saylor", "ceo statement"],
            EventType.MARKET_STRUCTURE: ["futures", "etf", "derivatives", "trading"],
            EventType.GEOPOLITICAL: ["war", "sanctions", "political", "china", "russia"]
        }
        
        for event_type, keywords in type_keywords.items():
            if any(keyword in text for keyword in keywords):
                return event_type
        
        return EventType.TECHNICAL
    
    def _calculate_urgency_score(self, article: NewsArticle, text: str) -> float:
        """긴급성 점수 계산"""
        urgency_indicators = [
            "breaking", "urgent", "alert", "emergency", "immediate",
            "flash", "just in", "developing", "live", "now"
        ]
        
        # 발행 시간 기반 점수 (최근일수록 높음)
        time_diff = datetime.now() - article.published_date
        time_score = max(0, 1 - time_diff.total_seconds() / (24 * 3600))  # 24시간 기준
        
        # 긴급성 키워드 점수
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in text)
        keyword_score = min(1.0, urgency_count / 3)
        
        return (time_score * 0.6 + keyword_score * 0.4)
    
    def _calculate_influence_score(self, text: str) -> float:
        """영향력 점수 계산"""
        influence_score = 0.0
        
        # 주요 인물 언급 점수
        for person, data in self.key_figures.items():
            if any(keyword in text for keyword in data["keywords"]):
                if data["impact"] == MarketImpactLevel.CRITICAL:
                    influence_score += 0.4
                elif data["impact"] == MarketImpactLevel.HIGH:
                    influence_score += 0.3
                else:
                    influence_score += 0.2
        
        # 기관 언급 점수
        for institution, data in self.institutions.items():
            if institution in text:
                if data["impact"] == MarketImpactLevel.CRITICAL:
                    influence_score += 0.3
                elif data["impact"] == MarketImpactLevel.HIGH:
                    influence_score += 0.2
                else:
                    influence_score += 0.1
        
        # 규제 기관 언급 점수
        for regulator, data in self.regulatory_bodies.items():
            if regulator in text:
                if data["impact"] == MarketImpactLevel.CRITICAL:
                    influence_score += 0.3
                elif data["impact"] == MarketImpactLevel.HIGH:
                    influence_score += 0.2
                else:
                    influence_score += 0.1
        
        return min(1.0, influence_score)
    
    def _calculate_sentiment_multiplier(self, article: NewsArticle) -> float:
        """감정 승수 계산"""
        if article.sentiment_score is None:
            return 0.0
        
        # 기본 감정 점수를 -2 to +2 범위로 확장
        return article.sentiment_score * 2
    
    def _calculate_confidence_score(self, article: NewsArticle, text: str) -> float:
        """신뢰도 점수 계산"""
        confidence = 0.5  # 기본 점수
        
        # 소스 신뢰도
        trusted_sources = [
            "reuters", "bloomberg", "cnbc", "wall street journal",
            "financial times", "coindesk", "cointelegraph"
        ]
        
        if any(source in article.source.lower() for source in trusted_sources):
            confidence += 0.3
        
        # 키워드 일치도
        bitcoin_keywords = ["bitcoin", "btc", "cryptocurrency", "crypto"]
        keyword_matches = sum(1 for keyword in bitcoin_keywords if keyword in text)
        confidence += min(0.2, keyword_matches * 0.05)
        
        return min(1.0, confidence)
    
    def _calculate_time_sensitivity(self, event_type: EventType, 
                                  impact_level: MarketImpactLevel) -> int:
        """시간 민감성 계산 (분 단위)"""
        base_sensitivity = {
            MarketImpactLevel.CRITICAL: 15,  # 15분
            MarketImpactLevel.HIGH: 60,      # 1시간  
            MarketImpactLevel.MEDIUM: 240,   # 4시간
            MarketImpactLevel.LOW: 1440      # 24시간
        }
        
        event_multiplier = {
            EventType.REGULATORY: 0.5,      # 빠른 반응
            EventType.MONETARY_POLICY: 0.3, # 매우 빠른 반응
            EventType.INSTITUTIONAL: 0.8,   # 중간 반응
            EventType.CELEBRITY: 0.2,       # 즉시 반응
            EventType.TECHNICAL: 1.5,       # 느린 반응
            EventType.MACRO_ECONOMIC: 1.0,  # 보통 반응
            EventType.MARKET_STRUCTURE: 0.7,# 빠른 반응
            EventType.GEOPOLITICAL: 1.2     # 중간 반응
        }
        
        base_time = base_sensitivity.get(impact_level, 1440)
        multiplier = event_multiplier.get(event_type, 1.0)
        
        return int(base_time * multiplier)
    
    def filter_high_impact_articles(self, articles: List[NewsArticle], 
                                  min_composite_score: float = 0.6) -> List[Tuple[NewsArticle, MarketRelevanceScore]]:
        """고영향 기사 필터링"""
        high_impact_articles = []
        
        for article in articles:
            relevance = self.analyze_article_relevance(article)
            
            if relevance.composite_score >= min_composite_score:
                high_impact_articles.append((article, relevance))
        
        # 종합 점수로 정렬
        high_impact_articles.sort(key=lambda x: x[1].composite_score, reverse=True)
        
        return high_impact_articles
    
    def get_market_moving_news(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """시장 영향 뉴스 종합 분석"""
        
        filtered_articles = self.filter_high_impact_articles(articles, min_composite_score=0.5)
        
        # 영향도별 분류
        impact_distribution = {}
        event_type_distribution = {}
        
        for article, relevance in filtered_articles:
            # 영향도 분포
            level = relevance.impact_level.value
            if level not in impact_distribution:
                impact_distribution[level] = []
            impact_distribution[level].append({
                "article": article,
                "score": relevance.composite_score,
                "time_sensitivity": relevance.time_sensitivity
            })
            
            # 이벤트 유형 분포
            event_type = relevance.event_type.value
            if event_type not in event_type_distribution:
                event_type_distribution[event_type] = 0
            event_type_distribution[event_type] += 1
        
        # 전체 시장 감정 계산
        total_sentiment = 0
        sentiment_count = 0
        
        for article, relevance in filtered_articles:
            if article.sentiment_score is not None:
                weighted_sentiment = article.sentiment_score * relevance.composite_score
                total_sentiment += weighted_sentiment
                sentiment_count += relevance.composite_score
        
        market_sentiment = total_sentiment / sentiment_count if sentiment_count > 0 else 0
        
        return {
            "total_articles": len(articles),
            "high_impact_articles": len(filtered_articles),
            "impact_distribution": impact_distribution,
            "event_type_distribution": event_type_distribution,
            "market_sentiment": {
                "score": market_sentiment,
                "label": self._get_sentiment_label(market_sentiment)
            },
            "top_articles": [
                {
                    "title": article.title,
                    "source": article.source,
                    "impact_level": relevance.impact_level.value,
                    "event_type": relevance.event_type.value,
                    "composite_score": relevance.composite_score,
                    "time_sensitivity_minutes": relevance.time_sensitivity,
                    "published": article.published_date.isoformat()
                }
                for article, relevance in filtered_articles[:10]
            ]
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """감정 점수를 라벨로 변환"""
        if score > 0.3:
            return "bullish"
        elif score < -0.3:
            return "bearish"
        else:
            return "neutral"


# 사용 예제
async def demo_bitcoin_futures_filter():
    """비트코인 선물시장 필터 데모"""
    
    # 샘플 기사 생성 (실제로는 뉴스 수집기에서 가져옴)
    sample_articles = [
        NewsArticle(
            id="1",
            title="Jerome Powell Hints at Emergency Rate Cut Amid Market Turmoil",
            content="Federal Reserve Chairman Jerome Powell suggested emergency measures...",
            summary="Fed Chairman considers emergency rate cut",
            url="https://example.com/1",
            source="Reuters",
            author="Financial Reporter",
            published_date=datetime.now() - timedelta(minutes=30),
            collected_date=datetime.now(),
            category=NewsCategory.MACRO,
            keywords=["jerome powell", "fed", "rate cut"],
            entities=["Jerome Powell", "Federal Reserve"],
            sentiment_score=0.7
        )
    ]
    
    # 필터 생성 및 분석
    btc_filter = BitcoinFuturesNewsFilter()
    
    # 시장 영향 분석
    market_analysis = btc_filter.get_market_moving_news(sample_articles)
    
    print("🚀 Bitcoin Futures Market Analysis")
    print("=" * 50)
    print(f"Total articles analyzed: {market_analysis['total_articles']}")
    print(f"High impact articles: {market_analysis['high_impact_articles']}")
    print(f"Market sentiment: {market_analysis['market_sentiment']['label']} ({market_analysis['market_sentiment']['score']:.3f})")
    
    print("\n📊 Top Market-Moving Articles:")
    for i, article_data in enumerate(market_analysis['top_articles'][:5]):
        print(f"{i+1}. {article_data['title']}")
        print(f"   Impact: {article_data['impact_level']} | Type: {article_data['event_type']}")
        print(f"   Score: {article_data['composite_score']:.3f} | Sensitivity: {article_data['time_sensitivity_minutes']}min")
        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_bitcoin_futures_filter())