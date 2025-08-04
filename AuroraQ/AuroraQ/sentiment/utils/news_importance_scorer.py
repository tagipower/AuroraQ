#!/usr/bin/env python3
"""
News Importance Scorer - 뉴스 중요도 점수화 시스템
출처, 내용, 정책 키워드 기반으로 0.0~1.0 중요도 점수 산출
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SourceTier(Enum):
    """뉴스 소스 등급"""
    TIER_1 = "tier_1"  # 최고급 (Reuters, Bloomberg, WSJ)
    TIER_2 = "tier_2"  # 주요 (CNBC, MarketWatch, CNN)
    TIER_3 = "tier_3"  # 일반 (Yahoo, Google News)
    TIER_4 = "tier_4"  # 기타

class EventCategory(Enum):
    """이벤트 카테고리"""
    MONETARY_POLICY = "monetary_policy"  # 통화정책
    REGULATION = "regulation"            # 규제
    MARKET_STRUCTURE = "market_structure" # 시장구조
    MACRO_ECONOMIC = "macro_economic"    # 거시경제
    CORPORATE = "corporate"              # 기업
    TECHNICAL = "technical"              # 기술
    GENERAL = "general"                  # 일반

@dataclass
class ImportanceFeatures:
    """중요도 점수 계산을 위한 특징"""
    source_score: float = 0.0          # 출처 점수 (0.0-0.3)
    policy_keyword_score: float = 0.0  # 정책 키워드 점수 (0.0-0.4)
    content_quality_score: float = 0.0 # 내용 품질 점수 (0.0-0.2)
    timing_score: float = 0.0          # 시기 점수 (0.0-0.1)
    
    @property
    def total_score(self) -> float:
        """총 중요도 점수 (0.0-1.0)"""
        return min(1.0, self.source_score + self.policy_keyword_score + 
                        self.content_quality_score + self.timing_score)

class NewsImportanceScorer:
    """뉴스 중요도 점수화 시스템"""
    
    def __init__(self):
        """초기화"""
        # 소스별 점수 매핑
        self.source_scores = {
            # Tier 1 - 최고급 금융 뉴스 (0.25-0.30)
            "reuters.com": 0.30,
            "bloomberg.com": 0.30,
            "wsj.com": 0.28,
            "ft.com": 0.28,
            "marketwatch.com": 0.25,
            
            # Tier 2 - 주요 뉴스 (0.15-0.24)
            "cnbc.com": 0.24,
            "cnn.com": 0.20,
            "bbc.com": 0.22,
            "investing.com": 0.18,
            "coindesk.com": 0.20,
            "cointelegraph.com": 0.18,
            
            # Tier 3 - 일반 뉴스 (0.08-0.14)
            "yahoo.com": 0.12,
            "news.google.com": 0.10,
            "finnhub.io": 0.14,
            "newsapi.org": 0.08,
            
            # Tier 4 - 기타 (0.05)
            "default": 0.05
        }
        
        # 정책 키워드 시스템 (이벤트 중심)
        self.policy_keywords = {
            # 통화정책 키워드 (최고 중요도 0.4)
            EventCategory.MONETARY_POLICY: {
                "critical": {  # 0.35-0.40
                    "fomc", "federal reserve", "fed meeting", "interest rate decision",
                    "rate cut", "rate hike", "monetary policy", "jerome powell",
                    "fed chair", "fed minutes", "quantitative easing", "qe",
                    "tightening", "dovish", "hawkish"
                },
                "high": {  # 0.25-0.34
                    "inflation target", "cpi", "pce", "core inflation",
                    "employment data", "nfp", "unemployment rate", "fed funds rate",
                    "federal open market", "fed policy", "central bank"
                },
                "medium": {  # 0.15-0.24
                    "fed speech", "fed official", "fed governor", "fed president",
                    "monetary accommodation", "policy stance"
                }
            },
            
            # 규제 키워드 (0.3-0.35)
            EventCategory.REGULATION: {
                "critical": {  # 0.30-0.35
                    "sec approval", "sec rejection", "sec investigation",
                    "regulatory approval", "regulatory ban", "regulatory clarity",
                    "cftc", "finra", "cryptocurrency regulation", "crypto ban"
                },
                "high": {  # 0.20-0.29
                    "compliance", "regulatory framework", "regulatory guidance",
                    "aml", "kyc", "anti-money laundering", "know your customer",
                    "sec filing", "regulatory filing"
                },
                "medium": {  # 0.10-0.19
                    "regulatory update", "compliance requirement", "regulatory news"
                }
            },
            
            # 시장구조 키워드 (0.25-0.30)
            EventCategory.MARKET_STRUCTURE: {
                "critical": {  # 0.25-0.30
                    "etf approval", "etf launch", "institutional adoption",
                    "blackrock", "vanguard", "fidelity", "grayscale",
                    "market manipulation", "whale movement", "large transfer"
                },
                "high": {  # 0.15-0.24
                    "institutional investment", "hedge fund", "pension fund",
                    "sovereign wealth fund", "etf inflow", "etf outflow"
                },
                "medium": {  # 0.08-0.14
                    "market maker", "liquidity provider", "trading volume"
                }
            },
            
            # 거시경제 키워드 (0.20-0.25)
            EventCategory.MACRO_ECONOMIC: {
                "critical": {  # 0.20-0.25
                    "recession", "economic crisis", "financial crisis",
                    "banking crisis", "sovereign debt", "currency crisis"
                },
                "high": {  # 0.12-0.19
                    "gdp", "economic growth", "trade war", "geopolitical",
                    "sanctions", "economic sanctions", "fiscal policy"
                },
                "medium": {  # 0.06-0.11
                    "economic indicator", "economic data", "economic outlook"
                }
            },
            
            # 기업 키워드 (0.15-0.20)
            EventCategory.CORPORATE: {
                "critical": {  # 0.15-0.20
                    "tesla bitcoin", "microstrategy", "coinbase", "binance",
                    "ftx", "major acquisition", "bankruptcy", "default"
                },
                "high": {  # 0.08-0.14
                    "earnings", "partnership", "collaboration", "investment",
                    "funding", "ipo", "merger", "acquisition"
                },
                "medium": {  # 0.04-0.07
                    "corporate news", "business update", "company announcement"
                }
            }
        }
        
        # 내용 품질 평가 패턴
        self.quality_patterns = {
            "high_quality": {  # +0.15-0.20
                r"\d+\.\d+%",  # 구체적 수치
                r"\$\d+(?:,\d{3})*(?:\.\d+)?",  # 금액
                r"according to.*(?:report|study|survey)",  # 신뢰할 수 있는 출처
                r"data (?:shows|indicates|reveals)",  # 데이터 기반
                r"(?:analysts?|experts?|economists?) (?:say|predict|expect)"  # 전문가 의견
            },
            "medium_quality": {  # +0.08-0.14
                r"(?:rising|falling|increasing|decreasing) by",  # 변화량
                r"market (?:expects|anticipates)",  # 시장 기대
                r"trading (?:volume|activity)",  # 거래 관련
                r"price (?:target|forecast)"  # 가격 예측
            },
            "low_quality": {  # +0.02-0.07
                r"could|might|may|possibly",  # 불확실성 표현
                r"rumors?|speculation",  # 추측성 내용
                r"social media|twitter|reddit"  # 소셜미디어 출처
            }
        }
        
        # 통계
        self.stats = {
            "total_scored": 0,
            "high_importance_count": 0,  # >0.7
            "medium_importance_count": 0,  # 0.4-0.7
            "low_importance_count": 0,   # <0.4
            "avg_score": 0.0,
            "last_update": datetime.now()
        }
    
    def calculate_importance_score(self, 
                                 title: str,
                                 content: str,
                                 source_url: str,
                                 published_at: datetime,
                                 metadata: Optional[Dict[str, Any]] = None) -> ImportanceFeatures:
        """
        뉴스 중요도 점수 계산
        
        Args:
            title: 뉴스 제목
            content: 뉴스 내용
            source_url: 출처 URL
            published_at: 발행 시간
            metadata: 추가 메타데이터
            
        Returns:
            ImportanceFeatures: 중요도 특징 및 점수
        """
        features = ImportanceFeatures()
        
        try:
            # 1. 출처 점수 계산 (0.0-0.3)
            features.source_score = self._calculate_source_score(source_url)
            
            # 2. 정책 키워드 점수 계산 (0.0-0.4)
            features.policy_keyword_score = self._calculate_policy_keyword_score(title, content)
            
            # 3. 내용 품질 점수 계산 (0.0-0.2)
            features.content_quality_score = self._calculate_content_quality_score(title, content)
            
            # 4. 시기 점수 계산 (0.0-0.1)
            features.timing_score = self._calculate_timing_score(published_at)
            
            # 통계 업데이트
            self._update_stats(features.total_score)
            
            logger.debug(f"Importance score calculated: {features.total_score:.3f} "
                        f"(source:{features.source_score:.2f}, "
                        f"policy:{features.policy_keyword_score:.2f}, "
                        f"quality:{features.content_quality_score:.2f}, "
                        f"timing:{features.timing_score:.2f})")
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating importance score: {e}")
            return ImportanceFeatures()  # 기본값 반환
    
    def _calculate_source_score(self, source_url: str) -> float:
        """출처 점수 계산"""
        if not source_url:
            return self.source_scores["default"]
        
        source_url_lower = source_url.lower()
        
        # 도메인 추출 및 매칭
        for domain, score in self.source_scores.items():
            if domain != "default" and domain in source_url_lower:
                return score
        
        return self.source_scores["default"]
    
    def _calculate_policy_keyword_score(self, title: str, content: str) -> float:
        """정책 키워드 점수 계산"""
        text = f"{title} {content}".lower()
        max_score = 0.0
        
        for category, levels in self.policy_keywords.items():
            for level, keywords in levels.items():
                for keyword in keywords:
                    if keyword in text:
                        # 키워드 레벨별 점수
                        if level == "critical":
                            score = 0.35 + (len(keyword.split()) - 1) * 0.01  # 복합 키워드 가산점
                        elif level == "high":
                            score = 0.25 + (len(keyword.split()) - 1) * 0.008
                        else:  # medium
                            score = 0.15 + (len(keyword.split()) - 1) * 0.005
                        
                        max_score = max(max_score, score)
        
        return min(0.4, max_score)  # 최대 0.4로 제한
    
    def _calculate_content_quality_score(self, title: str, content: str) -> float:
        """내용 품질 점수 계산"""
        text = f"{title} {content}"
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # 고품질 패턴 검사
        for pattern in self.quality_patterns["high_quality"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.04  # 각 패턴당 0.04점
        
        # 중품질 패턴 검사
        for pattern in self.quality_patterns["medium_quality"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.02  # 각 패턴당 0.02점
        
        # 저품질 패턴 검사 (감점)
        for pattern in self.quality_patterns["low_quality"]:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.01  # 각 패턴당 0.01점 감점
        
        # 내용 길이 기반 품질 점수
        content_length = len(content)
        if content_length > 500:  # 충분한 내용
            score += 0.03
        elif content_length > 200:  # 적절한 내용
            score += 0.02
        elif content_length < 50:  # 내용 부족
            score -= 0.02
        
        return max(0.0, min(0.2, score))  # 0.0-0.2 범위로 제한
    
    def _calculate_timing_score(self, published_at: datetime) -> float:
        """시기 점수 계산"""
        now = datetime.now()
        time_diff = now - published_at
        
        # 시간 기반 점수 (최신일수록 높음)
        if time_diff < timedelta(hours=1):
            return 0.1  # 1시간 이내: 최고점
        elif time_diff < timedelta(hours=6):
            return 0.08  # 6시간 이내
        elif time_diff < timedelta(hours=24):
            return 0.05  # 24시간 이내
        elif time_diff < timedelta(days=3):
            return 0.02  # 3일 이내
        else:
            return 0.0  # 3일 이후: 점수 없음
    
    def _update_stats(self, score: float):
        """통계 업데이트"""
        self.stats["total_scored"] += 1
        
        if score >= 0.7:
            self.stats["high_importance_count"] += 1
        elif score >= 0.4:
            self.stats["medium_importance_count"] += 1
        else:
            self.stats["low_importance_count"] += 1
        
        # 평균 점수 업데이트 (지수 이동 평균)
        alpha = 0.1
        if self.stats["total_scored"] == 1:
            self.stats["avg_score"] = score
        else:
            self.stats["avg_score"] = (alpha * score + 
                                     (1 - alpha) * self.stats["avg_score"])
        
        self.stats["last_update"] = datetime.now()
    
    def get_importance_threshold(self, level: str = "medium") -> float:
        """중요도 임계값 반환"""
        thresholds = {
            "high": 0.7,      # 고중요도: FinBERT 분석 우선
            "medium": 0.4,    # 중중요도: 조건부 FinBERT 분석
            "low": 0.2        # 저중요도: 키워드 분석만
        }
        return thresholds.get(level, 0.4)
    
    def should_analyze_with_finbert(self, importance_score: float, 
                                  confidence_threshold: float = 0.4) -> bool:
        """FinBERT 분석 필요 여부 판단"""
        return importance_score >= confidence_threshold
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """점수화 통계 반환"""
        total = self.stats["total_scored"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "high_importance_ratio": self.stats["high_importance_count"] / total,
            "medium_importance_ratio": self.stats["medium_importance_count"] / total,
            "low_importance_ratio": self.stats["low_importance_count"] / total,
            "effective_filtering_ratio": (
                self.stats["high_importance_count"] + self.stats["medium_importance_count"]
            ) / total
        }

# 글로벌 인스턴스 (싱글톤 패턴)
_global_scorer: Optional[NewsImportanceScorer] = None

def get_news_importance_scorer() -> NewsImportanceScorer:
    """뉴스 중요도 점수화 시스템 인스턴스 반환"""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = NewsImportanceScorer()
    return _global_scorer

def calculate_news_importance(title: str, content: str, source_url: str, 
                            published_at: datetime) -> float:
    """뉴스 중요도 점수 계산 (간편 함수)"""
    scorer = get_news_importance_scorer()
    features = scorer.calculate_importance_score(title, content, source_url, published_at)
    return features.total_score

if __name__ == "__main__":
    # 테스트 코드
    scorer = NewsImportanceScorer()
    
    # 테스트 케이스들
    test_cases = [
        {
            "title": "Fed Raises Interest Rates by 0.75% in Aggressive Move",
            "content": "The Federal Reserve raised interest rates by 0.75 percentage points, the largest increase since 1994. Jerome Powell indicated further hikes may be necessary to combat inflation.",
            "source_url": "https://reuters.com/markets/fed-decision",
            "published_at": datetime.now() - timedelta(hours=2)
        },
        {
            "title": "Bitcoin Price Fluctuates Amid Market Uncertainty",
            "content": "Bitcoin saw minor price movements today as traders await clarity on regulatory developments.",
            "source_url": "https://yahoo.com/finance/bitcoin",
            "published_at": datetime.now() - timedelta(days=1)
        },
        {
            "title": "SEC Approves First Bitcoin ETF Application",
            "content": "The Securities and Exchange Commission has approved the first spot Bitcoin ETF, marking a historic milestone for cryptocurrency adoption. The ETF will begin trading next week with major institutional backing.",
            "source_url": "https://bloomberg.com/news/sec-bitcoin-etf",
            "published_at": datetime.now() - timedelta(minutes=30)
        }
    ]
    
    print("=== News Importance Scoring Test ===\n")
    
    for i, case in enumerate(test_cases, 1):
        features = scorer.calculate_importance_score(**case)
        print(f"Test Case {i}:")
        print(f"  Title: {case['title'][:50]}...")
        print(f"  Source: {case['source_url']}")
        print(f"  Total Score: {features.total_score:.3f}")
        print(f"  Components:")
        print(f"    - Source: {features.source_score:.3f}")
        print(f"    - Policy Keywords: {features.policy_keyword_score:.3f}")
        print(f"    - Content Quality: {features.content_quality_score:.3f}")
        print(f"    - Timing: {features.timing_score:.3f}")
        print(f"  FinBERT Analysis: {'Yes' if scorer.should_analyze_with_finbert(features.total_score) else 'No'}")
        print()
    
    # 통계 출력
    stats = scorer.get_scoring_stats()
    print("=== Scoring Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")