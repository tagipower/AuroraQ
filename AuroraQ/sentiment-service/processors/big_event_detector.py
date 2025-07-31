#!/usr/bin/env python3
"""
Big Event Detection System for AuroraQ Sentiment Service
주요 시장 이벤트 감지 및 영향도 평가 시스템
"""

import asyncio
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import statistics
from collections import defaultdict

# 로컬 임포트
from ..collectors.enhanced_news_collector import NewsItem, NewsSource
from ..utils.data_quality_validator import DataQualityValidator, ContentItem, ContentType
from ..utils.content_cache_manager import ContentCacheManager

logger = logging.getLogger(__name__)

class EventType(Enum):
    """이벤트 유형"""
    FOMC_MEETING = "fomc_meeting"
    CPI_RELEASE = "cpi_release" 
    ETF_APPROVAL = "etf_approval"
    REGULATION = "regulation"
    INSTITUTIONAL_INVESTMENT = "institutional_investment"
    TECHNICAL_UPGRADE = "technical_upgrade"
    SECURITY_INCIDENT = "security_incident"
    PARTNERSHIP = "partnership"
    EARNINGS_REPORT = "earnings_report"
    MACROECONOMIC = "macroeconomic"
    GEOPOLITICAL = "geopolitical"

class EventImpact(Enum):
    """이벤트 영향도"""
    CRITICAL = "critical"     # 1.0
    HIGH = "high"             # 0.8
    MEDIUM = "medium"         # 0.6
    LOW = "low"               # 0.4
    MINIMAL = "minimal"       # 0.2

@dataclass
class BigEvent:
    """대형 이벤트"""
    event_id: str
    event_type: EventType
    title: str
    description: str
    impact_level: EventImpact
    base_impact_score: float  # 0.0 ~ 1.0
    sentiment_bias: float     # -1.0 ~ 1.0
    volatility_factor: float  # 0.0 ~ 1.0
    final_impact_score: float # 계산된 최종 점수
    confidence: float         # 0.0 ~ 1.0
    detected_at: datetime
    event_time: Optional[datetime] = None
    source_urls: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    symbols_affected: List[str] = field(default_factory=list)
    news_count: int = 0
    
    def calculate_final_impact(self) -> float:
        """최종 영향도 점수 계산"""
        # 공식: event_impact_score = impact_score * (1 + abs(sentiment_bias) * 0.5) * (1 + volatility_factor)
        sentiment_multiplier = 1 + abs(self.sentiment_bias) * 0.5
        volatility_multiplier = 1 + self.volatility_factor
        
        self.final_impact_score = self.base_impact_score * sentiment_multiplier * volatility_multiplier
        return min(2.0, self.final_impact_score)  # 최대 2.0으로 제한
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "impact_level": self.impact_level.value,
            "base_impact_score": round(self.base_impact_score, 3),
            "sentiment_bias": round(self.sentiment_bias, 3),
            "volatility_factor": round(self.volatility_factor, 3),
            "final_impact_score": round(self.final_impact_score, 3),
            "confidence": round(self.confidence, 3),
            "detected_at": self.detected_at.isoformat(),
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "source_urls": self.source_urls,
            "keywords": self.keywords,
            "symbols_affected": self.symbols_affected,
            "news_count": self.news_count
        }

class BigEventDetector:
    """대형 이벤트 감지기"""
    
    def __init__(self, 
                 data_validator: Optional[DataQualityValidator] = None,
                 cache_manager: Optional[ContentCacheManager] = None):
        """
        초기화
        
        Args:
            data_validator: 데이터 품질 검증기
            cache_manager: 캐시 매니저
        """
        self.data_validator = data_validator
        self.cache_manager = cache_manager
        
        # 이벤트 패턴 정의
        self.event_patterns = {
            EventType.FOMC_MEETING: {
                "keywords": ["fomc", "federal reserve", "fed meeting", "interest rate", "monetary policy"],
                "required_terms": ["fed", "meeting", "rate"],
                "base_impact": 0.9,
                "volatility_base": 0.7
            },
            EventType.CPI_RELEASE: {
                "keywords": ["cpi", "inflation", "consumer price index", "price index"],
                "required_terms": ["cpi", "inflation"],
                "base_impact": 0.8,
                "volatility_base": 0.6
            },
            EventType.ETF_APPROVAL: {
                "keywords": ["etf approval", "sec approval", "bitcoin etf", "crypto etf"],
                "required_terms": ["etf", "approval"],
                "base_impact": 0.9,
                "volatility_base": 0.8
            },
            EventType.REGULATION: {
                "keywords": ["regulation", "sec", "regulatory", "compliance", "policy"],
                "required_terms": ["regulation", "sec"],
                "base_impact": 0.7,
                "volatility_base": 0.6
            },
            EventType.INSTITUTIONAL_INVESTMENT: {
                "keywords": ["institutional", "investment", "fund", "grayscale", "blackrock"],
                "required_terms": ["institutional", "investment"],
                "base_impact": 0.6,
                "volatility_base": 0.5
            },
            EventType.TECHNICAL_UPGRADE: {
                "keywords": ["upgrade", "update", "fork", "ethereum 2.0", "merge"],
                "required_terms": ["upgrade", "update"],
                "base_impact": 0.5,
                "volatility_base": 0.4
            },
            EventType.SECURITY_INCIDENT: {
                "keywords": ["hack", "security", "breach", "exploit", "vulnerability"],
                "required_terms": ["hack", "security"],
                "base_impact": 0.8,
                "volatility_base": 0.9
            },
            EventType.PARTNERSHIP: {
                "keywords": ["partnership", "collaboration", "integration", "adoption"],
                "required_terms": ["partnership", "adoption"],
                "base_impact": 0.4,
                "volatility_base": 0.3
            },
            EventType.EARNINGS_REPORT: {
                "keywords": ["earnings", "quarterly", "revenue", "profit", "financial results"],
                "required_terms": ["earnings", "quarterly"],
                "base_impact": 0.5,
                "volatility_base": 0.4
            },
            EventType.MACROECONOMIC: {
                "keywords": ["gdp", "unemployment", "economic", "recession", "growth"],
                "required_terms": ["economic", "gdp"],
                "base_impact": 0.7,
                "volatility_base": 0.5
            },
            EventType.GEOPOLITICAL: {
                "keywords": ["war", "sanctions", "political", "government", "election"],
                "required_terms": ["political", "government"],
                "base_impact": 0.6,
                "volatility_base": 0.7
            }
        }
        
        # 심볼별 영향도 매핑
        self.symbol_impact_mapping = {
            "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
            "ETH": ["ethereum", "eth", "ether"],
            "STOCK": ["stock", "equity", "share", "market"],
            "FOREX": ["forex", "currency", "dollar", "euro"],
            "COMMODITY": ["gold", "oil", "commodity", "precious"]
        }
        
        # 감지된 이벤트 캐시
        self.detected_events: Dict[str, BigEvent] = {}
        self.event_ttl = 86400  # 24시간
        
        # 임계값 설정
        self.min_news_count = 3  # 최소 뉴스 개수
        self.min_confidence = 0.6  # 최소 신뢰도
        self.similarity_threshold = 0.8  # 중복 이벤트 임계값
        
        # 통계
        self.stats = {
            "total_detected": 0,
            "by_type": defaultdict(int),
            "by_impact": defaultdict(int),
            "avg_confidence": 0.0,
            "processing_time_avg": 0.0
        }
    
    def _extract_event_features(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """뉴스 아이템들에서 이벤트 특성 추출"""
        
        if not news_items:
            return {}
        
        # 텍스트 통합
        all_text = " ".join([f"{item.title} {item.content}" for item in news_items]).lower()
        
        # 시간 분석
        times = [item.published_at for item in news_items if item.published_at]
        time_span = (max(times) - min(times)).total_seconds() / 3600 if len(times) > 1 else 0
        
        # 소스 다양성
        sources = set(item.source.value for item in news_items)
        source_diversity = len(sources) / len(news_items) if news_items else 0
        
        # 관련성 점수 평균
        avg_relevance = statistics.mean(item.relevance_score for item in news_items)
        
        # 키워드 빈도
        word_freq = defaultdict(int)
        words = re.findall(r'\b\w+\b', all_text)
        for word in words:
            if len(word) >= 3:
                word_freq[word] += 1
        
        # 상위 키워드
        top_keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return {
            "text_length": len(all_text),
            "news_count": len(news_items),
            "time_span_hours": time_span,
            "source_diversity": source_diversity,
            "avg_relevance": avg_relevance,
            "top_keywords": top_keywords,
            "word_frequency": dict(word_freq),
            "sources": list(sources)
        }
    
    def _calculate_sentiment_bias(self, news_items: List[NewsItem]) -> float:
        """뉴스 아이템들의 감정 편향 계산"""
        
        if not news_items:
            return 0.0
        
        # 제목과 내용에서 긍정/부정 키워드 분석
        positive_keywords = ["surge", "rally", "growth", "approval", "positive", "bullish", "rise", "gain"]
        negative_keywords = ["crash", "drop", "decline", "negative", "bearish", "fall", "loss", "fear"]
        
        sentiment_scores = []
        
        for item in news_items:
            text = f"{item.title} {item.content}".lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # 간단한 감정 점수 계산
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0
            
            sentiment_scores.append(sentiment)
        
        return statistics.mean(sentiment_scores) if sentiment_scores else 0.0
    
    def _calculate_volatility_factor(self, news_items: List[NewsItem], event_type: EventType) -> float:
        """변동성 요인 계산"""
        
        base_volatility = self.event_patterns[event_type]["volatility_base"]
        
        # 뉴스 개수에 따른 조정
        news_count_factor = min(1.0, len(news_items) / 10.0)
        
        # 시간 집중도에 따른 조정
        features = self._extract_event_features(news_items)
        time_span = features.get("time_span_hours", 24)
        time_concentration = max(0.1, 1.0 - (time_span / 24.0))  # 24시간 기준
        
        # 소스 다양성에 따른 조정
        source_diversity = features.get("source_diversity", 0.5)
        
        # 최종 변동성 계산
        volatility = base_volatility * (
            0.4 * news_count_factor +
            0.3 * time_concentration +
            0.3 * source_diversity
        )
        
        return min(1.0, volatility)
    
    def _determine_symbols_affected(self, news_items: List[NewsItem]) -> List[str]:
        """영향받는 심볼 결정"""
        
        affected_symbols = set()
        
        for item in news_items:
            text = f"{item.title} {item.content}".lower()
            
            # 직접 심볼 매칭
            if item.symbol:
                affected_symbols.add(item.symbol.upper())
            
            # 키워드 기반 심볼 매칭
            for symbol, keywords in self.symbol_impact_mapping.items():
                if any(keyword in text for keyword in keywords):
                    affected_symbols.add(symbol)
        
        return list(affected_symbols)
    
    def _calculate_confidence(self, 
                            news_items: List[NewsItem], 
                            event_type: EventType,
                            features: Dict[str, Any]) -> float:
        """이벤트 신뢰도 계산"""
        
        confidence_factors = []
        
        # 1. 뉴스 개수 신뢰도
        news_count = len(news_items)
        news_confidence = min(1.0, news_count / 5.0)  # 5개 이상이면 최대
        confidence_factors.append(news_confidence)
        
        # 2. 소스 다양성 신뢰도
        source_diversity = features.get("source_diversity", 0.5)
        confidence_factors.append(source_diversity)
        
        # 3. 관련성 점수 신뢰도
        avg_relevance = features.get("avg_relevance", 0.5)
        confidence_factors.append(avg_relevance)
        
        # 4. 패턴 매칭 신뢰도
        pattern = self.event_patterns[event_type]
        text = " ".join([f"{item.title} {item.content}" for item in news_items]).lower()
        
        keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword in text)
        required_matches = sum(1 for term in pattern["required_terms"] if term in text)
        
        pattern_confidence = (keyword_matches / len(pattern["keywords"]) * 0.6 + 
                             required_matches / len(pattern["required_terms"]) * 0.4)
        confidence_factors.append(pattern_confidence)
        
        # 5. 시간 집중도 신뢰도
        time_span = features.get("time_span_hours", 24)
        time_confidence = max(0.3, 1.0 - (time_span / 48.0))  # 48시간 기준
        confidence_factors.append(time_confidence)
        
        return statistics.mean(confidence_factors)
    
    def _detect_event_type(self, news_items: List[NewsItem]) -> Optional[EventType]:
        """뉴스 아이템들에서 이벤트 유형 감지"""
        
        if not news_items:
            return None
        
        # 모든 텍스트 통합
        all_text = " ".join([f"{item.title} {item.content}" for item in news_items]).lower()
        
        # 각 이벤트 유형별 점수 계산
        type_scores = {}
        
        for event_type, pattern in self.event_patterns.items():
            score = 0.0
            
            # 키워드 매칭 점수
            keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword in all_text)
            keyword_score = keyword_matches / len(pattern["keywords"])
            
            # 필수 용어 매칭 점수
            required_matches = sum(1 for term in pattern["required_terms"] if term in all_text)
            required_score = required_matches / len(pattern["required_terms"])
            
            # 최소 필수 용어가 하나라도 매칭되어야 함
            if required_matches > 0:
                score = keyword_score * 0.6 + required_score * 0.4
                type_scores[event_type] = score
        
        # 가장 높은 점수의 이벤트 유형 선택
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] >= 0.3:  # 최소 임계값
                return best_type[0]
        
        return None
    
    def _generate_event_id(self, event_type: EventType, news_items: List[NewsItem]) -> str:
        """이벤트 ID 생성"""
        
        # 대표 제목과 시간 기반
        if news_items:
            representative_title = news_items[0].title
            time_str = news_items[0].published_at.strftime("%Y%m%d")
        else:
            representative_title = "unknown"
            time_str = datetime.now().strftime("%Y%m%d")
        
        content = f"{event_type.value}_{representative_title}_{time_str}"
        
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _is_duplicate_event(self, new_event: BigEvent) -> Optional[str]:
        """중복 이벤트 확인"""
        
        for existing_id, existing_event in self.detected_events.items():
            # 같은 유형이고 시간이 가까운 경우
            if (existing_event.event_type == new_event.event_type and
                abs((existing_event.detected_at - new_event.detected_at).total_seconds()) < 3600):
                
                # 제목 유사도 확인
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, existing_event.title, new_event.title).ratio()
                
                if similarity >= self.similarity_threshold:
                    return existing_id
        
        return None
    
    async def detect_events(self, news_items: List[NewsItem]) -> List[BigEvent]:
        """뉴스 아이템들에서 대형 이벤트 감지"""
        
        start_time = time.time()
        logger.info(f"Detecting big events from {len(news_items)} news items...")
        
        if not news_items:
            return []
        
        # 데이터 품질 검증
        if self.data_validator:
            content_items = []
            for item in news_items:
                content_item = ContentItem(
                    content_id=item.hash_id,
                    title=item.title,
                    content=item.content,
                    url=item.url,
                    source=item.source.value,
                    published_at=item.published_at,
                    content_type=ContentType.NEWS_ARTICLE,
                    category=item.category,
                    symbol=item.symbol
                )
                content_items.append(content_item)
            
            validated_items, quality_scores = self.data_validator.validate_batch(
                content_items, min_threshold=0.5
            )
            
            # 검증된 아이템만 사용
            validated_hashes = {item.content_id for item in validated_items}
            news_items = [item for item in news_items if item.hash_id in validated_hashes]
            
            logger.info(f"Data validation: {len(validated_items)}/{len(content_items)} items passed")
        
        # 이벤트 유형별로 뉴스 그룹화
        type_groups = defaultdict(list)
        
        for item in news_items:
            event_type = self._detect_event_type([item])
            if event_type:
                type_groups[event_type].append(item)
        
        detected_events = []
        
        # 각 이벤트 유형별로 처리
        for event_type, items in type_groups.items():
            if len(items) < self.min_news_count:
                logger.debug(f"Insufficient news count for {event_type.value}: {len(items)}")
                continue
            
            # 이벤트 특성 추출
            features = self._extract_event_features(items)
            
            # 감정 편향 계산
            sentiment_bias = self._calculate_sentiment_bias(items)
            
            # 변동성 요인 계산
            volatility_factor = self._calculate_volatility_factor(items, event_type)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(items, event_type, features)
            
            if confidence < self.min_confidence:
                logger.debug(f"Low confidence for {event_type.value}: {confidence:.3f}")
                continue
            
            # 기본 영향도 점수
            base_impact = self.event_patterns[event_type]["base_impact"]
            
            # 영향받는 심볼
            affected_symbols = self._determine_symbols_affected(items)
            
            # 대표 제목과 설명 생성
            title = items[0].title  # 가장 관련성 높은 뉴스의 제목
            description = f"{len(items)} news items detected for {event_type.value}"
            
            # 이벤트 생성
            event = BigEvent(
                event_id=self._generate_event_id(event_type, items),
                event_type=event_type,
                title=title,
                description=description,
                impact_level=self._determine_impact_level(base_impact),
                base_impact_score=base_impact,
                sentiment_bias=sentiment_bias,
                volatility_factor=volatility_factor,
                final_impact_score=0.0,  # 아래에서 계산
                confidence=confidence,
                detected_at=datetime.now(),
                event_time=min(item.published_at for item in items if item.published_at),
                source_urls=[item.url for item in items[:5]],  # 최대 5개 URL
                keywords=features.get("top_keywords", [])[:10],
                symbols_affected=affected_symbols,
                news_count=len(items)
            )
            
            # 최종 영향도 계산
            event.calculate_final_impact()
            
            # 중복 확인
            duplicate_id = self._is_duplicate_event(event)
            if duplicate_id:
                # 기존 이벤트 업데이트
                existing_event = self.detected_events[duplicate_id]
                existing_event.news_count += len(items)
                existing_event.source_urls.extend(event.source_urls)
                existing_event.confidence = max(existing_event.confidence, event.confidence)
                logger.info(f"Updated existing event: {duplicate_id}")
                continue
            
            # 새 이벤트 추가
            self.detected_events[event.event_id] = event
            detected_events.append(event)
            
            # 통계 업데이트
            self.stats["total_detected"] += 1
            self.stats["by_type"][event_type.value] += 1
            self.stats["by_impact"][event.impact_level.value] += 1
            
            logger.info(f"Big event detected: {event_type.value} "
                       f"(impact: {event.final_impact_score:.3f}, "
                       f"confidence: {confidence:.3f})")
        
        # 처리 시간 통계 업데이트
        processing_time = time.time() - start_time
        current_avg = self.stats["processing_time_avg"]
        total_count = self.stats["total_detected"]
        
        if total_count > 0:
            self.stats["processing_time_avg"] = (
                (current_avg * (total_count - len(detected_events)) + processing_time) / total_count
            )
        
        # 평균 신뢰도 업데이트
        if detected_events:
            confidences = [event.confidence for event in detected_events]
            self.stats["avg_confidence"] = statistics.mean(confidences)
        
        logger.info(f"Event detection completed: {len(detected_events)} events detected "
                   f"in {processing_time:.2f}s")
        
        return detected_events
    
    def _determine_impact_level(self, base_impact: float) -> EventImpact:
        """기본 영향도에서 영향 레벨 결정"""
        
        if base_impact >= 0.9:
            return EventImpact.CRITICAL
        elif base_impact >= 0.7:
            return EventImpact.HIGH
        elif base_impact >= 0.5:
            return EventImpact.MEDIUM
        elif base_impact >= 0.3:
            return EventImpact.LOW
        else:
            return EventImpact.MINIMAL
    
    def get_active_events(self, 
                         min_impact: float = 0.5,
                         max_age_hours: int = 24) -> List[BigEvent]:
        """활성 이벤트 조회"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        active_events = []
        for event in self.detected_events.values():
            if (event.final_impact_score >= min_impact and 
                event.detected_at >= cutoff_time):
                active_events.append(event)
        
        # 영향도 순으로 정렬
        active_events.sort(key=lambda x: x.final_impact_score, reverse=True)
        
        return active_events
    
    def get_events_by_symbol(self, symbol: str) -> List[BigEvent]:
        """심볼별 이벤트 조회"""
        
        symbol_events = []
        for event in self.detected_events.values():
            if symbol.upper() in event.symbols_affected:
                symbol_events.append(event)
        
        symbol_events.sort(key=lambda x: x.detected_at, reverse=True)
        return symbol_events
    
    def cleanup_old_events(self) -> int:
        """오래된 이벤트 정리"""
        
        cutoff_time = datetime.now() - timedelta(seconds=self.event_ttl)
        
        old_event_ids = []
        for event_id, event in self.detected_events.items():
            if event.detected_at < cutoff_time:
                old_event_ids.append(event_id)
        
        for event_id in old_event_ids:
            del self.detected_events[event_id]
        
        logger.info(f"Cleaned up {len(old_event_ids)} old events")
        return len(old_event_ids)
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """감지기 통계 반환"""
        
        return {
            "active_events": len(self.detected_events),
            "event_ttl_hours": self.event_ttl / 3600,
            "min_news_count": self.min_news_count,
            "min_confidence": self.min_confidence,
            "similarity_threshold": self.similarity_threshold,
            "supported_event_types": len(self.event_patterns),
            "symbol_mappings": len(self.symbol_impact_mapping),
            **dict(self.stats)
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_big_event_detector():
        """대형 이벤트 감지기 테스트"""
        
        detector = BigEventDetector()
        
        print("=== 대형 이벤트 감지기 테스트 ===")
        
        # 테스트 뉴스 아이템들 생성
        test_news = [
            NewsItem(
                title="Fed Raises Interest Rates by 0.75% in Aggressive Move",
                content="The Federal Reserve raised interest rates by 0.75 percentage points at its latest FOMC meeting, marking the most aggressive rate hike in decades...",
                url="https://reuters.com/fed-rate-hike",
                source=NewsSource.REUTERS,
                published_at=datetime.now() - timedelta(hours=2),
                category="economics",
                symbol="USD",
                relevance_score=0.95,
                hash_id="fed_news_1"
            ),
            NewsItem(
                title="FOMC Meeting Results: Federal Reserve Signals More Rate Hikes Ahead",
                content="Following today's FOMC meeting, Fed Chair Jerome Powell indicated that additional rate increases are likely as the central bank continues its fight against inflation...",
                url="https://bloomberg.com/fomc-results",
                source=NewsSource.BLOOMBERG,
                published_at=datetime.now() - timedelta(hours=1),
                category="economics", 
                symbol="USD",
                relevance_score=0.92,
                hash_id="fed_news_2"
            ),
            NewsItem(
                title="Markets React to Fed Rate Decision: Stocks Plunge on Hawkish Tone",
                content="Stock markets fell sharply after the Federal Reserve's rate decision, with investors concerned about the central bank's aggressive monetary policy stance...",
                url="https://cnbc.com/fed-market-reaction",
                source=NewsSource.CNBC,
                published_at=datetime.now() - timedelta(hours=1),
                category="markets",
                symbol="SPY",
                relevance_score=0.88,
                hash_id="fed_news_3"
            ),
            NewsItem(
                title="Bitcoin ETF Approval: SEC Gives Green Light to Spot Bitcoin ETFs",
                content="The Securities and Exchange Commission has approved the first spot Bitcoin ETF applications, marking a historic moment for cryptocurrency adoption...",
                url="https://coindesk.com/bitcoin-etf-approval",
                source=NewsSource.COINDESK,
                published_at=datetime.now() - timedelta(hours=3),
                category="crypto",
                symbol="BTC",
                relevance_score=0.98,
                hash_id="btc_etf_1"
            ),
            NewsItem(
                title="Historic Bitcoin ETF Launch: Billions in Trading Volume on First Day",
                content="The newly approved Bitcoin ETFs saw unprecedented trading volume on their first day, with billions of dollars in assets flowing into the products...",
                url="https://cointelegraph.com/bitcoin-etf-launch",
                source=NewsSource.COINTELEGRAPH,
                published_at=datetime.now() - timedelta(hours=2),
                category="crypto",
                symbol="BTC", 
                relevance_score=0.94,
                hash_id="btc_etf_2"
            )
        ]
        
        print(f"\n1. 이벤트 감지 테스트 ({len(test_news)}개 뉴스):")
        
        # 이벤트 감지 실행
        detected_events = await detector.detect_events(test_news)
        
        print(f"감지된 이벤트: {len(detected_events)}개\n")
        
        for i, event in enumerate(detected_events, 1):
            print(f"이벤트 {i}:")
            print(f"  ID: {event.event_id}")
            print(f"  유형: {event.event_type.value}")
            print(f"  제목: {event.title}")
            print(f"  영향 레벨: {event.impact_level.value}")
            print(f"  기본 영향도: {event.base_impact_score:.3f}")
            print(f"  감정 편향: {event.sentiment_bias:.3f}")
            print(f"  변동성 요인: {event.volatility_factor:.3f}")
            print(f"  최종 영향도: {event.final_impact_score:.3f}")
            print(f"  신뢰도: {event.confidence:.3f}")
            print(f"  뉴스 개수: {event.news_count}")
            print(f"  영향받는 심볼: {event.symbols_affected}")
            print(f"  키워드: {event.keywords[:5]}")
            print()
        
        # 활성 이벤트 조회
        print("2. 활성 이벤트 조회:")
        active_events = detector.get_active_events(min_impact=0.5)
        print(f"활성 이벤트: {len(active_events)}개")
        
        # 심볼별 이벤트 조회
        print("\n3. 심볼별 이벤트 조회:")
        btc_events = detector.get_events_by_symbol("BTC")
        print(f"BTC 관련 이벤트: {len(btc_events)}개")
        
        usd_events = detector.get_events_by_symbol("USD")
        print(f"USD 관련 이벤트: {len(usd_events)}개")
        
        # 통계 확인
        print("\n4. 감지기 통계:")
        stats = detector.get_detector_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_big_event_detector())