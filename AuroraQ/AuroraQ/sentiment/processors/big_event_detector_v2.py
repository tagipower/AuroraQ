#!/usr/bin/env python3
"""
Big Event Detection System V2 for AuroraQ Sentiment Service
VPS 최적화 버전 - 메모리 효율성, 알고리즘 개선, 실시간 알림 통합
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
from collections import defaultdict, deque
import hashlib

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
    MARKET_MANIPULATION = "market_manipulation"
    EXCHANGE_LISTING = "exchange_listing"

class EventImpact(Enum):
    """이벤트 영향도"""
    CRITICAL = "critical"     # 9.0+
    HIGH = "high"             # 7.0-8.9
    MEDIUM = "medium"         # 5.0-6.9
    LOW = "low"               # 3.0-4.9
    MINIMAL = "minimal"       # 0.0-2.9

class EventUrgency(Enum):
    """이벤트 긴급도"""
    IMMEDIATE = "immediate"   # 즉시 알림
    HIGH = "high"            # 5분 내 알림
    NORMAL = "normal"        # 15분 내 알림
    LOW = "low"              # 30분 내 알림

@dataclass
class BigEvent:
    """대형 이벤트 (최적화)"""
    event_id: str
    event_type: EventType
    title: str
    description: str
    impact_level: EventImpact
    urgency: EventUrgency
    base_impact_score: float  # 0.0 ~ 10.0
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
        """최종 영향도 점수 계산 (개선된 공식)"""
        # 기본 공식: base_score * (1 + sentiment_factor) * (1 + volatility_factor)
        sentiment_multiplier = 1 + abs(self.sentiment_bias) * 0.3
        volatility_multiplier = 1 + self.volatility_factor * 0.5
        
        # 신뢰도에 따른 조정
        confidence_factor = 0.5 + (self.confidence * 0.5)
        
        self.final_impact_score = (
            self.base_impact_score *
            sentiment_multiplier *
            volatility_multiplier *
            confidence_factor
        )
        
        return min(10.0, self.final_impact_score)  # 최대 10.0으로 제한
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (메모리 효율적)"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "title": self.title[:200],  # 길이 제한
            "description": self.description[:500],
            "impact_level": self.impact_level.value,
            "urgency": self.urgency.value,
            "base_impact_score": round(self.base_impact_score, 2),
            "sentiment_bias": round(self.sentiment_bias, 3),
            "volatility_factor": round(self.volatility_factor, 3),
            "final_impact_score": round(self.final_impact_score, 2),
            "confidence": round(self.confidence, 3),
            "detected_at": self.detected_at.isoformat(),
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "source_urls": self.source_urls[:5],  # 최대 5개 URL
            "keywords": self.keywords[:10],
            "symbols_affected": self.symbols_affected,
            "news_count": self.news_count
        }

class BigEventDetectorV2:
    """대형 이벤트 감지기 V2 - VPS 최적화"""
    
    def __init__(self, 
                 data_validator: Optional[Any] = None,
                 cache_manager: Optional[Any] = None,
                 notification_integration: Optional[Any] = None):
        """
        초기화
        
        Args:
            data_validator: 데이터 품질 검증기
            cache_manager: 캐시 매니저
            notification_integration: 알림 통합
        """
        self.data_validator = data_validator
        self.cache_manager = cache_manager
        self.notification_integration = notification_integration
        
        # 이벤트 패턴 정의 (최적화 및 확장)
        self.event_patterns = {
            EventType.FOMC_MEETING: {
                "keywords": ["fomc", "federal reserve", "fed meeting", "interest rate", "monetary policy", "jerome powell"],
                "required_terms": ["fed", "meeting"],
                "exclude_terms": ["past", "previous", "last month"],
                "base_impact": 8.5,
                "volatility_base": 0.8,
                "urgency": EventUrgency.IMMEDIATE
            },
            EventType.CPI_RELEASE: {
                "keywords": ["cpi", "inflation", "consumer price index", "price data", "inflation rate"],
                "required_terms": ["cpi", "inflation"],
                "exclude_terms": ["forecast", "prediction"],
                "base_impact": 7.5,
                "volatility_base": 0.7,
                "urgency": EventUrgency.HIGH
            },
            EventType.ETF_APPROVAL: {
                "keywords": ["etf approval", "sec approval", "bitcoin etf", "crypto etf", "spot etf"],
                "required_terms": ["etf"],
                "include_terms": ["approval", "approved", "sec"],
                "base_impact": 9.0,
                "volatility_base": 0.9,
                "urgency": EventUrgency.IMMEDIATE
            },
            EventType.REGULATION: {
                "keywords": ["regulation", "sec", "regulatory", "compliance", "policy", "law"],
                "required_terms": ["regulation", "regulatory"],
                "exclude_terms": ["future", "potential"],
                "base_impact": 6.5,
                "volatility_base": 0.6,
                "urgency": EventUrgency.HIGH
            },
            EventType.INSTITUTIONAL_INVESTMENT: {
                "keywords": ["institutional", "investment", "fund", "grayscale", "blackrock", "microstrategy"],
                "required_terms": ["institutional"],
                "include_terms": ["investment", "buy", "purchase"],
                "base_impact": 5.5,
                "volatility_base": 0.5,
                "urgency": EventUrgency.NORMAL
            },
            EventType.SECURITY_INCIDENT: {
                "keywords": ["hack", "security", "breach", "exploit", "vulnerability", "stolen"],
                "required_terms": ["hack", "breach", "exploit"],
                "exclude_terms": ["prevented", "failed"],
                "base_impact": 7.0,
                "volatility_base": 0.8,
                "urgency": EventUrgency.IMMEDIATE
            },
            EventType.EXCHANGE_LISTING: {
                "keywords": ["listing", "coinbase", "binance", "exchange", "trading pair"],
                "required_terms": ["listing", "exchange"],
                "include_terms": ["new", "added"],
                "base_impact": 4.0,
                "volatility_base": 0.4,
                "urgency": EventUrgency.NORMAL
            },
            EventType.MARKET_MANIPULATION: {
                "keywords": ["manipulation", "pump", "dump", "whale", "coordinated"],
                "required_terms": ["manipulation", "pump", "whale"],
                "base_impact": 6.0,
                "volatility_base": 0.7,
                "urgency": EventUrgency.HIGH
            }
        }
        
        # 심볼별 영향도 매핑 (확장)
        self.symbol_impact_mapping = {
            "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency", "digital gold"],
            "ETH": ["ethereum", "eth", "ether", "defi", "smart contract"],
            "ALTCOIN": ["altcoin", "alt", "alternative", "shitcoin"],
            "STOCK": ["stock", "equity", "share", "market", "s&p", "nasdaq"],
            "FOREX": ["forex", "currency", "dollar", "euro", "yen"],
            "COMMODITY": ["gold", "oil", "commodity", "precious", "energy"]
        }
        
        # 감지된 이벤트 캐시 (메모리 효율적)
        self.detected_events: Dict[str, BigEvent] = {}
        self.event_history: deque = deque(maxlen=1000)  # 최근 1000개 이벤트
        self.event_ttl = 86400  # 24시간
        
        # 임계값 설정
        self.min_news_count = 2  # 최소 뉴스 개수 (감소)
        self.min_confidence = 0.5  # 최소 신뢰도 (감소)
        self.similarity_threshold = 0.75  # 중복 이벤트 임계값
        
        # 성능 최적화
        self.text_cache = {}  # 텍스트 분석 캐시
        self.cache_size_limit = 500
        
        # 통계
        self.stats = {
            "total_detected": 0,
            "by_type": defaultdict(int),
            "by_impact": defaultdict(int),
            "by_urgency": defaultdict(int),
            "avg_confidence": 0.0,
            "processing_time_avg": 0.0,
            "notifications_sent": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _get_text_features_cached(self, text: str) -> Dict[str, Any]:
        """캐시된 텍스트 특성 추출"""
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        
        if cache_key in self.text_cache:
            self.stats["cache_hits"] += 1
            return self.text_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        features = self._extract_text_features(text)
        
        # 캐시 크기 제한
        if len(self.text_cache) >= self.cache_size_limit:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.text_cache))
            del self.text_cache[oldest_key]
        
        self.text_cache[cache_key] = features
        return features
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """텍스트 특성 추출 (최적화)"""
        if not text:
            return {}
        
        text_lower = text.lower()
        
        # 키워드 빈도 (중요 키워드만)
        important_keywords = {
            "bitcoin", "ethereum", "crypto", "fed", "sec", "etf", "regulation",
            "hack", "breach", "approval", "inflation", "cpi", "fomc"
        }
        
        keyword_freq = {}
        words = re.findall(r'\b\w+\b', text_lower)[:200]  # 최대 200단어
        
        for word in words:
            if word in important_keywords:
                keyword_freq[word] = keyword_freq.get(word, 0) + 1
        
        # 감정 키워드
        positive_keywords = ["surge", "rally", "growth", "approval", "positive", "bullish", "rise"]
        negative_keywords = ["crash", "drop", "decline", "negative", "bearish", "fall", "hack"]
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        sentiment_score = 0.0
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        # 긴급도 키워드
        urgency_keywords = ["breaking", "urgent", "immediate", "alert", "now", "just", "confirmed"]
        urgency_count = sum(1 for kw in urgency_keywords if kw in text_lower)
        
        return {
            "keyword_freq": keyword_freq,
            "sentiment_score": sentiment_score,
            "urgency_count": urgency_count,
            "text_length": len(text),
            "word_count": len(words)
        }
    
    def _detect_event_type_optimized(self, news_items: List[Any]) -> Optional[Tuple[EventType, float]]:
        """이벤트 유형 감지 (최적화)"""
        
        if not news_items:
            return None
        
        # 모든 텍스트 통합
        combined_text = " ".join([f"{item.title} {item.content}" for item in news_items[:10]]).lower()
        
        # 각 이벤트 유형별 점수 계산
        type_scores = {}
        
        for event_type, pattern in self.event_patterns.items():
            score = 0.0
            
            # 필수 용어 확인
            required_matches = sum(1 for term in pattern["required_terms"] if term.lower() in combined_text)
            if required_matches == 0:
                continue
            
            # 키워드 매칭 점수
            keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword.lower() in combined_text)
            keyword_score = keyword_matches / len(pattern["keywords"])
            
            # 포함 용어 보너스
            include_matches = 0
            if "include_terms" in pattern:
                include_matches = sum(1 for term in pattern["include_terms"] if term.lower() in combined_text)
                include_score = include_matches / len(pattern["include_terms"])
            else:
                include_score = 0
            
            # 제외 용어 페널티
            exclude_penalty = 0
            if "exclude_terms" in pattern:
                exclude_matches = sum(1 for term in pattern["exclude_terms"] if term.lower() in combined_text)
                exclude_penalty = exclude_matches * 0.2
            
            # 최종 점수 계산
            score = (keyword_score * 0.6 + include_score * 0.3 + required_matches * 0.1) - exclude_penalty
            
            if score > 0:
                type_scores[event_type] = score
        
        # 가장 높은 점수의 이벤트 유형 선택
        if type_scores:
            best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
            if best_score >= 0.3:  # 최소 임계값
                return best_type, best_score
        
        return None
    
    def _calculate_impact_metrics(self, news_items: List[Any], event_type: EventType) -> Dict[str, float]:
        """영향 메트릭 계산 (최적화)"""
        
        pattern = self.event_patterns[event_type]
        
        # 뉴스 메타데이터 분석
        news_count = len(news_items)
        sources = set(item.source.value for item in news_items)
        source_diversity = len(sources) / news_count if news_count > 0 else 0
        
        # 시간 집중도
        times = [item.published_at for item in news_items if item.published_at]
        if len(times) > 1:
            time_span = (max(times) - min(times)).total_seconds() / 3600  # 시간 단위
            time_concentration = max(0.1, 1.0 - (time_span / 12.0))  # 12시간 기준
        else:
            time_concentration = 1.0
        
        # 관련성 점수 평균
        avg_relevance = statistics.mean(item.relevance_score for item in news_items)
        
        # 텍스트 특성 분석
        combined_text = " ".join([f"{item.title} {item.content}" for item in news_items])
        text_features = self._get_text_features_cached(combined_text)
        
        # 감정 편향
        sentiment_bias = text_features.get("sentiment_score", 0.0)
        
        # 변동성 요인
        base_volatility = pattern["volatility_base"]
        news_factor = min(1.0, news_count / 5.0)
        urgency_factor = min(1.0, text_features.get("urgency_count", 0) / 3.0)
        
        volatility_factor = base_volatility * (
            0.4 * news_factor +
            0.3 * time_concentration +
            0.2 * source_diversity +
            0.1 * urgency_factor
        )
        
        # 신뢰도 계산
        confidence_factors = [
            min(1.0, news_count / 3.0),  # 뉴스 개수
            source_diversity,  # 소스 다양성
            avg_relevance,  # 관련성
            time_concentration,  # 시간 집중도
        ]
        confidence = statistics.mean(confidence_factors)
        
        return {
            "sentiment_bias": sentiment_bias,
            "volatility_factor": min(1.0, volatility_factor),
            "confidence": confidence,
            "news_count": news_count,
            "source_diversity": source_diversity,
            "time_concentration": time_concentration
        }
    
    def _determine_symbols_affected(self, news_items: List[Any]) -> List[str]:
        """영향받는 심볼 결정 (최적화)"""
        
        affected_symbols = set()
        combined_text = " ".join([f"{item.title} {item.content}" for item in news_items]).lower()
        
        # 직접 심볼 매칭
        for item in news_items:
            if item.symbol:
                affected_symbols.add(item.symbol.upper())
        
        # 키워드 기반 심볼 매칭
        for symbol, keywords in self.symbol_impact_mapping.items():
            if any(keyword in combined_text for keyword in keywords):
                affected_symbols.add(symbol)
        
        return list(affected_symbols)[:5]  # 최대 5개 심볼
    
    def _generate_event_id(self, event_type: EventType, news_items: List[Any]) -> str:
        """이벤트 ID 생성 (최적화)"""
        
        if news_items:
            representative_title = news_items[0].title[:50]  # 길이 제한
            time_str = news_items[0].published_at.strftime("%Y%m%d%H")
        else:
            representative_title = "unknown"
            time_str = datetime.now().strftime("%Y%m%d%H")
        
        content = f"{event_type.value}_{representative_title}_{time_str}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _is_duplicate_event(self, new_event: BigEvent) -> Optional[str]:
        """중복 이벤트 확인 (최적화)"""
        
        # 최근 6시간 내 같은 유형 이벤트만 확인
        cutoff_time = datetime.now() - timedelta(hours=6)
        
        for existing_id, existing_event in self.detected_events.items():
            if (existing_event.event_type == new_event.event_type and
                existing_event.detected_at >= cutoff_time):
                
                # 제목 유사도 확인 (간단한 방법)
                title1_words = set(new_event.title.lower().split())
                title2_words = set(existing_event.title.lower().split())
                
                if title1_words and title2_words:
                    intersection = len(title1_words & title2_words)
                    union = len(title1_words | title2_words)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity >= self.similarity_threshold:
                        return existing_id
        
        return None
    
    def _determine_urgency(self, event_type: EventType, impact_score: float, text_features: Dict) -> EventUrgency:
        """긴급도 결정"""
        
        # 패턴에서 기본 긴급도
        base_urgency = self.event_patterns[event_type].get("urgency", EventUrgency.NORMAL)
        
        # 영향도 기반 조정
        if impact_score >= 8.0:
            return EventUrgency.IMMEDIATE
        elif impact_score >= 6.0 and base_urgency == EventUrgency.IMMEDIATE:
            return EventUrgency.IMMEDIATE
        elif impact_score >= 6.0:
            return EventUrgency.HIGH
        
        # 긴급 키워드 기반 조정
        urgency_count = text_features.get("urgency_count", 0)
        if urgency_count >= 2:
            if base_urgency == EventUrgency.NORMAL:
                return EventUrgency.HIGH
            elif base_urgency == EventUrgency.LOW:
                return EventUrgency.NORMAL
        
        return base_urgency
    
    async def detect_events(self, news_items: List[Any]) -> List[BigEvent]:
        """뉴스 아이템들에서 대형 이벤트 감지 (최적화)"""
        
        start_time = time.time()
        logger.info(f"Detecting big events from {len(news_items)} news items...")
        
        if not news_items:
            return []
        
        # 데이터 품질 검증 (선택적)
        validated_items = news_items
        if self.data_validator and len(news_items) > 10:
            try:
                # 큰 배치만 검증하여 성능 최적화
                content_items = [self._create_content_item(item) for item in news_items]
                validated_content, _ = self.data_validator.validate_batch(
                    content_items, min_threshold=0.3
                )
                validated_hashes = {item.content_id for item in validated_content}
                validated_items = [item for item in news_items if item.hash_id in validated_hashes]
                
                logger.info(f"Data validation: {len(validated_items)}/{len(news_items)} items passed")
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
        
        # 이벤트 유형 감지
        detection_result = self._detect_event_type_optimized(validated_items)
        if not detection_result:
            logger.debug("No event type detected")
            return []
        
        event_type, type_confidence = detection_result
        
        # 최소 뉴스 개수 확인
        if len(validated_items) < self.min_news_count:
            logger.debug(f"Insufficient news count: {len(validated_items)}")
            return []
        
        # 영향 메트릭 계산
        metrics = self._calculate_impact_metrics(validated_items, event_type)
        
        # 신뢰도 확인
        if metrics["confidence"] < self.min_confidence:
            logger.debug(f"Low confidence: {metrics['confidence']:.3f}")
            return []
        
        # 영향받는 심볼
        affected_symbols = self._determine_symbols_affected(validated_items)
        
        # 기본 영향도 점수
        base_impact = self.event_patterns[event_type]["base_impact"]
        
        # 텍스트 특성
        combined_text = " ".join([f"{item.title} {item.content}" for item in validated_items])
        text_features = self._get_text_features_cached(combined_text)
        
        # 긴급도 결정
        urgency = self._determine_urgency(event_type, base_impact, text_features)
        
        # 대표 제목과 설명 생성
        title = validated_items[0].title
        description = f"{len(validated_items)} news items detected for {event_type.value}"
        
        # 키워드 추출 (상위 키워드만)
        keywords = list(text_features.get("keyword_freq", {}).keys())[:8]
        
        # 이벤트 생성
        event = BigEvent(
            event_id=self._generate_event_id(event_type, validated_items),
            event_type=event_type,
            title=title,
            description=description,
            impact_level=self._determine_impact_level(base_impact),
            urgency=urgency,
            base_impact_score=base_impact,
            sentiment_bias=metrics["sentiment_bias"],
            volatility_factor=metrics["volatility_factor"],
            final_impact_score=0.0,  # 아래에서 계산
            confidence=metrics["confidence"],
            detected_at=datetime.now(),
            event_time=min(item.published_at for item in validated_items if item.published_at),
            source_urls=[item.url for item in validated_items[:3]],  # 최대 3개 URL
            keywords=keywords,
            symbols_affected=affected_symbols,
            news_count=len(validated_items)
        )
        
        # 최종 영향도 계산
        event.calculate_final_impact()
        
        # 중복 확인
        duplicate_id = self._is_duplicate_event(event)
        if duplicate_id:
            # 기존 이벤트 업데이트
            existing_event = self.detected_events[duplicate_id]
            existing_event.news_count += len(validated_items)
            existing_event.confidence = max(existing_event.confidence, event.confidence)
            logger.info(f"Updated existing event: {duplicate_id}")
            return [existing_event]
        
        # 새 이벤트 추가
        self.detected_events[event.event_id] = event
        self.event_history.append(event)
        
        # 통계 업데이트
        self._update_stats(event)
        
        # 실시간 알림 발송
        await self._send_notification_if_needed(event)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Big event detected: {event_type.value} "
                   f"(impact: {event.final_impact_score:.2f}, "
                   f"confidence: {event.confidence:.3f}, "
                   f"urgency: {urgency.value}) in {processing_time:.2f}s")
        
        # 처리 시간 통계 업데이트
        self._update_processing_time_stats(processing_time)
        
        return [event]
    
    def _create_content_item(self, news_item: Any) -> Any:
        """뉴스 아이템을 ContentItem으로 변환"""
        # 실제 구현에서는 적절한 ContentItem 생성
        # 여기서는 간단한 더미 구현
        return type('ContentItem', (), {
            'content_id': news_item.hash_id,
            'title': news_item.title,
            'content': news_item.content
        })()
    
    def _update_stats(self, event: BigEvent):
        """통계 업데이트"""
        self.stats["total_detected"] += 1
        self.stats["by_type"][event.event_type.value] += 1
        self.stats["by_impact"][event.impact_level.value] += 1
        self.stats["by_urgency"][event.urgency.value] += 1
        
        # 평균 신뢰도 업데이트
        total = self.stats["total_detected"]
        current_avg = self.stats["avg_confidence"]
        self.stats["avg_confidence"] = (
            (current_avg * (total - 1) + event.confidence) / total
        )
    
    def _update_processing_time_stats(self, processing_time: float):
        """처리 시간 통계 업데이트"""
        total = self.stats["total_detected"]
        current_avg = self.stats["processing_time_avg"]
        
        if total > 0:
            self.stats["processing_time_avg"] = (
                (current_avg * (total - 1) + processing_time) / total
            )
    
    async def _send_notification_if_needed(self, event: BigEvent):
        """필요시 알림 발송"""
        
        if not self.notification_integration:
            return
        
        try:
            # 긴급도와 영향도에 따른 알림 발송
            should_notify = (
                event.urgency == EventUrgency.IMMEDIATE or
                event.final_impact_score >= 7.0 or
                (event.urgency == EventUrgency.HIGH and event.final_impact_score >= 5.0)
            )
            
            if should_notify:
                await self.notification_integration.send_big_event_notification(event)
                self.stats["notifications_sent"] += 1
                logger.info(f"Notification sent for event: {event.event_id}")
        
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _determine_impact_level(self, base_impact: float) -> EventImpact:
        """기본 영향도에서 영향 레벨 결정"""
        
        if base_impact >= 9.0:
            return EventImpact.CRITICAL
        elif base_impact >= 7.0:
            return EventImpact.HIGH
        elif base_impact >= 5.0:
            return EventImpact.MEDIUM
        elif base_impact >= 3.0:
            return EventImpact.LOW
        else:
            return EventImpact.MINIMAL
    
    def get_active_events(self, 
                         min_impact: float = 5.0,
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
        return symbol_events[:10]  # 최대 10개
    
    def cleanup_old_events(self) -> int:
        """오래된 이벤트 정리"""
        
        cutoff_time = datetime.now() - timedelta(seconds=self.event_ttl)
        
        old_event_ids = []
        for event_id, event in self.detected_events.items():
            if event.detected_at < cutoff_time:
                old_event_ids.append(event_id)
        
        for event_id in old_event_ids:
            del self.detected_events[event_id]
        
        # 텍스트 캐시도 정리
        if len(self.text_cache) > self.cache_size_limit // 2:
            self.text_cache.clear()
        
        logger.info(f"Cleaned up {len(old_event_ids)} old events")
        return len(old_event_ids)
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """감지기 통계 반환"""
        
        return {
            "active_events": len(self.detected_events),
            "event_history_size": len(self.event_history),
            "event_ttl_hours": self.event_ttl / 3600,
            "min_news_count": self.min_news_count,
            "min_confidence": self.min_confidence,
            "similarity_threshold": self.similarity_threshold,
            "supported_event_types": len(self.event_patterns),
            "symbol_mappings": len(self.symbol_impact_mapping),
            "cache_stats": {
                "text_cache_size": len(self.text_cache),
                "cache_hit_rate": (
                    self.stats["cache_hits"] / 
                    (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
                )
            },
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
        
        detector = BigEventDetectorV2()
        
        print("=== 대형 이벤트 감지기 V2 테스트 ===")
        
        # Mock NewsItem 클래스
        class MockNewsItem:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # 테스트 뉴스 아이템들 생성
        test_news = [
            MockNewsItem(
                hash_id="fed_news_1",
                title="Fed Raises Interest Rates by 0.75% in Aggressive Move",
                content="The Federal Reserve raised interest rates by 0.75 percentage points at its latest FOMC meeting...",
                url="https://reuters.com/fed-rate-hike",
                source=type('Source', (), {'value': 'reuters'})(),
                published_at=datetime.now() - timedelta(hours=2),
                category="economics",
                symbol="USD",
                relevance_score=0.95
            ),
            MockNewsItem(
                hash_id="fed_news_2",
                title="FOMC Meeting Results: Federal Reserve Signals More Rate Hikes",
                content="Following today's FOMC meeting, Fed Chair Jerome Powell indicated additional rate increases...",
                url="https://bloomberg.com/fomc-results",
                source=type('Source', (), {'value': 'bloomberg'})(),
                published_at=datetime.now() - timedelta(hours=1),
                category="economics", 
                symbol="USD",
                relevance_score=0.92
            ),
            MockNewsItem(
                hash_id="btc_etf_1",
                title="Bitcoin ETF Approval: SEC Gives Green Light to Spot Bitcoin ETFs",
                content="The Securities and Exchange Commission has approved the first spot Bitcoin ETF applications...",
                url="https://coindesk.com/bitcoin-etf-approval",
                source=type('Source', (), {'value': 'coindesk'})(),
                published_at=datetime.now() - timedelta(hours=3),
                category="crypto",
                symbol="BTC",
                relevance_score=0.98
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
            print(f"  긴급도: {event.urgency.value}")
            print(f"  기본 영향도: {event.base_impact_score:.2f}")
            print(f"  최종 영향도: {event.final_impact_score:.2f}")
            print(f"  신뢰도: {event.confidence:.3f}")
            print(f"  영향받는 심볼: {event.symbols_affected}")
            print(f"  키워드: {event.keywords[:5]}")
            print()
        
        # 통계 확인
        print("2. 감지기 통계:")
        stats = detector.get_detector_stats()
        for key, value in stats.items():
            if key not in ["by_type", "by_impact", "by_urgency", "cache_stats"]:
                print(f"  {key}: {value}")
        
        print(f"\n3. 캐시 통계:")
        cache_stats = stats.get("cache_stats", {})
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
    
    # 테스트 실행
    asyncio.run(test_big_event_detector())