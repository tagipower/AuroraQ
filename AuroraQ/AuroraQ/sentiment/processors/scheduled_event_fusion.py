#!/usr/bin/env python3
"""
Scheduled Event Fusion Processor for AuroraQ Sentiment Service
15분 간격 이벤트 융합 점수 처리기 - 모든 컴포넌트 통합
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np
from collections import defaultdict, deque
import statistics

# 로컬 임포트
from ..collectors.macro_indicator_collector import MacroIndicatorCollector, MacroIndicator, IndicatorType
from ..monitors.option_expiry_monitor import OptionExpiryMonitor, ExpiryEvent, ExpiryUrgency
from ..schedulers.event_schedule_loader import EventScheduleLoader, EconomicEvent, EventImportance
from ..processors.event_impact_manager import EventImpactManager, ImpactTimeframe, EventImpactLabel
from ..processors.sentiment_fusion_manager_v2 import SentimentFusionManagerV2, FusedSentimentV2

logger = logging.getLogger(__name__)

class FusionInterval(Enum):
    """융합 간격"""
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"

class EventWeight(Enum):
    """이벤트 가중치"""
    MACRO = 0.35      # 매크로 지표
    ECONOMIC = 0.30   # 경제 이벤트
    EXPIRY = 0.20     # 옵션/선물 만료
    SENTIMENT = 0.15  # 뉴스 감정

@dataclass
class FusionScore:
    """융합 점수"""
    timestamp: datetime
    symbol: str
    interval: FusionInterval
    
    # 개별 점수들
    macro_score: float = 0.0
    economic_score: float = 0.0
    expiry_score: float = 0.0
    sentiment_score: float = 0.0
    
    # 가중 통합 점수
    weighted_score: float = 0.0
    normalized_score: float = 0.0  # -1 ~ 1 정규화
    
    # 메타데이터
    confidence: float = 0.0
    volatility_expected: bool = False
    major_events: List[str] = field(default_factory=list)
    risk_level: str = "low"
    
    # 전략 추천
    strategy_recommendation: str = "neutral"
    position_sizing_factor: float = 1.0

@dataclass
class FusionContext:
    """융합 컨텍스트"""
    current_time: datetime
    symbol: str
    lookback_hours: int = 24
    
    # 시장 상태
    market_session: str = "regular"  # pre, regular, post, closed
    trading_day: bool = True
    volatility_regime: str = "normal"  # low, normal, high, extreme
    
    # 컨텍스트 정보
    recent_major_events: List[str] = field(default_factory=list)
    upcoming_events: List[str] = field(default_factory=list)
    market_stress_level: float = 0.0

class ScheduledEventFusion:
    """스케줄링된 이벤트 융합 처리기"""
    
    def __init__(self, 
                 macro_collector: MacroIndicatorCollector,
                 expiry_monitor: OptionExpiryMonitor,
                 event_loader: EventScheduleLoader,
                 impact_manager: EventImpactManager,
                 sentiment_manager: SentimentFusionManagerV2,
                 fusion_interval: int = 900):  # 15분 (900초)
        """
        초기화
        
        Args:
            macro_collector: 매크로 지표 수집기
            expiry_monitor: 옵션 만료 모니터
            event_loader: 이벤트 스케줄 로더
            impact_manager: 이벤트 영향 관리자
            sentiment_manager: 감정 융합 관리자
            fusion_interval: 융합 간격 (초)
        """
        self.macro_collector = macro_collector
        self.expiry_monitor = expiry_monitor
        self.event_loader = event_loader
        self.impact_manager = impact_manager
        self.sentiment_manager = sentiment_manager
        self.fusion_interval = fusion_interval
        
        # 추적 대상 심볼들
        self.target_symbols = ["BTC", "ETH", "USD", "GOLD", "SPX"]
        
        # 데이터 저장소
        self.fusion_scores: Dict[str, deque] = {}  # symbol -> deque of FusionScore
        self.fusion_history: Dict[str, deque] = {}  # symbol -> 히스토리
        
        # 실행 제어
        self.is_running = False
        self.fusion_task: Optional[asyncio.Task] = None
        
        # 캐시된 결과
        self.cached_contexts: Dict[str, FusionContext] = {}
        self.cache_ttl = 300  # 5분
        
        # 통계
        self.stats = {
            "total_fusions": 0,
            "successful_fusions": 0,
            "failed_fusions": 0,
            "last_fusion": None,
            "avg_fusion_time": 0.0,
            "symbols_processed": 0
        }
        
        # 심볼별 융합 점수 초기화
        for symbol in self.target_symbols:
            self.fusion_scores[symbol] = deque(maxlen=288)  # 3일 * 96 (15분 간격)
            self.fusion_history[symbol] = deque(maxlen=1000)  # 히스토리
    
    async def start(self):
        """융합 처리기 시작"""
        if self.is_running:
            logger.warning("Scheduled event fusion already running")
            return
        
        self.is_running = True
        self.fusion_task = asyncio.create_task(self._fusion_loop())
        logger.info(f"Scheduled event fusion started (interval: {self.fusion_interval}s)")
        
        # 초기 융합 실행
        await self._process_fusion_cycle()
    
    async def stop(self):
        """융합 처리기 중지"""
        self.is_running = False
        
        if self.fusion_task and not self.fusion_task.done():
            self.fusion_task.cancel()
            try:
                await self.fusion_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scheduled event fusion stopped")
    
    async def _fusion_loop(self):
        """융합 루프"""
        while self.is_running:
            try:
                await self._process_fusion_cycle()
                await asyncio.sleep(self.fusion_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fusion loop error: {e}")
                self.stats["failed_fusions"] += 1
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _process_fusion_cycle(self):
        """융합 사이클 처리"""
        start_time = time.time()
        
        try:
            logger.debug("Processing fusion cycle...")
            
            current_time = datetime.now(timezone.utc)
            successful_symbols = 0
            
            # 모든 심볼에 대해 융합 점수 계산
            for symbol in self.target_symbols:
                try:
                    fusion_score = await self._calculate_fusion_score(symbol, current_time)
                    if fusion_score:
                        self.fusion_scores[symbol].append(fusion_score)
                        self.fusion_history[symbol].append(fusion_score)
                        successful_symbols += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process fusion for {symbol}: {e}")
            
            # 통계 업데이트
            fusion_time = time.time() - start_time
            self._update_stats(successful_symbols, fusion_time)
            
            logger.debug(f"Fusion cycle completed: {successful_symbols}/{len(self.target_symbols)} symbols in {fusion_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Fusion cycle failed: {e}")
            self.stats["failed_fusions"] += 1
    
    async def _calculate_fusion_score(self, symbol: str, current_time: datetime) -> Optional[FusionScore]:
        """심볼별 융합 점수 계산"""
        try:
            # 융합 컨텍스트 생성
            context = await self._build_fusion_context(symbol, current_time)
            
            # 개별 점수 계산
            macro_score = await self._calculate_macro_score(symbol, context)
            economic_score = await self._calculate_economic_score(symbol, context)
            expiry_score = await self._calculate_expiry_score(symbol, context)
            sentiment_score = await self._calculate_sentiment_score(symbol, context)
            
            # 가중 통합 점수 계산
            weighted_score = (
                macro_score * EventWeight.MACRO.value +
                economic_score * EventWeight.ECONOMIC.value +
                expiry_score * EventWeight.EXPIRY.value +
                sentiment_score * EventWeight.SENTIMENT.value
            )
            
            # 정규화 (-1 ~ 1)
            normalized_score = np.tanh(weighted_score)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(symbol, context)
            
            # 변동성 예상
            volatility_expected = self._predict_volatility(symbol, context, weighted_score)
            
            # 주요 이벤트 식별
            major_events = self._identify_major_events(symbol, context)
            
            # 리스크 레벨 결정
            risk_level = self._determine_risk_level(weighted_score, context)
            
            # 전략 추천
            strategy_recommendation = self._recommend_strategy(normalized_score, context)
            position_sizing_factor = self._calculate_position_sizing(risk_level, confidence)
            
            fusion_score = FusionScore(
                timestamp=current_time,
                symbol=symbol,
                interval=FusionInterval.MINUTE_15,
                macro_score=macro_score,
                economic_score=economic_score,
                expiry_score=expiry_score,
                sentiment_score=sentiment_score,
                weighted_score=weighted_score,
                normalized_score=normalized_score,
                confidence=confidence,
                volatility_expected=volatility_expected,
                major_events=major_events,
                risk_level=risk_level,
                strategy_recommendation=strategy_recommendation,
                position_sizing_factor=position_sizing_factor
            )
            
            return fusion_score
            
        except Exception as e:
            logger.error(f"Failed to calculate fusion score for {symbol}: {e}")
            return None
    
    async def _build_fusion_context(self, symbol: str, current_time: datetime) -> FusionContext:
        """융합 컨텍스트 구성"""
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M')}"
        
        # 캐시 확인
        if cache_key in self.cached_contexts:
            cached = self.cached_contexts[cache_key]
            if (current_time - cached.current_time).total_seconds() < self.cache_ttl:
                return cached
        
        # 새로운 컨텍스트 생성
        context = FusionContext(
            current_time=current_time,
            symbol=symbol,
            lookback_hours=24
        )
        
        # 시장 세션 결정
        context.market_session = self._determine_market_session(current_time)
        context.trading_day = self._is_trading_day(current_time)
        
        # 변동성 체제 결정
        context.volatility_regime = self._determine_volatility_regime(symbol)
        
        # 최근 주요 이벤트
        context.recent_major_events = await self._get_recent_major_events(symbol, current_time)
        
        # 예정된 이벤트
        context.upcoming_events = await self._get_upcoming_events(symbol, current_time)
        
        # 시장 스트레스 레벨
        context.market_stress_level = await self._calculate_market_stress(current_time)
        
        # 캐시 저장
        self.cached_contexts[cache_key] = context
        
        return context
    
    async def _calculate_macro_score(self, symbol: str, context: FusionContext) -> float:
        """매크로 지표 점수 계산"""
        try:
            if not self.macro_collector.current_indicators:
                return 0.0
            
            relevant_scores = []
            
            # 심볼별 관련 지표 매핑
            symbol_indicators = {
                "BTC": ["^VIX", "DXY", "^GSPC", "^TNX", "GC=F"],
                "ETH": ["^VIX", "^VXN", "^GSPC", "^TNX"],
                "USD": ["DXY", "EURUSD=X", "^TNX"],
                "GOLD": ["GC=F", "DXY", "^TNX", "^VIX"],
                "SPX": ["^GSPC", "^VIX", "^TNX"]
            }
            
            indicators = symbol_indicators.get(symbol, ["^VIX", "DXY"])
            
            for indicator_symbol in indicators:
                indicator = self.macro_collector.get_indicator(indicator_symbol)
                if indicator:
                    # 변화율을 점수로 변환
                    change_score = np.tanh(indicator.change_percent / 5.0)  # 5% 기준 정규화
                    
                    # VIX는 역방향 (높을수록 부정적)
                    if indicator_symbol == "^VIX":
                        change_score = -change_score
                    
                    # 영향도 가중치 적용
                    weighted_score = change_score * indicator.impact_score
                    relevant_scores.append(weighted_score)
            
            return statistics.mean(relevant_scores) if relevant_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate macro score for {symbol}: {e}")
            return 0.0
    
    async def _calculate_economic_score(self, symbol: str, context: FusionContext) -> float:
        """경제 이벤트 점수 계산"""
        try:
            # 다음 24시간 내 이벤트들
            upcoming_events = self.event_loader.get_events_by_timeframe(24)
            
            if not upcoming_events:
                return 0.0
            
            event_scores = []
            
            for event in upcoming_events:
                # 이벤트 라벨링
                impact_label = self.impact_manager.label_economic_event(event)
                
                # 심볼별 영향도
                asset_impact = impact_label.asset_impacts.get(symbol, 0.0)
                
                # 시간 가중치 (가까운 이벤트일수록 높은 가중치)
                time_weight = max(0.1, min(1.0, 24 / max(1, event.time_to_event_hours)))
                
                # 최종 점수
                event_score = impact_label.final_impact_score * asset_impact * time_weight
                event_scores.append(event_score)
            
            return statistics.mean(event_scores) if event_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate economic score for {symbol}: {e}")
            return 0.0
    
    async def _calculate_expiry_score(self, symbol: str, context: FusionContext) -> float:
        """만료 이벤트 점수 계산"""
        try:
            # 긴급 만료 이벤트들
            urgent_expiries = self.expiry_monitor.get_urgent_expiries(ExpiryUrgency.HIGH)
            
            if not urgent_expiries:
                return 0.0
            
            # 심볼 관련 만료 이벤트 필터링
            relevant_expiries = [
                event for event in urgent_expiries
                if event.underlying_asset == symbol or symbol in ["BTC", "ETH"]
            ]
            
            if not relevant_expiries:
                return 0.0
            
            expiry_scores = []
            
            for event in relevant_expiries:
                # 긴급도별 기본 점수
                urgency_scores = {
                    ExpiryUrgency.IMMEDIATE: 1.0,
                    ExpiryUrgency.HIGH: 0.7,
                    ExpiryUrgency.MEDIUM: 0.4,
                    ExpiryUrgency.LOW: 0.2
                }
                
                base_score = urgency_scores.get(event.urgency, 0.2)
                
                # OI 가중치
                oi_weight = min(1.5, event.open_interest / 5000)  # 5000 기준
                
                # 최종 점수
                expiry_score = base_score * oi_weight * event.impact_score
                expiry_scores.append(expiry_score)
            
            return statistics.mean(expiry_scores) if expiry_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate expiry score for {symbol}: {e}")
            return 0.0
    
    async def _calculate_sentiment_score(self, symbol: str, context: FusionContext) -> float:
        """감정 점수 계산"""
        try:
            # 최근 융합된 감정 데이터 가져오기
            recent_sentiment = await self.sentiment_manager.get_latest_fused_sentiment(symbol)
            
            if not recent_sentiment:
                return 0.0
            
            # 감정 점수를 -1 ~ 1 범위로 정규화
            sentiment_score = np.tanh(recent_sentiment.weighted_sentiment)
            
            # 신뢰도 가중치 적용
            confidence_weight = recent_sentiment.confidence_score
            
            return sentiment_score * confidence_weight
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment score for {symbol}: {e}")
            return 0.0
    
    def _calculate_confidence(self, symbol: str, context: FusionContext) -> float:
        """신뢰도 계산"""
        confidence_factors = []
        
        # 데이터 신선도
        if symbol in self.macro_collector.current_indicators:
            indicator = self.macro_collector.current_indicators[symbol]
            data_age = (context.current_time - indicator.timestamp).total_seconds()
            freshness = max(0.5, 1.0 - (data_age / 1800))  # 30분 기준
            confidence_factors.append(freshness)
        
        # 시장 세션 (정규 시간 = 높은 신뢰도)
        session_confidence = 1.0 if context.market_session == "regular" else 0.7
        confidence_factors.append(session_confidence)
        
        # 변동성 체제 (정상 변동성 = 높은 신뢰도)
        volatility_confidence = {
            "low": 0.8,
            "normal": 1.0,
            "high": 0.9,
            "extreme": 0.6
        }.get(context.volatility_regime, 0.8)
        confidence_factors.append(volatility_confidence)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _predict_volatility(self, symbol: str, context: FusionContext, weighted_score: float) -> bool:
        """변동성 예상"""
        # 점수 기반 예측
        if abs(weighted_score) > 0.6:
            return True
        
        # 시장 스트레스 레벨
        if context.market_stress_level > 0.7:
            return True
        
        # 주요 이벤트 임박
        if len(context.upcoming_events) > 2:
            return True
        
        return False
    
    def _identify_major_events(self, symbol: str, context: FusionContext) -> List[str]:
        """주요 이벤트 식별"""
        major_events = []
        
        # 임박한 경제 이벤트
        upcoming = self.event_loader.get_high_impact_events(1)  # 1일 내
        for event in upcoming[:3]:  # 상위 3개만
            if event.importance in [EventImportance.CRITICAL, EventImportance.HIGH]:
                major_events.append(f"Economic: {event.title}")
        
        # 긴급 만료 이벤트
        urgent_expiries = self.expiry_monitor.get_urgent_expiries(ExpiryUrgency.IMMEDIATE)
        for expiry in urgent_expiries[:2]:  # 상위 2개만
            if expiry.underlying_asset == symbol or symbol in ["BTC", "ETH"]:
                major_events.append(f"Expiry: {expiry.instrument_name}")
        
        # 매크로 급변
        if symbol in self.macro_collector.current_indicators:
            indicator = self.macro_collector.current_indicators[symbol]
            if abs(indicator.change_percent) > 2.0:  # 2% 이상
                direction = "상승" if indicator.change_percent > 0 else "하락"
                major_events.append(f"Macro: {indicator.name} {direction} {abs(indicator.change_percent):.1f}%")
        
        return major_events
    
    def _determine_risk_level(self, weighted_score: float, context: FusionContext) -> str:
        """리스크 레벨 결정"""
        if abs(weighted_score) > 0.8 or context.market_stress_level > 0.8:
            return "extreme"
        elif abs(weighted_score) > 0.6 or context.market_stress_level > 0.6:
            return "high"
        elif abs(weighted_score) > 0.3 or context.market_stress_level > 0.4:
            return "medium"
        else:
            return "low"
    
    def _recommend_strategy(self, normalized_score: float, context: FusionContext) -> str:
        """전략 추천"""
        if normalized_score > 0.3:
            return "bullish"
        elif normalized_score < -0.3:
            return "bearish"
        elif context.volatility_regime == "high" and abs(normalized_score) > 0.1:
            return "volatility_trading"
        else:
            return "neutral"
    
    def _calculate_position_sizing(self, risk_level: str, confidence: float) -> float:
        """포지션 사이징 계산"""
        base_sizes = {
            "low": 1.0,
            "medium": 0.8,
            "high": 0.6,
            "extreme": 0.4
        }
        
        base_size = base_sizes.get(risk_level, 0.8)
        confidence_adjusted = base_size * confidence
        
        return max(0.2, min(1.0, confidence_adjusted))
    
    def _determine_market_session(self, current_time: datetime) -> str:
        """시장 세션 결정"""
        # UTC 시간 기준 대략적인 시장 시간
        hour = current_time.hour
        
        if 13 <= hour <= 21:  # 9 AM - 5 PM ET
            return "regular"
        elif 9 <= hour <= 13:  # Pre-market
            return "pre"
        elif 21 <= hour <= 24:  # After-hours
            return "post"
        else:
            return "closed"
    
    def _is_trading_day(self, current_time: datetime) -> bool:
        """거래일 여부"""
        return current_time.weekday() < 5  # 월-금
    
    def _determine_volatility_regime(self, symbol: str) -> str:
        """변동성 체제 결정"""
        vix = self.macro_collector.get_indicator("^VIX")
        if not vix:
            return "normal"
        
        if vix.current_value > 30:
            return "extreme"
        elif vix.current_value > 25:
            return "high"
        elif vix.current_value < 15:
            return "low"
        else:
            return "normal"
    
    async def _get_recent_major_events(self, symbol: str, current_time: datetime) -> List[str]:
        """최근 주요 이벤트"""
        # 과거 24시간 내 이벤트 (간단 구현)
        return []
    
    async def _get_upcoming_events(self, symbol: str, current_time: datetime) -> List[str]:
        """예정된 이벤트"""
        upcoming = self.event_loader.get_events_by_timeframe(24)
        return [event.title for event in upcoming[:5]]
    
    async def _calculate_market_stress(self, current_time: datetime) -> float:
        """시장 스트레스 계산"""
        stress_factors = []
        
        # VIX 기반
        vix = self.macro_collector.get_indicator("^VIX")
        if vix:
            vix_stress = min(1.0, max(0.0, (vix.current_value - 15) / 15))  # 15-30 범위
            stress_factors.append(vix_stress)
        
        # 환율 변동
        dxy = self.macro_collector.get_indicator("DXY")
        if dxy:
            dxy_stress = min(1.0, abs(dxy.change_percent) / 2.0)  # 2% 기준
            stress_factors.append(dxy_stress)
        
        return statistics.mean(stress_factors) if stress_factors else 0.0
    
    def _update_stats(self, successful_symbols: int, fusion_time: float):
        """통계 업데이트"""
        self.stats["total_fusions"] += 1
        self.stats["successful_fusions"] += 1 if successful_symbols > 0 else 0
        self.stats["symbols_processed"] += successful_symbols
        self.stats["last_fusion"] = datetime.now().isoformat()
        
        # 평균 융합 시간
        current_avg = self.stats["avg_fusion_time"]
        total_fusions = self.stats["total_fusions"]
        self.stats["avg_fusion_time"] = (
            (current_avg * (total_fusions - 1) + fusion_time) / total_fusions
        )
    
    def get_latest_fusion_score(self, symbol: str) -> Optional[FusionScore]:
        """최신 융합 점수 조회"""
        if symbol in self.fusion_scores and self.fusion_scores[symbol]:
            return self.fusion_scores[symbol][-1]
        return None
    
    def get_fusion_scores_history(self, symbol: str, hours: int = 24) -> List[FusionScore]:
        """융합 점수 히스토리"""
        if symbol not in self.fusion_scores:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            score for score in self.fusion_scores[symbol]
            if score.timestamp >= cutoff_time
        ]
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """융합 요약"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {},
            "overall_market_sentiment": "neutral",
            "volatility_outlook": "normal",
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "extreme": 0}
        }
        
        sentiment_scores = []
        volatility_count = 0
        
        for symbol in self.target_symbols:
            latest_score = self.get_latest_fusion_score(symbol)
            if latest_score:
                summary["symbols"][symbol] = {
                    "normalized_score": latest_score.normalized_score,
                    "confidence": latest_score.confidence,
                    "risk_level": latest_score.risk_level,
                    "strategy": latest_score.strategy_recommendation,
                    "volatility_expected": latest_score.volatility_expected,
                    "major_events_count": len(latest_score.major_events)
                }
                
                sentiment_scores.append(latest_score.normalized_score)
                if latest_score.volatility_expected:
                    volatility_count += 1
                
                # 리스크 분포
                summary["risk_distribution"][latest_score.risk_level] += 1
        
        # 전체 시장 감정
        if sentiment_scores:
            avg_sentiment = statistics.mean(sentiment_scores)
            if avg_sentiment > 0.2:
                summary["overall_market_sentiment"] = "bullish"
            elif avg_sentiment < -0.2:
                summary["overall_market_sentiment"] = "bearish"
            else:
                summary["overall_market_sentiment"] = "neutral"
        
        # 변동성 전망
        if volatility_count >= len(self.target_symbols) * 0.6:
            summary["volatility_outlook"] = "high"
        elif volatility_count >= len(self.target_symbols) * 0.3:
            summary["volatility_outlook"] = "elevated"
        else:
            summary["volatility_outlook"] = "normal"
        
        return summary
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """융합 통계"""
        return {
            "is_running": self.is_running,
            "fusion_interval": self.fusion_interval,
            "target_symbols": self.target_symbols,
            **self.stats,
            "cached_contexts_count": len(self.cached_contexts),
            "fusion_scores_count": {
                symbol: len(scores) for symbol, scores in self.fusion_scores.items()
            }
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_event_fusion():
        """스케줄링된 이벤트 융합 테스트"""
        
        print("=== Scheduled Event Fusion 테스트 ===")
        
        # 모의 컴포넌트들 (실제로는 초기화된 인스턴스 사용)
        print("주의: 이 테스트는 모든 의존 컴포넌트가 초기화되고 실행 중일 때만 작동합니다.")
        
        # 실제 사용 예시
        """
        # 컴포넌트 초기화
        macro_collector = MacroIndicatorCollector()
        expiry_monitor = OptionExpiryMonitor()
        event_loader = EventScheduleLoader()
        impact_manager = EventImpactManager()
        sentiment_manager = SentimentFusionManagerV2()
        
        # 모든 컴포넌트 시작
        await macro_collector.start()
        await expiry_monitor.start()
        await event_loader.start()
        await sentiment_manager.start()
        
        # 융합 처리기 초기화 및 시작
        fusion_processor = ScheduledEventFusion(
            macro_collector=macro_collector,
            expiry_monitor=expiry_monitor,
            event_loader=event_loader,
            impact_manager=impact_manager,
            sentiment_manager=sentiment_manager,
            fusion_interval=900  # 15분
        )
        
        await fusion_processor.start()
        
        # 30분 대기 후 결과 확인
        await asyncio.sleep(1800)
        
        # 융합 요약
        summary = fusion_processor.get_fusion_summary()
        print("융합 요약:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 심볼별 최신 점수
        for symbol in fusion_processor.target_symbols:
            score = fusion_processor.get_latest_fusion_score(symbol)
            if score:
                print(f"\n{symbol} 최신 융합 점수:")
                print(f"  정규화 점수: {score.normalized_score:.3f}")
                print(f"  신뢰도: {score.confidence:.3f}")
                print(f"  리스크 레벨: {score.risk_level}")
                print(f"  전략 추천: {score.strategy_recommendation}")
                print(f"  주요 이벤트: {score.major_events}")
        
        # 통계
        stats = fusion_processor.get_fusion_stats()
        print(f"\n융합 통계:")
        for key, value in stats.items():
            if key not in ["fusion_scores_count"]:
                print(f"  {key}: {value}")
        
        # 정리
        await fusion_processor.stop()
        await macro_collector.stop()
        await expiry_monitor.stop()
        await event_loader.stop()
        await sentiment_manager.stop()
        """
        
        print("\n=== 테스트 스킵 (의존성 컴포넌트 필요) ===")
    
    # 테스트 실행
    asyncio.run(test_event_fusion())