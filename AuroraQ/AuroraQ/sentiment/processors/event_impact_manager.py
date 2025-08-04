#!/usr/bin/env python3
"""
Event Impact Manager for AuroraQ Sentiment Service
이벤트 감도 라벨링 및 점수화 처리 매니저
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
from ..schedulers.event_schedule_loader import EconomicEvent, EventImportance, EventCategory
from ..monitors.option_expiry_monitor import ExpiryEvent, ExpiryUrgency
from ..collectors.macro_indicator_collector import MacroIndicator, IndicatorType

logger = logging.getLogger(__name__)

class ImpactTimeframe(Enum):
    """영향 시간대"""
    PRE_EVENT = "pre_event"       # 이벤트 전 (1-24시간)
    IMMEDIATE = "immediate"       # 즉시 (0-1시간)
    SHORT_TERM = "short_term"     # 단기 (1-6시간)
    MEDIUM_TERM = "medium_term"   # 중기 (6-24시간)
    LONG_TERM = "long_term"       # 장기 (1-7일)

class MarketRegime(Enum):
    """시장 상황"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"

@dataclass
class EventImpactLabel:
    """이벤트 영향 라벨"""
    event_id: str
    event_type: str  # "economic", "expiry", "macro"
    base_impact_score: float
    
    # 시간대별 영향도
    timeframe_impacts: Dict[ImpactTimeframe, float] = field(default_factory=dict)
    
    # 자산별 영향도
    asset_impacts: Dict[str, float] = field(default_factory=dict)
    
    # 시장 상황별 승수
    regime_multipliers: Dict[MarketRegime, float] = field(default_factory=dict)
    
    # 메타데이터
    confidence_score: float = 0.8
    historical_accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # 계산된 최종 점수
    final_impact_score: float = 0.0
    current_regime_adjusted_score: float = 0.0

@dataclass
class ImpactScenario:
    """영향 시나리오"""
    scenario_id: str
    scenario_name: str
    probability: float  # 0-1
    
    # 시나리오별 예상 영향
    price_impact_range: Tuple[float, float]  # (min%, max%)
    volatility_increase: float  # 변동성 증가율
    duration_hours: float  # 지속 시간
    
    # 트리거 조건
    trigger_conditions: List[str] = field(default_factory=list)
    
    # 관련 자산
    affected_assets: List[str] = field(default_factory=list)

@dataclass
class MarketSensitivityProfile:
    """시장 민감도 프로필"""
    profile_name: str
    
    # 이벤트 타입별 민감도
    economic_sensitivity: float = 1.0
    expiry_sensitivity: float = 1.0
    macro_sensitivity: float = 1.0
    
    # 현재 시장 상황
    current_regime: MarketRegime = MarketRegime.SIDEWAYS
    regime_confidence: float = 0.5
    
    # 과거 반응 패턴
    historical_overreactions: float = 0.0  # 과잉반응 경향
    historical_underreactions: float = 0.0  # 과소반응 경향
    
    # 업데이트 시간
    last_calibrated: datetime = field(default_factory=datetime.now)

class EventImpactManager:
    """이벤트 영향 관리자"""
    
    def __init__(self):
        """초기화"""
        
        # 데이터 저장소
        self.impact_labels: Dict[str, EventImpactLabel] = {}
        self.impact_scenarios: Dict[str, ImpactScenario] = {}
        self.market_profile = MarketSensitivityProfile("default_profile")
        
        # 이벤트 히스토리 (학습용)
        self.event_history: deque = deque(maxlen=500)
        self.impact_history: deque = deque(maxlen=500)
        
        # 캐시된 결과
        self.cached_assessments: Dict[str, Dict] = {}
        self.cache_ttl = 3600  # 1시간
        
        # 기본 라벨링 규칙 로드
        self._load_default_labeling_rules()
        
        # 기본 시나리오 생성
        self._create_default_scenarios()
        
        # 통계
        self.stats = {
            "total_events_labeled": 0,
            "scenarios_generated": 0,
            "market_regime_changes": 0,
            "prediction_accuracy": 0.0,
            "last_calibration": None
        }
    
    def _load_default_labeling_rules(self):
        """기본 라벨링 규칙 로드"""
        
        # 경제 이벤트 기본 라벨
        economic_labels = {
            EventImportance.CRITICAL: {
                "base_impact": 0.9,
                "timeframes": {
                    ImpactTimeframe.PRE_EVENT: 0.3,
                    ImpactTimeframe.IMMEDIATE: 0.9,
                    ImpactTimeframe.SHORT_TERM: 0.7,
                    ImpactTimeframe.MEDIUM_TERM: 0.5,
                    ImpactTimeframe.LONG_TERM: 0.3
                },
                "assets": {
                    "BTC": 0.8, "ETH": 0.7, "USD": 0.9,
                    "GOLD": 0.6, "SPX": 0.8
                }
            },
            EventImportance.HIGH: {
                "base_impact": 0.7,
                "timeframes": {
                    ImpactTimeframe.PRE_EVENT: 0.2,
                    ImpactTimeframe.IMMEDIATE: 0.7,
                    ImpactTimeframe.SHORT_TERM: 0.5,
                    ImpactTimeframe.MEDIUM_TERM: 0.3,
                    ImpactTimeframe.LONG_TERM: 0.2
                },
                "assets": {
                    "BTC": 0.6, "ETH": 0.5, "USD": 0.7,
                    "GOLD": 0.4, "SPX": 0.6
                }
            },
            EventImportance.MEDIUM: {
                "base_impact": 0.5,
                "timeframes": {
                    ImpactTimeframe.PRE_EVENT: 0.1,
                    ImpactTimeframe.IMMEDIATE: 0.5,
                    ImpactTimeframe.SHORT_TERM: 0.3,
                    ImpactTimeframe.MEDIUM_TERM: 0.2,
                    ImpactTimeframe.LONG_TERM: 0.1
                },
                "assets": {
                    "BTC": 0.4, "ETH": 0.3, "USD": 0.5,
                    "GOLD": 0.3, "SPX": 0.4
                }
            }
        }
        
        # 시장 상황별 승수
        regime_multipliers = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_VOLATILITY: 0.7,
            MarketRegime.BULL_MARKET: 1.2,
            MarketRegime.BEAR_MARKET: 1.3,
            MarketRegime.RISK_OFF: 1.4,
            MarketRegime.RISK_ON: 1.1,
            MarketRegime.SIDEWAYS: 1.0
        }
        
        # 기본 라벨 생성
        for importance, config in economic_labels.items():
            label_id = f"economic_{importance.value}"
            
            label = EventImpactLabel(
                event_id=label_id,
                event_type="economic",
                base_impact_score=config["base_impact"],
                timeframe_impacts=config["timeframes"],
                asset_impacts=config["assets"],
                regime_multipliers=regime_multipliers
            )
            
            self.impact_labels[label_id] = label
    
    def _create_default_scenarios(self):
        """기본 시나리오 생성"""
        
        scenarios = [
            # FOMC 매파적 시나리오
            {
                "id": "fomc_hawkish",
                "name": "FOMC Hawkish Surprise",
                "probability": 0.3,
                "price_impact": (-15.0, -5.0),  # 5-15% 하락
                "volatility_increase": 2.0,
                "duration": 24,
                "triggers": ["interest_rate_hike > expected", "hawkish_language"],
                "assets": ["BTC", "ETH", "NASDAQ", "GOLD"]
            },
            
            # FOMC 비둘기파 시나리오
            {
                "id": "fomc_dovish",
                "name": "FOMC Dovish Surprise", 
                "probability": 0.25,
                "price_impact": (5.0, 20.0),  # 5-20% 상승
                "volatility_increase": 1.5,
                "duration": 12,
                "triggers": ["no_rate_hike", "dovish_language", "pause_mentioned"],
                "assets": ["BTC", "ETH", "GOLD", "BONDS"]
            },
            
            # 높은 인플레이션 시나리오
            {
                "id": "high_inflation",
                "name": "High Inflation Print",
                "probability": 0.35,
                "price_impact": (-10.0, 5.0),  # 혼재 반응
                "volatility_increase": 1.8,
                "duration": 6,
                "triggers": ["cpi > 0.5% vs expected", "core_cpi > expected"],
                "assets": ["USD", "BONDS", "BTC", "GOLD"]
            },
            
            # 옵션 만료 대량 시나리오
            {
                "id": "large_options_expiry",
                "name": "Large Options Expiry Event",
                "probability": 0.4,
                "price_impact": (-8.0, 8.0),  # 양방향 변동성
                "volatility_increase": 1.6,
                "duration": 4,
                "triggers": ["options_oi > 50000", "max_pain_distance > 5%"],
                "assets": ["BTC", "ETH"]
            },
            
            # 매크로 쇼크 시나리오
            {
                "id": "macro_shock",
                "name": "Macro Indicator Shock",
                "probability": 0.2,
                "price_impact": (-20.0, -10.0),  # 급락
                "volatility_increase": 3.0,
                "duration": 48,
                "triggers": ["vix > 30", "dxy_change > 2%", "yields_spike > 20bp"],
                "assets": ["BTC", "ETH", "SPX", "GOLD", "USD"]
            }
        ]
        
        for scenario_config in scenarios:
            scenario = ImpactScenario(
                scenario_id=scenario_config["id"],
                scenario_name=scenario_config["name"],
                probability=scenario_config["probability"],
                price_impact_range=scenario_config["price_impact"],
                volatility_increase=scenario_config["volatility_increase"],
                duration_hours=scenario_config["duration"],
                trigger_conditions=scenario_config["triggers"],
                affected_assets=scenario_config["assets"]
            )
            
            self.impact_scenarios[scenario_config["id"]] = scenario
    
    def label_economic_event(self, event: EconomicEvent) -> EventImpactLabel:
        """경제 이벤트 라벨링"""
        
        # 기본 라벨 가져오기
        base_label_id = f"economic_{event.importance.value}"
        base_label = self.impact_labels.get(base_label_id)
        
        if not base_label:
            # 기본값 생성
            base_label = EventImpactLabel(
                event_id=event.event_id,
                event_type="economic",
                base_impact_score=0.5
            )
        
        # 개별 이벤트용 라벨 생성
        event_label = EventImpactLabel(
            event_id=event.event_id,
            event_type="economic",
            base_impact_score=base_label.base_impact_score,
            timeframe_impacts=base_label.timeframe_impacts.copy(),
            asset_impacts=base_label.asset_impacts.copy(),
            regime_multipliers=base_label.regime_multipliers.copy()
        )
        
        # 이벤트별 조정
        self._adjust_for_event_specifics(event_label, event)
        
        # 최종 점수 계산
        self._calculate_final_impact_score(event_label)
        
        # 저장
        self.impact_labels[event.event_id] = event_label
        self.stats["total_events_labeled"] += 1
        
        return event_label
    
    def label_expiry_event(self, event: ExpiryEvent) -> EventImpactLabel:
        """만료 이벤트 라벨링"""
        
        # 긴급도에 따른 기본 영향도
        urgency_impacts = {
            ExpiryUrgency.IMMEDIATE: 0.8,
            ExpiryUrgency.HIGH: 0.6,
            ExpiryUrgency.MEDIUM: 0.4,
            ExpiryUrgency.LOW: 0.2
        }
        
        base_impact = urgency_impacts.get(event.urgency, 0.3)
        
        # OI 크기에 따른 조정
        if event.open_interest > 10000:
            oi_multiplier = 1.3
        elif event.open_interest > 5000:
            oi_multiplier = 1.1
        else:
            oi_multiplier = 1.0
        
        event_label = EventImpactLabel(
            event_id=event.instrument_name,
            event_type="expiry",
            base_impact_score=base_impact * oi_multiplier,
            timeframe_impacts={
                ImpactTimeframe.PRE_EVENT: 0.2,
                ImpactTimeframe.IMMEDIATE: base_impact * oi_multiplier,
                ImpactTimeframe.SHORT_TERM: base_impact * 0.7,
                ImpactTimeframe.MEDIUM_TERM: base_impact * 0.3,
                ImpactTimeframe.LONG_TERM: 0.1
            },
            asset_impacts={
                event.underlying_asset: 0.8,
                "BTC" if event.underlying_asset != "BTC" else "ETH": 0.3
            }
        )
        
        # 최종 점수 계산
        self._calculate_final_impact_score(event_label)
        
        # 저장
        self.impact_labels[event.instrument_name] = event_label
        self.stats["total_events_labeled"] += 1
        
        return event_label
    
    def label_macro_change(self, indicator: MacroIndicator) -> EventImpactLabel:
        """매크로 지표 변화 라벨링"""
        
        # 지표 유형별 기본 영향도
        type_impacts = {
            IndicatorType.VOLATILITY: 0.9,  # VIX 등
            IndicatorType.CURRENCY: 0.8,    # DXY 등
            IndicatorType.BOND: 0.7,        # 10Y Treasury
            IndicatorType.EQUITY: 0.6,      # S&P 500
            IndicatorType.COMMODITY: 0.5,   # Gold, Oil
            IndicatorType.CRYPTO: 0.4       # GBTC 등
        }
        
        base_impact = type_impacts.get(indicator.indicator_type, 0.3)
        
        # 변화율에 따른 조정
        change_multiplier = min(2.0, 1.0 + abs(indicator.change_percent) / 5.0)
        
        # 변동성에 따른 조정
        volatility_multiplier = 1.0 + indicator.volatility_score * 0.5
        
        adjusted_impact = base_impact * change_multiplier * volatility_multiplier
        
        event_label = EventImpactLabel(
            event_id=f"macro_{indicator.symbol}",
            event_type="macro",
            base_impact_score=min(1.0, adjusted_impact),
            timeframe_impacts={
                ImpactTimeframe.IMMEDIATE: adjusted_impact,
                ImpactTimeframe.SHORT_TERM: adjusted_impact * 0.8,
                ImpactTimeframe.MEDIUM_TERM: adjusted_impact * 0.5,
                ImpactTimeframe.LONG_TERM: adjusted_impact * 0.2
            },
            asset_impacts=self._get_macro_asset_impacts(indicator)
        )
        
        # 최종 점수 계산
        self._calculate_final_impact_score(event_label)
        
        # 저장
        label_id = f"macro_{indicator.symbol}"
        self.impact_labels[label_id] = event_label
        self.stats["total_events_labeled"] += 1
        
        return event_label
    
    def _adjust_for_event_specifics(self, label: EventImpactLabel, event: EconomicEvent):
        """이벤트별 특수 조정"""
        
        # 카테고리별 조정
        category_adjustments = {
            EventCategory.INTEREST_RATE: 1.2,  # 금리 이벤트는 영향 증가
            EventCategory.INFLATION: 1.1,
            EventCategory.EMPLOYMENT: 1.0,
            EventCategory.GDP: 0.9,
            EventCategory.FED_SPEECH: 0.8
        }
        
        adjustment = category_adjustments.get(event.category, 1.0)
        label.base_impact_score *= adjustment
        
        # 시간까지 거리에 따른 조정
        if 0 <= event.time_to_event_hours <= 2:
            time_multiplier = 1.3  # 임박한 이벤트
        elif event.time_to_event_hours <= 24:
            time_multiplier = 1.1
        else:
            time_multiplier = 1.0
        
        label.base_impact_score *= time_multiplier
        
        # 주말/시장 시간 조정
        event_time = event.scheduled_time
        if event_time.weekday() >= 5:  # 주말
            for timeframe in label.timeframe_impacts:
                label.timeframe_impacts[timeframe] *= 0.7
    
    def _get_macro_asset_impacts(self, indicator: MacroIndicator) -> Dict[str, float]:
        """매크로 지표의 자산별 영향도"""
        
        impacts = {}
        
        if indicator.symbol in ["^VIX", "^VXN"]:  # 변동성 지수
            impacts = {
                "BTC": 0.8, "ETH": 0.7, "SPX": 0.9,
                "GOLD": 0.3, "USD": 0.4
            }
        elif indicator.symbol == "DXY":  # 달러 지수
            impacts = {
                "BTC": 0.7, "ETH": 0.6, "GOLD": 0.8,
                "EUR": 0.9, "JPY": 0.7
            }
        elif indicator.symbol in ["GC=F"]:  # 금
            impacts = {
                "BTC": 0.5, "USD": 0.6, "BONDS": 0.4,
                "GOLD": 1.0
            }
        elif indicator.symbol in ["^TNX"]:  # 10년 국채
            impacts = {
                "BTC": 0.6, "ETH": 0.5, "USD": 0.8,
                "GOLD": 0.7, "SPX": 0.7
            }
        else:
            # 기본값
            impacts = {
                "BTC": 0.4, "ETH": 0.3, "USD": 0.3
            }
        
        return impacts
    
    def _calculate_final_impact_score(self, label: EventImpactLabel):
        """최종 영향도 점수 계산"""
        
        # 기본 점수
        base_score = label.base_impact_score
        
        # 현재 시장 상황 승수 적용
        current_regime = self.market_profile.current_regime
        regime_multiplier = label.regime_multipliers.get(current_regime, 1.0)
        
        # 최종 점수
        label.final_impact_score = min(1.0, base_score * regime_multiplier)
        label.current_regime_adjusted_score = label.final_impact_score
        
        # 시간대별 점수도 조정
        for timeframe, impact in label.timeframe_impacts.items():
            label.timeframe_impacts[timeframe] = min(1.0, impact * regime_multiplier)
    
    def update_market_regime(self, 
                           vix_level: float,
                           market_trend: float,
                           volatility_regime: str) -> MarketRegime:
        """시장 상황 업데이트"""
        
        old_regime = self.market_profile.current_regime
        
        # VIX 기반 변동성 판단
        if vix_level > 30:
            volatility_regime = MarketRegime.HIGH_VOLATILITY
        elif vix_level < 15:
            volatility_regime = MarketRegime.LOW_VOLATILITY
        else:
            volatility_regime = None
        
        # 트렌드 기반 시장 상황
        if market_trend > 0.1:
            trend_regime = MarketRegime.BULL_MARKET
        elif market_trend < -0.1:
            trend_regime = MarketRegime.BEAR_MARKET
        else:
            trend_regime = MarketRegime.SIDEWAYS
        
        # 최종 판단 (변동성이 우선)
        if volatility_regime:
            new_regime = volatility_regime
        else:
            new_regime = trend_regime
        
        # 리스크 온/오프 판단 (추가 로직)
        if vix_level < 20 and market_trend > 0:
            new_regime = MarketRegime.RISK_ON
        elif vix_level > 25 and market_trend < 0:
            new_regime = MarketRegime.RISK_OFF
        
        # 변경 사항 기록
        if new_regime != old_regime:
            self.market_profile.current_regime = new_regime
            self.market_profile.last_calibrated = datetime.now()
            self.stats["market_regime_changes"] += 1
            
            logger.info(f"Market regime changed: {old_regime.value} -> {new_regime.value}")
            
            # 모든 라벨의 점수 재계산
            self._recalculate_all_impact_scores()
        
        return new_regime
    
    def _recalculate_all_impact_scores(self):
        """모든 영향도 점수 재계산"""
        for label in self.impact_labels.values():
            self._calculate_final_impact_score(label)
    
    def generate_impact_scenarios(self, events: List[Any]) -> List[ImpactScenario]:
        """영향 시나리오 생성"""
        
        active_scenarios = []
        
        for event in events:
            # 이벤트 타입에 따른 시나리오 매칭
            if hasattr(event, 'importance'):  # Economic Event
                if event.importance == EventImportance.CRITICAL:
                    if "fomc" in event.title.lower():
                        active_scenarios.extend([
                            self.impact_scenarios.get("fomc_hawkish"),
                            self.impact_scenarios.get("fomc_dovish")
                        ])
                    elif "cpi" in event.title.lower():
                        active_scenarios.append(
                            self.impact_scenarios.get("high_inflation")
                        )
            
            elif hasattr(event, 'open_interest'):  # Expiry Event
                if event.open_interest > 10000:
                    active_scenarios.append(
                        self.impact_scenarios.get("large_options_expiry")
                    )
            
            elif hasattr(event, 'volatility_score'):  # Macro Indicator
                if event.volatility_score > 0.8:
                    active_scenarios.append(
                        self.impact_scenarios.get("macro_shock")
                    )
        
        # None 제거 및 중복 제거
        active_scenarios = list(set(filter(None, active_scenarios)))
        
        # 확률 정규화
        if active_scenarios:
            total_prob = sum(s.probability for s in active_scenarios)
            for scenario in active_scenarios:
                scenario.probability = scenario.probability / total_prob
        
        self.stats["scenarios_generated"] = len(active_scenarios)
        
        return active_scenarios
    
    def get_combined_impact_assessment(self, 
                                     economic_events: List[EconomicEvent] = None,
                                     expiry_events: List[ExpiryEvent] = None,
                                     macro_indicators: List[MacroIndicator] = None,
                                     timeframe: ImpactTimeframe = ImpactTimeframe.SHORT_TERM) -> Dict[str, Any]:
        """통합 영향 평가"""
        
        cache_key = f"combined_{timeframe.value}_{hash(str([economic_events, expiry_events, macro_indicators]))}"
        
        # 캐시 확인
        if cache_key in self.cached_assessments:
            cached = self.cached_assessments[cache_key]
            if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                return cached["result"]
        
        # 새로운 평가 수행
        all_labels = []
        
        # 각 이벤트 타입별 라벨링
        if economic_events:
            for event in economic_events:
                label = self.label_economic_event(event)
                all_labels.append(label)
        
        if expiry_events:
            for event in expiry_events:
                label = self.label_expiry_event(event)
                all_labels.append(label)
        
        if macro_indicators:
            for indicator in macro_indicators:
                label = self.label_macro_change(indicator)
                all_labels.append(label)
        
        # 통합 점수 계산
        if not all_labels:
            combined_score = 0.0
            asset_impacts = {}
        else:
            # 시간대별 가중 평균
            timeframe_scores = [
                label.timeframe_impacts.get(timeframe, 0.0) 
                for label in all_labels
            ]
            combined_score = statistics.mean(timeframe_scores) if timeframe_scores else 0.0
            
            # 자산별 영향도 집계
            asset_impacts = defaultdict(list)
            for label in all_labels:
                for asset, impact in label.asset_impacts.items():
                    asset_impacts[asset].append(impact * label.final_impact_score)
            
            # 평균 계산
            asset_impacts = {
                asset: statistics.mean(impacts) 
                for asset, impacts in asset_impacts.items()
            }
        
        # 시나리오 생성
        all_events = (economic_events or []) + (expiry_events or []) + (macro_indicators or [])
        scenarios = self.generate_impact_scenarios(all_events)
        
        # 결과 구성
        result = {
            "combined_impact_score": round(combined_score, 3),
            "market_regime": self.market_profile.current_regime.value,
            "regime_confidence": self.market_profile.regime_confidence,
            "timeframe": timeframe.value,
            "asset_impacts": {k: round(v, 3) for k, v in asset_impacts.items()},
            "top_risk_assets": sorted(asset_impacts.items(), key=lambda x: x[1], reverse=True)[:5],
            "active_scenarios": [
                {
                    "name": scenario.scenario_name,
                    "probability": round(scenario.probability, 2),
                    "expected_impact": scenario.price_impact_range,
                    "duration_hours": scenario.duration_hours
                }
                for scenario in scenarios[:3]  # 상위 3개만
            ],
            "volatility_outlook": self._assess_volatility_outlook(all_labels),
            "confidence_score": statistics.mean([label.confidence_score for label in all_labels]) if all_labels else 0.0,
            "assessment_timestamp": datetime.now().isoformat()
        }
        
        # 캐시 저장
        self.cached_assessments[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
        
        return result
    
    def _assess_volatility_outlook(self, labels: List[EventImpactLabel]) -> Dict[str, Any]:
        """변동성 전망 평가"""
        
        if not labels:
            return {"level": "low", "confidence": 0.0, "duration": "short"}
        
        # 평균 영향도
        avg_impact = statistics.mean([label.final_impact_score for label in labels])
        
        # 변동성 레벨 결정
        if avg_impact > 0.8:
            level = "very_high"
        elif avg_impact > 0.6:
            level = "high"
        elif avg_impact > 0.4:
            level = "medium"
        elif avg_impact > 0.2:
            level = "low"
        else:
            level = "very_low"
        
        # 지속 시간 추정
        max_impact = max([label.final_impact_score for label in labels])
        if max_impact > 0.8:
            duration = "extended"  # 24시간+
        elif max_impact > 0.6:
            duration = "medium"    # 6-24시간
        else:
            duration = "short"     # 1-6시간
        
        return {
            "level": level,
            "confidence": round(avg_impact, 2),
            "duration": duration,
            "peak_expected_impact": round(max_impact, 2)
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """매니저 통계"""
        return {
            **self.stats,
            "impact_labels_count": len(self.impact_labels),
            "scenarios_count": len(self.impact_scenarios),
            "cached_assessments": len(self.cached_assessments),
            "current_market_regime": self.market_profile.current_regime.value,
            "regime_confidence": self.market_profile.regime_confidence,
            "last_regime_change": self.market_profile.last_calibrated.isoformat()
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def test_impact_manager():
        """이벤트 영향 매니저 테스트"""
        
        print("=== Event Impact Manager 테스트 ===")
        
        manager = EventImpactManager()
        
        # 1. 기본 라벨링 테스트
        print(f"\n1. 기본 설정:")
        print(f"   기본 라벨 수: {len(manager.impact_labels)}")
        print(f"   기본 시나리오 수: {len(manager.impact_scenarios)}")
        print(f"   현재 시장 상황: {manager.market_profile.current_regime.value}")
        
        # 2. 시장 상황 업데이트 테스트
        print(f"\n2. 시장 상황 업데이트:")
        new_regime = manager.update_market_regime(
            vix_level=25.0,
            market_trend=-0.05,
            volatility_regime="medium"
        )
        print(f"   업데이트된 시장 상황: {new_regime.value}")
        
        # 3. 가상의 경제 이벤트 라벨링
        from ..schedulers.event_schedule_loader import EconomicEvent, EventImportance, EventCategory
        
        mock_economic_event = EconomicEvent(
            event_id="test_fomc",
            title="FOMC Meeting Decision",
            category=EventCategory.INTEREST_RATE,
            importance=EventImportance.CRITICAL,
            scheduled_time=datetime.now(timezone.utc) + timedelta(hours=6)
        )
        
        economic_label = manager.label_economic_event(mock_economic_event)
        print(f"\n3. 경제 이벤트 라벨링:")
        print(f"   이벤트: {mock_economic_event.title}")
        print(f"   기본 영향도: {economic_label.base_impact_score:.2f}")
        print(f"   최종 영향도: {economic_label.final_impact_score:.2f}")
        print(f"   자산별 영향: {economic_label.asset_impacts}")
        
        # 4. 통합 평가 테스트
        print(f"\n4. 통합 영향 평가:")
        assessment = manager.get_combined_impact_assessment(
            economic_events=[mock_economic_event],
            timeframe=ImpactTimeframe.SHORT_TERM
        )
        
        print(f"   통합 영향 점수: {assessment['combined_impact_score']}")
        print(f"   변동성 전망: {assessment['volatility_outlook']['level']}")
        print(f"   상위 위험 자산: {assessment['top_risk_assets'][:3]}")
        if assessment['active_scenarios']:
            print(f"   활성 시나리오: {assessment['active_scenarios'][0]['name']}")
        
        # 5. 통계
        print(f"\n5. 매니저 통계:")
        stats = manager.get_manager_stats()
        for key, value in stats.items():
            if key not in ["last_regime_change"]:
                print(f"   {key}: {value}")
        
        print(f"\n=== 테스트 완료 ===")
    
    # 테스트 실행
    test_impact_manager()