#!/usr/bin/env python3
"""
시장 상황 감지기
변동성, 유동성, 거래량 등을 분석하여 현재 시장 상황을 판단
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging


@dataclass
class MarketCondition:
    """시장 상황 정보"""
    condition: str = "normal"
    confidence: float = 0.0
    
    # 변동성 지표
    volatility_level: str = "normal"  # low, normal, high, extreme
    volatility_percentile: float = 0.0
    
    # 유동성 지표
    liquidity_level: str = "normal"   # high, normal, low, very_low
    spread_widening: float = 0.0
    
    # 거래량 지표
    volume_level: str = "normal"      # low, normal, high, extreme
    volume_ratio: float = 1.0
    
    # 시간대 정보
    trading_session: str = "regular"  # pre_market, regular, after_hours
    
    # 특수 상황
    special_events: List[str] = field(default_factory=list)
    
    # 신뢰도 구성 요소
    data_quality: float = 1.0
    observation_period: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'condition': self.condition,
            'confidence': self.confidence,
            'volatility': {
                'level': self.volatility_level,
                'percentile': self.volatility_percentile
            },
            'liquidity': {
                'level': self.liquidity_level,
                'spread_widening': self.spread_widening
            },
            'volume': {
                'level': self.volume_level,
                'ratio': self.volume_ratio
            },
            'session': self.trading_session,
            'special_events': self.special_events,
            'quality': {
                'data_quality': self.data_quality,
                'observation_period': self.observation_period
            }
        }


class MarketConditionDetector:
    """시장 상황 감지기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 임계값 설정
        self.volatility_thresholds = {
            'low': 0.15,      # 15% 연변동성 이하
            'normal': 0.25,   # 15-25%
            'high': 0.40,     # 25-40%
            'extreme': 0.40   # 40% 초과
        }
        
        self.volume_thresholds = {
            'low': 0.5,       # 평균의 50% 이하
            'normal': 1.5,    # 50-150%
            'high': 3.0,      # 150-300%
            'extreme': 3.0    # 300% 초과
        }
        
        self.spread_thresholds = {
            'normal': 1.2,    # 평균의 120% 이하
            'wide': 2.0,      # 120-200%
            'very_wide': 2.0  # 200% 초과
        }
        
        # 시장 상황 조건
        self.condition_rules = {
            'normal': {
                'volatility': ['low', 'normal'],
                'liquidity': ['high', 'normal'],
                'volume': ['normal'],
                'special_events': []
            },
            'high_volatility': {
                'volatility': ['high', 'extreme'],
                'min_confidence': 0.7
            },
            'low_liquidity': {
                'liquidity': ['low', 'very_low'],
                'min_confidence': 0.6
            },
            'market_stress': {
                'volatility': ['extreme'],
                'liquidity': ['very_low'],
                'min_confidence': 0.8
            },
            'after_hours': {
                'trading_session': ['pre_market', 'after_hours'],
                'min_confidence': 0.9
            }
        }
        
        # 캐시
        self.condition_cache: Dict[str, Tuple[MarketCondition, datetime]] = {}
        self.cache_duration_minutes = 5
    
    def detect_current_condition(self, symbol: str = "MARKET") -> str:
        """현재 시장 상황 감지"""
        
        # 캐시 확인
        cache_key = symbol
        if self._is_cache_valid(cache_key):
            cached_condition, _ = self.condition_cache[cache_key]
            return cached_condition.condition
        
        # 시장 데이터 수집 및 분석
        market_data = self._collect_market_data(symbol)
        condition = self._analyze_market_condition(market_data, symbol)
        
        # 캐시 저장
        self.condition_cache[cache_key] = (condition, datetime.now())
        
        return condition.condition
    
    def get_detailed_condition(self, symbol: str = "MARKET") -> MarketCondition:
        """상세 시장 상황 정보 조회"""
        
        # 캐시 확인
        cache_key = symbol
        if self._is_cache_valid(cache_key):
            cached_condition, _ = self.condition_cache[cache_key]
            return cached_condition
        
        # 시장 데이터 수집 및 분석
        market_data = self._collect_market_data(symbol)
        condition = self._analyze_market_condition(market_data, symbol)
        
        # 캐시 저장
        self.condition_cache[cache_key] = (condition, datetime.now())
        
        return condition
    
    def _collect_market_data(self, symbol: str) -> Dict[str, Any]:
        """시장 데이터 수집"""
        
        # 실제 환경에서는 실시간 시장 데이터를 수집
        # 여기서는 시뮬레이션 데이터 생성
        
        current_time = datetime.now()
        
        # 기본 시장 데이터 (시뮬레이션)
        np.random.seed(int(current_time.timestamp()) % 1000)
        
        # 변동성 데이터 (연환산 기준)
        base_volatility = 0.20  # 20% 기본 변동성
        volatility_shock = np.random.normal(0, 0.05)
        current_volatility = max(0.05, base_volatility + volatility_shock)
        
        # 거래량 데이터
        avg_volume = 1000000  # 평균 거래량
        volume_multiplier = np.random.lognormal(0, 0.5)
        current_volume = avg_volume * volume_multiplier
        
        # 스프레드 데이터
        normal_spread = 0.001  # 0.1% 기본 스프레드
        spread_multiplier = np.random.lognormal(0, 0.3)
        current_spread = normal_spread * spread_multiplier
        
        # 시간대 확인
        trading_session = self._get_trading_session(current_time)
        
        # 특수 이벤트 시뮬레이션
        special_events = []
        if np.random.random() < 0.1:  # 10% 확률로 특수 이벤트
            events = ['earnings_announcement', 'fed_meeting', 'market_news', 'technical_issue']
            special_events.append(np.random.choice(events))
        
        return {
            'symbol': symbol,
            'timestamp': current_time,
            'volatility': {
                'current': current_volatility,
                'historical_avg': base_volatility,
                'percentile_rank': self._calculate_percentile_rank(current_volatility, base_volatility)
            },
            'volume': {
                'current': current_volume,
                'historical_avg': avg_volume,
                'ratio': current_volume / avg_volume
            },
            'liquidity': {
                'bid_ask_spread': current_spread,
                'normal_spread': normal_spread,
                'spread_ratio': current_spread / normal_spread
            },
            'trading_session': trading_session,
            'special_events': special_events,
            'data_quality': np.random.uniform(0.8, 1.0)
        }
    
    def _get_trading_session(self, timestamp: datetime) -> str:
        """거래 세션 확인"""
        
        # 미국 동부 시간 기준 (간단한 근사)
        hour = timestamp.hour
        
        if 4 <= hour < 9:
            return 'pre_market'
        elif 9 <= hour < 16:
            return 'regular'
        elif 16 <= hour < 20:
            return 'after_hours'
        else:
            return 'closed'
    
    def _calculate_percentile_rank(self, current_value: float, historical_avg: float) -> float:
        """백분위 순위 계산 (근사)"""
        
        # 정규분포 가정하에 백분위 계산
        z_score = (current_value - historical_avg) / (historical_avg * 0.3)
        
        # 누적분포함수를 통한 백분위
        from scipy.stats import norm
        percentile = norm.cdf(z_score) * 100
        
        return max(0, min(100, percentile))
    
    def _analyze_market_condition(self, market_data: Dict[str, Any], symbol: str) -> MarketCondition:
        """시장 상황 분석"""
        
        condition = MarketCondition()
        condition.data_quality = market_data.get('data_quality', 1.0)
        condition.observation_period = 1  # 실시간 관찰
        
        # 변동성 분석
        volatility_analysis = self._analyze_volatility(market_data['volatility'])
        condition.volatility_level = volatility_analysis['level']
        condition.volatility_percentile = volatility_analysis['percentile']
        
        # 유동성 분석
        liquidity_analysis = self._analyze_liquidity(market_data['liquidity'])
        condition.liquidity_level = liquidity_analysis['level']
        condition.spread_widening = liquidity_analysis['spread_ratio']
        
        # 거래량 분석
        volume_analysis = self._analyze_volume(market_data['volume'])
        condition.volume_level = volume_analysis['level']
        condition.volume_ratio = volume_analysis['ratio']
        
        # 거래 세션
        condition.trading_session = market_data['trading_session']
        
        # 특수 이벤트
        condition.special_events = market_data.get('special_events', [])
        
        # 전체 시장 상황 판단
        condition.condition, condition.confidence = self._determine_overall_condition(condition)
        
        return condition
    
    def _analyze_volatility(self, volatility_data: Dict[str, float]) -> Dict[str, Any]:
        """변동성 분석"""
        
        current_vol = volatility_data['current']
        percentile = volatility_data['percentile_rank']
        
        # 변동성 수준 분류
        if current_vol <= self.volatility_thresholds['low']:
            level = 'low'
        elif current_vol <= self.volatility_thresholds['normal']:
            level = 'normal'
        elif current_vol <= self.volatility_thresholds['high']:
            level = 'high'
        else:
            level = 'extreme'
        
        return {
            'level': level,
            'percentile': percentile,
            'current': current_vol
        }
    
    def _analyze_liquidity(self, liquidity_data: Dict[str, float]) -> Dict[str, Any]:
        """유동성 분석"""
        
        spread_ratio = liquidity_data['spread_ratio']
        
        # 유동성 수준 분류 (스프레드 역비례)
        if spread_ratio <= self.spread_thresholds['normal']:
            level = 'high'
        elif spread_ratio <= self.spread_thresholds['wide']:
            level = 'normal'
        elif spread_ratio <= self.spread_thresholds['very_wide']:
            level = 'low'
        else:
            level = 'very_low'
        
        return {
            'level': level,
            'spread_ratio': spread_ratio
        }
    
    def _analyze_volume(self, volume_data: Dict[str, float]) -> Dict[str, Any]:
        """거래량 분석"""
        
        volume_ratio = volume_data['ratio']
        
        # 거래량 수준 분류
        if volume_ratio <= self.volume_thresholds['low']:
            level = 'low'
        elif volume_ratio <= self.volume_thresholds['normal']:
            level = 'normal'
        elif volume_ratio <= self.volume_thresholds['high']:
            level = 'high'
        else:
            level = 'extreme'
        
        return {
            'level': level,
            'ratio': volume_ratio
        }
    
    def _determine_overall_condition(self, condition: MarketCondition) -> Tuple[str, float]:
        """전체 시장 상황 판단"""
        
        # 각 조건별 점수 계산
        condition_scores = {}
        
        for cond_name, rules in self.condition_rules.items():
            score = self._calculate_condition_score(condition, rules)
            condition_scores[cond_name] = score
        
        # 최고 점수 조건 선택
        best_condition = max(condition_scores, key=condition_scores.get)
        confidence = condition_scores[best_condition]
        
        # 최소 신뢰도 체크
        min_confidence = self.condition_rules[best_condition].get('min_confidence', 0.5)
        if confidence < min_confidence:
            best_condition = 'normal'
            confidence = 0.8
        
        return best_condition, confidence
    
    def _calculate_condition_score(self, 
                                 condition: MarketCondition, 
                                 rules: Dict[str, Any]) -> float:
        """조건별 점수 계산"""
        
        score = 0.0
        total_weight = 0.0
        
        # 변동성 조건
        if 'volatility' in rules:
            weight = 0.3
            if condition.volatility_level in rules['volatility']:
                score += weight
            total_weight += weight
        
        # 유동성 조건
        if 'liquidity' in rules:
            weight = 0.3
            if condition.liquidity_level in rules['liquidity']:
                score += weight
            total_weight += weight
        
        # 거래량 조건
        if 'volume' in rules:
            weight = 0.2
            if condition.volume_level in rules['volume']:
                score += weight
            total_weight += weight
        
        # 거래 세션 조건
        if 'trading_session' in rules:
            weight = 0.2
            if condition.trading_session in rules['trading_session']:
                score += weight
            total_weight += weight
        
        # 특수 이벤트 조건
        if 'special_events' in rules:
            weight = 0.1
            required_events = rules['special_events']
            if any(event in condition.special_events for event in required_events):
                score += weight
            total_weight += weight
        
        # 정규화
        return score / total_weight if total_weight > 0 else 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        
        if cache_key not in self.condition_cache:
            return False
        
        _, cache_time = self.condition_cache[cache_key]
        current_time = datetime.now()
        
        return (current_time - cache_time).total_seconds() < (self.cache_duration_minutes * 60)
    
    def get_condition_history(self, 
                            symbol: str = "MARKET",
                            lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """시장 상황 이력 조회 (시뮬레이션)"""
        
        # 실제 환경에서는 저장된 이력 데이터를 조회
        # 여기서는 시뮬레이션 데이터 생성
        
        history = []
        current_time = datetime.now()
        
        for i in range(lookback_hours):
            timestamp = current_time - timedelta(hours=i)
            
            # 시뮬레이션 조건 생성
            np.random.seed(int(timestamp.timestamp()) % 1000)
            
            conditions = ['normal', 'high_volatility', 'low_liquidity', 'after_hours']
            weights = [0.6, 0.2, 0.1, 0.1]
            condition = np.random.choice(conditions, p=weights)
            
            history.append({
                'timestamp': timestamp.isoformat(),
                'condition': condition,
                'confidence': np.random.uniform(0.6, 0.9)
            })
        
        return list(reversed(history))  # 시간순 정렬
    
    def clear_cache(self):
        """캐시 정리"""
        self.condition_cache.clear()
        self.logger.info("시장 상황 캐시 정리 완료")
    
    def update_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]):
        """임계값 업데이트"""
        
        if 'volatility' in new_thresholds:
            self.volatility_thresholds.update(new_thresholds['volatility'])
        
        if 'volume' in new_thresholds:
            self.volume_thresholds.update(new_thresholds['volume'])
        
        if 'spread' in new_thresholds:
            self.spread_thresholds.update(new_thresholds['spread'])
        
        # 캐시 무효화
        self.clear_cache()
        
        self.logger.info("시장 상황 감지 임계값 업데이트 완료")
    
    def get_condition_statistics(self, 
                               symbol: str = "MARKET",
                               lookback_days: int = 7) -> Dict[str, Any]:
        """시장 상황 통계"""
        
        # 이력 데이터 조회
        history = self.get_condition_history(symbol, lookback_days * 24)
        
        if not history:
            return {}
        
        # 조건별 빈도
        condition_counts = {}
        for record in history:
            condition = record['condition']
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        total_records = len(history)
        condition_percentages = {
            cond: (count / total_records) * 100 
            for cond, count in condition_counts.items()
        }
        
        # 신뢰도 통계
        confidences = [record['confidence'] for record in history]
        
        return {
            'period_days': lookback_days,
            'total_observations': total_records,
            'condition_distribution': condition_percentages,
            'confidence_stats': {
                'avg': np.mean(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'std': np.std(confidences)
            },
            'most_common_condition': max(condition_counts, key=condition_counts.get),
            'current_condition': self.detect_current_condition(symbol)
        }