#!/usr/bin/env python3
"""
시장 시뮬레이터 - 실제 거래 환경 모델링
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from .trade_executor import MarketData


@dataclass
class MarketState:
    """시장 상태"""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: float  # 변동성 수준
    volume_profile: str  # 'low', 'normal', 'high'
    market_impact: float  # 시장 충격 계수


class MarketSimulator:
    """시장 시뮬레이터"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 slippage: float = 0.0005,
                 market_impact_factor: float = 0.001):
        self.data = data.copy()
        self.slippage = slippage
        self.market_impact_factor = market_impact_factor
        
        # 현재 상태
        self.current_index = 0
        self.current_data = None
        self.market_state = MarketState(
            trend='sideways',
            volatility=0.02,
            volume_profile='normal',
            market_impact=0.001
        )
        
        # 시장 통계
        self._calculate_market_stats()
        
    def _calculate_market_stats(self):
        """시장 통계 사전 계산"""
        # 수익률 계산
        self.data['returns'] = self.data['close'].pct_change()
        
        # 변동성 계산 (20일 이동평균)
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        
        # 거래량 프로파일
        volume_mean = self.data['volume'].rolling(20).mean()
        volume_std = self.data['volume'].rolling(20).std()
        self.data['volume_zscore'] = (self.data['volume'] - volume_mean) / volume_std
        
        # 트렌드 계산 (50일 이동평균 기준)
        self.data['ma_50'] = self.data['close'].rolling(50).mean()
        self.data['trend_strength'] = (self.data['close'] - self.data['ma_50']) / self.data['ma_50']
        
        # 시장 충격 계수 계산
        self.data['market_impact'] = self._calculate_market_impact()
    
    def _calculate_market_impact(self) -> pd.Series:
        """시장 충격 계수 계산"""
        # 거래량 역수에 비례하는 시장 충격
        volume_impact = 1 / (self.data['volume'] / self.data['volume'].median()).clip(0.1, 10)
        
        # 변동성에 비례하는 충격
        volatility_impact = self.data['volatility'].fillna(0.02) / 0.02
        
        # 결합
        market_impact = (volume_impact * volatility_impact * self.market_impact_factor).clip(0, 0.01)
        
        return market_impact
    
    def update(self, timestamp: datetime, row: pd.Series) -> MarketData:
        """시장 데이터 업데이트"""
        
        # 현재 데이터 설정
        self.current_data = row
        
        # 시장 상태 업데이트
        self._update_market_state(row)
        
        # MarketData 객체 생성
        market_data = MarketData(
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            bid=self._calculate_bid_price(row['close']),
            ask=self._calculate_ask_price(row['close'])
        )
        
        self.current_index += 1
        
        return market_data
    
    def _update_market_state(self, row: pd.Series):
        """시장 상태 업데이트"""
        
        # 트렌드 판정
        if 'trend_strength' in row and not pd.isna(row['trend_strength']):
            if row['trend_strength'] > 0.02:
                self.market_state.trend = 'bullish'
            elif row['trend_strength'] < -0.02:
                self.market_state.trend = 'bearish'
            else:
                self.market_state.trend = 'sideways'
        
        # 변동성 업데이트
        if 'volatility' in row and not pd.isna(row['volatility']):
            self.market_state.volatility = row['volatility']
        
        # 거래량 프로파일
        if 'volume_zscore' in row and not pd.isna(row['volume_zscore']):
            zscore = row['volume_zscore']
            if zscore > 1.5:
                self.market_state.volume_profile = 'high'
            elif zscore < -1.0:
                self.market_state.volume_profile = 'low'
            else:
                self.market_state.volume_profile = 'normal'
        
        # 시장 충격 업데이트
        if 'market_impact' in row and not pd.isna(row['market_impact']):
            self.market_state.market_impact = row['market_impact']
    
    def _calculate_bid_price(self, close_price: float) -> float:
        """매수 호가 계산"""
        # 기본 스프레드
        base_spread = 0.001  # 0.1%
        
        # 변동성 기반 스프레드 조정
        volatility_adjustment = self.market_state.volatility * 2
        
        # 거래량 기반 스프레드 조정
        if self.market_state.volume_profile == 'low':
            volume_adjustment = 0.002
        elif self.market_state.volume_profile == 'high':
            volume_adjustment = 0.0005
        else:
            volume_adjustment = 0.001
        
        total_spread = base_spread + volatility_adjustment + volume_adjustment
        
        return close_price * (1 - total_spread / 2)
    
    def _calculate_ask_price(self, close_price: float) -> float:
        """매도 호가 계산"""
        # 기본 스프레드
        base_spread = 0.001  # 0.1%
        
        # 변동성 기반 스프레드 조정
        volatility_adjustment = self.market_state.volatility * 2
        
        # 거래량 기반 스프레드 조정
        if self.market_state.volume_profile == 'low':
            volume_adjustment = 0.002
        elif self.market_state.volume_profile == 'high':
            volume_adjustment = 0.0005
        else:
            volume_adjustment = 0.001
        
        total_spread = base_spread + volatility_adjustment + volume_adjustment
        
        return close_price * (1 + total_spread / 2)
    
    def simulate_market_impact(self, 
                             order_size: float, 
                             current_price: float,
                             order_type: str) -> float:
        """주문 시장 충격 시뮬레이션"""
        
        # 주문 크기에 비례한 충격
        if self.current_data is not None and 'volume' in self.current_data:
            avg_volume = self.current_data['volume']
            relative_size = order_size * current_price / (avg_volume * current_price / 1000)
        else:
            relative_size = 0.01
        
        # 기본 충격
        base_impact = self.market_state.market_impact * relative_size
        
        # 시장 상태별 조정
        if self.market_state.volume_profile == 'low':
            impact_multiplier = 2.0
        elif self.market_state.volume_profile == 'high':
            impact_multiplier = 0.5
        else:
            impact_multiplier = 1.0
        
        total_impact = base_impact * impact_multiplier
        
        # 방향별 적용
        if order_type == 'buy':
            return current_price * (1 + total_impact)
        else:
            return current_price * (1 - total_impact)
    
    def get_market_condition(self) -> Dict:
        """현재 시장 상황 반환"""
        return {
            'trend': self.market_state.trend,
            'volatility': self.market_state.volatility,
            'volume_profile': self.market_state.volume_profile,
            'market_impact': self.market_state.market_impact,
            'current_index': self.current_index,
            'total_bars': len(self.data)
        }
    
    def get_historical_volatility(self, periods: int = 20) -> float:
        """과거 변동성 계산"""
        if self.current_index < periods:
            return 0.02  # 기본값
        
        start_idx = max(0, self.current_index - periods)
        end_idx = self.current_index
        
        period_data = self.data.iloc[start_idx:end_idx]
        returns = period_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.02
        
        return returns.std() * np.sqrt(252)  # 연환산
    
    def get_volume_profile(self, periods: int = 20) -> Dict:
        """거래량 프로파일 분석"""
        if self.current_index < periods:
            return {
                'avg_volume': 1000000,
                'volume_trend': 'stable',
                'volume_percentile': 50
            }
        
        start_idx = max(0, self.current_index - periods)
        end_idx = self.current_index
        
        period_data = self.data.iloc[start_idx:end_idx]
        
        avg_volume = period_data['volume'].mean()
        current_volume = self.current_data['volume'] if self.current_data is not None else avg_volume
        
        # 거래량 트렌드
        if current_volume > avg_volume * 1.5:
            volume_trend = 'increasing'
        elif current_volume < avg_volume * 0.7:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
        
        # 거래량 백분위
        volume_percentile = (period_data['volume'] <= current_volume).mean() * 100
        
        return {
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'volume_trend': volume_trend,
            'volume_percentile': volume_percentile
        }
    
    def reset(self):
        """시뮬레이터 초기화"""
        self.current_index = 0
        self.current_data = None
        self.market_state = MarketState(
            trend='sideways',
            volatility=0.02,
            volume_profile='normal',
            market_impact=0.001
        )