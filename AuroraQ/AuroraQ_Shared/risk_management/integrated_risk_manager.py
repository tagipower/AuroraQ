#!/usr/bin/env python3
"""
AuroraQ Integrated Risk Management System
========================================

Consolidated risk management system combining:
1. Basic risk evaluation from core/risk_manager.py
2. Advanced VaR/CVaR analysis from advanced_risk_manager.py  
3. Enhanced position sizing and stop management
4. Transaction cost optimization
5. Market regime detection
6. Real-time risk monitoring

This module serves as the unified risk management hub for all AuroraQ components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..utils.logger import get_logger
from .advanced_risk_manager import AdvancedRiskManager
from .risk_models import RiskConfig, RiskMetrics, RiskAlert, RiskLevel, AlertType

logger = get_logger("IntegratedRiskManager")

# Market Regime Enums from core/risk_manager.py
class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"

class PositionSizeMethod(Enum):
    FIXED_AMOUNT = "fixed_amount"
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"

# Risk Parameters from core/risk_manager.py
@dataclass
class BasicRiskParameters:
    """기본 리스크 관리 파라미터"""
    # 가격 관련
    min_price: float = 1.0
    max_price_drop_pct: float = 0.20  # 20% 하락 시 경고
    
    # 감정 점수 관련
    sentiment_extreme_fear: float = 0.1
    sentiment_extreme_greed: float = 0.9
    sentiment_delta_threshold: float = 0.5
    
    # 변동성 관련
    max_volatility: float = 0.5
    min_volatility: float = 0.01
    
    # 포지션 관련
    default_stop_loss: float = 0.05  # 5%
    default_take_profit: float = 0.10  # 10%
    max_leverage: float = 5.0
    min_leverage: float = 0.5
    
    # 자본 배분
    max_position_size_pct: float = 0.25  # 전체 자본의 25%
    min_position_size: float = 100.0  # 최소 포지션 크기

# Strategy Risk Profiles from core/risk_manager.py
STRATEGY_RISK_PROFILES = {
    "RuleStrategyA": {"risk_level": "low", "max_allocation": 0.20, "stop_loss": 0.03},
    "RuleStrategyB": {"risk_level": "medium", "max_allocation": 0.25, "stop_loss": 0.05},
    "RuleStrategyC": {"risk_level": "medium", "max_allocation": 0.25, "stop_loss": 0.05},
    "RuleStrategyD": {"risk_level": "high", "max_allocation": 0.30, "stop_loss": 0.07},
    "RuleStrategyE": {"risk_level": "low", "max_allocation": 0.20, "stop_loss": 0.03},
    "PPOStrategy": {"risk_level": "adaptive", "max_allocation": 0.50, "stop_loss": 0.05},
}

class RiskMonitor:
    """실시간 리스크 모니터링"""
    
    def __init__(self, max_daily_loss: float = 0.05, max_trades: int = 50):
        self.max_daily_loss = max_daily_loss
        self.max_trades = max_trades
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """일일 한도 체크"""
        # 날짜 변경 시 리셋
        if datetime.now().date() > self.last_reset:
            self.reset_daily_stats()
        
        # 일일 손실 한도
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"일일 최대 손실 도달: {self.daily_pnl:.1%}"
        
        # 일일 거래 횟수 한도
        if self.daily_trades >= self.max_trades:
            return False, f"일일 최대 거래 횟수 도달: {self.daily_trades}"
        
        return True, "일일 한도 정상"
    
    def update_trade(self, pnl: float):
        """거래 업데이트"""
        self.daily_pnl += pnl
        self.daily_trades += 1
    
    def reset_daily_stats(self):
        """일일 통계 리셋"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()

class EnhancedRiskFilter:
    """향상된 리스크 필터 시스템"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def detect_market_regime(self, price_df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """시장 체제 감지"""
        if len(price_df) < self.lookback_period:
            return MarketRegime.SIDEWAYS, 0.5
        
        recent_data = price_df.tail(self.lookback_period)
        
        # 트렌드 분석
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
        
        # 변동성 분석
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # 체제 결정 로직
        confidence = 0.0
        
        if price_change > 0.1 and volatility < 0.3:
            regime = MarketRegime.BULL
            confidence = min(abs(price_change) * 5, 1.0)
        elif price_change < -0.1 and volatility < 0.3:
            regime = MarketRegime.BEAR  
            confidence = min(abs(price_change) * 5, 1.0)
        elif volatility > 0.5:
            regime = MarketRegime.CRISIS
            confidence = min((volatility - 0.3) * 1.5, 1.0)
        elif volatility > 0.3:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min((volatility - 0.2) * 2, 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = max(0.3, 1.0 - abs(price_change) * 2)
            
        return regime, confidence

class IntegratedRiskManager:
    """통합 리스크 관리 시스템"""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.basic_params = BasicRiskParameters()
        
        # 고급 리스크 관리자 초기화
        self.advanced_manager = AdvancedRiskManager(config=self.config)
        
        # 리스크 필터 및 모니터
        self.risk_filter = EnhancedRiskFilter()
        self.risk_monitor = RiskMonitor()
        
        # 상태 추적
        self.last_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        
        logger.info("IntegratedRiskManager initialized")
    
    # ========== Basic Risk Evaluation (from core/risk_manager.py) ==========
    
    def evaluate_risk(self,
                     price_df: pd.DataFrame,
                     strategy_name: str,
                     sentiment_score: Optional[float] = None,
                     sentiment_delta: Optional[float] = None,
                     position: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """종합 리스크 평가"""
        try:
            # 1. 데이터 유효성 검사
            if price_df is None or len(price_df) < 1:
                return False, "데이터 부족"
            
            close_series = price_df.get("close")
            if close_series is None or close_series.empty:
                return False, "가격 데이터 없음"
            
            current_price = float(close_series.iloc[-1])
            
            # 2. 가격 리스크 평가
            price_risk = self._evaluate_price_risk(price_df, current_price)
            if not price_risk[0]:
                return price_risk
            
            # 3. 감정 리스크 평가
            if sentiment_score is not None or sentiment_delta is not None:
                sentiment_risk = self._evaluate_sentiment_risk(sentiment_score, sentiment_delta)
                if not sentiment_risk[0]:
                    return sentiment_risk
            
            # 4. 변동성 리스크 평가
            volatility_risk = self._evaluate_volatility_risk(price_df)
            if not volatility_risk[0]:
                return volatility_risk
            
            # 5. 전략별 리스크 평가
            strategy_risk = self._evaluate_strategy_risk(strategy_name, price_df)
            if not strategy_risk[0]:
                return strategy_risk
            
            # 6. 포지션 리스크 평가 (이미 포지션이 있는 경우)
            if position:
                position_risk = self._evaluate_position_risk(position, current_price)
                if not position_risk[0]:
                    return position_risk
            
            # 7. 시장 체제 기반 추가 검사
            regime, confidence = self.risk_filter.detect_market_regime(price_df)
            self.last_regime = regime
            self.regime_confidence = confidence
            
            if regime == MarketRegime.CRISIS and confidence > 0.9:
                return False, f"극단적 위기 상황 감지 (신뢰도: {confidence:.2f})"
            
            return True, "리스크 평가 통과"
            
        except Exception as e:
            logger.error(f"리스크 평가 중 오류 발생: {e}")
            return False, f"리스크 평가 오류: {str(e)}"
    
    def _evaluate_price_risk(self, price_df: pd.DataFrame, current_price: float) -> Tuple[bool, str]:
        """가격 관련 리스크 평가"""
        # 최소 가격 체크
        if current_price < self.basic_params.min_price:
            return False, f"가격 너무 낮음: ${current_price:.2f}"
        
        # 급락 체크 (1시간 기준)
        if len(price_df) >= 12:  # 5분봉 12개 = 1시간
            hour_ago_price = float(price_df["close"].iloc[-12])
            price_drop = (hour_ago_price - current_price) / hour_ago_price
            
            if price_drop > self.basic_params.max_price_drop_pct:
                return False, f"급락 감지: {price_drop:.1%} in 1시간"
        
        return True, "가격 리스크 정상"
    
    def _evaluate_sentiment_risk(self,
                                sentiment_score: Optional[float],
                                sentiment_delta: Optional[float]) -> Tuple[bool, str]:
        """감정 관련 리스크 평가"""
        if sentiment_score is not None:
            # 극단적 공포
            if sentiment_score < self.basic_params.sentiment_extreme_fear:
                return False, f"극단적 공포 상태: {sentiment_score:.2f}"
            
            # 극단적 탐욕 (과열)
            if sentiment_score > self.basic_params.sentiment_extreme_greed:
                return False, f"극단적 탐욕 상태: {sentiment_score:.2f}"
        
        if sentiment_delta is not None:
            # 감정 급변
            if abs(sentiment_delta) > self.basic_params.sentiment_delta_threshold:
                return False, f"감정 급변: Δ{sentiment_delta:.2f}"
        
        return True, "감정 리스크 정상"
    
    def _evaluate_volatility_risk(self, price_df: pd.DataFrame) -> Tuple[bool, str]:
        """변동성 리스크 평가"""
        if len(price_df) < 20:
            return True, "데이터 부족으로 변동성 체크 생략"
        
        # 수익률 계산
        returns = price_df["close"].pct_change().dropna()
        volatility = returns.std()
        
        if volatility > self.basic_params.max_volatility:
            return False, f"과도한 변동성: {volatility:.2%}"
        
        if volatility < self.basic_params.min_volatility:
            return False, f"변동성 너무 낮음: {volatility:.4%}"
        
        return True, "변동성 정상"
    
    def _evaluate_strategy_risk(self, strategy_name: str, price_df: pd.DataFrame) -> Tuple[bool, str]:
        """전략별 특수 리스크 평가"""
        profile = STRATEGY_RISK_PROFILES.get(strategy_name, {})
        risk_level = profile.get("risk_level", "medium")
        
        # 고위험 전략의 경우 추가 체크
        if risk_level == "high":
            # 거래량 체크
            if "volume" in price_df.columns:
                avg_volume = price_df["volume"].mean()
                recent_volume = price_df["volume"].iloc[-1]
                
                if recent_volume < avg_volume * 0.5:
                    return False, "거래량 부족 (고위험 전략)"
        
        return True, "전략 리스크 정상"
    
    def _evaluate_position_risk(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """기존 포지션 리스크 평가"""
        entry_price = position.get("entry_price", 0)
        if entry_price <= 0:
            return True, "진입가 정보 없음"
        
        # 손실률 계산
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 최대 허용 손실 체크
        max_loss = position.get("max_loss", 0.10)  # 기본 10%
        if pnl_pct < -max_loss:
            return False, f"최대 손실 초과: {pnl_pct:.1%}"
        
        # 보유 기간 체크
        entry_time = position.get("timestamp")
        if entry_time:
            holding_time = datetime.now() - entry_time
            if holding_time > timedelta(days=30):
                return False, f"장기 보유: {holding_time.days}일"
        
        return True, "포지션 리스크 정상"
    
    # ========== Position Sizing and Capital Allocation ==========
    
    def allocate_capital(self,
                        total_capital: float,
                        strategy_name: str,
                        market_conditions: Optional[Dict[str, Any]] = None) -> float:
        """자본 배분"""
        # 전략 프로필 가져오기
        profile = STRATEGY_RISK_PROFILES.get(strategy_name, {})
        base_allocation = profile.get("max_allocation", 0.20)
        
        # 시장 상황에 따른 조정
        if market_conditions:
            volatility = market_conditions.get("volatility", 0.02)
            sentiment = market_conditions.get("sentiment", 0.5)
            
            # 높은 변동성 시 배분 축소
            if volatility > 0.03:
                base_allocation *= 0.8
            
            # 극단적 감정 시 배분 축소
            if sentiment < 0.2 or sentiment > 0.8:
                base_allocation *= 0.7
        
        # 시장 체제에 따른 조정
        if self.last_regime == MarketRegime.CRISIS:
            base_allocation *= 0.5
        elif self.last_regime == MarketRegime.HIGH_VOLATILITY:
            base_allocation *= 0.7
        
        # 최종 배분 계산
        allocated = total_capital * base_allocation
        
        # 최소/최대 제한
        min_allocation = max(self.basic_params.min_position_size, total_capital * 0.05)
        max_allocation = total_capital * self.basic_params.max_position_size_pct
        
        allocated = max(min_allocation, min(allocated, max_allocation))
        
        logger.info(
            f"자본 배분 - {strategy_name}: "
            f"${allocated:,.2f} ({allocated/total_capital:.1%})"
        )
        
        return allocated
    
    def calculate_position_size(self,
                              capital: float,
                              entry_price: float,
                              stop_loss_pct: float,
                              risk_per_trade: float = 0.02,
                              method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_BASED,
                              volatility: float = 0.2,
                              win_rate: float = 0.5,
                              avg_win: float = 0.02,
                              avg_loss: float = -0.02) -> int:
        """향상된 포지션 크기 계산"""
        
        if method == PositionSizeMethod.FIXED_AMOUNT:
            return int(50000 / entry_price)
            
        elif method == PositionSizeMethod.FIXED_FRACTIONAL:
            target_amount = capital * 0.15  # 15%
            return int(target_amount / entry_price)
            
        elif method == PositionSizeMethod.KELLY:
            if avg_loss >= 0:
                avg_loss = -0.02
            
            kelly_ratio = (win_rate * avg_win + (1 - win_rate) * avg_loss) / abs(avg_loss)
            kelly_ratio = max(0, min(kelly_ratio, 0.20))  # 0-20% 제한
            
            target_amount = capital * kelly_ratio
            return int(target_amount / entry_price)
            
        elif method == PositionSizeMethod.VOLATILITY_BASED:
            if volatility <= 0:
                volatility = 0.2
                
            # 변동성이 높을수록 작은 포지션
            vol_adjustment = min(0.15 / volatility, 1.0)
            base_fraction = 0.12  # 기본 12%
            
            target_amount = capital * base_fraction * vol_adjustment
            return int(target_amount / entry_price)
            
        elif method == PositionSizeMethod.RISK_PARITY:
            target_vol = 0.12  # 목표 변동성 12%
            position_vol = max(volatility, 0.05)
            
            leverage = target_vol / position_vol
            leverage = min(leverage, 1.5)  # 최대 1.5배
            
            target_amount = capital * 0.15 * leverage
            return int(target_amount / entry_price)
        
        else:
            # 기본 리스크 기반 계산
            if stop_loss_pct <= 0:
                stop_loss_pct = self.basic_params.default_stop_loss
            
            # 리스크 금액 계산
            risk_amount = capital * risk_per_trade
            
            # 포지션 크기 계산
            position_value = risk_amount / stop_loss_pct
            position_size = position_value / entry_price
            
            # 최대 포지션 제한
            max_position_value = capital * self.basic_params.max_position_size_pct
            max_position_size = max_position_value / entry_price
            
            final_size = min(position_size, max_position_size)
            return int(final_size)
    
    # ========== Stop Loss and Take Profit Management ==========
    
    def should_cut_loss_or_take_profit(self,
                                     entry_price: float,
                                     current_price: float,
                                     strategy_name: Optional[str] = None,
                                     position_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], str]:
        """손절/익절 판단"""
        try:
            if entry_price <= 0 or current_price <= 0:
                return None, "유효하지 않은 가격"
            
            # 손익률 계산
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 전략별 임계값 가져오기
            profile = STRATEGY_RISK_PROFILES.get(strategy_name, {})
            stop_loss = position_info.get("stop_loss") if position_info else None
            take_profit = position_info.get("take_profit") if position_info else None
            
            if stop_loss is None:
                stop_loss = profile.get("stop_loss", self.basic_params.default_stop_loss)
            if take_profit is None:
                take_profit = self.basic_params.default_take_profit
            
            # 시장 체제에 따른 동적 조정
            if self.last_regime == MarketRegime.HIGH_VOLATILITY:
                stop_loss *= 1.5  # 변동성 높을 때 손절 완화
                take_profit *= 1.2
            elif self.last_regime == MarketRegime.CRISIS:
                stop_loss *= 0.7  # 위기 시 손절 강화
                take_profit *= 0.8
            
            # 손절 체크
            if pnl_pct <= -stop_loss:
                return "cut_loss", f"손절 기준 도달: {pnl_pct:.1%}"
            
            # 익절 체크
            if pnl_pct >= take_profit:
                return "take_profit", f"익절 목표 달성: {pnl_pct:.1%}"
            
            # Trailing Stop (수익 중인 경우)
            if position_info and pnl_pct > 0.02:  # 2% 이상 수익
                highest_price = position_info.get("highest_price", current_price)
                if current_price < highest_price * 0.98:  # 최고점 대비 2% 하락
                    return "trailing_stop", f"트레일링 스탑: {pnl_pct:.1%}"
            
            return None, "보유 유지"
            
        except Exception as e:
            logger.error(f"손절/익절 판단 오류: {e}")
            return None, f"판단 오류: {str(e)}"
    
    def calculate_dynamic_stops(self,
                              entry_price: float,
                              volatility: float = 0.2,
                              regime: Optional[MarketRegime] = None) -> Dict[str, float]:
        """동적 스탑 레벨 계산"""
        
        if regime is None:
            regime = self.last_regime
        
        # 변동성 기반 조정
        vol_multiplier = max(0.7, min(1.5, volatility / 0.2))
        
        # 시장 체제 기반 조정
        regime_multipliers = {
            MarketRegime.BULL: 0.8,      # 상승장에서는 손절을 좁게
            MarketRegime.BEAR: 1.3,      # 하락장에서는 손절을 넓게
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.CRISIS: 2.0,
            MarketRegime.SIDEWAYS: 1.0
        }
        
        regime_mult = regime_multipliers.get(regime, 1.0)
        final_multiplier = vol_multiplier * regime_mult
        
        adjusted_stop_loss = self.basic_params.default_stop_loss * final_multiplier
        adjusted_take_profit = self.basic_params.default_take_profit * final_multiplier
        
        return {
            "stop_loss": entry_price * (1 - adjusted_stop_loss),
            "take_profit": entry_price * (1 + adjusted_take_profit),
            "trailing_stop": entry_price * (1 - 0.025 * final_multiplier)
        }
    
    # ========== Advanced Risk Management Integration ==========
    
    def comprehensive_risk_check(self,
                                price_df: pd.DataFrame,
                                strategy_name: str,
                                current_capital: float,
                                current_positions: Dict,
                                signal_strength: float = 1.0,
                                sentiment_score: Optional[float] = None) -> Dict:
        """종합 리스크 검사 (고급 기능 포함)"""
        
        result = {
            "approved": False,
            "position_size": 0,
            "stop_levels": {},
            "expected_costs": {},
            "reasons": [],
            "risk_metrics": None
        }
        
        try:
            # 1. 기본 리스크 평가
            basic_risk_ok, basic_reason = self.evaluate_risk(
                price_df, strategy_name, sentiment_score
            )
            if not basic_risk_ok:
                result["reasons"].append(f"Basic Risk: {basic_reason}")
                return result
            
            # 2. 일일 한도 체크
            daily_ok, daily_reason = self.risk_monitor.check_daily_limits()
            if not daily_ok:
                result["reasons"].append(f"Daily Limits: {daily_reason}")
                return result
            
            # 3. 고급 리스크 분석 (가능한 경우)
            if hasattr(self.advanced_manager, 'comprehensive_risk_check'):
                advanced_result = self.advanced_manager.comprehensive_risk_check(
                    price_df, strategy_name, current_capital, current_positions, signal_strength, sentiment_score
                )
                
                if not advanced_result.get("approved", False):
                    result["reasons"].extend(advanced_result.get("reasons", []))
                    return result
                
                # 고급 분석 결과 통합
                result.update({
                    "position_size": advanced_result.get("position_size", 0),
                    "stop_levels": advanced_result.get("stop_levels", {}),
                    "expected_costs": advanced_result.get("expected_costs", {}),
                    "risk_metrics": advanced_result.get("risk_metrics")
                })
            else:
                # 기본 포지션 사이징
                current_price = price_df['close'].iloc[-1]
                returns = price_df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
                
                position_size = self.calculate_position_size(
                    current_capital, current_price, 0.05, 0.02,
                    PositionSizeMethod.VOLATILITY_BASED, volatility
                )
                
                stop_levels = self.calculate_dynamic_stops(current_price, volatility)
                
                result.update({
                    "position_size": position_size,
                    "stop_levels": stop_levels,
                    "expected_costs": {},
                    "risk_metrics": None
                })
            
            result["approved"] = True
            return result
            
        except Exception as e:
            logger.error(f"종합 리스크 검사 오류: {e}")
            result["reasons"].append(f"Risk check error: {str(e)}")
            return result
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """통합 리스크 대시보드"""
        dashboard = {
            "basic_risk_status": {
                "last_regime": self.last_regime.value,
                "regime_confidence": self.regime_confidence,
                "daily_trades": self.risk_monitor.daily_trades,
                "daily_pnl": self.risk_monitor.daily_pnl
            }
        }
        
        # 고급 리스크 대시보드 통합
        if hasattr(self.advanced_manager, 'get_risk_dashboard'):
            advanced_dashboard = self.advanced_manager.get_risk_dashboard()
            dashboard["advanced_risk_status"] = advanced_dashboard
        
        return dashboard
    
    def update_trade(self, pnl: float, trade_info: Dict[str, Any]):
        """거래 업데이트"""
        # 기본 모니터 업데이트
        self.risk_monitor.update_trade(pnl)
        
        # 고급 리스크 관리자 업데이트 (가능한 경우)
        if hasattr(self.advanced_manager, '_on_trade_executed'):
            self.advanced_manager._on_trade_executed(trade_info)
        
        logger.debug(f"Trade updated: PnL={pnl:.4f}, Daily trades={self.risk_monitor.daily_trades}")

# 전역 통합 리스크 관리자
integrated_risk_manager = IntegratedRiskManager()