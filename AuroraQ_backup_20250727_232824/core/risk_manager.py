import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# 새로운 열거형 추가
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

# 리스크 파라미터 설정
@dataclass
class RiskParameters:
    """리스크 관리 파라미터"""
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

# 전역 파라미터
RISK_PARAMS = RiskParameters()

# 전략별 리스크 프로필
STRATEGY_RISK_PROFILES = {
    "RuleStrategyA": {"risk_level": "low", "max_allocation": 0.20, "stop_loss": 0.03},
    "RuleStrategyB": {"risk_level": "medium", "max_allocation": 0.25, "stop_loss": 0.05},
    "RuleStrategyC": {"risk_level": "medium", "max_allocation": 0.25, "stop_loss": 0.05},
    "RuleStrategyD": {"risk_level": "high", "max_allocation": 0.30, "stop_loss": 0.07},
    "RuleStrategyE": {"risk_level": "low", "max_allocation": 0.20, "stop_loss": 0.03},
    "PPOStrategy": {"risk_level": "adaptive", "max_allocation": 0.50, "stop_loss": 0.05},
}

def evaluate_risk(
    price_df: pd.DataFrame,
    strategy_name: str,
    sentiment_score: Optional[float] = None,
    sentiment_delta: Optional[float] = None,
    position: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    종합 리스크 평가 (개선 버전)
    
    Args:
        price_df: 가격 데이터
        strategy_name: 전략 이름
        sentiment_score: 현재 감정 점수
        sentiment_delta: 감정 점수 변화량
        position: 현재 포지션 정보
        
    Returns:
        (리스크 허용 여부, 사유)
    """
    try:
        # 1. 데이터 유효성 검사
        if price_df is None or len(price_df) < 1:
            return False, "데이터 부족"
        
        close_series = price_df.get("close")
        if close_series is None or close_series.empty:
            return False, "가격 데이터 없음"
        
        current_price = float(close_series.iloc[-1])
        
        # 2. 가격 리스크 평가
        price_risk = evaluate_price_risk(price_df, current_price)
        if not price_risk[0]:
            return price_risk
        
        # 3. 감정 리스크 평가
        if sentiment_score is not None or sentiment_delta is not None:
            sentiment_risk = evaluate_sentiment_risk(sentiment_score, sentiment_delta)
            if not sentiment_risk[0]:
                return sentiment_risk
        
        # 4. 변동성 리스크 평가
        volatility_risk = evaluate_volatility_risk(price_df)
        if not volatility_risk[0]:
            return volatility_risk
        
        # 5. 전략별 리스크 평가
        strategy_risk = evaluate_strategy_risk(strategy_name, price_df)
        if not strategy_risk[0]:
            return strategy_risk
        
        # 6. 포지션 리스크 평가 (이미 포지션이 있는 경우)
        if position:
            position_risk = evaluate_position_risk(position, current_price)
            if not position_risk[0]:
                return position_risk
        
        return True, "리스크 평가 통과"
        
    except Exception as e:
        logger.error(f"[RISK] 평가 중 오류 발생: {e}")
        return False, f"리스크 평가 오류: {str(e)}"

def evaluate_price_risk(price_df: pd.DataFrame, current_price: float) -> Tuple[bool, str]:
    """가격 관련 리스크 평가"""
    # 최소 가격 체크
    if current_price < RISK_PARAMS.min_price:
        return False, f"가격 너무 낮음: ${current_price:.2f}"
    
    # 급락 체크 (1시간 기준)
    if len(price_df) >= 12:  # 5분봉 12개 = 1시간
        hour_ago_price = float(price_df["close"].iloc[-12])
        price_drop = (hour_ago_price - current_price) / hour_ago_price
        
        if price_drop > RISK_PARAMS.max_price_drop_pct:
            return False, f"급락 감지: {price_drop:.1%} in 1시간"
    
    return True, "가격 리스크 정상"

def evaluate_sentiment_risk(
    sentiment_score: Optional[float],
    sentiment_delta: Optional[float]
) -> Tuple[bool, str]:
    """감정 관련 리스크 평가"""
    if sentiment_score is not None:
        # 극단적 공포
        if sentiment_score < RISK_PARAMS.sentiment_extreme_fear:
            return False, f"극단적 공포 상태: {sentiment_score:.2f}"
        
        # 극단적 탐욕 (과열)
        if sentiment_score > RISK_PARAMS.sentiment_extreme_greed:
            return False, f"극단적 탐욕 상태: {sentiment_score:.2f}"
    
    if sentiment_delta is not None:
        # 감정 급변
        if abs(sentiment_delta) > RISK_PARAMS.sentiment_delta_threshold:
            return False, f"감정 급변: Δ{sentiment_delta:.2f}"
    
    return True, "감정 리스크 정상"

def evaluate_volatility_risk(price_df: pd.DataFrame) -> Tuple[bool, str]:
    """변동성 리스크 평가"""
    if len(price_df) < 20:
        return True, "데이터 부족으로 변동성 체크 생략"
    
    # 수익률 계산
    returns = price_df["close"].pct_change().dropna()
    volatility = returns.std()
    
    if volatility > RISK_PARAMS.max_volatility:
        return False, f"과도한 변동성: {volatility:.2%}"
    
    if volatility < RISK_PARAMS.min_volatility:
        return False, f"변동성 너무 낮음: {volatility:.4%}"
    
    return True, "변동성 정상"

def evaluate_strategy_risk(strategy_name: str, price_df: pd.DataFrame) -> Tuple[bool, str]:
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

def evaluate_position_risk(position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
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

def allocate_capital(
    total_capital: float,
    strategy_name: str,
    market_conditions: Optional[Dict[str, Any]] = None
) -> float:
    """
    자본 배분 (개선 버전)
    
    Args:
        total_capital: 총 자본
        strategy_name: 전략 이름
        market_conditions: 시장 상황 정보
        
    Returns:
        배분된 자본
    """
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
    
    # 최종 배분 계산
    allocated = total_capital * base_allocation
    
    # 최소/최대 제한
    min_allocation = max(RISK_PARAMS.min_position_size, total_capital * 0.05)
    max_allocation = total_capital * RISK_PARAMS.max_position_size_pct
    
    allocated = max(min_allocation, min(allocated, max_allocation))
    
    logger.info(
        f"[RISK] 자본 배분 - {strategy_name}: "
        f"${allocated:,.2f} ({allocated/total_capital:.1%})"
    )
    
    return allocated

def adjust_leverage(
    base_leverage: float,
    sentiment_score: float,
    regime_label: str,
    volatility: Optional[float] = None
) -> float:
    """
    레버리지 동적 조정 (개선 버전)
    
    Args:
        base_leverage: 기본 레버리지
        sentiment_score: 감정 점수
        regime_label: 시장 레짐
        volatility: 변동성
        
    Returns:
        조정된 레버리지
    """
    leverage = base_leverage
    
    # 1. 감정 점수 기반 조정
    if sentiment_score > 0.7:
        leverage *= 1.2  # 긍정적 시장
    elif sentiment_score < 0.3:
        leverage *= 0.8  # 부정적 시장
    
    # 2. 시장 레짐 기반 조정
    regime_multipliers = {
        "bull": 1.3,
        "bear": 0.7,
        "neutral": 1.0,
        "volatile": 0.5
    }
    leverage *= regime_multipliers.get(regime_label, 1.0)
    
    # 3. 변동성 기반 조정
    if volatility:
        if volatility > 0.03:  # 고변동성
            leverage *= 0.7
        elif volatility < 0.01:  # 저변동성
            leverage *= 1.1
    
    # 4. 최종 제한
    leverage = max(RISK_PARAMS.min_leverage, 
                  min(leverage, RISK_PARAMS.max_leverage))
    
    logger.info(
        f"[RISK] 레버리지 조정: {base_leverage:.1f}x → {leverage:.1f}x "
        f"(감정: {sentiment_score:.2f}, 레짐: {regime_label})"
    )
    
    return leverage

def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_per_trade: float = 0.02
) -> float:
    """
    Kelly Criterion 기반 포지션 크기 계산
    
    Args:
        capital: 가용 자본
        entry_price: 진입 가격
        stop_loss_pct: 손절 비율
        risk_per_trade: 거래당 리스크 (기본 2%)
        
    Returns:
        포지션 크기 (수량)
    """
    if stop_loss_pct <= 0:
        stop_loss_pct = RISK_PARAMS.default_stop_loss
    
    # 리스크 금액 계산
    risk_amount = capital * risk_per_trade
    
    # 포지션 크기 계산
    position_value = risk_amount / stop_loss_pct
    position_size = position_value / entry_price
    
    # 최대 포지션 제한
    max_position_value = capital * RISK_PARAMS.max_position_size_pct
    max_position_size = max_position_value / entry_price
    
    final_size = min(position_size, max_position_size)
    
    logger.debug(
        f"[RISK] 포지션 크기: {final_size:.4f} "
        f"(리스크: ${risk_amount:.2f}, 손절: {stop_loss_pct:.1%})"
    )
    
    return final_size

def should_cut_loss_or_take_profit(
    entry_price: float,
    current_price: float,
    strategy_name: Optional[str] = None,
    position_info: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], str]:
    """
    손절/익절 판단 (개선 버전)
    
    Returns:
        (액션, 사유)
    """
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
            stop_loss = profile.get("stop_loss", RISK_PARAMS.default_stop_loss)
        if take_profit is None:
            take_profit = RISK_PARAMS.default_take_profit
        
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
        logger.error(f"[RISK] 손절/익절 판단 오류: {e}")
        return None, f"판단 오류: {str(e)}"

def calculate_risk_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """
    리스크 관련 지표 계산
    
    Args:
        trades: 거래 기록
        
    Returns:
        리스크 지표들
    """
    if trades.empty:
        return {}
    
    # 기본 통계
    returns = trades["pnl"].values
    
    # Sharpe Ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Maximum Drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown)
    
    # Value at Risk (VaR) - 95% 신뢰수준
    var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0
    
    # Calmar Ratio
    annual_return = mean_return * 252
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 3),
        "var_95": round(var_95, 4),
        "calmar_ratio": round(calmar, 3),
        "avg_loss": round(np.mean(returns[returns < 0]), 4) if any(returns < 0) else 0,
        "avg_win": round(np.mean(returns[returns > 0]), 4) if any(returns > 0) else 0,
        "risk_reward_ratio": abs(np.mean(returns[returns > 0]) / np.mean(returns[returns < 0])) 
                            if any(returns < 0) and any(returns > 0) else 0
    }

# 리스크 모니터링 클래스
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

# 전역 리스크 모니터
risk_monitor = RiskMonitor()

# 새로운 리스크 관리 클래스들 추가
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
    
    def should_trade_enhanced(self, price_df: pd.DataFrame, current_positions: int = 0) -> Tuple[bool, str]:
        """향상된 거래 허용 여부 판단"""
        if len(price_df) < self.lookback_period:
            return False, "Insufficient data"
        
        # 시장 체제 검사
        regime, confidence = self.detect_market_regime(price_df)
        
        # 매우 높은 확신도의 크라이시스만 거래 금지
        if regime == MarketRegime.CRISIS and confidence > 0.9:
            return False, f"Extreme crisis regime detected (confidence: {confidence:.2f})"
        
        # 고변동성 포지션 제한
        if regime == MarketRegime.HIGH_VOLATILITY and current_positions > 3:
            return False, f"High volatility - too many positions"
        
        return True, "Trading allowed"

class EnhancedPositionSizer:
    """향상된 포지션 사이징 시스템"""
    
    def __init__(self, method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_BASED):
        self.method = method
        
    def calculate_enhanced_position_size(self, 
                                       capital: float,
                                       price: float,
                                       volatility: float,
                                       win_rate: float = 0.5,
                                       avg_win: float = 0.02,
                                       avg_loss: float = -0.02,
                                       risk_per_trade: float = 0.02) -> int:
        """향상된 포지션 크기 계산"""
        
        if self.method == PositionSizeMethod.FIXED_AMOUNT:
            return int(50000 / price)
            
        elif self.method == PositionSizeMethod.FIXED_FRACTIONAL:
            target_amount = capital * 0.15  # 15%
            return int(target_amount / price)
            
        elif self.method == PositionSizeMethod.KELLY:
            if avg_loss >= 0:
                avg_loss = -0.02
            
            kelly_ratio = (win_rate * avg_win + (1 - win_rate) * avg_loss) / abs(avg_loss)
            kelly_ratio = max(0, min(kelly_ratio, 0.20))  # 0-20% 제한
            
            target_amount = capital * kelly_ratio
            return int(target_amount / price)
            
        elif self.method == PositionSizeMethod.VOLATILITY_BASED:
            if volatility <= 0:
                volatility = 0.2
                
            # 변동성이 높을수록 작은 포지션
            vol_adjustment = min(0.15 / volatility, 1.0)
            base_fraction = 0.12  # 기본 12%
            
            target_amount = capital * base_fraction * vol_adjustment
            return int(target_amount / price)
            
        elif self.method == PositionSizeMethod.RISK_PARITY:
            target_vol = 0.12  # 목표 변동성 12%
            position_vol = max(volatility, 0.05)
            
            leverage = target_vol / position_vol
            leverage = min(leverage, 1.5)  # 최대 1.5배
            
            target_amount = capital * 0.15 * leverage
            return int(target_amount / price)
        
        else:
            target_amount = capital * 0.1
            return int(target_amount / price)

class EnhancedStopManager:
    """향상된 손절/익절 관리"""
    
    def __init__(self):
        self.stop_loss_pct = 0.04      # 4% 손절
        self.take_profit_pct = 0.12    # 12% 익절  
        self.trailing_stop_pct = 0.025 # 2.5% 트레일링 스탑
        
    def calculate_dynamic_stops(self, 
                              entry_price: float, 
                              volatility: float = 0.2,
                              regime: MarketRegime = MarketRegime.SIDEWAYS) -> Dict[str, float]:
        """동적 스탑 레벨 계산"""
        
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
        
        adjusted_stop_loss = self.stop_loss_pct * final_multiplier
        adjusted_take_profit = self.take_profit_pct * final_multiplier
        
        return {
            "stop_loss": entry_price * (1 - adjusted_stop_loss),
            "take_profit": entry_price * (1 + adjusted_take_profit),
            "trailing_stop": entry_price * (1 - self.trailing_stop_pct * final_multiplier)
        }

class TransactionCostOptimizer:
    """거래 비용 최적화"""
    
    def __init__(self):
        self.commission_rate = 0.0015
        self.slippage_rate = 0.001
        self.min_commission = 1000
        
    def calculate_total_cost(self, trade_amount: float, volatility: float = 0.2) -> Dict[str, float]:
        """총 거래 비용 계산"""
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        adjusted_slippage = self.slippage_rate * (1 + volatility)
        slippage = trade_amount * adjusted_slippage
        
        total_cost = commission + slippage
        
        return {
            "commission": commission,
            "slippage": slippage,
            "total_cost": total_cost,
            "cost_rate": total_cost / trade_amount
        }
    
    def is_trade_profitable_enhanced(self, expected_return: float, trade_amount: float, volatility: float = 0.2) -> bool:
        """향상된 거래 수익성 판단"""
        costs = self.calculate_total_cost(trade_amount, volatility)
        total_cost_rate = costs["cost_rate"] * 2  # 왕복 거래 비용
        
        # 작은 거래는 기준 완화
        if trade_amount < 3_000_000:  # 300만원 미만
            min_required_return = total_cost_rate * 1.2
        else:
            min_required_return = total_cost_rate * 1.5
        
        return expected_return > min_required_return

# 통합 리스크 관리자 클래스
class IntegratedRiskManager:
    """통합 리스크 관리 시스템 (기존 + 향상된 기능)"""
    
    def __init__(self):
        self.risk_filter = EnhancedRiskFilter()
        self.position_sizer = EnhancedPositionSizer()
        self.stop_manager = EnhancedStopManager()
        self.cost_optimizer = TransactionCostOptimizer()
        
        # 전체 포트폴리오 제한
        self.max_portfolio_risk = 0.12
        self.max_concentration = 0.25
        self.max_daily_loss = 0.04
        
    def comprehensive_risk_check(self, 
                                price_df: pd.DataFrame,
                                strategy_name: str,
                                current_capital: float,
                                current_positions: Dict,
                                signal_strength: float = 1.0,
                                sentiment_score: Optional[float] = None) -> Dict:
        """종합 리스크 검사"""
        
        result = {
            "approved": False,
            "position_size": 0,
            "stop_levels": {},
            "expected_costs": {},
            "reasons": []
        }
        
        try:
            # 1. 기존 리스크 평가
            basic_risk_ok, basic_reason = evaluate_risk(
                price_df, strategy_name, sentiment_score
            )
            if not basic_risk_ok:
                result["reasons"].append(f"Basic Risk: {basic_reason}")
                return result
            
            # 2. 향상된 리스크 필터
            enhanced_risk_ok, enhanced_reason = self.risk_filter.should_trade_enhanced(
                price_df, len(current_positions)
            )
            if not enhanced_risk_ok:
                result["reasons"].append(f"Enhanced Risk: {enhanced_reason}")
                return result
            
            # 3. 시장 체제 및 변동성 분석
            regime, regime_confidence = self.risk_filter.detect_market_regime(price_df)
            
            current_price = price_df['close'].iloc[-1]
            returns = price_df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
            
            # 4. 포지션 크기 계산
            position_size = self.position_sizer.calculate_enhanced_position_size(
                current_capital, current_price, volatility
            )
            
            if position_size == 0:
                result["reasons"].append("Position size too small")
                return result
            
            # 5. 거래 비용 검사
            trade_amount = position_size * current_price
            expected_return = signal_strength * 0.025  # 2.5% 기대 수익률
            
            if not self.cost_optimizer.is_trade_profitable_enhanced(expected_return, trade_amount, volatility):
                result["reasons"].append("Trade not profitable after costs")
                return result
            
            # 6. 스탑 레벨 계산
            stop_levels = self.stop_manager.calculate_dynamic_stops(current_price, volatility, regime)
            
            # 7. 비용 계산
            expected_costs = self.cost_optimizer.calculate_total_cost(trade_amount, volatility)
            
            # 모든 검사 통과
            result.update({
                "approved": True,
                "position_size": position_size,
                "stop_levels": stop_levels,
                "expected_costs": expected_costs,
                "regime": regime.value,
                "regime_confidence": regime_confidence,
                "volatility": volatility
            })
            
            return result
            
        except Exception as e:
            logger.error(f"종합 리스크 검사 오류: {e}")
            result["reasons"].append(f"Risk check error: {str(e)}")
            return result

# 전역 통합 리스크 관리자
integrated_risk_manager = IntegratedRiskManager()