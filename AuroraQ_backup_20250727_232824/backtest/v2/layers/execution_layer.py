"""
실행 시뮬레이션 계층 (Execution Simulation Layer)
- RiskAdjustedEntry: Kelly 기준 기반 리스크 조정
- Slippage & Fees Model: 거래 수수료 및 동적 슬리피지
- Market Impact Simulation: 거래량 및 변동성 기반 체결가 조정
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """주문 정보"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY, SELL
    order_type: OrderType
    quantity: float
    price: float
    stop_price: Optional[float] = None
    status: str = "pending"  # pending, filled, cancelled
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    execution_timestamp: Optional[datetime] = None


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # LONG, SHORT
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    holding_time: float = 0.0  # seconds


class RiskAdjustedEntry:
    """리스크 조정 진입 시스템"""
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 max_position_pct: float = 0.1,
                 kelly_fraction: float = 0.25):
        """
        Args:
            initial_capital: 초기 자본
            max_position_pct: 최대 포지션 비율
            kelly_fraction: Kelly 기준 분수 (보수적 적용)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        
        # 거래 이력 추적
        self.win_rate_history = []
        self.avg_win_history = []
        self.avg_loss_history = []
    
    def calculate_kelly_criterion(self, 
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float) -> float:
        """Kelly 기준 계산"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly 공식: f = (p * b - q) / b
        # p: 승률, q: 패률, b: 평균 수익/평균 손실 비율
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 0
        
        if b <= 0:
            return 0.0
        
        kelly = (p * b - q) / b
        
        # 보수적 적용 (분수 Kelly)
        adjusted_kelly = kelly * self.kelly_fraction
        
        # 범위 제한
        return max(0.0, min(self.max_position_pct, adjusted_kelly))
    
    def update_statistics(self, trades: List[Dict[str, Any]]):
        """거래 통계 업데이트"""
        if not trades:
            return
        
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        
        if len(trades) > 0:
            win_rate = len(wins) / len(trades)
            self.win_rate_history.append(win_rate)
        
        if wins:
            avg_win = np.mean([t["pnl"] for t in wins])
            self.avg_win_history.append(avg_win)
        
        if losses:
            avg_loss = abs(np.mean([t["pnl"] for t in losses]))
            self.avg_loss_history.append(avg_loss)
    
    def calculate_position_size(self,
                              signal_confidence: float,
                              current_volatility: float,
                              recent_trades: List[Dict[str, Any]] = None) -> float:
        """리스크 조정된 포지션 크기 계산"""
        # 기본 포지션 크기
        base_size = self.current_capital * self.max_position_pct
        
        # Kelly 기준 적용 (거래 이력이 있는 경우)
        kelly_multiplier = 1.0
        
        if recent_trades and len(recent_trades) >= 10:
            self.update_statistics(recent_trades)
            
            if (self.win_rate_history and 
                self.avg_win_history and 
                self.avg_loss_history):
                
                # 최근 통계 사용
                recent_win_rate = np.mean(self.win_rate_history[-20:])
                recent_avg_win = np.mean(self.avg_win_history[-20:])
                recent_avg_loss = np.mean(self.avg_loss_history[-20:])
                
                kelly_pct = self.calculate_kelly_criterion(
                    recent_win_rate, recent_avg_win, recent_avg_loss
                )
                
                kelly_multiplier = kelly_pct / self.max_position_pct
        
        # 신뢰도 조정
        confidence_multiplier = signal_confidence
        
        # 변동성 조정 (변동성이 높을수록 포지션 감소)
        volatility_multiplier = 1.0
        if current_volatility > 0.03:
            volatility_multiplier = 0.7
        elif current_volatility > 0.05:
            volatility_multiplier = 0.4
        
        # 최종 포지션 크기
        position_size = base_size * kelly_multiplier * confidence_multiplier * volatility_multiplier
        
        # 최소/최대 제한
        min_position = self.current_capital * 0.01  # 최소 1%
        max_position = self.current_capital * self.max_position_pct
        
        return max(min_position, min(max_position, position_size))


class SlippageModel:
    """
    향상된 슬리피지 모델 - 유동성 기반 동적 계산
    
    Features:
    - 유동성 프로파일 기반 슬리피지
    - 시장 마이크로구조 고려
    - 주문 크기별 비선형 임팩트
    - 시간대별 유동성 조정
    """
    
    def __init__(self,
                 base_slippage_bps: float = 5.0,  # 기본 5 베이시스 포인트
                 volatility_multiplier: float = 2.0,
                 volume_impact_factor: float = 0.1,
                 liquidity_threshold: float = 0.01,  # 유동성 임계값
                 nonlinear_impact_factor: float = 1.5):  # 비선형 임팩트 계수
        """
        Args:
            base_slippage_bps: 기본 슬리피지 (베이시스 포인트)
            volatility_multiplier: 변동성 승수
            volume_impact_factor: 거래량 영향 계수
            liquidity_threshold: 유동성 임계값 (이하에서 슬리피지 급증)
            nonlinear_impact_factor: 비선형 임팩트 계수
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier
        self.volume_impact_factor = volume_impact_factor
        self.liquidity_threshold = liquidity_threshold
        self.nonlinear_impact_factor = nonlinear_impact_factor
    
    def calculate_slippage(self,
                         order_size: float,
                         current_price: float,
                         volatility: float,
                         volume: float,
                         order_side: str,
                         timestamp: Optional[datetime] = None) -> float:
        """
        향상된 슬리피지 계산 - 유동성 기반 동적 모델
        
        Args:
            order_size: 주문 크기 (가치)
            current_price: 현재 가격
            volatility: 변동성
            volume: 거래량
            order_side: 주문 방향 (BUY/SELL)
            timestamp: 시간 (유동성 조정용)
        """
        # 1. 기본 슬리피지
        base_slippage = current_price * (self.base_slippage_bps / 10000)
        
        # 2. 변동성 조정 (더 정교한 모델)
        volatility = max(0.001, min(volatility, 0.15)) if volatility > 0 else 0.02
        vol_factor = min(volatility * self.volatility_multiplier, 3.0)  # 최대 3배 제한
        volatility_adjustment = base_slippage * vol_factor
        
        # 3. 유동성 기반 임팩트 계산
        if volume > 0:
            # 주문 크기 대비 유동성 비율
            liquidity_ratio = order_size / (volume * current_price)
            
            # 유동성 점수 계산 (0-1, 낮을수록 유동성 부족)
            liquidity_score = self._calculate_liquidity_score(liquidity_ratio, volume)
            
            # 비선형 임팩트 모델
            if liquidity_ratio <= self.liquidity_threshold:
                # 정상 유동성: 선형 임팩트
                impact_factor = liquidity_ratio * self.volume_impact_factor
            else:
                # 유동성 부족: 비선형 임팩트 (급격히 증가)
                excess_ratio = liquidity_ratio - self.liquidity_threshold
                linear_impact = self.liquidity_threshold * self.volume_impact_factor
                nonlinear_impact = excess_ratio ** self.nonlinear_impact_factor * self.volume_impact_factor * 5
                impact_factor = linear_impact + nonlinear_impact
            
            # 유동성 조정
            volume_adjustment = current_price * impact_factor * (2 - liquidity_score)
        else:
            volume_adjustment = base_slippage * 2  # 거래량 데이터 없을 때 페널티
        
        # 4. 시간대별 유동성 조정
        time_multiplier = self._get_time_liquidity_multiplier(timestamp) if timestamp else 1.0
        
        # 5. 총 슬리피지 계산
        total_slippage = (base_slippage + volatility_adjustment + volume_adjustment) * time_multiplier
        
        # 6. 합리적 범위로 제한
        max_slippage = current_price * min(0.03, volatility * 10)  # 변동성에 비례한 최대 슬리피지
        total_slippage = min(abs(total_slippage), max_slippage)
        
        # 7. 방향성 적용
        if order_side == "BUY":
            return total_slippage
        else:
            return -total_slippage
    
    def _calculate_liquidity_score(self, liquidity_ratio: float, volume: float) -> float:
        """
        유동성 점수 계산 (0-1, 높을수록 유동성 좋음)
        """
        # 거래량 기반 기본 점수
        volume_score = min(volume / 1000000, 1.0)  # 100만 기준으로 정규화
        
        # 유동성 비율 기반 점수
        if liquidity_ratio <= 0.001:  # 0.1% 이하
            ratio_score = 1.0
        elif liquidity_ratio <= 0.01:  # 1% 이하
            ratio_score = 0.8
        elif liquidity_ratio <= 0.05:  # 5% 이하
            ratio_score = 0.5
        else:
            ratio_score = 0.2
        
        return (volume_score + ratio_score) / 2
    
    def _get_time_liquidity_multiplier(self, timestamp: datetime) -> float:
        """
        시간대별 유동성 승수 계산
        """
        if not timestamp:
            return 1.0
        
        hour = timestamp.hour
        
        # UTC 기준 주요 거래 시간
        if 9 <= hour <= 16:  # 아시아-유럽 겹치는 시간
            return 0.9  # 높은 유동성
        elif 13 <= hour <= 21:  # 유럽-미국 겹치는 시간
            return 0.8  # 최고 유동성
        elif 22 <= hour <= 6:  # 미국 시간대
            return 1.0  # 보통 유동성
        else:  # 유동성 낮은 시간
            return 1.3  # 슬리피지 증가


class FeeModel:
    """수수료 모델"""
    
    def __init__(self,
                 maker_fee_rate: float = 0.0002,  # 0.02%
                 taker_fee_rate: float = 0.0004,  # 0.04%
                 use_bnb_discount: bool = True):
        """
        Args:
            maker_fee_rate: 메이커 수수료율
            taker_fee_rate: 테이커 수수료율
            use_bnb_discount: BNB 할인 적용 여부
        """
        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate
        self.use_bnb_discount = use_bnb_discount
        self.bnb_discount_rate = 0.25  # 25% 할인
    
    def calculate_fee(self,
                     order_value: float,
                     order_type: OrderType,
                     is_maker: bool = False) -> float:
        """거래 수수료 계산"""
        # 기본 수수료율
        if is_maker or order_type == OrderType.LIMIT:
            fee_rate = self.maker_fee_rate
        else:
            fee_rate = self.taker_fee_rate
        
        # BNB 할인 적용
        if self.use_bnb_discount:
            fee_rate *= (1 - self.bnb_discount_rate)
        
        return order_value * fee_rate


class MarketImpactSimulator:
    """시장 영향 시뮬레이터"""
    
    def __init__(self,
                 permanent_impact_factor: float = 0.1,
                 temporary_impact_factor: float = 0.3):
        """
        Args:
            permanent_impact_factor: 영구적 시장 영향 계수
            temporary_impact_factor: 일시적 시장 영향 계수
        """
        self.permanent_impact_factor = permanent_impact_factor
        self.temporary_impact_factor = temporary_impact_factor
    
    def calculate_market_impact(self,
                              order_size: float,
                              average_volume: float,
                              volatility: float,
                              spread: float) -> Dict[str, float]:
        """시장 영향 계산"""
        if average_volume <= 0:
            participation_rate = 0.1
        else:
            participation_rate = order_size / average_volume
        
        # 영구적 영향 (제곱근 모델)
        permanent_impact = (self.permanent_impact_factor * 
                          np.sqrt(participation_rate) * 
                          volatility)
        
        # 일시적 영향
        temporary_impact = (self.temporary_impact_factor * 
                          participation_rate * 
                          spread)
        
        return {
            "permanent": permanent_impact,
            "temporary": temporary_impact,
            "total": permanent_impact + temporary_impact
        }


class ExecutionSimulator:
    """
    실행 시뮬레이션 메인 클래스
    모든 실행 관련 컴포넌트 통합
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 enable_risk_adjustment: bool = True,
                 enable_market_impact: bool = True):
        """
        Args:
            initial_capital: 초기 자본
            enable_risk_adjustment: 리스크 조정 활성화
            enable_market_impact: 시장 영향 시뮬레이션 활성화
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.enable_risk_adjustment = enable_risk_adjustment
        self.enable_market_impact = enable_market_impact
        
        # 컴포넌트 초기화
        self.risk_adjuster = RiskAdjustedEntry(initial_capital)
        self.slippage_model = SlippageModel()
        self.fee_model = FeeModel()
        self.market_impact = MarketImpactSimulator()
        
        # 포지션 및 주문 관리
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        self.order_counter = 0
    
    def execute_signal(self,
                      signal: Any,  # SignalResult from signal_layer
                      market_data: Dict[str, Any],
                      current_timestamp: datetime) -> Dict[str, Any]:
        """
        거래 신호 실행 시뮬레이션
        
        Returns:
            실행 결과 (주문, 체결가, 수수료 등)
        """
        try:
            price_data = market_data["price"]
            current_price = price_data['close'].iloc[-1]
            current_volume = price_data['volume'].iloc[-1] if 'volume' in price_data else 1000000
            
            # HOLD 신호는 실행하지 않음 (대소문자 구분 없이)
            if signal.action.upper() == "HOLD":
                return {
                    "executed": False,
                    "reason": "HOLD signal"
                }
            
            # 포지션 크기 계산
            if self.enable_risk_adjustment:
                # 변동성 계산
                volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1]
                
                # 리스크 조정된 포지션 크기
                risk_adjusted_size = self.risk_adjuster.calculate_position_size(
                    signal.confidence,
                    volatility,
                    self.trades[-50:] if len(self.trades) > 50 else self.trades
                )
                
                # 신호의 포지션 크기와 조합
                position_value = risk_adjusted_size * signal.position_size
            else:
                position_value = self.current_capital * 0.1 * signal.position_size
            
            # 수량 계산
            quantity = position_value / current_price
            
            # 주문 생성
            order = self._create_order(
                signal.action,
                quantity,
                current_price,
                current_timestamp
            )
            
            # 슬리피지 계산 (향상된 모델)
            volatility = market_data.get("volatility", 0.02)
            slippage = self.slippage_model.calculate_slippage(
                position_value,
                current_price,
                volatility,
                current_volume,
                signal.action,
                current_timestamp
            )
            
            # 시장 영향 계산
            market_impact_result = {"permanent": 0, "temporary": 0, "total": 0}
            if self.enable_market_impact:
                spread = current_price * 0.0001  # 0.01% 스프레드 가정
                market_impact_result = self.market_impact.calculate_market_impact(
                    position_value,
                    current_volume * current_price,  # 거래대금
                    volatility,
                    spread
                )
            
            # 체결가 계산
            execution_price = current_price + slippage
            if self.enable_market_impact:
                execution_price += current_price * market_impact_result["total"]
            
            # 수수료 계산
            order_value = quantity * execution_price
            commission = self.fee_model.calculate_fee(
                order_value,
                OrderType.MARKET,
                is_maker=False
            )
            
            # 주문 체결 처리
            order.status = "filled"
            order.filled_quantity = quantity
            order.filled_price = execution_price
            order.commission = commission
            order.slippage = slippage
            order.execution_timestamp = current_timestamp
            
            # 포지션 업데이트
            self._update_position(order, signal)
            
            # 거래 기록
            trade = {
                "timestamp": current_timestamp,
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": quantity,
                "price": current_price,
                "execution_price": execution_price,
                "slippage": slippage,
                "commission": commission,
                "market_impact": market_impact_result["total"],
                "signal_confidence": signal.confidence,
                "position_value": position_value
            }
            
            self.trades.append(trade)
            self.orders.append(order)
            
            # 자본 업데이트
            if signal.action == "BUY":
                self.current_capital -= (order_value + commission)
            else:  # SELL
                self.current_capital += (order_value - commission)
            
            return {
                "executed": True,
                "order": order,
                "trade": trade,
                "execution_details": {
                    "requested_price": current_price,
                    "execution_price": execution_price,
                    "slippage": slippage,
                    "commission": commission,
                    "market_impact": market_impact_result,
                    "total_cost": slippage + commission + 
                                 (current_price * market_impact_result["total"])
                }
            }
            
        except Exception as e:
            logger.error(f"실행 시뮬레이션 오류: {e}")
            return {
                "executed": False,
                "error": str(e)
            }
    
    def _create_order(self, 
                     side: str,
                     quantity: float,
                     price: float,
                     timestamp: datetime) -> Order:
        """주문 생성"""
        self.order_counter += 1
        
        return Order(
            order_id=f"ORD_{self.order_counter:06d}",
            timestamp=timestamp,
            symbol="BTCUSDT",  # 하드코딩 (추후 파라미터화)
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price
        )
    
    def _update_position(self, order: Order, signal: Any):
        """포지션 업데이트"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # 새 포지션 생성
            if order.side == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side="LONG",
                    quantity=order.filled_quantity,
                    entry_price=order.filled_price,
                    entry_timestamp=order.execution_timestamp,
                    current_price=order.filled_price
                )
        else:
            # 기존 포지션 업데이트
            position = self.positions[symbol]
            
            if order.side == "BUY":
                # 추가 매수
                total_value = (position.quantity * position.entry_price + 
                             order.filled_quantity * order.filled_price)
                position.quantity += order.filled_quantity
                position.entry_price = total_value / position.quantity
            else:
                # 매도 (포지션 청산)
                if position.quantity <= order.filled_quantity:
                    # 전체 청산
                    realized_pnl = (order.filled_price - position.entry_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    del self.positions[symbol]
                else:
                    # 부분 청산
                    realized_pnl = (order.filled_price - position.entry_price) * order.filled_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= order.filled_quantity
    
    def get_current_positions(self) -> Dict[str, Position]:
        """현재 포지션 조회"""
        return self.positions.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """실행 통계"""
        if not self.trades:
            return {
                "total_trades": 0,
                "avg_slippage": 0,
                "total_commission": 0,
                "avg_market_impact": 0
            }
        
        total_slippage = sum(t["slippage"] for t in self.trades)
        total_commission = sum(t["commission"] for t in self.trades)
        total_impact = sum(t["market_impact"] for t in self.trades)
        
        return {
            "total_trades": len(self.trades),
            "avg_slippage": total_slippage / len(self.trades),
            "total_commission": total_commission,
            "avg_market_impact": total_impact / len(self.trades),
            "current_capital": self.current_capital,
            "capital_usage": 1 - (self.current_capital / self.initial_capital)
        }