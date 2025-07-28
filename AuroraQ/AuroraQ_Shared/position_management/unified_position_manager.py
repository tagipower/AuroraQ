#!/usr/bin/env python3
"""
통합 포지션 관리자
백테스트와 실시간 거래에서 공통으로 사용하는 포지션 관리 로직
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from .position_models import Position, Trade, OrderSignal, PositionSide, OrderSide, TradeStatus


@dataclass
class TradingLimits:
    """거래 제한 설정"""
    max_position_size: float = 0.95  # 최대 포지션 크기 (자본 대비 %)
    max_daily_trades: int = 50  # 일일 최대 거래 수
    max_position_count: int = 10  # 동시 보유 가능한 포지션 수
    min_order_size: float = 0.0001  # 최소 주문 크기
    max_order_size: float = 1.0  # 최대 주문 크기
    emergency_stop_loss: float = 0.05  # 긴급 손절선 (5%)
    daily_loss_limit: float = 0.02  # 일일 손실 한도 (2%)
    enable_short_selling: bool = False  # 공매도 허용 여부
    max_leverage: float = 1.0  # 최대 레버리지
    
    # 리스크 관리 설정
    position_concentration_limit: float = 0.3  # 단일 포지션 집중도 한계
    correlation_limit: float = 0.7  # 포지션 간 상관관계 한계
    volatility_adjustment: bool = True  # 변동성 기반 포지션 조정


@dataclass 
class RiskState:
    """리스크 상태 추적"""
    daily_trades_count: int = 0
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR
    last_reset_date: datetime = field(default_factory=datetime.now)
    
    def reset_daily_stats(self):
        """일일 통계 초기화"""
        today = datetime.now().date()
        if self.last_reset_date.date() != today:
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = datetime.now()


class UnifiedPositionManager:
    """통합 포지션 관리자"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 trading_limits: Optional[TradingLimits] = None,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_limits = trading_limits or TradingLimits()
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 포지션 관리
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.equity_history: List[Tuple[datetime, float]] = []
        self.risk_state = RiskState(peak_equity=initial_capital)
        
        # 거래 추적
        self.all_trades: List[Trade] = []
        self.pending_orders: List[OrderSignal] = []
        
        # 리스크 관리 콜백
        self.risk_callbacks: List[Callable] = []
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # 성과 추적
        self._update_equity()
    
    def add_risk_callback(self, callback: Callable[[Dict], None]):
        """리스크 관리 콜백 추가"""
        self.risk_callbacks.append(callback)
    
    def get_equity(self) -> float:
        """총 자산가치 계산"""
        cash = self.cash
        position_value = sum(pos.state.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return cash + position_value + unrealized_pnl
    
    def get_available_capital(self) -> float:
        """사용 가능한 자본 계산"""
        equity = self.get_equity()
        used_capital = sum(pos.state.market_value for pos in self.positions.values())
        max_usable = equity * self.trading_limits.max_position_size
        return max(0, max_usable - used_capital)
    
    def get_position(self, symbol: str = "") -> Dict[str, Any]:
        """포지션 정보 반환 (기존 인터페이스 호환)"""
        if symbol and symbol in self.positions:
            return self.positions[symbol].get_position_info()
        
        # 전체 포지션 요약
        total_size = sum(abs(pos.size) for pos in self.positions.values())
        total_value = sum(pos.state.market_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'is_flat': len(self.positions) == 0 or total_size < 1e-8,
            'is_long': any(pos.is_long for pos in self.positions.values()),
            'is_short': any(pos.is_short for pos in self.positions.values()),
            'total_size': total_size,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'position_count': len(self.positions),
            'symbols': list(self.positions.keys())
        }
    
    def get_position_value(self) -> float:
        """포지션 가치 반환 (기존 인터페이스 호환)"""
        return sum(pos.state.market_value + pos.unrealized_pnl for pos in self.positions.values())
    
    def can_open_position(self, 
                         symbol: str,
                         size: float, 
                         price: float, 
                         side: OrderSide,
                         strategy_info: Optional[Dict] = None) -> Tuple[bool, str]:
        """포지션 개설 가능 여부 확인"""
        
        # 일일 통계 초기화
        self.risk_state.reset_daily_stats()
        
        # 1. 일일 거래 한도 체크
        if self.risk_state.daily_trades_count >= self.trading_limits.max_daily_trades:
            return False, f"일일 거래 한도 초과: {self.risk_state.daily_trades_count}/{self.trading_limits.max_daily_trades}"
        
        # 2. 포지션 수 제한 체크
        if len(self.positions) >= self.trading_limits.max_position_count:
            return False, f"최대 포지션 수 초과: {len(self.positions)}/{self.trading_limits.max_position_count}"
        
        # 3. 주문 크기 제한 체크
        if size < self.trading_limits.min_order_size:
            return False, f"최소 주문 크기 미달: {size} < {self.trading_limits.min_order_size}"
        
        if size > self.trading_limits.max_order_size:
            return False, f"최대 주문 크기 초과: {size} > {self.trading_limits.max_order_size}"
        
        # 4. 자본 충족 여부 체크
        required_capital = size * price * (1 + self.commission_rate + self.slippage_rate)
        available_capital = self.get_available_capital()
        
        if side == OrderSide.BUY and required_capital > available_capital:
            return False, f"자본 부족: 필요 ${required_capital:,.2f}, 가용 ${available_capital:,.2f}"
        
        # 5. 공매도 허용 여부 체크
        if side == OrderSide.SELL and not self.trading_limits.enable_short_selling:
            if symbol not in self.positions or self.positions[symbol].size < size:
                return False, "공매도가 허용되지 않음"
        
        # 6. 포지션 집중도 체크
        position_value = size * price
        total_equity = self.get_equity()
        concentration = position_value / total_equity
        
        if concentration > self.trading_limits.position_concentration_limit:
            return False, f"포지션 집중도 초과: {concentration:.1%} > {self.trading_limits.position_concentration_limit:.1%}"
        
        # 7. 일일 손실 한도 체크
        if self.risk_state.daily_pnl < -total_equity * self.trading_limits.daily_loss_limit:
            return False, f"일일 손실 한도 도달: {self.risk_state.daily_pnl/total_equity:.2%}"
        
        # 8. 레버리지 체크
        total_position_value = sum(pos.state.market_value for pos in self.positions.values()) + position_value
        leverage = total_position_value / total_equity
        
        if leverage > self.trading_limits.max_leverage:
            return False, f"레버리지 한도 초과: {leverage:.2f}x > {self.trading_limits.max_leverage:.2f}x"
        
        return True, "OK"
    
    def execute_trade(self, 
                     signal: OrderSignal, 
                     current_price: float,
                     strategy_id: str = "") -> Optional[Trade]:
        """거래 실행"""
        
        if signal.is_hold:
            return None
        
        # 신호 유효성 검증
        is_valid, reason = signal.validate()
        if not is_valid:
            self.logger.warning(f"Invalid signal: {reason}")
            return None
        
        side = OrderSide.BUY if signal.is_buy else OrderSide.SELL
        
        # 포지션 개설 가능 여부 확인
        can_open, reason = self.can_open_position(
            signal.symbol, signal.size, current_price, side
        )
        
        if not can_open:
            self.logger.warning(f"Cannot open position: {reason}")
            return None
        
        # 슬리피지 적용
        execution_price = self._apply_slippage(current_price, side, signal.size)
        
        # 수수료 계산
        commission = signal.size * execution_price * self.commission_rate
        
        # 거래 객체 생성
        trade = Trade(
            timestamp=datetime.now(),
            symbol=signal.symbol,
            side=side,
            size=signal.size,
            price=execution_price,
            commission=commission,
            slippage=abs(execution_price - current_price) / current_price,
            status=TradeStatus.EXECUTED,
            strategy_id=strategy_id,
            metadata=signal.metadata
        )
        
        # 포지션 업데이트
        self._update_position(trade)
        
        # 현금 업데이트
        if side == OrderSide.BUY:
            self.cash -= trade.total_cost
        else:
            self.cash += trade.total_cost
        
        # 거래 기록
        self.all_trades.append(trade)
        self.risk_state.daily_trades_count += 1
        
        # 자산가치 업데이트
        self._update_equity()
        
        # 리스크 콜백 실행
        self._trigger_risk_callbacks(trade)
        
        self.logger.info(f"Trade executed: {trade.trade_id} {side.value} {signal.size} @ {execution_price}")
        
        return trade
    
    def update_market_price(self, symbol: str, price: float):
        """시장 가격 업데이트"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
        
        # 자산가치 업데이트
        self._update_equity()
        
        # 리스크 상태 업데이트
        self._update_risk_state()
    
    def update_multiple_prices(self, prices: Dict[str, float]):
        """여러 심볼의 가격 일괄 업데이트"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        
        self._update_equity()
        self._update_risk_state()
    
    def close_position(self, 
                      symbol: str, 
                      size: Optional[float] = None,
                      reason: str = "") -> Optional[Trade]:
        """포지션 청산"""
        
        if symbol not in self.positions:
            self.logger.warning(f"No position to close for {symbol}")
            return None
        
        position = self.positions[symbol]
        close_size = size or abs(position.size)
        
        if close_size > abs(position.size):
            close_size = abs(position.size)
        
        # 청산 신호 생성
        signal = OrderSignal(
            action='sell' if position.is_long else 'buy',
            symbol=symbol,
            size=close_size,
            reason=reason or "Position close"
        )
        
        return self.execute_trade(signal, position.state.current_price)
    
    def close_all_positions(self, reason: str = "Close all") -> List[Trade]:
        """모든 포지션 청산"""
        closed_trades = []
        
        for symbol in list(self.positions.keys()):
            trade = self.close_position(symbol, reason=reason)
            if trade:
                closed_trades.append(trade)
        
        return closed_trades
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """리스크 지표 계산"""
        equity = self.get_equity()
        
        # 기본 지표
        metrics = {
            'equity': equity,
            'cash': self.cash,
            'total_position_value': sum(pos.state.market_value for pos in self.positions.values()),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
            'total_pnl': sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values()),
            'current_drawdown': self.risk_state.current_drawdown,
            'max_drawdown': self.risk_state.max_drawdown,
            'daily_trades': self.risk_state.daily_trades_count,
            'daily_pnl': self.risk_state.daily_pnl,
            'leverage': self._calculate_leverage(),
            'position_count': len(self.positions)
        }
        
        # VaR/CVaR 계산 (최근 거래 기반)
        if len(self.equity_history) >= 30:
            returns = self._calculate_returns()
            metrics.update({
                'var_95': np.percentile(returns, 5),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                'volatility': np.std(returns) * np.sqrt(252)  # 연환산
            })
        
        return metrics
    
    def check_emergency_conditions(self) -> List[str]:
        """긴급 상황 체크"""
        warnings = []
        equity = self.get_equity()
        
        # 1. 긴급 손절 체크
        total_loss_pct = (self.initial_capital - equity) / self.initial_capital
        if total_loss_pct >= self.trading_limits.emergency_stop_loss:
            warnings.append(f"EMERGENCY: Total loss {total_loss_pct:.2%} >= {self.trading_limits.emergency_stop_loss:.2%}")
        
        # 2. 일일 손실 한도 체크
        daily_loss_pct = -self.risk_state.daily_pnl / equity
        if daily_loss_pct >= self.trading_limits.daily_loss_limit:
            warnings.append(f"Daily loss limit reached: {daily_loss_pct:.2%}")
        
        # 3. 최대 낙폭 체크
        if self.risk_state.current_drawdown >= 0.2:  # 20% 낙폭
            warnings.append(f"High drawdown: {self.risk_state.current_drawdown:.2%}")
        
        # 4. 포지션 집중도 체크
        if self.positions:
            max_position_pct = max(pos.state.market_value / equity for pos in self.positions.values())
            if max_position_pct > 0.5:  # 50% 초과
                warnings.append(f"High position concentration: {max_position_pct:.1%}")
        
        return warnings
    
    def _apply_slippage(self, price: float, side: OrderSide, size: float) -> float:
        """슬리피지 적용"""
        base_slippage = self.slippage_rate
        
        # 거래 크기에 따른 슬리피지 조정
        size_multiplier = min(2.0, 1 + size * 10)  # 크기가 클수록 슬리피지 증가
        adjusted_slippage = base_slippage * size_multiplier
        
        if side == OrderSide.BUY:
            return price * (1 + adjusted_slippage)
        else:
            return price * (1 - adjusted_slippage)
    
    def _update_position(self, trade: Trade):
        """포지션 업데이트"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        self.positions[symbol].add_trade(trade)
        
        # 빈 포지션 제거
        if self.positions[symbol].is_flat:
            del self.positions[symbol]
    
    def _update_equity(self):
        """자산가치 기록 업데이트"""
        current_equity = self.get_equity()
        self.equity_history.append((datetime.now(), current_equity))
        
        # 메모리 관리 (최근 1000개만 유지)
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-500:]
    
    def _update_risk_state(self):
        """리스크 상태 업데이트"""
        current_equity = self.get_equity()
        
        # 최고점 갱신
        if current_equity > self.risk_state.peak_equity:
            self.risk_state.peak_equity = current_equity
        
        # 현재 낙폭 계산
        self.risk_state.current_drawdown = (self.risk_state.peak_equity - current_equity) / self.risk_state.peak_equity
        
        # 최대 낙폭 갱신
        self.risk_state.max_drawdown = max(self.risk_state.max_drawdown, self.risk_state.current_drawdown)
        
        # 일일 손익 계산
        if len(self.equity_history) >= 2:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_equity = [eq for ts, eq in self.equity_history if ts >= today_start]
            if len(today_equity) >= 2:
                self.risk_state.daily_pnl = today_equity[-1] - today_equity[0]
    
    def _calculate_leverage(self) -> float:
        """레버리지 계산"""
        equity = self.get_equity()
        if equity <= 0:
            return 0.0
        
        total_position_value = sum(pos.state.market_value for pos in self.positions.values())
        return total_position_value / equity
    
    def _calculate_returns(self) -> np.ndarray:
        """수익률 계산"""
        if len(self.equity_history) < 2:
            return np.array([])
        
        values = [eq for _, eq in self.equity_history]
        returns = np.diff(values) / values[:-1]
        return returns
    
    def _trigger_risk_callbacks(self, trade: Trade):
        """리스크 콜백 실행"""
        risk_data = {
            'trade': trade,
            'equity': self.get_equity(),
            'risk_metrics': self.get_risk_metrics(),
            'emergency_warnings': self.check_emergency_conditions()
        }
        
        for callback in self.risk_callbacks:
            try:
                callback(risk_data)
            except Exception as e:
                self.logger.error(f"Risk callback error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약"""
        equity = self.get_equity()
        total_return = (equity - self.initial_capital) / self.initial_capital
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_equity': equity,
            'total_return': total_return,
            'cash': self.cash,
            'total_trades': len(self.all_trades),
            'active_positions': len(self.positions),
            'total_commission': sum(t.commission for t in self.all_trades),
            'max_drawdown': self.risk_state.max_drawdown,
            'current_drawdown': self.risk_state.current_drawdown
        }
        
        # 거래별 수익률 분석
        if self.all_trades:
            buy_trades = [t for t in self.all_trades if t.side == OrderSide.BUY]
            sell_trades = [t for t in self.all_trades if t.side == OrderSide.SELL]
            
            summary.update({
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_trade_size': np.mean([t.size for t in self.all_trades]),
                'total_volume': sum(t.value for t in self.all_trades)
            })
        
        return summary
    
    def reset(self):
        """포지션 관리자 초기화"""
        self.positions.clear()
        self.cash = self.initial_capital
        self.current_capital = self.initial_capital
        self.all_trades.clear()
        self.equity_history.clear()
        self.risk_state = RiskState(peak_equity=self.initial_capital)
        self._update_equity()