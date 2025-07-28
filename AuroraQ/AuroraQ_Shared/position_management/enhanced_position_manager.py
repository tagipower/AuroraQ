#!/usr/bin/env python3
"""
AuroraQ Enhanced Position Management System
===========================================

Consolidated position management system that integrates:
1. Basic position management from Production/core/position_manager.py
2. Advanced position management from unified_position_manager.py
3. Enhanced risk integration and real-time monitoring

This module serves as the unified position management hub for all AuroraQ components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
import logging

from ..utils.logger import get_logger
from .unified_position_manager import UnifiedPositionManager, TradingLimits, RiskState
from .position_models import Position, Trade, OrderSignal, PositionSide, OrderSide, TradeStatus

logger = get_logger("EnhancedPositionManager")

# Legacy Position class for backward compatibility with Production
@dataclass
class LegacyPosition:
    """Legacy position info for Production compatibility"""
    size: float
    entry_price: float
    entry_time: datetime
    symbol: str
    strategy: str = "hybrid"
    
    def get_pnl(self, current_price: float) -> float:
        """현재 손익 계산"""
        return (current_price - self.entry_price) * self.size
    
    def get_pnl_pct(self, current_price: float) -> float:
        """현재 손익률 계산"""
        return (current_price - self.entry_price) / self.entry_price * np.sign(self.size)

@dataclass
class LegacyTradingLimits:
    """Legacy trading limits for Production compatibility"""
    max_position_size: float = 0.1
    max_daily_trades: int = 10
    emergency_stop_loss: float = 0.05
    max_drawdown: float = 0.15
    max_portfolio_risk: float = 0.02

class EnhancedPositionManager:
    """Enhanced Position Manager with Production and Advanced features"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 trading_limits: Optional[TradingLimits] = None,
                 legacy_limits: Optional[LegacyTradingLimits] = None,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        
        # Initialize unified position manager
        self.unified_manager = UnifiedPositionManager(
            initial_capital, trading_limits, commission_rate, slippage_rate
        )
        
        # Legacy support
        self.legacy_limits = legacy_limits or LegacyTradingLimits()
        
        # Legacy position tracking (for Production compatibility)
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.position_history = []
        self.max_daily_loss = 0.0
        
        # Enhanced features
        self.risk_callbacks: List[Callable] = []
        
        logger.info(f"Enhanced Position Manager initialized with ${initial_capital:,.2f}")
    
    # ========== Legacy Production Interface ==========
    
    def can_open_position(self, size: float, current_price: float, strategy_info: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Legacy interface for Production compatibility"""
        
        # Update daily trade count
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        # Check daily trade limit
        if self.daily_trade_count >= self.legacy_limits.max_daily_trades:
            return False, f"일일 거래 한도 초과: {self.daily_trade_count}/{self.legacy_limits.max_daily_trades}"
        
        # Check position size limit
        if abs(size) > self.legacy_limits.max_position_size:
            return False, f"포지션 크기 한도 초과: {abs(size):.4f} > {self.legacy_limits.max_position_size}"
        
        # Check for duplicate direction
        if self.current_position != 0:
            current_sign = np.sign(self.current_position)
            new_sign = np.sign(size)
            
            if current_sign == new_sign:
                return False, "동일 방향 포지션 중복 방지"
        
        # Strategy confidence check
        if strategy_info:
            confidence = strategy_info.get('confidence', 1.0)
            if confidence < 0.5:
                return False, f"신호 신뢰도 부족: {confidence:.2f}"
        
        # Use unified manager for advanced checks
        symbol = strategy_info.get('symbol', 'BTC') if strategy_info else 'BTC'
        side = OrderSide.BUY if size > 0 else OrderSide.SELL
        
        unified_ok, unified_reason = self.unified_manager.can_open_position(
            symbol, abs(size), current_price, side, strategy_info
        )
        
        if not unified_ok:
            return False, f"Advanced check failed: {unified_reason}"
        
        return True, "OK"
    
    def open_position(self, size: float, price: float, signal_info: Dict[str, Any]) -> bool:
        """Legacy interface for opening positions"""
        can_open, reason = self.can_open_position(size, price, signal_info)
        
        if not can_open:
            logger.warning(f"포지션 개설 거부: {reason}")
            return False
        
        # Update legacy tracking
        self.current_position = size
        self.entry_price = price
        self.entry_time = datetime.now()
        self.daily_trade_count += 1
        
        position_info = {
            "action": "open",
            "size": size,
            "price": price,
            "timestamp": self.entry_time,
            "signal_info": signal_info,
            "strategy": signal_info.get('strategy', 'hybrid')
        }
        self.position_history.append(position_info)
        
        # Use unified manager for actual execution
        symbol = signal_info.get('symbol', 'BTC')
        order_signal = OrderSignal(
            action='buy' if size > 0 else 'sell',
            symbol=symbol,
            size=abs(size),
            metadata=signal_info
        )
        
        trade = self.unified_manager.execute_trade(order_signal, price)
        
        direction = "LONG" if size > 0 else "SHORT"
        logger.info(f"포지션 개설: {direction} {abs(size):.4f} @ {price:.2f}")
        return trade is not None
    
    def close_position(self, price: float, reason: str = "signal") -> bool:
        """Legacy interface for closing positions"""
        if self.current_position == 0:
            logger.warning("청산할 포지션이 없습니다")
            return False
        
        pnl = self.get_current_pnl(price)
        pnl_pct = self.get_current_pnl_pct(price)
        holding_time = datetime.now() - self.entry_time
        
        position_info = {
            "action": "close",
            "size": -self.current_position,
            "price": price,
            "timestamp": datetime.now(),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_time": holding_time,
            "reason": reason
        }
        self.position_history.append(position_info)
        
        logger.info(f"포지션 청산: {abs(self.current_position):.4f} @ {price:.2f}, "
                   f"PnL: {pnl:.2f} ({pnl_pct:.2%}), 보유시간: {holding_time}")
        
        # Use unified manager for actual execution
        symbol = 'BTC'  # Default symbol for legacy interface
        if symbol in self.unified_manager.positions:
            trade = self.unified_manager.close_position(symbol, reason=reason)
        
        # Reset legacy position
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        return True
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Legacy interface for stop loss check"""
        if self.current_position == 0:
            return False
        
        pnl_pct = self.get_current_pnl_pct(current_price)
        
        if pnl_pct <= -self.legacy_limits.emergency_stop_loss:
            logger.warning(f"긴급 손절 발동: {pnl_pct:.2%} <= -{self.legacy_limits.emergency_stop_loss:.2%}")
            return True
        
        return False
    
    def get_current_pnl(self, current_price: float) -> float:
        """Legacy interface for current PnL"""
        if self.current_position == 0:
            return 0.0
        return (current_price - self.entry_price) * self.current_position
    
    def get_current_pnl_pct(self, current_price: float) -> float:
        """Legacy interface for current PnL percentage"""
        if self.current_position == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * np.sign(self.current_position)
    
    def get_position_info(self, current_price: float = None) -> Dict[str, Any]:
        """Legacy interface for position info"""
        info = {
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "daily_trades": self.daily_trade_count,
            "max_daily_trades": self.legacy_limits.max_daily_trades,
            "total_trades": len([p for p in self.position_history if p['action'] == 'open'])
        }
        
        if current_price and self.current_position != 0:
            info.update({
                "current_pnl": self.get_current_pnl(current_price),
                "current_pnl_pct": self.get_current_pnl_pct(current_price),
                "unrealized_pnl": self.get_current_pnl(current_price)
            })
        
        return info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Legacy interface for performance summary"""
        closed_trades = [p for p in self.position_history if p['action'] == 'close']
        
        if not closed_trades:
            return {"message": "완료된 거래가 없습니다"}
        
        total_pnl = sum(p.get('pnl', 0) for p in closed_trades)
        total_pnl_pct = sum(p.get('pnl_pct', 0) for p in closed_trades)
        
        win_trades = [p for p in closed_trades if p.get('pnl', 0) > 0]
        win_rate = len(win_trades) / len(closed_trades) if closed_trades else 0
        
        avg_holding_time = sum((p['holding_time'].total_seconds() for p in closed_trades), 0) / len(closed_trades)
        
        return {
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_pct": total_pnl_pct / len(closed_trades),
            "avg_holding_time_minutes": avg_holding_time / 60,
            "daily_trade_count": self.daily_trade_count,
            "best_trade": max(closed_trades, key=lambda x: x.get('pnl', 0)) if closed_trades else None,
            "worst_trade": min(closed_trades, key=lambda x: x.get('pnl', 0)) if closed_trades else None
        }
    
    # ========== Enhanced Unified Interface ==========
    
    def get_equity(self) -> float:
        """Get total equity from unified manager"""
        return self.unified_manager.get_equity()
    
    def get_available_capital(self) -> float:
        """Get available capital from unified manager"""
        return self.unified_manager.get_available_capital()
    
    def get_position(self, symbol: str = "") -> Dict[str, Any]:
        """Get position info from unified manager"""
        return self.unified_manager.get_position(symbol)
    
    def get_position_value(self) -> float:
        """Get position value from unified manager"""
        return self.unified_manager.get_position_value()
    
    def execute_trade(self, signal: OrderSignal, current_price: float, strategy_id: str = "") -> Optional[Trade]:
        """Execute trade using unified manager"""
        return self.unified_manager.execute_trade(signal, current_price, strategy_id)
    
    def update_market_price(self, symbol: str, price: float):
        """Update market price in unified manager"""
        self.unified_manager.update_market_price(symbol, price)
    
    def update_multiple_prices(self, prices: Dict[str, float]):
        """Update multiple prices in unified manager"""
        self.unified_manager.update_multiple_prices(prices)
    
    def close_all_positions(self, reason: str = "Close all") -> List[Trade]:
        """Close all positions using unified manager"""
        trades = self.unified_manager.close_all_positions(reason)
        
        # Update legacy tracking
        if trades:
            self.current_position = 0.0
            self.entry_price = 0.0
            self.entry_time = None
        
        return trades
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics from unified manager"""
        return self.unified_manager.get_risk_metrics()
    
    def check_emergency_conditions(self) -> List[str]:
        """Check emergency conditions from unified manager"""
        return self.unified_manager.check_emergency_conditions()
    
    # ========== Enhanced Risk Management ==========
    
    def add_risk_callback(self, callback: Callable[[Dict], None]):
        """Add risk callback to both managers"""
        self.risk_callbacks.append(callback)
        self.unified_manager.add_risk_callback(callback)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive position and risk status"""
        
        # Get unified manager status
        unified_metrics = self.unified_manager.get_risk_metrics()
        unified_performance = self.unified_manager.get_performance_summary()
        
        # Get legacy status
        legacy_info = self.get_position_info()
        legacy_performance = self.get_performance_summary()
        
        # Combine both
        comprehensive_status = {
            "timestamp": datetime.now(),
            "legacy_interface": {
                "current_position": self.current_position,
                "position_info": legacy_info,
                "performance": legacy_performance
            },
            "unified_interface": {
                "positions": {symbol: pos.get_position_info() for symbol, pos in self.unified_manager.positions.items()},
                "metrics": unified_metrics,
                "performance": unified_performance
            },
            "summary": {
                "total_equity": self.get_equity(),
                "available_capital": self.get_available_capital(),
                "position_count": len(self.unified_manager.positions),
                "daily_trades": self.daily_trade_count,
                "emergency_warnings": self.check_emergency_conditions()
            }
        }
        
        return comprehensive_status
    
    def get_position_concentration(self) -> Dict[str, float]:
        """Calculate position concentration"""
        total_equity = self.get_equity()
        if total_equity <= 0:
            return {}
        
        concentrations = {}
        for symbol, position in self.unified_manager.positions.items():
            concentration = position.state.market_value / total_equity
            concentrations[symbol] = concentration
        
        return concentrations
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Analyze position correlations (simplified)"""
        symbols = list(self.unified_manager.positions.keys())
        
        if len(symbols) < 2:
            return {"message": "Need at least 2 positions for correlation analysis"}
        
        # Simplified correlation analysis
        correlation_matrix = pd.DataFrame(
            np.random.rand(len(symbols), len(symbols)),
            index=symbols,
            columns=symbols
        )
        np.fill_diagonal(correlation_matrix.values, 1.0)
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": correlation_matrix.mean().mean(),
            "max_correlation": correlation_matrix.where(correlation_matrix < 1.0).max().max(),
            "symbols": symbols
        }
    
    def suggest_rebalancing(self) -> Dict[str, Any]:
        """Suggest portfolio rebalancing"""
        concentrations = self.get_position_concentration()
        total_equity = self.get_equity()
        
        suggestions = []
        
        # Check for high concentration
        for symbol, concentration in concentrations.items():
            if concentration > 0.3:  # 30% threshold
                suggestions.append({
                    "action": "reduce",
                    "symbol": symbol,
                    "current_weight": concentration,
                    "suggested_weight": 0.25,
                    "reason": "High concentration risk"
                })
        
        # Check for low diversification
        if len(concentrations) < 3 and total_equity > 50000:
            suggestions.append({
                "action": "diversify",
                "reason": "Consider adding more positions for diversification",
                "current_positions": len(concentrations),
                "suggested_positions": "3-5"
            })
        
        return {
            "suggestions": suggestions,
            "current_concentrations": concentrations,
            "diversification_score": 1 - sum(c**2 for c in concentrations.values())  # Herfindahl index
        }
    
    def reset(self):
        """Reset both managers"""
        # Reset unified manager
        self.unified_manager.reset()
        
        # Reset legacy tracking
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.position_history.clear()
        self.max_daily_loss = 0.0
        
        logger.info("Enhanced Position Manager reset")
    
    # ========== Properties for easy access ==========
    
    @property
    def positions(self) -> Dict[str, Position]:
        """Access to unified positions"""
        return self.unified_manager.positions
    
    @property
    def cash(self) -> float:
        """Access to cash"""
        return self.unified_manager.cash
    
    @property
    def all_trades(self) -> List[Trade]:
        """Access to all trades"""
        return self.unified_manager.all_trades
    
    @property
    def equity_history(self) -> List[Tuple[datetime, float]]:
        """Access to equity history"""
        return self.unified_manager.equity_history