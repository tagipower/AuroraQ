#!/usr/bin/env python3
"""
포지션 매니저
포지션 개설/청산, 리스크 관리, 거래 한도 체크
"""

import numpy as np
from datetime import datetime, date
from typing import Dict, Any, List
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("PositionManager")

@dataclass
class Position:
    """포지션 정보"""
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
class TradingLimits:
    """거래 한도"""
    max_position_size: float = 0.1
    max_daily_trades: int = 10
    emergency_stop_loss: float = 0.05
    max_drawdown: float = 0.15
    max_portfolio_risk: float = 0.02

class PositionManager:
    """포지션 관리자"""
    
    def __init__(self, limits: TradingLimits):
        self.limits = limits
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.position_history = []
        self.max_daily_loss = 0.0
        
    def can_open_position(self, size: float, current_price: float, strategy_info: Dict[str, Any] = None) -> tuple[bool, str]:
        """포지션 개설 가능 여부 확인"""
        
        # 일일 거래 한도 체크
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        if self.daily_trade_count >= self.limits.max_daily_trades:
            return False, f"일일 거래 한도 초과: {self.daily_trade_count}/{self.limits.max_daily_trades}"
        
        # 포지션 크기 한도 체크
        if abs(size) > self.limits.max_position_size:
            return False, f"포지션 크기 한도 초과: {abs(size):.4f} > {self.limits.max_position_size}"
        
        # 기존 포지션과 방향 체크
        if self.current_position != 0:
            current_sign = np.sign(self.current_position)
            new_sign = np.sign(size)
            
            if current_sign == new_sign:
                return False, "동일 방향 포지션 중복 방지"
        
        # 전략별 추가 체크
        if strategy_info:
            confidence = strategy_info.get('confidence', 1.0)
            if confidence < 0.5:
                return False, f"신호 신뢰도 부족: {confidence:.2f}"
        
        return True, "OK"
    
    def open_position(self, size: float, price: float, signal_info: Dict[str, Any]) -> bool:
        """포지션 개설"""
        can_open, reason = self.can_open_position(size, price, signal_info)
        
        if not can_open:
            logger.warning(f"포지션 개설 거부: {reason}")
            return False
        
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
        
        direction = "LONG" if size > 0 else "SHORT"
        logger.info(f"포지션 개설: {direction} {abs(size):.4f} @ {price:.2f}")
        return True
    
    def close_position(self, price: float, reason: str = "signal") -> bool:
        """포지션 청산"""
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
        
        # 포지션 초기화
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        return True
    
    def check_stop_loss(self, current_price: float) -> bool:
        """손절 체크"""
        if self.current_position == 0:
            return False
        
        pnl_pct = self.get_current_pnl_pct(current_price)
        
        if pnl_pct <= -self.limits.emergency_stop_loss:
            logger.warning(f"긴급 손절 발동: {pnl_pct:.2%} <= -{self.limits.emergency_stop_loss:.2%}")
            return True
        
        return False
    
    def get_current_pnl(self, current_price: float) -> float:
        """현재 손익 조회"""
        if self.current_position == 0:
            return 0.0
        return (current_price - self.entry_price) * self.current_position
    
    def get_current_pnl_pct(self, current_price: float) -> float:
        """현재 손익률 조회"""
        if self.current_position == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * np.sign(self.current_position)
    
    def get_position_info(self, current_price: float = None) -> Dict[str, Any]:
        """포지션 정보 조회"""
        info = {
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "daily_trades": self.daily_trade_count,
            "max_daily_trades": self.limits.max_daily_trades,
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
        """성과 요약 조회"""
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