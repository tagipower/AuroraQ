#!/usr/bin/env python3
"""
VPS 포지션 관리자 (통합 로깅 연동)
AuroraQ Production 포지션 관리 시스템을 VPS 환경에 최적화
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json

# VPS 통합 로깅 시스템
from vps_logging import get_vps_log_integrator, LogCategory, LogLevel

@dataclass
class VPSPosition:
    """VPS 최적화 포지션 정보"""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    strategy: str = "hybrid"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_pnl(self, current_price: float) -> float:
        """현재 손익 계산 (절대값)"""
        return (current_price - self.entry_price) * self.size
    
    def get_pnl_pct(self, current_price: float) -> float:
        """현재 손익률 계산 (%)"""
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * np.sign(self.size)
    
    def should_stop_loss(self, current_price: float) -> bool:
        """손절 조건 확인"""
        if self.stop_loss is None:
            return False
        
        pnl_pct = self.get_pnl_pct(current_price)
        return pnl_pct <= -abs(self.stop_loss)
    
    def should_take_profit(self, current_price: float) -> bool:
        """익절 조건 확인"""
        if self.take_profit is None:
            return False
        
        pnl_pct = self.get_pnl_pct(current_price)
        return pnl_pct >= self.take_profit
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'symbol': self.symbol,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'strategy': self.strategy,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }

@dataclass
class VPSTradingLimits:
    """VPS 거래 한도 설정"""
    max_position_size: float = 0.1      # 최대 포지션 크기 (10%)
    max_daily_trades: int = 10          # 일일 최대 거래
    emergency_stop_loss: float = 0.05   # 긴급 손절 (5%)
    max_drawdown: float = 0.15          # 최대 드로우다운 (15%)
    max_portfolio_risk: float = 0.02    # 최대 포트폴리오 리스크 (2%)
    max_positions: int = 5              # 최대 동시 포지션
    
    # VPS 최적화 설정
    risk_check_interval: int = 10       # 리스크 체크 간격 (초)
    auto_rebalance: bool = True         # 자동 리밸런싱
    emergency_mode_threshold: float = 0.08  # 긴급 모드 임계값

class VPSPositionManager:
    """VPS 최적화 포지션 관리자"""
    
    def __init__(self, limits: VPSTradingLimits, enable_logging: bool = True):
        """
        VPS 포지션 관리자 초기화
        
        Args:
            limits: 거래 한도 설정
            enable_logging: 통합 로깅 활성화
        """
        self.limits = limits
        self.enable_logging = enable_logging
        
        # 통합 로깅 시스템
        if enable_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("vps_position_manager")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 포지션 관리
        self.positions: Dict[str, VPSPosition] = {}
        self.position_history: List[VPSPosition] = []
        
        # 일일 거래 추적
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.daily_pnl = 0.0
        
        # 리스크 관리
        self.total_exposure = 0.0
        self.max_daily_loss = 0.0
        self.emergency_mode = False
        
        # 성능 통계
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0
        }
    
    async def can_open_position(self, 
                              symbol: str,
                              size: float, 
                              current_price: float, 
                              strategy_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        포지션 개설 가능 여부 확인
        
        Args:
            symbol: 거래 심볼
            size: 포지션 크기
            current_price: 현재 가격
            strategy_info: 전략 정보
            
        Returns:
            Tuple[bool, str]: (가능 여부, 사유)
        """
        try:
            # 일일 거래 한도 체크
            today = date.today()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today
                self.daily_pnl = 0.0
            
            if self.daily_trade_count >= self.limits.max_daily_trades:
                return False, f"일일 거래 한도 초과: {self.daily_trade_count}/{self.limits.max_daily_trades}"
            
            # 포지션 크기 한도 체크
            if abs(size) > self.limits.max_position_size:
                return False, f"포지션 크기 한도 초과: {abs(size):.4f} > {self.limits.max_position_size}"
            
            # 최대 포지션 수 체크
            if len(self.positions) >= self.limits.max_positions:
                return False, f"최대 포지션 수 초과: {len(self.positions)}/{self.limits.max_positions}"
            
            # 기존 동일 심볼 포지션 체크
            if symbol in self.positions:
                existing_position = self.positions[symbol]
                current_sign = np.sign(existing_position.size)
                new_sign = np.sign(size)
                
                if current_sign == new_sign:
                    return False, f"동일 방향 포지션 중복: {symbol}"
            
            # 총 노출 한도 체크
            new_exposure = abs(size * current_price)
            if self.total_exposure + new_exposure > self.limits.max_portfolio_risk:
                return False, f"포트폴리오 리스크 한도 초과"
            
            # 긴급 모드 체크
            if self.emergency_mode:
                return False, "긴급 모드 활성화 - 신규 포지션 금지"
            
            # 일일 손실 한도 체크
            if self.daily_pnl < -self.limits.emergency_stop_loss:
                return False, f"일일 손실 한도 초과: {self.daily_pnl:.4f} < -{self.limits.emergency_stop_loss}"
            
            # 전략별 추가 체크
            if strategy_info:
                strategy_check = await self._check_strategy_limits(symbol, size, strategy_info)
                if not strategy_check[0]:
                    return strategy_check
            
            return True, "포지션 개설 가능"
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position check error: {e}")
            return False, f"포지션 체크 오류: {str(e)}"
    
    async def _check_strategy_limits(self, 
                                   symbol: str, 
                                   size: float, 
                                   strategy_info: Dict[str, Any]) -> Tuple[bool, str]:
        """전략별 제한 체크"""
        try:
            strategy_name = strategy_info.get('strategy', 'unknown')
            confidence = strategy_info.get('confidence', 0.5)
            
            # 신뢰도 기반 크기 제한
            min_confidence = 0.6
            if confidence < min_confidence:
                return False, f"전략 신뢰도 부족: {confidence:.3f} < {min_confidence}"
            
            # 전략별 최대 포지션 수 제한
            strategy_positions = [p for p in self.positions.values() if p.strategy == strategy_name]
            max_strategy_positions = 3
            
            if len(strategy_positions) >= max_strategy_positions:
                return False, f"전략별 포지션 한도 초과: {strategy_name}"
            
            return True, "전략 체크 통과"
            
        except Exception as e:
            return False, f"전략 체크 오류: {str(e)}"
    
    async def open_position(self, 
                          symbol: str,
                          size: float,
                          entry_price: float,
                          strategy: str = "hybrid",
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        포지션 개설
        
        Args:
            symbol: 거래 심볼
            size: 포지션 크기
            entry_price: 진입 가격
            strategy: 전략명
            stop_loss: 손절선 (비율)
            take_profit: 익절선 (비율)
            metadata: 추가 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 포지션 생성
            position = VPSPosition(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                entry_time=datetime.now(),
                strategy=strategy,
                stop_loss=stop_loss or self.limits.emergency_stop_loss,
                take_profit=take_profit or 0.1,  # 기본 10% 익절
                metadata=metadata or {}
            )
            
            # 포지션 저장
            self.positions[symbol] = position
            
            # 노출 업데이트
            self.total_exposure += abs(size * entry_price)
            
            # 거래 카운트 업데이트
            self.daily_trade_count += 1
            self.stats["total_trades"] += 1
            
            # 로깅
            if self.logger:
                self.logger.info(
                    f"Position opened: {symbol} {size:+.4f} @ {entry_price:.4f}",
                    strategy=strategy,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # 통합 로깅 (Tagged 범주 - 중요 이벤트)
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_opened",
                    severity="medium",
                    description=f"Position opened: {symbol} {size:+.4f} @ {entry_price:.4f}",
                    symbol=symbol,
                    size=size,
                    entry_price=entry_price,
                    strategy=strategy,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    total_positions=len(self.positions)
                )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position open error: {e}")
            
            # 에러 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_open_failed",
                    severity="high",
                    description=f"Failed to open position: {str(e)}",
                    symbol=symbol,
                    size=size,
                    entry_price=entry_price,
                    error_details=str(e)
                )
            
            return False
    
    async def close_position(self, 
                           symbol: str, 
                           exit_price: float,
                           reason: str = "manual") -> Optional[Dict[str, Any]]:
        """
        포지션 청산
        
        Args:
            symbol: 거래 심볼
            exit_price: 청산 가격
            reason: 청산 사유
            
        Returns:
            Optional[Dict[str, Any]]: 청산 결과 정보
        """
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            
            # 손익 계산
            pnl = position.get_pnl(exit_price)
            pnl_pct = position.get_pnl_pct(exit_price)
            
            # 보유 기간 계산
            holding_duration = datetime.now() - position.entry_time
            
            # 청산 정보
            close_info = {
                "symbol": symbol,
                "size": position.size,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "holding_duration_seconds": holding_duration.total_seconds(),
                "strategy": position.strategy,
                "reason": reason,
                "close_time": datetime.now()
            }
            
            # 포지션 히스토리에 추가
            self.position_history.append(position)
            
            # 포지션 제거
            del self.positions[symbol]
            
            # 노출 업데이트
            self.total_exposure -= abs(position.size * position.entry_price)
            
            # 통계 업데이트
            await self._update_statistics(pnl, pnl_pct)
            
            # 일일 손익 업데이트
            self.daily_pnl += pnl
            
            # 로깅
            if self.logger:
                self.logger.info(
                    f"Position closed: {symbol} {position.size:+.4f} @ {exit_price:.4f} "
                    f"PnL: {pnl:+.4f} ({pnl_pct:+.2%}) Reason: {reason}",
                    holding_duration=holding_duration.total_seconds()
                )
            
            # 통합 로깅 (Tagged 범주)
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_closed",
                    severity="medium" if abs(pnl_pct) < 0.05 else "high",
                    description=f"Position closed: {symbol} PnL: {pnl:+.4f} ({pnl_pct:+.2%})",
                    **close_info,
                    remaining_positions=len(self.positions)
                )
            
            return close_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position close error: {e}")
            
            # 에러 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="position_close_failed",
                    severity="high",
                    description=f"Failed to close position: {str(e)}",
                    symbol=symbol,
                    exit_price=exit_price,
                    error_details=str(e)
                )
            
            return None
    
    async def _update_statistics(self, pnl: float, pnl_pct: float):
        """통계 업데이트"""
        try:
            # 승/패 판정
            if pnl > 0:
                self.stats["winning_trades"] += 1
                self.stats["avg_win"] = ((self.stats["avg_win"] * (self.stats["winning_trades"] - 1)) + pnl) / self.stats["winning_trades"]
            else:
                self.stats["losing_trades"] += 1
                self.stats["avg_loss"] = ((self.stats["avg_loss"] * (self.stats["losing_trades"] - 1)) + abs(pnl)) / self.stats["losing_trades"]
            
            # 총 손익 업데이트
            self.stats["total_pnl"] += pnl
            
            # 승률 계산
            total_closed = self.stats["winning_trades"] + self.stats["losing_trades"]
            if total_closed > 0:
                self.stats["win_rate"] = self.stats["winning_trades"] / total_closed
            
            # 수익 팩터 계산
            if self.stats["avg_loss"] > 0:
                self.stats["profit_factor"] = (self.stats["avg_win"] * self.stats["winning_trades"]) / (self.stats["avg_loss"] * self.stats["losing_trades"])
            
            # 최대 드로우다운 업데이트
            if self.stats["total_pnl"] < self.stats["max_drawdown"]:
                self.stats["max_drawdown"] = self.stats["total_pnl"]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Statistics update error: {e}")
    
    async def check_stop_conditions(self, current_prices: Dict[str, float]) -> List[str]:
        """
        손절/익절 조건 체크
        
        Args:
            current_prices: 현재 가격 딕셔너리 {symbol: price}
            
        Returns:
            List[str]: 청산해야 할 심볼 목록
        """
        symbols_to_close = []
        
        try:
            for symbol, position in self.positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                
                # 손절 체크
                if position.should_stop_loss(current_price):
                    symbols_to_close.append(symbol)
                    
                    if self.logger:
                        self.logger.warning(f"Stop loss triggered: {symbol} @ {current_price}")
                    continue
                
                # 익절 체크
                if position.should_take_profit(current_price):
                    symbols_to_close.append(symbol)
                    
                    if self.logger:
                        self.logger.info(f"Take profit triggered: {symbol} @ {current_price}")
                    continue
                
                # 긴급 손절 체크
                pnl_pct = position.get_pnl_pct(current_price)
                if pnl_pct <= -self.limits.emergency_stop_loss:
                    symbols_to_close.append(symbol)
                    
                    if self.logger:
                        self.logger.critical(f"Emergency stop loss: {symbol} @ {current_price} PnL: {pnl_pct:.2%}")
            
            return symbols_to_close
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stop condition check error: {e}")
            return []
    
    async def close_all_positions(self, current_prices: Dict[str, float], reason: str = "emergency"):
        """
        모든 포지션 청산
        
        Args:
            current_prices: 현재 가격 딕셔너리
            reason: 청산 사유
        """
        try:
            symbols_to_close = list(self.positions.keys())
            
            for symbol in symbols_to_close:
                if symbol in current_prices:
                    await self.close_position(symbol, current_prices[symbol], reason)
                else:
                    # 가격 정보가 없는 경우 진입 가격으로 청산 (보수적)
                    position = self.positions[symbol]
                    await self.close_position(symbol, position.entry_price, f"{reason}_no_price")
            
            # 긴급 모드 활성화
            if reason == "emergency":
                self.emergency_mode = True
                
                if self.log_integrator:
                    await self.log_integrator.log_security_event(
                        event_type="emergency_close_all",
                        severity="critical",
                        description=f"All positions closed - emergency mode activated",
                        total_positions_closed=len(symbols_to_close),
                        daily_pnl=self.daily_pnl
                    )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Close all positions error: {e}")
    
    def get_portfolio_risk(self, current_prices: Dict[str, float]) -> float:
        """
        포트폴리오 전체 리스크 계산
        
        Args:
            current_prices: 현재 가격 딕셔너리
            
        Returns:
            float: 포트폴리오 리스크 비율
        """
        try:
            if not self.positions:
                return 0.0
            
            total_risk = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    position_risk = abs(position.get_pnl_pct(current_price))
                    total_risk += position_risk
            
            return total_risk / len(self.positions) if self.positions else 0.0
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Portfolio risk calculation error: {e}")
            return 0.0
    
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        총 손익 계산
        
        Args:
            current_prices: 현재 가격 딕셔너리
            
        Returns:
            float: 총 손익
        """
        try:
            total_pnl = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    total_pnl += position.get_pnl(current_price)
            
            return total_pnl
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Total PnL calculation error: {e}")
            return 0.0
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        포트폴리오 총 가치 계산
        
        Args:
            current_prices: 현재 가격 딕셔너리
            
        Returns:
            float: 포트폴리오 총 가치
        """
        try:
            total_value = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    position_value = abs(position.size) * current_price
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Portfolio value calculation error: {e}")
            return 0.0
    
    def get_position_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """포지션 요약 정보"""
        try:
            summary = {
                "total_positions": len(self.positions),
                "total_exposure": self.total_exposure,
                "daily_trade_count": self.daily_trade_count,
                "daily_pnl": self.daily_pnl,
                "emergency_mode": self.emergency_mode,
                "positions": [],
                "statistics": self.stats.copy()
            }
            
            # 개별 포지션 정보
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    position_info = {
                        "symbol": symbol,
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "current_price": current_price,
                        "pnl": position.get_pnl(current_price),
                        "pnl_pct": position.get_pnl_pct(current_price),
                        "strategy": position.strategy,
                        "holding_duration": (datetime.now() - position.entry_time).total_seconds(),
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit
                    }
                    summary["positions"].append(position_info)
            
            # 전체 포트폴리오 메트릭
            summary["portfolio_pnl"] = self.get_total_pnl(current_prices)
            summary["portfolio_value"] = self.get_portfolio_value(current_prices)
            summary["portfolio_risk"] = self.get_portfolio_risk(current_prices)
            
            return summary
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position summary error: {e}")
            return {"error": str(e)}
    
    async def reset_emergency_mode(self):
        """긴급 모드 해제"""
        try:
            self.emergency_mode = False
            
            if self.logger:
                self.logger.info("Emergency mode reset")
            
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="emergency_mode_reset",
                    severity="medium",
                    description="Emergency mode has been reset - normal trading resumed"
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Emergency mode reset error: {e}")

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_position_manager(limits: Optional[VPSTradingLimits] = None) -> VPSPositionManager:
    """VPS 최적화된 포지션 관리자 생성"""
    if limits is None:
        limits = VPSTradingLimits()
    
    return VPSPositionManager(limits, enable_logging=True)

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_position_manager():
        limits = VPSTradingLimits()
        manager = create_vps_position_manager(limits)
        
        # 포지션 개설 테스트
        can_open, reason = await manager.can_open_position("BTCUSDT", 0.05, 50000.0)
        print(f"Can open position: {can_open}, Reason: {reason}")
        
        if can_open:
            success = await manager.open_position(
                symbol="BTCUSDT",
                size=0.05,
                entry_price=50000.0,
                strategy="test",
                stop_loss=0.05,
                take_profit=0.1
            )
            print(f"Position opened: {success}")
            
            # 포지션 요약
            current_prices = {"BTCUSDT": 51000.0}
            summary = manager.get_position_summary(current_prices)
            print("Position summary:", json.dumps(summary, indent=2, default=str))
            
            # 손절/익절 체크
            stop_symbols = await manager.check_stop_conditions(current_prices)
            print(f"Symbols to close: {stop_symbols}")
            
            # 포지션 청산
            close_info = await manager.close_position("BTCUSDT", 51000.0, "test_close")
            print("Close info:", close_info)
    
    asyncio.run(test_position_manager())