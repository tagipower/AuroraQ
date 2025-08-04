#!/usr/bin/env python3
"""
거래 전 안전성 검사 시스템
P6-2: 거래 전 안전성 검사 구현
"""

import sys
import os
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import json
import warnings

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """안전성 레벨"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    BLOCKED = "blocked"

class SafetyCheckType(Enum):
    """안전성 검사 타입"""
    POSITION_SIZE = "position_size"
    ACCOUNT_BALANCE = "account_balance"
    RISK_EXPOSURE = "risk_exposure"
    MARKET_CONDITIONS = "market_conditions"
    TRADING_HOURS = "trading_hours"
    SYMBOL_VALIDATION = "symbol_validation"
    ORDER_LIMITS = "order_limits"
    CORRELATION_RISK = "correlation_risk"
    VOLATILITY_CHECK = "volatility_check"
    LIQUIDITY_CHECK = "liquidity_check"

class ActionType(Enum):
    """액션 타입"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

@dataclass
class TradingSignal:
    """거래 신호"""
    symbol: str
    action: ActionType
    quantity: float
    price: Optional[float] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = ""
    strategy_name: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccountState:
    """계좌 상태"""
    total_balance: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> quantity
    open_orders: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class MarketConditions:
    """시장 상황"""
    market_open: bool = True
    volatility_index: float = 0.0  # VIX 등
    market_trend: str = "neutral"  # bullish, bearish, neutral
    trading_volume: float = 0.0
    bid_ask_spread: float = 0.0
    liquidity_score: float = 100.0  # 0-100
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SafetyCheckResult:
    """안전성 검사 결과"""
    check_type: SafetyCheckType
    level: SafetyLevel
    passed: bool
    score: float  # 0-100
    message: str
    recommendations: List[str] = field(default_factory=list)
    blocking_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'check_type': self.check_type.value,
            'level': self.level.value,
            'passed': self.passed,
            'score': self.score,
            'message': self.message,
            'recommendations': self.recommendations,
            'blocking_reason': self.blocking_reason,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class SafetyReport:
    """종합 안전성 보고서"""
    signal_id: str
    overall_level: SafetyLevel
    overall_score: float  # 0-100
    passed: bool
    check_results: List[SafetyCheckResult]
    final_recommendation: str
    trade_allowed: bool
    adjusted_quantity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'signal_id': self.signal_id,
            'overall_level': self.overall_level.value,
            'overall_score': self.overall_score,
            'passed': self.passed,
            'check_results': [result.to_dict() for result in self.check_results],
            'final_recommendation': self.final_recommendation,
            'trade_allowed': self.trade_allowed,
            'adjusted_quantity': self.adjusted_quantity,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class SafetyConfig:
    """안전성 설정"""
    # 포지션 크기 제한
    max_position_size_percent: float = 10.0  # 총 자금의 %
    max_single_trade_percent: float = 2.0    # 단일 거래 최대 %
    max_daily_loss_percent: float = 5.0      # 일일 최대 손실 %
    
    # 리스크 관리
    max_correlation_exposure: float = 30.0   # 상관관계 노출 한도 %
    min_liquidity_score: float = 50.0        # 최소 유동성 점수
    max_volatility_threshold: float = 50.0   # 최대 변동성 임계값
    
    # 거래 시간
    trading_start_time: time = time(9, 0)    # 거래 시작 시간
    trading_end_time: time = time(15, 30)    # 거래 종료 시간
    
    # 계좌 보호
    min_balance_threshold: float = 1000.0    # 최소 잔고
    max_leverage: float = 2.0                # 최대 레버리지
    
    # 알림 임계값
    caution_threshold: float = 70.0          # 주의 임계값
    warning_threshold: float = 50.0          # 경고 임계값
    danger_threshold: float = 30.0           # 위험 임계값

class PreTradeSafetyChecker:
    """거래 전 안전성 검사기"""
    
    def __init__(self, config: Optional[SafetyConfig] = None, config_file: str = "safety_config.json"):
        self.config = config or SafetyConfig()
        self.config_file = config_file
        
        # 상태 추적
        self.account_state = AccountState()
        self.market_conditions = MarketConditions()
        
        # 검사 히스토리
        self.check_history: List[SafetyReport] = []
        self.max_history_size = 1000
        
        # 일일 통계
        self.daily_stats = {
            "date": datetime.now().date(),
            "total_trades": 0,
            "blocked_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "risk_score": 0.0
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 설정 로드
        self._load_configuration()
        
        logger.info("Pre-trade safety checker initialized")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 업데이트
                for key, value in config_data.get('safety_config', {}).items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # 통계 로드
                if 'daily_stats' in config_data:
                    stats_date = datetime.fromisoformat(config_data['daily_stats']['date']).date()
                    if stats_date == datetime.now().date():
                        self.daily_stats.update(config_data['daily_stats'])
                
                logger.info(f"Safety configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load safety configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config_data = {
                'safety_config': {
                    'max_position_size_percent': self.config.max_position_size_percent,
                    'max_single_trade_percent': self.config.max_single_trade_percent,
                    'max_daily_loss_percent': self.config.max_daily_loss_percent,
                    'max_correlation_exposure': self.config.max_correlation_exposure,
                    'min_liquidity_score': self.config.min_liquidity_score,
                    'max_volatility_threshold': self.config.max_volatility_threshold,
                    'min_balance_threshold': self.config.min_balance_threshold,
                    'max_leverage': self.config.max_leverage,
                    'caution_threshold': self.config.caution_threshold,
                    'warning_threshold': self.config.warning_threshold,
                    'danger_threshold': self.config.danger_threshold
                },
                'daily_stats': {
                    **self.daily_stats,
                    'date': self.daily_stats['date'].isoformat()
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save safety configuration: {e}")
    
    async def perform_safety_check(self, signal: TradingSignal) -> SafetyReport:
        """종합 안전성 검사 수행"""
        try:
            logger.info(f"Starting safety check for signal: {signal.signal_id}")
            
            check_results = []
            
            # 1. 포지션 크기 검사
            result = await self._check_position_size(signal)
            check_results.append(result)
            
            # 2. 계좌 잔고 검사
            result = await self._check_account_balance(signal)
            check_results.append(result)
            
            # 3. 리스크 노출 검사
            result = await self._check_risk_exposure(signal)
            check_results.append(result)
            
            # 4. 시장 상황 검사
            result = await self._check_market_conditions(signal)
            check_results.append(result)
            
            # 5. 거래 시간 검사
            result = await self._check_trading_hours(signal)
            check_results.append(result)
            
            # 6. 심볼 유효성 검사
            result = await self._check_symbol_validation(signal)
            check_results.append(result)
            
            # 7. 주문 한도 검사
            result = await self._check_order_limits(signal)
            check_results.append(result)
            
            # 8. 상관관계 리스크 검사
            result = await self._check_correlation_risk(signal)
            check_results.append(result)
            
            # 9. 변동성 검사
            result = await self._check_volatility(signal)
            check_results.append(result)
            
            # 10. 유동성 검사
            result = await self._check_liquidity(signal)
            check_results.append(result)
            
            # 종합 평가
            report = self._generate_safety_report(signal, check_results)
            
            # 히스토리에 추가
            with self._lock:
                self.check_history.append(report)
                if len(self.check_history) > self.max_history_size:
                    self.check_history.pop(0)
            
            # 통계 업데이트
            self._update_daily_stats(report)
            
            logger.info(f"Safety check completed: {report.overall_level.value} (score: {report.overall_score:.1f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return SafetyReport(
                signal_id=signal.signal_id,
                overall_level=SafetyLevel.BLOCKED,
                overall_score=0.0,
                passed=False,
                check_results=[],
                final_recommendation="Safety check failed due to system error",
                trade_allowed=False
            )
    
    async def _check_position_size(self, signal: TradingSignal) -> SafetyCheckResult:
        """포지션 크기 검사"""
        try:
            # 계산된 포지션 크기 (가정)
            position_value = signal.quantity * (signal.price or 100.0)
            position_percent = (position_value / self.account_state.total_balance) * 100
            
            score = 100.0
            level = SafetyLevel.SAFE
            passed = True
            message = f"Position size: {position_percent:.1f}% of account"
            recommendations = []
            
            if position_percent > self.config.max_single_trade_percent:
                score = max(0, 100 - (position_percent - self.config.max_single_trade_percent) * 10)
                level = SafetyLevel.BLOCKED
                passed = False
                message += f" (exceeds {self.config.max_single_trade_percent}% limit)"
                recommendations.append(f"Reduce position size to max {self.config.max_single_trade_percent}% of account")
            elif position_percent > self.config.max_single_trade_percent * 0.8:
                score = 60.0
                level = SafetyLevel.WARNING
                recommendations.append("Consider reducing position size")
            elif position_percent > self.config.max_single_trade_percent * 0.6:
                score = 80.0
                level = SafetyLevel.CAUTION
                
            return SafetyCheckResult(
                check_type=SafetyCheckType.POSITION_SIZE,
                level=level,
                passed=passed,
                score=score,
                message=message,
                recommendations=recommendations,
                blocking_reason="Position size too large" if not passed else None
            )
            
        except Exception as e:
            logger.error(f"Position size check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.POSITION_SIZE,
                level=SafetyLevel.BLOCKED,
                passed=False,
                score=0.0,
                message="Position size check failed",
                blocking_reason="System error"
            )
    
    async def _check_account_balance(self, signal: TradingSignal) -> SafetyCheckResult:
        """계좌 잔고 검사"""
        try:
            available_balance = self.account_state.available_balance
            min_threshold = self.config.min_balance_threshold
            
            score = 100.0
            level = SafetyLevel.SAFE
            passed = True
            message = f"Available balance: ${available_balance:.2f}"
            recommendations = []
            
            if available_balance < min_threshold:
                score = 0.0
                level = SafetyLevel.BLOCKED
                passed = False
                message += f" (below ${min_threshold:.2f} minimum)"
                recommendations.append("Deposit funds to meet minimum balance requirement")
            elif available_balance < min_threshold * 2:
                score = 30.0
                level = SafetyLevel.DANGER
                recommendations.append("Consider depositing additional funds")
            elif available_balance < min_threshold * 5:
                score = 70.0
                level = SafetyLevel.CAUTION
                
            return SafetyCheckResult(
                check_type=SafetyCheckType.ACCOUNT_BALANCE,
                level=level,
                passed=passed,
                score=score,
                message=message,
                recommendations=recommendations,
                blocking_reason="Insufficient balance" if not passed else None
            )
            
        except Exception as e:
            logger.error(f"Account balance check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.ACCOUNT_BALANCE,
                level=SafetyLevel.BLOCKED,
                passed=False,
                score=0.0,
                message="Account balance check failed",
                blocking_reason="System error"
            )
    
    async def _check_risk_exposure(self, signal: TradingSignal) -> SafetyCheckResult:
        """리스크 노출 검사"""
        try:
            # 현재 총 포지션 가치
            total_exposure = sum(abs(qty * 100.0) for qty in self.account_state.positions.values())
            exposure_percent = (total_exposure / self.account_state.total_balance) * 100
            
            score = 100.0
            level = SafetyLevel.SAFE
            passed = True
            message = f"Risk exposure: {exposure_percent:.1f}% of account"
            recommendations = []
            
            if exposure_percent > 80.0:
                score = 20.0
                level = SafetyLevel.DANGER
                recommendations.append("High risk exposure - consider reducing positions")
            elif exposure_percent > 60.0:
                score = 50.0
                level = SafetyLevel.WARNING
                recommendations.append("Moderate risk exposure - monitor closely")
            elif exposure_percent > 40.0:
                score = 80.0
                level = SafetyLevel.CAUTION
                
            return SafetyCheckResult(
                check_type=SafetyCheckType.RISK_EXPOSURE,
                level=level,
                passed=passed,
                score=score,
                message=message,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Risk exposure check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.RISK_EXPOSURE,
                level=SafetyLevel.WARNING,
                passed=True,
                score=50.0,
                message="Risk exposure check failed",
                recommendations=["Manual risk assessment recommended"]
            )
    
    async def _check_market_conditions(self, signal: TradingSignal) -> SafetyCheckResult:
        """시장 상황 검사"""
        try:
            if not self.market_conditions.market_open:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.MARKET_CONDITIONS,
                    level=SafetyLevel.BLOCKED,
                    passed=False,
                    score=0.0,
                    message="Market is closed",
                    blocking_reason="Market closed"
                )
            
            volatility = self.market_conditions.volatility_index
            score = 100.0
            level = SafetyLevel.SAFE
            passed = True
            message = f"Market volatility: {volatility:.1f}"
            recommendations = []
            
            if volatility > self.config.max_volatility_threshold:
                score = 30.0
                level = SafetyLevel.DANGER
                recommendations.append("High market volatility - consider reducing position sizes")
            elif volatility > self.config.max_volatility_threshold * 0.8:
                score = 60.0
                level = SafetyLevel.WARNING
                recommendations.append("Elevated volatility - exercise caution")
            elif volatility > self.config.max_volatility_threshold * 0.6:
                score = 80.0
                level = SafetyLevel.CAUTION
                
            return SafetyCheckResult(
                check_type=SafetyCheckType.MARKET_CONDITIONS,
                level=level,
                passed=passed,
                score=score,
                message=message,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Market conditions check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.MARKET_CONDITIONS,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Market conditions check failed - assuming caution",
                recommendations=["Manual market assessment recommended"]
            )
    
    async def _check_trading_hours(self, signal: TradingSignal) -> SafetyCheckResult:
        """거래 시간 검사"""
        try:
            current_time = datetime.now().time()
            start_time = self.config.trading_start_time
            end_time = self.config.trading_end_time
            
            if start_time <= current_time <= end_time:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.TRADING_HOURS,
                    level=SafetyLevel.SAFE,
                    passed=True,
                    score=100.0,
                    message=f"Trading hours: {current_time.strftime('%H:%M')}"
                )
            else:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.TRADING_HOURS,
                    level=SafetyLevel.BLOCKED,
                    passed=False,
                    score=0.0,
                    message=f"Outside trading hours: {current_time.strftime('%H:%M')}",
                    blocking_reason="Outside trading hours"
                )
                
        except Exception as e:
            logger.error(f"Trading hours check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.TRADING_HOURS,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Trading hours check failed - assuming caution"
            )
    
    async def _check_symbol_validation(self, signal: TradingSignal) -> SafetyCheckResult:
        """심볼 유효성 검사"""
        try:
            # 간단한 심볼 검증 (실제로는 더 복잡한 검증 필요)
            symbol = signal.symbol.upper()
            
            if not symbol or len(symbol) < 2:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.SYMBOL_VALIDATION,
                    level=SafetyLevel.BLOCKED,
                    passed=False,
                    score=0.0,
                    message="Invalid symbol format",
                    blocking_reason="Invalid symbol"
                )
            
            # 허용된 심볼 목록 (예시)
            allowed_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
            
            if symbol not in allowed_symbols:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.SYMBOL_VALIDATION,
                    level=SafetyLevel.WARNING,
                    passed=True,
                    score=60.0,
                    message=f"Symbol {symbol} not in preferred list",
                    recommendations=["Verify symbol is tradeable and liquid"]
                )
            
            return SafetyCheckResult(
                check_type=SafetyCheckType.SYMBOL_VALIDATION,
                level=SafetyLevel.SAFE,
                passed=True,
                score=100.0,
                message=f"Symbol {symbol} validated"
            )
            
        except Exception as e:
            logger.error(f"Symbol validation check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.SYMBOL_VALIDATION,
                level=SafetyLevel.WARNING,
                passed=True,
                score=50.0,
                message="Symbol validation check failed"
            )
    
    async def _check_order_limits(self, signal: TradingSignal) -> SafetyCheckResult:
        """주문 한도 검사"""
        try:
            # 일일 거래 횟수 체크
            daily_trades = self.daily_stats["total_trades"]
            max_daily_trades = 50  # 설정 가능
            
            if daily_trades >= max_daily_trades:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.ORDER_LIMITS,
                    level=SafetyLevel.BLOCKED,
                    passed=False,
                    score=0.0,
                    message=f"Daily trade limit reached: {daily_trades}/{max_daily_trades}",
                    blocking_reason="Daily trade limit exceeded"
                )
            
            score = 100.0 - (daily_trades / max_daily_trades * 100)
            level = SafetyLevel.SAFE if score > 70 else SafetyLevel.CAUTION
            
            return SafetyCheckResult(
                check_type=SafetyCheckType.ORDER_LIMITS,
                level=level,
                passed=True,
                score=score,
                message=f"Daily trades: {daily_trades}/{max_daily_trades}"
            )
            
        except Exception as e:
            logger.error(f"Order limits check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.ORDER_LIMITS,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Order limits check failed"
            )
    
    async def _check_correlation_risk(self, signal: TradingSignal) -> SafetyCheckResult:
        """상관관계 리스크 검사"""
        try:
            # 현재 포지션과의 상관관계 분석 (간소화)
            # 실제로는 더 복잡한 상관관계 매트릭스 필요
            
            current_symbols = list(self.account_state.positions.keys())
            if signal.symbol in current_symbols:
                level = SafetyLevel.WARNING
                score = 60.0
                message = f"Already have position in {signal.symbol}"
                recommendations = ["Consider correlation with existing position"]
            else:
                level = SafetyLevel.SAFE
                score = 100.0
                message = "No correlation risk detected"
                recommendations = []
            
            return SafetyCheckResult(
                check_type=SafetyCheckType.CORRELATION_RISK,
                level=level,
                passed=True,
                score=score,
                message=message,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Correlation risk check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.CORRELATION_RISK,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Correlation risk check failed"
            )
    
    async def _check_volatility(self, signal: TradingSignal) -> SafetyCheckResult:
        """변동성 검사"""
        try:
            # 심볼별 변동성 체크 (예시)
            volatility = 25.0  # 실제로는 API에서 가져와야 함
            
            score = 100.0
            level = SafetyLevel.SAFE
            message = f"Symbol volatility: {volatility:.1f}%"
            recommendations = []
            
            if volatility > 50.0:
                score = 30.0
                level = SafetyLevel.DANGER
                recommendations.append("High volatility - consider smaller position")
            elif volatility > 35.0:
                score = 60.0
                level = SafetyLevel.WARNING
                recommendations.append("Elevated volatility - exercise caution")
            elif volatility > 25.0:
                score = 80.0
                level = SafetyLevel.CAUTION
            
            return SafetyCheckResult(
                check_type=SafetyCheckType.VOLATILITY_CHECK,
                level=level,
                passed=True,
                score=score,
                message=message,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Volatility check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.VOLATILITY_CHECK,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Volatility check failed"
            )
    
    async def _check_liquidity(self, signal: TradingSignal) -> SafetyCheckResult:
        """유동성 검사"""
        try:
            liquidity_score = self.market_conditions.liquidity_score
            
            score = liquidity_score
            level = SafetyLevel.SAFE
            message = f"Liquidity score: {liquidity_score:.1f}/100"
            recommendations = []
            
            if liquidity_score < self.config.min_liquidity_score:
                level = SafetyLevel.WARNING
                recommendations.append("Low liquidity - consider reducing position size")
            elif liquidity_score < self.config.min_liquidity_score * 1.5:
                level = SafetyLevel.CAUTION
                recommendations.append("Moderate liquidity - monitor execution")
            
            return SafetyCheckResult(
                check_type=SafetyCheckType.LIQUIDITY_CHECK,
                level=level,
                passed=True,
                score=score,
                message=message,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Liquidity check failed: {e}")
            return SafetyCheckResult(
                check_type=SafetyCheckType.LIQUIDITY_CHECK,
                level=SafetyLevel.CAUTION,
                passed=True,
                score=70.0,
                message="Liquidity check failed"
            )
    
    def _generate_safety_report(self, signal: TradingSignal, check_results: List[SafetyCheckResult]) -> SafetyReport:
        """안전성 보고서 생성"""
        try:
            # 전체 점수 계산 (가중 평균)
            total_score = sum(result.score for result in check_results) / len(check_results)
            
            # 블로킹 체크
            blocking_results = [r for r in check_results if not r.passed]
            trade_allowed = len(blocking_results) == 0
            
            # 전체 레벨 결정
            if not trade_allowed:
                overall_level = SafetyLevel.BLOCKED
            elif total_score >= self.config.caution_threshold:
                overall_level = SafetyLevel.SAFE
            elif total_score >= self.config.warning_threshold:
                overall_level = SafetyLevel.CAUTION
            elif total_score >= self.config.danger_threshold:
                overall_level = SafetyLevel.WARNING
            else:
                overall_level = SafetyLevel.DANGER
            
            # 최종 권장사항
            if not trade_allowed:
                final_recommendation = f"Trade blocked: {', '.join([r.blocking_reason for r in blocking_results if r.blocking_reason])}"
            elif overall_level == SafetyLevel.SAFE:
                final_recommendation = "Trade approved - all safety checks passed"
            elif overall_level == SafetyLevel.CAUTION:
                final_recommendation = "Trade approved with caution - monitor closely"
            elif overall_level == SafetyLevel.WARNING:
                final_recommendation = "Trade risky - consider reducing position size"
            else:
                final_recommendation = "Trade not recommended - high risk"
            
            # 조정된 수량 계산 (필요시)
            adjusted_quantity = None
            if trade_allowed and overall_level in [SafetyLevel.WARNING, SafetyLevel.CAUTION]:
                safety_factor = total_score / 100.0
                adjusted_quantity = signal.quantity * safety_factor
            
            return SafetyReport(
                signal_id=signal.signal_id,
                overall_level=overall_level,
                overall_score=total_score,
                passed=trade_allowed,
                check_results=check_results,
                final_recommendation=final_recommendation,
                trade_allowed=trade_allowed,
                adjusted_quantity=adjusted_quantity
            )
            
        except Exception as e:
            logger.error(f"Failed to generate safety report: {e}")
            return SafetyReport(
                signal_id=signal.signal_id,
                overall_level=SafetyLevel.BLOCKED,
                overall_score=0.0,
                passed=False,
                check_results=check_results,
                final_recommendation="Failed to generate safety report",
                trade_allowed=False
            )
    
    def _update_daily_stats(self, report: SafetyReport):
        """일일 통계 업데이트"""
        try:
            current_date = datetime.now().date()
            
            # 날짜가 바뀌면 초기화
            if self.daily_stats["date"] != current_date:
                self.daily_stats = {
                    "date": current_date,
                    "total_trades": 0,
                    "blocked_trades": 0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "risk_score": 0.0
                }
            
            # 통계 업데이트
            self.daily_stats["total_trades"] += 1
            if not report.trade_allowed:
                self.daily_stats["blocked_trades"] += 1
            
            # 위험 점수 업데이트 (이동 평균)
            current_risk = 100.0 - report.overall_score
            self.daily_stats["risk_score"] = (
                self.daily_stats["risk_score"] * 0.9 + current_risk * 0.1
            )
            
        except Exception as e:
            logger.error(f"Failed to update daily stats: {e}")
    
    def update_account_state(self, account_state: AccountState):
        """계좌 상태 업데이트"""
        with self._lock:
            self.account_state = account_state
    
    def update_market_conditions(self, market_conditions: MarketConditions):
        """시장 상황 업데이트"""
        with self._lock:
            self.market_conditions = market_conditions
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """안전성 통계 반환"""
        try:
            recent_reports = self.check_history[-100:] if self.check_history else []
            
            if not recent_reports:
                return {
                    "total_checks": 0,
                    "approval_rate": 0.0,
                    "average_score": 0.0,
                    "daily_stats": self.daily_stats
                }
            
            approved_count = sum(1 for r in recent_reports if r.trade_allowed)
            approval_rate = (approved_count / len(recent_reports)) * 100
            average_score = sum(r.overall_score for r in recent_reports) / len(recent_reports)
            
            level_distribution = {}
            for level in SafetyLevel:
                count = sum(1 for r in recent_reports if r.overall_level == level)
                level_distribution[level.value] = count
            
            return {
                "total_checks": len(recent_reports),
                "approval_rate": approval_rate,
                "average_score": average_score,
                "level_distribution": level_distribution,
                "daily_stats": self.daily_stats,
                "blocked_trades_today": self.daily_stats["blocked_trades"],
                "current_risk_score": self.daily_stats["risk_score"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get safety statistics: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self._save_configuration()
            self.check_history.clear()
            logger.info("Pre-trade safety checker cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 안전성 검사기
_global_safety_checker = None

def get_safety_checker(config: Optional[SafetyConfig] = None, config_file: str = None) -> PreTradeSafetyChecker:
    """전역 안전성 검사기 반환"""
    global _global_safety_checker
    if _global_safety_checker is None:
        _global_safety_checker = PreTradeSafetyChecker(
            config=config,
            config_file=config_file or "safety_config.json"
        )
    return _global_safety_checker

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_safety_checker():
        print("🧪 Pre-Trade Safety Checker 테스트")
        
        checker = get_safety_checker(config_file="test_safety_config.json")
        
        # 계좌 상태 설정
        account_state = AccountState(
            total_balance=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
            positions={"AAPL": 10.0, "GOOGL": 5.0}
        )
        checker.update_account_state(account_state)
        
        # 시장 상황 설정
        market_conditions = MarketConditions(
            market_open=True,
            volatility_index=20.0,
            market_trend="bullish",
            trading_volume=1000000.0,
            liquidity_score=85.0
        )
        checker.update_market_conditions(market_conditions)
        
        print("\n1️⃣ 안전한 거래 신호 테스트")
        safe_signal = TradingSignal(
            symbol="AAPL",
            action=ActionType.BUY,
            quantity=5.0,
            price=150.0,
            confidence=0.8,
            signal_id="test_001",
            strategy_name="test_strategy"
        )
        
        report = await checker.perform_safety_check(safe_signal)
        print(f"  결과: {report.overall_level.value} (점수: {report.overall_score:.1f})")
        print(f"  거래 허용: {report.trade_allowed}")
        print(f"  권장사항: {report.final_recommendation}")
        
        print("\n2️⃣ 위험한 거래 신호 테스트")
        risky_signal = TradingSignal(
            symbol="UNKN",
            action=ActionType.BUY,
            quantity=100.0,  # 너무 큰 수량
            price=150.0,
            confidence=0.3,
            signal_id="test_002",
            strategy_name="test_strategy"
        )
        
        report = await checker.perform_safety_check(risky_signal)
        print(f"  결과: {report.overall_level.value} (점수: {report.overall_score:.1f})")
        print(f"  거래 허용: {report.trade_allowed}")
        print(f"  권장사항: {report.final_recommendation}")
        
        print("\n3️⃣ 안전성 통계")
        stats = checker.get_safety_statistics()
        print(f"  총 검사 횟수: {stats['total_checks']}")
        print(f"  승인률: {stats['approval_rate']:.1f}%")
        print(f"  평균 점수: {stats['average_score']:.1f}")
        
        print("\n🎉 Pre-Trade Safety Checker 테스트 완료!")
        
        # 정리
        checker.cleanup()
        
        # 테스트 파일 정리
        test_file = Path("test_safety_config.json")
        if test_file.exists():
            test_file.unlink()
    
    # 테스트 실행
    asyncio.run(test_safety_checker())