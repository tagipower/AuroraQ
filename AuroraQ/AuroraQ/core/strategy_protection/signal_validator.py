#!/usr/bin/env python3
"""
전략 신호 검증 시스템
P6-1: 전략 신호 검증 시스템 설계
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
import threading
import warnings
from collections import deque, defaultdict

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """검증 결과"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    BLOCKED = "blocked"

class SignalType(Enum):
    """신호 타입"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class RiskLevel(Enum):
    """위험 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TradingSignal:
    """거래 신호"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    strategy_name: str
    price: Optional[float] = None
    volume: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'price': self.price,
            'volume': self.volume,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }

@dataclass
class ValidationRule:
    """검증 규칙"""
    rule_id: str
    name: str
    description: str
    enabled: bool = True
    priority: int = 1  # 1(높음) ~ 10(낮음)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationError:
    """검증 오류"""
    rule_id: str
    error_code: str
    message: str
    severity: RiskLevel
    signal_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'rule_id': self.rule_id,
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'signal_data': self.signal_data
        }

@dataclass
class ValidationReport:
    """검증 보고서"""
    signal_id: str
    timestamp: datetime
    result: ValidationResult
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    execution_time_ms: float = 0.0
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'result': self.result.value,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'warnings': [w.to_dict() for w in self.warnings],
            'errors': [e.to_dict() for e in self.errors],
            'execution_time_ms': self.execution_time_ms,
            'recommendation': self.recommendation
        }

class SignalValidator:
    """전략 신호 검증기"""
    
    def __init__(self, config_file: str = "signal_validator_config.json"):
        self.config_file = config_file
        
        # 검증 규칙
        self.validation_rules: Dict[str, ValidationRule] = {}
        self._setup_default_rules()
        
        # 신호 히스토리
        self.signal_history: deque = deque(maxlen=1000)
        self.validation_history: List[ValidationReport] = []
        
        # 실시간 통계
        self.stats = {
            "total_signals": 0,
            "passed_signals": 0,
            "warning_signals": 0,
            "failed_signals": 0,
            "blocked_signals": 0,
            "avg_validation_time_ms": 0.0,
            "last_validation": None
        }
        
        # 설정 및 제어
        self._lock = threading.RLock()
        self.enabled = True
        
        # 설정 로드
        self._load_configuration()
        
        logger.info("Signal validator initialized")
    
    def _setup_default_rules(self):
        """기본 검증 규칙 설정"""
        
        # 1. 기본 데이터 유효성 검사
        self.validation_rules["basic_data"] = ValidationRule(
            rule_id="basic_data",
            name="기본 데이터 유효성",
            description="신호의 기본 데이터가 유효한지 검사",
            priority=1,
            risk_level=RiskLevel.CRITICAL,
            parameters={
                "check_timestamp": True,
                "check_symbol": True,
                "check_signal_type": True,
                "check_strength_range": True,
                "check_confidence_range": True
            }
        )
        
        # 2. 신호 강도 검증
        self.validation_rules["signal_strength"] = ValidationRule(
            rule_id="signal_strength",
            name="신호 강도 검증",
            description="신호 강도가 최소 임계값을 넘는지 검사",
            priority=2,
            risk_level=RiskLevel.HIGH,
            parameters={
                "min_strength": 0.3,
                "min_confidence": 0.5,
                "require_both": True
            }
        )
        
        # 3. 가격 변동성 검사
        self.validation_rules["price_volatility"] = ValidationRule(
            rule_id="price_volatility",
            name="가격 변동성 검사",
            description="과도한 가격 변동 중 신호 발생 검사",
            priority=3,
            risk_level=RiskLevel.HIGH,
            parameters={
                "max_price_change_1h": 0.05,  # 1시간 내 5% 이상 변동
                "max_price_change_24h": 0.15,  # 24시간 내 15% 이상 변동
                "volatility_threshold": 0.1
            }
        )
        
        # 4. 신호 빈도 제한
        self.validation_rules["signal_frequency"] = ValidationRule(
            rule_id="signal_frequency",
            name="신호 빈도 제한",
            description="같은 심볼에 대한 신호 빈도 제한",
            priority=4,
            risk_level=RiskLevel.MEDIUM,
            parameters={
                "max_signals_per_hour": 10,
                "max_signals_per_day": 50,
                "min_interval_minutes": 5
            }
        )
        
        # 5. 포지션 충돌 검사
        self.validation_rules["position_conflict"] = ValidationRule(
            rule_id="position_conflict",
            name="포지션 충돌 검사",
            description="기존 포지션과 충돌하는 신호인지 검사",
            priority=5,
            risk_level=RiskLevel.HIGH,
            parameters={
                "check_opposite_signals": True,
                "check_position_size": True,
                "max_position_ratio": 0.3
            }
        )
        
        # 6. 시장 시간 검사
        self.validation_rules["market_hours"] = ValidationRule(
            rule_id="market_hours",
            name="시장 시간 검사",
            description="거래 가능한 시간인지 검사",
            priority=6,
            risk_level=RiskLevel.MEDIUM,
            parameters={
                "check_market_hours": True,
                "allow_weekend": False,
                "allow_holiday": False,
                "crypto_24h": True
            }
        )
        
        # 7. 전략 일관성 검사
        self.validation_rules["strategy_consistency"] = ValidationRule(
            rule_id="strategy_consistency",
            name="전략 일관성 검사",
            description="전략의 최근 신호와 일관성 있는지 검사",
            priority=7,
            risk_level=RiskLevel.MEDIUM,
            parameters={
                "check_recent_signals": 10,
                "max_contradiction_ratio": 0.3,
                "lookback_hours": 24
            }
        )
        
        # 8. 리스크 매니지먼트
        self.validation_rules["risk_management"] = ValidationRule(
            rule_id="risk_management",
            name="리스크 매니지먼트",
            description="리스크 관리 파라미터 검증",
            priority=8,
            risk_level=RiskLevel.HIGH,
            parameters={
                "require_stop_loss": True,
                "max_risk_per_trade": 0.02,  # 거래당 최대 2% 리스크
                "min_risk_reward_ratio": 1.5
            }
        )
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 규칙 설정 로드
                for rule_id, rule_config in config.get('rules', {}).items():
                    if rule_id in self.validation_rules:
                        rule = self.validation_rules[rule_id]
                        rule.enabled = rule_config.get('enabled', rule.enabled)
                        rule.priority = rule_config.get('priority', rule.priority)
                        rule.parameters.update(rule_config.get('parameters', {}))
                
                # 통계 로드
                self.stats.update(config.get('stats', {}))
                
                logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config = {
                'rules': {
                    rule_id: {
                        'enabled': rule.enabled,
                        'priority': rule.priority,
                        'parameters': rule.parameters
                    }
                    for rule_id, rule in self.validation_rules.items()
                },
                'stats': self.stats
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def validate_signal(self, signal: TradingSignal) -> ValidationReport:
        """신호 검증"""
        start_time = time.time()
        
        report = ValidationReport(
            signal_id=signal.signal_id,
            timestamp=datetime.now(),
            result=ValidationResult.PASS
        )
        
        try:
            if not self.enabled:
                report.result = ValidationResult.PASS
                report.recommendation = "Validator disabled - signal passed"
                return report
            
            # 신호 히스토리에 추가
            with self._lock:
                self.signal_history.append(signal)
            
            # 우선순위 순서로 규칙 정렬
            sorted_rules = sorted(
                [(rule_id, rule) for rule_id, rule in self.validation_rules.items() if rule.enabled],
                key=lambda x: x[1].priority
            )
            
            # 각 규칙에 대해 검증 수행
            for rule_id, rule in sorted_rules:
                try:
                    rule_result = await self._execute_validation_rule(signal, rule)
                    
                    if rule_result['passed']:
                        report.passed_rules.append(rule_id)
                    else:
                        report.failed_rules.append(rule_id)
                        
                        error = ValidationError(
                            rule_id=rule_id,
                            error_code=rule_result.get('error_code', 'VALIDATION_FAILED'),
                            message=rule_result.get('message', f'Rule {rule_id} failed'),
                            severity=rule.risk_level,
                            signal_data={'signal_id': signal.signal_id, 'rule': rule_id}
                        )
                        
                        if rule.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                            report.errors.append(error)
                        else:
                            report.warnings.append(error)
                        
                        # 중요한 규칙 실패 시 즉시 차단
                        if rule.risk_level == RiskLevel.CRITICAL:
                            report.result = ValidationResult.BLOCKED
                            report.recommendation = f"Signal blocked due to critical rule failure: {rule_id}"
                            break
                        elif rule.risk_level == RiskLevel.HIGH:
                            if report.result == ValidationResult.PASS:
                                report.result = ValidationResult.FAIL
                            
                except Exception as e:
                    logger.error(f"Error executing validation rule {rule_id}: {e}")
                    error = ValidationError(
                        rule_id=rule_id,
                        error_code='RULE_EXECUTION_ERROR',
                        message=f"Failed to execute rule {rule_id}: {str(e)}",
                        severity=RiskLevel.MEDIUM,
                        signal_data={'signal_id': signal.signal_id}
                    )
                    report.warnings.append(error)
            
            # 최종 결과 결정
            if report.result == ValidationResult.PASS:
                if report.warnings:
                    report.result = ValidationResult.WARNING
                    report.recommendation = f"Signal passed with {len(report.warnings)} warnings"
                else:
                    report.recommendation = "Signal passed all validations"
            elif report.result == ValidationResult.FAIL:
                report.recommendation = f"Signal failed {len(report.errors)} critical validations"
            
            # 실행 시간 기록
            report.execution_time_ms = (time.time() - start_time) * 1000
            
            # 통계 업데이트
            with self._lock:
                self.stats["total_signals"] += 1
                
                if report.result == ValidationResult.PASS:
                    self.stats["passed_signals"] += 1
                elif report.result == ValidationResult.WARNING:
                    self.stats["warning_signals"] += 1
                elif report.result == ValidationResult.FAIL:
                    self.stats["failed_signals"] += 1
                elif report.result == ValidationResult.BLOCKED:
                    self.stats["blocked_signals"] += 1
                
                # 평균 검증 시간 업데이트
                old_avg = self.stats["avg_validation_time_ms"]
                total = self.stats["total_signals"]
                self.stats["avg_validation_time_ms"] = (old_avg * (total - 1) + report.execution_time_ms) / total
                
                self.stats["last_validation"] = datetime.now().isoformat()
                
                # 검증 히스토리에 추가
                self.validation_history.append(report)
                if len(self.validation_history) > 500:
                    self.validation_history.pop(0)
            
            logger.info(f"Signal validation completed: {signal.signal_id} -> {report.result.value}")
            
        except Exception as e:
            logger.error(f"Failed to validate signal {signal.signal_id}: {e}")
            report.result = ValidationResult.FAIL
            report.recommendation = f"Validation failed due to system error: {str(e)}"
            
            error = ValidationError(
                rule_id="system",
                error_code="SYSTEM_ERROR",
                message=str(e),
                severity=RiskLevel.CRITICAL,
                signal_data={'signal_id': signal.signal_id}
            )
            report.errors.append(error)
        
        return report
    
    async def _execute_validation_rule(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """개별 검증 규칙 실행"""
        
        if rule.rule_id == "basic_data":
            return await self._validate_basic_data(signal, rule)
        elif rule.rule_id == "signal_strength":
            return await self._validate_signal_strength(signal, rule)
        elif rule.rule_id == "price_volatility":
            return await self._validate_price_volatility(signal, rule)
        elif rule.rule_id == "signal_frequency":
            return await self._validate_signal_frequency(signal, rule)
        elif rule.rule_id == "position_conflict":
            return await self._validate_position_conflict(signal, rule)
        elif rule.rule_id == "market_hours":
            return await self._validate_market_hours(signal, rule)
        elif rule.rule_id == "strategy_consistency":
            return await self._validate_strategy_consistency(signal, rule)
        elif rule.rule_id == "risk_management":
            return await self._validate_risk_management(signal, rule)
        else:
            return {'passed': True, 'message': f'Unknown rule: {rule.rule_id}'}
    
    async def _validate_basic_data(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """기본 데이터 유효성 검사"""
        
        # 타임스탬프 검사
        if rule.parameters.get("check_timestamp", True):
            if not signal.timestamp or signal.timestamp > datetime.now():
                return {
                    'passed': False,
                    'error_code': 'INVALID_TIMESTAMP',
                    'message': 'Invalid or future timestamp'
                }
        
        # 심볼 검사
        if rule.parameters.get("check_symbol", True):
            if not signal.symbol or len(signal.symbol) < 3:
                return {
                    'passed': False,
                    'error_code': 'INVALID_SYMBOL',
                    'message': 'Invalid trading symbol'
                }
        
        # 신호 타입 검사
        if rule.parameters.get("check_signal_type", True):
            if signal.signal_type not in SignalType:
                return {
                    'passed': False,
                    'error_code': 'INVALID_SIGNAL_TYPE',
                    'message': 'Invalid signal type'
                }
        
        # 강도 범위 검사
        if rule.parameters.get("check_strength_range", True):
            if not (0.0 <= signal.strength <= 1.0):
                return {
                    'passed': False,
                    'error_code': 'INVALID_STRENGTH_RANGE',
                    'message': f'Signal strength out of range: {signal.strength}'
                }
        
        # 신뢰도 범위 검사
        if rule.parameters.get("check_confidence_range", True):
            if not (0.0 <= signal.confidence <= 1.0):
                return {
                    'passed': False,
                    'error_code': 'INVALID_CONFIDENCE_RANGE',
                    'message': f'Signal confidence out of range: {signal.confidence}'
                }
        
        return {'passed': True, 'message': 'Basic data validation passed'}
    
    async def _validate_signal_strength(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """신호 강도 검증"""
        
        min_strength = rule.parameters.get("min_strength", 0.3)
        min_confidence = rule.parameters.get("min_confidence", 0.5)
        require_both = rule.parameters.get("require_both", True)
        
        strength_ok = signal.strength >= min_strength
        confidence_ok = signal.confidence >= min_confidence
        
        if require_both:
            if not (strength_ok and confidence_ok):
                return {
                    'passed': False,
                    'error_code': 'INSUFFICIENT_SIGNAL_STRENGTH',
                    'message': f'Signal strength ({signal.strength:.3f}) or confidence ({signal.confidence:.3f}) below threshold'
                }
        else:
            if not (strength_ok or confidence_ok):
                return {
                    'passed': False,
                    'error_code': 'INSUFFICIENT_SIGNAL_STRENGTH',
                    'message': f'Both signal strength ({signal.strength:.3f}) and confidence ({signal.confidence:.3f}) below threshold'
                }
        
        return {'passed': True, 'message': 'Signal strength validation passed'}
    
    async def _validate_price_volatility(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """가격 변동성 검사"""
        
        # 실제 구현에서는 시장 데이터에서 변동성 확인
        # 여기서는 시뮬레이션
        
        max_change_1h = rule.parameters.get("max_price_change_1h", 0.05)
        max_change_24h = rule.parameters.get("max_price_change_24h", 0.15)
        
        # 시뮬레이션된 변동성 (실제로는 시장 데이터에서 가져옴)
        simulated_change_1h = abs(np.random.normal(0, 0.02))
        simulated_change_24h = abs(np.random.normal(0, 0.08))
        
        if simulated_change_1h > max_change_1h:
            return {
                'passed': False,
                'error_code': 'HIGH_VOLATILITY_1H',
                'message': f'High volatility detected in 1h: {simulated_change_1h:.3f}'
            }
        
        if simulated_change_24h > max_change_24h:
            return {
                'passed': False,
                'error_code': 'HIGH_VOLATILITY_24H',
                'message': f'High volatility detected in 24h: {simulated_change_24h:.3f}'
            }
        
        return {'passed': True, 'message': 'Price volatility validation passed'}
    
    async def _validate_signal_frequency(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """신호 빈도 제한 검사"""
        
        max_per_hour = rule.parameters.get("max_signals_per_hour", 10)
        max_per_day = rule.parameters.get("max_signals_per_day", 50)
        min_interval = rule.parameters.get("min_interval_minutes", 5)
        
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        # 같은 심볼의 최근 신호들 확인
        recent_signals = [
            s for s in self.signal_history 
            if s.symbol == signal.symbol and s.timestamp > one_day_ago
        ]
        
        # 1시간 내 신호 수 확인
        signals_1h = [s for s in recent_signals if s.timestamp > one_hour_ago]
        if len(signals_1h) >= max_per_hour:
            return {
                'passed': False,
                'error_code': 'FREQUENCY_LIMIT_1H',
                'message': f'Too many signals in 1h for {signal.symbol}: {len(signals_1h)}/{max_per_hour}'
            }
        
        # 24시간 내 신호 수 확인
        if len(recent_signals) >= max_per_day:
            return {
                'passed': False,
                'error_code': 'FREQUENCY_LIMIT_24H',
                'message': f'Too many signals in 24h for {signal.symbol}: {len(recent_signals)}/{max_per_day}'
            }
        
        # 최소 간격 확인
        if recent_signals:
            latest_signal = max(recent_signals, key=lambda x: x.timestamp)
            time_diff = (signal.timestamp - latest_signal.timestamp).total_seconds() / 60
            
            if time_diff < min_interval:
                return {
                    'passed': False,
                    'error_code': 'MINIMUM_INTERVAL',
                    'message': f'Signal too soon after previous: {time_diff:.1f}min < {min_interval}min'
                }
        
        return {'passed': True, 'message': 'Signal frequency validation passed'}
    
    async def _validate_position_conflict(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """포지션 충돌 검사"""
        
        # 실제 구현에서는 포지션 매니저와 연동
        # 여기서는 기본 로직만 구현
        
        check_opposite = rule.parameters.get("check_opposite_signals", True)
        
        if check_opposite:
            # 최근 반대 신호가 있는지 확인
            recent_signals = [
                s for s in self.signal_history 
                if s.symbol == signal.symbol and 
                s.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            opposite_signals = []
            if signal.signal_type == SignalType.BUY:
                opposite_signals = [s for s in recent_signals if s.signal_type == SignalType.SELL]
            elif signal.signal_type == SignalType.SELL:
                opposite_signals = [s for s in recent_signals if s.signal_type == SignalType.BUY]
            
            if opposite_signals:
                return {
                    'passed': False,
                    'error_code': 'OPPOSITE_SIGNAL_CONFLICT',
                    'message': f'Opposite signal detected within 1h for {signal.symbol}'
                }
        
        return {'passed': True, 'message': 'Position conflict validation passed'}
    
    async def _validate_market_hours(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """시장 시간 검사"""
        
        crypto_24h = rule.parameters.get("crypto_24h", True)
        allow_weekend = rule.parameters.get("allow_weekend", False)
        
        # 암호화폐는 24시간 거래
        if crypto_24h and any(crypto in signal.symbol.upper() for crypto in ['BTC', 'ETH', 'USDT']):
            return {'passed': True, 'message': 'Crypto market 24h trading allowed'}
        
        # 주말 거래 확인
        if not allow_weekend:
            weekday = signal.timestamp.weekday()
            if weekday >= 5:  # 토요일(5), 일요일(6)
                return {
                    'passed': False,
                    'error_code': 'WEEKEND_TRADING_DISABLED',
                    'message': 'Weekend trading not allowed'
                }
        
        return {'passed': True, 'message': 'Market hours validation passed'}
    
    async def _validate_strategy_consistency(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """전략 일관성 검사"""
        
        check_recent = rule.parameters.get("check_recent_signals", 10)
        max_contradiction = rule.parameters.get("max_contradiction_ratio", 0.3)
        lookback_hours = rule.parameters.get("lookback_hours", 24)
        
        cutoff_time = signal.timestamp - timedelta(hours=lookback_hours)
        
        # 같은 전략의 최근 신호들 확인
        strategy_signals = [
            s for s in self.signal_history 
            if s.strategy_name == signal.strategy_name and 
            s.symbol == signal.symbol and
            s.timestamp > cutoff_time
        ][-check_recent:]
        
        if len(strategy_signals) < 3:
            return {'passed': True, 'message': 'Insufficient historical signals for consistency check'}
        
        # 모순되는 신호 비율 계산
        contradictory_count = 0
        for prev_signal in strategy_signals:
            if signal.signal_type == SignalType.BUY and prev_signal.signal_type == SignalType.SELL:
                contradictory_count += 1
            elif signal.signal_type == SignalType.SELL and prev_signal.signal_type == SignalType.BUY:
                contradictory_count += 1
        
        contradiction_ratio = contradictory_count / len(strategy_signals)
        
        if contradiction_ratio > max_contradiction:
            return {
                'passed': False,
                'error_code': 'STRATEGY_INCONSISTENCY',
                'message': f'High contradiction ratio: {contradiction_ratio:.2f} > {max_contradiction}'
            }
        
        return {'passed': True, 'message': 'Strategy consistency validation passed'}
    
    async def _validate_risk_management(self, signal: TradingSignal, rule: ValidationRule) -> Dict[str, Any]:
        """리스크 매니지먼트 검증"""
        
        require_stop_loss = rule.parameters.get("require_stop_loss", True)
        max_risk_per_trade = rule.parameters.get("max_risk_per_trade", 0.02)
        min_risk_reward = rule.parameters.get("min_risk_reward_ratio", 1.5)
        
        # 스탑로스 확인
        if require_stop_loss and not signal.stop_loss:
            return {
                'passed': False,
                'error_code': 'MISSING_STOP_LOSS',
                'message': 'Stop loss is required but not provided'
            }
        
        # 리스크 계산 (가격 정보가 있는 경우)
        if signal.price and signal.stop_loss:
            if signal.signal_type == SignalType.BUY:
                risk_ratio = (signal.price - signal.stop_loss) / signal.price
            elif signal.signal_type == SignalType.SELL:
                risk_ratio = (signal.stop_loss - signal.price) / signal.price
            else:
                risk_ratio = 0.0
            
            if risk_ratio > max_risk_per_trade:
                return {
                    'passed': False,
                    'error_code': 'EXCESSIVE_RISK',
                    'message': f'Risk per trade too high: {risk_ratio:.3f} > {max_risk_per_trade}'
                }
            
            # 리스크-리워드 비율 확인
            if signal.take_profit:
                if signal.signal_type == SignalType.BUY:
                    reward_ratio = (signal.take_profit - signal.price) / signal.price
                elif signal.signal_type == SignalType.SELL:
                    reward_ratio = (signal.price - signal.take_profit) / signal.price
                else:
                    reward_ratio = 0.0
                
                if risk_ratio > 0:
                    risk_reward_ratio = reward_ratio / risk_ratio
                    if risk_reward_ratio < min_risk_reward:
                        return {
                            'passed': False,
                            'error_code': 'POOR_RISK_REWARD_RATIO',
                            'message': f'Risk-reward ratio too low: {risk_reward_ratio:.2f} < {min_risk_reward}'
                        }
        
        return {'passed': True, 'message': 'Risk management validation passed'}
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        try:
            with self._lock:
                total = self.stats["total_signals"]
                
                if total == 0:
                    return {
                        'total_signals': 0,
                        'validation_rates': {
                            'pass_rate': 0.0,
                            'warning_rate': 0.0,
                            'fail_rate': 0.0,
                            'block_rate': 0.0
                        },
                        'performance': {
                            'avg_validation_time_ms': 0.0,
                            'total_rules': len(self.validation_rules),
                            'enabled_rules': len([r for r in self.validation_rules.values() if r.enabled])
                        },
                        'recent_activity': {
                            'last_validation': self.stats.get('last_validation'),
                            'recent_signals_count': len(self.signal_history),
                            'recent_validations_count': len(self.validation_history)
                        }
                    }
                
                return {
                    'total_signals': total,
                    'validation_rates': {
                        'pass_rate': self.stats["passed_signals"] / total,
                        'warning_rate': self.stats["warning_signals"] / total,
                        'fail_rate': self.stats["failed_signals"] / total,
                        'block_rate': self.stats["blocked_signals"] / total
                    },
                    'performance': {
                        'avg_validation_time_ms': self.stats["avg_validation_time_ms"],
                        'total_rules': len(self.validation_rules),
                        'enabled_rules': len([r for r in self.validation_rules.values() if r.enabled])
                    },
                    'recent_activity': {
                        'last_validation': self.stats.get('last_validation'),
                        'recent_signals_count': len(self.signal_history),
                        'recent_validations_count': len(self.validation_history)
                    },
                    'rule_performance': self._get_rule_performance_stats()
                }
                
        except Exception as e:
            logger.error(f"Failed to get validation statistics: {e}")
            return {'error': str(e)}
    
    def _get_rule_performance_stats(self) -> Dict[str, Any]:
        """규칙별 성능 통계"""
        rule_stats = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0})
        
        for report in self.validation_history[-100:]:  # 최근 100개 검증 결과
            for rule_id in report.passed_rules:
                rule_stats[rule_id]['passed'] += 1
                rule_stats[rule_id]['total'] += 1
            
            for rule_id in report.failed_rules:
                rule_stats[rule_id]['failed'] += 1
                rule_stats[rule_id]['total'] += 1
        
        # 각 규칙의 통과율 계산
        performance = {}
        for rule_id, stats in rule_stats.items():
            if stats['total'] > 0:
                performance[rule_id] = {
                    'pass_rate': stats['passed'] / stats['total'],
                    'fail_rate': stats['failed'] / stats['total'],
                    'total_executions': stats['total']
                }
        
        return performance
    
    def enable_rule(self, rule_id: str) -> bool:
        """규칙 활성화"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = True
            self._save_configuration()
            logger.info(f"Validation rule enabled: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """규칙 비활성화"""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = False
            self._save_configuration()
            logger.info(f"Validation rule disabled: {rule_id}")
            return True
        return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self._save_configuration()
            
            with self._lock:
                self.signal_history.clear()
                self.validation_history.clear()
            
            logger.info("Signal validator cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 신호 검증기
_global_signal_validator = None

def get_signal_validator(config_file: str = None) -> SignalValidator:
    """전역 신호 검증기 반환"""
    global _global_signal_validator
    if _global_signal_validator is None:
        _global_signal_validator = SignalValidator(
            config_file or "signal_validator_config.json"
        )
    return _global_signal_validator

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    import random
    
    async def test_signal_validator():
        print("🧪 Signal Validator 테스트")
        
        validator = get_signal_validator("test_signal_validator.json")
        
        print("\n1️⃣ 유효한 신호 검증")
        valid_signal = TradingSignal(
            signal_id="test_001",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.7,
            strategy_name="PPO_Strategy",
            price=45000.0,
            stop_loss=44000.0,
            take_profit=47000.0
        )
        
        report = await validator.validate_signal(valid_signal)
        print(f"  결과: {report.result.value}")
        print(f"  통과 규칙: {len(report.passed_rules)}개")
        print(f"  실패 규칙: {len(report.failed_rules)}개")
        print(f"  경고: {len(report.warnings)}개")
        print(f"  검증 시간: {report.execution_time_ms:.1f}ms")
        
        print("\n2️⃣ 무효한 신호 검증")
        invalid_signal = TradingSignal(
            signal_id="test_002",
            timestamp=datetime.now(),
            symbol="X",  # 잘못된 심볼
            signal_type=SignalType.BUY,
            strength=0.1,  # 낮은 강도
            confidence=0.2,  # 낮은 신뢰도
            strategy_name="Test_Strategy"
        )
        
        report = await validator.validate_signal(invalid_signal)
        print(f"  결과: {report.result.value}")
        print(f"  실패 규칙: {len(report.failed_rules)}개")
        print(f"  오류: {len(report.errors)}개")
        if report.errors:
            print(f"  첫 번째 오류: {report.errors[0].message}")
        
        print("\n3️⃣ 빈도 제한 테스트")
        for i in range(5):
            freq_signal = TradingSignal(
                signal_id=f"freq_test_{i}",
                timestamp=datetime.now(),
                symbol="ETHUSDT",
                signal_type=SignalType.BUY,
                strength=0.6,
                confidence=0.6,
                strategy_name="Frequency_Test"
            )
            
            report = await validator.validate_signal(freq_signal)
            print(f"    신호 {i+1}: {report.result.value}")
        
        print("\n4️⃣ 검증 통계")
        stats = validator.get_validation_statistics()
        print(f"  총 신호: {stats['total_signals']}개")
        print(f"  통과율: {stats['validation_rates']['pass_rate']*100:.1f}%")
        print(f"  경고율: {stats['validation_rates']['warning_rate']*100:.1f}%")
        print(f"  실패율: {stats['validation_rates']['fail_rate']*100:.1f}%")
        print(f"  평균 검증 시간: {stats['performance']['avg_validation_time_ms']:.1f}ms")
        
        print("\n🎉 Signal Validator 테스트 완료!")
        
        # 정리
        validator.cleanup()
        
        # 테스트 파일 정리
        import os
        if os.path.exists("test_signal_validator.json"):
            os.remove("test_signal_validator.json")
    
    # 테스트 실행
    asyncio.run(test_signal_validator())