#!/usr/bin/env python3
"""
비정상 전략 감지 및 차단 시스템
P6-3: 비정상 전략 감지 및 차단 로직
"""

import sys
import os
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import warnings
from collections import defaultdict, deque

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """이상 징후 타입"""
    FREQUENCY_ANOMALY = "frequency_anomaly"          # 비정상적 빈도
    VOLUME_ANOMALY = "volume_anomaly"                # 비정상적 거래량
    PATTERN_ANOMALY = "pattern_anomaly"              # 비정상적 패턴
    PERFORMANCE_ANOMALY = "performance_anomaly"      # 성능 이상
    BEHAVIOR_ANOMALY = "behavior_anomaly"            # 행동 이상
    CORRELATION_ANOMALY = "correlation_anomaly"      # 상관관계 이상
    TIMING_ANOMALY = "timing_anomaly"                # 타이밍 이상
    RISK_ANOMALY = "risk_anomaly"                    # 리스크 이상

class SeverityLevel(Enum):
    """심각도 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(Enum):
    """액션 타입"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class ResponseAction(Enum):
    """대응 액션"""
    MONITOR = "monitor"              # 모니터링 강화
    THROTTLE = "throttle"            # 거래 제한
    BLOCK_SIGNAL = "block_signal"    # 신호 차단
    BLOCK_STRATEGY = "block_strategy" # 전략 차단
    EMERGENCY_STOP = "emergency_stop" # 응급 중단

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
class StrategyPerformance:
    """전략 성능 데이터"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyDetection:
    """이상 징후 감지 결과"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    strategy_name: str
    signal_id: Optional[str] = None
    description: str = ""
    confidence_score: float = 0.0  # 0-1
    affected_signals: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'anomaly_id': self.anomaly_id,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'strategy_name': self.strategy_name,
            'signal_id': self.signal_id,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'affected_signals': self.affected_signals,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class BlockedStrategy:
    """차단된 전략"""
    strategy_name: str
    reason: str
    blocked_at: datetime
    blocked_until: Optional[datetime] = None
    block_count: int = 1
    anomaly_types: Set[AnomalyType] = field(default_factory=set)

@dataclass
class AnomalyThresholds:
    """이상 징후 임계값"""
    # 빈도 기반
    max_signals_per_minute: int = 10
    max_signals_per_hour: int = 100
    max_signals_per_day: int = 500
    
    # 거래량 기반
    max_position_size_multiplier: float = 3.0  # 평균 대비 배수
    min_position_size_ratio: float = 0.1       # 최소 비율
    
    # 성능 기반
    min_win_rate_threshold: float = 0.3        # 최소 승률
    max_drawdown_threshold: float = 0.2        # 최대 드로우다운
    min_sharpe_ratio: float = -1.0             # 최소 샤프 비율
    
    # 패턴 기반
    max_repeated_signals: int = 5              # 동일 신호 반복
    max_correlation_threshold: float = 0.9     # 최대 상관관계
    
    # 타이밍 기반
    min_signal_interval_seconds: int = 30      # 최소 신호 간격
    max_after_hours_signals: int = 5           # 시간외 최대 신호
    
    # 신뢰도 기반
    min_confidence_threshold: float = 0.3      # 최소 신뢰도
    max_low_confidence_ratio: float = 0.7      # 낮은 신뢰도 비율

class StrategyAnomalyDetector:
    """전략 이상 징후 감지기"""
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None, config_file: str = "anomaly_config.json"):
        self.thresholds = thresholds or AnomalyThresholds()
        self.config_file = config_file
        
        # 데이터 저장소
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # strategy -> signals
        self.performance_data: Dict[str, StrategyPerformance] = {}  # strategy -> performance
        self.anomaly_history: List[AnomalyDetection] = []
        self.blocked_strategies: Dict[str, BlockedStrategy] = {}
        
        # 실시간 추적
        self.active_signals: Dict[str, List[TradingSignal]] = defaultdict(list)  # strategy -> recent signals
        self.signal_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # strategy -> pattern counts
        
        # 통계 데이터
        self.detection_stats = {
            "total_detections": 0,
            "critical_detections": 0,
            "blocked_strategies": 0,
            "false_positives": 0,
            "last_detection": None
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 설정 로드
        self._load_configuration()
        
        logger.info("Strategy anomaly detector initialized")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 임계값 로드
                threshold_config = config.get('thresholds', {})
                for key, value in threshold_config.items():
                    if hasattr(self.thresholds, key):
                        setattr(self.thresholds, key, value)
                
                # 차단된 전략 로드
                blocked_config = config.get('blocked_strategies', {})
                for strategy_name, block_data in blocked_config.items():
                    self.blocked_strategies[strategy_name] = BlockedStrategy(
                        strategy_name=strategy_name,
                        reason=block_data.get('reason', 'Unknown'),
                        blocked_at=datetime.fromisoformat(block_data['blocked_at']),
                        blocked_until=datetime.fromisoformat(block_data['blocked_until']) if block_data.get('blocked_until') else None,
                        block_count=block_data.get('block_count', 1),
                        anomaly_types=set(AnomalyType(t) for t in block_data.get('anomaly_types', []))
                    )
                
                # 통계 로드
                self.detection_stats.update(config.get('detection_stats', {}))
                
                logger.info(f"Anomaly detector configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load anomaly detector configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config = {
                'thresholds': {
                    'max_signals_per_minute': self.thresholds.max_signals_per_minute,
                    'max_signals_per_hour': self.thresholds.max_signals_per_hour,
                    'max_signals_per_day': self.thresholds.max_signals_per_day,
                    'max_position_size_multiplier': self.thresholds.max_position_size_multiplier,
                    'min_position_size_ratio': self.thresholds.min_position_size_ratio,
                    'min_win_rate_threshold': self.thresholds.min_win_rate_threshold,
                    'max_drawdown_threshold': self.thresholds.max_drawdown_threshold,
                    'min_sharpe_ratio': self.thresholds.min_sharpe_ratio,
                    'max_repeated_signals': self.thresholds.max_repeated_signals,
                    'max_correlation_threshold': self.thresholds.max_correlation_threshold,
                    'min_signal_interval_seconds': self.thresholds.min_signal_interval_seconds,
                    'max_after_hours_signals': self.thresholds.max_after_hours_signals,
                    'min_confidence_threshold': self.thresholds.min_confidence_threshold,
                    'max_low_confidence_ratio': self.thresholds.max_low_confidence_ratio
                },
                'blocked_strategies': {
                    strategy_name: {
                        'reason': block.reason,
                        'blocked_at': block.blocked_at.isoformat(),
                        'blocked_until': block.blocked_until.isoformat() if block.blocked_until else None,
                        'block_count': block.block_count,
                        'anomaly_types': [t.value for t in block.anomaly_types]
                    }
                    for strategy_name, block in self.blocked_strategies.items()
                },
                'detection_stats': self.detection_stats
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save anomaly detector configuration: {e}")
    
    async def analyze_signal(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """신호 이상 징후 분석"""
        try:
            # 차단된 전략 체크
            if self._is_strategy_blocked(signal.strategy_name):
                blocked_info = self.blocked_strategies[signal.strategy_name]
                return [AnomalyDetection(
                    anomaly_id=f"blocked_{signal.signal_id}",
                    anomaly_type=AnomalyType.BEHAVIOR_ANOMALY,
                    severity=SeverityLevel.CRITICAL,
                    strategy_name=signal.strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Strategy blocked: {blocked_info.reason}",
                    confidence_score=1.0
                )]
            
            anomalies = []
            
            # 신호 기록
            with self._lock:
                self.signal_history[signal.strategy_name].append(signal)
                self.active_signals[signal.strategy_name].append(signal)
                
                # 최근 신호만 유지 (1시간)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.active_signals[signal.strategy_name] = [
                    s for s in self.active_signals[signal.strategy_name]
                    if s.timestamp > cutoff_time
                ]
            
            # 각종 이상 징후 검사
            anomalies.extend(await self._detect_frequency_anomalies(signal))
            anomalies.extend(await self._detect_volume_anomalies(signal))
            anomalies.extend(await self._detect_pattern_anomalies(signal))
            anomalies.extend(await self._detect_timing_anomalies(signal))
            anomalies.extend(await self._detect_confidence_anomalies(signal))
            
            # 이상 징후 기록
            for anomaly in anomalies:
                with self._lock:
                    self.anomaly_history.append(anomaly)
                    if len(self.anomaly_history) > 1000:
                        self.anomaly_history.pop(0)
                
                self._update_detection_stats(anomaly)
                
                # 자동 대응 액션
                await self._execute_response_action(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Signal anomaly analysis failed: {e}")
            return []
    
    async def _detect_frequency_anomalies(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """빈도 기반 이상 징후 감지"""
        anomalies = []
        strategy_name = signal.strategy_name
        
        try:
            recent_signals = self.active_signals[strategy_name]
            now = datetime.now()
            
            # 1분간 신호 수
            minute_ago = now - timedelta(minutes=1)
            signals_per_minute = len([s for s in recent_signals if s.timestamp > minute_ago])
            
            if signals_per_minute > self.thresholds.max_signals_per_minute:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"freq_min_{signal.signal_id}",
                    anomaly_type=AnomalyType.FREQUENCY_ANOMALY,
                    severity=SeverityLevel.HIGH,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Too many signals per minute: {signals_per_minute}",
                    confidence_score=min(1.0, signals_per_minute / self.thresholds.max_signals_per_minute),
                    metrics={'signals_per_minute': signals_per_minute, 'threshold': self.thresholds.max_signals_per_minute}
                ))
            
            # 1시간간 신호 수
            hour_ago = now - timedelta(hours=1)
            signals_per_hour = len([s for s in recent_signals if s.timestamp > hour_ago])
            
            if signals_per_hour > self.thresholds.max_signals_per_hour:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"freq_hour_{signal.signal_id}",
                    anomaly_type=AnomalyType.FREQUENCY_ANOMALY,
                    severity=SeverityLevel.MEDIUM,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Too many signals per hour: {signals_per_hour}",
                    confidence_score=min(1.0, signals_per_hour / self.thresholds.max_signals_per_hour),
                    metrics={'signals_per_hour': signals_per_hour, 'threshold': self.thresholds.max_signals_per_hour}
                ))
            
        except Exception as e:
            logger.error(f"Frequency anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_volume_anomalies(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """거래량 기반 이상 징후 감지"""
        anomalies = []
        strategy_name = signal.strategy_name
        
        try:
            recent_signals = self.active_signals[strategy_name]
            
            if len(recent_signals) < 10:  # 충분한 데이터가 없으면 스킵
                return anomalies
            
            # 평균 거래량 계산
            quantities = [s.quantity for s in recent_signals[:-1]]  # 현재 신호 제외
            if not quantities:
                return anomalies
            
            avg_quantity = statistics.mean(quantities)
            std_quantity = statistics.stdev(quantities) if len(quantities) > 1 else 0
            
            # 현재 신호가 평균보다 너무 큰 경우
            if signal.quantity > avg_quantity * self.thresholds.max_position_size_multiplier:
                severity = SeverityLevel.HIGH if signal.quantity > avg_quantity * 5 else SeverityLevel.MEDIUM
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"vol_high_{signal.signal_id}",
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=severity,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Unusually large position: {signal.quantity:.2f} vs avg {avg_quantity:.2f}",
                    confidence_score=min(1.0, signal.quantity / (avg_quantity * self.thresholds.max_position_size_multiplier)),
                    metrics={
                        'current_quantity': signal.quantity,
                        'average_quantity': avg_quantity,
                        'multiplier': signal.quantity / avg_quantity if avg_quantity > 0 else 0
                    }
                ))
            
            # 현재 신호가 평균보다 너무 작은 경우
            elif signal.quantity < avg_quantity * self.thresholds.min_position_size_ratio:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"vol_low_{signal.signal_id}",
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=SeverityLevel.LOW,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Unusually small position: {signal.quantity:.2f} vs avg {avg_quantity:.2f}",
                    confidence_score=0.3,
                    metrics={
                        'current_quantity': signal.quantity,
                        'average_quantity': avg_quantity,
                        'ratio': signal.quantity / avg_quantity if avg_quantity > 0 else 0
                    }
                ))
            
        except Exception as e:
            logger.error(f"Volume anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_pattern_anomalies(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """패턴 기반 이상 징후 감지"""
        anomalies = []
        strategy_name = signal.strategy_name
        
        try:
            recent_signals = self.active_signals[strategy_name]
            
            # 동일한 신호 반복 체크
            same_signals = [
                s for s in recent_signals[-self.thresholds.max_repeated_signals:]
                if s.symbol == signal.symbol and s.action == signal.action
            ]
            
            if len(same_signals) >= self.thresholds.max_repeated_signals:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"pattern_repeat_{signal.signal_id}",
                    anomaly_type=AnomalyType.PATTERN_ANOMALY,
                    severity=SeverityLevel.MEDIUM,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Repeated pattern detected: {signal.action.value} {signal.symbol} x{len(same_signals)}",
                    confidence_score=min(1.0, len(same_signals) / self.thresholds.max_repeated_signals),
                    metrics={
                        'pattern': f"{signal.action.value}_{signal.symbol}",
                        'repeat_count': len(same_signals),
                        'threshold': self.thresholds.max_repeated_signals
                    }
                ))
            
            # 신호 패턴 다양성 체크
            if len(recent_signals) >= 20:
                unique_patterns = set(f"{s.action.value}_{s.symbol}" for s in recent_signals[-20:])
                pattern_diversity = len(unique_patterns) / 20
                
                if pattern_diversity < 0.3:  # 다양성이 낮음
                    anomalies.append(AnomalyDetection(
                        anomaly_id=f"pattern_monotone_{signal.signal_id}",
                        anomaly_type=AnomalyType.PATTERN_ANOMALY,
                        severity=SeverityLevel.LOW,
                        strategy_name=strategy_name,
                        signal_id=signal.signal_id,
                        description=f"Low pattern diversity: {pattern_diversity:.2f}",
                        confidence_score=1.0 - pattern_diversity,
                        metrics={
                            'pattern_diversity': pattern_diversity,
                            'unique_patterns': len(unique_patterns),
                            'total_signals': 20
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_timing_anomalies(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """타이밍 기반 이상 징후 감지"""
        anomalies = []
        strategy_name = signal.strategy_name
        
        try:
            recent_signals = self.active_signals[strategy_name]
            
            # 신호 간격 체크
            if len(recent_signals) >= 2:
                last_signal = recent_signals[-2]  # 현재 신호 제외한 마지막 신호
                interval = (signal.timestamp - last_signal.timestamp).total_seconds()
                
                if interval < self.thresholds.min_signal_interval_seconds:
                    anomalies.append(AnomalyDetection(
                        anomaly_id=f"timing_interval_{signal.signal_id}",
                        anomaly_type=AnomalyType.TIMING_ANOMALY,
                        severity=SeverityLevel.MEDIUM,
                        strategy_name=strategy_name,
                        signal_id=signal.signal_id,
                        description=f"Signals too close in time: {interval:.1f}s interval",
                        confidence_score=max(0.1, 1.0 - interval / self.thresholds.min_signal_interval_seconds),
                        metrics={
                            'interval_seconds': interval,
                            'threshold': self.thresholds.min_signal_interval_seconds
                        }
                    ))
            
            # 시간외 거래 체크
            current_hour = signal.timestamp.hour
            if current_hour < 9 or current_hour > 16:  # 장외 시간 (간소화)
                after_hours_count = len([
                    s for s in recent_signals
                    if s.timestamp.hour < 9 or s.timestamp.hour > 16
                ])
                
                if after_hours_count > self.thresholds.max_after_hours_signals:
                    anomalies.append(AnomalyDetection(
                        anomaly_id=f"timing_afterhours_{signal.signal_id}",
                        anomaly_type=AnomalyType.TIMING_ANOMALY,
                        severity=SeverityLevel.LOW,
                        strategy_name=strategy_name,
                        signal_id=signal.signal_id,
                        description=f"Too many after-hours signals: {after_hours_count}",
                        confidence_score=min(1.0, after_hours_count / self.thresholds.max_after_hours_signals),
                        metrics={
                            'after_hours_count': after_hours_count,
                            'threshold': self.thresholds.max_after_hours_signals,
                            'signal_hour': current_hour
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Timing anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_confidence_anomalies(self, signal: TradingSignal) -> List[AnomalyDetection]:
        """신뢰도 기반 이상 징후 감지"""
        anomalies = []
        strategy_name = signal.strategy_name
        
        try:
            # 신뢰도가 너무 낮은 경우
            if signal.confidence < self.thresholds.min_confidence_threshold:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"conf_low_{signal.signal_id}",
                    anomaly_type=AnomalyType.BEHAVIOR_ANOMALY,
                    severity=SeverityLevel.LOW,
                    strategy_name=strategy_name,
                    signal_id=signal.signal_id,
                    description=f"Low confidence signal: {signal.confidence:.3f}",
                    confidence_score=1.0 - signal.confidence,
                    metrics={
                        'confidence': signal.confidence,
                        'threshold': self.thresholds.min_confidence_threshold
                    }
                ))
            
            # 최근 신호들의 평균 신뢰도 체크
            recent_signals = self.active_signals[strategy_name]
            if len(recent_signals) >= 10:
                low_confidence_signals = [
                    s for s in recent_signals[-10:]
                    if s.confidence < self.thresholds.min_confidence_threshold
                ]
                low_confidence_ratio = len(low_confidence_signals) / 10
                
                if low_confidence_ratio > self.thresholds.max_low_confidence_ratio:
                    anomalies.append(AnomalyDetection(
                        anomaly_id=f"conf_pattern_{signal.signal_id}",
                        anomaly_type=AnomalyType.BEHAVIOR_ANOMALY,
                        severity=SeverityLevel.MEDIUM,
                        strategy_name=strategy_name,
                        signal_id=signal.signal_id,
                        description=f"High ratio of low confidence signals: {low_confidence_ratio:.2f}",
                        confidence_score=low_confidence_ratio,
                        metrics={
                            'low_confidence_ratio': low_confidence_ratio,
                            'threshold': self.thresholds.max_low_confidence_ratio,
                            'low_confidence_count': len(low_confidence_signals)
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Confidence anomaly detection failed: {e}")
        
        return anomalies
    
    async def analyze_strategy_performance(self, strategy_name: str, performance: StrategyPerformance) -> List[AnomalyDetection]:
        """전략 성능 기반 이상 징후 분석"""
        anomalies = []
        
        try:
            self.performance_data[strategy_name] = performance
            
            # 승률 체크
            if performance.win_rate < self.thresholds.min_win_rate_threshold and performance.total_trades >= 10:
                severity = SeverityLevel.HIGH if performance.win_rate < 0.2 else SeverityLevel.MEDIUM
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"perf_winrate_{strategy_name}",
                    anomaly_type=AnomalyType.PERFORMANCE_ANOMALY,
                    severity=severity,
                    strategy_name=strategy_name,
                    description=f"Low win rate: {performance.win_rate:.2f}",
                    confidence_score=1.0 - performance.win_rate,
                    metrics={
                        'win_rate': performance.win_rate,
                        'threshold': self.thresholds.min_win_rate_threshold,
                        'total_trades': performance.total_trades
                    }
                ))
            
            # 드로우다운 체크
            if performance.max_drawdown > self.thresholds.max_drawdown_threshold:
                severity = SeverityLevel.CRITICAL if performance.max_drawdown > 0.5 else SeverityLevel.HIGH
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"perf_drawdown_{strategy_name}",
                    anomaly_type=AnomalyType.PERFORMANCE_ANOMALY,
                    severity=severity,
                    strategy_name=strategy_name,
                    description=f"High drawdown: {performance.max_drawdown:.2f}",
                    confidence_score=min(1.0, performance.max_drawdown / self.thresholds.max_drawdown_threshold),
                    metrics={
                        'max_drawdown': performance.max_drawdown,
                        'threshold': self.thresholds.max_drawdown_threshold
                    }
                ))
            
            # 샤프 비율 체크
            if performance.sharpe_ratio < self.thresholds.min_sharpe_ratio and performance.total_trades >= 20:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"perf_sharpe_{strategy_name}",
                    anomaly_type=AnomalyType.PERFORMANCE_ANOMALY,
                    severity=SeverityLevel.MEDIUM,
                    strategy_name=strategy_name,
                    description=f"Low Sharpe ratio: {performance.sharpe_ratio:.2f}",
                    confidence_score=max(0.1, abs(performance.sharpe_ratio - self.thresholds.min_sharpe_ratio)),
                    metrics={
                        'sharpe_ratio': performance.sharpe_ratio,
                        'threshold': self.thresholds.min_sharpe_ratio,
                        'total_trades': performance.total_trades
                    }
                ))
            
            # 이상 징후 기록 및 처리
            for anomaly in anomalies:
                with self._lock:
                    self.anomaly_history.append(anomaly)
                
                self._update_detection_stats(anomaly)
                await self._execute_response_action(anomaly)
            
        except Exception as e:
            logger.error(f"Strategy performance analysis failed: {e}")
        
        return anomalies
    
    def _is_strategy_blocked(self, strategy_name: str) -> bool:
        """전략이 차단되었는지 확인"""
        if strategy_name not in self.blocked_strategies:
            return False
        
        block = self.blocked_strategies[strategy_name]
        
        # 차단 만료 시간 체크
        if block.blocked_until and datetime.now() > block.blocked_until:
            del self.blocked_strategies[strategy_name]
            logger.info(f"Strategy {strategy_name} unblocked (time expired)")
            return False
        
        return True
    
    async def _execute_response_action(self, anomaly: AnomalyDetection):
        """이상 징후에 대한 대응 액션 실행"""
        try:
            action = self._determine_response_action(anomaly)
            
            if action == ResponseAction.BLOCK_STRATEGY:
                self._block_strategy(
                    anomaly.strategy_name,
                    f"{anomaly.anomaly_type.value}: {anomaly.description}",
                    anomaly.severity
                )
            elif action == ResponseAction.THROTTLE:
                # 실제로는 거래 빈도 제한 구현
                logger.warning(f"Throttling strategy {anomaly.strategy_name}")
            elif action == ResponseAction.EMERGENCY_STOP:
                # 실제로는 응급 중단 구현
                logger.critical(f"Emergency stop for strategy {anomaly.strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to execute response action: {e}")
    
    def _determine_response_action(self, anomaly: AnomalyDetection) -> ResponseAction:
        """대응 액션 결정"""
        if anomaly.severity == SeverityLevel.CRITICAL:
            return ResponseAction.BLOCK_STRATEGY
        elif anomaly.severity == SeverityLevel.HIGH:
            if anomaly.anomaly_type in [AnomalyType.FREQUENCY_ANOMALY, AnomalyType.PERFORMANCE_ANOMALY]:
                return ResponseAction.THROTTLE
            else:
                return ResponseAction.BLOCK_SIGNAL
        elif anomaly.severity == SeverityLevel.MEDIUM:
            return ResponseAction.MONITOR
        else:
            return ResponseAction.MONITOR
    
    def _block_strategy(self, strategy_name: str, reason: str, severity: SeverityLevel):
        """전략 차단"""
        try:
            # 차단 기간 결정
            if severity == SeverityLevel.CRITICAL:
                block_duration = timedelta(hours=24)
            elif severity == SeverityLevel.HIGH:
                block_duration = timedelta(hours=4)
            else:
                block_duration = timedelta(hours=1)
            
            blocked_until = datetime.now() + block_duration
            
            if strategy_name in self.blocked_strategies:
                # 기존 차단 연장
                existing_block = self.blocked_strategies[strategy_name]
                existing_block.block_count += 1
                existing_block.blocked_until = blocked_until
                existing_block.reason = f"{existing_block.reason}; {reason}"
            else:
                # 새로운 차단
                self.blocked_strategies[strategy_name] = BlockedStrategy(
                    strategy_name=strategy_name,
                    reason=reason,
                    blocked_at=datetime.now(),
                    blocked_until=blocked_until
                )
                self.detection_stats["blocked_strategies"] += 1
            
            logger.warning(f"Strategy {strategy_name} blocked until {blocked_until}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to block strategy {strategy_name}: {e}")
    
    def _update_detection_stats(self, anomaly: AnomalyDetection):
        """감지 통계 업데이트"""
        try:
            with self._lock:
                self.detection_stats["total_detections"] += 1
                if anomaly.severity == SeverityLevel.CRITICAL:
                    self.detection_stats["critical_detections"] += 1
                self.detection_stats["last_detection"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Failed to update detection stats: {e}")
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """감지 요약 정보 반환"""
        try:
            recent_anomalies = [
                a for a in self.anomaly_history
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            severity_counts = {}
            for severity in SeverityLevel:
                severity_counts[severity.value] = len([
                    a for a in recent_anomalies if a.severity == severity
                ])
            
            type_counts = {}
            for anomaly_type in AnomalyType:
                type_counts[anomaly_type.value] = len([
                    a for a in recent_anomalies if a.anomaly_type == anomaly_type
                ])
            
            return {
                "total_anomalies_24h": len(recent_anomalies),
                "severity_distribution": severity_counts,
                "type_distribution": type_counts,
                "blocked_strategies": len(self.blocked_strategies),
                "blocked_strategy_list": list(self.blocked_strategies.keys()),
                "detection_stats": self.detection_stats,
                "active_strategies": len(self.signal_history),
                "last_analysis": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get detection summary: {e}")
            return {"error": str(e)}
    
    def unblock_strategy(self, strategy_name: str, reason: str = "Manual unblock"):
        """전략 차단 해제"""
        try:
            if strategy_name in self.blocked_strategies:
                del self.blocked_strategies[strategy_name]
                self.detection_stats["blocked_strategies"] = max(0, self.detection_stats["blocked_strategies"] - 1)
                logger.info(f"Strategy {strategy_name} unblocked: {reason}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unblock strategy {strategy_name}: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self._save_configuration()
            self.signal_history.clear()
            self.active_signals.clear()
            self.anomaly_history.clear()
            logger.info("Strategy anomaly detector cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 이상 징후 감지기
_global_anomaly_detector = None

def get_anomaly_detector(thresholds: Optional[AnomalyThresholds] = None, config_file: str = None) -> StrategyAnomalyDetector:
    """전역 이상 징후 감지기 반환"""
    global _global_anomaly_detector
    if _global_anomaly_detector is None:
        _global_anomaly_detector = StrategyAnomalyDetector(
            thresholds=thresholds,
            config_file=config_file or "anomaly_config.json"
        )
    return _global_anomaly_detector

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_anomaly_detector():
        print("🧪 Strategy Anomaly Detector 테스트")
        
        detector = get_anomaly_detector(config_file="test_anomaly_config.json")
        
        print("\n1️⃣ 정상 신호 테스트")
        normal_signal = TradingSignal(
            symbol="AAPL",
            action=ActionType.BUY,
            quantity=10.0,
            price=150.0,
            confidence=0.8,
            signal_id="normal_001",
            strategy_name="test_strategy"
        )
        
        anomalies = await detector.analyze_signal(normal_signal)
        print(f"  감지된 이상 징후: {len(anomalies)}개")
        
        print("\n2️⃣ 이상 신호 테스트 (빈도)")
        for i in range(15):  # 임계값 초과
            frequent_signal = TradingSignal(
                symbol="AAPL",
                action=ActionType.BUY,
                quantity=10.0,
                confidence=0.8,
                signal_id=f"freq_{i:03d}",
                strategy_name="frequent_strategy"
            )
            anomalies = await detector.analyze_signal(frequent_signal)
            
        print(f"  감지된 이상 징후: {len(anomalies)}개")
        for anomaly in anomalies:
            print(f"    - {anomaly.anomaly_type.value}: {anomaly.description}")
        
        print("\n3️⃣ 이상 신호 테스트 (거래량)")
        volume_signal = TradingSignal(
            symbol="AAPL",
            action=ActionType.BUY,
            quantity=1000.0,  # 매우 큰 수량
            confidence=0.8,
            signal_id="volume_001",
            strategy_name="test_strategy"
        )
        
        anomalies = await detector.analyze_signal(volume_signal)
        print(f"  감지된 이상 징후: {len(anomalies)}개")
        for anomaly in anomalies:
            print(f"    - {anomaly.anomaly_type.value}: {anomaly.description}")
        
        print("\n4️⃣ 성능 기반 이상 징후 테스트")
        poor_performance = StrategyPerformance(
            strategy_name="poor_strategy",
            total_trades=50,
            winning_trades=10,
            losing_trades=40,
            win_rate=0.2,  # 낮은 승률
            max_drawdown=0.3,  # 높은 드로우다운
            sharpe_ratio=-0.5
        )
        
        perf_anomalies = await detector.analyze_strategy_performance("poor_strategy", poor_performance)
        print(f"  성능 이상 징후: {len(perf_anomalies)}개")
        for anomaly in perf_anomalies:
            print(f"    - {anomaly.anomaly_type.value}: {anomaly.description}")
        
        print("\n5️⃣ 감지 요약")
        summary = detector.get_detection_summary()
        print(f"  24시간 이상 징후: {summary['total_anomalies_24h']}개")
        print(f"  차단된 전략: {summary['blocked_strategies']}개")
        print(f"  활성 전략: {summary['active_strategies']}개")
        
        print("\n🎉 Strategy Anomaly Detector 테스트 완료!")
        
        # 정리
        detector.cleanup()
        
        # 테스트 파일 정리
        test_file = Path("test_anomaly_config.json")
        if test_file.exists():
            test_file.unlink()
    
    # 테스트 실행
    asyncio.run(test_anomaly_detector())