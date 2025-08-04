#!/usr/bin/env python3
"""
Enhanced VPS 전략 어댑터 (통합 점수 시스템)
모든 Rule 전략을 로드하고 점수 기반으로 최적 신호를 선택하는 통합 어댑터
"""

import os
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import importlib.util
from collections import defaultdict

# VPS 통합 로깅 시스템 - 절대 import 사용
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
except ImportError:
    # 로깅 시스템이 없는 경우 기본 로깅 사용
    import logging
    def get_vps_log_integrator():
        return logging.getLogger(__name__)
    LogCategory = type('LogCategory', (), {'TRADING': 'trading'})
    LogLevel = type('LogLevel', (), {'INFO': 'info', 'ERROR': 'error'})

# 통일된 신호 인터페이스 - 절대 import 사용
try:
    from trading.unified_signal_interface import (
        UnifiedSignalConverter, StandardSignal, SignalValidator,
        create_unified_signal_converter
    )
except ImportError:
    # 기본 신호 인터페이스 구현
    from dataclasses import dataclass
    from typing import Any, Dict
    
    @dataclass
    class StandardSignal:
        action: str
        score: float
        timestamp: str
        metadata: Dict[str, Any] = None
    
    class UnifiedSignalConverter:
        @staticmethod
        def convert(signal: Any) -> StandardSignal:
            return StandardSignal('HOLD', 0.5, datetime.now().isoformat())
    
    class SignalValidator:
        @staticmethod
        def validate(signal: Any) -> bool:
            return True
    
    def create_unified_signal_converter():
        return UnifiedSignalConverter()

# 선물 리스크 관리 시스템 - 절대 import 사용
try:
    from trading.futures_risk_coordinator import (
        FuturesRiskCoordinator, FuturesRiskConfig, SystemRiskState,
        SystemRiskLevel, MarketMetrics, PositionMetrics
    )
except ImportError:
    # 기본 리스크 관리 클래스 구현
    from enum import Enum
    
    class SystemRiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    @dataclass
    class SystemRiskState:
        level: SystemRiskLevel = SystemRiskLevel.LOW
        margin_ratio: float = 0.5
        position_risk: float = 0.1
    
    @dataclass
    class FuturesRiskConfig:
        max_leverage: float = 10.0
        position_limit: float = 0.1
    
    @dataclass 
    class MarketMetrics:
        volatility: float = 0.02
        trend_strength: float = 0.5
    
    @dataclass
    class PositionMetrics:
        size: float = 0.0
        pnl: float = 0.0
    
    class FuturesRiskCoordinator:
        def __init__(self, config=None):
            self.config = config or FuturesRiskConfig()
        
        def assess_risk(self, **kwargs):
            return SystemRiskState()

try:
    from trading.futures_leverage_manager import FuturesLeverageConfig, MarketCondition
except ImportError:
    from enum import Enum
    
    class MarketCondition(Enum):
        STABLE = "stable"
        VOLATILE = "volatile"
        TRENDING = "trending"
    
    @dataclass
    class FuturesLeverageConfig:
        default_leverage: float = 3.0
        max_leverage: float = 10.0

try:
    from trading.futures_margin_manager import MarginConfig
except ImportError:
    @dataclass
    class MarginConfig:
        maintenance_margin: float = 0.05
        initial_margin: float = 0.1

@dataclass
class StrategyScore:
    """전략 점수 정보"""
    composite_score: float
    confidence: float
    detail_scores: Dict[str, float]
    timestamp: datetime
    strategy_name: str

@dataclass 
class StrategyPerformance:
    """전략 성과 정보"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    recent_pnls: List[float] = field(default_factory=list)
    
    def update_trade(self, pnl: float):
        """거래 결과 업데이트"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        # Recent PnL 관리 (최근 20거래)
        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > 20:
            self.recent_pnls.pop(0)
            
        # 주요 지표 계산
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Profit Factor 계산 (총 수익 / 총 손실)
        if self.total_loss > 0:
            self.profit_factor = self.total_profit / self.total_loss
        else:
            self.profit_factor = float('inf') if self.total_profit > 0 else 0.0
        
        # 평균 수익/손실 계산
        if self.winning_trades > 0:
            self.avg_win = self.total_profit / self.winning_trades
        if self.losing_trades > 0:
            self.avg_loss = self.total_loss / self.losing_trades
        
        # Sharpe Ratio 계산
        if len(self.recent_pnls) > 1:
            mean_return = np.mean(self.recent_pnls)
            std_return = np.std(self.recent_pnls)
            self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        self.last_updated = datetime.now()

class EnhancedVPSStrategyAdapter:
    """통합 VPS 전략 어댑터 (점수 기반 선택)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Enhanced VPS 전략 어댑터 초기화
        
        Args:
            config: 어댑터 설정
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 통합 로깅 시스템
        try:
            self.log_integrator = get_vps_log_integrator()
            self.system_logger = self.log_integrator.get_logger("enhanced_strategy_adapter")
        except Exception as e:
            self.log_integrator = None
            self.system_logger = self.logger
            self.logger.warning(f"통합 로깅 시스템 초기화 실패: {e}")
        
        # 전략 관리
        self.strategies: Dict[str, Any] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.strategy_scores: Dict[str, StrategyScore] = {}
        
        # 캐시 및 성능
        self.score_cache: Dict[str, Tuple[StrategyScore, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)  # 5분 캐시
        
        # VPS 최적화 설정
        self.max_concurrent_strategies = self.config.get('max_concurrent_strategies', 5)
        self.min_score_threshold = self.config.get('min_score_threshold', 0.3)
        self.performance_weight = self.config.get('performance_weight', 0.25)
        
        # Telegram/WebSocket 알림 설정
        self.enable_notifications = self.config.get('enable_notifications', True)
        self.notification_threshold = self.config.get('notification_threshold', 0.7)
        
        # 통일된 신호 변환기
        signal_config = self.config.get('signal_converter', {})
        self.signal_converter = create_unified_signal_converter(signal_config)
        
        # 선물 리스크 관리 시스템 초기화
        self._initialize_futures_risk_management()
        
        # 전략 로드
        self._load_rule_strategies()
        
        # 성과 추적
        self.total_signals_generated = 0
        self.successful_signals = 0
        
        # 현재 포지션 및 마켓 데이터 캐시
        self.current_position = None
        self.current_market_data = None
        self.last_risk_assessment_time = None
        
        self.logger.info(f"Enhanced VPS Strategy Adapter 초기화 완료 - {len(self.strategies)}개 전략 로드, 선물 리스크 관리 활성화")
    
    def _initialize_futures_risk_management(self) -> None:
        """선물 리스크 관리 시스템 초기화"""
        try:
            # 설정에서 선물 리스크 관리 파라미터 로드
            futures_config_data = self.config.get('futures_risk', {})
            
            # 레버리지 설정
            leverage_config = FuturesLeverageConfig(
                min_leverage=futures_config_data.get('min_leverage', 1.0),
                max_leverage=futures_config_data.get('max_leverage', 10.0),
                default_leverage=futures_config_data.get('default_leverage', 3.0),
                volatility_threshold_high=futures_config_data.get('volatility_threshold_high', 0.05),
                volatility_threshold_low=futures_config_data.get('volatility_threshold_low', 0.02),
                max_portfolio_risk=futures_config_data.get('max_portfolio_risk', 0.02),
                adjustment_interval_seconds=futures_config_data.get('adjustment_interval_seconds', 30),
                performance_lookback_trades=futures_config_data.get('performance_lookback_trades', 20),
                enable_time_based_adjustment=futures_config_data.get('enable_time_based_adjustment', True)
            )
            
            # 마진 설정
            margin_config = MarginConfig(
                healthy_margin_ratio=futures_config_data.get('healthy_margin_ratio', 0.3),
                warning_margin_ratio=futures_config_data.get('warning_margin_ratio', 0.2),
                danger_margin_ratio=futures_config_data.get('danger_margin_ratio', 0.1),
                critical_margin_ratio=futures_config_data.get('critical_margin_ratio', 0.05),
                auto_add_margin=futures_config_data.get('auto_add_margin', True),
                auto_reduce_position=futures_config_data.get('auto_reduce_position', True),
                emergency_close_threshold=futures_config_data.get('emergency_close_threshold', 0.05),
                monitoring_interval_seconds=futures_config_data.get('monitoring_interval_seconds', 10)
            )
            
            # 통합 리스크 설정
            risk_config = FuturesRiskConfig(
                leverage_config=leverage_config,
                margin_config=margin_config,
                critical_risk_score=futures_config_data.get('critical_risk_score', 80.0),
                high_risk_score=futures_config_data.get('high_risk_score', 65.0),
                moderate_risk_score=futures_config_data.get('moderate_risk_score', 45.0),
                enable_auto_leverage_adjustment=futures_config_data.get('enable_auto_leverage_adjustment', True),
                enable_auto_margin_management=futures_config_data.get('enable_auto_margin_management', True),
                enable_emergency_protocols=futures_config_data.get('enable_emergency_protocols', True),
                risk_assessment_interval=futures_config_data.get('risk_assessment_interval', 15)
            )
            
            # 리스크 코디네이터 생성
            self.risk_coordinator = FuturesRiskCoordinator(risk_config, enable_logging=True)
            
            # 리스크 관리 활성화 여부
            self.futures_risk_enabled = futures_config_data.get('enabled', True)
            
            if self.futures_risk_enabled:
                self.system_logger.info("선물 리스크 관리 시스템 초기화 완료")
            else:
                self.system_logger.info("선물 리스크 관리 시스템 비활성화")
                
        except Exception as e:
            self.logger.error(f"선물 리스크 관리 시스템 초기화 실패: {e}")
            self.risk_coordinator = None
            self.futures_risk_enabled = False
    
    def _load_rule_strategies(self) -> None:
        """Rule 전략들을 자동으로 로드"""
        try:
            # 프로덕션 전략 경로 설정
            production_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "AuroraQ", "production", "strategies"
            )
            
            # rule_strategies.py 파일 로드
            rule_strategies_path = os.path.join(production_path, "rule_strategies.py")
            
            if os.path.exists(rule_strategies_path):
                # 동적 모듈 로드
                spec = importlib.util.spec_from_file_location("rule_strategies", rule_strategies_path)
                rule_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rule_module)
                
                # 전략 클래스들 로드
                strategy_classes = {
                    "RuleStrategyA": getattr(rule_module, "RuleStrategyA", None),
                    "RuleStrategyB": getattr(rule_module, "RuleStrategyB", None),
                    "RuleStrategyC": getattr(rule_module, "RuleStrategyC", None),
                    "RuleStrategyD": getattr(rule_module, "RuleStrategyD", None),
                    "RuleStrategyE": getattr(rule_module, "RuleStrategyE", None)
                }
                
                # PPO 전략 로드 시도
                try:
                    from trading.ppo_strategy import PPOStrategy, PPOConfig
                    ppo_config = PPOConfig(
                        model_path=self.config.get('ppo_model_path', '/app/models/ppo_model.zip'),
                        confidence_threshold=self.config.get('ppo_confidence_threshold', 0.7)
                    )
                    strategy_classes["PPOStrategy"] = PPOStrategy
                    self.ppo_config = ppo_config
                    logger.info("PPOStrategy 로드 성공")
                except ImportError as e:
                    logger.warning(f"PPOStrategy 로드 실패: {e}")
                    strategy_classes["PPOStrategy"] = None
                    self.ppo_config = None
                
                # 전략 인스턴스 생성
                for name, strategy_class in strategy_classes.items():
                    if strategy_class:
                        try:
                            # PPO 전략은 특별한 설정으로 생성
                            if name == "PPOStrategy" and self.ppo_config:
                                instance = strategy_class(self.ppo_config)
                            else:
                                instance = strategy_class()
                            
                            self.strategies[name] = instance
                            self.strategy_performances[name] = StrategyPerformance()
                            self.logger.info(f"전략 로드 성공: {name}")
                            
                            # PPO 전략의 경우 추가 정보 로깅
                            if name == "PPOStrategy" and hasattr(instance, 'get_ppo_statistics'):
                                ppo_stats = instance.get_ppo_statistics()
                                self.logger.info(f"PPO 통계: {ppo_stats}")
                                
                        except Exception as e:
                            self.logger.error(f"전략 {name} 인스턴스 생성 실패: {e}")
                            # PPO 전략 실패시 strategies에서 제거
                            if name == "PPOStrategy":
                                self.logger.warning("PPO 전략을 사용할 수 없습니다. Rule 전략만 사용됩니다.")
                
                self.logger.info(f"Rule 전략 로드 완료: {list(self.strategies.keys())}")
                
            else:
                self.logger.warning(f"rule_strategies.py 파일을 찾을 수 없습니다: {rule_strategies_path}")
                
        except Exception as e:
            self.logger.error(f"Rule 전략 로드 실패: {e}")
            # 폴백: 빈 전략 딕셔너리 사용
            self.strategies = {}
    
    async def get_best_trading_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        모든 전략을 평가하여 최적의 거래 신호 반환 (선물 리스크 관리 통합)
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            최적 거래 신호 딕셔너리
        """
        start_time = datetime.now()
        
        try:
            if not self.strategies:
                self.logger.warning("로드된 전략이 없습니다")
                return None
            
            price_data = market_data.get('price_data')
            if price_data is None or len(price_data) < 50:
                self.logger.warning("충분하지 않은 가격 데이터")
                return None
            
            # 1. 선물 리스크 관리 평가 (최우선)
            risk_assessment = None
            if self.futures_risk_enabled and self.risk_coordinator:
                # DataFrame을 dict로 변환
                price_dict = {
                    'close': price_data['close'].iloc[-1] if hasattr(price_data, 'iloc') else price_data.get('close', 0.0)
                }
                risk_assessment = await self._assess_futures_risk(market_data, price_dict)
                
                # 거래 제한 확인
                if risk_assessment and not risk_assessment.get('new_positions_allowed', True):
                    return self._create_hold_signal(market_data, f"risk_restriction: {risk_assessment.get('reason', 'high_risk')}")
            
            # 2. 모든 전략에서 신호 수집
            strategy_signals = await self._collect_strategy_signals(price_data, market_data)
            
            if not strategy_signals:
                return self._create_hold_signal(market_data, "no_valid_signals")
            
            # 3. 선물 리스크 기반 신호 필터링
            if self.futures_risk_enabled and self.risk_coordinator and risk_assessment:
                strategy_signals = self._filter_signals_by_risk(strategy_signals, risk_assessment)
            
            if not strategy_signals:
                return self._create_hold_signal(market_data, "signals_filtered_by_risk")
            
            # 4. 점수 기반 전략 선택
            best_signal = await self._select_best_signal(strategy_signals, market_data)
            
            # 5. 선물 리스크 기반 포지션 크기 조정
            if self.futures_risk_enabled and self.risk_coordinator and best_signal and risk_assessment:
                best_signal = await self._adjust_signal_for_futures_risk(best_signal, risk_assessment, market_data)
            
            # 6. 선택된 신호 로깅 및 알림
            if best_signal and best_signal.get('action') != 'HOLD':
                await self._log_signal_selection(best_signal, strategy_signals)
                
                if self.enable_notifications:
                    await self._send_signal_notification(best_signal)
            
            # 7. 성과 추적
            self.total_signals_generated += 1
            if best_signal and best_signal.get('action') != 'HOLD':
                self.successful_signals += 1
            
            # 8. 실행 시간 로깅
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > 1.0:  # 1초 이상 걸린 경우
                self.logger.warning(f"신호 생성 시간 지연: {execution_time:.2f}s")
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"최적 신호 생성 오류: {e}")
            return self._create_hold_signal(market_data, f"error: {str(e)}")
    
    async def _assess_futures_risk(self, market_data: Dict[str, Any], price_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """선물 리스크 평가"""
        try:
            if not self.risk_coordinator:
                return None
            
            # 시장 지표 생성
            market_metrics = MarketMetrics(
                price=price_data.get('close', 0.0),
                volatility_1h=market_data.get('volatility_1h', 0.03),
                volatility_4h=market_data.get('volatility_4h', 0.04), 
                volatility_24h=market_data.get('volatility_24h', 0.05),
                volume_24h=market_data.get('volume_24h', 0.0),
                funding_rate=market_data.get('funding_rate', 0.0),
                rsi_14=market_data.get('rsi_14', 50.0),
                atr_14=market_data.get('atr_14', 0.0),
                market_condition=MarketCondition.NORMAL
            )
            
            # 포지션 지표 생성 (현재 포지션이 있다면)
            position_metrics = None
            if market_data.get('current_position'):
                pos = market_data['current_position']
                position_metrics = PositionMetrics(
                    symbol=pos.get('symbol', 'BTCUSDT'),
                    side=pos.get('side', 'LONG'),
                    size=pos.get('size', 0.0),
                    entry_price=pos.get('entry_price', price_data.get('close', 0.0)),
                    mark_price=price_data.get('close', 0.0),
                    unrealized_pnl=pos.get('unrealized_pnl', 0.0),
                    margin_ratio=pos.get('margin_ratio', 0.3),
                    liquidation_price=pos.get('liquidation_price', 0.0)
                )
            
            # 계좌 잔고
            account_balance = market_data.get('account_balance', 10000.0)
            current_price = price_data.get('close', 0.0)
            
            # 시스템 리스크 평가
            risk_state = await self.risk_coordinator.assess_system_risk(
                market_metrics=market_metrics,
                position_metrics=position_metrics,
                account_balance=account_balance,
                current_price=current_price
            )
            
            return {
                'risk_level': risk_state.overall_risk_level.value,
                'risk_score': risk_state.overall_risk_score,
                'new_positions_allowed': risk_state.trading_restrictions.get('new_positions_allowed', True),
                'position_increase_allowed': risk_state.trading_restrictions.get('position_increase_allowed', True),
                'leverage_increase_allowed': risk_state.trading_restrictions.get('leverage_increase_allowed', True),
                'high_risk_strategies_allowed': risk_state.trading_restrictions.get('high_risk_strategies_allowed', True),
                'emergency_only_mode': risk_state.trading_restrictions.get('emergency_only_mode', False),
                'recommended_actions': risk_state.recommended_actions,
                'reason': f"System risk level: {risk_state.overall_risk_level.value} (score: {risk_state.overall_risk_score:.1f})",
                'component_scores': {
                    'leverage_risk': risk_state.leverage_risk_score,
                    'margin_risk': risk_state.margin_risk_score,
                    'market_risk': risk_state.market_risk_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Futures risk assessment error: {e}")
            # 에러 시 보수적으로 위험한 상태로 반환
            return {
                'risk_level': 'critical',
                'risk_score': 100.0,
                'new_positions_allowed': False,
                'position_increase_allowed': False,
                'leverage_increase_allowed': False,
                'high_risk_strategies_allowed': False,
                'emergency_only_mode': True,
                'recommended_actions': [],
                'reason': f"Risk assessment error: {str(e)}",
                'component_scores': {'leverage_risk': 100.0, 'margin_risk': 100.0, 'market_risk': 100.0}
            }
    
    def _filter_signals_by_risk(self, signals: List[Dict[str, Any]], risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """리스크 기반 신호 필터링"""
        try:
            if not risk_assessment:
                return signals
            
            filtered_signals = []
            
            for signal in signals:
                # 긴급 모드에서는 HOLD 신호만 허용
                if risk_assessment.get('emergency_only_mode', False):
                    if signal.get('action') == 'HOLD':
                        filtered_signals.append(signal)
                    continue
                
                # 신규 포지션 제한 확인
                if signal.get('action') in ['BUY', 'SELL'] and not risk_assessment.get('new_positions_allowed', True):
                    # 신규 포지션 신호를 HOLD로 변경
                    filtered_signal = signal.copy()
                    filtered_signal.update({
                        'action': 'HOLD',
                        'strength': 0.0,
                        'confidence': 0.0,
                        'reason': f"Risk restriction: {risk_assessment.get('reason', 'high_risk')}"
                    })
                    filtered_signals.append(filtered_signal)
                    continue
                
                # 고위험 전략 제한 확인
                if not risk_assessment.get('high_risk_strategies_allowed', True):
                    strategy_name = signal.get('strategy', '')
                    
                    # 고위험으로 분류되는 전략들
                    high_risk_strategies = ['RuleStrategyA', 'RuleStrategyD', 'PPOStrategy']
                    
                    if strategy_name in high_risk_strategies:
                        # 신호 강도를 낮춤
                        filtered_signal = signal.copy()
                        filtered_signal['strength'] *= 0.5
                        filtered_signal['confidence'] *= 0.7
                        filtered_signal['risk_adjusted'] = True
                        filtered_signals.append(filtered_signal)
                        continue
                
                # 기본적으로 신호 통과
                filtered_signals.append(signal)
            
            # 필터링 결과 로깅
            if len(filtered_signals) != len(signals):
                self.logger.info(
                    f"Risk-based signal filtering: {len(signals)} → {len(filtered_signals)} signals",
                    risk_level=risk_assessment.get('risk_level'),
                    risk_score=risk_assessment.get('risk_score')
                )
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Signal filtering error: {e}")
            # 에러 시 모든 신호를 HOLD로 변경
            return [{'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reason': f'Filter error: {str(e)}'}]
    
    async def _adjust_signal_for_futures_risk(self, signal: Dict[str, Any], risk_assessment: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """선물 리스크 기반 신호 조정"""
        try:
            if not signal or not risk_assessment:
                return signal
            
            adjusted_signal = signal.copy()
            
            # 1. 리스크 수준별 포지션 크기 조정
            risk_score = risk_assessment.get('risk_score', 50.0)
            
            # 리스크 점수에 따른 포지션 크기 감소 계수
            if risk_score >= 80:
                size_reduction_factor = 0.3  # 70% 감소
            elif risk_score >= 65:
                size_reduction_factor = 0.5  # 50% 감소
            elif risk_score >= 50:
                size_reduction_factor = 0.7  # 30% 감소
            else:
                size_reduction_factor = 1.0  # 감소 없음
            
            # 2. 동적 레버리지 적용
            if self.leverage_manager and signal.get('action') in ['BUY', 'SELL']:
                try:
                    # 시장 지표 생성
                    price_data = market_data.get('price_data', {})
                    market_metrics = MarketMetrics(
                        price=price_data.get('close', 0.0),
                        volatility_1h=market_data.get('volatility_1h', 0.03),
                        volatility_4h=market_data.get('volatility_4h', 0.04),
                        volatility_24h=market_data.get('volatility_24h', 0.05),
                        rsi_14=market_data.get('rsi_14', 50.0),
                        market_condition=MarketCondition.NORMAL
                    )
                    
                    # 최적 레버리지 계산
                    optimal_leverage, leverage_details = await self.leverage_manager.calculate_optimal_leverage(
                        market_metrics=market_metrics,
                        strategy_confidence=signal.get('confidence', 0.5)
                    )
                    
                    # 레버리지 기반 포지션 크기 계산
                    account_balance = market_data.get('account_balance', 10000.0)
                    entry_price = price_data.get('close', 0.0)
                    
                    if entry_price > 0:
                        position_size, size_details = await self.leverage_manager.calculate_position_size(
                            account_balance=account_balance,
                            entry_price=entry_price,
                            risk_per_trade=0.01  # 1% 리스크
                        )
                        
                        # 리스크 조정 적용
                        adjusted_position_size = position_size * size_reduction_factor
                        
                        adjusted_signal.update({
                            'position_size': adjusted_position_size,
                            'leverage': optimal_leverage,
                            'risk_adjusted_size': True,
                            'original_size': position_size,
                            'size_reduction_factor': size_reduction_factor,
                            'leverage_details': leverage_details,
                            'size_details': size_details
                        })
                
                except Exception as leverage_error:
                    self.logger.warning(f"Leverage calculation error: {leverage_error}")
                    # 폴백: 기본 포지션 크기 적용
                    adjusted_signal['position_size'] = 0.01 * size_reduction_factor
            
            # 3. 스톱로스 및 테이크프로핏 조정
            if risk_score >= 70:
                # 고위험 상황에서는 더 타이트한 스톱로스
                current_stop_loss = adjusted_signal.get('stop_loss_pct', 0.05)
                adjusted_signal['stop_loss_pct'] = min(current_stop_loss, 0.03)  # 최대 3%
                
                current_take_profit = adjusted_signal.get('take_profit_pct', 0.1)
                adjusted_signal['take_profit_pct'] = min(current_take_profit, 0.06)  # 최대 6%
            
            # 4. 신호 강도 및 신뢰도 조정
            confidence_adjustment = max(0.5, 1.0 - (risk_score - 50) / 100)  # 50점 이상에서 감소
            adjusted_signal['confidence'] *= confidence_adjustment
            adjusted_signal['strength'] *= confidence_adjustment
            
            # 5. 리스크 메타데이터 추가
            adjusted_signal.update({
                'risk_assessment': {
                    'risk_level': risk_assessment.get('risk_level'),
                    'risk_score': risk_score,
                    'component_scores': risk_assessment.get('component_scores', {}),
                    'size_reduction_applied': size_reduction_factor < 1.0,
                    'confidence_adjustment': confidence_adjustment
                },
                'futures_risk_adjusted': True,
                'adjustment_timestamp': datetime.now().isoformat()
            })
            
            # 6. 조정 내용 로깅
            if size_reduction_factor < 1.0 or confidence_adjustment < 1.0:
                self.logger.info(
                    f"Signal adjusted for futures risk - Strategy: {signal.get('strategy', 'unknown')}",
                    risk_score=risk_score,
                    size_reduction=f"{(1-size_reduction_factor)*100:.1f}%",
                    confidence_reduction=f"{(1-confidence_adjustment)*100:.1f}%",
                    original_strength=signal.get('strength', 0.0),
                    adjusted_strength=adjusted_signal.get('strength', 0.0)
                )
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Signal adjustment error: {e}")
            # 에러 시 보수적으로 HOLD 신호 반환
            return {
                'action': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'reason': f'Adjustment error: {str(e)}',
                'strategy': signal.get('strategy', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
    
    async def _collect_strategy_signals(self, price_data: pd.DataFrame, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """모든 전략에서 신호를 수집"""
        strategy_signals = []
        
        # 병렬 처리를 위한 태스크 생성
        tasks = []
        for strategy_name, strategy in self.strategies.items():
            task = asyncio.create_task(
                self._get_strategy_signal(strategy_name, strategy, price_data, market_data)
            )
            tasks.append(task)
        
        # 모든 전략 신호 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            strategy_name = list(self.strategies.keys())[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"전략 {strategy_name} 신호 생성 오류: {result}")
                continue
                
            if result and result.get('action') != 'HOLD':
                strategy_signals.append(result)
        
        return strategy_signals
    
    async def _get_strategy_signal(self, strategy_name: str, strategy: Any, 
                                 price_data: pd.DataFrame, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 전략에서 신호 생성"""
        try:
            # 캐시 확인
            cache_key = f"{strategy_name}_{hash(str(price_data.iloc[-1:].to_dict()))}"
            if cache_key in self.score_cache:
                cached_score, cache_time = self.score_cache[cache_key]
                if datetime.now() - cache_time < self.cache_ttl:
                    # 캐시된 점수 사용
                    pass
            
            # generate_signal 메서드 사용 (새로 추가된 메서드)
            if hasattr(strategy, 'generate_signal'):
                signal = strategy.generate_signal(price_data)
                
                if signal and signal.get('action') != 'HOLD':
                    # 메타데이터에서 점수 정보 추출
                    metadata = signal.get('metadata', {})
                    composite_score = metadata.get('composite_score', 0.0)
                    detail_scores = metadata.get('detail_scores', {})
                    confidence = metadata.get('confidence', 0.5)
                    
                    # StrategyScore 객체 생성
                    strategy_score = StrategyScore(
                        composite_score=composite_score,
                        confidence=confidence,
                        detail_scores=detail_scores,
                        timestamp=datetime.now(),
                        strategy_name=strategy_name
                    )
                    
                    # 점수 캐시 업데이트
                    self.strategy_scores[strategy_name] = strategy_score
                    self.score_cache[cache_key] = (strategy_score, datetime.now())
                    
                    # 신호에 전략명 추가
                    signal['strategy_name'] = strategy_name
                    signal['strategy_score'] = strategy_score
                    
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"전략 {strategy_name} 신호 생성 오류: {e}")
            return None
    
    async def _select_best_signal(self, strategy_signals: List[Dict[str, Any]], 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """점수 기반으로 최적 신호 선택"""
        if not strategy_signals:
            return self._create_hold_signal(market_data, "no_signals")
        
        # 각 신호에 대해 최종 점수 계산
        scored_signals = []
        
        for signal in strategy_signals:
            strategy_name = signal.get('strategy_name', 'Unknown')
            strategy_score = signal.get('strategy_score')
            
            if not strategy_score:
                continue
            
            # 성과 기반 가중치 계산
            performance = self.strategy_performances.get(strategy_name, StrategyPerformance())
            performance_score = self._calculate_performance_score(performance)
            
            # 성과 정보를 메타데이터에 추가
            signal['performance_info'] = {
                'profit_factor': performance.profit_factor if performance.profit_factor != float('inf') else 'Infinity',
                'win_rate': performance.win_rate,
                'total_trades': performance.total_trades,
                'sharpe_ratio': performance.sharpe_ratio
            }
            
            # 최근 성과에 따른 가중치
            recent_weight = self._calculate_recent_performance_weight(performance)
            
            # 최종 점수 계산 (전략점수 + 신뢰도 + 성과점수 + 최근성과)
            # PPO vs Rule 전략 차별화된 점수 계산
            if strategy_name == "PPOStrategy":
                # PPO 전략: 모델 신뢰도와 학습 성과를 더 중시
                final_score = (
                    strategy_score.composite_score * 0.40 +      # PPO 모델 점수 (높은 가중치)
                    strategy_score.confidence * 0.30 +          # 모델 신뢰도 (높은 가중치)
                    performance_score * 0.20 +                  # 과거 성과
                    recent_weight * 0.10                        # 최근 성과
                )
                # PPO 모델이 준비되어 있고 신뢰도가 높으면 보너스 점수
                if hasattr(signal.get('metadata', {}), 'model_loaded') and signal['metadata'].get('model_loaded', False):
                    final_score *= 1.1  # 10% 보너스
                    
            else:
                # Rule 전략: 전통적인 가중치 + 안정성 보너스
                final_score = (
                    strategy_score.composite_score * 0.35 +      # 전략 자체 점수
                    strategy_score.confidence * 0.25 +          # 신호 신뢰도
                    performance_score * 0.25 +                  # 과거 성과 (Rule 전략은 검증된 로직)
                    recent_weight * 0.15                        # 최근 성과 가중치
                )
                # Rule 전략의 안정성 보너스 (profit_factor > 1.5일 때)
                if performance_score > 0.7:
                    final_score *= 1.05  # 5% 보너스
            
            scored_signals.append({
                'signal': signal,
                'final_score': final_score,
                'strategy_name': strategy_name,
                'breakdown': {
                    'composite_score': strategy_score.composite_score,
                    'confidence': strategy_score.confidence,
                    'performance_score': performance_score,
                    'recent_weight': recent_weight,
                    'detail_scores': strategy_score.detail_scores
                }
            })
        
        if not scored_signals:
            return self._create_hold_signal(market_data, "no_scored_signals")
        
        # 적응형 전략 선택 로직
        best_scored = self._adaptive_strategy_selection(scored_signals, market_data)
        
        # 최소 임계값 확인
        if best_scored['final_score'] < self.min_score_threshold:
            return self._create_hold_signal(market_data, f"score_below_threshold_{best_scored['final_score']:.3f}")
        
        # 최적 신호 선택 및 표준화
        best_raw_signal = best_scored['signal'].copy()
        best_raw_signal['selection_metadata'] = {
            'final_score': best_scored['final_score'],
            'score_breakdown': best_scored['breakdown'],
            'total_candidates': len(strategy_signals),
            'selection_reason': f"highest_score_{best_scored['final_score']:.3f}"
        }
        
        # 통일된 인터페이스로 변환
        standard_signal = self.signal_converter.convert_strategy_signal(best_raw_signal)
        
        if standard_signal:
            # 유효성 검증
            is_valid, validation_errors = SignalValidator.validate_standard_signal(standard_signal)
            if not is_valid:
                self.logger.warning(f"신호 유효성 검증 실패: {validation_errors}")
                return self._create_hold_signal(market_data, "validation_failed")
            
            # 표준 신호를 VPS 딕셔너리로 변환하여 반환
            return standard_signal.to_vps_dict()
        else:
            return self._create_hold_signal(market_data, "conversion_failed")
    
    def _calculate_performance_score(self, performance: StrategyPerformance) -> float:
        """전략 성과 점수 계산 (Profit Factor 포함)"""
        if performance.total_trades == 0:
            return 0.5  # 중립
        
        # Win rate 점수 (0~1)
        win_rate_score = performance.win_rate
        
        # Profit Factor 점수 (0~1)
        # PF < 1: 손실, PF = 1: 손익분기, PF > 1: 수익
        # PF 1.5 이상을 우수로 평가
        if performance.profit_factor == float('inf'):
            profit_factor_score = 1.0  # 손실 없음
        elif performance.profit_factor > 2.0:
            profit_factor_score = 1.0  # 매우 우수
        elif performance.profit_factor > 1.5:
            profit_factor_score = 0.8 + (performance.profit_factor - 1.5) * 0.4  # 0.8~1.0
        elif performance.profit_factor > 1.0:
            profit_factor_score = 0.5 + (performance.profit_factor - 1.0) * 0.6  # 0.5~0.8
        elif performance.profit_factor > 0.5:
            profit_factor_score = performance.profit_factor  # 0.5~1.0 -> 0.25~0.5
        else:
            profit_factor_score = max(0.0, performance.profit_factor * 0.5)  # 0~0.25
        
        # Sharpe ratio 점수 (0~1)
        sharpe_score = min(1.0, max(0.0, (performance.sharpe_ratio + 1) / 3))  # -1~2 범위를 0~1로
        
        # 총 PnL 점수 (0~1)
        pnl_score = min(1.0, max(0.0, performance.total_pnl / abs(performance.total_pnl) * 0.1 + 0.5)) if performance.total_pnl != 0 else 0.5
        
        # 가중 평균 (Profit Factor 중요도 증가)
        return (
            win_rate_score * 0.25 +       # Win rate 25%
            profit_factor_score * 0.35 +   # Profit Factor 35% (핵심 지표)
            sharpe_score * 0.25 +          # Sharpe Ratio 25%
            pnl_score * 0.15               # 총 PnL 15%
        )
    
    def _calculate_recent_performance_weight(self, performance: StrategyPerformance) -> float:
        """최근 성과에 따른 가중치 계산"""
        if len(performance.recent_pnls) < 3:
            return 0.5  # 중립
        
        # 최근 5거래 성과
        recent_pnls = performance.recent_pnls[-5:]
        win_count = sum(1 for pnl in recent_pnls if pnl > 0)
        recent_win_rate = win_count / len(recent_pnls)
        
        # 최근 평균 수익률
        recent_avg = np.mean(recent_pnls)
        recent_score = min(1.0, max(0.0, recent_avg * 10 + 0.5))  # -5%~5% 범위
        
        return (recent_win_rate * 0.6 + recent_score * 0.4)
    
    def _adaptive_strategy_selection(self, scored_signals: List[Dict[str, Any]], 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        적응형 전략 선택 알고리즘
        시장 상황과 전략 특성을 고려하여 최적 전략을 선택
        
        Args:
            scored_signals: 점수가 매겨진 전략 신호들
            market_data: 현재 시장 데이터
            
        Returns:
            선택된 최적 전략 신호
        """
        if not scored_signals:
            return None
        
        try:
            # 1. 기본 점수 정렬
            sorted_signals = sorted(scored_signals, key=lambda x: x['final_score'], reverse=True)
            
            # 2. 시장 조건 평가
            market_volatility = market_data.get('volatility_24h', 0.03)
            market_trend_strength = abs(market_data.get('trend_strength', 0.0))
            market_sentiment = market_data.get('sentiment', 0.0)
            
            # 시장 조건별 가중치 조정
            market_condition_weights = self._calculate_market_condition_weights(
                volatility=market_volatility,
                trend_strength=market_trend_strength,
                sentiment=market_sentiment
            )
            
            # 3. 전략별 적합도 계산
            adjusted_signals = []
            
            for signal_data in sorted_signals:
                strategy_name = signal_data['strategy_name']
                base_score = signal_data['final_score']
                
                # 전략별 시장 적합도 계산
                market_fitness = self._calculate_strategy_market_fitness(
                    strategy_name, market_condition_weights
                )
                
                # PPO vs Rule 전략 차별화
                strategy_bonus = self._calculate_strategy_bonus(
                    strategy_name, signal_data, market_data
                )
                
                # 최근 성과 기반 동적 조정
                recent_performance_adjustment = self._calculate_recent_performance_adjustment(
                    strategy_name
                )
                
                # 최종 조정 점수 계산
                adjusted_score = (
                    base_score * 0.60 +                           # 기본 점수 60%
                    market_fitness * 0.25 +                      # 시장 적합도 25%
                    strategy_bonus * 0.10 +                      # 전략 보너스 10%
                    recent_performance_adjustment * 0.05          # 최근 성과 조정 5%
                )
                
                adjusted_signals.append({
                    **signal_data,
                    'adjusted_final_score': adjusted_score,
                    'market_fitness': market_fitness,
                    'strategy_bonus': strategy_bonus,
                    'recent_adjustment': recent_performance_adjustment,
                    'selection_factors': {
                        'market_volatility': market_volatility,
                        'trend_strength': market_trend_strength,
                        'sentiment': market_sentiment,
                        'market_weights': market_condition_weights
                    }
                })
            
            # 4. 조정된 점수로 재정렬
            adjusted_signals.sort(key=lambda x: x['adjusted_final_score'], reverse=True)
            
            # 5. 최종 선택 검증
            best_candidate = adjusted_signals[0]
            
            # 점수 차이가 미미할 경우 다양성 고려
            if len(adjusted_signals) > 1:
                score_diff = best_candidate['adjusted_final_score'] - adjusted_signals[1]['adjusted_final_score']
                
                # 점수 차이가 5% 미만이고 다른 전략이면 다양성 선택
                if score_diff < 0.05 and best_candidate['strategy_name'] != adjusted_signals[1]['strategy_name']:
                    # 최근 선택 이력 확인
                    if hasattr(self, 'recent_selections'):
                        recent_strategies = [s['strategy'] for s in self.recent_selections[-5:]]
                        
                        # 최근에 같은 전략을 너무 많이 선택했다면 다양성 선택
                        if recent_strategies.count(best_candidate['strategy_name']) >= 3:
                            best_candidate = adjusted_signals[1]
                            self.logger.info(f"다양성을 위해 전략 변경: {best_candidate['strategy_name']}")
            
            # 6. 선택 이력 기록
            if not hasattr(self, 'recent_selections'):
                self.recent_selections = []
            
            self.recent_selections.append({
                'strategy': best_candidate['strategy_name'],
                'timestamp': datetime.now(),
                'adjusted_score': best_candidate['adjusted_final_score'],
                'base_score': best_candidate['final_score']
            })
            
            # 이력은 최대 20개까지만 유지
            if len(self.recent_selections) > 20:
                self.recent_selections.pop(0)
            
            # 7. 선택 사유 로깅
            self.logger.info(
                f"적응형 전략 선택: {best_candidate['strategy_name']} "
                f"(조정점수: {best_candidate['adjusted_final_score']:.3f}, "
                f"기본점수: {best_candidate['final_score']:.3f}, "
                f"시장적합도: {best_candidate['market_fitness']:.3f})"
            )
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"적응형 전략 선택 오류: {e}")
            # 폴백: 기본 점수로 선택
            return max(scored_signals, key=lambda x: x['final_score'])
    
    def _calculate_market_condition_weights(self, volatility: float, trend_strength: float, 
                                          sentiment: float) -> Dict[str, float]:
        """시장 조건별 가중치 계산"""
        try:
            # 변동성 기반 가중치 (0.01 ~ 0.10 범위)
            if volatility > 0.08:
                volatility_regime = "high"  # 고변동성
            elif volatility > 0.04:
                volatility_regime = "medium"  # 중변동성
            else:
                volatility_regime = "low"  # 저변동성
            
            # 트렌드 강도 기반 가중치 (0.0 ~ 1.0 범위)
            if trend_strength > 0.7:
                trend_regime = "strong"  # 강한 트렌드
            elif trend_strength > 0.4:
                trend_regime = "moderate"  # 중간 트렌드
            else:
                trend_regime = "weak"  # 약한 트렌드/횡보
            
            # 감정 기반 가중치 (-1.0 ~ 1.0 범위)
            if sentiment > 0.3:
                sentiment_regime = "bullish"  # 강세
            elif sentiment < -0.3:
                sentiment_regime = "bearish"  # 약세
            else:
                sentiment_regime = "neutral"  # 중립
            
            return {
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'sentiment_regime': sentiment_regime,
                'volatility_value': volatility,
                'trend_value': trend_strength,
                'sentiment_value': sentiment
            }
            
        except Exception as e:
            self.logger.error(f"시장 조건 가중치 계산 오류: {e}")
            return {
                'volatility_regime': "medium",
                'trend_regime': "moderate", 
                'sentiment_regime': "neutral",
                'volatility_value': 0.03,
                'trend_value': 0.5,
                'sentiment_value': 0.0
            }
    
    def _calculate_strategy_market_fitness(self, strategy_name: str, 
                                         market_weights: Dict[str, float]) -> float:
        """전략별 시장 적합도 계산"""
        try:
            volatility_regime = market_weights['volatility_regime']
            trend_regime = market_weights['trend_regime'] 
            sentiment_regime = market_weights['sentiment_regime']
            
            # 전략별 시장 조건 선호도 매트릭스
            strategy_preferences = {
                "PPOStrategy": {
                    "high": 0.9, "medium": 0.8, "low": 0.6,  # 변동성 대응
                    "strong": 0.9, "moderate": 0.7, "weak": 0.5,  # 트렌드 대응
                    "bullish": 0.8, "bearish": 0.8, "neutral": 0.7  # 감정 대응
                },
                "RuleStrategyA": {  # 공격적 전략
                    "high": 0.8, "medium": 0.9, "low": 0.7,
                    "strong": 0.9, "moderate": 0.8, "weak": 0.6,
                    "bullish": 0.9, "bearish": 0.7, "neutral": 0.8
                },
                "RuleStrategyB": {  # 균형 전략
                    "high": 0.7, "medium": 0.9, "low": 0.8,
                    "strong": 0.8, "moderate": 0.9, "weak": 0.8,
                    "bullish": 0.8, "bearish": 0.8, "neutral": 0.9
                },
                "RuleStrategyC": {  # 보수적 전략
                    "high": 0.6, "medium": 0.8, "low": 0.9,
                    "strong": 0.7, "moderate": 0.8, "weak": 0.9,
                    "bullish": 0.8, "bearish": 0.9, "neutral": 0.8
                },
                "RuleStrategyD": {  # 역추세 전략
                    "high": 0.9, "medium": 0.7, "low": 0.5,
                    "strong": 0.6, "moderate": 0.7, "weak": 0.9,
                    "bullish": 0.6, "bearish": 0.9, "neutral": 0.8
                },
                "RuleStrategyE": {  # 모멘텀 전략
                    "high": 0.8, "medium": 0.8, "low": 0.6,
                    "strong": 0.9, "moderate": 0.8, "weak": 0.5,
                    "bullish": 0.9, "bearish": 0.6, "neutral": 0.7
                }
            }
            
            # 기본값 설정
            if strategy_name not in strategy_preferences:
                return 0.6  # 중립적 적합도
            
            prefs = strategy_preferences[strategy_name]
            
            # 각 시장 조건별 적합도 계산
            volatility_fitness = prefs.get(volatility_regime, 0.6)
            trend_fitness = prefs.get(trend_regime, 0.6)
            sentiment_fitness = prefs.get(sentiment_regime, 0.6)
            
            # 가중 평균 계산 (변동성 40%, 트렌드 35%, 감정 25%)
            market_fitness = (
                volatility_fitness * 0.40 +
                trend_fitness * 0.35 +
                sentiment_fitness * 0.25
            )
            
            return min(1.0, max(0.0, market_fitness))
            
        except Exception as e:
            self.logger.error(f"전략 시장 적합도 계산 오류: {e}")
            return 0.6  # 에러 시 중립값
    
    def _calculate_strategy_bonus(self, strategy_name: str, signal_data: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> float:
        """전략별 보너스 점수 계산"""
        try:
            bonus = 0.0
            
            # PPO 전략 보너스
            if strategy_name == "PPOStrategy":
                # 모델 준비도 보너스
                signal = signal_data.get('signal', {})
                metadata = signal.get('metadata', {})
                
                if metadata.get('model_loaded', False):
                    bonus += 0.1  # 모델 로드 보너스
                
                confidence = signal_data.get('breakdown', {}).get('confidence', 0.0)
                if confidence > 0.8:
                    bonus += 0.05  # 고신뢰도 보너스
                
                # 최근 학습 성과 보너스
                if hasattr(self.strategies.get("PPOStrategy"), 'recent_training_score'):
                    training_score = self.strategies["PPOStrategy"].recent_training_score
                    if training_score > 0.8:
                        bonus += 0.03
            
            # Rule 전략 보너스
            else:
                # 전략별 특수 조건 보너스
                performance = self.strategy_performances.get(strategy_name)
                if performance:
                    # 높은 Profit Factor 보너스
                    if performance.profit_factor > 2.0:
                        bonus += 0.05
                    elif performance.profit_factor > 1.5:
                        bonus += 0.03
                    
                    # 높은 win rate 보너스
                    if performance.win_rate > 0.7:
                        bonus += 0.03
                    
                    # 안정성 보너스 (Sharpe ratio)
                    if performance.sharpe_ratio > 1.0:
                        bonus += 0.02
            
            # 시장 조건별 특별 보너스
            volatility = market_data.get('volatility_24h', 0.03)
            
            # 고변동성에서 PPO 전략 추가 보너스
            if strategy_name == "PPOStrategy" and volatility > 0.06:
                bonus += 0.05
            
            # 저변동성에서 안정적 Rule 전략 보너스
            elif strategy_name in ["RuleStrategyB", "RuleStrategyC"] and volatility < 0.02:
                bonus += 0.03
            
            return min(0.2, max(0.0, bonus))  # 0~20% 보너스 제한
            
        except Exception as e:
            self.logger.error(f"전략 보너스 계산 오류: {e}")
            return 0.0
    
    def _calculate_recent_performance_adjustment(self, strategy_name: str) -> float:
        """최근 성과 기반 동적 조정"""
        try:
            performance = self.strategy_performances.get(strategy_name)
            if not performance or len(performance.recent_pnls) < 3:
                return 0.5  # 중립
            
            # 최근 10거래 성과 분석
            recent_pnls = performance.recent_pnls[-10:]
            
            # 최근 승률
            recent_wins = sum(1 for pnl in recent_pnls if pnl > 0)
            recent_win_rate = recent_wins / len(recent_pnls)
            
            # 최근 평균 수익률
            recent_avg_pnl = np.mean(recent_pnls)
            
            # 최근 변동성 (일관성)
            recent_std = np.std(recent_pnls) if len(recent_pnls) > 1 else 0.0
            consistency_score = 1.0 / (1.0 + recent_std) if recent_std > 0 else 1.0
            
            # 조정 점수 계산
            adjustment = (
                recent_win_rate * 0.4 +                    # 최근 승률 40%
                min(1.0, max(0.0, recent_avg_pnl * 20 + 0.5)) * 0.4 +  # 최근 수익률 40%
                consistency_score * 0.2                    # 일관성 20%
            )
            
            return min(1.0, max(0.0, adjustment))
            
        except Exception as e:
            self.logger.error(f"최근 성과 조정 계산 오류: {e}")
            return 0.5
    
    def _handle_ppo_feedback(self, strategy_name: str, signal: Dict[str, Any], 
                            market_outcome: Optional[float] = None) -> None:
        """
        🎯 향상된 PPO 전략 피드백 처리
        정밀화된 보상 shaping 시스템과 Rule 전략 대비 성과 강화
        """
        try:
            if strategy_name != "PPOStrategy" or "PPOStrategy" not in self.strategies:
                return
            
            ppo_strategy = self.strategies["PPOStrategy"]
            
            # 고급 PPO 보상 피드백 처리
            if hasattr(ppo_strategy, 'add_score_based_reward') and market_outcome is not None:
                # 신호 메타데이터에서 상세 정보 추출
                metadata = signal.get('metadata', {})
                strategy_score = metadata.get('composite_score', 0.0)
                confidence = metadata.get('confidence', 0.5)
                action_str = signal.get('action', 'HOLD')
                
                # 상태 벡터 복원/생성
                current_state = self._reconstruct_state_vector(signal, metadata)
                if current_state is None:
                    self.logger.warning("PPO 상태 벡터 복원 실패 - 피드백 건너뛰기")
                    return
                
                # 액션 매핑
                action_map = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
                action = action_map.get(action_str, 2)
                
                # 다음 상태 추정 (현재는 간단화)
                next_state = self._estimate_next_state(current_state, action, market_outcome)
                
                # Rule 전략 대비 성과 컨텍스트 제공
                rule_performance_context = self._get_rule_performance_context()
                
                # PPO 전략에 Rule 성과 벤치마크 제공
                if hasattr(ppo_strategy, 'set_external_rule_performance'):
                    ppo_strategy.set_external_rule_performance(rule_performance_context['average_performance'])
                
                # 정밀화된 보상 계산 및 피드백
                final_reward = ppo_strategy.add_score_based_reward(
                    state=current_state,
                    action=action,
                    strategy_score=strategy_score,
                    market_outcome=market_outcome,
                    next_state=next_state,
                    done=False,
                    confidence=confidence,
                    action_str=action_str
                )
                
                # Rule 전략 대비 성과 업데이트
                self._update_ppo_vs_rule_performance(
                    ppo_reward=final_reward,
                    ppo_score=strategy_score,
                    market_outcome=market_outcome,
                    rule_context=rule_performance_context
                )
                
                # 상세 로깅
                self.logger.info(
                    f"🎯 고급 PPO 피드백: "
                    f"전략점수={strategy_score:.3f}, "
                    f"시장결과={market_outcome:.3f}, "
                    f"최종보상={final_reward:.3f}, "
                    f"신뢰도={confidence:.3f}, "
                    f"액션={action_str}"
                )
                
                # 성능 벤치마킹 로깅
                if hasattr(self, 'ppo_vs_rule_metrics'):
                    self.logger.debug(
                        f"PPO vs Rule 성과: "
                        f"PPO평균보상={self.ppo_vs_rule_metrics.get('ppo_avg_reward', 0):.3f}, "
                        f"Rule평균성과={self.ppo_vs_rule_metrics.get('rule_avg_performance', 0):.3f}"
                    )
                
            else:
                self.logger.warning("PPO 보상 시스템 또는 시장 결과가 없어 피드백 건너뛰기")
            
        except Exception as e:
            self.logger.error(f"향상된 PPO 피드백 처리 실패: {e}")
            import traceback
            self.logger.debug(f"PPO 피드백 오류 상세: {traceback.format_exc()}")
    
    def _reconstruct_state_vector(self, signal: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """신호 정보에서 상태 벡터 복원/생성"""
        try:
            # 기존 특성이 있다면 사용
            if 'features' in metadata and metadata['features'] is not None:
                features = metadata['features']
                if isinstance(features, (list, np.ndarray)):
                    return np.array(features, dtype=np.float32)
            
            # 신호 정보에서 기본 특성 생성
            state_features = []
            
            # 가격 정보
            price = signal.get('price', 50000.0)
            state_features.append(price / 50000.0)  # 정규화
            
            # 메타데이터에서 기술적 지표 추출
            detail_scores = metadata.get('detail_scores', {})
            
            # 기본 26개 특성으로 구성
            basic_features = [
                price / 50000.0,                              # 0: 정규화된 가격
                1.0,                                          # 1: 거래량 비율 (기본값)
                detail_scores.get('rsi', 0.5),                # 2: RSI
                detail_scores.get('macd', 0.0),               # 3: MACD
                detail_scores.get('adx', 0.25),               # 4: ADX
                0.0, 0.0, 0.0,                               # 5-7: EMA 관련
                detail_scores.get('bb_position', 0.5),        # 8: 볼린저 밴드 위치
                detail_scores.get('stoch_k', 0.5),            # 9: 스토캐스틱 K
                detail_scores.get('stoch_d', 0.5),            # 10: 스토캐스틱 D
                0.0, 0.0,                                    # 11-12: 가격 변화율
                detail_scores.get('volatility', 0.02),        # 13: 변동성
                0.0,                                         # 14: 트렌드 강도
            ]
            
            # 감정 분석 특성 (6개)
            sentiment_features = metadata.get('sentiment_features', [0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
            if len(sentiment_features) < 6:
                sentiment_features.extend([0.0] * (6 - len(sentiment_features)))
            
            # 최종 특성 조합
            final_features = basic_features[:20] + sentiment_features[:6]
            
            # 26개로 맞추기
            while len(final_features) < 26:
                final_features.append(0.0)
            
            return np.array(final_features[:26], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"상태 벡터 복원 실패: {e}")
            return None
    
    def _estimate_next_state(self, current_state: np.ndarray, action: int, 
                           market_outcome: float) -> np.ndarray:
        """다음 상태 추정 (간단한 모델)"""
        try:
            next_state = current_state.copy()
            
            # 가격 변화 반영 (인덱스 0은 정규화된 가격)
            price_change = market_outcome * 0.01  # 1% 기준
            next_state[0] = max(0.1, min(2.0, next_state[0] * (1 + price_change)))
            
            # 변동성 업데이트 (인덱스 13)
            if len(next_state) > 13:
                volatility_change = abs(market_outcome) * 0.1
                next_state[13] = min(1.0, next_state[13] + volatility_change)
            
            # RSI 간단 추정 (인덱스 2)
            if len(next_state) > 2:
                if market_outcome > 0:
                    next_state[2] = min(1.0, next_state[2] + 0.05)
                else:
                    next_state[2] = max(0.0, next_state[2] - 0.05)
            
            return next_state
            
        except Exception as e:
            self.logger.error(f"다음 상태 추정 실패: {e}")
            return current_state.copy()
    
    def _get_rule_performance_context(self) -> Dict[str, float]:
        """Rule 전략들의 성과 컨텍스트 수집"""
        try:
            rule_performances = {}
            total_rule_performance = 0.0
            rule_count = 0
            
            for strategy_name, performance in self.strategy_performances.items():
                if strategy_name != "PPOStrategy" and strategy_name.startswith("RuleStrategy"):
                    if performance.total_trades > 0:
                        # Rule 전략 성과 점수 계산
                        performance_score = (
                            performance.win_rate * 0.3 +
                            min(1.0, performance.profit_factor / 2.0) * 0.4 +
                            min(1.0, max(0.0, performance.sharpe_ratio + 1) / 3) * 0.3
                        )
                        
                        rule_performances[strategy_name] = performance_score
                        total_rule_performance += performance_score
                        rule_count += 1
            
            avg_rule_performance = total_rule_performance / rule_count if rule_count > 0 else 0.6
            
            return {
                'individual_performances': rule_performances,
                'average_performance': avg_rule_performance,
                'rule_count': rule_count,
                'best_rule_performance': max(rule_performances.values()) if rule_performances else 0.6,
                'worst_rule_performance': min(rule_performances.values()) if rule_performances else 0.4
            }
            
        except Exception as e:
            self.logger.error(f"Rule 성과 컨텍스트 수집 실패: {e}")
            return {
                'individual_performances': {},
                'average_performance': 0.6,
                'rule_count': 0,
                'best_rule_performance': 0.6,
                'worst_rule_performance': 0.4
            }
    
    def _update_ppo_vs_rule_performance(self, ppo_reward: float, ppo_score: float, 
                                      market_outcome: float, rule_context: Dict[str, float]):
        """PPO vs Rule 전략 성과 비교 업데이트"""
        try:
            if not hasattr(self, 'ppo_vs_rule_metrics'):
                self.ppo_vs_rule_metrics = {
                    'ppo_rewards': [],
                    'ppo_scores': [],
                    'market_outcomes': [],
                    'rule_performances': [],
                    'comparison_history': []
                }
            
            # 메트릭 업데이트
            metrics = self.ppo_vs_rule_metrics
            metrics['ppo_rewards'].append(ppo_reward)
            metrics['ppo_scores'].append(ppo_score)
            metrics['market_outcomes'].append(market_outcome)
            metrics['rule_performances'].append(rule_context['average_performance'])
            
            # 비교 기록
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'ppo_reward': ppo_reward,
                'ppo_score': ppo_score,
                'market_outcome': market_outcome,
                'rule_avg': rule_context['average_performance'],
                'ppo_vs_rule_ratio': ppo_reward / rule_context['average_performance'] if rule_context['average_performance'] > 0 else 1.0
            }
            metrics['comparison_history'].append(comparison)
            
            # 최근 50개만 유지
            for key in ['ppo_rewards', 'ppo_scores', 'market_outcomes', 'rule_performances']:
                if len(metrics[key]) > 50:
                    metrics[key].pop(0)
            
            if len(metrics['comparison_history']) > 100:
                metrics['comparison_history'].pop(0)
            
            # 평균 성과 계산
            if len(metrics['ppo_rewards']) > 0:
                metrics['ppo_avg_reward'] = np.mean(metrics['ppo_rewards'])
                metrics['rule_avg_performance'] = np.mean(metrics['rule_performances'])
                metrics['ppo_success_rate'] = sum(1 for r in metrics['ppo_rewards'] if r > 0) / len(metrics['ppo_rewards'])
                
                # 상대적 성과 비교
                if metrics['rule_avg_performance'] > 0:
                    metrics['ppo_vs_rule_performance_ratio'] = metrics['ppo_avg_reward'] / metrics['rule_avg_performance']
                else:
                    metrics['ppo_vs_rule_performance_ratio'] = 1.0
            
        except Exception as e:
            self.logger.error(f"PPO vs Rule 성과 업데이트 실패: {e}")
    
    def get_ppo_vs_rule_comparison(self) -> Dict[str, Any]:
        """PPO vs Rule 전략 성과 비교 통계 반환"""
        try:
            if not hasattr(self, 'ppo_vs_rule_metrics'):
                return {'error': 'No comparison data available'}
            
            metrics = self.ppo_vs_rule_metrics
            
            if len(metrics['ppo_rewards']) == 0:
                return {'error': 'No performance data available'}
            
            # 최근 성과 분석
            recent_rewards = metrics['ppo_rewards'][-10:] if len(metrics['ppo_rewards']) >= 10 else metrics['ppo_rewards']
            recent_rule_perf = metrics['rule_performances'][-10:] if len(metrics['rule_performances']) >= 10 else metrics['rule_performances']
            
            return {
                'overall_performance': {
                    'ppo_avg_reward': metrics.get('ppo_avg_reward', 0.0),
                    'rule_avg_performance': metrics.get('rule_avg_performance', 0.0),
                    'ppo_vs_rule_ratio': metrics.get('ppo_vs_rule_performance_ratio', 1.0),
                    'ppo_success_rate': metrics.get('ppo_success_rate', 0.0)
                },
                'recent_performance': {
                    'recent_ppo_avg': np.mean(recent_rewards),
                    'recent_rule_avg': np.mean(recent_rule_perf),
                    'recent_ratio': np.mean(recent_rewards) / np.mean(recent_rule_perf) if np.mean(recent_rule_perf) > 0 else 1.0,
                    'trend': 'improving' if len(recent_rewards) >= 5 and np.mean(recent_rewards[-5:]) > np.mean(recent_rewards[:-5]) else 'declining'
                },
                'statistics': {
                    'total_comparisons': len(metrics['ppo_rewards']),
                    'ppo_reward_variance': np.var(metrics['ppo_rewards']),
                    'rule_performance_variance': np.var(metrics['rule_performances']),
                    'correlation': np.corrcoef(metrics['ppo_rewards'], metrics['market_outcomes'])[0, 1] if len(metrics['ppo_rewards']) > 1 else 0.0
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"PPO vs Rule 비교 통계 계산 실패: {e}")
            return {'error': str(e)}
    
    def _log_rule_strategy_feedback(self, strategy_name: str, signal: Dict[str, Any], 
                                   market_outcome: Optional[float] = None) -> None:
        """Rule 전략 로깅 처리"""
        try:
            # PPO 전략은 별도 처리하므로 제외
            if strategy_name == "PPOStrategy":
                return
            
            # Rule 전략 로거 가져오기
            try:
                from trading.rule_strategy_logger import get_rule_strategy_logger
                rule_logger = get_rule_strategy_logger()
            except ImportError:
                self.logger.warning("Rule 전략 로거를 불러올 수 없습니다")
                return
            
            # 전략 정보 추출
            metadata = signal.get('metadata', {})
            composite_score = metadata.get('composite_score', 0.0)
            confidence = metadata.get('confidence', 0.5)
            
            # 전략 인스턴스에서 성과 정보 가져오기
            total_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            total_pnl = 0.0
            
            if strategy_name in self.strategies:
                strategy_instance = self.strategies[strategy_name]
                if hasattr(strategy_instance, 'position_count'):
                    total_trades = strategy_instance.position_count
                if hasattr(strategy_instance, 'win_rate'):
                    win_rate = strategy_instance.win_rate
                if hasattr(strategy_instance, 'profit_factor'):
                    profit_factor = strategy_instance.profit_factor
                if hasattr(strategy_instance, 'total_pnl'):
                    total_pnl = strategy_instance.total_pnl
            
            # 선택 여부 확인
            selection_metadata = signal.get('selection_metadata', {})
            selected = selection_metadata.get('final_score', 0) > 0
            selection_rank = selection_metadata.get('selection_rank', 0)
            
            # 로깅 실행
            rule_logger.log_strategy_signal(
                strategy_name=strategy_name,
                action=signal.get('action', 'HOLD'),
                strength=signal.get('strength', 0.0),
                strategy_score=composite_score,
                confidence=confidence,
                market_outcome=market_outcome,
                selected=selected,
                selection_rank=selection_rank,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_pnl=total_pnl
            )
            
            self.logger.debug(f"Rule 전략 {strategy_name} 로깅 완료")
            
        except Exception as e:
            self.logger.error(f"Rule 전략 로깅 실패: {e}")
    
    async def add_market_outcome_feedback(self, strategy_name: str, signal: Dict[str, Any], 
                                         market_outcome: float) -> None:
        """전략에 시장 결과 피드백 추가 (PPO 특별 처리 포함)"""
        try:
            # PPO 전략에 대한 특별 처리
            self._handle_ppo_feedback(strategy_name, signal, market_outcome)
            
            # Rule 전략 로깅 추가
            self._log_rule_strategy_feedback(strategy_name, signal, market_outcome)
            
            # 일반 전략 성과 기록 업데이트
            if strategy_name in self.strategy_performances:
                performance = self.strategy_performances[strategy_name]
                
                # 거래 결과 기록
                if market_outcome > 0:
                    performance.winning_trades += 1
                    performance.total_profit += market_outcome
                else:
                    performance.losing_trades += 1
                    performance.total_loss += abs(market_outcome)
                
                performance.total_trades += 1
                performance.total_pnl += market_outcome
                
                # Win rate 업데이트
                performance.win_rate = performance.winning_trades / performance.total_trades
                
                # Profit factor 업데이트
                if performance.total_loss > 0:
                    performance.profit_factor = performance.total_profit / performance.total_loss
                else:
                    performance.profit_factor = float('inf') if performance.total_profit > 0 else 1.0
                
                # 최근 성과 추가
                performance.recent_outcomes.append(market_outcome)
                
                self.logger.info(f"전략 {strategy_name} 성과 업데이트: "
                               f"outcome={market_outcome:.3f}, "
                               f"win_rate={performance.win_rate:.3f}, "
                               f"profit_factor={performance.profit_factor:.3f}")
            
        except Exception as e:
            self.logger.error(f"시장 결과 피드백 처리 실패: {e}")
    
    def _create_hold_signal(self, market_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """HOLD 신호 생성"""
        current_price = 0.0
        if 'current_price' in market_data:
            current_price = market_data['current_price']
        elif 'price_data' in market_data and len(market_data['price_data']) > 0:
            current_price = market_data['price_data']['close'].iloc[-1]
        
        return {
            'action': 'HOLD',
            'strength': 0.0,
            'price': current_price,
            'metadata': {
                'strategy': 'EnhancedVPSAdapter',
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def _log_signal_selection(self, selected_signal: Dict[str, Any], all_signals: List[Dict[str, Any]]) -> None:
        """신호 선택 과정 로깅 (Profit Factor 포함)"""
        try:
            strategy_name = selected_signal.get('strategy_name', 'Unknown')
            selection_meta = selected_signal.get('selection_metadata', {})
            
            # 통합 로깅 시스템 사용 (Profit Factor 포함)
            if self.log_integrator:
                try:
                    from vps_logging.vps_integration import log_strategy_sig
                    await log_strategy_sig(
                        strategy_name=strategy_name,
                        signal_data=selected_signal,
                        selection_metadata=selection_meta,
                        candidates_count=len(all_signals),
                        timestamp=datetime.now().isoformat()
                    )
                except ImportError:
                    # 폴백: 기존 방식
                    pass
                
                # Summary 로그 (CSV 형태) - Profit Factor 추가
                performance_info = selected_signal.get('performance_info', {})
                pf_display = performance_info.get('profit_factor', 0)
                if pf_display == 'Infinity':
                    pf_display = '∞'
                elif isinstance(pf_display, (int, float)):
                    pf_display = f"{pf_display:.3f}"
                
                summary_data = {
                    'timestamp': datetime.now().isoformat(),
                    'selected_strategy': strategy_name,
                    'action': selected_signal.get('action'),
                    'strength': selected_signal.get('strength', 0),
                    'final_score': selection_meta.get('final_score', 0),
                    'candidates_count': len(all_signals),
                    'composite_score': selection_meta.get('score_breakdown', {}).get('composite_score', 0),
                    'confidence': selection_meta.get('score_breakdown', {}).get('confidence', 0),
                    'performance_score': selection_meta.get('score_breakdown', {}).get('performance_score', 0),
                    'profit_factor': pf_display,
                    'win_rate': performance_info.get('win_rate', 0),
                    'total_trades': performance_info.get('total_trades', 0),
                    'sharpe_ratio': performance_info.get('sharpe_ratio', 0)
                }
                
                await self.log_integrator.log_async(
                    LogCategory.SUMMARY,
                    LogLevel.INFO,
                    "strategy_selection",
                    summary_data
                )
            
            # Detailed 로그 (JSON 형태) - Profit Factor 포함
            detail_log = {
                'event': 'strategy_selection',
                'selected_strategy': strategy_name,
                'signal': selected_signal,
                'all_candidates': [
                    {
                        'strategy': s.get('strategy_name'),
                        'action': s.get('action'),
                        'strength': s.get('strength'),
                        'composite_score': s.get('strategy_score', {}).composite_score if hasattr(s.get('strategy_score', {}), 'composite_score') else 0
                    } for s in all_signals
                ],
                'selection_reason': selection_meta.get('selection_reason'),
                'profit_factor_info': selected_signal.get('performance_info', {})
            }
            
            self.system_logger.info(f"전략 선택 (PF 포함): {json.dumps(detail_log, ensure_ascii=False, default=str)}")
            
        except Exception as e:
            self.logger.error(f"신호 선택 로깅 오류: {e}")
    
    async def _send_signal_notification(self, signal: Dict[str, Any]) -> None:
        """Telegram/WebSocket 알림 전송"""
        try:
            strategy_name = signal.get('strategy_name', 'Unknown')
            selection_meta = signal.get('selection_metadata', {})
            score_breakdown = selection_meta.get('score_breakdown', {})
            performance_info = signal.get('performance_info', {})
            
            # Profit Factor 표시 처리
            pf_display = performance_info.get('profit_factor', 0)
            if pf_display == 'Infinity':
                pf_display = '∞'
            else:
                pf_display = f"{pf_display:.2f}"
            
            # 알림 메시지 구성
            message = f"""
🎯 **전략 신호 발생**
📊 전략: {strategy_name}
📈 액션: {signal.get('action')} ({signal.get('strength', 0):.2f})
💯 최종점수: {selection_meta.get('final_score', 0):.3f}

📋 **점수 상세:**
• 전략점수: {score_breakdown.get('composite_score', 0):.3f}
• 신뢰도: {score_breakdown.get('confidence', 0):.3f}  
• 성과점수: {score_breakdown.get('performance_score', 0):.3f}
• 최근성과: {score_breakdown.get('recent_weight', 0):.3f}

📊 **전략 성과:**
• Profit Factor: {pf_display}
• Win Rate: {performance_info.get('win_rate', 0):.1%}
• 총 거래: {performance_info.get('total_trades', 0)}회
• Sharpe: {performance_info.get('sharpe_ratio', 0):.2f}

💰 가격: ${signal.get('price', 0):.2f}
⏰ 시간: {datetime.now().strftime('%H:%M:%S')}
            """.strip()
            
            # WebSocket 알림 (구현 필요)
            # Telegram 알림 (구현 필요)
            
            self.logger.info(f"알림 전송: {strategy_name} - {signal.get('action')}")
            
        except Exception as e:
            self.logger.error(f"알림 전송 오류: {e}")
    
    def update_strategy_performance(self, strategy_name: str, pnl: float) -> None:
        """전략 성과 업데이트 (Profit Factor 포함 로깅)"""
        if strategy_name in self.strategy_performances:
            self.strategy_performances[strategy_name].update_trade(pnl)
            
            # Profit Factor 포함 로깅
            performance = self.strategy_performances[strategy_name]
            
            try:
                # 비동기 로깅을 위해 태스크 생성
                async def log_performance():
                    try:
                        from vps_logging.vps_integration import log_trade_result, log_strategy_perf
                        
                        # 거래 결과 로깅
                        await log_trade_result(strategy_name, pnl)
                        
                        # 5거래마다 성과 요약 로깅
                        if performance.total_trades % 5 == 0:
                            metrics = {
                                'total_trades': performance.total_trades,
                                'winning_trades': performance.winning_trades,
                                'losing_trades': performance.losing_trades,
                                'win_rate': performance.win_rate,
                                'profit_factor': performance.profit_factor,
                                'total_profit': performance.total_profit,
                                'total_loss': performance.total_loss,
                                'avg_win': performance.avg_win,
                                'avg_loss': performance.avg_loss,
                                'sharpe_ratio': performance.sharpe_ratio,
                                'total_pnl': performance.total_pnl
                            }
                            await log_strategy_perf(strategy_name, metrics)
                    except ImportError:
                        # 폴백: 기존 로깅
                        pass
                
                # 태스크 생성 (fire-and-forget)
                import asyncio
                try:
                    asyncio.create_task(log_performance())
                except RuntimeError:
                    # 이벤트 루프가 없는 경우 무시
                    pass
                    
            except Exception as e:
                self.logger.error(f"성과 로깅 오류: {e}")
            
            self.logger.debug(f"전략 {strategy_name} 성과 업데이트: PnL={pnl:.4f}, PF={performance.profit_factor:.3f}")
    
    async def async_update_strategy_performance(self, strategy_name: str, pnl: float) -> None:
        """비동기 전략 성과 업데이트 (권장)"""
        if strategy_name in self.strategy_performances:
            self.strategy_performances[strategy_name].update_trade(pnl)
            performance = self.strategy_performances[strategy_name]
            
            try:
                from vps_logging.vps_integration import log_trade_result, log_strategy_perf
                
                # 거래 결과 로깅
                await log_trade_result(strategy_name, pnl)
                
                # 5거래마다 성과 요약 로깅
                if performance.total_trades % 5 == 0:
                    metrics = {
                        'total_trades': performance.total_trades,
                        'winning_trades': performance.winning_trades,
                        'losing_trades': performance.losing_trades,
                        'win_rate': performance.win_rate,
                        'profit_factor': performance.profit_factor,
                        'total_profit': performance.total_profit,
                        'total_loss': performance.total_loss,
                        'avg_win': performance.avg_win,
                        'avg_loss': performance.avg_loss,
                        'sharpe_ratio': performance.sharpe_ratio,
                        'total_pnl': performance.total_pnl
                    }
                    await log_strategy_perf(strategy_name, metrics)
                    
            except ImportError:
                # 폴백: 기존 로깅
                pass
            
            self.logger.info(f"전략 {strategy_name} 성과 업데이트: PnL={pnl:.4f}, PF={performance.profit_factor:.3f}, WR={performance.win_rate:.1%}")
    
    def update_ppo_prediction_result(self, strategy_name: str, success: bool) -> None:
        """PPO 예측 결과 업데이트"""
        if strategy_name == "PPOStrategy" and strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            if hasattr(strategy, 'update_prediction_result'):
                strategy.update_prediction_result(success)
                self.logger.debug(f"PPO 예측 결과 업데이트: {strategy_name}, 성공: {success}")
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """전략 통계 정보 반환"""
        stats = {
            'total_strategies': len(self.strategies),
            'total_signals_generated': self.total_signals_generated,
            'successful_signals': self.successful_signals,
            'success_rate': self.successful_signals / self.total_signals_generated if self.total_signals_generated > 0 else 0,
            'strategy_performances': {},
            'signal_conversion_stats': self.signal_converter.get_conversion_statistics()
        }
        
        for name, perf in self.strategy_performances.items():
            stats['strategy_performances'][name] = {
                'total_trades': perf.total_trades,
                'winning_trades': perf.winning_trades,
                'losing_trades': perf.losing_trades,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor if perf.profit_factor != float('inf') else 'Infinity',
                'total_pnl': perf.total_pnl,
                'total_profit': perf.total_profit,
                'total_loss': perf.total_loss,
                'avg_win': perf.avg_win,
                'avg_loss': perf.avg_loss,
                'sharpe_ratio': perf.sharpe_ratio,
                'last_updated': perf.last_updated.isoformat()
            }
        
        # PPO 전략 통계 추가
        if 'PPOStrategy' in self.strategies:
            ppo_strategy = self.strategies['PPOStrategy']
            if hasattr(ppo_strategy, 'get_ppo_statistics'):
                stats['ppo_statistics'] = ppo_strategy.get_ppo_statistics()
        
        return stats
    
    def get_available_strategies(self) -> Dict[str, Any]:
        """사용 가능한 전략 목록 반환"""
        strategies_info = {}
        
        for name, strategy in self.strategies.items():
            performance = self.strategy_performances.get(name)
            score = self.strategy_scores.get(name)
            
            strategies_info[name] = {
                'name': name,
                'class': strategy.__class__.__name__ if strategy else 'Unknown',
                'active': performance.is_active if performance else False,
                'performance': {
                    'total_trades': performance.total_trades if performance else 0,
                    'win_rate': performance.win_rate if performance else 0.0,
                    'profit_factor': performance.profit_factor if performance else 0.0,
                    'avg_signal_strength': performance.avg_signal_strength if performance else 0.0
                } if performance else {},
                'score': {
                    'composite': score.composite_score if score else 0.0,
                    'confidence': score.confidence if score else 0.0,
                    'last_updated': score.last_updated.isoformat() if score and score.last_updated else None
                } if score else {},
                'enabled': name in self.strategies,
                'description': getattr(strategy, 'description', f'{name} 전략') if strategy else f'{name} 전략'
            }
        
        return strategies_info

# 별칭 생성 (호환성)
VPSStrategyAdapter = EnhancedVPSStrategyAdapter

# 팩토리 함수
def create_enhanced_vps_strategy_adapter(config: Dict[str, Any] = None) -> EnhancedVPSStrategyAdapter:
    """Enhanced VPS Strategy Adapter 생성 (PPO 포함)"""
    default_config = {
        'max_concurrent_strategies': 6,  # PPO 포함으로 증가
        'min_score_threshold': 0.3,
        'performance_weight': 0.25,
        'enable_notifications': True,
        'notification_threshold': 0.7,
        'signal_converter': {
            'min_strength_threshold': 0.3,
            'min_confidence_threshold': 0.5,
            'default_position_size_pct': 0.02,
            'min_position_size': 0.001,
            'max_position_size': 0.05
        },
        # PPO 설정 추가
        'ppo_model_path': '/app/models/ppo_model.zip',
        'ppo_confidence_threshold': 0.7
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedVPSStrategyAdapter(default_config)

if __name__ == "__main__":
    # 테스트 코드
    async def test_enhanced_adapter():
        print("🧪 Enhanced VPS Strategy Adapter 테스트")
        
        adapter = create_enhanced_vps_strategy_adapter()
        
        # 더미 시장 데이터
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50200,
            'low': np.random.randn(100).cumsum() + 49800,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        market_data = {
            'symbol': 'BTCUSDT',
            'current_price': 50500.0,
            'sentiment': 0.1,
            'volatility': 0.02,
            'price_data': price_data
        }
        
        # 신호 생성 테스트
        signal = await adapter.get_best_trading_signal(market_data)
        print(f"생성된 신호: {signal}")
        
        # 통계 정보
        stats = adapter.get_strategy_statistics()
        print(f"어댑터 통계: {json.dumps(stats, indent=2, default=str, ensure_ascii=False)}")
    
    asyncio.run(test_enhanced_adapter())