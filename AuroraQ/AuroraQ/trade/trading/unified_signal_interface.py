#!/usr/bin/env python3
"""
통일된 VPS 신호 변환 인터페이스
모든 전략 신호를 VPS 실행 가능한 표준 형태로 변환하는 통합 시스템
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SignalAction(Enum):
    """신호 액션 타입"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class SignalStrength(Enum):
    """신호 강도 분류"""
    WEAK = "WEAK"        # 0.0 - 0.3
    MODERATE = "MODERATE" # 0.3 - 0.7
    STRONG = "STRONG"    # 0.7 - 1.0

@dataclass
class StandardSignal:
    """표준화된 VPS 신호 구조"""
    # 필수 필드
    action: SignalAction
    strength: float  # 0.0 - 1.0
    price: float
    timestamp: datetime
    
    # 전략 정보
    strategy_name: str
    composite_score: float
    confidence: float
    
    # 상세 점수
    detail_scores: Dict[str, float] = field(default_factory=dict)
    
    # 리스크 관리
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    
    # 메타데이터
    reason: str = ""
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 분류
    strength_category: SignalStrength = field(init=False)
    risk_level: str = field(init=False)
    
    def __post_init__(self):
        """후처리: 자동 분류"""
        # 강도 분류
        if self.strength <= 0.3:
            self.strength_category = SignalStrength.WEAK
        elif self.strength <= 0.7:
            self.strength_category = SignalStrength.MODERATE
        else:
            self.strength_category = SignalStrength.STRONG
        
        # 리스크 레벨 계산
        self.risk_level = self._calculate_risk_level()
    
    def _calculate_risk_level(self) -> str:
        """리스크 레벨 계산"""
        # 신뢰도와 점수 기반 리스크 평가
        if self.confidence >= 0.8 and self.composite_score >= 0.7:
            return "LOW"
        elif self.confidence >= 0.6 and self.composite_score >= 0.5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def to_vps_dict(self) -> Dict[str, Any]:
        """VPS 실행용 딕셔너리로 변환"""
        return {
            'action': self.action.value,
            'strength': self.strength,
            'strength_category': self.strength_category.value,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'composite_score': self.composite_score,
            'confidence': self.confidence,
            'detail_scores': self.detail_scores,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'reason': self.reason,
            'risk_level': self.risk_level,
            'selection_metadata': self.selection_metadata
        }
    
    def is_actionable(self, min_strength: float = 0.3, min_confidence: float = 0.5) -> bool:
        """실행 가능한 신호인지 확인"""
        if self.action == SignalAction.HOLD:
            return False
        
        return (self.strength >= min_strength and 
                self.confidence >= min_confidence)

class UnifiedSignalConverter:
    """통일된 신호 변환기"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        신호 변환기 초기화
        
        Args:
            config: 변환 설정
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 변환 설정
        self.min_strength_threshold = self.config.get('min_strength_threshold', 0.3)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.5)
        self.default_position_size_pct = self.config.get('default_position_size_pct', 0.02)  # 2%
        
        # 변환 통계
        self.total_conversions = 0
        self.successful_conversions = 0
        self.conversion_errors = 0
    
    def convert_strategy_signal(self, raw_signal: Dict[str, Any]) -> Optional[StandardSignal]:
        """
        전략 원시 신호를 표준 신호로 변환
        
        Args:
            raw_signal: 전략에서 생성된 원시 신호
            
        Returns:
            StandardSignal 또는 None
        """
        try:
            self.total_conversions += 1
            
            # 필수 필드 검증
            if not self._validate_raw_signal(raw_signal):
                self.conversion_errors += 1
                return None
            
            # 액션 변환
            action = self._convert_action(raw_signal.get('action', 'HOLD'))
            
            # 메타데이터 추출
            metadata = raw_signal.get('metadata', {})
            
            # 표준 신호 생성
            standard_signal = StandardSignal(
                action=action,
                strength=float(raw_signal.get('strength', 0.0)),
                price=float(raw_signal.get('price', 0.0)),
                timestamp=datetime.now(),
                strategy_name=raw_signal.get('strategy_name', metadata.get('strategy', 'Unknown')),
                composite_score=metadata.get('composite_score', 0.0),
                confidence=metadata.get('confidence', 0.5),
                detail_scores=metadata.get('detail_scores', {}),
                stop_loss=metadata.get('stop_loss'),
                take_profit=metadata.get('take_profit'),
                reason=metadata.get('reason', ''),
                selection_metadata=raw_signal.get('selection_metadata', {})
            )
            
            # 포지션 크기 계산
            standard_signal.position_size = self._calculate_position_size(standard_signal)
            
            self.successful_conversions += 1
            
            self.logger.debug(f"신호 변환 완료: {standard_signal.strategy_name} -> {standard_signal.action.value}")
            
            return standard_signal
            
        except Exception as e:
            self.conversion_errors += 1
            self.logger.error(f"신호 변환 오류: {e}")
            return None
    
    def convert_multiple_signals(self, raw_signals: List[Dict[str, Any]]) -> List[StandardSignal]:
        """
        여러 신호를 일괄 변환
        
        Args:
            raw_signals: 원시 신호 리스트
            
        Returns:
            변환된 표준 신호 리스트
        """
        converted_signals = []
        
        for raw_signal in raw_signals:
            converted = self.convert_strategy_signal(raw_signal)
            if converted:
                converted_signals.append(converted)
        
        return converted_signals
    
    def _validate_raw_signal(self, raw_signal: Dict[str, Any]) -> bool:
        """원시 신호 유효성 검증"""
        required_fields = ['action', 'strength', 'price']
        
        for field in required_fields:
            if field not in raw_signal:
                self.logger.warning(f"필수 필드 누락: {field}")
                return False
        
        # 값 범위 검증
        strength = raw_signal.get('strength', 0)
        if not (0.0 <= strength <= 1.0):
            self.logger.warning(f"strength 값 범위 오류: {strength}")
            return False
        
        # 가격 검증
        price = raw_signal.get('price', 0)
        if price <= 0:
            self.logger.warning(f"가격 값 오류: {price}")
            return False
        
        return True
    
    def _convert_action(self, action_str: str) -> SignalAction:
        """액션 문자열을 SignalAction으로 변환"""
        action_mapping = {
            'BUY': SignalAction.BUY,
            'SELL': SignalAction.SELL,
            'HOLD': SignalAction.HOLD,
            'CLOSE_LONG': SignalAction.CLOSE_LONG,
            'CLOSE_SHORT': SignalAction.CLOSE_SHORT,
            'LONG': SignalAction.BUY,  # 호환성
            'SHORT': SignalAction.SELL  # 호환성
        }
        
        return action_mapping.get(action_str.upper(), SignalAction.HOLD)
    
    def _calculate_position_size(self, signal: StandardSignal) -> float:
        """신호 강도와 리스크에 기반한 포지션 크기 계산"""
        base_size = self.default_position_size_pct
        
        # 신호 강도에 따른 조정
        strength_multiplier = signal.strength
        
        # 신뢰도에 따른 조정
        confidence_multiplier = signal.confidence
        
        # 리스크 레벨에 따른 조정
        risk_multipliers = {
            'LOW': 1.0,
            'MEDIUM': 0.8,
            'HIGH': 0.5
        }
        risk_multiplier = risk_multipliers.get(signal.risk_level, 0.5)
        
        # 최종 포지션 크기 계산
        position_size = base_size * strength_multiplier * confidence_multiplier * risk_multiplier
        
        # 최소/최대 제한
        min_size = self.config.get('min_position_size', 0.001)  # 0.1%
        max_size = self.config.get('max_position_size', 0.05)   # 5%
        
        return max(min_size, min(max_size, position_size))
    
    def filter_actionable_signals(self, signals: List[StandardSignal]) -> List[StandardSignal]:
        """실행 가능한 신호만 필터링"""
        actionable = []
        
        for signal in signals:
            if signal.is_actionable(
                min_strength=self.min_strength_threshold,
                min_confidence=self.min_confidence_threshold
            ):
                actionable.append(signal)
        
        return actionable
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """변환 통계 반환"""
        success_rate = (self.successful_conversions / self.total_conversions 
                       if self.total_conversions > 0 else 0.0)
        
        return {
            'total_conversions': self.total_conversions,
            'successful_conversions': self.successful_conversions,
            'conversion_errors': self.conversion_errors,
            'success_rate': success_rate,
            'config': self.config
        }

class SignalValidator:
    """신호 유효성 검증기"""
    
    @staticmethod
    def validate_standard_signal(signal: StandardSignal) -> Tuple[bool, List[str]]:
        """표준 신호 유효성 검증"""
        errors = []
        
        # 필수 필드 확인
        if not signal.strategy_name:
            errors.append("전략명이 없습니다")
        
        if signal.price <= 0:
            errors.append("가격이 유효하지 않습니다")
        
        if not (0.0 <= signal.strength <= 1.0):
            errors.append("신호 강도가 범위를 벗어났습니다")
        
        if not (0.0 <= signal.confidence <= 1.0):
            errors.append("신뢰도가 범위를 벗어났습니다")
        
        # 스톱로스/익절 검증
        if signal.stop_loss and signal.action == SignalAction.BUY:
            if signal.stop_loss >= signal.price:
                errors.append("매수 신호의 스톱로스가 현재가보다 높습니다")
        
        if signal.take_profit and signal.action == SignalAction.BUY:
            if signal.take_profit <= signal.price:
                errors.append("매수 신호의 익절가가 현재가보다 낮습니다")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_signal_consistency(signals: List[StandardSignal]) -> Tuple[bool, List[str]]:
        """여러 신호 간 일관성 검증"""
        errors = []
        
        if not signals:
            return True, []
        
        # 동일한 심볼에 대한 상충 신호 확인
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]
        
        if buy_signals and sell_signals:
            errors.append("매수와 매도 신호가 동시에 존재합니다")
        
        # 가격 일관성 확인
        prices = [s.price for s in signals]
        if len(set(prices)) > 1:
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            if price_range / avg_price > 0.001:  # 0.1% 이상 차이
                errors.append("신호 간 가격 차이가 큽니다")
        
        return len(errors) == 0, errors

# 팩토리 함수
def create_unified_signal_converter(config: Dict[str, Any] = None) -> UnifiedSignalConverter:
    """통일된 신호 변환기 생성"""
    default_config = {
        'min_strength_threshold': 0.3,
        'min_confidence_threshold': 0.5,
        'default_position_size_pct': 0.02,
        'min_position_size': 0.001,
        'max_position_size': 0.05
    }
    
    if config:
        default_config.update(config)
    
    return UnifiedSignalConverter(default_config)

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 통일된 VPS 신호 변환 인터페이스 테스트")
    
    # 변환기 생성
    converter = create_unified_signal_converter()
    
    # 테스트 원시 신호
    raw_signals = [
        {
            'action': 'BUY',
            'strength': 0.8,
            'price': 50000.0,
            'strategy_name': 'RuleStrategyA',
            'metadata': {
                'strategy': 'RuleStrategyA',
                'composite_score': 0.75,
                'confidence': 0.82,
                'detail_scores': {
                    'ema_cross': 0.9,
                    'adx_strength': 0.7,
                    'momentum': 0.6,
                    'volume': 0.8
                },
                'reason': 'EMA 골든크로스 + ADX 강세',
                'stop_loss': 49000.0,
                'take_profit': 52000.0
            }
        },
        {
            'action': 'HOLD',
            'strength': 0.2,
            'price': 50000.0,
            'strategy_name': 'RuleStrategyB',
            'metadata': {
                'strategy': 'RuleStrategyB',
                'composite_score': 0.25,
                'confidence': 0.4,
                'reason': '신호 강도 부족'
            }
        }
    ]
    
    # 신호 변환 테스트
    converted_signals = converter.convert_multiple_signals(raw_signals)
    
    print(f"\n📊 변환 결과: {len(converted_signals)}개 신호 변환 완료")
    
    for signal in converted_signals:
        print(f"\n🎯 {signal.strategy_name}:")
        print(f"   액션: {signal.action.value} ({signal.strength_category.value})")
        print(f"   강도: {signal.strength:.3f}, 신뢰도: {signal.confidence:.3f}")
        print(f"   점수: {signal.composite_score:.3f}, 리스크: {signal.risk_level}")
        print(f"   포지션 크기: {signal.position_size:.3f}")
        print(f"   실행가능: {signal.is_actionable()}")
    
    # 통계 정보
    stats = converter.get_conversion_statistics()
    print(f"\n📈 변환 통계: {stats}")
    
    # 유효성 검증 테스트
    for signal in converted_signals:
        is_valid, errors = SignalValidator.validate_standard_signal(signal)
        print(f"\n✅ {signal.strategy_name} 유효성: {is_valid}")
        if errors:
            for error in errors:
                print(f"   ❌ {error}")