"""
전략 어댑터 (Strategy Adapter)
기존 전략들을 백테스트 시스템에 통합하는 어댑터
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


class StrategyAdapter:
    """
    기존 전략을 백테스트 시스템에 연결하는 어댑터
    """
    
    def __init__(self, strategy_instance, strategy_name: str):
        """
        Args:
            strategy_instance: 기존 전략 인스턴스
            strategy_name: 전략 이름
        """
        self.strategy = strategy_instance
        self.strategy_name = strategy_name
        self.current_position = None
        self.backtest_indicators = {}  # 백테스트 시스템에서 계산된 지표 저장
        
    def _safe_series_to_float(self, value, default=0.0):
        """Series나 다른 타입을 안전하게 float으로 변환"""
        try:
            if value is None:
                return default
            # pandas Series인 경우
            if hasattr(value, 'iloc') and hasattr(value, '__len__'):
                if len(value) > 0:
                    return float(value.iloc[-1])
                else:
                    return default
            # 이미 숫자인 경우
            if isinstance(value, (int, float)):
                return float(value)
            # numpy 타입인 경우
            if hasattr(value, 'item'):
                return float(value.item())
            # 기타 변환 시도
            return float(value)
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logger.warning(f"float 변환 실패: {value} ({type(value)}) -> {default} 사용, 오류: {e}")
            return default

    def set_indicators(self, indicators: Dict[str, Any]):
        """백테스트 시스템에서 계산된 지표 설정"""
        self.backtest_indicators = indicators or {}
        
        # 전략의 get_cached_indicator 메서드를 패치하여 백테스트 지표를 사용하도록 함
        if hasattr(self.strategy, 'get_cached_indicator'):
            original_method = self.strategy.get_cached_indicator
            
            def patched_get_cached_indicator(indicator, data, **params):
                # 먼저 백테스트 시스템의 지표에서 찾기
                if indicator in self.backtest_indicators:
                    cached_value = self.backtest_indicators[indicator]
                    logger.debug(f"백테스트 캐시에서 {indicator} 지표 반환: {type(cached_value)}")
                    
                    # 안전한 반환값 처리
                    if cached_value is None:
                        return None
                    
                    # pandas Series나 DataFrame인 경우 그대로 반환
                    if hasattr(cached_value, 'iloc'):
                        return cached_value
                    
                    # dict 형태 (bollinger 같은 통합 지표)인 경우
                    if isinstance(cached_value, dict):
                        return cached_value
                    
                    return cached_value
                
                # bollinger 요청인데 개별 지표들이 있는 경우 자동 조합
                if indicator == "bollinger" and all(k in self.backtest_indicators for k in ["bb_upper", "bb_middle", "bb_lower"]):
                    return {
                        'upper': self.backtest_indicators["bb_upper"],
                        'middle': self.backtest_indicators["bb_middle"],
                        'lower': self.backtest_indicators["bb_lower"]
                    }
                
                # MACD DataFrame에서 개별 지표 추출
                if indicator == "macd" and "macd" in self.backtest_indicators:
                    macd_df = self.backtest_indicators["macd"]
                    if isinstance(macd_df, pd.DataFrame) and 'macd' in macd_df.columns:
                        return macd_df['macd']
                    else:
                        return macd_df
                        
                if indicator == "macd_signal" and "macd" in self.backtest_indicators:
                    macd_df = self.backtest_indicators["macd"]
                    if isinstance(macd_df, pd.DataFrame) and 'signal' in macd_df.columns:
                        return macd_df['signal']
                    # 개별 지표가 있으면 그것 사용
                    elif "macd_signal" in self.backtest_indicators:
                        return self.backtest_indicators["macd_signal"]
                
                # 백테스트 지표에 없으면 원래 메서드 호출
                try:
                    result = original_method(indicator, data, **params)
                    logger.debug(f"전략 캐시에서 {indicator} 지표 반환: {type(result)}")
                    return result
                except Exception as e:
                    logger.warning(f"지표 {indicator} 조회 실패: {e}")
                    return None
            
            # 메서드 패치
            self.strategy.get_cached_indicator = patched_get_cached_indicator
            
        # 전략에 안전한 float 변환 함수 추가
        if hasattr(self.strategy, '__dict__'):
            self.strategy._safe_float = self._safe_series_to_float
        
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        가격 데이터를 기반으로 신호 생성
        
        Args:
            price_data: OHLCV 데이터프레임
            
        Returns:
            신호 딕셔너리 {action, strength, price, metadata}
        """
        try:
            # 데이터 충분성 확인
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": price_data['close'].iloc[-1] if len(price_data) > 0 else 0.0,
                    "metadata": {"reason": "insufficient_data"}
                }
            
            current_price = price_data['close'].iloc[-1]
            
            # 포지션 상태에 따른 신호 생성
            if self.current_position is None:
                # 진입 신호 확인
                entry_signal = self.strategy.should_enter(price_data)
                if entry_signal:
                    # 포지션 기록
                    self.current_position = {
                        "side": entry_signal["side"],
                        "entry_price": current_price,
                        "entry_time": datetime.now(),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit"),
                        "metadata": entry_signal
                    }
                    
                    return {
                        "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                        "strength": entry_signal.get("confidence", 0.5),
                        "price": current_price,
                        "metadata": {
                            "strategy": self.strategy_name,
                            "reason": entry_signal.get("reason", ""),
                            "confidence": entry_signal.get("confidence", 0.5),
                            "stop_loss": entry_signal.get("stop_loss"),
                            "take_profit": entry_signal.get("take_profit")
                        }
                    }
            else:
                # 청산 신호 확인
                exit_reason = self.strategy.should_exit(
                    self._create_position_object(), price_data
                )
                if exit_reason:
                    # 포지션 청산
                    side = self.current_position["side"]
                    confidence = self.current_position.get("confidence", 0.5)
                    self.current_position = None
                    
                    return {
                        "action": "SELL" if side == "LONG" else "BUY",
                        "strength": confidence,
                        "price": current_price,
                        "metadata": {
                            "strategy": self.strategy_name,
                            "reason": exit_reason,
                            "is_exit": True
                        }
                    }
            
            # 신호 없음
            return {
                "action": "HOLD",
                "strength": 0.5,
                "price": current_price,
                "metadata": {"strategy": self.strategy_name, "reason": "no_signal"}
            }
            
        except Exception as e:
            logger.error(f"전략 {self.strategy_name} 신호 생성 오류: {e}")
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": price_data['close'].iloc[-1] if len(price_data) > 0 else 0.0,
                "metadata": {"error": str(e)}
            }
    
    def _create_position_object(self):
        """포지션 객체 생성 (기존 전략 호환용)"""
        if not self.current_position:
            return None
            
        class PositionObject:
            def __init__(self, pos_data):
                self.entry_price = pos_data["entry_price"]
                self.entry_time = pos_data["entry_time"]
                self.side = pos_data["side"]
                self.confidence = pos_data.get("confidence", 0.5)
                self.stop_loss = pos_data.get("stop_loss")
                self.take_profit = pos_data.get("take_profit")
                
            @property
            def holding_time(self):
                return datetime.now() - self.entry_time
        
        return PositionObject(self.current_position)
    
    def reset(self):
        """포지션 리셋"""
        self.current_position = None


class StrategyRegistry:
    """전략 등록 및 관리 시스템"""
    
    def __init__(self):
        self.strategies = {}
        self.strategy_adapters = {}
        
    def register_strategy(self, strategy_class, strategy_name: str, **kwargs):
        """전략 등록"""
        try:
            # 전략 인스턴스 생성
            strategy_instance = strategy_class(**kwargs)
            
            # 어댑터 생성
            adapter = StrategyAdapter(strategy_instance, strategy_name)
            
            # 등록
            self.strategies[strategy_name] = strategy_instance
            self.strategy_adapters[strategy_name] = adapter
            
            logger.info(f"전략 등록 완료: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"전략 {strategy_name} 등록 실패: {e}")
            return False
    
    def get_strategy_adapter(self, strategy_name: str) -> Optional[StrategyAdapter]:
        """전략 어댑터 반환"""
        return self.strategy_adapters.get(strategy_name)
    
    def get_all_strategy_names(self) -> List[str]:
        """등록된 모든 전략 이름 반환"""
        return list(self.strategies.keys())
    
    def reset_all_strategies(self):
        """모든 전략 리셋"""
        for adapter in self.strategy_adapters.values():
            adapter.reset()


# 전역 전략 레지스트리
strategy_registry = StrategyRegistry()


def register_builtin_strategies():
    """내장 전략들 등록"""
    try:
        # 통합 룰 전략들 등록
        try:
            # 절대 import 시도
            try:
                from rule_strategies import (
                    RuleStrategyA, RuleStrategyB, RuleStrategyC, 
                    RuleStrategyD, RuleStrategyE
                )
            except ImportError:
                # 상대 import 시도
                from .rule_strategies import (
                    RuleStrategyA, RuleStrategyB, RuleStrategyC, 
                    RuleStrategyD, RuleStrategyE
                )
            
            # 모든 룰 전략 등록
            rule_strategies = [
                (RuleStrategyA, "RuleStrategyA"),
                (RuleStrategyB, "RuleStrategyB"), 
                (RuleStrategyC, "RuleStrategyC"),
                (RuleStrategyD, "RuleStrategyD"),
                (RuleStrategyE, "RuleStrategyE")
            ]
            
            for strategy_class, strategy_name in rule_strategies:
                try:
                    strategy_registry.register_strategy(strategy_class, strategy_name)
                    logger.info(f"통합 룰 전략 등록 성공: {strategy_name}")
                except Exception as e:
                    logger.warning(f"{strategy_name} 등록 실패: {e}")
                    
        except ImportError as e:
            logger.warning(f"통합 룰 전략 모듈 로드 실패: {e}")
            
            # 개별 전략 로드 시도 (Fallback)
            try:
                from strategy.rule_strategy_a import RuleStrategyA
                strategy_registry.register_strategy(RuleStrategyA, "RuleStrategyA")
            except ImportError as e:
                logger.warning(f"RuleStrategyA 등록 실패: {e}")
            
            try:
                from strategy.rule_strategy_b import RuleStrategyB
                strategy_registry.register_strategy(RuleStrategyB, "RuleStrategyB")
            except ImportError as e:
                logger.warning(f"RuleStrategyB 등록 실패: {e}")
            
            try:
                from strategy.rule_strategy_c import RuleStrategyC
                strategy_registry.register_strategy(RuleStrategyC, "RuleStrategyC")
            except ImportError as e:
                logger.warning(f"RuleStrategyC 등록 실패: {e}")
            
            try:
                from strategy.rule_strategy_d import RuleStrategyD
                strategy_registry.register_strategy(RuleStrategyD, "RuleStrategyD")
            except ImportError as e:
                logger.warning(f"RuleStrategyD 등록 실패: {e}")
            
            try:
                from strategy.rule_strategy_e import RuleStrategyE
                strategy_registry.register_strategy(RuleStrategyE, "RuleStrategyE")
            except ImportError as e:
                logger.warning(f"RuleStrategyE 등록 실패: {e}")
        
        # OptimizedRuleStrategyE 등록 (최적화 버전)
        try:
            # 절대 경로로 임포트 시도
            import sys
            import os
            
            # 전략 모듈 경로 추가
            strategy_path = os.path.join(os.path.dirname(__file__), '..', 'strategies')
            if strategy_path not in sys.path:
                sys.path.insert(0, strategy_path)
            
            from optimized_rule_strategy_e import OptimizedRuleStrategyE
            strategy_registry.register_strategy(OptimizedRuleStrategyE, "OptimizedRuleStrategyE")
            logger.info("최적화된 RuleStrategyE 등록 성공")
        except ImportError as e:
            logger.warning(f"OptimizedRuleStrategyE 등록 실패: {e}")
            # 대안: 직접 로드 시도
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "optimized_rule_strategy_e", 
                    os.path.join(os.path.dirname(__file__), '..', 'strategies', 'optimized_rule_strategy_e.py')
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                OptimizedRuleStrategyE = module.OptimizedRuleStrategyE
                strategy_registry.register_strategy(OptimizedRuleStrategyE, "OptimizedRuleStrategyE")
                logger.info("최적화된 RuleStrategyE 대안 등록 성공")
            except Exception as e2:
                logger.error(f"최적화된 전략 대안 등록도 실패: {e2}")
        
        # PPO 전략 등록
        try:
            from .ppo_strategy_adapter import PPOStrategyAdapter
            
            # PPO 어댑터를 전략으로 등록
            class PPOStrategyWrapper:
                def __init__(self):
                    self.adapter = PPOStrategyAdapter()
                    self.strategy_name = "PPOStrategy"
                
                def should_enter(self, price_data):
                    return self.adapter.should_enter(price_data)
                
                def should_exit(self, position, price_data):
                    return self.adapter.should_exit(position, price_data)
                
                def reset(self):
                    self.adapter.reset()
            
            strategy_registry.register_strategy(PPOStrategyWrapper, "PPOStrategy")
            logger.info("PPO 전략 등록 성공")
            
        except ImportError as e:
            logger.warning(f"PPO 전략 등록 실패: {e}")
        except Exception as e:
            logger.error(f"PPO 전략 등록 중 오류: {e}")
        
        logger.info(f"등록된 전략 수: {len(strategy_registry.get_all_strategy_names())}")
        logger.info(f"등록된 전략들: {strategy_registry.get_all_strategy_names()}")
        
    except Exception as e:
        logger.error(f"내장 전략 등록 중 오류: {e}")


def get_strategy_registry() -> StrategyRegistry:
    """전략 레지스트리 반환"""
    # 전략이 등록되어 있지 않으면 자동으로 등록
    if len(strategy_registry.get_all_strategy_names()) == 0:
        register_builtin_strategies()
    
    return strategy_registry