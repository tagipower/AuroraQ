"""
PPO/MAB 피드백 루프 통합 브리지
백테스트 결과를 PPO 학습과 MAB 선택에 반영
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from collections import deque, defaultdict

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


class PPOExperienceBuffer:
    """PPO 학습을 위한 경험 버퍼"""
    
    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: 최대 버퍼 크기
        """
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        
    def add_experience(self,
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      next_state: np.ndarray,
                      done: bool,
                      metadata: Optional[Dict[str, Any]] = None):
        """경험 추가"""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.experiences.append(experience)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """배치 데이터 반환"""
        if len(self.experiences) < batch_size:
            return None
        
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        
        batch = {
            "states": np.array([self.states[i] for i in indices]),
            "actions": np.array([self.actions[i] for i in indices]),
            "rewards": np.array([self.rewards[i] for i in indices]),
            "next_states": np.array([self.next_states[i] for i in indices]),
            "dones": np.array([self.dones[i] for i in indices])
        }
        
        return batch
    
    def clear(self):
        """버퍼 초기화"""
        self.experiences.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()


class MABFeedbackProcessor:
    """MAB 피드백 처리기"""
    
    def __init__(self, 
                 decay_factor: float = 0.95,
                 exploration_bonus: float = 0.1):
        """
        Args:
            decay_factor: 시간 할인 계수
            exploration_bonus: 탐색 보너스
        """
        self.decay_factor = decay_factor
        self.exploration_bonus = exploration_bonus
        
        # 전략별 성과 추적
        self.strategy_rewards = defaultdict(list)
        self.strategy_confidence = defaultdict(list)
        self.strategy_exploration_count = defaultdict(int)
        
        # 적응적 파라미터
        self.adaptive_epsilon = {}
        self.performance_trend = defaultdict(deque)
    
    def process_feedback(self,
                        strategy: str,
                        reward: float,
                        confidence: float,
                        is_exploration: bool = False,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        MAB 피드백 처리
        
        Returns:
            업데이트된 전략 점수
        """
        # 기본 보상 처리
        adjusted_reward = reward
        
        # 탐색 보너스
        if is_exploration:
            self.strategy_exploration_count[strategy] += 1
            adjusted_reward += self.exploration_bonus
        
        # 신뢰도 가중치
        confidence_weight = max(0.5, confidence)
        weighted_reward = adjusted_reward * confidence_weight
        
        # 기록 업데이트
        self.strategy_rewards[strategy].append(weighted_reward)
        self.strategy_confidence[strategy].append(confidence)
        
        # 성과 추세 추적 (최근 10개)
        self.performance_trend[strategy].append(weighted_reward)
        if len(self.performance_trend[strategy]) > 10:
            self.performance_trend[strategy].popleft()
        
        # 적응적 엡실론 계산
        self._update_adaptive_epsilon(strategy)
        
        return {
            "raw_reward": reward,
            "adjusted_reward": adjusted_reward,
            "weighted_reward": weighted_reward,
            "confidence": confidence,
            "exploration_count": self.strategy_exploration_count[strategy],
            "adaptive_epsilon": self.adaptive_epsilon.get(strategy, 0.1)
        }
    
    def _update_adaptive_epsilon(self, strategy: str):
        """적응적 엡실론 업데이트"""
        if len(self.performance_trend[strategy]) < 3:
            self.adaptive_epsilon[strategy] = 0.1
            return
        
        # 최근 성과 분산
        recent_rewards = list(self.performance_trend[strategy])
        variance = np.var(recent_rewards)
        
        # 평균 성과
        avg_performance = np.mean(recent_rewards)
        
        # 적응적 엡실론 계산
        # 성과가 나쁘거나 분산이 높으면 더 많은 탐색
        base_epsilon = 0.05
        variance_penalty = min(0.1, variance * 2)
        performance_penalty = max(0, (0.5 - avg_performance) * 0.2)
        
        adaptive_epsilon = base_epsilon + variance_penalty + performance_penalty
        self.adaptive_epsilon[strategy] = min(0.3, adaptive_epsilon)
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """전략별 통계 반환"""
        stats = {}
        
        for strategy in self.strategy_rewards:
            rewards = self.strategy_rewards[strategy]
            confidences = self.strategy_confidence[strategy]
            
            if rewards:
                stats[strategy] = {
                    "avg_reward": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "avg_confidence": np.mean(confidences),
                    "total_trades": len(rewards),
                    "exploration_count": self.strategy_exploration_count[strategy],
                    "exploration_ratio": self.strategy_exploration_count[strategy] / len(rewards),
                    "adaptive_epsilon": self.adaptive_epsilon.get(strategy, 0.1),
                    "performance_trend": list(self.performance_trend[strategy])[-5:]  # 최근 5개
                }
        
        return stats


class StateEncoder:
    """시장 상태를 PPO 학습용 벡터로 인코딩"""
    
    def __init__(self, 
                 price_window: int = 20,
                 indicator_window: int = 14):
        """
        Args:
            price_window: 가격 윈도우 크기
            indicator_window: 지표 윈도우 크기
        """
        self.price_window = price_window
        self.indicator_window = indicator_window
        
        # 정규화 파라미터
        self.price_scaler = None
        self.indicator_scalers = {}
        
    def encode_state(self,
                    price_data: pd.DataFrame,
                    indicators: Dict[str, pd.Series],
                    sentiment_score: float,
                    regime: str,
                    volatility: float) -> np.ndarray:
        """시장 상태 인코딩"""
        features = []
        
        # 1. 가격 정보 (정규화된 수익률)
        if len(price_data) >= self.price_window:
            prices = price_data['close'].tail(self.price_window).values
            returns = np.diff(prices) / prices[:-1]
            features.extend(returns)
        else:
            # 부족한 경우 패딩
            features.extend([0.0] * (self.price_window - 1))
        
        # 2. 기술적 지표
        for indicator_name in ["rsi", "atr"]:
            if indicator_name in indicators:
                values = indicators[indicator_name].tail(self.indicator_window)
                if len(values) > 0:
                    # 정규화 (0-1 범위)
                    if indicator_name == "rsi":
                        normalized = values.iloc[-1] / 100.0
                    elif indicator_name == "atr":
                        normalized = min(1.0, values.iloc[-1] / price_data['close'].iloc[-1])
                    else:
                        normalized = np.tanh(values.iloc[-1])  # 기본 정규화
                    
                    features.append(normalized)
                else:
                    features.append(0.5)  # 기본값
        
        # 3. 감정 점수
        features.append(sentiment_score)
        
        # 4. 시장 레짐 (원핫 인코딩)
        regime_mapping = {
            "bull": [1, 0, 0, 0],
            "bear": [0, 1, 0, 0],
            "neutral": [0, 0, 1, 0],
            "volatile": [0, 0, 0, 1]
        }
        regime_encoded = regime_mapping.get(regime, [0, 0, 1, 0])
        features.extend(regime_encoded)
        
        # 5. 변동성
        features.append(min(1.0, volatility * 20))  # 변동성 정규화
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_size(self) -> int:
        """특성 벡터 크기 반환"""
        # 가격 수익률 + 지표 + 감정 + 레짐 + 변동성
        return (self.price_window - 1) + 2 + 1 + 4 + 1


class BacktestFeedbackBridge:
    """
    백테스트와 학습 시스템 간의 브리지
    백테스트 결과를 PPO와 MAB에 피드백
    """
    
    def __init__(self,
                 enable_ppo_feedback: bool = True,
                 enable_mab_feedback: bool = True,
                 ppo_buffer_size: int = 10000):
        """
        Args:
            enable_ppo_feedback: PPO 피드백 활성화
            enable_mab_feedback: MAB 피드백 활성화
            ppo_buffer_size: PPO 버퍼 크기
        """
        self.enable_ppo_feedback = enable_ppo_feedback
        self.enable_mab_feedback = enable_mab_feedback
        
        # 컴포넌트 초기화
        if enable_ppo_feedback:
            self.ppo_buffer = PPOExperienceBuffer(ppo_buffer_size)
            self.state_encoder = StateEncoder()
        
        if enable_mab_feedback:
            self.mab_processor = MABFeedbackProcessor()
        
        # 세션 추적
        self.current_states = {}
        self.episode_rewards = defaultdict(list)
    
    def process_backtest_step(self,
                            strategy: str,
                            market_data: Dict[str, Any],
                            signal_result: Any,  # SignalResult
                            execution_result: Dict[str, Any],
                            is_exploration: bool = False) -> Dict[str, Any]:
        """백테스트 단계 처리"""
        feedback_results = {}
        
        # 보상 계산
        reward = self._calculate_reward(signal_result, execution_result)
        
        # MAB 피드백
        if self.enable_mab_feedback:
            mab_feedback = self.mab_processor.process_feedback(
                strategy=strategy,
                reward=reward,
                confidence=signal_result.confidence,
                is_exploration=is_exploration,
                metadata={
                    "signal_action": signal_result.action,
                    "position_size": signal_result.position_size,
                    "execution_price": execution_result.get("trade", {}).get("execution_price", 0)
                }
            )
            feedback_results["mab"] = mab_feedback
        
        # PPO 피드백 (PPO 전략인 경우)
        if self.enable_ppo_feedback and strategy == "PPOStrategy":
            ppo_feedback = self._process_ppo_feedback(
                strategy, market_data, signal_result, execution_result, reward
            )
            feedback_results["ppo"] = ppo_feedback
        
        return feedback_results
    
    def _calculate_reward(self, 
                         signal_result: Any,
                         execution_result: Dict[str, Any]) -> float:
        """보상 계산"""
        # 기본 보상 (신뢰도 기반)
        base_reward = signal_result.confidence
        
        # 실행 성공 보너스
        execution_bonus = 0.1 if execution_result.get("executed") else -0.1
        
        # 비용 페널티
        details = execution_result.get("execution_details", {})
        cost_penalty = details.get("total_cost", 0) / details.get("requested_price", 1)
        
        # 최종 보상
        total_reward = base_reward + execution_bonus - cost_penalty
        
        return np.clip(total_reward, 0, 1)
    
    def _process_ppo_feedback(self,
                            strategy: str,
                            market_data: Dict[str, Any],
                            signal_result: Any,
                            execution_result: Dict[str, Any],
                            reward: float) -> Dict[str, Any]:
        """PPO 피드백 처리"""
        try:
            # 상태 인코딩
            price_data = market_data["price"]
            indicators = market_data.get("indicators", {})
            
            current_state = self.state_encoder.encode_state(
                price_data=price_data,
                indicators=indicators,
                sentiment_score=market_data.get("sentiment_score", 0.5),
                regime=signal_result.metadata.get("regime", "neutral") if signal_result.metadata else "neutral",
                volatility=signal_result.metadata.get("volatility", 0.02) if signal_result.metadata else 0.02
            )
            
            # 액션 매핑
            action_mapping = {"HOLD": 0, "BUY": 1, "SELL": 2}
            action = action_mapping.get(signal_result.action, 0)
            
            # 이전 상태가 있으면 경험 추가
            if strategy in self.current_states:
                prev_state = self.current_states[strategy]["state"]
                prev_action = self.current_states[strategy]["action"]
                
                # 경험 버퍼에 추가
                self.ppo_buffer.add_experience(
                    state=prev_state,
                    action=prev_action,
                    reward=reward,
                    next_state=current_state,
                    done=False,  # 연속적인 거래이므로 False
                    metadata={
                        "strategy": strategy,
                        "timestamp": execution_result.get("trade", {}).get("timestamp"),
                        "signal_confidence": signal_result.confidence
                    }
                )
            
            # 현재 상태 업데이트
            self.current_states[strategy] = {
                "state": current_state,
                "action": action,
                "timestamp": datetime.now()
            }
            
            return {
                "state_size": len(current_state),
                "action": action,
                "buffer_size": len(self.ppo_buffer.experiences),
                "reward": reward
            }
            
        except Exception as e:
            logger.error(f"PPO 피드백 처리 오류: {e}")
            return {"error": str(e)}
    
    def get_ppo_training_data(self, batch_size: int = 32) -> Optional[Dict[str, np.ndarray]]:
        """PPO 훈련 데이터 반환"""
        if not self.enable_ppo_feedback:
            return None
        
        return self.ppo_buffer.get_batch(batch_size)
    
    def get_mab_statistics(self) -> Dict[str, Any]:
        """MAB 통계 반환"""
        if not self.enable_mab_feedback:
            return {}
        
        return self.mab_processor.get_strategy_statistics()
    
    def save_feedback_data(self, filepath: str):
        """피드백 데이터 저장"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "ppo_enabled": self.enable_ppo_feedback,
            "mab_enabled": self.enable_mab_feedback
        }
        
        if self.enable_ppo_feedback:
            data["ppo"] = {
                "buffer_size": len(self.ppo_buffer.experiences),
                "feature_size": self.state_encoder.get_feature_size()
            }
        
        if self.enable_mab_feedback:
            data["mab"] = self.get_mab_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def reset(self):
        """피드백 시스템 리셋"""
        if self.enable_ppo_feedback:
            self.ppo_buffer.clear()
            self.current_states.clear()
        
        if self.enable_mab_feedback:
            self.mab_processor = MABFeedbackProcessor()
        
        self.episode_rewards.clear()