# mab_selector.py - 개선 버전
import random
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from core.strategy_score_manager import get_all_current_scores, update_strategy_metrics

logger = logging.getLogger(__name__)


class MABSelector:
    """
    Multi-Armed Bandit 기반 메타 전략 선택기 (개선 버전)
    - UCB (Upper Confidence Bound) 알고리즘 추가
    - Epsilon decay 기능
    - Thompson Sampling 옵션
    """

    def __init__(
        self, 
        strategy_names: List[str], 
        epsilon: float = 0.1,
        algorithm: str = "epsilon_greedy",  # epsilon_greedy, ucb, thompson
        decay_rate: float = 0.995,
        min_epsilon: float = 0.01,
        ucb_c: float = 2.0
    ):
        """
        Args:
            strategy_names: 전략 이름 리스트
            epsilon: 탐색 확률 (0~1)
            algorithm: MAB 알고리즘 타입
            decay_rate: epsilon 감소율
            min_epsilon: 최소 epsilon 값
            ucb_c: UCB 탐색 파라미터
        """
        self.strategy_names = strategy_names
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.algorithm = algorithm
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.ucb_c = ucb_c
        
        # 통계 추적 - 증분 업데이트를 위한 추가 필드
        self.counts = defaultdict(int)
        self.total_rewards = defaultdict(float)
        self.squared_rewards = defaultdict(float)  # 분산 계산용
        self.running_mean = defaultdict(float)  # 증분 평균
        self.running_variance = defaultdict(float)  # 증분 분산
        
        # Thompson Sampling용 베타 분포 파라미터
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        
        self.total_selections = 0
        
        logger.info(
            f"MAB Selector 초기화: algorithm={algorithm}, "
            f"epsilon={epsilon}, strategies={len(strategy_names)}"
        )

    def select(self) -> str:
        """전략 선택 (알고리즘별 분기)"""
        self.total_selections += 1
        
        if self.algorithm == "epsilon_greedy":
            return self._select_epsilon_greedy()
        elif self.algorithm == "ucb":
            return self._select_ucb()
        elif self.algorithm == "thompson":
            return self._select_thompson()
        else:
            return self._select_epsilon_greedy()

    def _select_epsilon_greedy(self) -> str:
        """Epsilon-Greedy 선택"""
        current_scores = get_all_current_scores()
        
        # 미탐색 전략 우선
        unexplored = [s for s in self.strategy_names if self.counts[s] == 0]
        if unexplored:
            chosen = random.choice(unexplored)
            logger.info(f"[MAB-EG] 🔍 미탐색 전략: {chosen}")
            return chosen
        
        # Epsilon decay
        self._decay_epsilon()
        
        # 탐색 vs 활용
        if random.random() < self.epsilon:
            chosen = random.choice(self.strategy_names)
            logger.info(f"[MAB-EG] 🎲 탐색: {chosen} (ε={self.epsilon:.3f})")
        else:
            # 현재 점수 + 과거 평균 보상 고려
            combined_scores = {}
            for s in self.strategy_names:
                current_score = current_scores.get(s, 0)
                avg_reward = self._get_average_reward(s)
                combined_scores[s] = 0.7 * current_score + 0.3 * avg_reward
            
            chosen = max(combined_scores, key=combined_scores.get)
            logger.info(
                f"[MAB-EG] ✅ 활용: {chosen} "
                f"(score={combined_scores[chosen]:.3f})"
            )
        
        return chosen

    def _select_ucb(self) -> str:
        """Upper Confidence Bound 선택"""
        current_scores = get_all_current_scores()
        
        # 미탐색 전략 우선
        unexplored = [s for s in self.strategy_names if self.counts[s] == 0]
        if unexplored:
            chosen = random.choice(unexplored)
            logger.info(f"[MAB-UCB] 🔍 미탐색 전략: {chosen}")
            return chosen
        
        # UCB 계산
        ucb_values = {}
        for s in self.strategy_names:
            avg_reward = self._get_average_reward(s)
            exploration_bonus = self.ucb_c * np.sqrt(
                np.log(self.total_selections) / self.counts[s]
            )
            ucb_values[s] = avg_reward + exploration_bonus
        
        chosen = max(ucb_values, key=ucb_values.get)
        logger.info(
            f"[MAB-UCB] 📊 선택: {chosen} "
            f"(UCB={ucb_values[chosen]:.3f})"
        )
        
        return chosen

    def _select_thompson(self) -> str:
        """Thompson Sampling 선택"""
        # 각 전략에서 샘플링
        samples = {}
        for s in self.strategy_names:
            # Beta 분포에서 샘플링
            sample = np.random.beta(self.alpha[s], self.beta[s])
            samples[s] = sample
        
        chosen = max(samples, key=samples.get)
        logger.info(
            f"[MAB-TS] 🎯 선택: {chosen} "
            f"(sample={samples[chosen]:.3f})"
        )
        
        return chosen

    def update(self, strategy_name: str, reward: float, metrics: Optional[Dict] = None):
        """보상 업데이트 (증분 통계 사용)"""
        # 증분 통계 업데이트
        n = self.counts[strategy_name]
        old_mean = self.running_mean[strategy_name]
        
        # Welford's online algorithm로 평균과 분산 증분 업데이트
        self.counts[strategy_name] += 1
        new_n = self.counts[strategy_name]
        
        # 평균 업데이트
        delta = reward - old_mean
        self.running_mean[strategy_name] = old_mean + delta / new_n
        
        # 분산 업데이트
        delta2 = reward - self.running_mean[strategy_name]
        self.running_variance[strategy_name] += delta * delta2
        
        # 전체 통계 (호환성 유지)
        self.total_rewards[strategy_name] += reward
        self.squared_rewards[strategy_name] += reward ** 2
        
        # Thompson Sampling용 베타 분포 업데이트
        if reward > 0.5:  # 성공으로 간주
            self.alpha[strategy_name] += 1
        else:
            self.beta[strategy_name] += 1
        
        # Score Manager 업데이트
        if metrics is None:
            metrics = {"reward_shaping_score": reward}
        else:
            metrics["reward_shaping_score"] = reward
            
        update_strategy_metrics(strategy_name, metrics)
        
        logger.info(
            f"[MAB] 📈 업데이트: {strategy_name} → "
            f"reward={reward:.4f}, count={self.counts[strategy_name]}, "
            f"mean={self.running_mean[strategy_name]:.4f}"
        )

    def _get_average_reward(self, strategy_name: str) -> float:
        """평균 보상 계산 (증분 평균 사용)"""
        if self.counts[strategy_name] == 0:
            return 0.0
        # 증분 평균 사용 (더 정확하고 빠름)
        return self.running_mean[strategy_name]

    def _get_reward_std(self, strategy_name: str) -> float:
        """보상 표준편차 계산 (증분 분산 사용)"""
        if self.counts[strategy_name] < 2:
            return 0.0
        
        # 증분 분산에서 표준편차 계산
        variance = self.running_variance[strategy_name] / (self.counts[strategy_name] - 1)
        return np.sqrt(max(0, variance))

    def _decay_epsilon(self):
        """Epsilon 감소"""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate
            self.epsilon = max(self.epsilon, self.min_epsilon)

    def get_statistics(self) -> Dict[str, Dict]:
        """상세 통계 반환"""
        stats = {}
        
        for s in self.strategy_names:
            avg_reward = self._get_average_reward(s)
            std_reward = self._get_reward_std(s)
            
            stats[s] = {
                "count": self.counts[s],
                "total_reward": round(self.total_rewards[s], 4),
                "avg_reward": round(avg_reward, 4),
                "std_reward": round(std_reward, 4),
                "selection_rate": round(
                    self.counts[s] / max(1, self.total_selections), 3
                )
            }
            
            # Thompson Sampling 파라미터
            if self.algorithm == "thompson":
                stats[s]["alpha"] = round(self.alpha[s], 2)
                stats[s]["beta"] = round(self.beta[s], 2)
        
        return stats

    def reset_strategy(self, strategy_name: str):
        """특정 전략 통계 리셋"""
        self.counts[strategy_name] = 0
        self.total_rewards[strategy_name] = 0.0
        self.squared_rewards[strategy_name] = 0.0
        self.running_mean[strategy_name] = 0.0
        self.running_variance[strategy_name] = 0.0
        self.alpha[strategy_name] = 1.0
        self.beta[strategy_name] = 1.0
        
        logger.info(f"[MAB] 전략 리셋: {strategy_name}")

    def get_best_strategy(self) -> Tuple[str, float]:
        """최고 성과 전략 반환"""
        if not any(self.counts.values()):
            return self.strategy_names[0], 0.0
        
        avg_rewards = {
            s: self._get_average_reward(s) 
            for s in self.strategy_names 
            if self.counts[s] > 0
        }
        
        best_strategy = max(avg_rewards, key=avg_rewards.get)
        return best_strategy, avg_rewards[best_strategy]

    def should_explore(self) -> bool:
        """탐색 여부 결정 (디버깅용)"""
        if self.algorithm == "epsilon_greedy":
            return random.random() < self.epsilon
        return False