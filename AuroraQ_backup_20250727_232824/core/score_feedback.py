import logging
from core.strategy_score_manager import update_strategy_metrics, strategy_metrics

logger = logging.getLogger(__name__)

def reward_to_score(reward: float) -> float:
    """
    보상 값을 정규화하여 0~1 사이 전략 점수로 변환
    """
    try:
        if reward is None:
            logger.warning("⚠️ [보상→점수] reward가 None입니다. 0.0으로 대체합니다.")
            return 0.0
        score = max(0.0, min(1.0, (reward + 1) / 2))  # reward -1 ~ +1 기준
        return round(score, 4)
    except Exception as e:
        logger.warning(f"[보상→점수] 변환 실패: {e}")
        return 0.0


def apply_reward_to_score(strategy_name: str, reward: float, is_real_trade: bool = True):
    """
    보상 기반 전략 메트릭 보정 (avg_return, win_rate, sharpe_ratio 등)

    Args:
        strategy_name (str): 전략명 ("PPO" 포함 가능)
        reward (float): PPO 또는 룰전략 실행 후 평가된 보상
        is_real_trade (bool): 실거래 결과 여부 (학습결과는 전략 점수 반영 안함)
    """
    try:
        if reward is None:
            logger.warning(f"⚠️ [{strategy_name}] 보상이 None입니다. 전략 점수 반영을 건너뜁니다.")
            return

        if not is_real_trade:
            logger.info(f"ℹ️ [{strategy_name}] 학습 보상({reward})은 전략 점수에 반영되지 않음.")
            return

        current = strategy_metrics.get(strategy_name, {
            "win_rate": 0.5,
            "avg_return": 0.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.1
        })

        old_avg = current["avg_return"]
        old_win = current["win_rate"]
        old_sharpe = current["sharpe_ratio"]

        # ✅ avg_return 업데이트 (보상 기반 EMA)
        new_avg = round((old_avg * 0.9 + reward * 0.1), 4)

        # ✅ win_rate 업데이트 (보상 조건 따라 점진적 개선)
        win_delta = 0.01 if reward > 0.5 else -0.01
        new_win = min(1.0, max(0.0, round(old_win + win_delta, 4)))

        # ✅ Sharpe ratio 업데이트
        new_sharpe = round(min(3.0, max(0.1, old_sharpe + (reward - 0.5) * 0.2)), 4)

        updated = {
            **current,
            "avg_return": new_avg,
            "win_rate": new_win,
            "sharpe_ratio": new_sharpe
        }

        update_strategy_metrics(strategy_name, updated)
        logger.info(
            f"📊 [{strategy_name}] 실거래 보상 반영 → "
            f"avg_return: {old_avg:.4f}→{new_avg:.4f}, "
            f"win_rate: {old_win:.4f}→{new_win:.4f}, "
            f"sharpe_ratio: {old_sharpe:.4f}→{new_sharpe:.4f}"
        )

    except Exception as e:
        logger.warning(f"⚠️ [{strategy_name}] 보상 점수 연동 실패: {e}")


def score(strategy_name: str, reward: float, is_real_trade: bool = True) -> float:
    """
    PPO 및 룰 전략 모두 사용할 수 있는 점수 평가 메서드
    - 점수 반환용
    - 보상 기반 전략 평가
    """
    try:
        score = reward_to_score(reward)
        apply_reward_to_score(strategy_name, reward, is_real_trade)
        return score
    except Exception as e:
        logger.warning(f"⚠️ [{strategy_name}] .score() 계산 실패: {e}")
        return 0.0
