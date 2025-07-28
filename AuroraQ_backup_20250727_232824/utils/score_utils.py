# score_utils.py

import numpy as np
from config.score_config_loader import load_score_weights


# ✅ 기본 성과 메트릭 계산 함수들
def calculate_sharpe(returns: list[float]) -> float:
    if not returns:
        return 0.0
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return mean_return / std_return


def calculate_win_rate(returns: list[float]) -> float:
    if not returns:
        return 0.0
    wins = [r for r in returns if r > 0]
    return len(wins) / len(returns)


def calculate_mdd(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (peak - value) / peak
        mdd = max(mdd, drawdown)
    return mdd


def calculate_profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def calculate_expectancy(returns: list[float]) -> float:
    if not returns:
        return 0.0
    win_rate = calculate_win_rate(returns)
    avg_gain = np.mean([r for r in returns if r > 0]) if win_rate > 0 else 0
    avg_loss = -np.mean([r for r in returns if r < 0]) if win_rate < 1 else 0
    return (win_rate * avg_gain) - ((1 - win_rate) * avg_loss)


# ✅ 전략 점수 계산 (감정 점수 포함)
def calculate_total_score(metrics: dict) -> float:
    """
    성과 지표 + 감정 기반 요소에 따라 가중 평균 점수를 계산
    """
    weights = load_score_weights()

    score = (
        weights.get("sharpe", 0.0) * metrics.get("sharpe", 0.0)
        + weights.get("win_rate", 0.0) * metrics.get("win_rate", 0.0)
        + weights.get("profit_factor", 0.0) * metrics.get("profit_factor", 0.0)
        + weights.get("expectancy", 0.0) * metrics.get("expectancy", 0.0)
        - weights.get("mdd", 0.0) * metrics.get("mdd", 0.0)
        + weights.get("sentiment_score", 0.0) * metrics.get("sentiment_score", 0.0)
        + weights.get("sentiment_delta", 0.0) * metrics.get("sentiment_delta", 0.0)
        + weights.get("confidence", 0.0) * metrics.get("confidence", 0.0)
        - weights.get("scenario_penalty", 0.0) * metrics.get("scenario_penalty", 0.0)
    )

    return round(score, 4)
