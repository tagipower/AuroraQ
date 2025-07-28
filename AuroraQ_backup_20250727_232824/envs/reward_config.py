# reward_config.py

REWARD_WEIGHTS = {
    "profit": 1.0,              # 순이익 보상
    "drawdown_penalty": -0.5,   # 최대 낙폭 패널티
    "volatility_penalty": -0.2, # 변동성 페널티
    "duration_penalty": -0.1,   # 포지션 보유 시간 페널티
    "invalid_action_penalty": -1.0,  # 잘못된 액션 보정
    "stop_loss_penalty": -0.3,  # 손절 발생 시 패널티
    "take_profit_bonus": 0.5,   # 익절 성공 시 보너스
    "regime_alignment_bonus": 0.4,  # 시장 추세와 정합성 보너스
    "sentiment_alignment_bonus": 0.3  # 감정 점수와 정합성 보너스
}

# 예: 개별 학습 전략에 따라 중요도를 다르게 할 수 있음
REWARD_PROFILES = {
    "conservative": {
        "profit": 0.7,
        "drawdown_penalty": -0.8,
        "volatility_penalty": -0.5,
        "duration_penalty": -0.2,
        "stop_loss_penalty": -0.6,
        "take_profit_bonus": 0.6
    },
    "aggressive": {
        "profit": 1.2,
        "drawdown_penalty": -0.3,
        "volatility_penalty": -0.1,
        "stop_loss_penalty": -0.2,
        "take_profit_bonus": 0.8,
        "invalid_action_penalty": -1.5
    },
    "balanced": REWARD_WEIGHTS
}
