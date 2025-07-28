# state_preprocessor.py

import numpy as np

# 시나리오 태그를 숫자로 매핑 (확장 가능)
SCENARIO_TAG_TO_INDEX = {
    "낙관 기대감": 1,
    "시장 낙관": 2,
    "인플레이션 완화": 3,
    "공포 상승": 4,
    "금리/물가 우려": 5,
    "시장 불안": 6,
    "중립 흐름": 0,
    "분석 실패": 0,
    "기타": 0
}

def preprocess_state(price_data, sentiment_score=None, regime_score=None, mab_score=None):
    """
    PPO 학습을 위한 상태 전처리 함수
    - 시계열 가격 변화율: 4종 * 19 = 76차원
    - 감정 지표: sentiment_score, confidence, scenario_tag (총 3차원)
    - 기타 지표: regime_score, mab_score (2차원)
    → 총 81차원 벡터
    """

    def normalize_series(series, length=20):
        if len(series) < length:
            series = [0.0] * (length - len(series)) + list(series)
        else:
            series = series[-length:]
        # 변화율 계산
        returns = [0.0] + [(series[i] - series[i - 1]) / series[i - 1] if series[i - 1] != 0 else 0.0 for i in range(1, length)]
        return returns

    # 시계열 입력
    close_returns = normalize_series(price_data.get("close", []))
    high_returns = normalize_series(price_data.get("high", []))
    low_returns = normalize_series(price_data.get("low", []))
    volume_returns = normalize_series(price_data.get("volume", []))

    # 기본 감정 점수 세팅
    if isinstance(sentiment_score, dict):
        score = sentiment_score.get("sentiment_score", 0.0)
        confidence = sentiment_score.get("confidence", 0.5)
        tag = sentiment_score.get("scenario_tag", "기타")
    else:
        score = sentiment_score if sentiment_score is not None else 0.0
        confidence = 0.5
        tag = "기타"

    sentiment = (score + 1) / 2
    scenario_index = SCENARIO_TAG_TO_INDEX.get(tag, 0)

    # 기타 점수 정규화
    regime = (regime_score + 1) / 2 if regime_score is not None else 0.5
    mab = mab_score if mab_score is not None else 0.5

    # 최종 상태 벡터 구성
    state_vector = close_returns + high_returns + low_returns + volume_returns
    state_vector += [sentiment, confidence, scenario_index, regime, mab]

    return np.array(state_vector, dtype=np.float32)
