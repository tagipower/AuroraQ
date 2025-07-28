import os
import pandas as pd
import numpy as np

def generate_dummy_merged_data(
    start_date="2024-12-01",
    end_date="2025-01-01",
    freq="5min",
    output_path="data/merged_data.csv"
):
    """
    AuroraQ 백테스트 시나리오용 업그레이드 더미 데이터 생성기 (1번 호환 방식)
    
    - 모든 전략(RuleA~E, PPO)에서 필요한 'price' 필드를 포함
    - sentiment_score, regime_score, volatility 등 시나리오 태그 필드 포함
    - 5분봉 기준으로 기본적인 변동성, 볼륨, 추세 패턴 반영
    """

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 시간대 생성 (5분봉)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)

    # 랜덤 가격 시뮬레이션 (기본 추세 + 노이즈)
    base_price = 42000
    trend = np.linspace(-500, 800, n)  # 장기 추세 성분
    noise = np.random.normal(0, 50, size=n).cumsum()  # 단기 노이즈
    close = base_price + trend + noise

    # OHLC 계산
    open_ = close + np.random.normal(0, 10, size=n)
    high = np.maximum(open_, close) + np.random.uniform(5, 20, size=n)
    low = np.minimum(open_, close) - np.random.uniform(5, 20, size=n)

    # 거래량, 감정/레짐 점수 등 추가 필드
    volume = np.abs(np.random.normal(1.5e6, 5e5, size=n))
    sentiment_score = np.random.uniform(-1, 1, size=n)
    regime_score = np.random.uniform(0, 1, size=n)
    volatility = np.random.uniform(0, 5, size=n)

    # 시나리오 분석용 보조 필드
    close_drop = np.random.uniform(0, 10, size=n)
    volume_spike = np.random.uniform(1e6, 3e6, size=n)

    # 최종 DataFrame 구성
    df = pd.DataFrame({
        "datetime": date_range,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price": close,  # AuroraQ 전용 핵심 필드 (모든 전략 호환성)
        "volume": volume,
        "sentiment_score": sentiment_score,
        "regime_score": regime_score,
        "close_drop": close_drop,
        "volume_spike": volume_spike,
        "volatility": volatility
    })

    # CSV 저장
    df.to_csv(output_path, index=False)
    print(f"✅ 1번 방식 호환 더미 merged_data.csv 생성 완료: {output_path} (총 {len(df)} 행)")

if __name__ == "__main__":
    generate_dummy_merged_data()
