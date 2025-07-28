# 📁 utils/market_filter.py

def filter_market_condition(price_data: dict) -> bool:
    """
    감정 점수 및 추세 기반으로 매매 가능 여부 판단.

    조건:
    - 감정 점수가 -0.4 이하이면 부정적 시장 → 매매 금지
    - 5일선이 20일선보다 낮으면 추세 약세 → 매매 금지

    Args:
        price_data (dict): 시세 및 지표 정보 포함된 딕셔너리

    Returns:
        bool: True면 전략 실행 허용, False면 스킵
    """
    sentiment = price_data.get("sentiment", 0)
    short_ma = price_data.get("ma_5")
    long_ma = price_data.get("ma_20")

    # 감정 점수 필터
    if sentiment < -0.4:
        return False

    # 추세 필터
    if short_ma is not None and long_ma is not None:
        if short_ma < long_ma:
            return False

    return True
