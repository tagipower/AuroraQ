# utils/price_data_sanitizer.py

def ensure_price_data_fields(price_data, regime_predictor):
    """
    PPO 및 전략 환경에서 기대하는 필수 필드를 누락 없이 보완합니다.
    필드 누락 시 기본값을 할당하여 KeyError를 방지하고,
    PPO 상태 입력의 일관성을 보장합니다.
    """
    price_data.setdefault("price", price_data.get("close", [0])[-1])
    price_data.setdefault("sentiment", 0.5)
    price_data.setdefault("regime", "neutral")
    price_data.setdefault("trend", regime_predictor.get_long_term_trend())
    price_data.setdefault("event_flag", 0)  # 기본적으로 이벤트 없음
    return price_data