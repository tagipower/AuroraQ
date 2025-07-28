import time
import pandas as pd
import json
from utils.logger import get_logger
from utils.price_buffer import get_latest_price_data
from sentiment.sentiment_event_manager import get_sentiment_delta
from core.risk_manager import trigger_emergency_exit
from core.position_tracker_file import get_current_position
from utils.telegram_notifier import send_telegram_message  # ✅ 텔레그램 연동

logger = get_logger("PositionMonitor")

# ⚙️ 감시 기준 설정
SENTIMENT_THRESHOLD = 0.35
VOLUME_SPIKE_MULTIPLIER = 3
VOLATILITY_THRESHOLD = 0.05
MONITOR_INTERVAL = 30  # 초

def is_aligned_with_position(price_data: pd.DataFrame) -> bool:
    position = get_current_position()
    if not position:
        return False
    last = price_data.iloc[-1]
    price_change = (last["close"] - last["open"]) / last["open"]
    if position == "long" and price_change > 0:
        return True
    if position == "short" and price_change < 0:
        return True
    return False

def detect_sentiment_spike():
    delta = get_sentiment_delta()
    position = get_current_position()
    if delta is None or position is None:
        return False, delta
    if abs(delta) >= SENTIMENT_THRESHOLD:
        if (position == "long" and delta > 0) or (position == "short" and delta < 0):
            logger.info(f"💚 감정 급변이지만 포지션 방향과 일치 (ΔSentiment: {delta:.3f})")
            return False, delta
        logger.warning(f"[감정 급변 감지] ΔSentiment: {delta:.3f}")
        return True, delta
    return False, delta

def detect_volume_spike(price_data: pd.DataFrame) -> bool:
    recent_volume = price_data["volume"].iloc[-1]
    avg_volume = price_data["volume"].rolling(window=20).mean().iloc[-1]
    if recent_volume > avg_volume * VOLUME_SPIKE_MULTIPLIER:
        if is_aligned_with_position(price_data):
            logger.info("📊 이상 거래량이지만 포지션 방향과 일치")
            return False
        logger.warning(f"⚠️ 이상 거래량 감지: {recent_volume:.2f} > 평균 {avg_volume:.2f}")
        return True
    return False

def detect_candle_anomaly(price_data: pd.DataFrame) -> bool:
    last = price_data.iloc[-1]
    high, low, close = last["high"], last["low"], last["close"]
    if (high - low) / close > VOLATILITY_THRESHOLD:
        if is_aligned_with_position(price_data):
            logger.info("📈 변동성 급등이지만 포지션 방향과 일치")
            return False
        logger.warning(f"⚠️ 변동성 급등 감지: 고저차 {(high - low):.2f}")
        return True
    return False

def save_event_log(event_data: dict):
    try:
        with open("logs/event_monitor_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")
    except Exception as e:
        logger.error(f"이벤트 로그 저장 실패: {e}")

def send_alert(event_data: dict):
    try:
        message = (
            f"🚨 포지션 위험 감지!\n"
            f"🕒 {event_data['timestamp']}\n"
            f"📈 가격: {event_data['price']:.2f}\n"
            f"📌 포지션: {event_data['position']}\n"
            f"💥 Δ감정: {event_data['sentiment_delta']:.3f}\n"
            f"🔍 Flags: {json.dumps(event_data['flags'])}"
        )
        send_telegram_message(message)
    except Exception as e:
        logger.error(f"⚠️ 텔레그램 알림 실패: {e}")

def monitor_active_position():
    logger.info("📡 position_monitor 루프 시작")
    while True:
        try:
            price_data = get_latest_price_data()
            if price_data is None or len(price_data) < 20:
                logger.warning("📉 가격 데이터 부족")
                time.sleep(MONITOR_INTERVAL)
                continue

            sentiment_flag, delta = detect_sentiment_spike()
            volume_flag = detect_volume_spike(price_data)
            volatility_flag = detect_candle_anomaly(price_data)

            flags = {
                "sentiment_spike": sentiment_flag,
                "volume_spike": volume_flag,
                "volatility_spike": volatility_flag,
            }

            position = get_current_position()
            last_price = price_data.iloc[-1]["close"]
            event_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "position": position,
                "price": last_price,
                "sentiment_delta": delta,
                "flags": flags
            }

            save_event_log(event_data)

            if any(flags.values()):
                logger.error(f"🚨 위험 이벤트 발생: {flags}")
                send_alert(event_data)  # ✅ 텔레그램 경고
                trigger_emergency_exit(reason="position_monitor triggered")

            time.sleep(MONITOR_INTERVAL)

        except Exception as e:
            logger.exception(f"📛 모니터링 루프 예외 발생: {e}")
            time.sleep(MONITOR_INTERVAL)

# 🟡 실행 예시
if __name__ == "__main__":
    monitor_active_position()
