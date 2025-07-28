import os
import json
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# ✅ 텔레그램 연동
try:
    from utils.telegram_notifier import send_telegram_message
except ImportError:
    def send_telegram_message(msg): 
        print("[텔레그램 비활성] " + msg)

# ✅ 로깅 설정
logger = logging.getLogger("ExecutionMonitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ✅ 경로 설정
POSITION_STATE_PATH = "data/position_state.json"
ORDER_LOG_PATH = "logs/order_log.csv"

# ✅ 체결 실패 감지 기준
ORDER_TIMEOUT = timedelta(minutes=5)
MAX_FAILURE_THRESHOLD = 3

# ✅ 실패 누적 추적
failure_counter = {}


def load_position_state():
    try:
        with open(POSITION_STATE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        msg = "⚠️ position_state.json 파일이 없습니다."
        logger.warning(msg)
        send_telegram_message(msg)
        return {}
    except Exception as e:
        msg = f"❌ position_state.json 로딩 실패: {e}"
        logger.error(msg)
        send_telegram_message(msg)
        return {}


def load_order_log():
    try:
        return pd.read_csv(ORDER_LOG_PATH)
    except FileNotFoundError:
        msg = "⚠️ order_log.csv 파일이 없습니다."
        logger.warning(msg)
        send_telegram_message(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"❌ order_log.csv 로딩 실패: {e}"
        logger.error(msg)
        send_telegram_message(msg)
        return pd.DataFrame()


def detect_execution_anomalies():
    global failure_counter
    pos_state = load_position_state()
    orders = load_order_log()

    if orders.empty:
        logger.info("ℹ️ 주문 로그가 비어 있습니다.")
        return

    now = datetime.utcnow()
    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")
    pending_orders = orders[(orders["status"] == "PENDING") & (now - orders["timestamp"] > ORDER_TIMEOUT)]

    for _, row in pending_orders.iterrows():
        key = f"{row['strategy']}_{row['symbol']}_{row['action']}"
        failure_counter[key] = failure_counter.get(key, 0) + 1
        msg = f"⏱️ 미체결 주문 감지: {key} ({failure_counter[key]}회)"
        logger.warning(msg)
        send_telegram_message(msg)

        if failure_counter[key] >= MAX_FAILURE_THRESHOLD:
            critical_msg = f"🚨 반복된 주문 실패 감지! [{key}] → 매매 중단 고려 필요"
            logger.error(critical_msg)
            send_telegram_message(critical_msg)

    # ✅ 포지션 정보 누락 확인
    active_symbols = orders[orders["status"] == "FILLED"]["symbol"].unique()
    for symbol in active_symbols:
        if symbol not in pos_state:
            msg = f"❓ 체결된 {symbol} 포지션 정보 누락 감지"
            logger.warning(msg)
            send_telegram_message(msg)


def run_execution_monitor(interval_sec=30):
    logger.info("📡 Execution Monitor 루프 시작")
    send_telegram_message("✅ Execution Monitor 루프 시작됨")
    while True:
        try:
            detect_execution_anomalies()
        except Exception as e:
            err_msg = f"❌ 모니터 루프 예외 발생: {e}"
            logger.exception(err_msg)
            send_telegram_message(err_msg)
        time.sleep(interval_sec)


if __name__ == "__main__":
    run_execution_monitor()
