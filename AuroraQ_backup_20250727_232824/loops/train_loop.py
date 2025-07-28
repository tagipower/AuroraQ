# 📁 loops/train_loop.py

import time
import os
import traceback
from core.train_ppo import train_ppo
from utils.telegram_notifier import send_telegram_message

LOOP_INTERVAL_HOURS = int(os.getenv("TRAIN_LOOP_INTERVAL_HOURS", 72))
MODEL_LOCK_PATH = os.getenv("MODEL_LOCK_PATH", "models/ppo_model.lock")
RUN_ONCE = os.getenv("RUN_ONCE", "False").lower() == "true"

def wait_for_previous_training():
    return not os.path.exists(MODEL_LOCK_PATH)

def run_train_loop():
    send_telegram_message("🧠 PPO 학습 루프 시작")

    while True:
        if wait_for_previous_training():
            try:
                train_ppo()
            except Exception as e:
                error_msg = traceback.format_exc()
                send_telegram_message(f"❌ 학습 루프 예외 발생:\n{error_msg}")
            finally:
                if os.path.exists(MODEL_LOCK_PATH):
                    os.remove(MODEL_LOCK_PATH)
        else:
            print("🛑 락파일 존재 → 다음 주기까지 대기")

        if RUN_ONCE:
            break

        sleep_seconds = LOOP_INTERVAL_HOURS * 3600
        print(f"🕒 {LOOP_INTERVAL_HOURS}시간 후 재시도")
        send_telegram_message(f"⏱ PPO 학습 루프 대기 중: {LOOP_INTERVAL_HOURS}시간 후 재시작")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_train_loop()
