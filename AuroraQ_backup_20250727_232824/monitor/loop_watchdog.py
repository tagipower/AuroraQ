import os
import time
import subprocess
import psutil
from utils.telegram_notifier import send_telegram_message

# ✅ 감시 대상 루프
MONITORED_SCRIPTS = {
    "run_loop.py": "python loops/run_loop.py",
    "execution_monitor.py": "python monitor/execution_monitor.py",
    "position_monitor.py": "python monitor/position_monitor.py",
}

CHECK_INTERVAL = 30  # 감시 주기 (초)
RESTART_COOLDOWN = 60  # 같은 스크립트 재시작 쿨다운 (초)
LOG_PATH = "logs/watchdog.log"

# ✅ 최근 재시작 시각 기록용 딕셔너리
last_restart_time = {}

# ✅ 로컬 로그 기록 함수
def log_local(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(log_entry + "\n")
    print(log_entry)  # 콘솔에도 출력

# ✅ 프로세스 실행 여부 확인
def is_process_running(script_name):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if script_name in proc.info['cmdline']:
                return True
        except Exception:
            continue
    return False

# ✅ 재시작 로직 (중복 방지 포함)
def restart_script(script_name, command):
    now = time.time()
    if script_name in last_restart_time and now - last_restart_time[script_name] < RESTART_COOLDOWN:
        msg = f"⚠️ [Watchdog] `{script_name}` 최근 재시도 있음 → 재시작 생략"
        log_local(msg)
        return

    try:
        subprocess.Popen(command, shell=True)
        msg = f"🔁 [Watchdog] `{script_name}` 감지되지 않아 재시작함."
        send_telegram_message(msg)
        log_local(msg)
        last_restart_time[script_name] = now
    except Exception as e:
        err_msg = f"❌ [Watchdog] `{script_name}` 재시작 실패: {e}"
        send_telegram_message(err_msg)
        log_local(err_msg)

# ✅ 메인 감시 루프
def monitor_loops():
    startup_msg = "🟢 [Watchdog] 루프 감시 시작됨."
    send_telegram_message(startup_msg)
    log_local(startup_msg)

    while True:
        for script, cmd in MONITORED_SCRIPTS.items():
            if not is_process_running(script):
                restart_script(script, cmd)
        time.sleep(CHECK_INTERVAL)

# ✅ 실행 진입점
if __name__ == "__main__":
    try:
        monitor_loops()
    except Exception as e:
        err_msg = f"🚨 [Watchdog] 감시 루프 자체 오류 발생: {e}"
        send_telegram_message(err_msg)
        log_local(err_msg)
