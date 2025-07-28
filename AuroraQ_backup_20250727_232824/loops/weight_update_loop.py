import os
import sys
import time
import logging
from datetime import datetime

# ✅ 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ✅ 가중치 업데이트 함수 불러오기
from core.update_strategy_weights import update_weights

# ✅ 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeightUpdateLoop")

# ✅ 가중치 업데이트 주기 (초) - 기본 1시간
try:
    UPDATE_INTERVAL = int(os.getenv("WEIGHT_UPDATE_INTERVAL", 3600))
except ValueError:
    UPDATE_INTERVAL = 3600
    logger.warning("⚠️ WEIGHT_UPDATE_INTERVAL 환경변수 형식 오류. 기본값 3600초로 설정됨.")

def run_weight_update_loop():
    logger.info("🔁 [전략 가중치 자동 보정 루프 시작]")

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"⏰ {now} - 전략 가중치 업데이트 시작")
            update_weights()
            logger.info("✅ 전략 가중치 업데이트 완료")
        except Exception as e:
            logger.exception(f"❗ 가중치 업데이트 중 예외 발생: {e}")

        logger.info(f"💤 {UPDATE_INTERVAL}초 대기 후 다음 업데이트 수행")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    run_weight_update_loop()
