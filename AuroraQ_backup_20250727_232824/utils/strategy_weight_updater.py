import subprocess
from datetime import datetime, timedelta

class StrategyWeightUpdater:
    """
    일정 시간 간격으로 update_strategy_weights.py를 자동 실행하는 클래스
    """

    def __init__(self, interval_minutes=360):
        self.last_update = datetime.min
        self.interval = timedelta(minutes=interval_minutes)

    def maybe_update(self):
        """
        현재 시각이 업데이트 주기 이상 경과했을 경우 업데이트 실행
        """
        now = datetime.now()
        if now - self.last_update >= self.interval:
            print(f"[WeightUpdater] ⏱️ 자동 전략 가중치 업데이트 실행: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                subprocess.run(["python", "core/update_strategy_weights.py"], check=True)
                self.last_update = now
            except subprocess.CalledProcessError as e:
                print(f"[WeightUpdater] ❌ 업데이트 실패: {e}")
