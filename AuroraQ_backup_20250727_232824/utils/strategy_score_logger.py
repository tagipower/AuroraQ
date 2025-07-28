# 📁 utils/strategy_score_logger.py

import csv
from datetime import datetime
from pathlib import Path

# 저장 경로 설정
LOG_PATH = Path("report/strategy_score_log.csv")

def log_strategy_score(strategy_name: str, metrics: dict, score: float):
    """
    전략별 평가 점수와 메트릭을 CSV 파일에 누적 기록합니다.

    Args:
        strategy_name (str): 전략 이름
        metrics (dict): 전략 평가 메트릭 (예: win_rate, avg_profit 등)
        score (float): calculate_strategy_score()로 계산된 최종 점수
    """
    # 디렉토리 생성 (없을 경우 자동 생성)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not LOG_PATH.exists()

    # CSV 열 정의
    fieldnames = [
        "timestamp", "strategy", "score",
        "win_rate", "avg_profit", "max_drawdown",
        "stability", "recent_performance"
    ]

    # metrics 유효성 검사
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics는 dict여야 합니다. 현재 타입: {type(metrics)}")

    # CSV 기록
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 새로운 파일일 경우 헤더 작성
        if is_new_file:
            writer.writerow(fieldnames)

        # 한 줄 데이터 기록
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategy_name,
            round(score, 4),
            metrics.get("win_rate", 0.0),
            metrics.get("avg_profit", 0.0),
            metrics.get("max_drawdown", 0.0),
            metrics.get("stability", 0.0),
            metrics.get("recent_performance", 0.0),
        ])