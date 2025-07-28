import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any
from utils.logger import get_logger
from utils.async_file_writer import write_yaml_async, get_async_writer

logger = get_logger("StrategyScoreManager")

# 전략별 메트릭 히스토리
strategy_metrics = defaultdict(list)
log_path = "logs/strategy_score_log.yaml"

# 성과 지표 가중치 설정
METRIC_WEIGHTS = {
    "sharpe": 0.10,      # Sharpe (안정성)
    "win_rate": 0.15,    # 승률
    "roi": 0.35,         # 수익률 (중요)
    "profit_factor": 0.35,  # Profit Factor (중요)
    "reward_score": 0.025,  # 보상 점수
    "sentiment": 0.025,     # 감정 점수
    "mdd": -0.05           # 최대 낙폭 (패널티)
}

def calculate_strategy_score(trades: List[Dict], price_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    전략별 성과 메트릭 계산 (개선 버전)
    
    Args:
        trades: 거래 기록 리스트
        price_data: 가격 데이터 (벤치마크 계산용)
    
    Returns:
        계산된 메트릭 딕셔너리
    """
    # 빈 거래 처리
    if not trades:
        return {
            "roi": 0.0, "sharpe": 0.0, "win_rate": 0.0,
            "mdd": 0.0, "profit_factor": 0.0, "trade_count": 0,
            "baseline_roi": 0.0, "volatility": 0.0
        }

    # PnL 추출 및 검증
    profits = []
    for t in trades:
        pnl = t.get("pnl", 0)
        if pnl is not None and not pd.isna(pnl):
            profits.append(float(pnl))
    
    if not profits:
        return {
            "roi": 0.0, "sharpe": 0.0, "win_rate": 0.0,
            "mdd": 0.0, "profit_factor": 0.0, "trade_count": len(trades),
            "baseline_roi": 0.0, "volatility": 0.0
        }

    profits = np.array(profits)
    
    # 1. ROI: 복리 수익률 계산
    # 각 거래의 수익률을 1+r 형태로 변환하여 누적곱
    roi_cumulative = (np.prod(1 + profits) - 1) * 100
    
    # 2. 승률
    wins = np.sum(profits > 0)
    losses = np.sum(profits < 0)
    win_rate = wins / len(profits) if len(profits) > 0 else 0.0
    
    # 3. Sharpe Ratio (개선된 버전)
    mean_profit = np.mean(profits)
    std_profit = np.std(profits, ddof=1) if len(profits) > 1 else 0.001
    
    # 연간화 factor (일 거래 기준)
    annualization_factor = np.sqrt(252)
    
    # 표준편차가 매우 작을 때 처리
    if std_profit < 0.001:
        std_profit = 0.001
    
    # Sharpe 계산 및 스케일링
    raw_sharpe = (mean_profit / std_profit) * annualization_factor
    # tanh 함수로 -1~1 범위로 부드럽게 스케일링
    sharpe = np.tanh(raw_sharpe / 2)
    
    # 4. Profit Factor
    total_gains = np.sum(profits[profits > 0])
    total_losses = abs(np.sum(profits[profits < 0]))
    
    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = total_gains  # 손실 없음
    else:
        profit_factor = 0.0
    
    # 5. 최대 낙폭 (MDD) - 개선된 계산
    if len(profits) > 0:
        equity_curve = np.cumsum(profits)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / (running_max + 1e-8)  # 0으로 나누기 방지
        mdd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    else:
        mdd = 0.0
    
    # 6. 벤치마크 및 변동성 계산
    baseline_roi, volatility = calculate_benchmark_metrics(price_data)
    
    return {
        "roi": round(roi_cumulative, 4),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "mdd": round(mdd, 4),
        "profit_factor": round(profit_factor, 4),
        "trade_count": len(trades),
        "win_count": int(wins),
        "loss_count": int(losses),
        "baseline_roi": round(baseline_roi, 4),
        "volatility": round(volatility, 4)
    }

def calculate_benchmark_metrics(price_data: Optional[pd.DataFrame]) -> tuple:
    """벤치마크 ROI와 변동성 계산"""
    baseline_roi, volatility = 0.0, 0.0
    
    if isinstance(price_data, pd.DataFrame) and not price_data.empty and "close" in price_data.columns:
        try:
            # 시작/종료 가격
            start_price = price_data["close"].iloc[0]
            end_price = price_data["close"].iloc[-1]
            
            # Buy & Hold 수익률
            if start_price > 0:
                baseline_roi = ((end_price - start_price) / start_price) * 100
            
            # 일일 변동성 (표준편차)
            returns = price_data["close"].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)  # 연간화
                
        except Exception as e:
            logger.debug(f"벤치마크 계산 중 오류: {e}")
    
    return baseline_roi, volatility

def calculate_total_score(metrics: Dict[str, float]) -> float:
    """
    복합 점수 계산 (가중치 기반)
    
    Args:
        metrics: 성과 지표 딕셔너리
        
    Returns:
        종합 점수 (0~1 범위로 정규화)
    """
    # 메트릭 추출
    sharpe = metrics.get("sharpe", 0)
    roi = metrics.get("roi", 0)
    win_rate = metrics.get("win_rate", 0)
    mdd = metrics.get("mdd", 0)
    profit_factor = min(metrics.get("profit_factor", 0), 10)  # 극단값 제한
    sentiment = metrics.get("sentiment_score", 0)
    reward_score = metrics.get("reward_shaping_score", 0)
    
    # ROI를 0~1 범위로 스케일링 (100% = 1.0)
    roi_scaled = np.tanh(roi / 100)
    
    # Profit Factor를 0~1 범위로 스케일링
    pf_scaled = np.tanh((profit_factor - 1) / 2) if profit_factor > 0 else 0
    
    # 가중 합계
    total_score = (
        METRIC_WEIGHTS["sharpe"] * sharpe +
        METRIC_WEIGHTS["win_rate"] * win_rate +
        METRIC_WEIGHTS["roi"] * roi_scaled +
        METRIC_WEIGHTS["profit_factor"] * pf_scaled +
        METRIC_WEIGHTS["reward_score"] * reward_score +
        METRIC_WEIGHTS["sentiment"] * sentiment +
        METRIC_WEIGHTS["mdd"] * mdd
    )
    
    # 0~1 범위로 클리핑
    total_score = max(0, min(1, total_score))
    
    return round(total_score, 4)

def update_strategy_metrics(
    strategy_name: str, 
    metrics: Dict[str, Any], 
    price_data: Optional[pd.DataFrame] = None
) -> float:
    """
    전략별 메트릭 업데이트 및 로깅
    
    Args:
        strategy_name: 전략 이름
        metrics: 메트릭 정보 (trades 포함)
        price_data: 가격 데이터
        
    Returns:
        종합 점수
    """
    try:
        # trades 추출 및 검증
        trades = metrics.get("trades", [])
        if not isinstance(trades, list):
            trades = []
        
        # 성과 지표 계산
        stats = calculate_strategy_score(trades, price_data)
        metrics.update(stats)
        
        # 종합 점수 계산
        total_score = calculate_total_score(metrics)
        
        # 로그 엔트리 생성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "total_score": total_score
        }
        
        # 메모리 저장
        strategy_metrics[strategy_name].append(log_entry)
        
        # 파일 저장 - 비동기 쓰기 사용
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_yaml_async(log_path, [log_entry], mode='a')
        
        # 로깅
        logger.info(
            f"[Metrics] {strategy_name} → "
            f"ROI={stats.get('roi', 0):.2f}%, "
            f"Sharpe={stats.get('sharpe', 0):.3f}, "
            f"WinRate={stats.get('win_rate', 0):.2%}, "
            f"PF={stats.get('profit_factor', 0):.2f}, "
            f"MDD={stats.get('mdd', 0):.2%}, "
            f"Trades={stats.get('trade_count', 0)}, "
            f"Score={total_score:.3f}"
        )
        
        return total_score
        
    except Exception as e:
        logger.error(f"[Metrics] {strategy_name} 메트릭 업데이트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def load_score_history(strategy_name: str, limit: int = 10) -> List[Dict]:
    """전략별 점수 히스토리 조회"""
    return strategy_metrics.get(strategy_name, [])[-limit:]

def get_average_score(strategy_name: str, limit: int = 10) -> float:
    """최근 N개 평균 점수 계산"""
    scores = load_score_history(strategy_name, limit)
    if not scores:
        return 0.0
    
    total_scores = [entry.get("total_score", 0) for entry in scores]
    return round(sum(total_scores) / len(total_scores), 4)

def get_all_current_scores() -> Dict[str, float]:
    """모든 전략의 현재 점수 반환"""
    current_scores = {}
    
    for strategy, logs in strategy_metrics.items():
        if logs:
            current_scores[strategy] = logs[-1].get("total_score", 0)
    
    return current_scores

def get_strategy_summary(strategy_name: str) -> Dict[str, Any]:
    """전략 상세 요약 정보"""
    history = strategy_metrics.get(strategy_name, [])
    if not history:
        return {"error": "No data available"}
    
    latest = history[-1]
    metrics = latest.get("metrics", {})
    
    return {
        "strategy": strategy_name,
        "last_updated": latest.get("timestamp"),
        "current_score": latest.get("total_score"),
        "average_score": get_average_score(strategy_name),
        "performance": {
            "roi": f"{metrics.get('roi', 0):.2f}%",
            "sharpe": metrics.get('sharpe', 0),
            "win_rate": f"{metrics.get('win_rate', 0):.2%}",
            "profit_factor": metrics.get('profit_factor', 0),
            "mdd": f"{metrics.get('mdd', 0):.2%}",
            "trades": metrics.get('trade_count', 0)
        },
        "benchmark": {
            "baseline_roi": f"{metrics.get('baseline_roi', 0):.2f}%",
            "volatility": f"{metrics.get('volatility', 0):.2%}"
        }
    }

def reset_strategy_metrics(strategy_name: Optional[str] = None):
    """전략 메트릭 초기화"""
    if strategy_name:
        if strategy_name in strategy_metrics:
            strategy_metrics[strategy_name].clear()
            logger.info(f"[Metrics] {strategy_name} 메트릭 초기화")
    else:
        strategy_metrics.clear()
        logger.info("[Metrics] 모든 전략 메트릭 초기화")

# 메트릭 검증 함수
def validate_metrics(metrics: Dict[str, float]) -> bool:
    """메트릭 유효성 검증"""
    required_fields = ["roi", "sharpe", "win_rate", "mdd", "profit_factor"]
    
    for field in required_fields:
        if field not in metrics:
            return False
    
    # 범위 검증
    if not -1 <= metrics["sharpe"] <= 1:
        return False
    if not 0 <= metrics["win_rate"] <= 1:
        return False
    if metrics["mdd"] < 0 or metrics["mdd"] > 1:
        return False
    
    return True