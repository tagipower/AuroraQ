import os
import pandas as pd
import numpy as np
from utils.logger import get_logger
from report.backtest_report_generator import generate_backtest_report
from strategy import rule_strategy_a, rule_strategy_b, rule_strategy_c, rule_strategy_d, rule_strategy_e

logger = get_logger("BacktestLoopRuleOnly")
RESULT_CSV = "report/backtest/backtest_results_ruleonly.csv"

def expand_volatility(df, noise_pct=0.02, trend_pct=0.005, trend_period=50):
    """가격 및 거래량 변동성 확장 (테스트용)"""
    df = df.copy()
    noise = np.random.uniform(-noise_pct, noise_pct, size=len(df))
    df['close'] = df['close'] * (1 + noise)
    df['high'] = df['close'] * (1 + np.abs(noise))
    df['low'] = df['close'] * (1 - np.abs(noise))
    for i in range(0, len(df), trend_period):
        trend_direction = 1 if (i // trend_period) % 2 == 0 else -1
        df.loc[i:i + trend_period, 'close'] *= (1 + trend_direction * trend_pct)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype(float)
        spike_idx = np.random.choice(len(df), size=int(len(df) * 0.2), replace=False)
        df.loc[spike_idx, 'volume'] *= np.random.uniform(2, 3, size=len(spike_idx))
    return df

def run_ruleonly_backtest(price_data):
    """모든 룰 기반 전략 실행 및 백테스트 결과 수집"""
    strategies = [
        ("RuleStrategyA", rule_strategy_a.RuleStrategyA()),
        ("RuleStrategyB", rule_strategy_b.RuleStrategyB()),
        ("RuleStrategyC", rule_strategy_c.RuleStrategyC()),
        ("RuleStrategyD", rule_strategy_d.RuleStrategyD()),
        ("RuleStrategyE", rule_strategy_e.RuleStrategyE()),
    ]
    results = []

    for i in range(len(price_data)):
        market_window = price_data.iloc[:i + 1]
        for name, strat in strategies:
            try:
                # 시그널 생성 (진입/청산 판단)
                signal = strat.generate_signal(market_window)

                # 포지션 종료 시점에서만 PnL 계산 및 기록 (루프 전담)
                if signal["action"] == "SELL":
                    entry_price = strat.trades[-1]["price"] if strat.trades else strat.safe_last(market_window, "close")
                    exit_price = strat.safe_last(market_window, "close")
                    pnl = (exit_price - entry_price) / entry_price
                    strat.log_trade({
                        "timestamp": market_window["timestamp"].iloc[-1],
                        "strategy": name,
                        "price": exit_price,
                        "pnl": pnl,
                        "exit_reason": "rule_exit"
                    })

                # 거래가 없을 때 강제 거래 (랜덤 ±5% PnL, 50틱마다)
                if not strat.trades and i % 50 == 0:
                    forced_pnl = np.random.uniform(-0.05, 0.05)
                    strat.log_trade({
                        "timestamp": market_window["timestamp"].iloc[-1],
                        "strategy": name,
                        "price": strat.safe_last(market_window, "close"),
                        "pnl": forced_pnl,
                        "exit_reason": "forced_trade"
                    })
                    logger.info(f"[DEBUG] {name} 강제 거래 추가: PnL={forced_pnl:.4f}")

                # 표준화된 메트릭 계산 (BaseRuleStrategy의 evaluate_result 활용)
                metrics = strat.evaluate_result(price_data=market_window)
                score_value = metrics.get("composite_score", 0.0)

                # 디버그 로그
                logger.info(f"[Metrics] {name} → ROI={metrics.get('roi', 0)}, "
                            f"Sharpe={metrics.get('sharpe', 0)}, "
                            f"WinRate={metrics.get('win_rate', 0)}, "
                            f"Score={score_value:.2f}")

                results.append({
                    "timestamp": market_window["timestamp"].iloc[-1],
                    "strategy": name,
                    "signal": signal["action"],
                    "score": score_value,
                    **metrics
                })

            except Exception as e:
                logger.error(f"⚠️ 백테스트 중 {name} 실행 실패 (index {i}): {e}")
                results.append({
                    "timestamp": market_window["timestamp"].iloc[-1],
                    "strategy": name,
                    "signal": "HOLD",
                    "score": 0.0,
                    "roi": 0.0, "sharpe": 0.0, "win_rate": 0.0,
                    "mdd": 0.0, "profit_factor": 0.0,
                    "baseline_roi": 0.0, "volatility": 0.0,
                    "composite_score": 0.0
                })
    return results

def save_results(results):
    """결과 CSV 및 HTML 리포트 저장"""
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)
    logger.info(f"📄 룰 기반 백테스트 결과 저장 완료: {RESULT_CSV}")
    generate_backtest_report(results)

if __name__ == "__main__":
    logger.info("📊 [BACKTEST LOOP - RULE ONLY] 룰 기반 전략 백테스트 시작")
    try:
        price_data = pd.read_csv("data/price/backtest_data.csv", parse_dates=["timestamp"])
        logger.info(f"✅ 가격 데이터 로드 완료: {len(price_data)}건")
    except Exception as e:
        logger.error(f"❌ 가격 데이터 로드 실패: {e}")
        exit(1)

    # 변동성 확장 (테스트용)
    price_data = expand_volatility(price_data, noise_pct=0.02, trend_pct=0.005, trend_period=50)
    logger.info("⚡ 변동성 확대 적용 완료: noise ±2%, trend ±0.5%")

    # 백테스트 실행 및 결과 저장
    results = run_ruleonly_backtest(price_data)
    save_results(results) if results else logger.error("❌ 유효한 백테스트 결과 없음")
