# 📁 run_loop.py
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime

# ✅ 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.strategy_selector import StrategySelector
from core.strategy_score_manager import update_strategy_metrics, calculate_strategy_score
from core.score_feedback import reward_to_score
from utils.strategy_score_logger import log_strategy_score
from core.risk_manager import evaluate_risk, allocate_capital, adjust_leverage, should_cut_loss_or_take_profit
from core.regime_predictor import RegimePredictor
from utils.price_buffer import get_latest_price
from utils.logger import get_logger
from utils.telegram_notifier import send_telegram_message
from sentiment.sentiment_event_manager import SentimentEventManager
from config.env_loader import load_env
from core.order_simulator import simulate_trade, simulate_exit
from config.rule_param_loader import get_rule_params
from report.strategy_param_logger import log_strategy_params
from report.html_report_generator import generate_html_report

logger = get_logger("RunLoop")
env_config = load_env()

REAL_TRADE = os.getenv("REAL_TRADE", "False") == "True"
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", 300)) if os.getenv("LOOP_INTERVAL") else 300

loop_durations = deque(maxlen=20)
strategy_health = defaultdict(lambda: {"fail": 0, "success": 0, "nosignal": 0})
predictor = RegimePredictor()
sentiment_manager = SentimentEventManager()
selector = StrategySelector()

# 상태 변수
total_capital = 100000
position_status = None
entry_price = None
entry_time = None
entry_signal = None

strategy_logged = set()
strategy_report_logged = set()

def run_live_loop():
    global position_status, entry_price, entry_time, entry_signal

    logger.info("▶️ [LIVE LOOP] 실시간 자동매매 루프 시작")

    while True:
        loop_start = time.time()
        try:
            if sentiment_manager.get_upcoming_events(within_minutes=30):
                logger.info("🔕 예정된 이벤트로 인한 매매 회피")
                time.sleep(60)
                continue

            price_data = get_latest_price()
            if not price_data or "close" not in price_data:
                logger.warning("⚠️ 가격 데이터 누락 또는 'close' 없음")
                time.sleep(30)
                continue

            price_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            sentiment_score = sentiment_manager.get_sentiment_score() or 0.0
            prev_sentiment = sentiment_manager.get_previous_sentiment() or sentiment_score
            sentiment_delta = sentiment_score - prev_sentiment
            price_data["sentiment"] = sentiment_score

            logger.info(f"[감정 점수] 현재: {sentiment_score:.2f} / Δ: {sentiment_delta:.2f}")

            try:
                regime = predictor.predict(price_data, sentiment_score=sentiment_score)
                price_data["regime"] = regime
                logger.info(f"📉 예측된 시장 레짐: {regime}")
            except Exception as e:
                logger.warning(f"❗ 레짐 예측 실패: {e}")
                continue

            try:
                strategy_result = selector.select(price_data)
            except Exception as e:
                logger.warning(f"❗ 전략 선택 실패: {e}")
                strategy_result = None

            if not strategy_result:
                logger.warning("⚠️ 유효한 전략 없음 → 루프 스킵")
                time.sleep(30)
                continue

            strategy_name = strategy_result.get("strategy")
            strategy_obj = strategy_result.get("strategy_object")
            signal = strategy_result.get("signal")

            if strategy_name and strategy_name not in strategy_logged:
                params = get_rule_params(strategy_name)
                log_strategy_params(strategy_name, params)
                strategy_logged.add(strategy_name)

            if not strategy_obj or not hasattr(strategy_obj, "execute"):
                logger.warning(f"❗ 전략 '{strategy_name}' 실행 메서드 누락")
                continue

            if not signal:
                strategy_health[strategy_name]["nosignal"] += 1
                logger.info(f"💤 전략 '{strategy_name}' 신호 없음 → 매매 생략")
                continue
            else:
                strategy_health[strategy_name]["success"] += 1
                logger.info(f"🎯 전략 선택 완료: {strategy_name}")

            is_safe = evaluate_risk(
                price_df=price_data,
                position_status=position_status,
                strategy_name=strategy_name,
                regime=regime,
                sentiment_score=sentiment_score,
                sentiment_delta=sentiment_delta,
                current_price=price_data["close"][-1] if isinstance(price_data["close"], list) else price_data["close"]
            )
            if not is_safe:
                logger.info(f"🚫 리스크 필터링 차단: {strategy_name}")
                continue

            current_price = price_data["close"]
            if position_status in ["LONG", "SHORT"] and entry_price and current_price:
                result = should_cut_loss_or_take_profit(entry_price, current_price, strategy_name)
                if result in ["STOP", "TAKE"]:
                    logger.info(f"📤 포지션 청산 조건({result}) 충족")
                    simulate_exit(entry_price, current_price, signal, strategy_name)
                    position_status = None
                    entry_price = None
                    entry_signal = None
                    entry_time = None
                    continue

            if REAL_TRADE:
                action = strategy_obj.execute(signal)
                entry_price = current_price
                position_status = getattr(action, "direction", None)
                entry_signal = signal
                entry_time = datetime.now()
                logger.info("✅ 실거래 주문 완료")
            else:
                simulate_trade(current_price, signal, strategy_name)
                entry_price = current_price
                position_status = signal.get("position")
                entry_signal = signal
                entry_time = datetime.now()
                logger.info(f"🧪 모의 주문: {signal}")

            # ✅ 전략 평가 (.score 우선 → 없으면 fallback)
            try:
                if hasattr(strategy_obj, "score"):
                    score = strategy_obj.score(price_data, signal)
                    log_strategy_score(strategy_name, {"custom_score": True}, score)
                    logger.info(f"📊 커스텀 점수 기록 완료: {strategy_name} → {score:.4f}")
                elif hasattr(strategy_obj, "run_with_reward"):
                    _, reward = strategy_obj.run_with_reward(price_data)
                    score = reward_to_score(reward, is_real_trade=True)
                    update_strategy_metrics(strategy_name, {"reward": reward})
                    log_strategy_score(strategy_name, {"reward": reward}, score)
                    logger.info(f"📊 보상 점수 기록 완료: {strategy_name} → {score:.4f}")
                elif hasattr(strategy_obj, "evaluate_result"):
                    metrics = strategy_obj.evaluate_result(signal, price_data)
                    if metrics:
                        update_strategy_metrics(strategy_name, metrics)
                        score = calculate_strategy_score(metrics)
                        log_strategy_score(strategy_name, metrics, score)
                        logger.info(f"📊 평가 점수 기록 완료: {strategy_name} → {score:.4f}")
                else:
                    logger.warning(f"⚠️ 전략 '{strategy_name}' 평가 함수 없음")
            except Exception as e:
                logger.warning(f"❗ 전략 평가 중 오류 발생: {e}")

            # ✅ 리포트 생성 (하루 1회)
            report_key = f"{strategy_name}_{entry_time.date()}"
            if report_key not in strategy_report_logged:
                result = {
                    "strategy": strategy_name,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A",
                    "pnl_pct": 0.0,
                    "leverage": signal.get("leverage", 1),
                    "position": position_status,
                    "signal": signal,
                    "sentiment": sentiment_score,
                    "regime": regime
                }
                generate_html_report(strategy_name, result)
                strategy_report_logged.add(report_key)

        except Exception as loop_error:
            logger.exception(f"🔥 루프 예외 발생: {loop_error}")
            send_telegram_message(f"[루프 오류] {loop_error}")

        finally:
            loop_end = time.time()
            elapsed = loop_end - loop_start
            loop_durations.append(elapsed)
            avg_duration = sum(loop_durations) / len(loop_durations) if loop_durations else 0
            sleep_time = max(0, LOOP_INTERVAL - elapsed)
            logger.info(f"⏱ 루프 시간: {elapsed:.2f}s | 평균: {avg_duration:.2f}s | 대기: {sleep_time:.2f}s")
            time.sleep(sleep_time)

if __name__ == "__main__":
    run_live_loop()
