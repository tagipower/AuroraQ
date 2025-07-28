import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from utils.logger import get_logger
from core.strategy_selector import StrategySelector
from core.path_config import get_log_path, get_data_path
from core.risk_manager import integrated_risk_manager, risk_monitor
from sentiment.sentiment_score import get_sentiment_score_by_date, get_sentiment_score_range
from report.backtest_report_generator import generate_backtest_report

logger = get_logger("BacktestLoop")

# 경로 설정 (path_config 사용)
RESULT_CSV = "report/backtest/backtest_results.csv"
PPO_BUFFER = str(get_log_path("ppo_training_buffer"))

def apply_enhanced_risk_management(
    market_window: pd.DataFrame, 
    strategy_name: str, 
    original_signal: str, 
    sentiment_score: float
) -> Tuple[str, str]:
    """향상된 리스크 관리 적용"""
    try:
        # 가격 데이터 컬럼명 매핑 (backtest_loop 형식 → risk_manager 형식)
        price_data = market_window.copy()
        if 'close' not in price_data.columns and 'Close' in price_data.columns:
            price_data['close'] = price_data['Close']
        if 'close' not in price_data.columns and 'price' in price_data.columns:
            price_data['close'] = price_data['price']
        
        # 최소 데이터 확인
        if len(price_data) < 20 or 'close' not in price_data.columns:
            return original_signal, "Insufficient data for risk check"
        
        # 시장 체제 감지
        regime, confidence = integrated_risk_manager.risk_filter.detect_market_regime(price_data)
        
        # 극단적인 시장 상황에서 거래 제한
        if regime.value == "crisis" and confidence > 0.8:
            return "HOLD", f"Crisis regime detected (confidence: {confidence:.2f})"
        
        # 고변동성 시장에서 신호 강도 조정
        if regime.value == "high_volatility" and confidence > 0.6:
            # 30% 확률로 신호를 HOLD로 변경
            import random
            if random.random() < 0.3:
                return "HOLD", f"High volatility regime (confidence: {confidence:.2f})"
        
        # 감정 극값에서 역방향 신호 제한
        if sentiment_score < 0.2 and original_signal == "SELL":
            return "HOLD", f"Extreme fear - avoid panic selling (sentiment: {sentiment_score:.2f})"
        
        if sentiment_score > 0.8 and original_signal == "BUY":
            return "HOLD", f"Extreme greed - avoid FOMO buying (sentiment: {sentiment_score:.2f})"
        
        # 기본적으로 원래 신호 유지
        return original_signal, "Risk check passed"
        
    except Exception as e:
        logger.error(f"리스크 관리 적용 오류: {e}")
        # 오류 시 보수적으로 HOLD
        return "HOLD", f"Risk management error: {str(e)}"


def calculate_metrics(trades):
    """Sharpe, ROI, 승률, MDD, Profit Factor 계산"""
    if not trades:
        return {"sharpe": 0, "roi": 0, "win_rate": 0, "mdd": 0, "profit_factor": 0}
    profits = [t["pnl"] for t in trades]
    roi = sum(profits)
    wins = sum(1 for p in profits if p > 0)
    win_rate = wins / len(profits) if profits else 0
    mdd = min([sum(profits[:i + 1]) for i in range(len(profits))], default=0)
    profit_factor = abs(sum(p for p in profits if p > 0) / (sum(p for p in profits if p < 0) or 1))
    sharpe = (roi / (pd.Series(profits).std() or 1)) * (len(profits) ** 0.5)
    return {
        "sharpe": round(sharpe, 4),
        "roi": round(roi, 4),
        "win_rate": round(win_rate, 4),
        "mdd": round(mdd, 4),
        "profit_factor": round(profit_factor, 4)
    }


def save_ppo_buffer(data: Dict[str, Any]) -> None:
    """PPO 학습용 버퍼에 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(PPO_BUFFER), exist_ok=True)
        
        if os.path.exists(PPO_BUFFER):
            with open(PPO_BUFFER, "r", encoding="utf-8") as f:
                buffer = json.load(f)
        else:
            buffer = []
        
        buffer.append(data)
        
        with open(PPO_BUFFER, "w", encoding="utf-8") as f:
            json.dump(buffer, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"PPO buffer updated: {len(buffer)} entries")
        
    except Exception as e:
        logger.error(f"Failed to save PPO buffer: {e}")


def get_sentiment_for_timestamp(timestamp: datetime) -> float:
    """
    특정 타임스탬프에 대한 감정 점수 조회
    
    Args:
        timestamp: 조회할 타임스탬프
        
    Returns:
        감정 점수 (-1.0 ~ 1.0)
    """
    try:
        date_str = timestamp.strftime("%Y-%m-%d")
        score = get_sentiment_score_by_date(date_str)
        return score
    except Exception as e:
        logger.warning(f"Failed to get sentiment for {timestamp}: {e}")
        return 0.0


def run_backtest_loop(
    price_data: pd.DataFrame, 
    sentiment_data: Optional[pd.DataFrame] = None,
    sentiment_file: Optional[str] = None,
    start_index: int = 0,
    max_iterations: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    개선된 백테스트 루프 (감정 점수 통합)
    
    Args:
        price_data: 가격 데이터 DataFrame
        sentiment_data: 감정 데이터 DataFrame (사용 안 함, 호환성용)
        sentiment_file: 감정 점수 파일 경로
        start_index: 시작 인덱스
        max_iterations: 최대 반복 횟수 (None이면 전체)
        
    Returns:
        백테스트 결과 리스트
    """
    logger.info("백테스트 루프 시작")
    
    # 감정 파일 경로 설정
    if sentiment_file is None:
        sentiment_file = str(get_data_path("sentiment"))
    
    # StrategySelector 초기화
    try:
        selector = StrategySelector(sentiment_file=sentiment_file)
        logger.info(f"StrategySelector 초기화 완료: {sentiment_file}")
    except Exception as e:
        logger.error(f"StrategySelector 초기화 실패: {e}")
        return []
    
    results = []
    total_length = len(price_data)
    
    # 반복 범위 설정
    end_index = min(start_index + max_iterations, total_length) if max_iterations else total_length
    
    logger.info(f"백테스트 범위: {start_index} ~ {end_index} (총 {total_length}개 중 {end_index - start_index}개 처리)")

    for i in range(start_index, end_index):
        # 진행률 로깅
        if i % 100 == 0:
            progress = ((i - start_index) / (end_index - start_index)) * 100
            logger.info(f"진행률: {progress:.1f}% ({i}/{end_index})")
        
        # 시장 데이터 윈도우 (누적)
        market_window = price_data.iloc[:i + 1]
        current_timestamp = market_window["timestamp"].iloc[-1]

        try:
            # 1. 일일 리스크 한도 체크
            daily_limit_ok, daily_reason = risk_monitor.check_daily_limits()
            if not daily_limit_ok:
                logger.warning(f"일일 한도 초과: {daily_reason}")
                # HOLD 신호로 처리
                signal_action = "HOLD"
                strat_name = "RISK_LIMITED"
                score = 0
                base_score = 0
                sentiment_score = 0
                regime = "neutral"
                volatility = 0
                trend = "sideways"
                trades = []
            else:
                # 2. 전략 선택 실행
                selection = selector.select(market_window)
                
                # 결과 추출
                strat_name = selection.get("strategy", "UNKNOWN")
                signal_data = selection.get("signal", {})
                original_signal = signal_data.get("action", "HOLD") if isinstance(signal_data, dict) else str(signal_data)
                score = selection.get("score", 0)
                base_score = selection.get("base_score", 0)
                sentiment_score = selection.get("sentiment_score", 0)
                regime = selection.get("regime", "neutral")
                volatility = selection.get("volatility", 0)
                trend = selection.get("trend", "sideways")
                
                # 3. 향상된 리스크 검사 (BUY/SELL 신호인 경우만)
                signal_action = original_signal
                if original_signal in ["BUY", "SELL"]:
                    # 리스크 관리 적용된 신호 조정
                    signal_action, risk_reason = apply_enhanced_risk_management(
                        market_window, strat_name, original_signal, sentiment_score
                    )
                    
                    if signal_action != original_signal:
                        logger.info(f"리스크 관리로 신호 조정: {original_signal} → {signal_action} ({risk_reason})")
                
                # 4. 거래 실행 시 리스크 모니터 업데이트
                if signal_action in ["BUY", "SELL"]:
                    # 모의 PnL 계산 (실제로는 전략에서 계산해야 함)
                    mock_pnl = 0.0  # 추후 실제 PnL로 교체
                    risk_monitor.update_trade(mock_pnl)

            # 전략 객체에서 거래 정보 추출
            strategy_obj = selection.get("strategy_object")
            trades = []
            if strategy_obj and hasattr(strategy_obj, 'trades'):
                trades = strategy_obj.trades
            
            # 성과 지표 계산
            metrics = calculate_metrics(trades)

            # 추가 감정 점수 조회 (검증용)
            external_sentiment = get_sentiment_for_timestamp(current_timestamp)

            # 결과 레코드 생성
            result = {
                "timestamp": current_timestamp,
                "strategy": strat_name,
                "signal_action": signal_action,
                "signal_data": signal_data,
                "base_score": base_score,
                "adjusted_score": score,
                "sentiment_score": sentiment_score,
                "external_sentiment": external_sentiment,
                "regime": regime,
                "volatility": volatility,
                "trend": trend,
                "trade_count": len(trades),
                **metrics
            }
            
            results.append(result)

            # PPO 전략인 경우 버퍼에 저장
            if strat_name == "PPOStrategy":
                save_ppo_buffer({
                    **result,
                    "price_data": {
                        "close": float(market_window["close"].iloc[-1]),
                        "volume": float(market_window["volume"].iloc[-1]) if "volume" in market_window else 0
                    }
                })

        except Exception as e:
            logger.error(f"백테스트 중 전략 실행 실패 (index {i}): {e}", exc_info=True)
            
            # 오류 발생 시 기본 결과 추가
            error_result = {
                "timestamp": current_timestamp,
                "strategy": "ERROR",
                "signal_action": "HOLD",
                "signal_data": {},
                "base_score": 0,
                "adjusted_score": 0,
                "sentiment_score": 0,
                "external_sentiment": 0,
                "regime": "unknown",
                "volatility": 0,
                "trend": "unknown",
                "trade_count": 0,
                "error": str(e),
                **calculate_metrics([])
            }
            results.append(error_result)

    logger.info(f"백테스트 완료: {len(results)}개 결과 생성")
    return results


def save_results(results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    백테스트 결과 저장
    
    Args:
        results: 백테스트 결과 리스트
        output_path: 출력 파일 경로 (None이면 기본 경로)
        
    Returns:
        저장된 파일 경로
    """
    if output_path is None:
        output_path = RESULT_CSV
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # DataFrame으로 변환 및 저장
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        logger.info(f"백테스트 결과 저장 완료: {output_path}")
        logger.info(f"저장된 레코드 수: {len(df)}")
        
        # 요약 통계 로깅
        if len(df) > 0:
            logger.info("결과 요약:")
            logger.info(f"  - 사용된 전략: {df['strategy'].value_counts().to_dict()}")
            logger.info(f"  - 평균 조정 점수: {df['adjusted_score'].mean():.4f}")
            logger.info(f"  - 평균 감정 점수: {df['sentiment_score'].mean():.4f}")
            logger.info(f"  - 평균 ROI: {df['roi'].mean():.4f}")
            logger.info(f"  - 평균 Sharpe: {df['sharpe'].mean():.4f}")
        
        # 백테스트 리포트 생성
        try:
            generate_backtest_report(results)
            logger.info("백테스트 리포트 생성 완료")
        except Exception as e:
            logger.warning(f"백테스트 리포트 생성 실패: {e}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("📊 [BACKTEST LOOP] AuroraQ 통합 전략 백테스트 시작")

    try:
        # 가격 데이터 로드
        price_file = str(get_data_path("backtest_data"))
        if not os.path.exists(price_file):
            # 대체 경로들 시도
            alt_paths = [
                "data/price/backtest_data.csv",
                "data/price/test_backtest_data.csv",
                "data/btc_price_data.csv"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    price_file = alt_path
                    break
            else:
                logger.error("❌ 가격 데이터 파일을 찾을 수 없습니다.")
                logger.info("다음 위치에서 파일을 찾았지만 존재하지 않음:")
                for path in [str(get_data_path("backtest_data"))] + alt_paths:
                    logger.info(f"  - {path}")
                exit(1)
        
        price_data = pd.read_csv(price_file, parse_dates=["timestamp"])
        logger.info(f"✅ 가격 데이터 로드 완료: {price_file} ({len(price_data)}건)")
        
    except Exception as e:
        logger.error(f"❌ 가격 데이터 로드 실패: {e}")
        exit(1)

    try:
        # 감정 데이터 파일 확인
        sentiment_file = str(get_data_path("sentiment"))
        if os.path.exists(sentiment_file):
            logger.info(f"✅ 감정 데이터 파일 확인: {sentiment_file}")
        else:
            logger.warning(f"⚠️ 감정 데이터 파일 없음: {sentiment_file}")
            logger.info("기본 감정 점수(0.0)로 진행합니다.")
        
        # 백테스트 실행 (테스트를 위해 처음 1000개만)
        max_iter = 1000 if len(price_data) > 1000 else None
        if max_iter:
            logger.info(f"테스트 모드: 처음 {max_iter}개 데이터만 처리")
        
        results = run_backtest_loop(
            price_data=price_data,
            sentiment_data=None,
            sentiment_file=sentiment_file,
            max_iterations=max_iter
        )

        if results:
            output_file = save_results(results)
            logger.info(f"✅ 백테스트 완료: {output_file}")
        else:
            logger.error("❌ 유효한 백테스트 결과 없음")
            
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 백테스트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 백테스트 실행 중 오류: {e}", exc_info=True)
