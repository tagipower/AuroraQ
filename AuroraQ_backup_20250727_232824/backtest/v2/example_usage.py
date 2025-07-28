"""
AuroraQ 백테스트 시스템 v2 사용 예제
적응형·확률적·다중프레임 백테스트 환경 사용법
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.v2.layers.controller_layer import BacktestController, BacktestOrchestrator, BacktestMode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """샘플 데이터 생성"""
    # 가격 데이터 생성 (5분봉 데이터)
    start_date = datetime.now() - timedelta(days=30)
    periods = 30 * 24 * 12  # 30일 * 24시간 * 12 (5분봉)
    
    timestamps = pd.date_range(start_date, periods=periods, freq='5T')
    
    # 가격 시뮬레이션 (간단한 랜덤워크)
    import numpy as np
    np.random.seed(42)
    
    initial_price = 50000
    returns = np.random.normal(0, 0.001, periods)  # 0.1% 변동성
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # OHLCV 데이터 생성
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.002))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.002))
        volume = np.random.uniform(100, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # 데이터 저장
    os.makedirs('data/sample', exist_ok=True)
    price_file = 'data/sample/sample_price_data.csv'
    df.to_csv(price_file, index=False)
    
    # 감정 데이터 생성
    sentiment_data = []
    for timestamp in timestamps[::12]:  # 1시간마다
        sentiment_score = 0.5 + 0.3 * np.sin(len(sentiment_data) * 0.1) + np.random.normal(0, 0.1)
        sentiment_score = np.clip(sentiment_score, 0, 1)
        
        sentiment_data.append({
            'timestamp': timestamp,
            'sentiment_score': sentiment_score,
            'confidence': 0.8 + 0.2 * np.random.random()
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_file = 'data/sample/sample_sentiment_data.csv'
    sentiment_df.to_csv(sentiment_file, index=False)
    
    return price_file, sentiment_file


def example_simple_backtest():
    """간단한 백테스트 예제"""
    logger.info("=== 간단한 백테스트 예제 ===")
    
    # 샘플 데이터 생성
    price_file, sentiment_file = create_sample_data()
    
    # 백테스트 컨트롤러 생성
    controller = BacktestController(
        initial_capital=1000000,
        mode=BacktestMode.NORMAL,
        enable_multiframe=True,
        enable_exploration=False
    )
    
    # 전략 시스템 초기화
    controller.initialize_strategies(
        sentiment_file=sentiment_file,
        enable_ppo=False  # 예제에서는 간단히 비활성화
    )
    
    # 백테스트 실행
    result = controller.run_backtest(
        price_data_path=price_file,
        sentiment_data_path=sentiment_file,
        window_size=50,
        indicators=["sma_20", "rsi", "macd", "atr"]
    )
    
    if result["success"]:
        logger.info("백테스트 성공!")
        logger.info(f"최고 전략: {result['metrics']['best_strategy']}")
        logger.info(f"실행 시간: {result['stats']['execution_time']:.2f}초")
        logger.info(f"총 거래: {result['stats']['executed_trades']}")
        logger.info(f"캐시 히트율: {result['stats']['cache_stats']['hit_rate']:.2%}")
    else:
        logger.error(f"백테스트 실패: {result['error']}")


def example_exploration_mode():
    """탐색 모드 백테스트 예제"""
    logger.info("=== 탐색 모드 백테스트 예제 ===")
    
    # 샘플 데이터 생성
    price_file, sentiment_file = create_sample_data()
    
    # 탐색 모드로 백테스트 컨트롤러 생성
    controller = BacktestController(
        initial_capital=1000000,
        mode=BacktestMode.EXPLORATION,
        enable_multiframe=True,
        enable_exploration=True  # 탐색 모드 활성화
    )
    
    # 전략 시스템 초기화
    controller.initialize_strategies(
        sentiment_file=sentiment_file,
        enable_ppo=False
    )
    
    # 백테스트 실행
    result = controller.run_backtest(
        price_data_path=price_file,
        sentiment_data_path=sentiment_file,
        window_size=50
    )
    
    if result["success"]:
        logger.info("탐색 모드 백테스트 성공!")
        logger.info(f"탐색 거래: {result['stats']['exploration_trades']}")
        logger.info(f"전체 거래: {result['stats']['executed_trades']}")
        logger.info(f"탐색 비율: {result['stats']['exploration_trades']/result['stats']['executed_trades']:.2%}")


def example_multiple_backtests():
    """복수 백테스트 예제"""
    logger.info("=== 복수 백테스트 예제 ===")
    
    # 샘플 데이터 생성
    price_file, sentiment_file = create_sample_data()
    
    # 여러 백테스트 설정
    configurations = [
        {
            "name": "normal_mode",
            "price_data_path": price_file,
            "sentiment_data_path": sentiment_file,
            "initial_capital": 1000000,
            "mode": BacktestMode.NORMAL,
            "enable_multiframe": True,
            "enable_exploration": False,
            "window_size": 50
        },
        {
            "name": "exploration_mode",
            "price_data_path": price_file,
            "sentiment_data_path": sentiment_file,
            "initial_capital": 1000000,
            "mode": BacktestMode.EXPLORATION,
            "enable_multiframe": True,
            "enable_exploration": True,
            "window_size": 50
        },
        {
            "name": "validation_mode",
            "price_data_path": price_file,
            "sentiment_data_path": sentiment_file,
            "initial_capital": 1000000,
            "mode": BacktestMode.VALIDATION,
            "enable_multiframe": False,
            "enable_exploration": False,
            "window_size": 100
        }
    ]
    
    # 오케스트레이터 생성
    orchestrator = BacktestOrchestrator(n_workers=2)
    
    # 병렬 백테스트 실행
    results = orchestrator.run_multiple_backtests(
        configurations, 
        parallel=True
    )
    
    # 결과 비교
    logger.info("백테스트 결과 비교:")
    for result in results:
        if result["success"]:
            config = result["config"]
            best_roi = result["metrics"]["best_metrics"].roi
            logger.info(f"{config['name']}: ROI {best_roi:.2%}")
        else:
            logger.error(f"{result['config']['name']}: 실패 - {result['error']}")


def example_walk_forward_analysis():
    """워크포워드 분석 예제"""
    logger.info("=== 워크포워드 분석 예제 ===")
    
    # 샘플 데이터 생성 (더 긴 기간)
    price_file, sentiment_file = create_sample_data()
    
    # 기본 설정
    base_config = {
        "price_data_path": price_file,
        "sentiment_data_path": sentiment_file,
        "initial_capital": 1000000,
        "enable_multiframe": True,
        "enable_exploration": False,
        "window_size": 50
    }
    
    # 오케스트레이터 생성
    orchestrator = BacktestOrchestrator()
    
    # 워크포워드 분석 실행
    wf_result = orchestrator.walk_forward_analysis(
        base_config,
        n_windows=5,
        train_ratio=0.7
    )
    
    # 결과 출력
    if wf_result["statistics"]:
        stats = wf_result["statistics"]
        logger.info(f"훈련 평균 ROI: {stats['train_avg_roi']:.2%}")
        logger.info(f"테스트 평균 ROI: {stats['test_avg_roi']:.2%}")
        logger.info(f"효율성 비율: {stats['efficiency_ratio']:.2f}")
        logger.info(f"일관성: {stats['consistency']:.2f}")


def main():
    """메인 함수"""
    try:
        # 1. 간단한 백테스트
        example_simple_backtest()
        
        # 2. 탐색 모드
        example_exploration_mode()
        
        # 3. 복수 백테스트
        example_multiple_backtests()
        
        # 4. 워크포워드 분석
        example_walk_forward_analysis()
        
        logger.info("=== 모든 예제 완료 ===")
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()