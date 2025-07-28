"""
전체 백테스트 v2 시스템 테스트
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_test_data():
    """테스트 데이터 생성"""
    print("📊 테스트 데이터 생성 중...")
    
    # 200개 데이터 포인트 (약 16시간 분량의 5분봉)
    periods = 200
    start_date = datetime.now() - timedelta(hours=periods//12)
    timestamps = pd.date_range(start_date, periods=periods, freq='5min')
    
    # 현실적인 가격 시뮬레이션
    np.random.seed(42)
    initial_price = 50000
    prices = [initial_price]
    
    for i in range(1, periods):
        # 트렌드 + 노이즈
        trend = 0.0001 * np.sin(i * 0.1)  # 약간의 사인파 트렌드
        noise = np.random.normal(0, 0.002)  # 0.2% 변동성
        change = trend + noise
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLCV 데이터
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.003))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.003))
        volume = np.random.uniform(500, 5000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    price_df = pd.DataFrame(data)
    
    # 감정 데이터 (시간당)
    sentiment_timestamps = pd.date_range(start_date, periods=periods//12, freq='1h')
    sentiment_data = []
    
    for ts in sentiment_timestamps:
        score = 0.5 + 0.3 * np.sin(len(sentiment_data) * 0.2) + np.random.normal(0, 0.1)
        score = np.clip(score, 0, 1)
        
        sentiment_data.append({
            'timestamp': ts,
            'sentiment_score': score,
            'confidence': 0.8 + 0.2 * np.random.random()
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # 파일 저장
    os.makedirs('data/test', exist_ok=True)
    price_file = 'data/test/price_data.csv'
    sentiment_file = 'data/test/sentiment_data.csv'
    
    price_df.to_csv(price_file, index=False)
    sentiment_df.to_csv(sentiment_file, index=False)
    
    print(f"✅ 데이터 저장 완료:")
    print(f"  - 가격: {price_file} ({len(price_df)}개)")
    print(f"  - 감정: {sentiment_file} ({len(sentiment_df)}개)")
    
    return price_file, sentiment_file

def test_full_controller():
    """전체 컨트롤러 테스트"""
    print("🎮 백테스트 컨트롤러 테스트...")
    
    try:
        from backtest.v2.layers.controller_layer import BacktestController, BacktestMode
        
        # 테스트 데이터 생성
        price_file, sentiment_file = create_test_data()
        
        # 컨트롤러 생성
        controller = BacktestController(
            initial_capital=100000,  # 더 작은 금액으로 테스트
            mode=BacktestMode.NORMAL,
            enable_multiframe=False,  # 단순화
            enable_exploration=False
        )
        print("✅ 백테스트 컨트롤러 생성 완료")
        
        # 전략 초기화 (더미 모드)
        controller.initialize_strategies(
            sentiment_file=sentiment_file,
            enable_ppo=False
        )
        print("✅ 전략 시스템 초기화 완료")
        
        # 백테스트 실행 (작은 윈도우로)
        print("🚀 백테스트 실행 중...")
        result = controller.run_backtest(
            price_data_path=price_file,
            sentiment_data_path=sentiment_file,
            window_size=20,  # 작은 윈도우
            indicators=["sma_20", "rsi"]  # 간단한 지표만
        )
        
        if result["success"]:
            print("✅ 백테스트 성공!")
            print(f"  - 실행 시간: {result['stats']['execution_time']:.2f}초")
            print(f"  - 총 신호: {result['stats']['total_signals']}")
            print(f"  - 실행 거래: {result['stats']['executed_trades']}")
            print(f"  - 캐시 히트율: {result['stats']['cache_stats']['hit_rate']:.2%}")
            
            # 메트릭 확인
            if result['metrics']['best_strategy']:
                print(f"  - 최고 전략: {result['metrics']['best_strategy']}")
            
            return True
        else:
            print(f"❌ 백테스트 실패: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 컨트롤러 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_layer():
    """평가 계층 테스트"""
    print("📊 평가 계층 테스트...")
    
    try:
        from backtest.v2.layers.evaluation_layer import MetricsEvaluator
        
        evaluator = MetricsEvaluator(initial_capital=100000)
        print("✅ 메트릭 평가기 생성 완료")
        
        # 더미 거래 데이터 추가
        for i in range(10):
            trade_data = {
                "timestamp": datetime.now() - timedelta(hours=i),
                "signal_action": ["BUY", "SELL"][i % 2],
                "entry_price": 50000 + i * 100,
                "pnl": np.random.normal(100, 500),  # 랜덤 수익
                "commission": 5,
                "signal_confidence": 0.5 + 0.3 * np.random.random()
            }
            evaluator.add_trade("TestStrategy", trade_data)
        
        # 평가 실행
        metrics = evaluator.evaluate_strategy("TestStrategy")
        print(f"✅ 전략 평가 완료:")
        print(f"  - 총 거래: {metrics.total_trades}")
        print(f"  - ROI: {metrics.roi:.2%}")
        print(f"  - 승률: {metrics.win_rate:.2%}")
        print(f"  - 종합 점수: {metrics.composite_score:.3f}")
        
        # 보고서 생성
        reports = evaluator.generate_reports("TestStrategy", format="json")
        print(f"✅ 보고서 생성 완료: {list(reports.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 평가 계층 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_execution_layer():
    """실행 계층 테스트"""
    print("⚡ 실행 계층 테스트...")
    
    try:
        from backtest.v2.layers.execution_layer import ExecutionSimulator
        from backtest.v2.layers.signal_layer import SignalResult
        
        simulator = ExecutionSimulator(initial_capital=100000)
        print("✅ 실행 시뮬레이터 생성 완료")
        
        # 더미 신호
        signal = SignalResult(
            action="BUY",
            confidence=0.8,
            position_size=0.1,
            entry_price=50000
        )
        
        # 더미 시장 데이터
        market_data = {
            "price": pd.DataFrame({
                'close': [50000],
                'volume': [1000]
            }),
            "volatility": 0.02
        }
        
        # 실행 시뮬레이션
        result = simulator.execute_signal(
            signal,
            market_data,
            datetime.now()
        )
        
        if result["executed"]:
            print("✅ 거래 실행 시뮬레이션 완료:")
            details = result["execution_details"]
            print(f"  - 요청 가격: {details['requested_price']:,.0f}")
            print(f"  - 체결 가격: {details['execution_price']:,.0f}")
            print(f"  - 슬리피지: {details['slippage']:.2f}")
            print(f"  - 수수료: {details['commission']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 실행 계층 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 AuroraQ 백테스트 시스템 v2 전체 테스트")
    print("=" * 60)
    
    tests = [
        ("실행 계층", test_execution_layer),
        ("평가 계층", test_evaluation_layer),
        ("전체 컨트롤러", test_full_controller)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'-' * 30}")
        success = test_func()
        results.append((name, success))
        print(f"{'-' * 30}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 최종 테스트 결과:")
    
    for name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {name}: {status}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\n총 {total_success}/{len(results)}개 테스트 성공")
    
    if total_success == len(results):
        print("🎉 모든 테스트 성공! 새로운 백테스트 시스템이 정상 작동합니다.")
    else:
        print("⚠️ 일부 테스트 실패 - 추가 디버깅이 필요합니다.")

if __name__ == "__main__":
    main()