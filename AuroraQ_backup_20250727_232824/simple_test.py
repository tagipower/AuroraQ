"""
간단한 백테스트 v2 테스트
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_simple_test_data():
    """간단한 테스트 데이터 생성"""
    print("📊 테스트 데이터 생성 중...")
    
    # 100개 데이터 포인트 생성 (더 적은 양으로)
    periods = 100
    start_date = datetime.now() - timedelta(hours=periods//12)
    timestamps = pd.date_range(start_date, periods=periods, freq='5min')
    
    # 간단한 가격 시뮬레이션
    np.random.seed(42)
    initial_price = 50000
    prices = [initial_price]
    
    for i in range(1, periods):
        change = np.random.normal(0, 0.001)  # 0.1% 변동성
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLCV 데이터
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.002))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.002))
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # 파일 저장
    os.makedirs('data/test', exist_ok=True)
    price_file = 'data/test/simple_price.csv'
    df.to_csv(price_file, index=False)
    
    print(f"✅ 가격 데이터 저장: {price_file} ({len(df)}개 레코드)")
    return price_file

def test_data_layer():
    """데이터 계층 테스트"""
    print("🔧 데이터 계층 테스트...")
    
    try:
        from backtest.v2.layers.data_layer import DataLayer
        
        # 데이터 레이어 생성
        data_layer = DataLayer(cache_size=100, enable_multiframe=False)
        
        # 테스트 데이터 생성
        price_file = create_simple_test_data()
        
        # 데이터 로드
        price_data = data_layer.load_price_data(price_file)
        print(f"✅ 가격 데이터 로드: {len(price_data)}개 레코드")
        
        # 지표 계산
        indicators = data_layer.calculate_indicators(
            price_data, 
            ["sma_20", "rsi", "volatility"]
        )
        print(f"✅ 지표 계산 완료: {list(indicators.keys())}")
        
        # 캐시 통계
        cache_stats = data_layer.get_cache_stats()
        print(f"✅ 캐시 통계: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 계층 테스트 실패: {e}")
        return False

def test_signal_layer():
    """시그널 계층 테스트"""
    print("🎯 시그널 계층 테스트...")
    
    try:
        from backtest.v2.layers.signal_layer import SignalProcessor
        
        processor = SignalProcessor()
        print("✅ 시그널 프로세서 생성 완료")
        
        # 더미 시그널 데이터
        strategy_signal = {
            "action": "BUY",
            "strength": 0.7
        }
        
        # 더미 시장 데이터
        timestamps = pd.date_range(datetime.now(), periods=50, freq='5min')
        market_data = {
            "price": pd.DataFrame({
                'timestamp': timestamps,
                'close': [50000 + i * 10 for i in range(50)],
                'high': [50010 + i * 10 for i in range(50)],
                'low': [49990 + i * 10 for i in range(50)],
                'open': [50000 + i * 10 for i in range(50)]
            })
        }
        
        # 더미 지표
        indicators = {
            "rsi": pd.Series([50 + i for i in range(50)]),
            "atr": pd.Series([100 + i for i in range(50)])
        }
        
        # 신호 처리
        result = processor.process_signal(
            strategy_signal,
            market_data,
            indicators,
            sentiment_score=0.6
        )
        
        print(f"✅ 신호 처리 완료: {result.action}, 신뢰도: {result.confidence:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 시그널 계층 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 AuroraQ 백테스트 시스템 v2 간단 테스트")
    print("=" * 50)
    
    results = []
    
    # 1. 데이터 계층 테스트
    results.append(("데이터 계층", test_data_layer()))
    
    # 2. 시그널 계층 테스트
    results.append(("시그널 계층", test_signal_layer()))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약:")
    
    for name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {name}: {status}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\n총 {total_success}/{len(results)}개 테스트 성공")
    
    if total_success == len(results):
        print("🎉 모든 테스트 성공!")
    else:
        print("⚠️ 일부 테스트 실패")

if __name__ == "__main__":
    main()