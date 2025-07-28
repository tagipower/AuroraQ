#!/usr/bin/env python3
"""
전략 통합 테스트
=================

rule_strategies.py와 strategy_adapter.py 통합 동작 테스트
"""

import sys
import os

# 현재 디렉토리를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_rule_strategies_import():
    """룰 전략 직접 import 테스트"""
    print("🧪 Rule Strategies 직접 Import 테스트")
    
    try:
        from rule_strategies import (
            RuleStrategyA, RuleStrategyB, RuleStrategyC, 
            RuleStrategyD, RuleStrategyE, get_available_strategies
        )
        
        print("✅ 룰 전략 모듈 import 성공")
        
        # 사용 가능한 전략 확인
        strategies = get_available_strategies()
        print(f"📋 사용 가능한 전략: {strategies}")
        
        # 각 전략 인스턴스 생성 테스트
        for strategy_name in strategies:
            strategy_class = globals()[strategy_name]
            try:
                instance = strategy_class()
                print(f"✅ {strategy_name} 인스턴스 생성 성공")
            except Exception as e:
                print(f"❌ {strategy_name} 인스턴스 생성 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 룰 전략 import 실패: {e}")
        return False

def test_strategy_adapter_integration():
    """Strategy Adapter 통합 테스트"""
    print("\n🔗 Strategy Adapter 통합 테스트")
    
    try:
        from strategy_adapter import get_strategy_registry, register_builtin_strategies
        
        print("✅ Strategy Adapter import 성공")
        
        # 전략 레지스트리 가져오기
        registry = get_strategy_registry()
        print(f"📋 레지스트리 전략 수: {len(registry.get_all_strategy_names())}")
        
        # 등록된 전략 목록
        registered_strategies = registry.get_all_strategy_names()
        print(f"📊 등록된 전략: {registered_strategies}")
        
        # 각 전략 어댑터 테스트
        for strategy_name in registered_strategies:
            adapter = registry.get_strategy_adapter(strategy_name)
            if adapter:
                print(f"✅ {strategy_name} 어댑터 조회 성공")
            else:
                print(f"❌ {strategy_name} 어댑터 조회 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy Adapter 테스트 실패: {e}")
        return False

def test_strategy_signal_generation():
    """전략 신호 생성 테스트"""
    print("\n📊 전략 신호 생성 테스트")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 테스트 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        # 가격 시뮬레이션
        price_changes = np.random.normal(0, 0.02, 100)
        prices = 50000 * np.cumprod(1 + price_changes)
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.998, 1.002, 100),
            'high': prices * np.random.uniform(1.001, 1.005, 100),
            'low': prices * np.random.uniform(0.995, 0.999, 100),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        print(f"📈 테스트 데이터 생성: {len(test_data)}개 캔들")
        
        # 전략 테스트
        from rule_strategies import get_rule_strategy, get_available_strategies
        
        success_count = 0
        for strategy_name in get_available_strategies():
            try:
                strategy = get_rule_strategy(strategy_name)
                if strategy:
                    # 진입 신호 테스트
                    entry_signal = strategy.should_enter(test_data)
                    
                    if entry_signal:
                        print(f"✅ {strategy_name} 진입 신호 생성: {entry_signal.get('reason', 'No reason')}")
                        print(f"   신뢰도: {entry_signal.get('confidence', 0):.3f}")
                    else:
                        print(f"📊 {strategy_name} 진입 신호 없음 (정상)")
                    
                    success_count += 1
                else:
                    print(f"❌ {strategy_name} 전략 생성 실패")
                    
            except Exception as e:
                print(f"❌ {strategy_name} 신호 생성 테스트 실패: {e}")
        
        print(f"\n📊 테스트 결과: {success_count}/{len(get_available_strategies())} 전략 성공")
        return success_count == len(get_available_strategies())
        
    except Exception as e:
        print(f"❌ 신호 생성 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 AuroraQ Production 전략 통합 테스트")
    print("=" * 60)
    
    # 테스트 실행
    tests = [
        ("Rule Strategies Import", test_rule_strategies_import),
        ("Strategy Adapter Integration", test_strategy_adapter_integration),
        ("Strategy Signal Generation", test_strategy_signal_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name} 시작...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 성공" if result else "❌ 실패"
            print(f"📊 {test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name} 예외 발생: {e}")
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 전체 테스트 결과")
    print("=" * 60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{status} {test_name}")
        if result:
            success_count += 1
    
    print(f"\n🎯 총 성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("🎉 모든 테스트 통과! rule_strategies.py 통합 완료")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. 문제를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)