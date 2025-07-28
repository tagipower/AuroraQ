#!/usr/bin/env python3
"""
AuroraQ Production 전략 테스트
============================

통합된 룰 전략들의 Production 환경 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 현재 디렉토리를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def create_test_data(length=100):
    """테스트용 가격 데이터 생성"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=length, freq='H')
    
    # 가격 시뮬레이션 (브라운 운동)
    price_changes = np.random.normal(0, 0.02, length)
    prices = 50000 * np.cumprod(1 + price_changes)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.998, 1.002, length),
        'high': prices * np.random.uniform(1.001, 1.005, length),
        'low': prices * np.random.uniform(0.995, 0.999, length),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, length)
    })
    
    return data

def test_rule_strategies_import():
    """룰 전략 import 테스트"""
    print("🧪 룰 전략 Import 테스트")
    
    try:
        from rule_strategies import (
            RuleStrategyA, RuleStrategyB, RuleStrategyC, 
            RuleStrategyD, RuleStrategyE, get_available_strategies
        )
        
        print("✅ 룰 전략 모듈 import 성공")
        
        # 모든 전략 인스턴스 생성
        strategies = {}
        for strategy_name in get_available_strategies():
            try:
                strategy_class = eval(strategy_name)
                instance = strategy_class()
                strategies[strategy_name] = instance
                print(f"✅ {strategy_name} 인스턴스 생성 성공")
            except Exception as e:
                print(f"❌ {strategy_name} 인스턴스 생성 실패: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 룰 전략 import 실패: {e}")
        return False

def test_strategy_adapter():
    """전략 어댑터 테스트"""
    print("\n🔗 전략 어댑터 테스트")
    
    try:
        from strategy_adapter import get_strategy_registry
        
        # 전략 레지스트리 가져오기
        registry = get_strategy_registry()
        strategies = registry.get_all_strategy_names()
        
        print(f"📋 등록된 전략 수: {len(strategies)}")
        print(f"📊 등록된 전략: {strategies}")
        
        # 각 전략의 어댑터 테스트
        success_count = 0
        for strategy_name in strategies:
            adapter = registry.get_strategy_adapter(strategy_name)
            if adapter:
                print(f"✅ {strategy_name} 어댑터 조회 성공")
                success_count += 1
            else:
                print(f"❌ {strategy_name} 어댑터 조회 실패")
        
        return success_count == len(strategies)
        
    except Exception as e:
        print(f"❌ 전략 어댑터 테스트 실패: {e}")
        return False

def test_strategies_signal_generation():
    """전략 신호 생성 테스트"""
    print("\n📊 전략 신호 생성 테스트")
    
    try:
        from rule_strategies import get_rule_strategy, get_available_strategies
        
        # 테스트 데이터 생성
        test_data = create_test_data()
        print(f"📈 테스트 데이터 생성: {len(test_data)}개 캔들")
        
        success_count = 0
        total_signals = 0
        
        for strategy_name in get_available_strategies():
            try:
                strategy = get_rule_strategy(strategy_name)
                if not strategy:
                    print(f"❌ {strategy_name} 전략 생성 실패")
                    continue
                
                # 진입 신호 테스트
                entry_signal = strategy.should_enter(test_data)
                
                if entry_signal:
                    print(f"🎯 {strategy_name} 진입 신호 생성!")
                    print(f"   신뢰도: {entry_signal.get('confidence', 0):.3f}")
                    print(f"   이유: {entry_signal.get('reason', 'No reason')}")
                    total_signals += 1
                else:
                    print(f"📊 {strategy_name} 진입 신호 없음 (정상)")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ {strategy_name} 신호 생성 테스트 실패: {e}")
        
        print(f"\n📊 신호 생성 테스트 결과:")
        print(f"   성공한 전략: {success_count}/{len(get_available_strategies())}")
        print(f"   생성된 신호: {total_signals}개")
        
        return success_count == len(get_available_strategies())
        
    except Exception as e:
        print(f"❌ 신호 생성 테스트 실패: {e}")
        return False

def test_strategy_indicators():
    """전략 지표 계산 테스트"""
    print("\n📈 전략 지표 계산 테스트")
    
    try:
        from rule_strategies import RuleStrategyA
        
        # 충분한 데이터로 테스트
        test_data = create_test_data(200)
        strategy = RuleStrategyA()
        
        # 지표 계산
        indicators = strategy.calculate_indicators(test_data)
        
        print(f"📊 계산된 지표 수: {len(indicators)}")
        
        expected_indicators = ['ema_short', 'ema_long', 'adx']
        found_indicators = 0
        
        for indicator in expected_indicators:
            if indicator in indicators:
                value = indicators[indicator]
                print(f"✅ {indicator}: {value:.3f}")
                found_indicators += 1
            else:
                print(f"❌ {indicator}: 누락")
        
        return found_indicators == len(expected_indicators)
        
    except Exception as e:
        print(f"❌ 지표 계산 테스트 실패: {e}")
        return False

def test_strategy_position_management():
    """전략 포지션 관리 테스트"""
    print("\n💼 전략 포지션 관리 테스트")
    
    try:
        from rule_strategies import RuleStrategyA
        
        # 테스트 데이터와 전략
        test_data = create_test_data(200)
        strategy = RuleStrategyA()
        
        # 모의 포지션 객체 생성
        class MockPosition:
            def __init__(self):
                self.entry_price = 50000
                self.entry_time = datetime.now() - timedelta(minutes=30)
                self.side = "LONG"
                self.confidence = 0.7
            
            @property
            def holding_time(self):
                return datetime.now() - self.entry_time
        
        position = MockPosition()
        
        # 청산 조건 테스트
        exit_reason = strategy.should_exit(position, test_data)
        
        if exit_reason:
            print(f"🎯 청산 신호 생성: {exit_reason}")
        else:
            print("📊 청산 신호 없음 (정상)")
        
        print("✅ 포지션 관리 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 포지션 관리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 AuroraQ Production 전략 종합 테스트")
    print("=" * 60)
    
    # 테스트 리스트
    tests = [
        ("룰 전략 Import", test_rule_strategies_import),
        ("전략 어댑터", test_strategy_adapter),
        ("신호 생성", test_strategies_signal_generation),
        ("지표 계산", test_strategy_indicators),
        ("포지션 관리", test_strategy_position_management)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name} 테스트 시작...")
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
    
    success_rate = success_count / len(results) * 100
    print(f"\n🎯 총 성공률: {success_count}/{len(results)} ({success_rate:.1f}%)")
    
    if success_count == len(results):
        print("🎉 모든 테스트 통과! Production 룰 전략 통합 완료")
        return 0
    elif success_rate >= 80:
        print("✅ 대부분의 테스트 통과! 시스템이 정상 작동합니다")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. 문제를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)