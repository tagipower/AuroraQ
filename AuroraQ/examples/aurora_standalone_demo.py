#!/usr/bin/env python3
"""
AuroraQ 독립 실행 모드 데모
리소스 사용량 및 성능 비교
"""

import asyncio
import time
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from AuroraQ.standalone_runner import AuroraQStandalone
from SharedCore.utils.resource_monitor import get_resource_monitor
from SharedCore.utils.cache_manager import create_cache_manager, CacheMode


async def demo_aurora_standalone():
    """AuroraQ 독립 실행 데모"""
    print("🎯 AuroraQ Standalone Mode Demo")
    print("=" * 50)
    
    # 리소스 모니터 초기화
    monitor = get_resource_monitor()
    
    # 시작 전 리소스 상태
    print("📊 Initial Resource Status:")
    monitor.log_resource_status()
    
    # AuroraQ 독립 실행 초기화
    runner = AuroraQStandalone()
    
    try:
        # 초기화
        start_time = time.time()
        await runner.initialize()
        init_time = time.time() - start_time
        
        print(f"⚡ Initialization completed in {init_time:.2f} seconds")
        
        # 초기화 후 리소스 상태
        print("\n📊 Post-initialization Resource Status:")
        monitor.log_resource_status()
        
        # 캐시 성능 테스트
        print("\n🔄 Testing cache performance...")
        await test_cache_performance(runner)
        
        # 상태 조회
        print("\n📋 AuroraQ Status:")
        status = await runner.get_status()
        print(f"  Status: {status['status']}")
        print(f"  Loaded modules: {status.get('loaded_modules', [])}")
        
        # 간단한 백테스트
        print("\n📈 Running quick backtest...")
        backtest_result = await runner.run_backtest("2025-01-01", "2025-01-31")
        
        # 최종 리소스 상태
        print("\n📊 Final Resource Status:")
        monitor.log_resource_status()
        
        # 리소스 최적화 제안
        suggestions = monitor.get_optimization_suggestions(mode="aurora")
        if any(suggestions.values()):
            print("\n💡 Optimization Suggestions:")
            for category, items in suggestions.items():
                if items:
                    print(f"  {category.title()}:")
                    for item in items:
                        print(f"    • {item}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        await runner.shutdown()


async def test_cache_performance(runner: AuroraQStandalone):
    """캐시 성능 테스트"""
    if not runner.data_provider:
        return
        
    # 캐시 미스 테스트 (첫 번째 요청)
    start_time = time.time()
    data1 = await runner.data_provider.get_market_data("crypto", "BTC/USDT", "1h")
    first_request_time = time.time() - start_time
    
    # 캐시 히트 테스트 (두 번째 요청)
    start_time = time.time()
    data2 = await runner.data_provider.get_market_data("crypto", "BTC/USDT", "1h")
    second_request_time = time.time() - start_time
    
    print(f"  First request (cache miss): {first_request_time:.3f}s")
    print(f"  Second request (cache hit): {second_request_time:.3f}s")
    print(f"  Cache speedup: {first_request_time/second_request_time:.1f}x")


async def compare_modes():
    """모드별 리소스 사용량 비교"""
    print("\n🔄 Comparing resource usage across modes...")
    print("=" * 50)
    
    monitor = get_resource_monitor()
    results = {}
    
    # Aurora Only 모드
    print("Testing Aurora-only mode...")
    runner_aurora = AuroraQStandalone()
    
    baseline = monitor.get_current_usage()
    await runner_aurora.initialize()
    aurora_usage = monitor.get_current_usage()
    
    results['aurora_only'] = {
        'cpu_increase': aurora_usage.cpu_percent - baseline.cpu_percent,
        'memory_increase_mb': aurora_usage.memory_used_mb - baseline.memory_used_mb
    }
    
    await runner_aurora.shutdown()
    
    # 결과 출력
    print("\n📊 Resource Usage Comparison:")
    for mode, metrics in results.items():
        print(f"  {mode.replace('_', ' ').title()}:")
        print(f"    CPU increase: +{metrics['cpu_increase']:.1f}%")
        print(f"    Memory increase: +{metrics['memory_increase_mb']:.0f}MB")


async def benchmark_cache_modes():
    """캐시 모드별 성능 벤치마크"""
    print("\n⚡ Cache Mode Benchmarks...")
    print("=" * 50)
    
    cache_modes = [
        (CacheMode.AURORA_ONLY, "Aurora Only"),
        (CacheMode.FULL, "Full Cache"),
        (CacheMode.MINIMAL, "Minimal Cache")
    ]
    
    for mode, name in cache_modes:
        print(f"Testing {name} cache mode...")
        
        cache_manager = create_cache_manager(mode.value)
        await cache_manager.connect()
        
        # 성능 테스트
        start_time = time.time()
        for i in range(100):
            await cache_manager.set(f"test:{i}", {"data": f"value_{i}"})
        
        set_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            await cache_manager.get(f"test:{i}")
        
        get_time = time.time() - start_time
        
        stats = cache_manager.get_stats()
        
        print(f"  Set operations: {set_time:.3f}s")
        print(f"  Get operations: {get_time:.3f}s")
        print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"  Memory usage: {stats['memory_usage_mb']:.1f}MB")
        
        await cache_manager.close()


async def main():
    """메인 데모 실행"""
    print("🚀 QuantumAI - AuroraQ Standalone Demo")
    print("Testing resource-optimized independent execution")
    print("=" * 60)
    
    try:
        # 기본 데모
        await demo_aurora_standalone()
        
        # 모드 비교 (선택사항)
        print("\n" + "=" * 60)
        choice = input("Run detailed benchmarks? (y/N): ").lower().strip()
        
        if choice == 'y':
            await compare_modes()
            await benchmark_cache_modes()
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    print("\n✅ Demo completed")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())