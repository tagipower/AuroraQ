#!/usr/bin/env python3
"""
VPS Deployment 성능 최적화 시스템
메모리 관리, CPU 최적화, I/O 최적화, 캐싱 전략
"""

import asyncio
import gc
import json
import logging
import os
import psutil
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Union
import resource


class MemoryManager:
    """메모리 관리 시스템"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_limit_mb = 3072  # 3GB VPS 제한
        self.gc_stats = deque(maxlen=100)
        self.memory_snapshots = deque(maxlen=1000)
        self.large_objects = weakref.WeakSet()
        
        # 메모리 모니터링 스레드
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def _monitor_memory(self):
        """메모리 모니터링 스레드"""
        while self.monitoring:
            try:
                usage = self.get_memory_usage()
                self.memory_snapshots.append({
                    'timestamp': time.time(),
                    'usage': usage
                })
                
                # 임계치 확인
                if usage['percent'] > self.critical_threshold:
                    self.emergency_cleanup()
                elif usage['percent'] > self.warning_threshold:
                    self.optimize_memory()
                
                time.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                self.logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(60)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량"""
        process = psutil.Process()
        memory_info = process.memory_info()
        vm_info = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': vm_info.available / 1024 / 1024,
            'system_percent': vm_info.percent,
            'swap_mb': psutil.swap_memory().used / 1024 / 1024,
            'limit_mb': self.memory_limit_mb,
            'within_limit': memory_info.rss / 1024 / 1024 < self.memory_limit_mb
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        before_usage = self.get_memory_usage()
        
        # 1. 가비지 컬렉션
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        # 2. 큰 객체 정리
        large_objects_count = len(self.large_objects)
        
        # 3. 시스템 캐시 정리 (필요시)
        if hasattr(gc, 'freeze'):
            gc.freeze()
        
        after_usage = self.get_memory_usage()
        
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'before_memory_mb': before_usage['rss_mb'],
            'after_memory_mb': after_usage['rss_mb'],
            'memory_freed_mb': before_usage['rss_mb'] - after_usage['rss_mb'],
            'gc_collected': sum(collected),
            'large_objects_tracked': large_objects_count,
            'optimization_type': 'routine'
        }
        
        self.gc_stats.append(optimization_result)
        self.logger.info(f"메모리 최적화 완료: {optimization_result['memory_freed_mb']:.2f}MB 확보")
        
        return optimization_result
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """긴급 메모리 정리"""
        self.logger.warning("긴급 메모리 정리 시작")
        
        before_usage = self.get_memory_usage()
        
        # 1. 강제 가비지 컬렉션
        collected = []
        for _ in range(3):  # 여러 번 실행
            for generation in range(3):
                collected.append(gc.collect(generation))
        
        # 2. 메모리 매핑 해제
        if hasattr(resource, 'RLIMIT_DATA'):
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (self.memory_limit_mb * 1024 * 1024, -1))
            except:
                pass
        
        # 3. 캐시 정리
        self._clear_all_caches()
        
        after_usage = self.get_memory_usage()
        
        cleanup_result = {
            'timestamp': datetime.now().isoformat(),
            'before_memory_mb': before_usage['rss_mb'],
            'after_memory_mb': after_usage['rss_mb'],
            'memory_freed_mb': before_usage['rss_mb'] - after_usage['rss_mb'],
            'gc_collected': sum(collected),
            'optimization_type': 'emergency'
        }
        
        self.gc_stats.append(cleanup_result)
        self.logger.warning(f"긴급 메모리 정리 완료: {cleanup_result['memory_freed_mb']:.2f}MB 확보")
        
        return cleanup_result
    
    def _clear_all_caches(self):
        """모든 캐시 정리"""
        # LRU 캐시 정리
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear'):
                try:
                    obj.cache_clear()
                except:
                    pass
    
    def track_large_object(self, obj, threshold_mb: float = 10.0):
        """큰 객체 추적"""
        try:
            obj_size = sys.getsizeof(obj) / 1024 / 1024
            if obj_size > threshold_mb:
                self.large_objects.add(obj)
                self.logger.debug(f"큰 객체 추적: {type(obj).__name__} ({obj_size:.2f}MB)")
        except:
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        if not self.memory_snapshots:
            return {"message": "No memory data available"}
        
        recent_snapshots = list(self.memory_snapshots)[-100:]  # 최근 100개
        memory_values = [s['usage']['rss_mb'] for s in recent_snapshots]
        
        return {
            'current_usage': self.get_memory_usage(),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'max_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'gc_optimizations': len(self.gc_stats),
            'large_objects_tracked': len(self.large_objects),
            'recent_optimizations': list(self.gc_stats)[-5:] if self.gc_stats else []
        }
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False


class CPUOptimizer:
    """CPU 최적화 시스템"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cpu_stats = deque(maxlen=1000)
        self.process_stats = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """CPU 사용량"""
        process = psutil.Process()
        
        cpu_usage = {
            'process_percent': process.cpu_percent(interval=1),
            'system_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            'threads': process.num_threads(),
            'context_switches': process.num_ctx_switches() if hasattr(process, 'num_ctx_switches') else None
        }
        
        self.cpu_stats.append({
            'timestamp': time.time(),
            'usage': cpu_usage
        })
        
        return cpu_usage
    
    def optimize_cpu_bound_task(self, func: Callable, *args, **kwargs):
        """CPU 집약적 작업 최적화"""
        return self.thread_pool.submit(func, *args, **kwargs)
    
    async def optimize_async_cpu_task(self, func: Callable, *args, **kwargs):
        """비동기 CPU 작업 최적화"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """CPU 통계"""
        if not self.cpu_stats:
            return {"message": "No CPU data available"}
        
        recent_stats = list(self.cpu_stats)[-100:]  # 최근 100개
        cpu_values = [s['usage']['process_percent'] for s in recent_stats]
        
        return {
            'current_usage': self.get_cpu_usage(),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'max_workers': self.max_workers,
            'active_threads': threading.active_count()
        }


class SmartCache:
    """지능형 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
        # TTL 정리 스레드
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired(self):
        """만료된 캐시 정리"""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, (value, timestamp) in self.cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.cache.pop(key, None)
                    self.access_times.pop(key, None)
                
                time.sleep(60)  # 1분마다 정리
                
            except Exception:
                time.sleep(60)
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return value
            else:
                # 만료된 항목 제거
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """캐시에 값 저장"""
        current_time = time.time()
        
        # 크기 제한 확인
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = (value, current_time)
        self.access_times[key] = current_time
    
    def _evict_lru(self):
        """LRU 기반 항목 제거"""
        if not self.access_times:
            return
        
        # 가장 오래 사용되지 않은 항목 제거
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.cache.pop(lru_key, None)
        self.access_times.pop(lru_key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
    
    def clear(self):
        """캐시 비우기"""
        self.cache.clear()
        self.access_times.clear()


class PerformanceOptimizer:
    """성능 최적화 통합 시스템"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cpu_optimizer = CPUOptimizer()
        self.cache = SmartCache()
        
        self.logger = logging.getLogger(__name__)
        self.optimization_history = deque(maxlen=100)
        
    def optimize_function(self, cache_key: str = None, cpu_bound: bool = False):
        """함수 최적화 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 캐시 확인
                if cache_key:
                    cache_result = self.cache.get(f"{func.__name__}:{cache_key}")
                    if cache_result is not None:
                        return cache_result
                
                # CPU 집약적 작업 처리
                if cpu_bound:
                    result = await self.cpu_optimizer.optimize_async_cpu_task(func, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                
                # 결과 캐싱
                if cache_key:
                    self.cache.set(f"{func.__name__}:{cache_key}", result)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 캐시 확인
                if cache_key:
                    cache_result = self.cache.get(f"{func.__name__}:{cache_key}")
                    if cache_result is not None:
                        return cache_result
                
                # CPU 집약적 작업 처리
                if cpu_bound:
                    future = self.cpu_optimizer.optimize_cpu_bound_task(func, *args, **kwargs)
                    result = future.result()
                else:
                    result = func(*args, **kwargs)
                
                # 결과 캐싱
                if cache_key:
                    self.cache.set(f"{func.__name__}:{cache_key}", result)
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def performance_context(self, label: str = ""):
        """성능 컨텍스트 매니저"""
        start_time = time.time()
        start_memory = self.memory_manager.get_memory_usage()
        start_cpu = self.cpu_optimizer.get_cpu_usage()
        
        try:
            yield self
        finally:
            end_time = time.time()
            end_memory = self.memory_manager.get_memory_usage()
            end_cpu = self.cpu_optimizer.get_cpu_usage()
            
            performance_data = {
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': end_time - start_time,
                'memory_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
                'cpu_usage_avg': (start_cpu['process_percent'] + end_cpu['process_percent']) / 2,
                'start_memory_mb': start_memory['rss_mb'],
                'end_memory_mb': end_memory['rss_mb']
            }
            
            self.optimization_history.append(performance_data)
            self.logger.info(f"Performance context '{label}': {performance_data['duration_seconds']:.2f}s, "
                           f"Memory: {performance_data['memory_delta_mb']:+.2f}MB")
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태"""
        memory_usage = self.memory_manager.get_memory_usage()
        cpu_usage = self.cpu_optimizer.get_cpu_usage()
        cache_stats = self.cache.get_stats()
        
        # 건강 점수 계산 (0-100)
        health_score = 100
        
        # 메모리 점수 (최대 -40점)
        if memory_usage['percent'] > 90:
            health_score -= 40
        elif memory_usage['percent'] > 80:
            health_score -= 20
        elif memory_usage['percent'] > 70:
            health_score -= 10
        
        # CPU 점수 (최대 -30점)
        if cpu_usage['process_percent'] > 80:
            health_score -= 30
        elif cpu_usage['process_percent'] > 60:
            health_score -= 15
        elif cpu_usage['process_percent'] > 40:
            health_score -= 5
        
        # 캐시 점수 (최대 -20점)
        if cache_stats['hit_rate_percent'] < 50:
            health_score -= 20
        elif cache_stats['hit_rate_percent'] < 70:
            health_score -= 10
        elif cache_stats['hit_rate_percent'] < 85:
            health_score -= 5
        
        # 스왑 사용량 (최대 -10점)
        if memory_usage['swap_mb'] > 100:
            health_score -= 10
        elif memory_usage['swap_mb'] > 50:
            health_score -= 5
        
        health_status = "Excellent" if health_score >= 90 else \
                       "Good" if health_score >= 75 else \
                       "Fair" if health_score >= 60 else \
                       "Poor" if health_score >= 40 else "Critical"
        
        return {
            'health_score': max(0, health_score),
            'health_status': health_status,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'cache_stats': cache_stats,
            'recommendations': self._generate_recommendations(memory_usage, cpu_usage, cache_stats)
        }
    
    def _generate_recommendations(self, memory_usage, cpu_usage, cache_stats) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        if memory_usage['percent'] > 80:
            recommendations.append("메모리 사용량이 높습니다. 가비지 컬렉션을 실행하거나 불필요한 객체를 정리하세요.")
        
        if cpu_usage['process_percent'] > 70:
            recommendations.append("CPU 사용량이 높습니다. CPU 집약적 작업을 백그라운드로 이동하세요.")
        
        if cache_stats['hit_rate_percent'] < 70:
            recommendations.append("캐시 히트률이 낮습니다. 캐싱 전략을 검토하세요.")
        
        if memory_usage['swap_mb'] > 50:
            recommendations.append("스왑 사용량이 높습니다. 메모리를 확보하거나 시스템을 재시작하세요.")
        
        if len(recommendations) == 0:
            recommendations.append("시스템이 최적화된 상태입니다.")
        
        return recommendations
    
    def auto_optimize(self) -> Dict[str, Any]:
        """자동 최적화 실행"""
        optimization_results = {}
        
        # 메모리 최적화
        memory_result = self.memory_manager.optimize_memory()
        optimization_results['memory'] = memory_result
        
        # 캐시 정리 (히트률이 낮은 경우)
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate_percent'] < 50:
            self.cache.clear()
            optimization_results['cache'] = {'action': 'cleared', 'reason': 'low hit rate'}
        
        return optimization_results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """종합 성능 통계"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'cpu_stats': self.cpu_optimizer.get_cpu_stats(),
            'cache_stats': self.cache.get_stats(),
            'optimization_history': list(self.optimization_history)[-10:],  # 최근 10개
            'process_info': {
                'pid': os.getpid(),
                'threads': threading.active_count(),
                'file_descriptors': len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 'N/A'
            }
        }


# 전역 최적화기 인스턴스
global_optimizer = PerformanceOptimizer()


# 편의 함수들
def optimize(cache_key: str = None, cpu_bound: bool = False):
    """최적화 데코레이터 (편의 함수)"""
    return global_optimizer.optimize_function(cache_key=cache_key, cpu_bound=cpu_bound)


def performance_context(label: str = ""):
    """성능 컨텍스트 (편의 함수)"""
    return global_optimizer.performance_context(label)


def system_health():
    """시스템 건강 상태 (편의 함수)"""
    return global_optimizer.get_system_health()


def auto_optimize():
    """자동 최적화 (편의 함수)"""
    return global_optimizer.auto_optimize()


# 사용 예시
if __name__ == "__main__":
    
    # 최적화 예시
    @optimize(cache_key="example", cpu_bound=True)
    def cpu_intensive_function(n: int):
        return sum(i * i for i in range(n))
    
    @optimize(cache_key="async_example")
    async def async_function():
        await asyncio.sleep(0.1)
        return "async result"
    
    # 테스트 실행
    async def test_performance_system():
        print("⚡ 성능 최적화 시스템 테스트")
        
        with performance_context("performance_test"):
            # CPU 집약적 작업
            result1 = cpu_intensive_function(10000)
            print(f"CPU 집약적 작업 결과: {result1}")
            
            # 비동기 작업
            result2 = await async_function()
            print(f"비동기 작업 결과: {result2}")
            
            # 캐시된 결과 (두 번째 호출)
            result3 = cpu_intensive_function(10000)
            print(f"캐시된 결과: {result3}")
        
        # 시스템 건강 상태 확인
        health = system_health()
        print(f"\n🏥 시스템 건강 상태: {health['health_status']} ({health['health_score']}/100)")
        
        # 자동 최적화
        optimization_result = auto_optimize()
        print(f"🔧 자동 최적화 결과: {optimization_result}")
        
        # 종합 통계
        stats = global_optimizer.get_comprehensive_stats()
        print(f"\n📊 종합 성능 통계:")
        print(json.dumps(stats, indent=2, default=str))
    
    # 테스트 실행
    asyncio.run(test_performance_system())