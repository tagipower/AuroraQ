#!/usr/bin/env python3
"""
VPS Deployment 고급 디버깅 시스템
실시간 디버깅, 프로파일링, 트레이싱, 성능 모니터링
"""

import asyncio
import functools
import inspect
import json
import logging
import psutil
import sys
import time
import traceback
import gc
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from collections import defaultdict, deque
import threading
import contextlib


class PerformanceProfiler:
    """성능 프로파일러"""
    
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.function_calls = deque(maxlen=max_records)
        self.memory_snapshots = deque(maxlen=max_records)
        self.slow_calls = deque(maxlen=100)
        self.error_calls = deque(maxlen=100)
        
    def record_call(self, func_name: str, duration: float, memory_before: float, 
                   memory_after: float, args_info: str = "", error: Optional[str] = None):
        """함수 호출 기록"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'duration_ms': duration * 1000,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before,
            'args_info': args_info,
            'error': error
        }
        
        self.function_calls.append(record)
        
        # 느린 호출 기록 (100ms 이상)
        if duration > 0.1:
            self.slow_calls.append(record)
        
        # 에러 호출 기록
        if error:
            self.error_calls.append(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if not self.function_calls:
            return {"message": "No function calls recorded"}
        
        calls = list(self.function_calls)
        durations = [call['duration_ms'] for call in calls]
        memory_deltas = [call['memory_delta_mb'] for call in calls]
        
        function_stats = defaultdict(list)
        for call in calls:
            function_stats[call['function']].append(call['duration_ms'])
        
        return {
            'total_calls': len(calls),
            'avg_duration_ms': sum(durations) / len(durations),
            'max_duration_ms': max(durations),
            'min_duration_ms': min(durations),
            'slow_calls_count': len(self.slow_calls),
            'error_calls_count': len(self.error_calls),
            'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
            'max_memory_delta_mb': max(memory_deltas),
            'top_slow_functions': self._get_top_slow_functions(function_stats),
            'recent_errors': list(self.error_calls)[-5:] if self.error_calls else []
        }
    
    def _get_top_slow_functions(self, function_stats: Dict[str, List[float]], top_n: int = 5) -> List[Dict]:
        """가장 느린 함수들 반환"""
        avg_times = [(func, sum(times)/len(times), len(times)) 
                    for func, times in function_stats.items()]
        avg_times.sort(key=lambda x: x[1], reverse=True)
        
        return [{'function': func, 'avg_duration_ms': avg_time, 'call_count': count} 
                for func, avg_time, count in avg_times[:top_n]]


class DebugTracer:
    """디버깅 트레이서"""
    
    def __init__(self, max_trace_depth: int = 10):
        self.max_trace_depth = max_trace_depth
        self.trace_stack = []
        self.call_graph = defaultdict(list)
        self.enabled = False
        
    def enable(self):
        """트레이싱 활성화"""
        self.enabled = True
        sys.settrace(self._trace_calls)
    
    def disable(self):
        """트레이싱 비활성화"""
        self.enabled = False
        sys.settrace(None)
    
    def _trace_calls(self, frame, event, arg):
        """함수 호출 트레이스"""
        if not self.enabled:
            return None
        
        if event == 'call':
            func_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            if len(self.trace_stack) < self.max_trace_depth:
                call_info = {
                    'function': func_name,
                    'file': filename,
                    'line': lineno,
                    'timestamp': time.time(),
                    'locals': dict(frame.f_locals) if len(frame.f_locals) < 10 else {}
                }
                self.trace_stack.append(call_info)
                
                # 호출 그래프 구성
                if len(self.trace_stack) > 1:
                    caller = self.trace_stack[-2]['function']
                    self.call_graph[caller].append(func_name)
        
        elif event == 'return':
            if self.trace_stack:
                call_info = self.trace_stack.pop()
                duration = time.time() - call_info['timestamp']
                call_info['duration'] = duration
        
        return self._trace_calls


class MemoryTracker:
    """메모리 추적기"""
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
        self.snapshots = []
        self.gc_stats = []
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def snapshot(self, label: str = ""):
        """메모리 스냅샷"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'memory': self._get_memory_usage(),
            'gc_stats': {
                'collections': gc.get_stats(),
                'objects': len(gc.get_objects())
            }
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def analyze_leaks(self) -> Dict[str, Any]:
        """메모리 누수 분석"""
        if len(self.snapshots) < 2:
            return {"message": "Need at least 2 snapshots for leak analysis"}
        
        first = self.snapshots[0]['memory']
        last = self.snapshots[-1]['memory']
        
        leak_analysis = {
            'memory_increase_mb': last['rss_mb'] - first['rss_mb'],
            'memory_increase_percent': ((last['rss_mb'] - first['rss_mb']) / first['rss_mb']) * 100,
            'snapshots_analyzed': len(self.snapshots),
            'potential_leak': False,
            'recommendations': []
        }
        
        # 메모리 증가가 50MB 이상이면 잠재적 누수
        if leak_analysis['memory_increase_mb'] > 50:
            leak_analysis['potential_leak'] = True
            leak_analysis['recommendations'].append("메모리 사용량이 지속적으로 증가하고 있습니다.")
            leak_analysis['recommendations'].append("가비지 컬렉션을 수동으로 실행해보세요: gc.collect()")
            leak_analysis['recommendations'].append("큰 객체들이 제대로 해제되는지 확인하세요.")
        
        return leak_analysis


class AsyncDebugger:
    """비동기 코드 디버거"""
    
    def __init__(self):
        self.pending_tasks = {}
        self.completed_tasks = deque(maxlen=100)
        self.task_stats = defaultdict(int)
        
    def track_task(self, task_name: str, coro):
        """태스크 추적"""
        task_id = id(coro)
        self.pending_tasks[task_id] = {
            'name': task_name,
            'created_at': time.time(),
            'coro': coro
        }
        
        async def tracked_coro():
            try:
                start_time = time.time()
                result = await coro
                duration = time.time() - start_time
                
                # 완료된 태스크 기록
                self.completed_tasks.append({
                    'name': task_name,
                    'duration': duration,
                    'completed_at': time.time(),
                    'success': True
                })
                
                self.task_stats[f"{task_name}_success"] += 1
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 실패한 태스크 기록
                self.completed_tasks.append({
                    'name': task_name,
                    'duration': duration,
                    'completed_at': time.time(),
                    'success': False,
                    'error': str(e)
                })
                
                self.task_stats[f"{task_name}_error"] += 1
                raise
            finally:
                # pending에서 제거
                self.pending_tasks.pop(task_id, None)
        
        return tracked_coro()
    
    def get_async_stats(self) -> Dict[str, Any]:
        """비동기 통계"""
        current_time = time.time()
        
        # 오래 실행중인 태스크 찾기
        long_running = []
        for task_id, task_info in self.pending_tasks.items():
            duration = current_time - task_info['created_at']
            if duration > 10:  # 10초 이상
                long_running.append({
                    'name': task_info['name'],
                    'duration': duration,
                    'task_id': task_id
                })
        
        return {
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'long_running_tasks': long_running,
            'task_stats': dict(self.task_stats),
            'recent_completions': list(self.completed_tasks)[-5:]
        }


class AdvancedDebugger:
    """고급 디버깅 시스템 메인 클래스"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.profiler = PerformanceProfiler()
        self.tracer = DebugTracer()
        self.memory_tracker = MemoryTracker()
        self.async_debugger = AsyncDebugger()
        
        # 로거 설정
        self.logger = logging.getLogger('AdvancedDebugger')
        self.logger.setLevel(log_level)
        
        # 파일 핸들러
        handler = logging.FileHandler('debug_system.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # 기본 메모리 스냅샷
        self.memory_tracker.snapshot("system_start")
    
    def profile_function(self, include_args: bool = False):
        """함수 프로파일링 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                args_info = ""
                if include_args:
                    args_info = f"args: {args[:3]}..." if args else ""
                
                try:
                    result = func(*args, **kwargs)
                    error = None
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    self.profiler.record_call(
                        func.__name__, duration, memory_before, memory_after, args_info, error
                    )
                
                return result
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                args_info = ""
                if include_args:
                    args_info = f"args: {args[:3]}..." if args else ""
                
                try:
                    result = await func(*args, **kwargs)
                    error = None
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    self.profiler.record_call(
                        func.__name__, duration, memory_before, memory_after, args_info, error
                    )
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def debug_context(self, label: str = ""):
        """디버그 컨텍스트 매니저"""
        @contextlib.contextmanager
        def context():
            self.logger.info(f"DEBUG CONTEXT START: {label}")
            start_snapshot = self.memory_tracker.snapshot(f"{label}_start")
            
            try:
                yield self
            except Exception as e:
                self.logger.error(f"ERROR in {label}: {e}")
                traceback.print_exc()
                raise
            finally:
                end_snapshot = self.memory_tracker.snapshot(f"{label}_end")
                memory_delta = end_snapshot['memory']['rss_mb'] - start_snapshot['memory']['rss_mb']
                self.logger.info(f"DEBUG CONTEXT END: {label} (Memory delta: {memory_delta:.2f}MB)")
        
        return context()
    
    def track_async_task(self, task_name: str):
        """비동기 태스크 추적 데코레이터"""
        def decorator(coro_func):
            @functools.wraps(coro_func)
            async def wrapper(*args, **kwargs):
                coro = coro_func(*args, **kwargs)
                return await self.async_debugger.track_task(task_name, coro)
            return wrapper
        return decorator
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """종합 디버그 리포트"""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.profiler.get_stats(),
            'memory_analysis': self.memory_tracker.analyze_leaks(),
            'async_stats': self.async_debugger.get_async_stats(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'process_threads': psutil.Process().num_threads()
            }
        }
    
    def optimize_memory(self):
        """메모리 최적화 수행"""
        before_snapshot = self.memory_tracker.snapshot("before_optimization")
        
        # 가비지 컬렉션 수행
        collected = gc.collect()
        
        after_snapshot = self.memory_tracker.snapshot("after_optimization")
        
        memory_freed = before_snapshot['memory']['rss_mb'] - after_snapshot['memory']['rss_mb']
        
        self.logger.info(f"Memory optimization: freed {memory_freed:.2f}MB, collected {collected} objects")
        
        return {
            'memory_freed_mb': memory_freed,
            'objects_collected': collected,
            'before_memory_mb': before_snapshot['memory']['rss_mb'],
            'after_memory_mb': after_snapshot['memory']['rss_mb']
        }
    
    def emergency_debug_dump(self, filename: str = None):
        """긴급 디버그 덤프"""
        if filename is None:
            filename = f"emergency_debug_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dump_data = {
            'timestamp': datetime.now().isoformat(),
            'emergency_dump': True,
            'system_state': self.get_comprehensive_report(),
            'call_stack': traceback.format_stack(),
            'environment': dict(os.environ),
            'memory_snapshots': list(self.memory_tracker.snapshots),
            'recent_function_calls': list(self.profiler.function_calls)[-50:],
            'gc_objects_count': len(gc.get_objects()),
            'threads': [str(thread) for thread in threading.enumerate()]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(dump_data, f, indent=2, default=str)
            
            self.logger.critical(f"Emergency debug dump saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency dump: {e}")
            return None


# 전역 디버거 인스턴스
global_debugger = AdvancedDebugger()


# 편의 함수들
def profile(include_args: bool = False):
    """프로파일링 데코레이터 (편의 함수)"""
    return global_debugger.profile_function(include_args)


def debug_context(label: str = ""):
    """디버그 컨텍스트 (편의 함수)"""
    return global_debugger.debug_context(label)


def track_async(task_name: str):
    """비동기 추적 데코레이터 (편의 함수)"""
    return global_debugger.track_async_task(task_name)


def memory_snapshot(label: str = ""):
    """메모리 스냅샷 (편의 함수)"""
    return global_debugger.memory_tracker.snapshot(label)


def debug_report():
    """디버그 리포트 (편의 함수)"""
    report = global_debugger.get_comprehensive_report()
    print(json.dumps(report, indent=2, default=str))
    return report


def emergency_dump():
    """긴급 덤프 (편의 함수)"""
    return global_debugger.emergency_debug_dump()


# 사용 예시
if __name__ == "__main__":
    
    # 프로파일링 예시
    @profile(include_args=True)
    def example_function(n: int):
        time.sleep(0.1)  # 시뮬레이션
        return sum(range(n))
    
    @profile()
    async def example_async_function():
        await asyncio.sleep(0.05)
        return "async result"
    
    # 테스트 실행
    async def test_debug_system():
        print("🐛 고급 디버깅 시스템 테스트")
        
        # 메모리 스냅샷
        memory_snapshot("test_start")
        
        # 디버그 컨텍스트 사용
        with debug_context("test_context"):
            # 함수 호출 테스트
            result = example_function(1000)
            print(f"동기 함수 결과: {result}")
            
            # 비동기 함수 테스트
            async_result = await example_async_function()
            print(f"비동기 함수 결과: {async_result}")
        
        # 메모리 스냅샷
        memory_snapshot("test_end")
        
        # 메모리 최적화
        optimization_result = global_debugger.optimize_memory()
        print(f"메모리 최적화: {optimization_result}")
        
        # 리포트 생성
        print("\n📊 디버그 리포트:")
        debug_report()
    
    # 테스트 실행
    asyncio.run(test_debug_system())