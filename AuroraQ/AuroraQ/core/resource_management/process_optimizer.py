#!/usr/bin/env python3
"""
프로세스 최적화 관리자
P5: 시스템 리소스 관리 및 최적화 - 프로세스 레벨 최적화
"""

import sys
import os
import psutil
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import gc
import warnings
from collections import defaultdict, deque

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """최적화 레벨"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"

class ProcessPriority(Enum):
    """프로세스 우선순위"""
    CRITICAL = "critical"      # 중요한 프로세스 (예: 거래 엔진)
    HIGH = "high"             # 높은 우선순위 (예: 데이터 수집)
    NORMAL = "normal"         # 일반 프로세스
    LOW = "low"              # 낮은 우선순위 (예: 백업)
    BACKGROUND = "background"  # 백그라운드 (예: 로그 처리)

class OptimizationStrategy(Enum):
    """최적화 전략"""
    MEMORY_FOCUSED = "memory_focused"
    CPU_FOCUSED = "cpu_focused"
    BALANCED = "balanced"
    LATENCY_FOCUSED = "latency_focused"

@dataclass
class ProcessInfo:
    """프로세스 정보"""
    pid: int
    name: str
    priority: ProcessPriority = ProcessPriority.NORMAL
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    thread_count: int = 0
    file_descriptors: int = 0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    status: str = "running"
    create_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'pid': self.pid,
            'name': self.name,
            'priority': self.priority.value,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'thread_count': self.thread_count,
            'file_descriptors': self.file_descriptors,
            'io_read_mb': self.io_read_mb,
            'io_write_mb': self.io_write_mb,
            'status': self.status,
            'create_time': self.create_time.isoformat()
        }

@dataclass
class OptimizationAction:
    """최적화 액션"""
    action_type: str
    target_pid: Optional[int] = None
    target_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_benefit: str = ""
    risk_level: str = "low"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'action_type': self.action_type,
            'target_pid': self.target_pid,
            'target_name': self.target_name,
            'parameters': self.parameters,
            'expected_benefit': self.expected_benefit,
            'risk_level': self.risk_level
        }

@dataclass
class OptimizationResult:
    """최적화 결과"""
    action: OptimizationAction
    success: bool
    timestamp: datetime
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def calculate_improvement(self) -> Dict[str, float]:
        """개선도 계산"""
        improvements = {}
        for metric, after_value in self.after_metrics.items():
            if metric in self.before_metrics:
                before_value = self.before_metrics[metric]
                if before_value > 0:
                    improvement = ((before_value - after_value) / before_value) * 100
                    improvements[metric] = improvement
        return improvements

class ProcessOptimizer:
    """프로세스 최적화 관리자"""
    
    def __init__(self, config_file: str = "process_optimizer_config.json"):
        self.config_file = config_file
        
        # 프로세스 우선순위 맵핑
        self.process_priorities = {
            "python": ProcessPriority.HIGH,  # AuroraQ 메인 프로세스
            "chrome": ProcessPriority.LOW,
            "firefox": ProcessPriority.LOW,
            "notepad": ProcessPriority.BACKGROUND,
            "explorer": ProcessPriority.NORMAL
        }
        
        # 최적화 설정
        self.optimization_settings = {
            "enable_auto_optimization": True,
            "optimization_interval_seconds": 60,
            "memory_threshold_mb": 1000,
            "cpu_threshold_percent": 80,
            "max_optimizations_per_hour": 10,
            "emergency_memory_threshold_mb": 2000
        }
        
        # 프로세스 추적
        self.tracked_processes: Dict[int, ProcessInfo] = {}
        self.process_history: deque = deque(maxlen=1000)
        self.optimization_history: List[OptimizationResult] = []
        
        # 최적화 통계
        self.stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "memory_freed_mb": 0.0,
            "cpu_saved_percent": 0.0,
            "last_optimization": None,
            "optimization_start_time": None
        }
        
        # 제어 변수
        self._optimization_active = False
        self._optimizer_thread = None
        self._lock = threading.RLock()
        
        # 설정 로드
        self._load_configuration()
        
        logger.info("Process optimizer initialized")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 프로세스 우선순위 로드
                priorities = config.get('process_priorities', {})
                for name, priority_str in priorities.items():
                    try:
                        self.process_priorities[name] = ProcessPriority(priority_str)
                    except ValueError:
                        continue
                
                # 최적화 설정 로드
                self.optimization_settings.update(config.get('optimization_settings', {}))
                
                # 통계 로드
                self.stats.update(config.get('stats', {}))
                
                logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config = {
                'process_priorities': {
                    name: priority.value 
                    for name, priority in self.process_priorities.items()
                },
                'optimization_settings': self.optimization_settings,
                'stats': self.stats
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def scan_processes(self) -> List[ProcessInfo]:
        """프로세스 스캔"""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 
                                           'memory_percent', 'num_threads', 'status', 'create_time']):
                try:
                    info = proc.info
                    if not info or not info.get('pid'):
                        continue
                    
                    # 프로세스 우선순위 결정
                    process_name = info.get('name', '').lower()
                    priority = ProcessPriority.NORMAL
                    for name_pattern, proc_priority in self.process_priorities.items():
                        if name_pattern.lower() in process_name:
                            priority = proc_priority
                            break
                    
                    # 메모리 정보 계산
                    memory_info = info.get('memory_info')
                    memory_mb = 0.0
                    if memory_info:
                        memory_mb = memory_info.rss / 1024 / 1024
                    
                    # 파일 디스크립터 수 (Unix에서만)
                    file_descriptors = 0
                    try:
                        file_descriptors = proc.num_fds()
                    except (AttributeError, psutil.AccessDenied):
                        pass
                    
                    # I/O 정보 (가능한 경우)
                    io_read_mb = 0.0
                    io_write_mb = 0.0
                    try:
                        io_counters = proc.io_counters()
                        io_read_mb = io_counters.read_bytes / 1024 / 1024
                        io_write_mb = io_counters.write_bytes / 1024 / 1024
                    except (AttributeError, psutil.AccessDenied):
                        pass
                    
                    # 생성 시간
                    create_time = datetime.now()
                    try:
                        create_time = datetime.fromtimestamp(info.get('create_time', time.time()))
                    except (OSError, ValueError):
                        pass
                    
                    process_info = ProcessInfo(
                        pid=info['pid'],
                        name=info.get('name', 'unknown'),
                        priority=priority,
                        cpu_percent=info.get('cpu_percent', 0.0) or 0.0,
                        memory_mb=memory_mb,
                        memory_percent=info.get('memory_percent', 0.0) or 0.0,
                        thread_count=info.get('num_threads', 0) or 0,
                        file_descriptors=file_descriptors,
                        io_read_mb=io_read_mb,
                        io_write_mb=io_write_mb,
                        status=info.get('status', 'unknown'),
                        create_time=create_time
                    )
                    
                    processes.append(process_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    logger.debug(f"Error scanning process {info.get('pid', 'unknown')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to scan processes: {e}")
        
        return processes
    
    def analyze_processes(self, processes: List[ProcessInfo]) -> List[OptimizationAction]:
        """프로세스 분석 및 최적화 액션 생성"""
        actions = []
        
        try:
            # 메모리 사용량 기준 정렬
            memory_sorted = sorted(processes, key=lambda p: p.memory_mb, reverse=True)
            
            # CPU 사용량 기준 정렬
            cpu_sorted = sorted(processes, key=lambda p: p.cpu_percent, reverse=True)
            
            total_memory = sum(p.memory_mb for p in processes)
            total_cpu = sum(p.cpu_percent for p in processes)
            
            # 1. 메모리 최적화
            if total_memory > self.optimization_settings["memory_threshold_mb"]:
                # 메모리 사용량이 높은 low/background 우선순위 프로세스 찾기
                for proc in memory_sorted[:5]:  # 상위 5개만 확인
                    if (proc.priority in [ProcessPriority.LOW, ProcessPriority.BACKGROUND] and
                        proc.memory_mb > 100):  # 100MB 이상 사용
                        
                        actions.append(OptimizationAction(
                            action_type="reduce_memory",
                            target_pid=proc.pid,
                            target_name=proc.name,
                            parameters={"target_reduction_mb": min(proc.memory_mb * 0.3, 500)},
                            expected_benefit=f"메모리 {proc.memory_mb * 0.3:.0f}MB 절약",
                            risk_level="low"
                        ))
            
            # 2. CPU 최적화
            if total_cpu > self.optimization_settings["cpu_threshold_percent"]:
                for proc in cpu_sorted[:3]:  # 상위 3개만 확인
                    if (proc.priority in [ProcessPriority.LOW, ProcessPriority.BACKGROUND] and
                        proc.cpu_percent > 5.0):  # 5% 이상 사용
                        
                        actions.append(OptimizationAction(
                            action_type="reduce_cpu",
                            target_pid=proc.pid,
                            target_name=proc.name,
                            parameters={"target_priority": "below_normal"},
                            expected_benefit=f"CPU {proc.cpu_percent:.1f}% 절약",
                            risk_level="low"
                        ))
            
            # 3. 스레드 최적화
            high_thread_processes = [p for p in processes if p.thread_count > 50]
            for proc in high_thread_processes:
                if proc.priority in [ProcessPriority.LOW, ProcessPriority.BACKGROUND]:
                    actions.append(OptimizationAction(
                        action_type="optimize_threads",
                        target_pid=proc.pid,
                        target_name=proc.name,
                        parameters={"max_threads": min(proc.thread_count, 25)},
                        expected_benefit=f"스레드 {proc.thread_count - 25}개 절약",
                        risk_level="medium"
                    ))
            
            # 4. 응급 상황 처리
            emergency_memory = self.optimization_settings["emergency_memory_threshold_mb"]
            if total_memory > emergency_memory:
                # 가장 큰 비중요 프로세스 중단
                for proc in memory_sorted:
                    if proc.priority == ProcessPriority.BACKGROUND and proc.memory_mb > 200:
                        actions.append(OptimizationAction(
                            action_type="suspend_process",
                            target_pid=proc.pid,
                            target_name=proc.name,
                            parameters={"suspend_duration_minutes": 5},
                            expected_benefit=f"응급 메모리 {proc.memory_mb:.0f}MB 확보",
                            risk_level="high"
                        ))
                        break
            
        except Exception as e:
            logger.error(f"Failed to analyze processes: {e}")
        
        return actions
    
    def execute_optimization_action(self, action: OptimizationAction) -> OptimizationResult:
        """최적화 액션 실행"""
        result = OptimizationResult(
            action=action,
            success=False,
            timestamp=datetime.now()
        )
        
        try:
            # 실행 전 메트릭 수집
            if action.target_pid:
                try:
                    proc = psutil.Process(action.target_pid)
                    result.before_metrics = {
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'thread_count': proc.num_threads()
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result.error_message = "Process not found or access denied"
                    return result
            
            # 액션 타입별 실행
            if action.action_type == "reduce_memory":
                result.success = self._reduce_memory_usage(action)
            elif action.action_type == "reduce_cpu":
                result.success = self._reduce_cpu_usage(action)
            elif action.action_type == "optimize_threads":
                result.success = self._optimize_threads(action)
            elif action.action_type == "suspend_process":
                result.success = self._suspend_process(action)
            else:
                result.error_message = f"Unknown action type: {action.action_type}"
                return result
            
            # 실행 후 메트릭 수집 (약간의 지연 후)
            if action.target_pid and result.success:
                time.sleep(1)  # 1초 대기
                try:
                    proc = psutil.Process(action.target_pid)
                    result.after_metrics = {
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'thread_count': proc.num_threads()
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # 프로세스가 종료되었을 수 있음
                    result.after_metrics = {
                        'cpu_percent': 0.0,
                        'memory_mb': 0.0,
                        'thread_count': 0
                    }
            
            # 통계 업데이트
            if result.success:
                self.stats["successful_optimizations"] += 1
                improvements = result.calculate_improvement()
                if "memory_mb" in improvements:
                    self.stats["memory_freed_mb"] += improvements["memory_mb"]
                if "cpu_percent" in improvements:
                    self.stats["cpu_saved_percent"] += improvements["cpu_percent"]
            
            self.stats["total_optimizations"] += 1
            self.stats["last_optimization"] = datetime.now().isoformat()
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Failed to execute optimization action: {e}")
        
        return result
    
    def _reduce_memory_usage(self, action: OptimizationAction) -> bool:
        """메모리 사용량 감소"""
        try:
            if not action.target_pid:
                return False
            
            proc = psutil.Process(action.target_pid)
            
            # 실제 구현에서는 프로세스별 메모리 최적화 로직
            # 여기서는 시뮬레이션
            logger.info(f"Simulating memory reduction for process {proc.name()} (PID: {action.target_pid})")
            
            # 가비지 컬렉션 (Python 프로세스인 경우)
            if "python" in proc.name().lower():
                gc.collect()
            
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Cannot reduce memory usage: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to reduce memory usage: {e}")
            return False
    
    def _reduce_cpu_usage(self, action: OptimizationAction) -> bool:
        """CPU 사용량 감소"""
        try:
            if not action.target_pid:
                return False
            
            proc = psutil.Process(action.target_pid)
            
            # 프로세스 우선순위 낮추기
            try:
                if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
                    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    # Unix 계열
                    current_nice = proc.nice()
                    proc.nice(min(current_nice + 5, 19))  # nice 값 증가 (우선순위 감소)
                
                logger.info(f"Reduced CPU priority for process {proc.name()} (PID: {action.target_pid})")
                return True
                
            except psutil.AccessDenied:
                logger.warning(f"Access denied to change priority for PID {action.target_pid}")
                return False
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Cannot reduce CPU usage: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to reduce CPU usage: {e}")
            return False
    
    def _optimize_threads(self, action: OptimizationAction) -> bool:
        """스레드 최적화"""
        try:
            # 실제 구현에서는 애플리케이션별 스레드 풀 조정
            # 여기서는 시뮬레이션
            logger.info(f"Simulating thread optimization for PID {action.target_pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize threads: {e}")
            return False
    
    def _suspend_process(self, action: OptimizationAction) -> bool:
        """프로세스 일시 중단"""
        try:
            if not action.target_pid:
                return False
            
            proc = psutil.Process(action.target_pid)
            
            # 중요한 프로세스는 중단하지 않음
            if action.target_name and any(critical in action.target_name.lower() 
                                        for critical in ['python', 'system', 'explorer', 'winlogon']):
                logger.warning(f"Refusing to suspend critical process: {action.target_name}")
                return False
            
            # 프로세스 일시 중단
            proc.suspend()
            logger.warning(f"Suspended process {proc.name()} (PID: {action.target_pid})")
            
            # 지정된 시간 후 재개하는 스레드 시작
            duration = action.parameters.get("suspend_duration_minutes", 5)
            
            def resume_process():
                time.sleep(duration * 60)
                try:
                    proc.resume()
                    logger.info(f"Resumed process {proc.name()} (PID: {action.target_pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            resume_thread = threading.Thread(target=resume_process, daemon=True)
            resume_thread.start()
            
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Cannot suspend process: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to suspend process: {e}")
            return False
    
    def run_optimization_cycle(self):
        """최적화 사이클 실행"""
        try:
            logger.debug("Running process optimization cycle...")
            
            # 프로세스 스캔
            processes = self.scan_processes()
            
            # 프로세스 정보 업데이트
            with self._lock:
                self.tracked_processes.clear()
                for proc in processes:
                    self.tracked_processes[proc.pid] = proc
                
                # 히스토리에 추가
                self.process_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'process_count': len(processes),
                    'total_memory_mb': sum(p.memory_mb for p in processes),
                    'total_cpu_percent': sum(p.cpu_percent for p in processes)
                })
            
            # 최적화 액션 분석
            actions = self.analyze_processes(processes)
            
            if not actions:
                logger.debug("No optimization actions needed")
                return
            
            # 시간당 최적화 제한 확인
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_optimizations = [
                r for r in self.optimization_history 
                if r.timestamp > one_hour_ago
            ]
            
            max_per_hour = self.optimization_settings["max_optimizations_per_hour"]
            if len(recent_optimizations) >= max_per_hour:
                logger.info(f"Optimization limit reached ({len(recent_optimizations)}/{max_per_hour} per hour)")
                return
            
            # 액션 실행 (위험도 낮은 것부터)
            actions.sort(key=lambda a: {"low": 1, "medium": 2, "high": 3}[a.risk_level])
            
            for action in actions[:3]:  # 최대 3개 액션만 실행
                result = self.execute_optimization_action(action)
                
                with self._lock:
                    self.optimization_history.append(result)
                    if len(self.optimization_history) > 100:
                        self.optimization_history.pop(0)
                
                if result.success:
                    logger.info(f"Optimization successful: {action.action_type} for {action.target_name}")
                    improvements = result.calculate_improvement()
                    if improvements:
                        logger.info(f"Improvements: {improvements}")
                else:
                    logger.warning(f"Optimization failed: {result.error_message}")
                
                # 액션 간 간격
                time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    def start_optimization(self, interval_seconds: int = None):
        """자동 최적화 시작"""
        if self._optimization_active:
            logger.warning("Process optimization already active")
            return
        
        if interval_seconds is None:
            interval_seconds = self.optimization_settings["optimization_interval_seconds"]
        
        if not self.optimization_settings["enable_auto_optimization"]:
            logger.info("Auto optimization is disabled")
            return
        
        self._optimization_active = True
        self.stats["optimization_start_time"] = datetime.now().isoformat()
        
        def optimization_loop():
            while self._optimization_active:
                try:
                    self.run_optimization_cycle()
                    
                    # 설정 저장 (주기적으로)
                    if self.stats["total_optimizations"] % 10 == 0:
                        self._save_configuration()
                    
                    # 대기
                    for _ in range(interval_seconds):
                        if not self._optimization_active:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    time.sleep(30)  # 에러 시 30초 후 재시도
        
        self._optimizer_thread = threading.Thread(target=optimization_loop, daemon=True)
        self._optimizer_thread.start()
        
        logger.info(f"Process optimization started (interval: {interval_seconds}s)")
    
    def stop_optimization(self):
        """자동 최적화 중지"""
        if not self._optimization_active:
            return
        
        self._optimization_active = False
        if self._optimizer_thread and self._optimizer_thread.is_alive():
            self._optimizer_thread.join(timeout=10)
        
        self._save_configuration()
        logger.info("Process optimization stopped")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """최적화 요약"""
        try:
            current_processes = self.scan_processes()
            
            # 최근 최적화 결과
            recent_results = self.optimization_history[-10:] if self.optimization_history else []
            successful_recent = [r for r in recent_results if r.success]
            
            # 프로세스 통계
            total_memory = sum(p.memory_mb for p in current_processes)
            total_cpu = sum(p.cpu_percent for p in current_processes)
            high_priority_processes = [p for p in current_processes 
                                     if p.priority in [ProcessPriority.CRITICAL, ProcessPriority.HIGH]]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'optimization_active': self._optimization_active,
                'current_status': {
                    'total_processes': len(current_processes),
                    'total_memory_mb': total_memory,
                    'total_cpu_percent': total_cpu,
                    'high_priority_processes': len(high_priority_processes)
                },
                'optimization_stats': self.stats,
                'recent_optimizations': {
                    'total_recent': len(recent_results),
                    'successful_recent': len(successful_recent),
                    'success_rate': len(successful_recent) / len(recent_results) * 100 if recent_results else 0
                },
                'top_memory_consumers': [
                    {'name': p.name, 'pid': p.pid, 'memory_mb': p.memory_mb, 'priority': p.priority.value}
                    for p in sorted(current_processes, key=lambda x: x.memory_mb, reverse=True)[:5]
                ],
                'top_cpu_consumers': [
                    {'name': p.name, 'pid': p.pid, 'cpu_percent': p.cpu_percent, 'priority': p.priority.value}
                    for p in sorted(current_processes, key=lambda x: x.cpu_percent, reverse=True)[:5]
                ],
                'settings': self.optimization_settings
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization summary: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_optimization()
            
            with self._lock:
                self.tracked_processes.clear()
                self.process_history.clear()
                self.optimization_history.clear()
            
            logger.info("Process optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 프로세스 최적화기
_global_process_optimizer = None

def get_process_optimizer(config_file: str = None) -> ProcessOptimizer:
    """전역 프로세스 최적화기 반환"""
    global _global_process_optimizer
    if _global_process_optimizer is None:
        _global_process_optimizer = ProcessOptimizer(
            config_file or "process_optimizer_config.json"
        )
    return _global_process_optimizer

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_process_optimizer():
        print("🧪 Process Optimizer 테스트")
        
        optimizer = get_process_optimizer("test_process_config.json")
        
        print("\n1️⃣ 프로세스 스캔")
        processes = optimizer.scan_processes()
        print(f"  감지된 프로세스: {len(processes)}개")
        
        # 상위 메모리 사용 프로세스
        top_memory = sorted(processes, key=lambda p: p.memory_mb, reverse=True)[:3]
        for proc in top_memory:
            print(f"    {proc.name} (PID: {proc.pid}): {proc.memory_mb:.1f}MB")
        
        print("\n2️⃣ 최적화 액션 분석")
        actions = optimizer.analyze_processes(processes)
        print(f"  권장 액션: {len(actions)}개")
        for action in actions:
            print(f"    {action.action_type}: {action.expected_benefit} (위험도: {action.risk_level})")
        
        print("\n3️⃣ 최적화 실행 (시뮬레이션)")
        if actions:
            result = optimizer.execute_optimization_action(actions[0])
            print(f"    결과: {'성공' if result.success else '실패'}")
            if result.error_message:
                print(f"    오류: {result.error_message}")
        
        print("\n4️⃣ 자동 최적화 시작 (5초)")
        optimizer.start_optimization(interval_seconds=3)
        await asyncio.sleep(5)
        
        print("\n5️⃣ 최적화 요약")
        summary = optimizer.get_optimization_summary()
        print(f"    최적화 활성: {summary['optimization_active']}")
        print(f"    총 프로세스: {summary['current_status']['total_processes']}")
        print(f"    전체 메모리: {summary['current_status']['total_memory_mb']:.1f}MB")
        print(f"    총 최적화: {summary['optimization_stats']['total_optimizations']}")
        
        print("\n🎉 Process Optimizer 테스트 완료!")
        
        # 정리
        optimizer.cleanup()
        
        # 테스트 파일 정리
        test_file = Path("test_process_config.json")
        if test_file.exists():
            test_file.unlink()
    
    # 테스트 실행
    asyncio.run(test_process_optimizer())