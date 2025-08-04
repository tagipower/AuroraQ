#!/usr/bin/env python3
"""
Batch Processing Scheduler V2 for AuroraQ Sentiment Service
VPS 최적화 버전 - 리소스 효율성, 스마트 스케줄링, 에러 복구 강화
"""

import asyncio
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import statistics
import threading
from contextlib import asynccontextmanager

# 스케줄링 라이브러리 (fallback 포함)
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    # logger는 아래에서 정의되므로 일단 None으로 설정
    
    # 간단한 스케줄러 폴백 구현
    class AsyncIOScheduler:
        def __init__(self):
            self.jobs = []
            self.running = False
        
        def add_job(self, func, trigger=None, id=None, **kwargs):
            job = {'func': func, 'trigger': trigger, 'id': id, 'kwargs': kwargs}
            self.jobs.append(job)
            return job
        
        def start(self):
            self.running = True
            print("Simple scheduler started")
        
        def shutdown(self):
            self.running = False
            print("Simple scheduler stopped")
    
    class CronTrigger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    
    class IntervalTrigger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    
    class DateTrigger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

logger = logging.getLogger(__name__)

# VPS 환경 최적화 설정
os.environ.setdefault('TZ', 'UTC')

class ScheduleType(Enum):
    """스케줄 유형"""
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"
    ADAPTIVE = "adaptive"  # 리소스 기반 동적 스케줄링

class TaskPriority(Enum):
    """작업 우선순위"""
    CRITICAL = 1    # 즉시 실행
    HIGH = 2        # 리소스 여유 시 우선
    NORMAL = 3      # 일반적인 우선순위
    LOW = 4         # 시스템 여유 시만
    MAINTENANCE = 5  # 유지보수용

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    THROTTLED = "throttled"
    RETRYING = "retrying"

class ResourceThreshold(Enum):
    """리소스 임계값"""
    LOW = 0.3       # 30% 이하
    NORMAL = 0.6    # 60% 이하
    HIGH = 0.8      # 80% 이하
    CRITICAL = 0.95 # 95% 이하

@dataclass
class TaskMetrics:
    """작업 메트릭"""
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_runtime: float = 0.0
    avg_runtime: float = 0.0
    min_runtime: float = float('inf')
    max_runtime: float = 0.0
    last_execution: Optional[datetime] = None
    consecutive_failures: int = 0
    throttle_count: int = 0
    
    def update(self, runtime: float, success: bool):
        """메트릭 업데이트"""
        self.execution_count += 1
        self.total_runtime += runtime
        self.avg_runtime = self.total_runtime / self.execution_count
        
        if runtime < self.min_runtime:
            self.min_runtime = runtime
        if runtime > self.max_runtime:
            self.max_runtime = runtime
            
        if success:
            self.success_count += 1
            self.consecutive_failures = 0
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            
        self.last_execution = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        return self.success_count / self.execution_count if self.execution_count > 0 else 0.0

@dataclass
class ScheduledTaskV2:
    """향상된 스케줄된 작업"""
    task_id: str
    name: str
    function: Callable
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_runtime: int = 1800  # 30분 (VPS 최적화)
    retry_count: int = 2     # 재시도 감소
    timeout: int = 900       # 15분 (VPS 최적화)
    
    # 리소스 요구사항
    min_memory_mb: int = 100
    max_memory_mb: int = 1024  # 1GB 제한
    cpu_intensive: bool = False
    
    # 실행 제어
    allow_parallel: bool = False
    max_parallel_instances: int = 1
    throttle_on_failure: bool = True
    backoff_multiplier: float = 2.0
    
    # 상태
    status: TaskStatus = TaskStatus.PENDING
    current_instances: int = 0
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    execution_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # 적응형 스케줄링
    adaptive_interval: Optional[int] = None
    base_interval: Optional[int] = None
    
    def should_run(self, current_resources: Dict[str, float]) -> tuple[bool, str]:
        """실행 가능 여부 확인"""
        # 메모리 체크
        if current_resources.get('memory_mb', 0) + self.min_memory_mb > self.max_memory_mb:
            return False, "Insufficient memory"
        
        # CPU 집약적 작업의 경우 CPU 사용률 체크
        if self.cpu_intensive and current_resources.get('cpu_percent', 0) > 70:
            return False, "High CPU usage"
        
        # 병렬 인스턴스 체크
        if not self.allow_parallel and self.current_instances > 0:
            return False, "Already running"
        
        if self.current_instances >= self.max_parallel_instances:
            return False, "Max instances reached"
        
        # 연속 실패 시 스로틀링
        if self.throttle_on_failure and self.metrics.consecutive_failures >= 3:
            return False, "Throttled due to failures"
        
        return True, "Ready"

class ResourceMonitorV2:
    """향상된 리소스 모니터"""
    
    def __init__(self, history_size: int = 120):  # 2시간 히스토리
        self.process = psutil.Process()
        self.history_size = history_size
        self.cpu_history: deque = deque(maxlen=history_size)
        self.memory_history: deque = deque(maxlen=history_size)
        self.disk_history: deque = deque(maxlen=history_size)
        self.running_tasks: Dict[str, Dict] = {}
        self.monitor_lock = threading.RLock()
        
        # 알림 임계값
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_mb': 3072.0,     # 3GB
            'memory_percent': 80.0,
            'disk_percent': 85.0
        }
        
        # 마지막 알림 시간 (스팸 방지)
        self.last_alerts = {}
        self.alert_cooldown = 300  # 5분
    
    def get_current_usage(self) -> Dict[str, float]:
        """현재 리소스 사용량"""
        with self.monitor_lock:
            try:
                # CPU 사용률
                cpu_percent = self.process.cpu_percent(interval=0.1)
                
                # 메모리 사용률
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()
                
                # 디스크 사용률
                disk_usage = psutil.disk_usage('/')
                disk_percent = disk_usage.percent
                
                usage = {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'running_tasks': len(self.running_tasks),
                    'timestamp': time.time()
                }
                
                # 히스토리 업데이트
                self.cpu_history.append({'timestamp': time.time(), 'value': cpu_percent})
                self.memory_history.append({'timestamp': time.time(), 'value': memory_mb})
                self.disk_history.append({'timestamp': time.time(), 'value': disk_percent})
                
                return usage
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                return {'cpu_percent': 0, 'memory_mb': 0, 'memory_percent': 0, 
                       'disk_percent': 0, 'running_tasks': 0, 'timestamp': time.time()}
    
    def get_average_usage(self, minutes: int = 10) -> Dict[str, float]:
        """평균 리소스 사용량"""
        cutoff_time = time.time() - (minutes * 60)
        
        def get_recent_avg(history):
            recent = [h['value'] for h in history if h['timestamp'] > cutoff_time]
            return sum(recent) / len(recent) if recent else 0.0
        
        return {
            'cpu_percent': get_recent_avg(self.cpu_history),
            'memory_mb': get_recent_avg(self.memory_history),
            'disk_percent': get_recent_avg(self.disk_history)
        }
    
    def get_resource_level(self) -> ResourceThreshold:
        """현재 리소스 레벨"""
        current = self.get_current_usage()
        
        # 가장 높은 사용률을 기준으로 판단
        max_usage = max(
            current['cpu_percent'] / 100,
            current['memory_percent'] / 100,
            current['disk_percent'] / 100
        )
        
        if max_usage >= ResourceThreshold.CRITICAL.value:
            return ResourceThreshold.CRITICAL
        elif max_usage >= ResourceThreshold.HIGH.value:
            return ResourceThreshold.HIGH
        elif max_usage >= ResourceThreshold.NORMAL.value:
            return ResourceThreshold.NORMAL
        else:
            return ResourceThreshold.LOW
    
    def should_throttle(self, priority: TaskPriority) -> bool:
        """스로틀링 필요 여부 판단"""
        resource_level = self.get_resource_level()
        
        # CRITICAL 우선순위는 항상 실행
        if priority == TaskPriority.CRITICAL:
            return False
        
        # 리소스 레벨에 따른 스로틀링
        if resource_level == ResourceThreshold.CRITICAL:
            return priority.value > TaskPriority.HIGH.value
        elif resource_level == ResourceThreshold.HIGH:
            return priority.value > TaskPriority.NORMAL.value
        
        return False
    
    def register_task(self, task_id: str, task_info: Dict[str, Any]):
        """실행 중인 작업 등록"""
        with self.monitor_lock:
            self.running_tasks[task_id] = {
                **task_info,
                'start_time': time.time(),
                'start_memory': self.get_current_usage()['memory_mb']
            }
    
    def unregister_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """실행 완료된 작업 해제"""
        with self.monitor_lock:
            if task_id in self.running_tasks:
                task_info = self.running_tasks.pop(task_id)
                task_info['end_time'] = time.time()
                task_info['duration'] = task_info['end_time'] - task_info['start_time']
                task_info['end_memory'] = self.get_current_usage()['memory_mb']
                return task_info
            return None
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """알림 체크"""
        current = self.get_current_usage()
        alerts = []
        now = time.time()
        
        for metric, threshold in self.alert_thresholds.items():
            if current.get(metric, 0) > threshold:
                # 쿨다운 체크
                last_alert = self.last_alerts.get(metric, 0)
                if now - last_alert > self.alert_cooldown:
                    alerts.append({
                        'type': 'resource_alert',
                        'metric': metric,
                        'current': current[metric],
                        'threshold': threshold,
                        'timestamp': now
                    })
                    self.last_alerts[metric] = now
        
        return alerts

class BatchSchedulerV2:
    """향상된 배치 처리 스케줄러 V2"""
    
    def __init__(self,
                 max_concurrent_tasks: int = 2,  # VPS 환경에 맞게 감소
                 resource_check_interval: int = 30,
                 adaptive_scheduling: bool = True):
        """
        초기화
        
        Args:
            max_concurrent_tasks: 최대 동시 작업 수 (VPS 최적화)
            resource_check_interval: 리소스 체크 간격 (초)
            adaptive_scheduling: 적응형 스케줄링 활성화
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_check_interval = resource_check_interval
        self.adaptive_scheduling = adaptive_scheduling
        
        # 스케줄러 및 모니터 초기화
        self.scheduler = AsyncIOScheduler()
        self.resource_monitor = ResourceMonitorV2()
        
        # 작업 관리
        self.scheduled_tasks: Dict[str, ScheduledTaskV2] = {}
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 상태
        self.is_running = False
        self.start_time = time.time()
        
        # 통계
        self.global_stats = {
            "total_scheduled": 0,
            "total_executed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_throttled": 0,
            "avg_execution_time": 0.0,
            "scheduler_uptime": 0,
            "last_resource_check": None,
            "resource_alerts": []
        }
        
        # 백그라운드 작업
        self.monitor_task: Optional[asyncio.Task] = None
        self.maintenance_task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()
    
    async def start(self):
        """스케줄러 시작"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        logger.info("Starting batch scheduler V2...")
        
        try:
            # 스케줄러 시작
            self.scheduler.start()
            
            # 백그라운드 모니터링 시작
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            self.is_running = True
            logger.info("Batch scheduler V2 started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    async def stop(self):
        """스케줄러 중지"""
        if not self.is_running:
            return
        
        logger.info("Stopping batch scheduler V2...")
        
        try:
            self.is_running = False
            
            # 백그라운드 작업 중지
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            if self.maintenance_task and not self.maintenance_task.done():
                self.maintenance_task.cancel()
                try:
                    await self.maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # 실행 중인 작업들 취소
            for task_id, task in list(self.running_tasks.items()):
                if not task.done():
                    task.cancel()
                    logger.info(f"Cancelled running task: {task_id}")
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # 스케줄러 종료
            self.scheduler.shutdown(wait=True)
            
            logger.info("Batch scheduler V2 stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    async def schedule_task(self,
                          task_id: str,
                          name: str,
                          function: Callable,
                          schedule_type: ScheduleType,
                          schedule_config: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL,
                          **task_options) -> bool:
        """작업 스케줄링"""
        
        try:
            # 기존 작업 확인
            if task_id in self.scheduled_tasks:
                logger.warning(f"Task already scheduled: {task_id}")
                return False
            
            # 작업 생성
            task = ScheduledTaskV2(
                task_id=task_id,
                name=name,
                function=function,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                priority=priority,
                **task_options
            )
            
            # 적응형 스케줄링 설정
            if schedule_type == ScheduleType.ADAPTIVE:
                task.base_interval = schedule_config.get('base_interval', 300)  # 5분
                task.adaptive_interval = task.base_interval
            
            # 스케줄러에 등록
            success = await self._register_task_to_scheduler(task)
            
            if success:
                self.scheduled_tasks[task_id] = task
                self.global_stats["total_scheduled"] += 1
                task.status = TaskStatus.SCHEDULED
                
                logger.info(f"Task scheduled: {name} ({task_id})")
                return True
            else:
                logger.error(f"Failed to register task to scheduler: {task_id}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {e}")
            return False
    
    async def _register_task_to_scheduler(self, task: ScheduledTaskV2) -> bool:
        """스케줄러에 작업 등록"""
        
        try:
            job_config = {
                'func': self._execute_task_wrapper,
                'args': [task.task_id],
                'id': task.task_id,
                'name': task.name,
                'max_instances': task.max_parallel_instances,
                'coalesce': True,
                'misfire_grace_time': 300  # 5분
            }
            
            if task.schedule_type == ScheduleType.INTERVAL:
                trigger = IntervalTrigger(**task.schedule_config)
            elif task.schedule_type == ScheduleType.CRON:
                trigger = CronTrigger(**task.schedule_config)
            elif task.schedule_type == ScheduleType.ONCE:
                run_date = task.schedule_config.get('run_date', datetime.now() + timedelta(seconds=10))
                trigger = DateTrigger(run_date=run_date)
            elif task.schedule_type == ScheduleType.ADAPTIVE:
                # 적응형은 interval로 시작
                trigger = IntervalTrigger(seconds=task.adaptive_interval)
            else:
                logger.error(f"Unknown schedule type: {task.schedule_type}")
                return False
            
            self.scheduler.add_job(trigger=trigger, **job_config)
            return True
            
        except Exception as e:
            logger.error(f"Failed to register task to scheduler: {e}")
            return False
    
    async def _execute_task_wrapper(self, task_id: str):
        """작업 실행 래퍼"""
        
        task = self.scheduled_tasks.get(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return
        
        # 리소스 및 제약 조건 체크
        current_resources = self.resource_monitor.get_current_usage()
        can_run, reason = task.should_run(current_resources)
        
        if not can_run:
            logger.debug(f"Task {task_id} skipped: {reason}")
            if reason.startswith("Throttled"):
                self.global_stats["total_throttled"] += 1
                task.status = TaskStatus.THROTTLED
            return
        
        # 동시 실행 제한
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"Max concurrent tasks reached, queuing: {task_id}")
            self.task_queue.append(task_id)
            return
        
        # 스로틀링 체크
        if self.resource_monitor.should_throttle(task.priority):
            logger.warning(f"Resource throttling active, skipping: {task_id}")
            self.global_stats["total_throttled"] += 1
            task.status = TaskStatus.THROTTLED
            return
        
        # 작업 실행
        execution_task = asyncio.create_task(self._execute_task(task))
        self.running_tasks[task_id] = execution_task
        
        try:
            await execution_task
        except Exception as e:
            logger.error(f"Task execution wrapper failed: {task_id} - {e}")
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # 큐에서 대기 중인 작업 처리
            await self._process_queued_tasks()
    
    async def _execute_task(self, task: ScheduledTaskV2):
        """작업 실행"""
        
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        task.current_instances += 1
        
        # 리소스 모니터에 등록
        self.resource_monitor.register_task(task.task_id, {
            'name': task.name,
            'priority': task.priority.value,
            'cpu_intensive': task.cpu_intensive
        })
        
        success = False
        error_message = None
        
        try:
            logger.info(f"Executing task: {task.name} (Priority: {task.priority.name})")
            
            # 타임아웃 적용하여 작업 실행
            await asyncio.wait_for(
                task.function(),
                timeout=task.timeout
            )
            
            success = True
            task.status = TaskStatus.COMPLETED
            self.global_stats["total_successful"] += 1
            
            logger.info(f"Task completed successfully: {task.name}")
            
        except asyncio.TimeoutError:
            error_message = f"Task timed out after {task.timeout} seconds"
            task.status = TaskStatus.FAILED
            
        except Exception as e:
            error_message = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task execution failed: {task.name} - {error_message}")
            
        finally:
            runtime = time.time() - start_time
            task.current_instances = max(0, task.current_instances - 1)
            
            # 메트릭 업데이트
            task.metrics.update(runtime, success)
            
            # 실행 히스토리 추가
            task.execution_history.append({
                'timestamp': datetime.now(),
                'runtime': runtime,
                'success': success,
                'error': error_message,
                'resource_usage': self.resource_monitor.get_current_usage()
            })
            
            # 글로벌 통계 업데이트
            self.global_stats["total_executed"] += 1
            if not success:
                self.global_stats["total_failed"] += 1
            
            # 평균 실행 시간 업데이트
            total_tasks = self.global_stats["total_executed"]
            current_avg = self.global_stats["avg_execution_time"]
            self.global_stats["avg_execution_time"] = (
                (current_avg * (total_tasks - 1) + runtime) / total_tasks
            )
            
            # 적응형 스케줄링 조정
            if task.schedule_type == ScheduleType.ADAPTIVE:
                await self._adjust_adaptive_schedule(task, success, runtime)
            
            # 리소스 모니터에서 해제
            task_info = self.resource_monitor.unregister_task(task.task_id)
            if task_info:
                logger.debug(f"Task resource usage: {task_info}")
    
    async def _adjust_adaptive_schedule(self, task: ScheduledTaskV2, success: bool, runtime: float):
        """적응형 스케줄링 조정"""
        
        if not task.base_interval:
            return
        
        try:
            # 성공률과 리소스 사용량을 기반으로 간격 조정
            success_rate = task.metrics.success_rate
            resource_level = self.resource_monitor.get_resource_level()
            
            # 기본 간격에서 시작
            new_interval = task.base_interval
            
            # 성공률이 낮으면 간격 증가
            if success_rate < 0.7:
                new_interval = int(new_interval * 1.5)
            elif success_rate > 0.95:
                new_interval = int(new_interval * 0.8)
            
            # 리소스 상황에 따른 조정
            if resource_level == ResourceThreshold.CRITICAL:
                new_interval = int(new_interval * 2.0)
            elif resource_level == ResourceThreshold.HIGH:
                new_interval = int(new_interval * 1.3)
            elif resource_level == ResourceThreshold.LOW:
                new_interval = int(new_interval * 0.7)
            
            # 최소/최대 제한
            new_interval = max(60, min(3600, new_interval))  # 1분 ~ 1시간
            
            # 간격이 변경되었으면 스케줄 업데이트
            if new_interval != task.adaptive_interval:
                task.adaptive_interval = new_interval
                
                # 기존 작업 제거하고 새로 등록
                self.scheduler.remove_job(task.task_id)
                
                self.scheduler.add_job(
                    func=self._execute_task_wrapper,
                    trigger=IntervalTrigger(seconds=new_interval),
                    args=[task.task_id],
                    id=task.task_id,
                    name=task.name,
                    max_instances=task.max_parallel_instances,
                    coalesce=True,
                    misfire_grace_time=300
                )
                
                logger.info(f"Adaptive schedule adjusted for {task.task_id}: {new_interval}s")
                
        except Exception as e:
            logger.error(f"Failed to adjust adaptive schedule for {task.task_id}: {e}")
    
    async def _process_queued_tasks(self):
        """큐에 대기 중인 작업들 처리"""
        
        while (self.task_queue and 
               len(self.running_tasks) < self.max_concurrent_tasks and
               not self.resource_monitor.should_throttle(TaskPriority.NORMAL)):
            
            task_id = self.task_queue.popleft()
            execution_task = asyncio.create_task(self._execute_task_wrapper(task_id))
            # 백그라운드에서 실행하도록 함
            asyncio.create_task(self._handle_queued_task(task_id, execution_task))
    
    async def _handle_queued_task(self, task_id: str, execution_task: asyncio.Task):
        """큐에서 처리된 작업 핸들링"""
        try:
            await execution_task
        except Exception as e:
            logger.error(f"Queued task execution failed: {task_id} - {e}")
    
    async def _monitor_loop(self):
        """리소스 모니터링 루프"""
        
        while self.is_running:
            try:
                # 리소스 사용량 체크
                current_usage = self.resource_monitor.get_current_usage()
                self.global_stats["last_resource_check"] = datetime.now().isoformat()
                
                # 알림 체크
                alerts = self.resource_monitor.check_alerts()
                if alerts:
                    self.global_stats["resource_alerts"].extend(alerts)
                    # 최근 100개만 유지
                    self.global_stats["resource_alerts"] = self.global_stats["resource_alerts"][-100:]
                    
                    for alert in alerts:
                        logger.warning(f"Resource alert: {alert}")
                
                # 다음 체크까지 대기
                await asyncio.sleep(self.resource_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _maintenance_loop(self):
        """유지보수 루프"""
        
        while self.is_running:
            try:
                # 1시간마다 유지보수 실행
                await asyncio.sleep(3600)
                
                if not self.is_running:
                    break
                
                await self._perform_maintenance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(300)  # 오류 시 5분 대기
    
    async def _perform_maintenance(self):
        """시스템 유지보수"""
        
        logger.info("Performing system maintenance...")
        
        try:
            # 1. 실패한 작업 재시작
            failed_tasks = [
                task for task in self.scheduled_tasks.values()
                if task.metrics.consecutive_failures >= 3 and 
                   task.status == TaskStatus.FAILED
            ]
            
            for task in failed_tasks:
                if task.throttle_on_failure:
                    # 백오프 적용하여 재시작
                    backoff_delay = min(300, task.metrics.consecutive_failures * 60)
                    logger.info(f"Restarting failed task {task.task_id} after {backoff_delay}s")
                    
                    # 일시적으로 스케줄 조정
                    self.scheduler.modify_job(
                        task.task_id, 
                        next_run_time=datetime.now() + timedelta(seconds=backoff_delay)
                    )
                    
                    task.status = TaskStatus.SCHEDULED
            
            # 2. 메모리 정리
            import gc
            gc.collect()
            
            # 3. 통계 정리
            self.global_stats["scheduler_uptime"] = time.time() - self.start_time
            
            # 4. 오래된 히스토리 정리
            for task in self.scheduled_tasks.values():
                if len(task.execution_history) > 50:
                    # 최근 50개만 유지
                    task.execution_history = deque(
                        list(task.execution_history)[-50:], 
                        maxlen=50
                    )
            
            logger.info("System maintenance completed")
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
    
    # 편의 메서드들
    async def add_simple_interval_task(self, 
                                     task_id: str, 
                                     name: str, 
                                     function: Callable, 
                                     interval_seconds: int,
                                     priority: TaskPriority = TaskPriority.NORMAL) -> bool:
        """간단한 간격 작업 추가"""
        
        return await self.schedule_task(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={'seconds': interval_seconds},
            priority=priority
        )
    
    async def add_cron_task(self, 
                          task_id: str, 
                          name: str, 
                          function: Callable, 
                          cron_config: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL) -> bool:
        """크론 작업 추가"""
        
        return await self.schedule_task(
            task_id=task_id,
            name=name,
            function=function,
            schedule_type=ScheduleType.CRON,
            schedule_config=cron_config,
            priority=priority
        )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        
        task = self.scheduled_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority.name,
            "schedule_type": task.schedule_type.value,
            "current_instances": task.current_instances,
            "metrics": {
                "execution_count": task.metrics.execution_count,
                "success_count": task.metrics.success_count,
                "failure_count": task.metrics.failure_count,
                "success_rate": round(task.metrics.success_rate, 3),
                "avg_runtime": round(task.metrics.avg_runtime, 2),
                "consecutive_failures": task.metrics.consecutive_failures,
                "last_execution": task.metrics.last_execution.isoformat() if task.metrics.last_execution else None
            },
            "is_running": task_id in self.running_tasks,
            "adaptive_interval": task.adaptive_interval if task.schedule_type == ScheduleType.ADAPTIVE else None
        }
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        
        current_resources = self.resource_monitor.get_current_usage()
        resource_level = self.resource_monitor.get_resource_level()
        
        return {
            **self.global_stats,
            "scheduler_uptime": time.time() - self.start_time,
            "is_running": self.is_running,
            "scheduled_tasks_count": len(self.scheduled_tasks),
            "running_tasks_count": len(self.running_tasks),
            "queued_tasks_count": len(self.task_queue),
            "resource_usage": current_resources,
            "resource_level": resource_level.name,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "adaptive_scheduling_enabled": self.adaptive_scheduling,
            "recent_alerts": self.global_stats["resource_alerts"][-10:] if self.global_stats["resource_alerts"] else []
        }
    
    async def pause_task(self, task_id: str) -> bool:
        """작업 일시 정지"""
        
        try:
            self.scheduler.pause_job(task_id)
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].status = TaskStatus.PENDING
            logger.info(f"Task paused: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """작업 재개"""
        
        try:
            self.scheduler.resume_job(task_id)
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].status = TaskStatus.SCHEDULED
            logger.info(f"Task resumed: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            return False
    
    async def remove_task(self, task_id: str) -> bool:
        """작업 제거"""
        
        try:
            # 실행 중인 작업 취소
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                del self.running_tasks[task_id]
            
            # 스케줄러에서 제거
            self.scheduler.remove_job(task_id)
            
            # 로컬 저장소에서 제거
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
            
            # 큐에서도 제거
            self.task_queue = deque([tid for tid in self.task_queue if tid != task_id])
            
            logger.info(f"Task removed: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove task {task_id}: {e}")
            return False


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def sample_task():
        """샘플 작업"""
        print("Sample task executed")
        await asyncio.sleep(2)
        print("Sample task completed")
    
    async def cpu_intensive_task():
        """CPU 집약적 작업"""
        print("CPU intensive task started")
        # 의도적으로 CPU 사용
        total = 0
        for i in range(1000000):
            total += i
        await asyncio.sleep(1)
        print(f"CPU intensive task completed: {total}")
    
    async def failing_task():
        """실패하는 작업"""
        print("Failing task started")
        await asyncio.sleep(1)
        raise Exception("Intentional failure for testing")
    
    async def test_scheduler():
        """스케줄러 테스트"""
        
        print("=== Batch Scheduler V2 테스트 ===")
        
        async with BatchSchedulerV2(max_concurrent_tasks=2) as scheduler:
            
            # 1. 간단한 간격 작업
            await scheduler.add_simple_interval_task(
                "sample_task",
                "Sample Task",
                sample_task,
                interval_seconds=10,
                priority=TaskPriority.NORMAL
            )
            
            # 2. CPU 집약적 작업
            await scheduler.schedule_task(
                "cpu_task",
                "CPU Intensive Task",
                cpu_intensive_task,
                ScheduleType.INTERVAL,
                {'seconds': 15},
                priority=TaskPriority.LOW,
                cpu_intensive=True,
                max_memory_mb=512
            )
            
            # 3. 적응형 작업
            await scheduler.schedule_task(
                "adaptive_task",
                "Adaptive Task",
                sample_task,
                ScheduleType.ADAPTIVE,
                {'base_interval': 20},
                priority=TaskPriority.NORMAL
            )
            
            # 4. 실패하는 작업 (테스트용)
            await scheduler.add_simple_interval_task(
                "failing_task",
                "Failing Task", 
                failing_task,
                interval_seconds=25,
                priority=TaskPriority.LOW
            )
            
            # 5. 크론 작업 (매분)
            await scheduler.add_cron_task(
                "cron_task",
                "Cron Task",
                sample_task,
                {'second': 0},  # 매분 0초에 실행
                priority=TaskPriority.HIGH
            )
            
            print(f"\n5개 작업이 스케줄링되었습니다.")
            
            # 60초 동안 실행
            for i in range(12):  # 5초씩 12번 = 60초
                await asyncio.sleep(5)
                
                # 통계 출력
                stats = scheduler.get_scheduler_stats()
                print(f"\n=== {i*5+5}초 후 통계 ===")
                print(f"실행된 작업: {stats['total_executed']}")
                print(f"성공: {stats['total_successful']}, 실패: {stats['total_failed']}")
                print(f"스로틀됨: {stats['total_throttled']}")
                print(f"현재 실행 중: {stats['running_tasks_count']}")
                print(f"대기 중: {stats['queued_tasks_count']}")
                print(f"리소스 레벨: {stats['resource_level']}")
                print(f"CPU: {stats['resource_usage']['cpu_percent']:.1f}%, "
                      f"메모리: {stats['resource_usage']['memory_mb']:.0f}MB")
                
                # 작업별 상태
                for task_id in ['sample_task', 'cpu_task', 'adaptive_task']:
                    task_status = scheduler.get_task_status(task_id)
                    if task_status:
                        metrics = task_status['metrics']
                        print(f"  {task_id}: {metrics['execution_count']}회 실행, "
                              f"성공률 {metrics['success_rate']:.1%}")
            
            print(f"\n=== 최종 통계 ===")
            final_stats = scheduler.get_scheduler_stats()
            print(json.dumps(final_stats, indent=2, default=str))
    
    # 테스트 실행
    asyncio.run(test_scheduler())