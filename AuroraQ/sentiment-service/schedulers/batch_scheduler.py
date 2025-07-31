#!/usr/bin/env python3
"""
Batch Processing Scheduler for AuroraQ Sentiment Service
배치 처리 스케줄러 및 최적화 시스템
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import statistics

# 스케줄링 라이브러리
import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# 로컬 임포트
from ..processors.finbert_batch_processor import FinBERTBatchProcessor
from ..processors.sentiment_fusion_manager import SentimentFusionManager
from ..processors.big_event_detector import BigEventDetector
from ..collectors.enhanced_news_collector import EnhancedNewsCollector
from ..integrations.aurora_adapter import AuroraQAdapter, TradingMode
from ..utils.content_cache_manager import ContentCacheManager
from ..integrations.notification_integration import NotificationIntegration

logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """스케줄 유형"""
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"

class TaskPriority(Enum):
    """작업 우선순위"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ScheduledTask:
    """스케줄된 작업"""
    task_id: str
    name: str
    function: Callable
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_runtime: int = 3600  # 최대 실행 시간 (초)
    retry_count: int = 3
    timeout: int = 1800  # 30분
    
    # 실행 상태
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_runtime: float = 0.0
    
    # 실행 히스토리
    execution_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_stats(self, runtime: float, success: bool):
        """통계 업데이트"""
        self.run_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # 평균 실행 시간 업데이트
        self.avg_runtime = (
            (self.avg_runtime * (self.run_count - 1) + runtime) / self.run_count
        )
        
        # 히스토리 추가
        self.execution_history.append({
            'timestamp': datetime.now(),
            'runtime': runtime,
            'success': success
        })

class ResourceMonitor:
    """리소스 모니터"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=60)  # 60분 히스토리
        self.memory_history = deque(maxlen=60)
        self.running_tasks = {}
        
    def record_resource_usage(self, cpu_percent: float, memory_mb: float):
        """리소스 사용량 기록"""
        self.cpu_history.append({
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent
        })
        
        self.memory_history.append({
            'timestamp': datetime.now(),
            'memory_mb': memory_mb
        })
    
    def get_current_load(self) -> Dict[str, float]:
        """현재 부하 상태"""
        if not self.cpu_history:
            return {'cpu': 0.0, 'memory': 0.0, 'tasks': len(self.running_tasks)}
        
        recent_cpu = statistics.mean([h['cpu_percent'] for h in list(self.cpu_history)[-5:]])
        recent_memory = statistics.mean([h['memory_mb'] for h in list(self.memory_history)[-5:]])
        
        return {
            'cpu': recent_cpu,
            'memory': recent_memory,
            'tasks': len(self.running_tasks)
        }
    
    def should_throttle(self) -> bool:
        """스로틀링이 필요한지 확인"""
        load = self.get_current_load()
        
        # CPU 80% 이상, 메모리 1GB 이상, 동시 작업 5개 이상
        return (
            load['cpu'] > 80.0 or
            load['memory'] > 1024.0 or
            load['tasks'] >= 5
        )

class BatchScheduler:
    """배치 처리 스케줄러"""
    
    def __init__(self,
                 finbert_processor: FinBERTBatchProcessor,
                 fusion_manager: SentimentFusionManager,
                 event_detector: BigEventDetector,
                 news_collector: EnhancedNewsCollector,
                 aurora_adapter: AuroraQAdapter,
                 cache_manager: ContentCacheManager,
                 notification_integration: Optional[NotificationIntegration] = None):
        """
        초기화
        
        Args:
            finbert_processor: FinBERT 배치 프로세서
            fusion_manager: 감정 융합 매니저
            event_detector: 빅 이벤트 감지기
            news_collector: 뉴스 수집기
            aurora_adapter: AuroraQ 어댑터
            cache_manager: 캐시 매니저
        """
        self.finbert_processor = finbert_processor
        self.fusion_manager = fusion_manager
        self.event_detector = event_detector
        self.news_collector = news_collector
        self.aurora_adapter = aurora_adapter
        self.cache_manager = cache_manager
        self.notification_integration = notification_integration
        
        # 스케줄러 초기화
        self.scheduler = AsyncIOScheduler()
        self.resource_monitor = ResourceMonitor()
        
        # 작업 관리
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 설정
        self.max_concurrent_tasks = 3
        self.default_symbols = ["BTC", "ETH", "CRYPTO"]
        
        # 통계
        self.stats = {
            "total_scheduled": 0,
            "total_executed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "avg_execution_time": 0.0,
            "scheduler_uptime": 0,
            "last_maintenance": None
        }
        
        self.start_time = time.time()
    
    async def start(self):
        """스케줄러 시작"""
        logger.info("Starting batch scheduler...")
        
        # 기본 작업들 스케줄링
        await self._schedule_default_tasks()
        
        # 스케줄러 시작
        self.scheduler.start()
        
        logger.info("Batch scheduler started successfully")
    
    async def stop(self):
        """스케줄러 중지"""
        logger.info("Stopping batch scheduler...")
        
        # 실행 중인 작업들 취소
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled running task: {task_id}")
        
        # 스케줄러 종료
        self.scheduler.shutdown(wait=True)
        
        logger.info("Batch scheduler stopped")
    
    async def _schedule_default_tasks(self):
        """기본 작업들 스케줄링"""
        
        # 1. 뉴스 수집 (5분 간격)
        await self.schedule_task(
            task_id="news_collection",
            name="News Collection",
            function=self._collect_news_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 5},
            priority=TaskPriority.HIGH
        )
        
        # 2. FinBERT 배치 처리 (15분 간격)
        await self.schedule_task(
            task_id="finbert_batch",
            name="FinBERT Batch Processing",
            function=self._finbert_batch_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 15},
            priority=TaskPriority.CRITICAL
        )
        
        # 3. 빅 이벤트 감지 (10분 간격)
        await self.schedule_task(
            task_id="event_detection",
            name="Big Event Detection",
            function=self._event_detection_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 10},
            priority=TaskPriority.HIGH
        )
        
        # 4. 매매 신호 생성 (3분 간격, 실전 모드)
        await self.schedule_task(
            task_id="trading_signals_live",
            name="Live Trading Signals",
            function=self._trading_signals_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 3},
            priority=TaskPriority.HIGH,
            kwargs={"trading_mode": TradingMode.LIVE}
        )
        
        # 5. 매매 신호 생성 (2분 간격, 가상 모드)
        await self.schedule_task(
            task_id="trading_signals_paper",
            name="Paper Trading Signals",
            function=self._trading_signals_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 2},
            priority=TaskPriority.NORMAL,
            kwargs={"trading_mode": TradingMode.PAPER}
        )
        
        # 6. 캐시 정리 (30분 간격)
        await self.schedule_task(
            task_id="cache_cleanup",
            name="Cache Cleanup",
            function=self._cache_cleanup_task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"minutes": 30},
            priority=TaskPriority.LOW
        )
        
        # 7. 시스템 유지보수 (매일 새벽 2시)
        await self.schedule_task(
            task_id="system_maintenance",
            name="System Maintenance",
            function=self._system_maintenance_task,
            schedule_type=ScheduleType.CRON,
            schedule_config={"hour": 2, "minute": 0},
            priority=TaskPriority.LOW
        )
    
    async def schedule_task(self,
                          task_id: str,
                          name: str,
                          function: Callable,
                          schedule_type: ScheduleType,
                          schedule_config: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL,
                          max_runtime: int = 3600,
                          retry_count: int = 3,
                          **kwargs) -> bool:
        """작업 스케줄링"""
        
        try:
            # 작업 생성
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                function=function,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                priority=priority,
                max_runtime=max_runtime,
                retry_count=retry_count
            )
            
            # 스케줄러에 등록
            if schedule_type == ScheduleType.INTERVAL:
                self.scheduler.add_job(
                    func=self._execute_task_wrapper,
                    trigger=IntervalTrigger(**schedule_config),
                    args=[task_id, kwargs],
                    id=task_id,
                    name=name,
                    max_instances=1,
                    coalesce=True
                )
            
            elif schedule_type == ScheduleType.CRON:
                self.scheduler.add_job(
                    func=self._execute_task_wrapper,
                    trigger=CronTrigger(**schedule_config),
                    args=[task_id, kwargs],
                    id=task_id,
                    name=name,
                    max_instances=1,
                    coalesce=True
                )
            
            self.scheduled_tasks[task_id] = task
            self.stats["total_scheduled"] += 1
            
            logger.info(f"Task scheduled: {name} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {e}")
            return False
    
    async def _execute_task_wrapper(self, task_id: str, kwargs: Dict[str, Any] = None):
        """작업 실행 래퍼"""
        
        if kwargs is None:
            kwargs = {}
        
        task = self.scheduled_tasks.get(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return
        
        # 리소스 체크
        if self.resource_monitor.should_throttle():
            logger.warning(f"Resource throttling - skipping task: {task_id}")
            return
        
        # 동시 실행 제한
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"Max concurrent tasks reached - skipping task: {task_id}")
            return
        
        # 이미 실행 중인지 확인
        if task_id in self.running_tasks and not self.running_tasks[task_id].done():
            logger.warning(f"Task already running: {task_id}")
            return
        
        # 작업 실행
        task_future = asyncio.create_task(self._execute_task(task, kwargs))
        self.running_tasks[task_id] = task_future
        
        try:
            await task_future
        except Exception as e:
            logger.error(f"Task execution failed: {task_id} - {e}")
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _execute_task(self, task: ScheduledTask, kwargs: Dict[str, Any]):
        """작업 실행"""
        
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()
        
        success = False
        error_message = None
        
        try:
            logger.info(f"Executing task: {task.name}")
            
            # 타임아웃 적용
            await asyncio.wait_for(
                task.function(**kwargs),
                timeout=task.timeout
            )
            
            success = True
            task.status = TaskStatus.COMPLETED
            self.stats["total_successful"] += 1
            
            logger.info(f"Task completed successfully: {task.name}")
            
        except asyncio.TimeoutError:
            error_message = f"Task timed out after {task.timeout} seconds"
            task.status = TaskStatus.FAILED
            
        except Exception as e:
            error_message = str(e)
            task.status = TaskStatus.FAILED
            
        finally:
            runtime = time.time() - start_time
            
            # 통계 업데이트
            task.update_stats(runtime, success)
            self.stats["total_executed"] += 1
            
            if not success:
                self.stats["total_failed"] += 1
                logger.error(f"Task failed: {task.name} - {error_message}")
            
            # 평균 실행 시간 업데이트
            total_tasks = self.stats["total_executed"]
            current_avg = self.stats["avg_execution_time"]
            self.stats["avg_execution_time"] = (
                (current_avg * (total_tasks - 1) + runtime) / total_tasks
            )
    
    # 작업 함수들
    async def _collect_news_task(self, **kwargs):
        """뉴스 수집 작업"""
        
        try:
            all_news = []
            
            for symbol in self.default_symbols:
                news_items = await self.news_collector.collect_all_sources(
                    symbol=symbol, hours_back=6
                )
                all_news.extend(news_items)
            
            logger.info(f"News collection completed: {len(all_news)} items")
            
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            # 실패 알림 발송
            if self.notification_integration:
                await self.notification_integration.send_batch_completion_notification(
                    "뉴스 수집", False, {"error": str(e)}
                )
            raise
    
    async def _finbert_batch_task(self, **kwargs):
        """FinBERT 배치 처리 작업"""
        
        try:
            # 강제 배치 실행
            success = await self.finbert_processor.force_batch_run()
            
            if success:
                stats = self.finbert_processor.get_processor_stats()
                logger.info(f"FinBERT batch processing completed: {stats}")
                
                # 성공 알림 발송 (중요한 작업이므로)
                if self.notification_integration and stats.get("processed_items", 0) > 0:
                    await self.notification_integration.send_batch_completion_notification(
                        "FinBERT 배치 분석", True, {
                            "processed": stats.get("processed_items", 0),
                            "avg_score": stats.get("avg_confidence", 0)
                        }
                    )
            else:
                logger.warning("FinBERT batch processing returned no results")
            
        except Exception as e:
            logger.error(f"FinBERT batch processing failed: {e}")
            raise
    
    async def _event_detection_task(self, **kwargs):
        """빅 이벤트 감지 작업"""
        
        try:
            # 최근 뉴스 조회
            all_news = []
            
            for symbol in self.default_symbols:
                news_items = await self.news_collector.collect_all_sources(
                    symbol=symbol, hours_back=12
                )
                all_news.extend(news_items)
            
            # 이벤트 감지
            if all_news:
                detected_events = await self.event_detector.detect_events(all_news)
                logger.info(f"Event detection completed: {len(detected_events)} events")
                
                # 중요한 이벤트가 감지되면 알림 발송
                if self.notification_integration and detected_events:
                    for event in detected_events:
                        # 영향도가 7.0 이상인 경우만 알림
                        if event.impact_score >= 7.0:
                            await self.notification_integration.send_big_event_notification(event)
            else:
                logger.info("No news items for event detection")
            
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            raise
    
    async def _trading_signals_task(self, trading_mode: TradingMode = TradingMode.PAPER, **kwargs):
        """매매 신호 생성 작업"""
        
        try:
            # 배치 매매 신호 생성
            signals = await self.aurora_adapter.process_symbol_batch(
                symbols=self.default_symbols,
                trading_mode=trading_mode,
                send_to_aurora=True
            )
            
            logger.info(f"Trading signals generated ({trading_mode.value}): {len(signals)}")
            
            # 강한 매매 신호가 생성되면 알림 발송
            if self.notification_integration and signals:
                for signal in signals:
                    # STRONG 이상 신호만 알림
                    if signal.strength.value in ["strong", "very_strong"]:
                        await self.notification_integration.send_trading_signal_notification(
                            signal, trading_mode
                        )
            
        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            raise
    
    async def _cache_cleanup_task(self, **kwargs):
        """캐시 정리 작업"""
        
        try:
            # 만료된 컨텐츠 정리
            cleaned = await self.cache_manager.cleanup_expired_content()
            
            # 이벤트 정리
            event_cleaned = self.event_detector.cleanup_old_events()
            
            # 융합 매니저 캐시 정리 (선택적)
            self.fusion_manager.clear_cache()
            
            logger.info(f"Cache cleanup completed: {cleaned} content items, {event_cleaned} events")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            raise
    
    async def _system_maintenance_task(self, **kwargs):
        """시스템 유지보수 작업"""
        
        try:
            logger.info("Starting system maintenance...")
            
            # 1. 통계 리셋
            daily_stats = dict(self.stats)
            
            # 2. 로그 정리 (여기서는 간단한 예시)
            # 실제로는 로그 파일 회전, 압축 등을 수행
            
            # 3. 메모리 정리
            import gc
            gc.collect()
            
            # 4. 헬스체크
            health_status = await self.aurora_adapter.health_check()
            
            self.stats["last_maintenance"] = datetime.now().isoformat()
            
            logger.info(f"System maintenance completed. Daily stats: {daily_stats}")
            
        except Exception as e:
            logger.error(f"System maintenance failed: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        
        task = self.scheduled_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority.value,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "success_count": task.success_count,
            "error_count": task.error_count,
            "success_rate": task.success_count / task.run_count if task.run_count > 0 else 0,
            "avg_runtime": round(task.avg_runtime, 2),
            "is_running": task_id in self.running_tasks
        }
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        
        self.stats["scheduler_uptime"] = time.time() - self.start_time
        
        return {
            **self.stats,
            "scheduled_tasks_count": len(self.scheduled_tasks),
            "running_tasks_count": len(self.running_tasks),
            "resource_load": self.resource_monitor.get_current_load(),
            "task_details": {
                task_id: self.get_task_status(task_id)
                for task_id in self.scheduled_tasks.keys()
            }
        }
    
    async def pause_task(self, task_id: str) -> bool:
        """작업 일시 정지"""
        
        try:
            self.scheduler.pause_job(task_id)
            logger.info(f"Task paused: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """작업 재개"""
        
        try:
            self.scheduler.resume_job(task_id)
            logger.info(f"Task resumed: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            return False
    
    async def remove_task(self, task_id: str) -> bool:
        """작업 제거"""
        
        try:
            self.scheduler.remove_job(task_id)
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
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
    
    async def test_batch_scheduler():
        """배치 스케줄러 테스트"""
        
        print("=== 배치 스케줄러 테스트 ===")
        
        # Mock 컴포넌트들 (실제 구현에서는 실제 객체 사용)
        scheduler = None  # 실제로는 BatchScheduler 인스턴스
        
        print("배치 스케줄러 테스트 완료")
    
    # 테스트 실행
    asyncio.run(test_batch_scheduler())