#!/usr/bin/env python3
"""
Sentiment Service Runner for VPS Deployment
VPS 환경에서 감정 서비스를 실행하는 통합 런너
"""

import asyncio
import signal
import sys
import os
import time
import logging
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# 로컬 임포트
from config.sentiment_service_config import SentimentServiceConfig, get_config, validate_vps_environment
from collectors.enhanced_news_collector_v2 import EnhancedNewsCollectorV2
from collectors.macro_indicator_collector import MacroIndicatorCollector
from processors.finbert_batch_processor_v2 import FinBERTBatchProcessorV2
from processors.big_event_detector_v2 import BigEventDetectorV2
from processors.sentiment_fusion_manager_v2 import SentimentFusionManagerV2
from processors.event_impact_manager import EventImpactManager
from processors.scheduled_event_fusion import ScheduledEventFusion
from monitors.option_expiry_monitor import OptionExpiryMonitor
from schedulers.event_schedule_loader import EventScheduleLoader
from schedulers.batch_scheduler_v2 import BatchSchedulerV2, TaskPriority, ScheduleType

logger = logging.getLogger(__name__)

class ServiceState:
    """서비스 상태 관리"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class HealthMonitor:
    """헬스 모니터"""
    
    def __init__(self, config: SentimentServiceConfig):
        self.config = config
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.health_status = {
            "status": "healthy",
            "uptime": 0,
            "components": {},
            "last_error": None,
            "resource_usage": {}
        }
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self):
        """모니터링 시작"""
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self):
        """모니터링 중지"""
        self.is_running = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                await self._update_health_status()
                await asyncio.sleep(30)  # 30초마다 체크
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _update_health_status(self):
        """헬스 상태 업데이트"""
        current_time = time.time()
        
        # 기본 정보
        self.health_status["uptime"] = current_time - self.start_time
        self.health_status["timestamp"] = datetime.now().isoformat()
        
        # 리소스 사용량 업데이트
        try:
            import psutil
            process = psutil.Process()
            
            self.health_status["resource_usage"] = {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
            
            # 시스템 리소스
            self.health_status["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
        except Exception as e:
            self.health_status["resource_usage"] = {"error": str(e)}
        
        self.last_health_check = current_time
    
    def get_health_status(self) -> Dict[str, Any]:
        """헬스 상태 반환"""
        return self.health_status.copy()
    
    def update_component_status(self, component: str, status: str, details: Optional[Dict] = None):
        """컴포넌트 상태 업데이트"""
        self.health_status["components"][component] = {
            "status": status,
            "last_updated": datetime.now().isoformat(),
            "details": details or {}
        }
    
    def report_error(self, error: str, component: Optional[str] = None):
        """에러 보고"""
        self.health_status["last_error"] = {
            "message": error,
            "component": component,
            "timestamp": datetime.now().isoformat()
        }
        
        if component:
            self.update_component_status(component, "error", {"error": error})

class SentimentServiceRunner:
    """감정 서비스 런너"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로 (선택사항)
        """
        self.config = SentimentServiceConfig()
        if config_path and os.path.exists(config_path):
            self.config = SentimentServiceConfig.from_file(config_path)
        
        self.state = ServiceState.STOPPED
        self.health_monitor = HealthMonitor(self.config)
        
        # 서비스 컴포넌트들
        self.news_collector: Optional[EnhancedNewsCollectorV2] = None
        self.macro_collector: Optional[MacroIndicatorCollector] = None
        self.expiry_monitor: Optional[OptionExpiryMonitor] = None
        self.event_loader: Optional[EventScheduleLoader] = None
        self.finbert_processor: Optional[FinBERTBatchProcessorV2] = None
        self.event_detector: Optional[BigEventDetectorV2] = None
        self.impact_manager: Optional[EventImpactManager] = None
        self.fusion_manager: Optional[SentimentFusionManagerV2] = None
        self.event_fusion: Optional[ScheduledEventFusion] = None
        self.scheduler: Optional[BatchSchedulerV2] = None
        
        # 실행 제어
        self.shutdown_event = asyncio.Event()
        self.startup_complete = False
        
        # 통계
        self.startup_time: Optional[float] = None
        self.error_count = 0
        
        # 로깅 설정
        self._setup_logging()
        
        # 신호 핸들러 설정
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_config = self.config.logging
        
        # 로그 레벨 설정
        log_level = getattr(logging, log_config.level.value)
        logging.basicConfig(
            level=log_level,
            format=log_config.format,
            handlers=[]
        )
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # 콘솔 핸들러
        if log_config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(log_config.format)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_config.enable_file and log_config.file_path:
            try:
                # 로그 디렉토리 생성
                log_dir = os.path.dirname(log_config.file_path)
                os.makedirs(log_dir, exist_ok=True)
                
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_config.file_path,
                    maxBytes=log_config.max_file_size,
                    backupCount=log_config.backup_count
                )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(log_config.format)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                print(f"Failed to setup file logging: {e}")
    
    def _setup_signal_handlers(self):
        """신호 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """서비스 시작"""
        try:
            self.state = ServiceState.STARTING
            self.startup_time = time.time()
            
            logger.info(f"Starting {self.config.service_name} v{self.config.version}")
            logger.info(f"Deployment mode: {self.config.deployment_mode.value}")
            
            # 환경 검증
            issues = validate_vps_environment()
            if issues:
                logger.warning("Environment issues detected:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            # 헬스 모니터 시작
            await self.health_monitor.start()
            
            # 지연 시작 (VPS 안정화)
            if self.config.startup_delay > 0:
                logger.info(f"Startup delay: {self.config.startup_delay} seconds")
                await asyncio.sleep(self.config.startup_delay)
            
            # 컴포넌트들 초기화 및 시작
            await self._initialize_components()
            await self._start_components()
            
            # 스케줄러 작업 등록
            await self._setup_scheduled_tasks()
            
            self.state = ServiceState.RUNNING
            self.startup_complete = True
            
            startup_duration = time.time() - self.startup_time
            logger.info(f"Service started successfully in {startup_duration:.2f} seconds")
            
            self.health_monitor.update_component_status("service", "running", {
                "startup_duration": startup_duration,
                "components_initialized": 10
            })
            
            # 메인 루프
            await self._main_loop()
            
        except Exception as e:
            self.state = ServiceState.ERROR
            self.error_count += 1
            logger.error(f"Failed to start service: {e}", exc_info=True)
            self.health_monitor.report_error(str(e), "service")
            raise
    
    async def _initialize_components(self):
        """컴포넌트 초기화"""
        logger.info("Initializing components...")
        
        try:
            # 1. News Collector
            logger.info("Initializing news collector...")
            api_keys = {
                "newsapi_key": self.config.api_keys.newsapi_key,
                "finnhub_key": self.config.api_keys.finnhub_key
            }
            
            self.news_collector = EnhancedNewsCollectorV2(
                api_keys=api_keys,
                max_concurrent_requests=self.config.vps_limits.max_concurrent_requests,
                request_timeout=self.config.vps_limits.request_timeout,
                max_retries=self.config.vps_limits.max_retries
            )
            self.health_monitor.update_component_status("news_collector", "initialized")
            
            # 2. Macro Indicator Collector
            logger.info("Initializing macro indicator collector...")
            self.macro_collector = MacroIndicatorCollector(
                update_interval=300  # 5분 간격
            )
            self.health_monitor.update_component_status("macro_collector", "initialized")
            
            # 3. Option Expiry Monitor
            logger.info("Initializing option expiry monitor...")
            self.expiry_monitor = OptionExpiryMonitor(
                update_interval=3600,  # 1시간 간격
                max_days_ahead=90
            )
            self.health_monitor.update_component_status("expiry_monitor", "initialized")
            
            # 4. Event Schedule Loader
            logger.info("Initializing event schedule loader...")
            self.event_loader = EventScheduleLoader(
                update_interval=21600  # 6시간 간격
            )
            self.health_monitor.update_component_status("event_loader", "initialized")
            
            # 5. FinBERT Processor
            logger.info("Initializing FinBERT processor...")
            self.finbert_processor = FinBERTBatchProcessorV2(
                initial_batch_size=self.config.processor.initial_batch_size,
                max_batch_size=self.config.processor.max_batch_size,
                min_batch_size=self.config.processor.min_batch_size,
                max_sequence_length=self.config.processor.max_sequence_length
            )
            self.health_monitor.update_component_status("finbert_processor", "initialized")
            
            # 6. Big Event Detector
            logger.info("Initializing event detector...")
            self.event_detector = BigEventDetectorV2(
                impact_threshold=self.config.processor.impact_threshold,
                cache_size=500,  # VPS에서 캐시 크기 제한
                enable_real_time_alerts=True
            )
            self.health_monitor.update_component_status("event_detector", "initialized")
            
            # 7. Event Impact Manager
            logger.info("Initializing event impact manager...")
            self.impact_manager = EventImpactManager()
            self.health_monitor.update_component_status("impact_manager", "initialized")
            
            # 8. Fusion Manager
            logger.info("Initializing fusion manager...")
            self.fusion_manager = SentimentFusionManagerV2(
                cache_size=self.config.fusion.cache_size,
                cache_ttl=self.config.fusion.cache_ttl,
                enable_adaptive_weights=self.config.fusion.enable_adaptive_weights
            )
            self.health_monitor.update_component_status("fusion_manager", "initialized")
            
            # 9. Scheduled Event Fusion
            logger.info("Initializing scheduled event fusion...")
            self.event_fusion = ScheduledEventFusion(
                macro_collector=self.macro_collector,
                expiry_monitor=self.expiry_monitor,
                event_loader=self.event_loader,
                impact_manager=self.impact_manager,
                sentiment_manager=self.fusion_manager,
                fusion_interval=900  # 15분 간격
            )
            self.health_monitor.update_component_status("event_fusion", "initialized")
            
            # 10. Scheduler
            logger.info("Initializing scheduler...")
            self.scheduler = BatchSchedulerV2(
                max_concurrent_tasks=self.config.scheduler.max_concurrent_tasks,
                resource_check_interval=self.config.scheduler.resource_check_interval,
                adaptive_scheduling=self.config.scheduler.adaptive_scheduling
            )
            self.health_monitor.update_component_status("scheduler", "initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def _start_components(self):
        """컴포넌트 시작"""
        logger.info("Starting components...")
        
        try:
            # 1. Macro Indicator Collector 시작
            logger.info("Starting macro indicator collector...")
            await self.macro_collector.start()
            self.health_monitor.update_component_status("macro_collector", "running")
            
            # 2. Option Expiry Monitor 시작
            logger.info("Starting option expiry monitor...")
            await self.expiry_monitor.start()
            self.health_monitor.update_component_status("expiry_monitor", "running")
            
            # 3. Event Schedule Loader 시작
            logger.info("Starting event schedule loader...")
            await self.event_loader.start()
            self.health_monitor.update_component_status("event_loader", "running")
            
            # 4. FinBERT 프로세서 시작 (모델 로딩 시간이 오래 걸림)
            logger.info("Starting FinBERT processor...")
            await self.finbert_processor.start()
            self.health_monitor.update_component_status("finbert_processor", "running")
            
            # 5. Fusion Manager 시작
            logger.info("Starting fusion manager...")
            await self.fusion_manager.start()
            self.health_monitor.update_component_status("fusion_manager", "running")
            
            # 6. Event Detector 시작
            logger.info("Starting event detector...")
            await self.event_detector.start()
            self.health_monitor.update_component_status("event_detector", "running")
            
            # 7. Scheduled Event Fusion 시작
            logger.info("Starting scheduled event fusion...")
            await self.event_fusion.start()
            self.health_monitor.update_component_status("event_fusion", "running")
            
            # 8. Scheduler 시작 (마지막에 시작)
            logger.info("Starting scheduler...")
            await self.scheduler.start()
            self.health_monitor.update_component_status("scheduler", "running")
            
            logger.info("All components started successfully")
            
        except Exception as e:
            logger.error(f"Component startup failed: {e}")
            raise
    
    async def _setup_scheduled_tasks(self):
        """스케줄된 작업 설정"""
        logger.info("Setting up scheduled tasks...")
        
        try:
            # 1. 뉴스 수집 작업
            await self.scheduler.add_simple_interval_task(
                task_id="news_collection",
                name="News Collection",
                function=self._news_collection_task,
                interval_seconds=self.config.scheduler.task_intervals["news_collection"],
                priority=TaskPriority.HIGH
            )
            
            # 2. FinBERT 배치 처리
            await self.scheduler.add_simple_interval_task(
                task_id="finbert_batch",
                name="FinBERT Batch Processing",
                function=self._finbert_batch_task,
                interval_seconds=self.config.scheduler.task_intervals["finbert_batch"],
                priority=TaskPriority.CRITICAL
            )
            
            # 3. 이벤트 감지
            await self.scheduler.add_simple_interval_task(
                task_id="event_detection",
                name="Event Detection",
                function=self._event_detection_task,
                interval_seconds=self.config.scheduler.task_intervals["event_detection"],
                priority=TaskPriority.HIGH
            )
            
            # 4. 캐시 정리
            await self.scheduler.add_simple_interval_task(
                task_id="cache_cleanup",
                name="Cache Cleanup",
                function=self._cache_cleanup_task,
                interval_seconds=self.config.scheduler.task_intervals["cache_cleanup"],
                priority=TaskPriority.LOW
            )
            
            # 5. 시스템 유지보수 (매일 새벽 2시)
            await self.scheduler.add_cron_task(
                task_id="system_maintenance",
                name="System Maintenance",
                function=self._system_maintenance_task,
                cron_config={"hour": 2, "minute": 0},
                priority=TaskPriority.LOW
            )
            
            logger.info("Scheduled tasks configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduled tasks: {e}")
            raise
    
    async def _main_loop(self):
        """메인 실행 루프"""
        logger.info("Entering main loop...")
        
        try:
            while not self.shutdown_event.is_set():
                # 주기적 헬스 체크 및 통계 업데이트
                await self._update_service_stats()
                
                # 1분마다 체크
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=60.0)
                    break  # shutdown_event가 설정됨
                except asyncio.TimeoutError:
                    continue  # 타임아웃은 정상 - 계속 루프
                
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            self.health_monitor.report_error(str(e), "main_loop")
    
    async def _update_service_stats(self):
        """서비스 통계 업데이트"""
        try:
            # 각 컴포넌트 상태 업데이트
            if self.news_collector:
                collector_stats = self.news_collector.get_collection_stats()
                self.health_monitor.update_component_status("news_collector", "running", collector_stats)
            
            if self.macro_collector:
                macro_stats = self.macro_collector.get_collector_stats()
                self.health_monitor.update_component_status("macro_collector", "running", macro_stats)
            
            if self.expiry_monitor:
                expiry_stats = self.expiry_monitor.get_monitor_stats()
                self.health_monitor.update_component_status("expiry_monitor", "running", expiry_stats)
            
            if self.event_loader:
                loader_stats = self.event_loader.get_loader_stats()
                self.health_monitor.update_component_status("event_loader", "running", loader_stats)
            
            if self.finbert_processor:
                processor_stats = self.finbert_processor.get_processor_stats()
                self.health_monitor.update_component_status("finbert_processor", "running", processor_stats)
            
            if self.event_detector:
                detector_stats = self.event_detector.get_detector_stats()
                self.health_monitor.update_component_status("event_detector", "running", detector_stats)
            
            if self.impact_manager:
                impact_stats = self.impact_manager.get_manager_stats()
                self.health_monitor.update_component_status("impact_manager", "running", impact_stats)
            
            if self.fusion_manager:
                fusion_stats = self.fusion_manager.get_fusion_stats_v2()
                self.health_monitor.update_component_status("fusion_manager", "running", fusion_stats)
            
            if self.event_fusion:
                event_fusion_stats = self.event_fusion.get_fusion_stats()
                self.health_monitor.update_component_status("event_fusion", "running", event_fusion_stats)
            
            if self.scheduler:
                scheduler_stats = self.scheduler.get_scheduler_stats()
                self.health_monitor.update_component_status("scheduler", "running", scheduler_stats)
            
        except Exception as e:
            logger.error(f"Failed to update service stats: {e}")
    
    # 작업 함수들
    async def _news_collection_task(self):
        """뉴스 수집 작업"""
        logger.debug("Running news collection task...")
        
        try:
            if not self.news_collector:
                return
            
            default_symbols = ["BTC", "ETH", "CRYPTO"]
            all_news = []
            
            async with self.news_collector:
                for symbol in default_symbols:
                    news_items = await self.news_collector.collect_all_sources(
                        symbol=symbol,
                        hours_back=6,
                        max_per_source=self.config.collector.max_items_per_source
                    )
                    all_news.extend(news_items)
            
            logger.info(f"News collection completed: {len(all_news)} items")
            
            # FinBERT 프로세서 큐에 추가
            if self.finbert_processor and all_news:
                for news_item in all_news:
                    await self.finbert_processor.add_to_queue(
                        content_hash=news_item.hash_id,
                        title=news_item.title,
                        content=news_item.content,
                        source=news_item.source.value,
                        url=news_item.url,
                        published_at=news_item.published_at,
                        symbol=news_item.symbol,
                        category=news_item.category
                    )
                
                logger.debug(f"Added {len(all_news)} items to FinBERT queue")
            
        except Exception as e:
            logger.error(f"News collection task failed: {e}")
            self.health_monitor.report_error(str(e), "news_collection")
            raise
    
    async def _finbert_batch_task(self):
        """FinBERT 배치 처리 작업"""
        logger.debug("Running FinBERT batch task...")
        
        try:
            if not self.finbert_processor:
                return
            
            success = await self.finbert_processor.force_batch_run()
            
            if success:
                stats = self.finbert_processor.get_processor_stats()
                logger.info(f"FinBERT batch completed: {stats.get('queue_size', 0)} in queue, "
                           f"{stats.get('completed_items', 0)} completed")
            else:
                logger.warning("FinBERT batch processing returned no results")
            
        except Exception as e:
            logger.error(f"FinBERT batch task failed: {e}")
            self.health_monitor.report_error(str(e), "finbert_batch")
            raise
    
    async def _event_detection_task(self):
        """이벤트 감지 작업"""
        logger.debug("Running event detection task...")
        
        try:
            if not self.event_detector or not self.news_collector:
                return
            
            # 최근 뉴스 수집
            default_symbols = ["BTC", "ETH", "CRYPTO"]
            all_news = []
            
            async with self.news_collector:
                for symbol in default_symbols:
                    news_items = await self.news_collector.collect_all_sources(
                        symbol=symbol,
                        hours_back=12,
                        max_per_source=10
                    )
                    all_news.extend(news_items)
            
            if all_news:
                detected_events = await self.event_detector.detect_events_batch(all_news)
                logger.info(f"Event detection completed: {len(detected_events)} events detected")
                
                # 중요 이벤트 로깅
                for event in detected_events:
                    if event.urgency.value in ["immediate", "high"]:
                        logger.warning(f"Important event detected: {event.title} "
                                      f"(Impact: {event.impact_score:.1f}, Urgency: {event.urgency.value})")
            
        except Exception as e:
            logger.error(f"Event detection task failed: {e}")
            self.health_monitor.report_error(str(e), "event_detection")
            raise
    
    async def _cache_cleanup_task(self):
        """캐시 정리 작업"""
        logger.debug("Running cache cleanup task...")
        
        try:
            # Fusion Manager 캐시 정리
            if self.fusion_manager:
                cleaned = self.fusion_manager.cache_manager.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired fusion cache entries")
            
            # Event Detector 캐시 정리
            if self.event_detector:
                self.event_detector.cleanup_cache()
            
            # FinBERT Processor 캐시 정리
            if self.finbert_processor:
                self.finbert_processor.clear_cache()
            
            # 가비지 컬렉션
            import gc
            collected = gc.collect()
            logger.debug(f"Garbage collection: {collected} objects collected")
            
        except Exception as e:
            logger.error(f"Cache cleanup task failed: {e}")
            self.health_monitor.report_error(str(e), "cache_cleanup")
    
    async def _system_maintenance_task(self):
        """시스템 유지보수 작업"""
        logger.info("Running system maintenance task...")
        
        try:
            # 1. 로그 회전 (간단한 구현)
            # 실제로는 logrotate 사용 권장
            
            # 2. 메모리 정리
            import gc
            gc.collect()
            
            # 3. 통계 리셋 (선택적)
            if self.news_collector:
                stats = self.news_collector.get_collection_stats()
                if stats.get("total_requests", 0) > 10000:
                    await self.news_collector.cleanup()
                    logger.info("News collector statistics reset")
            
            # 4. 헬스 체크
            health_status = self.health_monitor.get_health_status()
            logger.info(f"System maintenance completed. Uptime: {health_status['uptime']:.0f}s")
            
        except Exception as e:
            logger.error(f"System maintenance task failed: {e}")
            self.health_monitor.report_error(str(e), "system_maintenance")
    
    async def shutdown(self):
        """서비스 종료"""
        if self.state == ServiceState.STOPPING:
            return
        
        self.state = ServiceState.STOPPING
        logger.info("Initiating graceful shutdown...")
        
        try:
            # 종료 이벤트 설정
            self.shutdown_event.set()
            
            # 컴포넌트들 순차적 종료 (의존성 역순)
            components = [
                ("scheduler", self.scheduler),
                ("event_fusion", self.event_fusion),
                ("event_detector", self.event_detector),
                ("fusion_manager", self.fusion_manager),
                ("impact_manager", self.impact_manager),
                ("finbert_processor", self.finbert_processor),
                ("event_loader", self.event_loader),
                ("expiry_monitor", self.expiry_monitor),
                ("macro_collector", self.macro_collector),
                ("news_collector", self.news_collector)
            ]
            
            for name, component in components:
                if component:
                    try:
                        logger.info(f"Stopping {name}...")
                        if hasattr(component, 'stop'):
                            await component.stop()
                        elif hasattr(component, '__aexit__'):
                            await component.__aexit__(None, None, None)
                        
                        self.health_monitor.update_component_status(name, "stopped")
                        logger.info(f"{name} stopped successfully")
                        
                    except Exception as e:
                        logger.error(f"Error stopping {name}: {e}")
            
            # 헬스 모니터 중지
            await self.health_monitor.stop()
            
            self.state = ServiceState.STOPPED
            logger.info("Service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.state = ServiceState.ERROR
    
    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            "service": {
                "name": self.config.service_name,
                "version": self.config.version,
                "state": self.state,
                "startup_complete": self.startup_complete,
                "startup_time": self.startup_time,
                "error_count": self.error_count
            },
            "health": self.health_monitor.get_health_status(),
            "config": {
                "deployment_mode": self.config.deployment_mode.value,
                "debug_enabled": self.config.enable_debug,
                "vps_limits": {
                    "max_memory_mb": self.config.vps_limits.max_memory_mb,
                    "max_concurrent_requests": self.config.vps_limits.max_concurrent_requests,
                    "max_batch_size": self.config.vps_limits.max_batch_size
                }
            }
        }

async def main():
    """메인 실행 함수"""
    # 명령줄 인수 처리
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # 서비스 런너 생성 및 실행
    runner = SentimentServiceRunner(config_path)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        return 1
    finally:
        await runner.shutdown()
    
    return 0

if __name__ == "__main__":
    # 서비스 실행
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nService interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Service failed to start: {e}")
        sys.exit(1)