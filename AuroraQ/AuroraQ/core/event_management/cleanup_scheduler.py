#!/usr/bin/env python3
"""
TTL 기반 자동 정리 시스템
P8-3: TTL-based Automatic Cleanup System
"""

import sys
import os
import asyncio
import sqlite3
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import time
from collections import defaultdict
import zipfile
import tarfile

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .ttl_event_manager import EventEntry, EventStatus, TTLEventManager, get_ttl_event_manager
from .expiry_processor import ExpiryProcessor, get_expiry_processor, ProcessingResult

class CleanupScope(Enum):
    """정리 범위"""
    EXPIRED_ONLY = "expired_only"           # 만료된 이벤트만
    PROCESSED_ONLY = "processed_only"       # 처리된 이벤트만
    ARCHIVED_ONLY = "archived_only"         # 아카이브된 이벤트만
    CANCELLED_ONLY = "cancelled_only"       # 취소된 이벤트만
    ALL_INACTIVE = "all_inactive"           # 모든 비활성 이벤트
    DATABASE_OPTIMIZATION = "db_optimization"  # 데이터베이스 최적화

class CleanupAction(Enum):
    """정리 액션"""
    DELETE = "delete"                       # 완전 삭제
    ARCHIVE_TO_FILE = "archive_to_file"    # 파일로 아카이브
    COMPRESS = "compress"                   # 압축 저장
    BACKUP_AND_DELETE = "backup_and_delete"  # 백업 후 삭제
    MOVE_TO_COLD_STORAGE = "move_to_cold"  # 콜드 스토리지 이동
    SUMMARIZE = "summarize"                 # 요약 정보만 보관

class CleanupFrequency(Enum):
    """정리 주기"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

@dataclass
class CleanupPolicy:
    """정리 정책"""
    policy_id: str
    scope: CleanupScope
    action: CleanupAction
    frequency: CleanupFrequency
    
    # 조건 설정
    age_threshold_hours: int = 24           # 최소 나이 (시간)
    max_records_to_keep: Optional[int] = None  # 유지할 최대 레코드 수
    size_threshold_mb: Optional[float] = None  # 크기 임계값 (MB)
    
    # 필터 조건
    event_type_patterns: List[str] = field(default_factory=list)  # 이벤트 타입 패턴
    tag_filters: List[str] = field(default_factory=list)         # 태그 필터
    priority_filters: List[str] = field(default_factory=list)    # 우선순위 필터
    
    # 실행 조건
    enabled: bool = True
    dry_run: bool = False                   # 시뮬레이션 모드
    require_confirmation: bool = False      # 확인 필요
    
    # 보관 설정
    archive_path: Optional[str] = None      # 아카이브 경로
    compression_level: int = 6              # 압축 레벨 (1-9)
    encryption_enabled: bool = False        # 암호화 사용
    
    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    description: str = ""

@dataclass
class CleanupResult:
    """정리 결과"""
    policy_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # 처리 통계
    records_scanned: int = 0
    records_cleaned: int = 0
    records_archived: int = 0
    records_failed: int = 0
    
    # 크기 정보
    bytes_cleaned: int = 0
    bytes_archived: int = 0
    
    # 성능 정보
    execution_time_seconds: float = 0.0
    throughput_records_per_second: float = 0.0
    
    # 오류 정보
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 상태
    success: bool = False
    dry_run: bool = False

@dataclass
class CleanupMetrics:
    """정리 메트릭"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    
    total_records_cleaned: int = 0
    total_bytes_cleaned: int = 0
    
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    
    # 정책별 통계
    policy_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class CleanupScheduler:
    """TTL 기반 자동 정리 스케줄러"""
    
    def __init__(self, ttl_manager=None, expiry_processor=None):
        # 로거 초기화
        self.logger = logging.getLogger(__name__)
        
        # 매니저 참조
        self.ttl_manager = ttl_manager or get_ttl_event_manager()
        self.expiry_processor = expiry_processor or get_expiry_processor()
        
        # 정책 관리
        self.cleanup_policies: Dict[str, CleanupPolicy] = {}
        
        # 실행 제어
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.policy_tasks: Dict[str, asyncio.Task] = {}
        
        # 메트릭
        self.metrics = CleanupMetrics()
        self.execution_history: List[CleanupResult] = []
        
        # 설정
        self.base_archive_path = Path("archives")
        self.base_archive_path.mkdir(parents=True, exist_ok=True)
        
        self.temp_path = Path("temp_cleanup")
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 정책 로드
        self._load_default_policies()
        
        self.logger.info("Cleanup Scheduler initialized")
    
    def _load_default_policies(self):
        """기본 정리 정책 로드"""
        
        default_policies = [
            # 일일 만료 이벤트 정리
            CleanupPolicy(
                policy_id="daily_expired_cleanup",
                scope=CleanupScope.EXPIRED_ONLY,
                action=CleanupAction.BACKUP_AND_DELETE,
                frequency=CleanupFrequency.DAILY,
                age_threshold_hours=24,
                archive_path=str(self.base_archive_path / "expired"),
                description="매일 24시간 이상 된 만료 이벤트 백업 후 삭제"
            ),
            
            # 주간 처리된 이벤트 아카이브
            CleanupPolicy(
                policy_id="weekly_processed_archive",
                scope=CleanupScope.PROCESSED_ONLY,
                action=CleanupAction.ARCHIVE_TO_FILE,
                frequency=CleanupFrequency.WEEKLY,
                age_threshold_hours=168,  # 1주일
                max_records_to_keep=1000,
                archive_path=str(self.base_archive_path / "processed"),
                compression_level=7,
                description="주간 처리된 이벤트 압축 아카이브"
            ),
            
            # 월간 아카이브된 이벤트 압축
            CleanupPolicy(
                policy_id="monthly_archive_compress",
                scope=CleanupScope.ARCHIVED_ONLY,
                action=CleanupAction.COMPRESS,
                frequency=CleanupFrequency.MONTHLY,
                age_threshold_hours=720,  # 30일
                archive_path=str(self.base_archive_path / "compressed"),
                compression_level=9,
                description="월간 오래된 아카이브 이벤트 고압축"
            ),
            
            # 시간별 취소된 이벤트 정리
            CleanupPolicy(
                policy_id="hourly_cancelled_cleanup",
                scope=CleanupScope.CANCELLED_ONLY,
                action=CleanupAction.DELETE,
                frequency=CleanupFrequency.HOURLY,
                age_threshold_hours=1,
                description="시간별 취소된 이벤트 즉시 삭제"
            ),
            
            # 데이터베이스 최적화
            CleanupPolicy(
                policy_id="weekly_db_optimization",
                scope=CleanupScope.DATABASE_OPTIMIZATION,
                action=CleanupAction.SUMMARIZE,
                frequency=CleanupFrequency.WEEKLY,
                description="주간 데이터베이스 최적화 및 통계 갱신"
            )
        ]
        
        for policy in default_policies:
            self.cleanup_policies[policy.policy_id] = policy
        
        self.logger.info(f"Loaded {len(default_policies)} default cleanup policies")
    
    def add_cleanup_policy(self, policy: CleanupPolicy):
        """정리 정책 추가"""
        self.cleanup_policies[policy.policy_id] = policy
        self.logger.info(f"Added cleanup policy: {policy.policy_id}")
    
    def remove_cleanup_policy(self, policy_id: str) -> bool:
        """정리 정책 제거"""
        if policy_id in self.cleanup_policies:
            # 실행 중인 태스크 취소
            if policy_id in self.policy_tasks:
                self.policy_tasks[policy_id].cancel()
                del self.policy_tasks[policy_id]
            
            del self.cleanup_policies[policy_id]
            self.logger.info(f"Removed cleanup policy: {policy_id}")
            return True
        return False
    
    async def start_scheduler(self):
        """스케줄러 시작"""
        if self.is_running:
            self.logger.warning("Cleanup scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Cleanup scheduler started")
    
    async def stop_scheduler(self):
        """스케줄러 중지"""
        self.is_running = False
        
        # 메인 스케줄러 태스크 취소
        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 모든 정책 태스크 취소
        for task in self.policy_tasks.values():
            if not task.done():
                task.cancel()
        
        # 태스크 완료 대기
        if self.policy_tasks:
            await asyncio.gather(*self.policy_tasks.values(), return_exceptions=True)
        
        self.policy_tasks.clear()
        self.logger.info("Cleanup scheduler stopped")
    
    async def _scheduler_loop(self):
        """스케줄러 루프"""
        while self.is_running:
            try:
                # 실행할 정책들 확인
                policies_to_execute = self._get_policies_to_execute()
                
                # 정책 실행
                for policy in policies_to_execute:
                    if policy.policy_id not in self.policy_tasks or self.policy_tasks[policy.policy_id].done():
                        self.policy_tasks[policy.policy_id] = asyncio.create_task(
                            self._execute_cleanup_policy(policy)
                        )
                
                # 완료된 태스크 정리
                completed_tasks = [
                    policy_id for policy_id, task in self.policy_tasks.items()
                    if task.done()
                ]
                for policy_id in completed_tasks:
                    del self.policy_tasks[policy_id]
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(300)  # 5분 대기
    
    def _get_policies_to_execute(self) -> List[CleanupPolicy]:
        """실행할 정책들 반환"""
        now = datetime.now()
        policies_to_execute = []
        
        for policy in self.cleanup_policies.values():
            if not policy.enabled:
                continue
            
            # 이미 실행 중인 경우 스킵
            if policy.policy_id in self.policy_tasks and not self.policy_tasks[policy.policy_id].done():
                continue
            
            should_execute = False
            
            # 주기별 실행 조건 확인
            if policy.frequency == CleanupFrequency.HOURLY:
                if not policy.last_executed or (now - policy.last_executed).total_seconds() >= 3600:
                    should_execute = True
            
            elif policy.frequency == CleanupFrequency.DAILY:
                if not policy.last_executed or (now - policy.last_executed).days >= 1:
                    should_execute = True
            
            elif policy.frequency == CleanupFrequency.WEEKLY:
                if not policy.last_executed or (now - policy.last_executed).days >= 7:
                    should_execute = True
            
            elif policy.frequency == CleanupFrequency.MONTHLY:
                if not policy.last_executed or (now - policy.last_executed).days >= 30:
                    should_execute = True
            
            if should_execute:
                policies_to_execute.append(policy)
        
        return policies_to_execute
    
    async def _execute_cleanup_policy(self, policy: CleanupPolicy) -> CleanupResult:
        """정리 정책 실행"""
        execution_id = str(uuid.uuid4())
        result = CleanupResult(
            policy_id=policy.policy_id,
            execution_id=execution_id,
            started_at=datetime.now(),
            dry_run=policy.dry_run
        )
        
        try:
            self.logger.info(f"Executing cleanup policy: {policy.policy_id} ({execution_id})")
            
            # 범위별 처리
            if policy.scope == CleanupScope.DATABASE_OPTIMIZATION:
                await self._execute_database_optimization(policy, result)
            else:
                await self._execute_event_cleanup(policy, result)
            
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            
            if result.execution_time_seconds > 0:
                result.throughput_records_per_second = result.records_scanned / result.execution_time_seconds
            
            result.success = len(result.errors) == 0
            
            # 정책 업데이트
            policy.last_executed = result.completed_at
            
            # 메트릭 업데이트
            self._update_metrics(result)
            
            # 히스토리 저장
            self.execution_history.append(result)
            if len(self.execution_history) > 100:  # 최대 100개 유지
                self.execution_history.pop(0)
            
            self.logger.info(
                f"Cleanup policy completed: {policy.policy_id} - "
                f"Cleaned: {result.records_cleaned}, "
                f"Archived: {result.records_archived}, "
                f"Time: {result.execution_time_seconds:.2f}s"
            )
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            result.success = False
            result.errors.append(f"Policy execution failed: {e}")
            self.logger.error(f"Cleanup policy failed: {policy.policy_id} - {e}")
        
        return result
    
    async def _execute_event_cleanup(self, policy: CleanupPolicy, result: CleanupResult):
        """이벤트 정리 실행"""
        # 대상 이벤트 조회
        target_events = await self._get_target_events(policy)
        result.records_scanned = len(target_events)
        
        if not target_events:
            self.logger.debug(f"No events to clean for policy: {policy.policy_id}")
            return
        
        # 액션별 처리
        for event in target_events:
            try:
                if policy.action == CleanupAction.DELETE:
                    await self._delete_event(event, policy, result)
                
                elif policy.action == CleanupAction.ARCHIVE_TO_FILE:
                    await self._archive_event_to_file(event, policy, result)
                
                elif policy.action == CleanupAction.COMPRESS:
                    await self._compress_event(event, policy, result)
                
                elif policy.action == CleanupAction.BACKUP_AND_DELETE:
                    await self._backup_and_delete_event(event, policy, result)
                
                elif policy.action == CleanupAction.MOVE_TO_COLD_STORAGE:
                    await self._move_to_cold_storage(event, policy, result)
                
                elif policy.action == CleanupAction.SUMMARIZE:
                    await self._summarize_event(event, policy, result)
                
                result.records_cleaned += 1
                
            except Exception as e:
                result.records_failed += 1
                result.errors.append(f"Failed to process event {event.event_id}: {e}")
    
    async def _get_target_events(self, policy: CleanupPolicy) -> List[EventEntry]:
        """대상 이벤트 조회"""
        cutoff_time = datetime.now() - timedelta(hours=policy.age_threshold_hours)
        
        # 상태별 필터링
        target_statuses = []
        if policy.scope == CleanupScope.EXPIRED_ONLY:
            target_statuses = [EventStatus.EXPIRED]
        elif policy.scope == CleanupScope.PROCESSED_ONLY:
            target_statuses = [EventStatus.PROCESSED]
        elif policy.scope == CleanupScope.ARCHIVED_ONLY:
            target_statuses = [EventStatus.ARCHIVED]
        elif policy.scope == CleanupScope.CANCELLED_ONLY:
            target_statuses = [EventStatus.CANCELLED]
        elif policy.scope == CleanupScope.ALL_INACTIVE:
            target_statuses = [EventStatus.EXPIRED, EventStatus.PROCESSED, 
                             EventStatus.ARCHIVED, EventStatus.CANCELLED]
        
        all_events = []
        for status in target_statuses:
            events = await self.ttl_manager.list_events(status=status)
            all_events.extend(events)
        
        # 나이 필터링
        filtered_events = [
            event for event in all_events
            if (event.processed_at or event.created_at) < cutoff_time
        ]
        
        # 추가 필터링
        if policy.event_type_patterns:
            import re
            filtered_events = [
                event for event in filtered_events
                if any(re.match(pattern, event.event_type) for pattern in policy.event_type_patterns)
            ]
        
        if policy.tag_filters:
            filtered_events = [
                event for event in filtered_events
                if any(tag in event.tags for tag in policy.tag_filters)
            ]
        
        if policy.priority_filters:
            filtered_events = [
                event for event in filtered_events
                if event.priority.value in policy.priority_filters
            ]
        
        # 최대 레코드 수 제한
        if policy.max_records_to_keep:
            # 최신 순으로 정렬 후 제한
            filtered_events.sort(key=lambda x: x.created_at, reverse=True)
            if len(filtered_events) > policy.max_records_to_keep:
                filtered_events = filtered_events[policy.max_records_to_keep:]
        
        return filtered_events
    
    async def _delete_event(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """이벤트 삭제"""
        if not policy.dry_run:
            await self.ttl_manager.delete_event(event.event_id)
        
        # 크기 계산 (JSON 직렬화 크기 추정)
        event_size = len(json.dumps(event.to_dict()).encode('utf-8'))
        result.bytes_cleaned += event_size
    
    async def _archive_event_to_file(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """이벤트를 파일로 아카이브"""
        if not policy.archive_path:
            raise ValueError("Archive path not specified")
        
        archive_dir = Path(policy.archive_path)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 날짜별 디렉토리 구성
        date_dir = archive_dir / event.created_at.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 아카이브 파일 경로
        archive_file = date_dir / f"{event.event_id}.json"
        
        if not policy.dry_run:
            # JSON으로 저장
            event_data = {
                "archived_at": datetime.now().isoformat(),
                "policy_id": policy.policy_id,
                "event": event.to_dict()
            }
            
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, ensure_ascii=False)
            
            # 압축 옵션
            if policy.compression_level > 0:
                await self._compress_file(archive_file, policy.compression_level)
            
            # 원본 삭제
            await self.ttl_manager.delete_event(event.event_id)
        
        result.records_archived += 1
        result.bytes_archived += archive_file.stat().st_size if archive_file.exists() else 0
    
    async def _compress_event(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """이벤트 압축"""
        await self._archive_event_to_file(event, policy, result)
    
    async def _backup_and_delete_event(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """이벤트 백업 후 삭제"""
        # 백업 (아카이브와 동일)
        await self._archive_event_to_file(event, policy, result)
    
    async def _move_to_cold_storage(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """콜드 스토리지로 이동"""
        # 간단 구현: 특별한 아카이브 디렉토리 사용
        cold_policy = CleanupPolicy(
            policy_id=f"{policy.policy_id}_cold",
            scope=policy.scope,
            action=CleanupAction.ARCHIVE_TO_FILE,
            frequency=policy.frequency,
            archive_path=str(Path(policy.archive_path or "cold_storage") / "cold"),
            compression_level=9,  # 최대 압축
            dry_run=policy.dry_run
        )
        
        await self._archive_event_to_file(event, cold_policy, result)
    
    async def _summarize_event(self, event: EventEntry, policy: CleanupPolicy, result: CleanupResult):
        """이벤트 요약"""
        # 요약 정보만 남기고 원본 삭제
        summary = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "status": event.status.value,
            "priority": event.priority.value,
            "created_at": event.created_at.isoformat(),
            "expires_at": event.expires_at.isoformat(),
            "processed_at": event.processed_at.isoformat() if event.processed_at else None,
            "ttl_seconds": event.ttl_seconds,
            "access_count": event.access_count,
            "tags": event.tags,
            "summarized_at": datetime.now().isoformat(),
            "summary_policy": policy.policy_id
        }
        
        if policy.archive_path:
            summary_dir = Path(policy.archive_path) / "summaries"
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = summary_dir / f"summary_{event.event_id}.json"
            
            if not policy.dry_run:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                # 원본 삭제
                await self.ttl_manager.delete_event(event.event_id)
    
    async def _compress_file(self, file_path: Path, compression_level: int):
        """파일 압축"""
        if compression_level <= 0:
            return
        
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            import gzip
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 원본 파일 삭제
            file_path.unlink()
            
        except Exception as e:
            self.logger.warning(f"Failed to compress file {file_path}: {e}")
    
    async def _execute_database_optimization(self, policy: CleanupPolicy, result: CleanupResult):
        """데이터베이스 최적화"""
        try:
            if not policy.dry_run:
                # SQLite VACUUM 실행
                with sqlite3.connect(str(self.ttl_manager.db_path)) as conn:
                    # 통계 업데이트
                    conn.execute("ANALYZE")
                    
                    # 데이터베이스 압축
                    conn.execute("VACUUM")
                    
                    # 인덱스 재구성
                    conn.execute("REINDEX")
                    
                    conn.commit()
            
            # 데이터베이스 크기 정보
            if self.ttl_manager.db_path.exists():
                db_size = self.ttl_manager.db_path.stat().st_size
                result.bytes_cleaned = db_size  # 최적화된 크기
            
            result.records_cleaned = 1  # 최적화 작업 1회
            
        except Exception as e:
            result.errors.append(f"Database optimization failed: {e}")
    
    def _update_metrics(self, result: CleanupResult):
        """메트릭 업데이트"""
        self.metrics.total_executions += 1
        
        if result.success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        self.metrics.total_records_cleaned += result.records_cleaned
        self.metrics.total_bytes_cleaned += result.bytes_cleaned
        
        # 평균 실행 시간 업데이트
        if self.metrics.total_executions > 0:
            total_time = (self.metrics.average_execution_time * (self.metrics.total_executions - 1) + 
                         result.execution_time_seconds)
            self.metrics.average_execution_time = total_time / self.metrics.total_executions
        
        self.metrics.last_execution_time = result.completed_at
        
        # 정책별 통계
        if result.policy_id not in self.metrics.policy_stats:
            self.metrics.policy_stats[result.policy_id] = {
                "executions": 0,
                "records_cleaned": 0,
                "bytes_cleaned": 0,
                "last_execution": None
            }
        
        policy_stats = self.metrics.policy_stats[result.policy_id]
        policy_stats["executions"] += 1
        policy_stats["records_cleaned"] += result.records_cleaned
        policy_stats["bytes_cleaned"] += result.bytes_cleaned
        policy_stats["last_execution"] = result.completed_at.isoformat() if result.completed_at else None
    
    async def execute_policy_now(self, policy_id: str) -> CleanupResult:
        """정책 즉시 실행"""
        policy = self.cleanup_policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy not found: {policy_id}")
        
        return await self._execute_cleanup_policy(policy)
    
    def get_cleanup_policies(self) -> List[CleanupPolicy]:
        """정리 정책 목록 조회"""
        return list(self.cleanup_policies.values())
    
    def get_execution_history(self, limit: int = 50) -> List[CleanupResult]:
        """실행 히스토리 조회"""
        return self.execution_history[-limit:]
    
    def get_scheduler_metrics(self) -> CleanupMetrics:
        """스케줄러 메트릭 조회"""
        return self.metrics
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        return {
            "is_running": self.is_running,
            "active_policies": len([p for p in self.cleanup_policies.values() if p.enabled]),
            "total_policies": len(self.cleanup_policies),
            "running_tasks": len(self.policy_tasks),
            "metrics": {
                "total_executions": self.metrics.total_executions,
                "success_rate": (
                    self.metrics.successful_executions / max(1, self.metrics.total_executions) * 100
                ),
                "total_records_cleaned": self.metrics.total_records_cleaned,
                "total_bytes_cleaned": self.metrics.total_bytes_cleaned,
                "average_execution_time": self.metrics.average_execution_time,
                "last_execution": self.metrics.last_execution_time.isoformat() if self.metrics.last_execution_time else None
            },
            "next_executions": self._get_next_execution_times()
        }
    
    def _get_next_execution_times(self) -> Dict[str, str]:
        """다음 실행 시간 예측"""
        next_executions = {}
        now = datetime.now()
        
        for policy in self.cleanup_policies.values():
            if not policy.enabled:
                continue
            
            next_time = None
            
            if policy.frequency == CleanupFrequency.HOURLY:
                if policy.last_executed:
                    next_time = policy.last_executed + timedelta(hours=1)
                else:
                    next_time = now + timedelta(hours=1)
            
            elif policy.frequency == CleanupFrequency.DAILY:
                if policy.last_executed:
                    next_time = policy.last_executed + timedelta(days=1)
                else:
                    next_time = now + timedelta(days=1)
            
            elif policy.frequency == CleanupFrequency.WEEKLY:
                if policy.last_executed:
                    next_time = policy.last_executed + timedelta(weeks=1)
                else:
                    next_time = now + timedelta(weeks=1)
            
            elif policy.frequency == CleanupFrequency.MONTHLY:
                if policy.last_executed:
                    next_time = policy.last_executed + timedelta(days=30)
                else:
                    next_time = now + timedelta(days=30)
            
            if next_time:
                next_executions[policy.policy_id] = next_time.isoformat()
        
        return next_executions
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            await self.stop_scheduler()
            
            # 임시 파일 정리
            if self.temp_path.exists():
                shutil.rmtree(self.temp_path, ignore_errors=True)
            
            self.logger.info("Cleanup Scheduler cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Cleanup Scheduler cleanup failed: {e}")

# 전역 스케줄러
_global_cleanup_scheduler = None

def get_cleanup_scheduler(ttl_manager=None, expiry_processor=None) -> CleanupScheduler:
    """전역 정리 스케줄러 반환"""
    global _global_cleanup_scheduler
    if _global_cleanup_scheduler is None:
        _global_cleanup_scheduler = CleanupScheduler(ttl_manager, expiry_processor)
    return _global_cleanup_scheduler

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_cleanup_scheduler():
        print("🧪 Cleanup Scheduler 테스트")
        
        # 스케줄러 초기화
        scheduler = CleanupScheduler()
        
        print("\n1️⃣ 기본 정책 확인")
        policies = scheduler.get_cleanup_policies()
        print(f"  기본 정책 수: {len(policies)}")
        for policy in policies[:3]:
            print(f"  - {policy.policy_id}: {policy.description}")
        
        print("\n2️⃣ 커스텀 정책 추가")
        custom_policy = CleanupPolicy(
            policy_id="test_custom_policy",
            scope=CleanupScope.EXPIRED_ONLY,
            action=CleanupAction.DELETE,
            frequency=CleanupFrequency.HOURLY,
            age_threshold_hours=1,
            dry_run=True,  # 테스트용 시뮬레이션
            description="테스트용 커스텀 정책"
        )
        scheduler.add_cleanup_policy(custom_policy)
        print(f"  추가된 정책: {custom_policy.policy_id}")
        
        print("\n3️⃣ 정책 즉시 실행 (시뮬레이션)")
        try:
            result = await scheduler.execute_policy_now("test_custom_policy")
            print(f"  실행 결과:")
            print(f"  - 스캔된 레코드: {result.records_scanned}")
            print(f"  - 정리된 레코드: {result.records_cleaned}")
            print(f"  - 실행 시간: {result.execution_time_seconds:.2f}초")
            print(f"  - 성공 여부: {result.success}")
        except Exception as e:
            print(f"  실행 오류: {e}")
        
        print("\n4️⃣ 스케줄러 상태 확인")
        status = scheduler.get_scheduler_status()
        print(f"  활성 정책: {status['active_policies']}")
        print(f"  총 정책: {status['total_policies']}")
        print(f"  성공률: {status['metrics']['success_rate']:.1f}%")
        
        print("\n5️⃣ 다음 실행 시간")
        next_executions = status['next_executions']
        for policy_id, next_time in list(next_executions.items())[:3]:
            print(f"  {policy_id}: {next_time}")
        
        print("\n6️⃣ 메트릭 조회")
        metrics = scheduler.get_scheduler_metrics()
        print(f"  총 실행 횟수: {metrics.total_executions}")
        print(f"  총 정리된 레코드: {metrics.total_records_cleaned}")
        print(f"  평균 실행 시간: {metrics.average_execution_time:.2f}초")
        
        print("\n🎉 Cleanup Scheduler 테스트 완료!")
        
        # 정리
        await scheduler.cleanup()
    
    # 테스트 실행
    asyncio.run(test_cleanup_scheduler())