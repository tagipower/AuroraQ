#!/usr/bin/env python3
"""
TTL 이벤트 관리자
P8-1: TTL Event Manager Implementation
"""

import sys
import os
import asyncio
import threading
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import time
from collections import defaultdict

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class EventStatus(Enum):
    """이벤트 상태"""
    ACTIVE = "active"           # 활성 상태
    EXPIRED = "expired"         # 만료됨
    PROCESSED = "processed"     # 처리됨
    ARCHIVED = "archived"       # 아카이브됨
    CANCELLED = "cancelled"     # 취소됨

class EventPriority(Enum):
    """이벤트 우선순위"""
    CRITICAL = "critical"       # 즉시 처리 필요
    HIGH = "high"              # 높은 우선순위
    MEDIUM = "medium"          # 보통 우선순위
    LOW = "low"                # 낮은 우선순위

class TTLAction(Enum):
    """TTL 액션 타입"""
    DELETE = "delete"          # 삭제
    ARCHIVE = "archive"        # 아카이브
    CALLBACK = "callback"      # 콜백 실행
    NOTIFY = "notify"          # 알림
    EXTEND = "extend"          # 만료 시간 연장

@dataclass
class TTLConfig:
    """TTL 설정"""
    # 기본 설정
    db_path: str = "event_ttl.db"
    temp_dir: str = "temp_events"
    
    # TTL 기본값 (초)
    default_ttl_seconds: int = 3600  # 1시간
    max_ttl_seconds: int = 86400 * 7  # 1주일
    min_ttl_seconds: int = 60        # 1분
    
    # 청소 설정
    cleanup_interval_minutes: int = 10  # 청소 주기 (분)
    expired_retention_hours: int = 24   # 만료된 이벤트 보관 시간
    batch_size: int = 100              # 배치 처리 크기
    
    # 성능 설정
    enable_monitoring: bool = True
    enable_callbacks: bool = True
    max_concurrent_callbacks: int = 5
    callback_timeout_seconds: int = 30
    
    # 알림 설정
    enable_notifications: bool = True
    notification_lead_time_minutes: int = 5  # 만료 전 알림 시간

@dataclass
class EventEntry:
    """이벤트 엔트리"""
    event_id: str
    event_type: str
    status: EventStatus
    priority: EventPriority
    created_at: datetime
    expires_at: datetime
    ttl_seconds: int
    
    # 이벤트 데이터
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # TTL 액션
    actions: List[TTLAction] = field(default_factory=list)
    callback_url: Optional[str] = None
    callback_data: Optional[Dict[str, Any]] = None
    
    # 처리 정보
    processed_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    
    # 메타데이터
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        return datetime.now() >= self.expires_at
    
    def is_near_expiry(self, lead_time_minutes: int = 5) -> bool:
        """만료 임박 여부 확인"""
        warning_time = self.expires_at - timedelta(minutes=lead_time_minutes)
        return datetime.now() >= warning_time
    
    def time_to_expiry(self) -> timedelta:
        """만료까지 남은 시간"""
        return self.expires_at - datetime.now()
    
    def extend_ttl(self, additional_seconds: int):
        """TTL 연장"""
        self.expires_at += timedelta(seconds=additional_seconds)
        self.ttl_seconds += additional_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'data': self.data,
            'tags': self.tags,
            'actions': [action.value for action in self.actions],
            'callback_url': self.callback_url,
            'callback_data': self.callback_data,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'last_accessed_at': self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            'access_count': self.access_count,
            'source': self.source,
            'metadata': self.metadata
        }

@dataclass
class TTLMetrics:
    """TTL 메트릭"""
    total_events: int = 0
    active_events: int = 0
    expired_events: int = 0
    processed_events: int = 0
    archived_events: int = 0
    cancelled_events: int = 0
    
    # 처리 통계
    events_per_minute: float = 0.0
    average_ttl_seconds: float = 0.0
    callback_success_rate: float = 100.0
    cleanup_efficiency: float = 100.0
    
    # 시간 분포
    ttl_distribution: Dict[str, int] = field(default_factory=dict)
    priority_distribution: Dict[str, int] = field(default_factory=dict)

class TTLEventManager:
    """TTL 이벤트 관리자"""
    
    def __init__(self, config: Optional[TTLConfig] = None):
        # 로거 초기화 (가장 먼저)
        self.logger = logging.getLogger(__name__)
        
        # 설정 초기화
        self.config = config or TTLConfig()
        
        # 데이터베이스 경로 설정
        self.db_path = Path(self.config.db_path)
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 인메모리 캐시
        self.events_cache: Dict[str, EventEntry] = {}
        self.expiry_queue: List[EventEntry] = []  # 만료 시간 순 정렬
        
        # 콜백 관리
        self.callback_handlers: Dict[str, Callable] = {}
        
        # 동기화
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # 메트릭
        self.metrics = TTLMetrics()
        self._last_metrics_update = datetime.now()
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 기존 이벤트 로드
        self._load_events_from_database()
        
        self.logger.info("TTL Event Manager initialized")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    data TEXT,
                    tags TEXT,
                    actions TEXT,
                    callback_url TEXT,
                    callback_data TEXT,
                    processed_at TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'unknown',
                    metadata TEXT
                )
                """)
                
                # 인덱스 생성
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_expires_at ON events(expires_at)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_status ON events(status)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_priority ON events(priority)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_events_from_database(self):
        """데이터베이스에서 이벤트 로드"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                SELECT * FROM events 
                WHERE status IN ('active', 'expired') 
                ORDER BY expires_at
                """)
                
                loaded_count = 0
                for row in cursor.fetchall():
                    try:
                        event = self._row_to_event(row)
                        self.events_cache[event.event_id] = event
                        
                        # 만료 큐에 추가 (활성 이벤트만)
                        if event.status == EventStatus.ACTIVE:
                            self.expiry_queue.append(event)
                        
                        loaded_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load event from row: {e}")
                
                # 만료 큐 정렬
                self.expiry_queue.sort(key=lambda x: x.expires_at)
                
                self.logger.info(f"Loaded {loaded_count} events from database")
                
        except Exception as e:
            self.logger.error(f"Failed to load events from database: {e}")
    
    def _row_to_event(self, row) -> EventEntry:
        """데이터베이스 행을 EventEntry로 변환"""
        return EventEntry(
            event_id=row[0],
            event_type=row[1],
            status=EventStatus(row[2]),
            priority=EventPriority(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            expires_at=datetime.fromisoformat(row[5]),
            ttl_seconds=row[6],
            data=json.loads(row[7]) if row[7] else {},
            tags=json.loads(row[8]) if row[8] else [],
            actions=[TTLAction(action) for action in json.loads(row[9])] if row[9] else [],
            callback_url=row[10],
            callback_data=json.loads(row[11]) if row[11] else None,
            processed_at=datetime.fromisoformat(row[12]) if row[12] else None,
            last_accessed_at=datetime.fromisoformat(row[13]) if row[13] else None,
            access_count=row[14] or 0,
            source=row[15] or "unknown",
            metadata=json.loads(row[16]) if row[16] else {}
        )
    
    async def create_event(self, 
                          event_type: str,
                          ttl_seconds: Optional[int] = None,
                          priority: EventPriority = EventPriority.MEDIUM,
                          data: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None,
                          actions: Optional[List[TTLAction]] = None,
                          callback_url: Optional[str] = None,
                          callback_data: Optional[Dict[str, Any]] = None,
                          source: str = "unknown",
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """이벤트 생성"""
        
        # TTL 검증 및 기본값 설정
        if ttl_seconds is None:
            ttl_seconds = self.config.default_ttl_seconds
        
        ttl_seconds = max(self.config.min_ttl_seconds, 
                         min(ttl_seconds, self.config.max_ttl_seconds))
        
        # 이벤트 생성
        event_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        event = EventEntry(
            event_id=event_id,
            event_type=event_type,
            status=EventStatus.ACTIVE,
            priority=priority,
            created_at=now,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
            data=data or {},
            tags=tags or [],
            actions=actions or [],
            callback_url=callback_url,
            callback_data=callback_data,
            source=source,
            metadata=metadata or {}
        )
        
        # 캐시에 저장
        with self._lock:
            self.events_cache[event_id] = event
            
            # 만료 큐에 삽입 (정렬 유지)
            import bisect
            bisect.insort(self.expiry_queue, event, key=lambda x: x.expires_at)
        
        # 데이터베이스에 저장
        await self._save_event_to_database(event)
        
        self.logger.debug(f"Created event {event_id} with TTL {ttl_seconds}s")
        return event_id
    
    async def _save_event_to_database(self, event: EventEntry):
        """이벤트를 데이터베이스에 저장"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO events 
                (event_id, event_type, status, priority, created_at, expires_at, 
                 ttl_seconds, data, tags, actions, callback_url, callback_data,
                 processed_at, last_accessed_at, access_count, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type,
                    event.status.value,
                    event.priority.value,
                    event.created_at.isoformat(),
                    event.expires_at.isoformat(),
                    event.ttl_seconds,
                    json.dumps(event.data),
                    json.dumps(event.tags),
                    json.dumps([action.value for action in event.actions]),
                    event.callback_url,
                    json.dumps(event.callback_data) if event.callback_data else None,
                    event.processed_at.isoformat() if event.processed_at else None,
                    event.last_accessed_at.isoformat() if event.last_accessed_at else None,
                    event.access_count,
                    event.source,
                    json.dumps(event.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save event {event.event_id}: {e}")
    
    async def get_event(self, event_id: str) -> Optional[EventEntry]:
        """이벤트 조회"""
        with self._lock:
            event = self.events_cache.get(event_id)
            
            if event:
                # 접근 정보 업데이트
                event.last_accessed_at = datetime.now()
                event.access_count += 1
                
                # 데이터베이스 업데이트 (비동기)
                asyncio.create_task(self._save_event_to_database(event))
            
            return event
    
    async def update_event(self, event_id: str, **kwargs) -> bool:
        """이벤트 업데이트"""
        with self._lock:
            event = self.events_cache.get(event_id)
            
            if not event:
                return False
            
            # 업데이트 가능한 필드들
            updatable_fields = ['data', 'tags', 'actions', 'callback_url', 'callback_data', 'metadata']
            
            for field, value in kwargs.items():
                if field in updatable_fields and hasattr(event, field):
                    setattr(event, field, value)
            
            # TTL 연장 처리
            if 'extend_ttl_seconds' in kwargs:
                event.extend_ttl(kwargs['extend_ttl_seconds'])
                # 만료 큐 재정렬 필요
                self.expiry_queue.sort(key=lambda x: x.expires_at)
            
            # 데이터베이스 업데이트
            await self._save_event_to_database(event)
            
            return True
    
    async def delete_event(self, event_id: str) -> bool:
        """이벤트 삭제"""
        with self._lock:
            event = self.events_cache.pop(event_id, None)
            
            if event:
                # 만료 큐에서 제거
                try:
                    self.expiry_queue.remove(event)
                except ValueError:
                    pass  # 이미 제거됨
                
                # 데이터베이스에서 삭제
                try:
                    with sqlite3.connect(str(self.db_path)) as conn:
                        conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
                        conn.commit()
                except Exception as e:
                    self.logger.error(f"Failed to delete event {event_id} from database: {e}")
                
                return True
            
            return False
    
    async def list_events(self, 
                         status: Optional[EventStatus] = None,
                         event_type: Optional[str] = None,
                         priority: Optional[EventPriority] = None,
                         tags: Optional[List[str]] = None,
                         limit: Optional[int] = None) -> List[EventEntry]:
        """이벤트 목록 조회"""
        
        events = []
        
        with self._lock:
            for event in self.events_cache.values():
                # 필터링
                if status and event.status != status:
                    continue
                
                if event_type and event.event_type != event_type:
                    continue
                
                if priority and event.priority != priority:
                    continue
                
                if tags and not any(tag in event.tags for tag in tags):
                    continue
                
                events.append(event)
        
        # 만료 시간순 정렬
        events.sort(key=lambda x: x.expires_at)
        
        # 제한 적용
        if limit:
            events = events[:limit]
        
        return events
    
    async def process_expired_events(self) -> int:
        """만료된 이벤트 처리"""
        processed_count = 0
        now = datetime.now()
        
        with self._lock:
            # 만료된 이벤트 찾기
            expired_events = []
            remaining_events = []
            
            for event in self.expiry_queue:
                if event.expires_at <= now:
                    expired_events.append(event)
                else:
                    remaining_events.append(event)
            
            # 만료 큐 업데이트
            self.expiry_queue = remaining_events
        
        # 만료된 이벤트 처리
        for event in expired_events:
            try:
                await self._process_expired_event(event)
                processed_count += 1
            except Exception as e:
                self.logger.error(f"Failed to process expired event {event.event_id}: {e}")
        
        if processed_count > 0:
            self.logger.info(f"Processed {processed_count} expired events")
        
        return processed_count
    
    async def _process_expired_event(self, event: EventEntry):
        """만료된 이벤트 처리"""
        self.logger.debug(f"Processing expired event {event.event_id}")
        
        # 상태 업데이트
        event.status = EventStatus.EXPIRED
        event.processed_at = datetime.now()
        
        # 액션 실행
        for action in event.actions:
            try:
                await self._execute_ttl_action(event, action)
            except Exception as e:
                self.logger.error(f"Failed to execute action {action} for event {event.event_id}: {e}")
        
        # 콜백 실행
        if event.callback_url and self.config.enable_callbacks:
            try:
                await self._execute_callback(event)
            except Exception as e:
                self.logger.error(f"Failed to execute callback for event {event.event_id}: {e}")
        
        # 데이터베이스 업데이트
        await self._save_event_to_database(event)
    
    async def _execute_ttl_action(self, event: EventEntry, action: TTLAction):
        """TTL 액션 실행"""
        if action == TTLAction.DELETE:
            await self.delete_event(event.event_id)
        
        elif action == TTLAction.ARCHIVE:
            event.status = EventStatus.ARCHIVED
            # 아카이브 로직 (필요시 구현)
        
        elif action == TTLAction.CALLBACK:
            if event.callback_url:
                await self._execute_callback(event)
        
        elif action == TTLAction.NOTIFY:
            # 알림 로직 (필요시 구현)
            self.logger.info(f"Notification for expired event {event.event_id}")
        
        elif action == TTLAction.EXTEND:
            # 연장 로직 (조건부)
            if event.access_count > 0:  # 접근된 이벤트만 연장
                event.extend_ttl(self.config.default_ttl_seconds)
                event.status = EventStatus.ACTIVE
                
                # 만료 큐에 다시 추가
                with self._lock:
                    import bisect
                    bisect.insort(self.expiry_queue, event, key=lambda x: x.expires_at)
    
    async def _execute_callback(self, event: EventEntry):
        """콜백 실행"""
        if not event.callback_url:
            return
        
        try:
            import aiohttp
            
            callback_payload = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'status': event.status.value,
                'expired_at': datetime.now().isoformat(),
                'data': event.data,
                'callback_data': event.callback_data
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config.callback_timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(event.callback_url, json=callback_payload) as response:
                    if response.status >= 400:
                        self.logger.warning(f"Callback failed for event {event.event_id}: {response.status}")
                    else:
                        self.logger.debug(f"Callback executed for event {event.event_id}")
        
        except Exception as e:
            self.logger.error(f"Callback execution failed for event {event.event_id}: {e}")
    
    def register_callback_handler(self, event_type: str, handler: Callable):
        """콜백 핸들러 등록"""
        self.callback_handlers[event_type] = handler
    
    async def start_cleanup_scheduler(self):
        """정리 스케줄러 시작"""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Cleanup scheduler started")
    
    async def stop_cleanup_scheduler(self):
        """정리 스케줄러 중지"""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Cleanup scheduler stopped")
    
    async def _cleanup_loop(self):
        """정리 루프"""
        while self._is_running:
            try:
                # 만료된 이벤트 처리
                await self.process_expired_events()
                
                # 메트릭 업데이트
                await self._update_metrics()
                
                # 오래된 처리 완료 이벤트 정리
                await self._cleanup_old_events()
                
                # 대기
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # 1분 대기 후 재시도
    
    async def _cleanup_old_events(self):
        """오래된 이벤트 정리"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.expired_retention_hours)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                DELETE FROM events 
                WHERE status IN ('processed', 'archived') 
                AND processed_at < ?
                """, (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old events")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old events: {e}")
    
    async def _update_metrics(self):
        """메트릭 업데이트"""
        try:
            with self._lock:
                # 기본 카운트
                self.metrics.total_events = len(self.events_cache)
                
                status_counts = defaultdict(int)
                priority_counts = defaultdict(int)
                ttl_distribution = defaultdict(int)
                
                for event in self.events_cache.values():
                    status_counts[event.status.value] += 1
                    priority_counts[event.priority.value] += 1
                    
                    # TTL 분포 (시간 단위)
                    ttl_hours = event.ttl_seconds // 3600
                    if ttl_hours < 1:
                        ttl_distribution["< 1h"] += 1
                    elif ttl_hours < 24:
                        ttl_distribution["1-24h"] += 1
                    elif ttl_hours < 168:  # 7일
                        ttl_distribution["1-7d"] += 1
                    else:
                        ttl_distribution["> 7d"] += 1
                
                self.metrics.active_events = status_counts.get("active", 0)
                self.metrics.expired_events = status_counts.get("expired", 0)
                self.metrics.processed_events = status_counts.get("processed", 0)
                self.metrics.archived_events = status_counts.get("archived", 0)
                self.metrics.cancelled_events = status_counts.get("cancelled", 0)
                
                self.metrics.priority_distribution = dict(priority_counts)
                self.metrics.ttl_distribution = dict(ttl_distribution)
                
                # 평균 TTL 계산
                if self.events_cache:
                    total_ttl = sum(event.ttl_seconds for event in self.events_cache.values())
                    self.metrics.average_ttl_seconds = total_ttl / len(self.events_cache)
            
            self._last_metrics_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def get_metrics(self) -> TTLMetrics:
        """메트릭 조회"""
        return self.metrics
    
    async def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        await self._update_metrics()
        
        return {
            "is_running": self._is_running,
            "total_events": self.metrics.total_events,
            "active_events": self.metrics.active_events,
            "expired_events": self.metrics.expired_events,
            "expiry_queue_size": len(self.expiry_queue),
            "next_expiry": self.expiry_queue[0].expires_at.isoformat() if self.expiry_queue else None,
            "metrics": self.metrics,
            "config": {
                "cleanup_interval_minutes": self.config.cleanup_interval_minutes,
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "max_ttl_seconds": self.config.max_ttl_seconds,
                "enable_callbacks": self.config.enable_callbacks
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            await self.stop_cleanup_scheduler()
            self.logger.info("TTL Event Manager cleanup completed")
        
        except Exception as e:
            self.logger.error(f"TTL Event Manager cleanup failed: {e}")

# 전역 매니저
_global_ttl_manager = None

def get_ttl_event_manager(config: Optional[TTLConfig] = None) -> TTLEventManager:
    """전역 TTL 이벤트 매니저 반환"""
    global _global_ttl_manager
    if _global_ttl_manager is None:
        _global_ttl_manager = TTLEventManager(config)
    return _global_ttl_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_ttl_manager():
        print("🧪 TTL Event Manager 테스트")
        
        # 매니저 초기화
        manager = TTLEventManager()
        
        print("\n1️⃣ 이벤트 생성")
        event_id = await manager.create_event(
            event_type="test_event",
            ttl_seconds=60,  # 1분
            priority=EventPriority.HIGH,
            data={"test": "data"},
            tags=["test", "example"],
            actions=[TTLAction.NOTIFY, TTLAction.ARCHIVE]
        )
        print(f"  생성된 이벤트 ID: {event_id}")
        
        print("\n2️⃣ 이벤트 조회")
        event = await manager.get_event(event_id)
        if event:
            print(f"  이벤트 타입: {event.event_type}")
            print(f"  만료 시간: {event.expires_at}")
            print(f"  남은 시간: {event.time_to_expiry()}")
        
        print("\n3️⃣ 상태 조회")
        status = await manager.get_status()
        print(f"  총 이벤트: {status['total_events']}")
        print(f"  활성 이벤트: {status['active_events']}")
        
        print("\n4️⃣ 정리 스케줄러 시작")
        await manager.start_cleanup_scheduler()
        
        print("\n5️⃣ 짧은 대기 후 정리")
        await asyncio.sleep(2)
        await manager.stop_cleanup_scheduler()
        
        print("\n🎉 TTL Event Manager 테스트 완료!")
        
        # 정리
        await manager.cleanup()
    
    # 테스트 실행
    asyncio.run(test_ttl_manager())