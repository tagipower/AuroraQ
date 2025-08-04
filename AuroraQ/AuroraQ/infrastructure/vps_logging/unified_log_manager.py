#!/usr/bin/env python3
"""
AuroraQ 통합 로그 관리 시스템 v1.0
4가지 로그 범주를 통합 관리하는 방어적 보안 분석 시스템

범주별 처리:
- Raw Logs: 디버깅/추적용 (.jsonl, 3-7일)
- Summary Logs: 분석/리포트용 (.csv, 수개월)  
- Training Logs: 학습/검증용 (.pkl/.npz, 장기보존)
- Tagged Logs: 고의미 이벤트 (.jsonl, 조건부 영구)
"""

import os
import json
import pickle
import gzip
import csv
import sqlite3
import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import structlog
from collections import defaultdict, deque
import psutil

class LogCategory(Enum):
    """로그 범주 정의"""
    RAW = "raw"           # 디버깅/추적용
    SUMMARY = "summary"   # 분석/리포트용
    TRAINING = "training" # 학습/검증용
    TAGGED = "tagged"     # 고의미 이벤트

class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class RetentionPolicy(Enum):
    """보존 정책"""
    SHORT = (3, 7)        # 3-7일
    MEDIUM = (30, 90)     # 1-3개월
    LONG = (365, 1825)    # 1-5년
    PERMANENT = (0, 0)    # 영구보존

@dataclass
class LogEntry:
    """통합 로그 엔트리"""
    timestamp: datetime
    category: LogCategory
    level: LogLevel
    component: str
    event_type: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class StorageConfig:
    """저장소 설정"""
    base_dir: Path
    raw_dir: Path
    summary_dir: Path
    training_dir: Path
    tagged_dir: Path
    archive_dir: Path
    max_file_size_mb: int = 100
    compression_enabled: bool = True
    backup_enabled: bool = True

class UnifiedLogManager:
    """통합 로그 관리자"""
    
    def __init__(self, 
                 base_dir: str = "/app/logs",
                 vps_optimized: bool = True,
                 max_memory_mb: int = 512):
        """
        통합 로그 관리자 초기화
        
        Args:
            base_dir: 로그 기본 디렉토리
            vps_optimized: VPS 최적화 모드
            max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self.base_dir = Path(base_dir)
        self.vps_optimized = vps_optimized
        self.max_memory_mb = max_memory_mb
        
        # 디렉토리 구조 생성
        self.storage_config = self._setup_directories()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 내부 상태
        self._buffer = defaultdict(deque)
        self._buffer_lock = threading.Lock()
        self._stats = defaultdict(int)
        self._last_cleanup = datetime.now()
        
        # 비동기 작업자
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._running = False
        
        # 보존 정책 매핑
        self.retention_policies = {
            LogCategory.RAW: RetentionPolicy.SHORT,
            LogCategory.SUMMARY: RetentionPolicy.MEDIUM,
            LogCategory.TRAINING: RetentionPolicy.LONG,
            LogCategory.TAGGED: RetentionPolicy.PERMANENT
        }
        
        # VPS 최적화 설정
        if vps_optimized:
            self._apply_vps_optimizations()
    
    def _setup_directories(self) -> StorageConfig:
        """디렉토리 구조 설정"""
        base = self.base_dir
        
        # 주요 디렉토리
        dirs = {
            'raw': base / "raw",
            'summary': base / "summary", 
            'training': base / "training",
            'tagged': base / "tagged",
            'archive': base / "archive"
        }
        
        # 디렉토리 생성
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 서브디렉토리 생성 (날짜별)
        today = datetime.now().strftime("%Y%m%d")
        for category_dir in dirs.values():
            (category_dir / today).mkdir(exist_ok=True)
        
        return StorageConfig(
            base_dir=base,
            raw_dir=dirs['raw'],
            summary_dir=dirs['summary'],
            training_dir=dirs['training'],
            tagged_dir=dirs['tagged'],
            archive_dir=dirs['archive']
        )
    
    def _setup_logger(self) -> structlog.stdlib.BoundLogger:
        """구조화된 로거 설정"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger("UnifiedLogManager")
    
    def _apply_vps_optimizations(self):
        """VPS 최적화 적용"""
        # 메모리 제한
        self._buffer_size_limit = min(1000, self.max_memory_mb // 4)
        
        # 압축 활성화
        self.storage_config.compression_enabled = True
        
        # 배치 처리 크기 조정
        self._batch_size = 50
        
        # 자동 정리 간격 (30분)
        self._cleanup_interval = 1800
    
    async def log(self, 
                  category: LogCategory,
                  level: LogLevel,
                  component: str,
                  event_type: str,
                  message: str,
                  **kwargs) -> bool:
        """
        통합 로그 기록
        
        Args:
            category: 로그 범주
            level: 로그 레벨
            component: 컴포넌트명
            event_type: 이벤트 타입
            message: 메시지
            **kwargs: 추가 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 로그 엔트리 생성
            entry = LogEntry(
                timestamp=datetime.now(),
                category=category,
                level=level,
                component=component,
                event_type=event_type,
                message=message,
                metadata=kwargs.get('metadata', {}),
                tags=kwargs.get('tags', []),
                session_id=kwargs.get('session_id'),
                user_id=kwargs.get('user_id'),
                correlation_id=kwargs.get('correlation_id')
            )
            
            # 버퍼에 추가
            with self._buffer_lock:
                self._buffer[category].append(entry)
                self._stats[f"{category.value}_count"] += 1
            
            # 즉시 처리가 필요한 경우
            if level in [LogLevel.ERROR, LogLevel.CRITICAL] or category == LogCategory.TAGGED:
                await self._flush_category(category)
            
            # 버퍼 크기 체크
            if len(self._buffer[category]) >= self._batch_size:
                await self._flush_category(category)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to log entry", error=str(e), category=category.value)
            return False
    
    async def _flush_category(self, category: LogCategory):
        """범주별 버퍼 플러시"""
        try:
            with self._buffer_lock:
                entries = list(self._buffer[category])
                self._buffer[category].clear()
            
            if not entries:
                return
            
            # 범주별 처리
            if category == LogCategory.RAW:
                await self._write_raw_logs(entries)
            elif category == LogCategory.SUMMARY:
                await self._write_summary_logs(entries)
            elif category == LogCategory.TRAINING:
                await self._write_training_logs(entries)
            elif category == LogCategory.TAGGED:
                await self._write_tagged_logs(entries)
            
            self._stats[f"{category.value}_flushed"] += len(entries)
            
        except Exception as e:
            self.logger.error("Failed to flush category", category=category.value, error=str(e))
    
    async def _write_raw_logs(self, entries: List[LogEntry]):
        """Raw 로그 작성 (.jsonl 형식)"""
        date_str = datetime.now().strftime("%Y%m%d")
        hour_str = datetime.now().strftime("%H")
        
        file_path = (self.storage_config.raw_dir / date_str / 
                    f"raw_logs_{hour_str}.jsonl")
        file_path.parent.mkdir(exist_ok=True)
        
        # JSONL 형식으로 작성
        with open(file_path, 'a', encoding='utf-8') as f:
            for entry in entries:
                log_line = {
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level.value,
                    'component': entry.component,
                    'event_type': entry.event_type,
                    'message': entry.message,
                    'metadata': entry.metadata,
                    'tags': entry.tags,
                    'session_id': entry.session_id,
                    'correlation_id': entry.correlation_id
                }
                f.write(json.dumps(log_line, ensure_ascii=False) + '\n')
    
    async def _write_summary_logs(self, entries: List[LogEntry]):
        """Summary 로그 작성 (.csv 형식)"""
        date_str = datetime.now().strftime("%Y%m%d")
        
        file_path = (self.storage_config.summary_dir / date_str / 
                    f"summary_{date_str}.csv")
        file_path.parent.mkdir(exist_ok=True)
        
        # CSV 형식으로 작성
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'level', 'component', 'event_type', 
                         'message', 'tags', 'session_id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for entry in entries:
                writer.writerow({
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level.value,
                    'component': entry.component,
                    'event_type': entry.event_type,
                    'message': entry.message[:500],  # 메시지 길이 제한
                    'tags': ','.join(entry.tags),
                    'session_id': entry.session_id or ''
                })
    
    async def _write_training_logs(self, entries: List[LogEntry]):
        """Training 로그 작성 (.pkl/.npz 형식)"""
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 학습 데이터 구조화
        training_data = {
            'timestamps': [],
            'features': [],
            'labels': [],
            'metadata': []
        }
        
        for entry in entries:
            training_data['timestamps'].append(entry.timestamp.timestamp())
            
            # 특성 벡터 생성 (간단한 예시)
            features = [
                len(entry.message),
                len(entry.tags),
                1 if entry.level == LogLevel.ERROR else 0,
                hash(entry.component) % 1000
            ]
            training_data['features'].append(features)
            
            # 레이블 (로그 레벨을 숫자로 변환)
            level_map = {LogLevel.DEBUG: 0, LogLevel.INFO: 1, 
                        LogLevel.WARNING: 2, LogLevel.ERROR: 3, LogLevel.CRITICAL: 4}
            training_data['labels'].append(level_map[entry.level])
            
            training_data['metadata'].append({
                'component': entry.component,
                'event_type': entry.event_type,
                'tags': entry.tags
            })
        
        # NumPy 배열로 변환
        features_array = np.array(training_data['features'], dtype=np.float32)
        labels_array = np.array(training_data['labels'], dtype=np.int32)
        timestamps_array = np.array(training_data['timestamps'], dtype=np.float64)
        
        # .npz 형식으로 저장
        file_path = (self.storage_config.training_dir / date_str / 
                    f"training_{date_str}_{datetime.now().strftime('%H%M')}.npz")
        file_path.parent.mkdir(exist_ok=True)
        
        np.savez_compressed(
            file_path,
            features=features_array,
            labels=labels_array,
            timestamps=timestamps_array
        )
        
        # 메타데이터는 별도 pickle 파일로 저장
        metadata_path = file_path.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(training_data['metadata'], f)
    
    async def _write_tagged_logs(self, entries: List[LogEntry]):
        """Tagged 로그 작성 (.jsonl 형식, 영구보존)"""
        date_str = datetime.now().strftime("%Y%m%d")
        
        file_path = (self.storage_config.tagged_dir / date_str / 
                    f"tagged_events_{date_str}.jsonl")
        file_path.parent.mkdir(exist_ok=True)
        
        # 고의미 이벤트를 상세히 기록
        with open(file_path, 'a', encoding='utf-8') as f:
            for entry in entries:
                tagged_entry = {
                    'id': f"{entry.timestamp.isoformat()}_{hash(entry.message) % 10000}",
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level.value,
                    'component': entry.component,
                    'event_type': entry.event_type,
                    'message': entry.message,
                    'metadata': entry.metadata,
                    'tags': entry.tags,
                    'session_id': entry.session_id,
                    'user_id': entry.user_id,
                    'correlation_id': entry.correlation_id,
                    'system_info': {
                        'memory_usage': psutil.virtual_memory().percent,
                        'cpu_usage': psutil.cpu_percent(),
                        'disk_usage': psutil.disk_usage('/').percent
                    }
                }
                f.write(json.dumps(tagged_entry, ensure_ascii=False) + '\n')
    
    async def start_background_tasks(self):
        """백그라운드 작업 시작"""
        self._running = True
        
        # 정기적인 버퍼 플러시
        asyncio.create_task(self._periodic_flush())
        
        # 정기적인 정리 작업
        asyncio.create_task(self._periodic_cleanup())
        
        self.logger.info("Background tasks started")
    
    async def _periodic_flush(self):
        """정기적 버퍼 플러시"""
        while self._running:
            try:
                for category in LogCategory:
                    if self._buffer[category]:
                        await self._flush_category(category)
                
                await asyncio.sleep(60)  # 1분마다
                
            except Exception as e:
                self.logger.error("Periodic flush error", error=str(e))
    
    async def _periodic_cleanup(self):
        """정기적 정리 작업"""
        while self._running:
            try:
                await self._cleanup_old_logs()
                await self._compress_old_files()
                
                self._last_cleanup = datetime.now()
                
                # VPS 최적화: 30분마다
                interval = self._cleanup_interval if self.vps_optimized else 3600
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error("Periodic cleanup error", error=str(e))
    
    async def _cleanup_old_logs(self):
        """오래된 로그 정리"""
        now = datetime.now()
        
        for category, policy in self.retention_policies.items():
            if policy == RetentionPolicy.PERMANENT:
                continue
            
            min_days, max_days = policy.value
            cutoff_date = now - timedelta(days=max_days)
            
            category_dir = getattr(self.storage_config, f"{category.value}_dir")
            
            # 오래된 파일 삭제
            for date_dir in category_dir.iterdir():
                if date_dir.is_dir():
                    try:
                        dir_date = datetime.strptime(date_dir.name, "%Y%m%d")
                        if dir_date < cutoff_date:
                            shutil.rmtree(date_dir)
                            self.logger.info("Cleaned up old logs", 
                                           category=category.value, 
                                           date=date_dir.name)
                    except ValueError:
                        continue  # 날짜 형식이 아닌 디렉토리는 건너뛰기
    
    async def _compress_old_files(self):
        """오래된 파일 압축"""
        if not self.storage_config.compression_enabled:
            return
        
        cutoff_date = datetime.now() - timedelta(days=1)
        
        for category_dir in [self.storage_config.raw_dir, 
                           self.storage_config.summary_dir,
                           self.storage_config.tagged_dir]:
            
            for date_dir in category_dir.iterdir():
                if date_dir.is_dir():
                    try:
                        dir_date = datetime.strptime(date_dir.name, "%Y%m%d")
                        if dir_date < cutoff_date:
                            await self._compress_directory(date_dir)
                    except ValueError:
                        continue
    
    async def _compress_directory(self, directory: Path):
        """디렉토리 압축"""
        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.gz'):
                    compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                    
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    file_path.unlink()  # 원본 파일 삭제
                    
                    self.logger.debug("Compressed file", 
                                    original=str(file_path), 
                                    compressed=str(compressed_path))
        
        except Exception as e:
            self.logger.error("Compression failed", directory=str(directory), error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'buffer_sizes': {cat.value: len(self._buffer[cat]) for cat in LogCategory},
            'stats': dict(self._stats),
            'last_cleanup': self._last_cleanup.isoformat(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'storage_paths': {
                'raw': str(self.storage_config.raw_dir),
                'summary': str(self.storage_config.summary_dir),
                'training': str(self.storage_config.training_dir),
                'tagged': str(self.storage_config.tagged_dir)
            }
        }
    
    async def shutdown(self):
        """안전한 종료"""
        self._running = False
        
        # 모든 버퍼 플러시
        for category in LogCategory:
            await self._flush_category(category)
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        self.logger.info("UnifiedLogManager shutdown complete")

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_log_manager(base_dir: str = "/app/logs") -> UnifiedLogManager:
    """VPS 최적화된 로그 관리자 생성"""
    return UnifiedLogManager(
        base_dir=base_dir,
        vps_optimized=True,
        max_memory_mb=512
    )

# 기존 로깅 시스템과의 호환성을 위한 어댑터
class LoggingAdapter:
    """기존 로깅 시스템 어댑터"""
    
    def __init__(self, log_manager: UnifiedLogManager, component: str):
        self.log_manager = log_manager
        self.component = component
    
    def info(self, message: str, **kwargs):
        asyncio.create_task(
            self.log_manager.log(LogCategory.RAW, LogLevel.INFO, 
                               self.component, "info", message, **kwargs)
        )
    
    def error(self, message: str, **kwargs):
        asyncio.create_task(
            self.log_manager.log(LogCategory.RAW, LogLevel.ERROR, 
                               self.component, "error", message, **kwargs)
        )
    
    def warning(self, message: str, **kwargs):
        asyncio.create_task(
            self.log_manager.log(LogCategory.RAW, LogLevel.WARNING, 
                               self.component, "warning", message, **kwargs)
        )
    
    def debug(self, message: str, **kwargs):
        asyncio.create_task(
            self.log_manager.log(LogCategory.RAW, LogLevel.DEBUG, 
                               self.component, "debug", message, **kwargs)
        )

# UnifiedLogManager 클래스에 get_logger 메서드 추가 (호환성 위해)
def _add_get_logger_method():
    """UnifiedLogManager에 get_logger 메서드 동적 추가"""
    def get_logger(self, component_name: str) -> LoggingAdapter:
        """컴포넌트별 로거 생성 (VPS 호환성)"""
        return LoggingAdapter(self, component_name)
    
    # 메서드를 클래스에 동적으로 추가
    UnifiedLogManager.get_logger = get_logger

# 모듈 로드 시 메서드 추가
_add_get_logger_method()

if __name__ == "__main__":
    # 테스트 실행
    async def test_log_manager():
        manager = create_vps_log_manager()
        await manager.start_background_tasks()
        
        # 테스트 로그들
        await manager.log(LogCategory.RAW, LogLevel.INFO, "test", "startup", "System started")
        await manager.log(LogCategory.TAGGED, LogLevel.CRITICAL, "security", "auth_failure", 
                         "Authentication failed", metadata={"user": "test", "ip": "127.0.0.1"})
        
        print("Stats:", manager.get_stats())
        
        await asyncio.sleep(2)
        await manager.shutdown()
    
    asyncio.run(test_log_manager())