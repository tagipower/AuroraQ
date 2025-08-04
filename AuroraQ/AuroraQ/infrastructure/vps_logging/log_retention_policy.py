#!/usr/bin/env python3
"""
로그 보존 정책 및 자동 정리 시스템
4가지 범주별 차별화된 보존 전략 구현
"""

import os
import gzip
import shutil
import sqlite3
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import csv
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

from .unified_log_manager import LogCategory, RetentionPolicy

class ArchiveFormat(Enum):
    """아카이브 형식"""
    GZIP = "gzip"
    TAR_GZ = "tar.gz"
    ZIP = "zip"

class CompressionLevel(Enum):
    """압축 레벨"""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9

@dataclass
class RetentionRule:
    """보존 규칙"""
    category: LogCategory
    active_days: int          # 활성 보관 기간
    archive_days: int         # 아카이브 보관 기간  
    compression_after: int    # 압축 시작 일수
    delete_after: Optional[int] = None  # 삭제 일수 (None = 영구보존)
    archive_format: ArchiveFormat = ArchiveFormat.GZIP
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    metadata_retention: bool = True    # 메타데이터 보존 여부

@dataclass
class StorageStats:
    """저장소 사용량 통계"""
    total_size_mb: float
    active_size_mb: float
    archived_size_mb: float
    file_count: int
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None

class LogRetentionManager:
    """로그 보존 정책 관리자"""
    
    def __init__(self, 
                 base_log_dir: str = "/app/logs",
                 vps_optimized: bool = True):
        """
        로그 보존 정책 관리자 초기화
        
        Args:
            base_log_dir: 로그 기본 디렉토리
            vps_optimized: VPS 최적화 모드
        """
        self.base_log_dir = Path(base_log_dir)
        self.vps_optimized = vps_optimized
        
        # VPS 최적화 설정
        if vps_optimized:
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.max_concurrent_operations = 2
        else:
            self.executor = ThreadPoolExecutor(max_workers=3)
            self.max_concurrent_operations = 5
        
        # 기본 보존 규칙 설정
        self.retention_rules = self._setup_default_rules()
        
        # 메타데이터 DB 설정
        self.metadata_db_path = self.base_log_dir / "retention_metadata.db"
        self._setup_metadata_db()
        
        # 통계
        self.cleanup_stats = {
            "last_cleanup": None,
            "files_processed": 0,
            "bytes_saved": 0,
            "errors": 0
        }
        
        self.logger = logging.getLogger("LogRetentionManager")
    
    def _setup_default_rules(self) -> Dict[LogCategory, RetentionRule]:
        """기본 보존 규칙 설정"""
        if self.vps_optimized:
            # VPS 최적화: 저장공간 절약 우선
            return {
                LogCategory.RAW: RetentionRule(
                    category=LogCategory.RAW,
                    active_days=3,
                    archive_days=7, 
                    compression_after=1,
                    delete_after=7,
                    compression_level=CompressionLevel.HIGH
                ),
                LogCategory.SUMMARY: RetentionRule(
                    category=LogCategory.SUMMARY,
                    active_days=30,
                    archive_days=90,
                    compression_after=7,
                    delete_after=90,
                    compression_level=CompressionLevel.MEDIUM
                ),
                LogCategory.TRAINING: RetentionRule(
                    category=LogCategory.TRAINING,
                    active_days=90,
                    archive_days=365,
                    compression_after=30,
                    delete_after=None,  # 영구보존
                    compression_level=CompressionLevel.LOW,
                    archive_format=ArchiveFormat.TAR_GZ
                ),
                LogCategory.TAGGED: RetentionRule(
                    category=LogCategory.TAGGED,
                    active_days=365,
                    archive_days=1825,  # 5년
                    compression_after=90,
                    delete_after=None,  # 영구보존
                    compression_level=CompressionLevel.LOW,
                    metadata_retention=True
                )
            }
        else:
            # 표준 설정: 보존 기간 우선
            return {
                LogCategory.RAW: RetentionRule(
                    category=LogCategory.RAW,
                    active_days=7,
                    archive_days=14,
                    compression_after=2,
                    delete_after=14
                ),
                LogCategory.SUMMARY: RetentionRule(
                    category=LogCategory.SUMMARY,
                    active_days=90,
                    archive_days=180,
                    compression_after=14,
                    delete_after=180
                ),
                LogCategory.TRAINING: RetentionRule(
                    category=LogCategory.TRAINING,
                    active_days=180,
                    archive_days=730,
                    compression_after=60,
                    delete_after=None
                ),
                LogCategory.TAGGED: RetentionRule(
                    category=LogCategory.TAGGED,
                    active_days=730,
                    archive_days=2555,  # 7년
                    compression_after=180,
                    delete_after=None
                )
            }
    
    def _setup_metadata_db(self):
        """메타데이터 데이터베이스 설정"""
        self.metadata_db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    created_date TEXT,
                    archived_date TEXT,
                    compression_ratio REAL,
                    checksum TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cleanup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cleanup_date TEXT NOT NULL,
                    category TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    files_processed INTEGER,
                    bytes_processed INTEGER,
                    errors INTEGER,
                    duration_seconds REAL
                )
            ''')
            
            conn.commit()
    
    async def run_retention_policy(self) -> Dict[str, Any]:
        """보존 정책 실행"""
        start_time = datetime.now()
        total_stats = {
            "start_time": start_time.isoformat(),
            "categories_processed": 0,
            "files_compressed": 0,
            "files_deleted": 0,
            "bytes_saved": 0,
            "errors": 0
        }
        
        try:
            # 각 범주별로 보존 정책 적용
            for category, rule in self.retention_rules.items():
                try:
                    category_stats = await self._process_category(category, rule)
                    
                    # 통계 업데이트
                    total_stats["categories_processed"] += 1
                    total_stats["files_compressed"] += category_stats.get("files_compressed", 0)
                    total_stats["files_deleted"] += category_stats.get("files_deleted", 0)
                    total_stats["bytes_saved"] += category_stats.get("bytes_saved", 0)
                    
                except Exception as e:
                    total_stats["errors"] += 1
                    self.logger.error(f"Failed to process category {category.value}: {e}")
            
            # 정리 기록
            duration = (datetime.now() - start_time).total_seconds()
            total_stats["duration_seconds"] = duration
            total_stats["end_time"] = datetime.now().isoformat()
            
            await self._record_cleanup_history(total_stats)
            
            return total_stats
            
        except Exception as e:
            self.logger.error(f"Retention policy execution failed: {e}")
            total_stats["errors"] += 1
            return total_stats
    
    async def _process_category(self, 
                               category: LogCategory, 
                               rule: RetentionRule) -> Dict[str, Any]:
        """범주별 보존 정책 처리"""
        category_dir = self.base_log_dir / category.value
        if not category_dir.exists():
            return {}
        
        stats = {
            "files_compressed": 0,
            "files_deleted": 0,
            "bytes_saved": 0
        }
        
        now = datetime.now()
        
        # 1. 압축 대상 파일 처리
        compress_cutoff = now - timedelta(days=rule.compression_after)
        compression_tasks = []
        
        for file_path in self._find_files_by_date(category_dir, compress_cutoff, older=True):
            if not file_path.name.endswith('.gz') and not file_path.name.endswith('.tar.gz'):
                compression_tasks.append(self._compress_file(file_path, rule))
        
        # 압축 작업 실행 (병렬)
        if compression_tasks:
            compression_results = await asyncio.gather(*compression_tasks, return_exceptions=True)
            for result in compression_results:
                if isinstance(result, dict):
                    stats["files_compressed"] += 1
                    stats["bytes_saved"] += result.get("bytes_saved", 0)
        
        # 2. 아카이브 대상 처리
        archive_cutoff = now - timedelta(days=rule.archive_days)
        archive_dir = self.base_log_dir / "archive" / category.value
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in self._find_files_by_date(category_dir, archive_cutoff, older=True):
            if file_path.is_file():
                await self._archive_file(file_path, archive_dir, rule)
        
        # 3. 삭제 대상 처리 (영구보존이 아닌 경우)
        if rule.delete_after is not None:
            delete_cutoff = now - timedelta(days=rule.delete_after)
            
            for file_path in self._find_files_by_date(category_dir, delete_cutoff, older=True):
                if file_path.is_file():
                    await self._delete_file_with_metadata(file_path, rule)
                    stats["files_deleted"] += 1
        
        return stats
    
    def _find_files_by_date(self, 
                           directory: Path, 
                           cutoff_date: datetime,
                           older: bool = True) -> List[Path]:
        """날짜 기준으로 파일 검색"""
        files = []
        
        if not directory.exists():
            return files
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    # 파일 수정 시간 확인
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if older and file_mtime < cutoff_date:
                        files.append(file_path)
                    elif not older and file_mtime >= cutoff_date:
                        files.append(file_path)
                        
                except (OSError, ValueError):
                    continue
        
        return files
    
    async def _compress_file(self, 
                           file_path: Path, 
                           rule: RetentionRule) -> Dict[str, Any]:
        """파일 압축"""
        try:
            original_size = file_path.stat().st_size
            
            if rule.archive_format == ArchiveFormat.GZIP:
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                
                # 압축 실행
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self._gzip_compress, 
                    file_path, 
                    compressed_path, 
                    rule.compression_level.value
                )
                
            elif rule.archive_format == ArchiveFormat.TAR_GZ:
                compressed_path = file_path.with_suffix('.tar.gz')
                
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._tar_compress,
                    file_path,
                    compressed_path,
                    rule.compression_level.value
                )
            
            # 압축 성공시 원본 파일 삭제
            compressed_size = compressed_path.stat().st_size
            bytes_saved = original_size - compressed_size
            
            # 메타데이터 저장
            await self._save_file_metadata(
                compressed_path, 
                rule.category,
                original_size,
                compressed_size,
                datetime.now()
            )
            
            file_path.unlink()  # 원본 삭제
            
            return {
                "success": True,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "bytes_saved": bytes_saved,
                "compression_ratio": compressed_size / original_size if original_size > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Compression failed for {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def _gzip_compress(self, source: Path, target: Path, level: int):
        """GZIP 압축 (동기 실행)"""
        with open(source, 'rb') as f_in:
            with gzip.open(target, 'wb', compresslevel=level) as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _tar_compress(self, source: Path, target: Path, level: int):
        """TAR.GZ 압축 (동기 실행)"""
        import tarfile
        
        with tarfile.open(target, 'w:gz', compresslevel=level) as tar:
            tar.add(source, arcname=source.name)
    
    async def _archive_file(self, 
                          file_path: Path, 
                          archive_dir: Path,
                          rule: RetentionRule):
        """파일 아카이브"""
        try:
            # 날짜별 아카이브 디렉토리
            file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
            date_dir = archive_dir / file_date.strftime("%Y%m%d")
            date_dir.mkdir(exist_ok=True)
            
            # 아카이브 파일 이동
            archive_path = date_dir / file_path.name
            shutil.move(str(file_path), str(archive_path))
            
            # 메타데이터 업데이트
            await self._save_file_metadata(
                archive_path,
                rule.category,
                archive_path.stat().st_size,
                archive_path.stat().st_size,
                datetime.now(),
                archived=True
            )
            
        except Exception as e:
            self.logger.error(f"Archive failed for {file_path}: {e}")
    
    async def _delete_file_with_metadata(self, 
                                       file_path: Path,
                                       rule: RetentionRule):
        """메타데이터와 함께 파일 삭제"""
        try:
            # 메타데이터 보존이 필요한 경우
            if rule.metadata_retention:
                await self._backup_file_metadata(file_path)
            
            file_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Delete failed for {file_path}: {e}")
    
    async def _save_file_metadata(self,
                                file_path: Path,
                                category: LogCategory,
                                original_size: int,
                                compressed_size: int,
                                processed_date: datetime,
                                archived: bool = False):
        """파일 메타데이터 저장"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO file_metadata 
                    (file_path, category, original_size, compressed_size, 
                     created_date, archived_date, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(file_path),
                    category.value,
                    original_size,
                    compressed_size,
                    processed_date.isoformat(),
                    processed_date.isoformat() if archived else None,
                    compressed_size / original_size if original_size > 0 else 0
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata for {file_path}: {e}")
    
    async def _backup_file_metadata(self, file_path: Path):
        """파일 메타데이터 백업"""
        try:
            # 파일 기본 정보
            stat = file_path.stat()
            metadata = {
                "path": str(file_path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "deleted": datetime.now().isoformat()
            }
            
            # 메타데이터 파일로 저장
            metadata_dir = self.base_log_dir / "metadata" / "deleted"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = metadata_dir / f"{file_path.stem}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to backup metadata for {file_path}: {e}")
    
    async def _record_cleanup_history(self, stats: Dict[str, Any]):
        """정리 기록 저장"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute('''
                    INSERT INTO cleanup_history 
                    (cleanup_date, category, operation, files_processed, 
                     bytes_processed, errors, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats["start_time"],
                    "all",
                    "retention_policy",
                    stats["files_compressed"] + stats["files_deleted"],
                    stats["bytes_saved"],
                    stats["errors"],
                    stats.get("duration_seconds", 0)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to record cleanup history: {e}")
    
    async def get_storage_stats(self) -> Dict[LogCategory, StorageStats]:
        """저장소 사용량 통계"""
        stats = {}
        
        for category in LogCategory:
            category_dir = self.base_log_dir / category.value
            
            if not category_dir.exists():
                stats[category] = StorageStats(0, 0, 0, 0)
                continue
            
            total_size = 0
            active_size = 0
            archived_size = 0
            file_count = 0
            oldest_file = None
            newest_file = None
            
            # 활성 파일들
            for file_path in category_dir.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    total_size += file_size
                    active_size += file_size
                    file_count += 1
                    
                    if oldest_file is None or file_mtime < oldest_file:
                        oldest_file = file_mtime
                    if newest_file is None or file_mtime > newest_file:
                        newest_file = file_mtime
            
            # 아카이브 파일들
            archive_dir = self.base_log_dir / "archive" / category.value
            if archive_dir.exists():
                for file_path in archive_dir.rglob('*'):
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        archived_size += file_size
                        file_count += 1
            
            stats[category] = StorageStats(
                total_size_mb=total_size / (1024 * 1024),
                active_size_mb=active_size / (1024 * 1024),
                archived_size_mb=archived_size / (1024 * 1024),
                file_count=file_count,
                oldest_file=oldest_file,
                newest_file=newest_file
            )
        
        return stats
    
    async def cleanup_by_disk_usage(self, target_usage_percent: float = 80.0):
        """디스크 사용량 기준 정리"""
        try:
            # 현재 디스크 사용량 확인
            disk_usage = psutil.disk_usage(str(self.base_log_dir))
            current_usage = (disk_usage.used / disk_usage.total) * 100
            
            if current_usage < target_usage_percent:
                return {"message": "Disk usage within target", "current_usage": current_usage}
            
            # 긴급 정리 모드
            emergency_rules = {}
            for category, rule in self.retention_rules.items():
                # 보존 기간을 절반으로 줄임
                emergency_rule = RetentionRule(
                    category=rule.category,
                    active_days=max(1, rule.active_days // 2),
                    archive_days=max(3, rule.archive_days // 2),
                    compression_after=max(1, rule.compression_after // 2),
                    delete_after=rule.delete_after // 2 if rule.delete_after else 30,
                    compression_level=CompressionLevel.HIGH
                )
                emergency_rules[category] = emergency_rule
            
            # 임시로 긴급 규칙 적용
            original_rules = self.retention_rules
            self.retention_rules = emergency_rules
            
            try:
                cleanup_stats = await self.run_retention_policy()
                return {
                    "message": "Emergency cleanup completed",
                    "original_usage": current_usage,
                    "cleanup_stats": cleanup_stats
                }
            finally:
                # 원래 규칙 복원
                self.retention_rules = original_rules
                
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
            return {"error": str(e)}

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_retention_manager(base_log_dir: str = "/app/logs") -> LogRetentionManager:
    """VPS 최적화된 보존 정책 관리자 생성"""
    return LogRetentionManager(base_log_dir=base_log_dir, vps_optimized=True)

if __name__ == "__main__":
    # 테스트 실행
    async def test_retention_manager():
        manager = create_vps_retention_manager("/tmp/test_logs")
        
        # 보존 정책 실행
        stats = await manager.run_retention_policy()
        print("Retention policy stats:", stats)
        
        # 저장소 통계
        storage_stats = await manager.get_storage_stats()
        for category, stat in storage_stats.items():
            print(f"{category.value}: {stat.total_size_mb:.2f}MB, {stat.file_count} files")
    
    asyncio.run(test_retention_manager())