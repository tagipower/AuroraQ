#!/usr/bin/env python3
"""
백업 관리 시스템
P7-2: 백업 관리 시스템 구현
"""

import sys
import os
import shutil
import gzip
import zipfile
import tarfile
import sqlite3
import asyncio
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
from collections import defaultdict
import tempfile

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)

class BackupType(Enum):
    """백업 타입"""
    FULL = "full"          # 전체 백업
    INCREMENTAL = "incremental"  # 증분 백업
    DIFFERENTIAL = "differential"  # 차등 백업
    SNAPSHOT = "snapshot"   # 스냅샷

class CompressionType(Enum):
    """압축 타입"""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"

class BackupStatus(Enum):
    """백업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BackupConfig:
    """백업 설정"""
    # 기본 설정
    backup_dir: str = "backups"
    temp_dir: str = "temp_backup"
    
    # 백업 대상
    source_dirs: List[str] = field(default_factory=lambda: ["logs", "data", "config"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*.tmp", "*.cache", "__pycache__"])
    include_databases: List[str] = field(default_factory=list)
    
    # 백업 전략
    backup_type: BackupType = BackupType.INCREMENTAL
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    
    # 스케줄링
    auto_backup_enabled: bool = True
    full_backup_interval_days: int = 7
    incremental_backup_interval_hours: int = 6
    max_concurrent_backups: int = 2
    
    # 보존 정책
    retention_days: int = 30
    max_backup_count: int = 50
    max_backup_size_gb: float = 10.0
    
    # 검증
    verify_backups: bool = True
    checksum_algorithm: str = "sha256"
    
    # 성능
    chunk_size_mb: int = 64
    parallel_compression: bool = True
    bandwidth_limit_mbps: float = 0.0  # 0 = 무제한

@dataclass
class BackupEntry:
    """백업 엔트리"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    source_path: str = ""
    backup_path: str = ""
    compression: CompressionType = CompressionType.NONE
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    file_count: int = 0
    checksum: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'source_path': self.source_path,
            'backup_path': self.backup_path,
            'compression': self.compression.value,
            'original_size_bytes': self.original_size_bytes,
            'compressed_size_bytes': self.compressed_size_bytes,
            'compression_ratio': self.compression_ratio,
            'file_count': self.file_count,
            'checksum': self.checksum,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

@dataclass
class BackupMetrics:
    """백업 메트릭"""
    total_backups: int = 0
    successful_backups: int = 0
    failed_backups: int = 0
    total_size_bytes: int = 0
    avg_compression_ratio: float = 0.0
    last_backup_time: Optional[datetime] = None
    next_scheduled_backup: Optional[datetime] = None
    backup_frequency_hours: float = 0.0
    storage_efficiency: float = 0.0

class BackupManager:
    """백업 관리 시스템"""
    
    def __init__(self, config: Optional[BackupConfig] = None, config_file: str = "backup_config.json"):
        self.config = config or BackupConfig()
        self.config_file = config_file
        
        # Logger 먼저 초기화
        try:
            import logging
            self.logger = logging.getLogger("BackupManager")
        except Exception:
            import logging
            logging.basicConfig()
            self.logger = logging.getLogger("BackupManager")
        
        # 디렉토리 설정
        self.backup_dir = Path(self.config.backup_dir)
        self.temp_dir = Path(self.config.temp_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 설정
        self.db_path = self.backup_dir / "backup_registry.db"
        self._init_database()
        
        # 백업 상태 추적
        self.active_backups: Dict[str, BackupEntry] = {}
        self.backup_queue: List[str] = []
        self.metrics = BackupMetrics()
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 스케줄러
        self._scheduler_active = False
        self._scheduler_thread = None
        
        # 설정 로드
        self._load_configuration()
        
        # 스케줄러 시작
        if self.config.auto_backup_enabled:
            self.start_scheduler()
        
        self.logger.info("Backup manager initialized")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    backup_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    source_path TEXT,
                    backup_path TEXT,
                    compression TEXT DEFAULT 'none',
                    original_size_bytes INTEGER DEFAULT 0,
                    compressed_size_bytes INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 0.0,
                    file_count INTEGER DEFAULT 0,
                    checksum TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backup_created_at 
                ON backups(created_at)
                """)
                
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backup_status 
                ON backups(status)
                """)
                
                # 기존 데이터베이스에 compression 컬럼 추가 (마이그레이션)
                try:
                    conn.execute("ALTER TABLE backups ADD COLUMN compression TEXT DEFAULT 'none'")
                except sqlite3.OperationalError:
                    # 이미 컬럼이 있는 경우 무시
                    pass
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 업데이트
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        if key in ['backup_type', 'compression']:
                            # Enum 타입 처리
                            enum_class = BackupType if key == 'backup_type' else CompressionType
                            setattr(self.config, key, enum_class(value))
                        else:
                            setattr(self.config, key, value)
                
                self.logger.info(f"Backup configuration loaded from {self.config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load backup configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config_data = {
                'backup_dir': self.config.backup_dir,
                'temp_dir': self.config.temp_dir,
                'source_dirs': self.config.source_dirs,
                'exclude_patterns': self.config.exclude_patterns,
                'include_databases': self.config.include_databases,
                'backup_type': self.config.backup_type.value,
                'compression': self.config.compression.value,
                'compression_level': self.config.compression_level,
                'auto_backup_enabled': self.config.auto_backup_enabled,
                'full_backup_interval_days': self.config.full_backup_interval_days,
                'incremental_backup_interval_hours': self.config.incremental_backup_interval_hours,
                'max_concurrent_backups': self.config.max_concurrent_backups,
                'retention_days': self.config.retention_days,
                'max_backup_count': self.config.max_backup_count,
                'max_backup_size_gb': self.config.max_backup_size_gb,
                'verify_backups': self.config.verify_backups,
                'checksum_algorithm': self.config.checksum_algorithm,
                'chunk_size_mb': self.config.chunk_size_mb,
                'parallel_compression': self.config.parallel_compression,
                'bandwidth_limit_mbps': self.config.bandwidth_limit_mbps
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save backup configuration: {e}")
    
    async def create_backup(self, backup_type: Optional[BackupType] = None, 
                          source_path: Optional[str] = None) -> str:
        """백업 생성"""
        try:
            # 백업 ID 생성
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{backup_type.value if backup_type else self.config.backup_type.value}"
            
            # 백업 엔트리 생성
            backup_entry = BackupEntry(
                backup_id=backup_id,
                backup_type=backup_type or self.config.backup_type,
                status=BackupStatus.PENDING,
                created_at=datetime.now(),
                source_path=source_path or str(Path.cwd()),
                compression=self.config.compression
            )
            
            # 데이터베이스에 등록
            self._save_backup_entry(backup_entry)
            
            # 백업 실행
            await self._execute_backup(backup_entry)
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise
    
    async def _execute_backup(self, backup_entry: BackupEntry):
        """백업 실행"""
        try:
            with self._lock:
                self.active_backups[backup_entry.backup_id] = backup_entry
            
            # 상태 업데이트
            backup_entry.status = BackupStatus.RUNNING
            self._save_backup_entry(backup_entry)
            
            self.logger.info(f"Starting backup: {backup_entry.backup_id}")
            
            # 백업 타입에 따른 처리
            if backup_entry.backup_type == BackupType.FULL:
                await self._create_full_backup(backup_entry)
            elif backup_entry.backup_type == BackupType.INCREMENTAL:
                await self._create_incremental_backup(backup_entry)
            elif backup_entry.backup_type == BackupType.DIFFERENTIAL:
                await self._create_differential_backup(backup_entry)
            else:  # SNAPSHOT
                await self._create_snapshot_backup(backup_entry)
            
            # 검증
            if self.config.verify_backups:
                await self._verify_backup(backup_entry)
            
            # 완료 처리
            backup_entry.status = BackupStatus.COMPLETED
            backup_entry.completed_at = datetime.now()
            backup_entry.compression_ratio = (
                (backup_entry.original_size_bytes - backup_entry.compressed_size_bytes) / 
                max(1, backup_entry.original_size_bytes)
            ) if backup_entry.original_size_bytes > 0 else 0.0
            
            self._save_backup_entry(backup_entry)
            self.logger.info(f"Backup completed: {backup_entry.backup_id}")
            
        except Exception as e:
            # 에러 처리
            backup_entry.status = BackupStatus.FAILED
            backup_entry.error_message = str(e)
            backup_entry.completed_at = datetime.now()
            self._save_backup_entry(backup_entry)
            
            self.logger.error(f"Backup failed {backup_entry.backup_id}: {e}")
            raise
        
        finally:
            # 활성 백업에서 제거
            with self._lock:
                self.active_backups.pop(backup_entry.backup_id, None)
    
    async def _create_full_backup(self, backup_entry: BackupEntry):
        """전체 백업 생성"""
        try:
            # 백업 파일명
            backup_filename = f"{backup_entry.backup_id}.{self._get_extension()}"
            backup_path = self.backup_dir / backup_filename
            backup_entry.backup_path = str(backup_path)
            
            # 임시 디렉토리에서 작업
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                temp_path = Path(temp_dir)
                
                # 소스 디렉토리들 수집
                total_size = 0
                file_count = 0
                
                for source_dir in self.config.source_dirs:
                    source_path = Path(source_dir)
                    if source_path.exists():
                        size, count = await self._copy_directory(
                            source_path, 
                            temp_path / source_path.name,
                            backup_entry
                        )
                        total_size += size
                        file_count += count
                
                backup_entry.original_size_bytes = total_size
                backup_entry.file_count = file_count
                
                # 압축
                await self._compress_backup(temp_path, backup_path, backup_entry)
                
        except Exception as e:
            self.logger.error(f"Full backup creation failed: {e}")
            raise
    
    async def _create_incremental_backup(self, backup_entry: BackupEntry):
        """증분 백업 생성"""
        try:
            # 마지막 백업 시간 찾기
            last_backup_time = self._get_last_backup_time()
            
            # 백업 파일명
            backup_filename = f"{backup_entry.backup_id}.{self._get_extension()}"
            backup_path = self.backup_dir / backup_filename
            backup_entry.backup_path = str(backup_path)
            
            # 임시 디렉토리에서 작업
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                temp_path = Path(temp_dir)
                
                # 변경된 파일들만 수집
                total_size = 0
                file_count = 0
                
                for source_dir in self.config.source_dirs:
                    source_path = Path(source_dir)
                    if source_path.exists():
                        size, count = await self._copy_changed_files(
                            source_path,
                            temp_path / source_path.name,
                            last_backup_time,
                            backup_entry
                        )
                        total_size += size
                        file_count += count
                
                backup_entry.original_size_bytes = total_size
                backup_entry.file_count = file_count
                
                # 압축
                await self._compress_backup(temp_path, backup_path, backup_entry)
                
        except Exception as e:
            self.logger.error(f"Incremental backup creation failed: {e}")
            raise
    
    async def _create_differential_backup(self, backup_entry: BackupEntry):
        """차등 백업 생성"""
        try:
            # 마지막 전체 백업 시간 찾기
            last_full_backup_time = self._get_last_full_backup_time()
            
            # 백업 파일명
            backup_filename = f"{backup_entry.backup_id}.{self._get_extension()}"
            backup_path = self.backup_dir / backup_filename
            backup_entry.backup_path = str(backup_path)
            
            # 임시 디렉토리에서 작업
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                temp_path = Path(temp_dir)
                
                # 마지막 전체 백업 이후 변경된 파일들 수집
                total_size = 0
                file_count = 0
                
                for source_dir in self.config.source_dirs:
                    source_path = Path(source_dir)
                    if source_path.exists():
                        size, count = await self._copy_changed_files(
                            source_path,
                            temp_path / source_path.name,
                            last_full_backup_time,
                            backup_entry
                        )
                        total_size += size
                        file_count += count
                
                backup_entry.original_size_bytes = total_size
                backup_entry.file_count = file_count
                
                # 압축
                await self._compress_backup(temp_path, backup_path, backup_entry)
                
        except Exception as e:
            self.logger.error(f"Differential backup creation failed: {e}")
            raise
    
    async def _create_snapshot_backup(self, backup_entry: BackupEntry):
        """스냅샷 백업 생성"""
        # 스냅샷은 현재 상태의 메타데이터만 저장
        try:
            snapshot_data = {
                'timestamp': datetime.now().isoformat(),
                'source_dirs': self.config.source_dirs,
                'file_listing': {}
            }
            
            total_size = 0
            file_count = 0
            
            for source_dir in self.config.source_dirs:
                source_path = Path(source_dir)
                if source_path.exists():
                    file_info = await self._get_directory_snapshot(source_path)
                    snapshot_data['file_listing'][source_dir] = file_info
                    
                    # 크기 계산
                    for info in file_info.values():
                        total_size += info.get('size', 0)
                        file_count += 1
            
            backup_entry.original_size_bytes = total_size
            backup_entry.file_count = file_count
            
            # 스냅샷 데이터 저장
            snapshot_file = self.backup_dir / f"{backup_entry.backup_id}_snapshot.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            backup_entry.backup_path = str(snapshot_file)
            backup_entry.compressed_size_bytes = snapshot_file.stat().st_size
            
        except Exception as e:
            self.logger.error(f"Snapshot backup creation failed: {e}")
            raise
    
    async def _copy_directory(self, src: Path, dst: Path, backup_entry: BackupEntry) -> Tuple[int, int]:
        """디렉토리 복사"""
        total_size = 0
        file_count = 0
        
        try:
            dst.mkdir(parents=True, exist_ok=True)
            
            for item in src.rglob("*"):
                if item.is_file() and not self._should_exclude(item):
                    rel_path = item.relative_to(src)
                    dst_file = dst / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(item, dst_file)
                    
                    file_size = item.stat().st_size
                    total_size += file_size
                    file_count += 1
                    
                    # 진행률 업데이트
                    if file_count % 100 == 0:
                        await asyncio.sleep(0.001)  # 비동기 처리를 위한 양보
            
            return total_size, file_count
            
        except Exception as e:
            self.logger.error(f"Directory copy failed: {e}")
            raise
    
    async def _copy_changed_files(self, src: Path, dst: Path, since_time: datetime, 
                                backup_entry: BackupEntry) -> Tuple[int, int]:
        """변경된 파일들만 복사"""
        total_size = 0
        file_count = 0
        
        try:
            dst.mkdir(parents=True, exist_ok=True)
            
            for item in src.rglob("*"):
                if item.is_file() and not self._should_exclude(item):
                    # 수정 시간 확인
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime > since_time:
                        rel_path = item.relative_to(src)
                        dst_file = dst / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(item, dst_file)
                        
                        file_size = item.stat().st_size
                        total_size += file_size
                        file_count += 1
                        
                        # 진행률 업데이트
                        if file_count % 50 == 0:
                            await asyncio.sleep(0.001)
            
            return total_size, file_count
            
        except Exception as e:
            self.logger.error(f"Changed files copy failed: {e}")
            raise
    
    async def _get_directory_snapshot(self, path: Path) -> Dict[str, Any]:
        """디렉토리 스냅샷 정보 수집"""
        snapshot = {}
        
        try:
            for item in path.rglob("*"):
                if item.is_file() and not self._should_exclude(item):
                    rel_path = str(item.relative_to(path))
                    stat = item.stat()
                    
                    snapshot[rel_path] = {
                        'size': stat.st_size,
                        'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'checksum': await self._calculate_file_checksum(item)
                    }
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Directory snapshot failed: {e}")
            return {}
    
    def _should_exclude(self, path: Path) -> bool:
        """파일 제외 여부 확인"""
        import fnmatch
        
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return True
        return False
    
    async def _compress_backup(self, source_path: Path, backup_path: Path, backup_entry: BackupEntry):
        """백업 압축"""
        try:
            if self.config.compression == CompressionType.NONE:
                # 압축 없이 복사
                if source_path.is_dir():
                    shutil.copytree(source_path, backup_path)
                else:
                    shutil.copy2(source_path, backup_path)
                backup_entry.compressed_size_bytes = backup_entry.original_size_bytes
                
            elif self.config.compression == CompressionType.GZIP:
                # gzip 압축
                with open(backup_path, 'wb') as f_out:
                    with gzip.GzipFile(fileobj=f_out, compresslevel=self.config.compression_level) as gz_out:
                        await self._write_tar_to_stream(source_path, gz_out)
                
            elif self.config.compression == CompressionType.ZIP:
                # ZIP 압축
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED, 
                                   compresslevel=self.config.compression_level) as zf:
                    await self._add_to_zip(source_path, zf, source_path)
                
            elif self.config.compression == CompressionType.TAR_GZ:
                # tar.gz 압축
                with tarfile.open(backup_path, 'w:gz', compresslevel=self.config.compression_level) as tar:
                    await self._add_to_tar(source_path, tar, source_path)
                    
            elif self.config.compression == CompressionType.TAR_BZ2:
                # tar.bz2 압축
                with tarfile.open(backup_path, 'w:bz2', compresslevel=self.config.compression_level) as tar:
                    await self._add_to_tar(source_path, tar, source_path)
            
            # 압축된 크기 계산
            backup_entry.compressed_size_bytes = backup_path.stat().st_size
            
        except Exception as e:
            self.logger.error(f"Backup compression failed: {e}")
            raise
    
    async def _write_tar_to_stream(self, source_path: Path, stream):
        """TAR을 스트림에 쓰기"""
        import tarfile
        # GzipFile 스트림에 직접 TAR 쓰기
        with tarfile.open(fileobj=stream, mode='w') as tar:
            if source_path.is_dir():
                for item in source_path.rglob("*"):
                    if item.is_file():
                        arc_name = str(item.relative_to(source_path))
                        tar.add(item, arcname=arc_name)
                        await asyncio.sleep(0.001)  # 비동기 처리
            elif source_path.is_file():
                tar.add(source_path, arcname=source_path.name)
    
    async def _add_to_zip(self, source_path: Path, zip_file: zipfile.ZipFile, base_path: Path):
        """ZIP에 파일 추가"""
        for item in source_path.rglob("*"):
            if item.is_file():
                arc_name = str(item.relative_to(base_path))
                zip_file.write(item, arc_name)
                await asyncio.sleep(0.001)  # 비동기 처리
    
    async def _add_to_tar(self, source_path: Path, tar_file: tarfile.TarFile, base_path: Path):
        """TAR에 파일 추가"""
        for item in source_path.rglob("*"):
            if item.is_file():
                arc_name = str(item.relative_to(base_path))
                tar_file.add(item, arc_name)
                await asyncio.sleep(0.001)  # 비동기 처리
    
    def _get_extension(self) -> str:
        """압축 타입에 따른 확장자 반환"""
        if self.config.compression == CompressionType.GZIP:
            return "tar.gz"
        elif self.config.compression == CompressionType.ZIP:
            return "zip"
        elif self.config.compression == CompressionType.TAR_GZ:
            return "tar.gz"
        elif self.config.compression == CompressionType.TAR_BZ2:
            return "tar.bz2"
        else:
            return "backup"
    
    async def _verify_backup(self, backup_entry: BackupEntry):
        """백업 검증"""
        try:
            backup_path = Path(backup_entry.backup_path)
            if not backup_path.exists():
                raise Exception("Backup file not found")
            
            # 체크섬 계산
            checksum = await self._calculate_file_checksum(backup_path)
            backup_entry.checksum = checksum
            
            # 압축 파일 무결성 검사
            if self.config.compression == CompressionType.ZIP:
                with zipfile.ZipFile(backup_path, 'r') as zf:
                    bad_files = zf.testzip()
                    if bad_files:
                        raise Exception(f"Corrupted files in backup: {bad_files}")
            
            elif self.config.compression in [CompressionType.TAR_GZ, CompressionType.TAR_BZ2]:
                mode = 'r:gz' if self.config.compression == CompressionType.TAR_GZ else 'r:bz2'
                with tarfile.open(backup_path, mode) as tar:
                    # TAR 파일 검증은 단순히 읽기 시도
                    tar.getnames()
            
            self.logger.info(f"Backup verification successful: {backup_entry.backup_id}")
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            raise
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        try:
            hash_obj = hashlib.new(self.config.checksum_algorithm)
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.config.chunk_size_mb * 1024 * 1024):
                    hash_obj.update(chunk)
                    await asyncio.sleep(0.001)  # 비동기 처리
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _get_last_backup_time(self) -> datetime:
        """마지막 백업 시간 반환"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                SELECT created_at FROM backups 
                WHERE status = 'completed' 
                ORDER BY created_at DESC LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    return datetime.fromisoformat(result[0])
                else:
                    return datetime.min
                    
        except Exception:
            return datetime.min
    
    def _get_last_full_backup_time(self) -> datetime:
        """마지막 전체 백업 시간 반환"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                SELECT created_at FROM backups 
                WHERE status = 'completed' AND backup_type = 'full'
                ORDER BY created_at DESC LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    return datetime.fromisoformat(result[0])
                else:
                    return datetime.min
                    
        except Exception:
            return datetime.min
    
    def _save_backup_entry(self, entry: BackupEntry):
        """백업 엔트리 저장"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO backups 
                (backup_id, backup_type, status, created_at, completed_at, 
                 source_path, backup_path, compression, original_size_bytes, compressed_size_bytes,
                 compression_ratio, file_count, checksum, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.backup_id,
                    entry.backup_type.value,
                    entry.status.value,
                    entry.created_at.isoformat(),
                    entry.completed_at.isoformat() if entry.completed_at else None,
                    entry.source_path,
                    entry.backup_path,
                    entry.compression.value,
                    entry.original_size_bytes,
                    entry.compressed_size_bytes,
                    entry.compression_ratio,
                    entry.file_count,
                    entry.checksum,
                    entry.error_message,
                    json.dumps(entry.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save backup entry: {e}")
    
    def get_backup_list(self, limit: int = 50) -> List[BackupEntry]:
        """백업 목록 반환"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                SELECT * FROM backups 
                ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                
                backups = []
                for row in cursor.fetchall():
                    entry = BackupEntry(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        status=BackupStatus(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                        source_path=row[5] or "",
                        backup_path=row[6] or "",
                        compression=CompressionType(row[7]) if row[7] else CompressionType.NONE,
                        original_size_bytes=row[8],
                        compressed_size_bytes=row[9],
                        compression_ratio=row[10],
                        file_count=row[11],
                        checksum=row[12] or "",
                        error_message=row[13] or "",
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    backups.append(entry)
                
                return backups
                
        except Exception as e:
            self.logger.error(f"Failed to get backup list: {e}")
            return []
    
    async def restore_backup(self, backup_id: str, restore_path: str) -> bool:
        """백업 복원"""
        try:
            # 백업 정보 조회
            backup_entry = self._get_backup_entry(backup_id)
            if not backup_entry:
                raise Exception(f"Backup not found: {backup_id}")
            
            backup_path = Path(backup_entry.backup_path)
            if not backup_path.exists():
                raise Exception(f"Backup file not found: {backup_path}")
            
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Restoring backup {backup_id} to {restore_path}")
            
            # 압축 타입에 따른 복원
            if self.config.compression == CompressionType.ZIP:
                with zipfile.ZipFile(backup_path, 'r') as zf:
                    zf.extractall(restore_dir)
                    
            elif self.config.compression in [CompressionType.TAR_GZ, CompressionType.TAR_BZ2]:
                mode = 'r:gz' if self.config.compression == CompressionType.TAR_GZ else 'r:bz2'
                with tarfile.open(backup_path, mode) as tar:
                    tar.extractall(restore_dir)
                    
            else:
                # 압축되지 않은 백업
                if backup_path.is_dir():
                    shutil.copytree(backup_path, restore_dir / backup_path.name)
                else:
                    shutil.copy2(backup_path, restore_dir)
            
            self.logger.info(f"Backup restoration completed: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup restoration failed: {e}")
            return False
    
    def _get_backup_entry(self, backup_id: str) -> Optional[BackupEntry]:
        """백업 엔트리 조회"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                SELECT * FROM backups WHERE backup_id = ?
                """, (backup_id,))
                
                row = cursor.fetchone()
                if row:
                    return BackupEntry(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        status=BackupStatus(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                        source_path=row[5] or "",
                        backup_path=row[6] or "",
                        compression=CompressionType(row[7]) if row[7] else CompressionType.NONE,
                        original_size_bytes=row[8],
                        compressed_size_bytes=row[9],
                        compression_ratio=row[10],
                        file_count=row[11],
                        checksum=row[12] or "",
                        error_message=row[13] or "",
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get backup entry: {e}")
            return None
    
    def start_scheduler(self):
        """백업 스케줄러 시작"""
        if self._scheduler_active:
            return
        
        self._scheduler_active = True
        
        def scheduler_loop():
            while self._scheduler_active:
                try:
                    # 스케줄 확인 및 백업 실행
                    self._check_backup_schedule()
                    
                    # 정리 작업
                    self._cleanup_old_backups()
                    
                    # 대기 (1시간)
                    for _ in range(3600):
                        if not self._scheduler_active:
                            break
                        import time
                    time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Scheduler error: {e}")
                    time.sleep(300)  # 5분 후 재시도
        
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("Backup scheduler started")
    
    def stop_scheduler(self):
        """백업 스케줄러 중지"""
        if not self._scheduler_active:
            return
        
        self._scheduler_active = False
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=10)
        
        self.logger.info("Backup scheduler stopped")
    
    def _check_backup_schedule(self):
        """백업 스케줄 확인"""
        try:
            now = datetime.now()
            
            # 마지막 전체 백업 확인
            last_full_backup = self._get_last_full_backup_time()
            if now - last_full_backup > timedelta(days=self.config.full_backup_interval_days):
                asyncio.create_task(self.create_backup(BackupType.FULL))
                return
            
            # 마지막 증분 백업 확인
            last_backup = self._get_last_backup_time()
            if now - last_backup > timedelta(hours=self.config.incremental_backup_interval_hours):
                asyncio.create_task(self.create_backup(BackupType.INCREMENTAL))
                
        except Exception as e:
            self.logger.error(f"Schedule check failed: {e}")
    
    def _cleanup_old_backups(self):
        """오래된 백업 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # 오래된 백업 조회
                cursor = conn.execute("""
                SELECT backup_id, backup_path FROM backups 
                WHERE created_at < ? AND status = 'completed'
                """, (cutoff_date.isoformat(),))
                
                old_backups = cursor.fetchall()
                
                # 백업 파일 삭제
                for backup_id, backup_path in old_backups:
                    try:
                        path = Path(backup_path)
                        if path.exists():
                            path.unlink()
                        
                        # 데이터베이스에서 삭제
                        conn.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id,))
                        
                        self.logger.info(f"Cleaned up old backup: {backup_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup backup {backup_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """백업 통계 반환"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 기본 통계
                cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(compressed_size_bytes) as total_size,
                    AVG(compression_ratio) as avg_compression
                FROM backups
                """)
                
                stats = cursor.fetchone()
                
                # 최근 백업 시간
                cursor = conn.execute("""
                SELECT created_at FROM backups 
                WHERE status = 'completed' 
                ORDER BY created_at DESC LIMIT 1
                """)
                last_backup = cursor.fetchone()
                
                return {
                    'total_backups': stats[0],
                    'successful_backups': stats[1],
                    'failed_backups': stats[2],
                    'total_size_bytes': stats[3] or 0,
                    'total_size_gb': (stats[3] or 0) / (1024**3),
                    'avg_compression_ratio': f"{(stats[4] or 0) * 100:.1f}%",
                    'last_backup_time': last_backup[0] if last_backup else None,
                    'storage_usage_percent': ((stats[3] or 0) / (1024**3)) / self.config.max_backup_size_gb * 100,
                    'active_backups': len(self.active_backups),
                    'scheduler_active': self._scheduler_active
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get backup statistics: {e}")
            return {'error': str(e)}
    
    async def get_backup_by_id(self, backup_id: str) -> Optional[BackupEntry]:
        """백업 ID로 백업 엔트리 조회"""
        return self._get_backup_entry(backup_id)
    
    async def list_backups(self, status: Optional[BackupStatus] = None, 
                          backup_type: Optional[BackupType] = None,
                          limit: Optional[int] = None) -> List[BackupEntry]:
        """백업 목록 조회"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                query = "SELECT * FROM backups"
                params = []
                conditions = []
                
                if status:
                    conditions.append("status = ?")
                    params.append(status.value)
                
                if backup_type:
                    conditions.append("backup_type = ?")
                    params.append(backup_type.value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                backups = []
                for row in rows:
                    # CompressionType 안전 변환
                    compression_value = row[7]
                    if isinstance(compression_value, str):
                        try:
                            compression = CompressionType(compression_value)
                        except ValueError:
                            compression = CompressionType.NONE
                    elif isinstance(compression_value, int):
                        # 정수로 저장된 경우 (오래된 데이터)
                        compression_map = {
                            0: CompressionType.NONE,
                            1: CompressionType.GZIP,
                            2: CompressionType.ZIP,
                            3: CompressionType.TAR_GZ,
                            4: CompressionType.TAR_BZ2
                        }
                        compression = compression_map.get(compression_value, CompressionType.NONE)
                    else:
                        compression = CompressionType.NONE
                    
                    backup = BackupEntry(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        status=BackupStatus(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                        source_path=row[5] or "",
                        backup_path=row[6] or "",
                        compression=CompressionType(row[7]) if row[7] else CompressionType.NONE,
                        original_size_bytes=row[8] or 0,
                        compressed_size_bytes=row[9] or 0,
                        compression_ratio=row[10] or 0.0,
                        file_count=row[11] or 0,
                        checksum=row[12] or "",
                        error_message=row[13] or "",
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    backups.append(backup)
                
                return backups
                
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    async def verify_backup(self, backup_entry: BackupEntry) -> bool:
        """백업 검증"""
        try:
            backup_path = Path(backup_entry.backup_path)
            
            # 파일 존재 확인
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # 체크섬 검증
            if backup_entry.checksum:
                calculated_checksum = self._calculate_checksum(backup_path)
                if calculated_checksum != backup_entry.checksum:
                    self.logger.error(f"Checksum mismatch for {backup_entry.backup_id}")
                    return False
            
            # 압축 파일 무결성 검사
            if backup_entry.compression != CompressionType.NONE:
                try:
                    if backup_entry.compression == CompressionType.GZIP:
                        with gzip.open(backup_path, 'rb') as f:
                            f.read(1024)  # 첫 1KB 읽기 테스트
                    elif backup_entry.compression == CompressionType.ZIP:
                        import zipfile
                        with zipfile.ZipFile(backup_path, 'r') as zf:
                            zf.testzip()  # 압축 파일 무결성 테스트
                except Exception as e:
                    self.logger.error(f"Backup compression integrity check failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """파일 체크섬 계산"""
        try:
            import hashlib
            
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                # 대용량 파일을 위한 청크 단위 읽기
                chunk_size = 64 * 1024  # 64KB
                while chunk := f.read(chunk_size):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return ""
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_scheduler()
            self._save_configuration()
            
            # 임시 디렉토리 정리
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            self.logger.info("Backup manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Backup manager cleanup failed: {e}")

# 전역 백업 관리자
_global_backup_manager = None

def get_backup_manager(config: Optional[BackupConfig] = None, config_file: str = None) -> BackupManager:
    """전역 백업 관리자 반환"""
    global _global_backup_manager
    if _global_backup_manager is None:
        _global_backup_manager = BackupManager(
            config=config,
            config_file=config_file or "backup_config.json"
        )
    return _global_backup_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_backup_manager():
        print("🧪 Backup Manager 테스트")
        
        # 설정
        config = BackupConfig(
            backup_dir="test_backups",
            temp_dir="test_temp",
            source_dirs=["logs", "config"],
            backup_type=BackupType.FULL,
            compression=CompressionType.GZIP,
            auto_backup_enabled=False,
            verify_backups=True,
            retention_days=7
        )
        
        manager = get_backup_manager(config, "test_backup_config.json")
        
        print("\n1️⃣ 전체 백업 생성")
        backup_id = await manager.create_backup(BackupType.FULL)
        print(f"  백업 ID: {backup_id}")
        
        print("\n2️⃣ 백업 목록 조회")
        backups = manager.get_backup_list(5)
        for backup in backups:
            print(f"  {backup.backup_id}: {backup.status.value} ({backup.file_count} files)")
        
        print("\n3️⃣ 백업 통계")
        stats = manager.get_backup_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n4️⃣ 증분 백업 생성")
        await asyncio.sleep(1)  # 시간 차이를 위한 대기
        incremental_id = await manager.create_backup(BackupType.INCREMENTAL)
        print(f"  증분 백업 ID: {incremental_id}")
        
        print("\n5️⃣ 백업 복원 테스트")
        restore_success = await manager.restore_backup(backup_id, "test_restore")
        print(f"  복원 성공: {restore_success}")
        
        print("\n🎉 Backup Manager 테스트 완료!")
        
        # 정리
        manager.cleanup()
        
        # 테스트 파일 정리
        import shutil
        for dir_name in ["test_backups", "test_temp", "test_restore"]:
            test_dir = Path(dir_name)
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
        
        test_config = Path("test_backup_config.json")
        if test_config.exists():
            test_config.unlink()
    
    # 테스트 실행
    asyncio.run(test_backup_manager())