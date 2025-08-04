#!/usr/bin/env python3
"""
로그 아카이브 및 순환 관리 시스템
P7-3: 로그 순환 및 아카이브 시스템
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
import fnmatch

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)

class ArchivePolicy(Enum):
    """아카이브 정책"""
    TIME_BASED = "time_based"      # 시간 기반
    SIZE_BASED = "size_based"      # 크기 기반
    COUNT_BASED = "count_based"    # 개수 기반
    HYBRID = "hybrid"              # 복합 정책

class CompressionLevel(Enum):
    """압축 레벨"""
    NONE = 0
    FAST = 1
    NORMAL = 6
    BEST = 9

class StorageTier(Enum):
    """스토리지 계층"""
    HOT = "hot"          # 빠른 액세스 (SSD)
    WARM = "warm"        # 중간 액세스 (HDD)
    COLD = "cold"        # 느린 액세스 (아카이브)
    FROZEN = "frozen"    # 장기 보관 (테이프/클라우드)

@dataclass
class ArchiveConfig:
    """아카이브 설정"""
    # 기본 설정
    archive_dir: str = "archives"
    temp_dir: str = "temp_archive"
    source_patterns: List[str] = field(default_factory=lambda: ["*.log", "*.log.*"])
    
    # 아카이브 정책
    policy: ArchivePolicy = ArchivePolicy.HYBRID
    
    # 시간 기반 설정
    archive_after_days: int = 7
    delete_after_days: int = 90
    
    # 크기 기반 설정
    max_file_size_mb: int = 100
    max_total_size_gb: float = 10.0
    
    # 개수 기반 설정
    max_files_per_directory: int = 1000
    max_archive_count: int = 500
    
    # 압축 설정
    compression_enabled: bool = True
    compression_level: CompressionLevel = CompressionLevel.NORMAL
    compression_format: str = "gzip"  # gzip, bzip2, lzma
    
    # 스토리지 계층
    enable_tiered_storage: bool = True
    hot_storage_days: int = 7
    warm_storage_days: int = 30
    cold_storage_days: int = 90
    
    # 검증 및 무결성
    verify_archives: bool = True
    checksum_algorithm: str = "sha256"
    duplicate_detection: bool = True
    
    # 인덱싱
    enable_indexing: bool = True
    index_content: bool = True
    full_text_search: bool = False
    
    # 성능
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size_mb: int = 64
    
    # 정리
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24

@dataclass
class ArchiveEntry:
    """아카이브 엔트리"""
    archive_id: str
    original_path: str
    archive_path: str
    storage_tier: StorageTier
    created_at: datetime
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexed: bool = False
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'archive_id': self.archive_id,
            'original_path': self.original_path,
            'archive_path': self.archive_path,
            'storage_tier': self.storage_tier.value,
            'created_at': self.created_at.isoformat(),
            'original_size_bytes': self.original_size_bytes,
            'compressed_size_bytes': self.compressed_size_bytes,
            'compression_ratio': self.compression_ratio,
            'checksum': self.checksum,
            'metadata': self.metadata,
            'indexed': self.indexed,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

@dataclass
class ArchiveIndex:
    """아카이브 인덱스"""
    archive_id: str
    file_path: str
    line_number: int
    content_hash: str
    keywords: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    level: Optional[str] = None
    module: Optional[str] = None

@dataclass
class ArchiveMetrics:
    """아카이브 메트릭"""
    total_archives: int = 0
    total_original_size_bytes: int = 0
    total_compressed_size_bytes: int = 0
    avg_compression_ratio: float = 0.0
    storage_efficiency: float = 0.0
    hot_storage_count: int = 0
    warm_storage_count: int = 0
    cold_storage_count: int = 0
    frozen_storage_count: int = 0
    last_archive_time: Optional[datetime] = None
    last_cleanup_time: Optional[datetime] = None

class ArchiveManager:
    """로그 아카이브 및 순환 관리 시스템"""
    
    def __init__(self, config: Optional[ArchiveConfig] = None, config_file: str = "archive_config.json"):
        self.config = config or ArchiveConfig()
        self.config_file = config_file
        
        # Logger 먼저 초기화
        try:
            import logging
            self.logger = logging.getLogger("ArchiveManager")
        except Exception:
            import logging
            logging.basicConfig()
            self.logger = logging.getLogger("ArchiveManager")
        
        # 디렉토리 설정
        self.archive_dir = Path(self.config.archive_dir)
        self.temp_dir = Path(self.config.temp_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 스토리지 계층 디렉토리
        if self.config.enable_tiered_storage:
            for tier in StorageTier:
                tier_dir = self.archive_dir / tier.value
                tier_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 설정
        self.db_path = self.archive_dir / "archive_registry.db"
        self._init_database()
        
        # 아카이브 상태 추적
        self.active_archives: Dict[str, ArchiveEntry] = {}
        self.metrics = ArchiveMetrics()
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 백그라운드 작업
        self._cleanup_active = False
        self._cleanup_thread = None
        
        # 설정 로드
        self._load_configuration()
        
        # 자동 정리 시작
        if self.config.auto_cleanup:
            self.start_auto_cleanup()
        
        self.logger.info("Archive manager initialized")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 아카이브 테이블
                conn.execute("""
                CREATE TABLE IF NOT EXISTS archives (
                    archive_id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    storage_tier TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    original_size_bytes INTEGER DEFAULT 0,
                    compressed_size_bytes INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 0.0,
                    checksum TEXT,
                    metadata TEXT,
                    indexed BOOLEAN DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
                """)
                
                # 인덱스 테이블
                conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    archive_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    content_hash TEXT,
                    keywords TEXT,
                    timestamp TEXT,
                    level TEXT,
                    module TEXT,
                    FOREIGN KEY (archive_id) REFERENCES archives (archive_id)
                )
                """)
                
                # 중복 감지 테이블
                conn.execute("""
                CREATE TABLE IF NOT EXISTS duplicates (
                    content_hash TEXT PRIMARY KEY,
                    archive_id TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (archive_id) REFERENCES archives (archive_id)
                )
                """)
                
                # 인덱스 생성
                conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_created_at ON archives(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_tier ON archives(storage_tier)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_checksum ON archives(checksum)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_index_archive_id ON archive_index(archive_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_index_keywords ON archive_index(keywords)")
                
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
                        if key in ['policy', 'compression_level']:
                            # Enum 타입 처리
                            enum_class = ArchivePolicy if key == 'policy' else CompressionLevel
                            setattr(self.config, key, enum_class(value))
                        else:
                            setattr(self.config, key, value)
                
                self.logger.info(f"Archive configuration loaded from {self.config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load archive configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config_data = {
                'archive_dir': self.config.archive_dir,
                'temp_dir': self.config.temp_dir,
                'source_patterns': self.config.source_patterns,
                'policy': self.config.policy.value,
                'archive_after_days': self.config.archive_after_days,
                'delete_after_days': self.config.delete_after_days,
                'max_file_size_mb': self.config.max_file_size_mb,
                'max_total_size_gb': self.config.max_total_size_gb,
                'max_files_per_directory': self.config.max_files_per_directory,
                'max_archive_count': self.config.max_archive_count,
                'compression_enabled': self.config.compression_enabled,
                'compression_level': self.config.compression_level.value,
                'compression_format': self.config.compression_format,
                'enable_tiered_storage': self.config.enable_tiered_storage,
                'hot_storage_days': self.config.hot_storage_days,
                'warm_storage_days': self.config.warm_storage_days,
                'cold_storage_days': self.config.cold_storage_days,
                'verify_archives': self.config.verify_archives,
                'checksum_algorithm': self.config.checksum_algorithm,
                'duplicate_detection': self.config.duplicate_detection,
                'enable_indexing': self.config.enable_indexing,
                'index_content': self.config.index_content,
                'full_text_search': self.config.full_text_search,
                'parallel_processing': self.config.parallel_processing,
                'max_workers': self.config.max_workers,
                'chunk_size_mb': self.config.chunk_size_mb,
                'auto_cleanup': self.config.auto_cleanup,
                'cleanup_interval_hours': self.config.cleanup_interval_hours
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save archive configuration: {e}")
    
    async def archive_files(self, source_dir: Union[str, Path], 
                          pattern: Optional[str] = None) -> List[str]:
        """파일들을 아카이브"""
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                raise Exception(f"Source directory not found: {source_path}")
            
            # 패턴 설정
            patterns = [pattern] if pattern else self.config.source_patterns
            
            # 아카이브할 파일들 찾기
            files_to_archive = []
            for pattern in patterns:
                files_to_archive.extend(source_path.glob(pattern))
            
            # 아카이브 정책에 따른 필터링
            filtered_files = await self._filter_files_for_archive(files_to_archive)
            
            # 병렬 처리
            if self.config.parallel_processing and len(filtered_files) > 1:
                archive_ids = await self._archive_files_parallel(filtered_files)
            else:
                archive_ids = []
                for file_path in filtered_files:
                    archive_id = await self._archive_single_file(file_path)
                    if archive_id:
                        archive_ids.append(archive_id)
            
            self.logger.info(f"Archived {len(archive_ids)} files from {source_dir}")
            return archive_ids
            
        except Exception as e:
            self.logger.error(f"File archiving failed: {e}")
            return []
    
    async def _filter_files_for_archive(self, files: List[Path]) -> List[Path]:
        """아카이브할 파일 필터링"""
        filtered_files = []
        current_time = datetime.now()
        
        try:
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # 정책에 따른 필터링
                should_archive = False
                file_stat = file_path.stat()
                file_age = current_time - datetime.fromtimestamp(file_stat.st_mtime)
                file_size_mb = file_stat.st_size / (1024 * 1024)
                
                if self.config.policy == ArchivePolicy.TIME_BASED:
                    should_archive = file_age.days >= self.config.archive_after_days
                    
                elif self.config.policy == ArchivePolicy.SIZE_BASED:
                    should_archive = file_size_mb >= self.config.max_file_size_mb
                    
                elif self.config.policy == ArchivePolicy.COUNT_BASED:
                    # 디렉토리당 파일 수 확인
                    parent_files = len(list(file_path.parent.glob("*")))
                    should_archive = parent_files > self.config.max_files_per_directory
                    
                else:  # HYBRID
                    should_archive = (
                        file_age.days >= self.config.archive_after_days or
                        file_size_mb >= self.config.max_file_size_mb
                    )
                
                # 중복 검사
                if should_archive and self.config.duplicate_detection:
                    checksum = await self._calculate_file_checksum(file_path)
                    if await self._is_duplicate(checksum):
                        self.logger.debug(f"Skipping duplicate file: {file_path}")
                        continue
                
                if should_archive:
                    filtered_files.append(file_path)
            
            return filtered_files
            
        except Exception as e:
            self.logger.error(f"File filtering failed: {e}")
            return []
    
    async def _archive_files_parallel(self, files: List[Path]) -> List[str]:
        """병렬 파일 아카이브"""
        import concurrent.futures
        
        archive_ids = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 작업 제출
                future_to_file = {
                    executor.submit(self._archive_single_file_sync, file_path): file_path 
                    for file_path in files
                }
                
                # 결과 수집
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        archive_id = future.result()
                        if archive_id:
                            archive_ids.append(archive_id)
                    except Exception as e:
                        self.logger.error(f"Failed to archive {file_path}: {e}")
            
            return archive_ids
            
        except Exception as e:
            self.logger.error(f"Parallel archiving failed: {e}")
            return []
    
    def _archive_single_file_sync(self, file_path: Path) -> Optional[str]:
        """단일 파일 아카이브 (동기 버전)"""
        import asyncio
        return asyncio.run(self._archive_single_file(file_path))
    
    async def _archive_single_file(self, file_path: Path) -> Optional[str]:
        """단일 파일 아카이브"""
        try:
            # 아카이브 ID 생성
            archive_id = f"arch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.stem}"
            
            # 스토리지 계층 결정
            storage_tier = self._determine_storage_tier(file_path)
            
            # 아카이브 경로 결정
            tier_dir = self.archive_dir / storage_tier.value if self.config.enable_tiered_storage else self.archive_dir
            archive_filename = f"{archive_id}.{self._get_archive_extension()}"
            archive_path = tier_dir / archive_filename
            
            # 원본 파일 정보
            file_stat = file_path.stat()
            original_size = file_stat.st_size
            
            # 체크섬 계산
            checksum = await self._calculate_file_checksum(file_path)
            
            # 압축 및 아카이브
            if self.config.compression_enabled:
                compressed_size = await self._compress_file(file_path, archive_path)
            else:
                shutil.copy2(file_path, archive_path)
                compressed_size = original_size
            
            # 압축 비율 계산
            compression_ratio = (original_size - compressed_size) / max(1, original_size) if original_size > 0 else 0.0
            
            # 아카이브 엔트리 생성
            archive_entry = ArchiveEntry(
                archive_id=archive_id,
                original_path=str(file_path),
                archive_path=str(archive_path),
                storage_tier=storage_tier,
                created_at=datetime.now(),
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio,
                checksum=checksum
            )
            
            # 데이터베이스에 저장
            self._save_archive_entry(archive_entry)
            
            # 중복 검사 정보 저장
            if self.config.duplicate_detection:
                await self._save_duplicate_info(checksum, archive_id, str(file_path))
            
            # 인덱싱
            if self.config.enable_indexing:
                await self._index_archive(archive_entry)
            
            # 검증
            if self.config.verify_archives:
                await self._verify_archive(archive_entry)
            
            # 원본 파일 삭제 (아카이브 완료 후)
            file_path.unlink()
            
            self.logger.debug(f"Archived file: {file_path} -> {archive_path}")
            return archive_id
            
        except Exception as e:
            self.logger.error(f"Failed to archive file {file_path}: {e}")
            return None
    
    def _determine_storage_tier(self, file_path: Path) -> StorageTier:
        """스토리지 계층 결정"""
        try:
            if not self.config.enable_tiered_storage:
                return StorageTier.HOT
            
            file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_age.days <= self.config.hot_storage_days:
                return StorageTier.HOT
            elif file_age.days <= self.config.warm_storage_days:
                return StorageTier.WARM
            elif file_age.days <= self.config.cold_storage_days:
                return StorageTier.COLD
            else:
                return StorageTier.FROZEN
                
        except Exception:
            return StorageTier.HOT
    
    def _get_archive_extension(self) -> str:
        """아카이브 확장자 반환"""
        if not self.config.compression_enabled:
            return "archive"
        
        if self.config.compression_format == "gzip":
            return "gz"
        elif self.config.compression_format == "bzip2":
            return "bz2"
        elif self.config.compression_format == "lzma":
            return "xz"
        else:
            return "archive"
    
    async def _compress_file(self, source_path: Path, archive_path: Path) -> int:
        """파일 압축"""
        try:
            if self.config.compression_format == "gzip":
                with open(source_path, 'rb') as f_in:
                    with gzip.open(archive_path, 'wb', compresslevel=self.config.compression_level.value) as f_out:
                        await self._copy_file_async(f_in, f_out)
                        
            elif self.config.compression_format == "bzip2":
                import bz2
                with open(source_path, 'rb') as f_in:
                    with bz2.open(archive_path, 'wb', compresslevel=self.config.compression_level.value) as f_out:
                        await self._copy_file_async(f_in, f_out)
                        
            elif self.config.compression_format == "lzma":
                import lzma
                with open(source_path, 'rb') as f_in:
                    with lzma.open(archive_path, 'wb', preset=self.config.compression_level.value) as f_out:
                        await self._copy_file_async(f_in, f_out)
            
            return archive_path.stat().st_size
            
        except Exception as e:
            self.logger.error(f"File compression failed: {e}")
            raise
    
    async def _copy_file_async(self, f_in, f_out):
        """비동기 파일 복사"""
        chunk_size = self.config.chunk_size_mb * 1024 * 1024
        
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            await asyncio.sleep(0.001)  # 비동기 처리를 위한 양보
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        try:
            hash_obj = hashlib.new(self.config.checksum_algorithm)
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.config.chunk_size_mb * 1024 * 1024):
                    hash_obj.update(chunk)
                    await asyncio.sleep(0.001)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    async def _is_duplicate(self, checksum: str) -> bool:
        """중복 파일 확인"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM duplicates WHERE content_hash = ?", (checksum,))
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception:
            return False
    
    async def _save_duplicate_info(self, checksum: str, archive_id: str, original_path: str):
        """중복 정보 저장"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO duplicates 
                (content_hash, archive_id, original_path, created_at)
                VALUES (?, ?, ?, ?)
                """, (checksum, archive_id, original_path, datetime.now().isoformat()))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save duplicate info: {e}")
    
    async def _index_archive(self, archive_entry: ArchiveEntry):
        """아카이브 인덱싱"""
        try:
            if not self.config.index_content:
                return
            
            # 압축된 파일 읽기
            content_lines = await self._read_archive_content(archive_entry)
            
            # 키워드 추출 및 인덱싱
            for line_num, line in enumerate(content_lines, 1):
                # 로그 레벨 추출
                level = self._extract_log_level(line)
                
                # 모듈명 추출
                module = self._extract_module_name(line)
                
                # 타임스탬프 추출
                timestamp = self._extract_timestamp(line)
                
                # 키워드 추출
                keywords = self._extract_keywords(line)
                
                # 콘텐츠 해시
                content_hash = hashlib.sha256(line.encode()).hexdigest()[:16]
                
                # 인덱스 저장
                await self._save_index_entry(
                    archive_entry.archive_id,
                    archive_entry.original_path,
                    line_num,
                    content_hash,
                    keywords,
                    timestamp,
                    level,
                    module
                )
            
            # 인덱싱 완료 표시
            archive_entry.indexed = True
            self._save_archive_entry(archive_entry)
            
        except Exception as e:
            self.logger.error(f"Archive indexing failed: {e}")
    
    async def _read_archive_content(self, archive_entry: ArchiveEntry) -> List[str]:
        """아카이브 내용 읽기"""
        try:
            archive_path = Path(archive_entry.archive_path)
            
            if self.config.compression_format == "gzip":
                with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
                    return f.readlines()
                    
            elif self.config.compression_format == "bzip2":
                import bz2
                with bz2.open(archive_path, 'rt', encoding='utf-8') as f:
                    return f.readlines()
                    
            elif self.config.compression_format == "lzma":
                import lzma
                with lzma.open(archive_path, 'rt', encoding='utf-8') as f:
                    return f.readlines()
            else:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
                    
        except Exception as e:
            self.logger.error(f"Failed to read archive content: {e}")
            return []
    
    def _extract_log_level(self, line: str) -> Optional[str]:
        """로그 레벨 추출"""
        import re
        
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in levels:
            if re.search(rf'\b{level}\b', line, re.IGNORECASE):
                return level.upper()
        return None
    
    def _extract_module_name(self, line: str) -> Optional[str]:
        """모듈명 추출"""
        import re
        
        # 일반적인 로그 패턴에서 모듈명 추출
        patterns = [
            r'- ([a-zA-Z_][a-zA-Z0-9_.]*) -',  # logger name pattern
            r'\.([a-zA-Z_][a-zA-Z0-9_]*):',    # module.function: pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """타임스탬프 추출"""
        import re
        
        # 일반적인 타임스탬프 패턴들
        patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                try:
                    # 다양한 포맷 시도
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                except ValueError:
                    pass
        
        return None
    
    def _extract_keywords(self, line: str) -> List[str]:
        """키워드 추출"""
        import re
        
        # 단어 추출 (3글자 이상)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', line.lower())
        
        # 로그 레벨, 일반적인 단어 제외
        exclude_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'way', 'when', 'what', 'with'}
        
        keywords = [word for word in set(words) if word not in exclude_words and len(word) >= 3]
        return keywords[:10]  # 최대 10개 키워드
    
    async def _save_index_entry(self, archive_id: str, file_path: str, line_number: int,
                              content_hash: str, keywords: List[str], timestamp: Optional[datetime],
                              level: Optional[str], module: Optional[str]):
        """인덱스 엔트리 저장"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                INSERT INTO archive_index 
                (archive_id, file_path, line_number, content_hash, keywords, timestamp, level, module)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    archive_id,
                    file_path,
                    line_number,
                    content_hash,
                    json.dumps(keywords),
                    timestamp.isoformat() if timestamp else None,
                    level,
                    module
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save index entry: {e}")
    
    async def _verify_archive(self, archive_entry: ArchiveEntry):
        """아카이브 검증"""
        try:
            archive_path = Path(archive_entry.archive_path)
            if not archive_path.exists():
                raise Exception("Archive file not found")
            
            # 체크섬 검증
            calculated_checksum = await self._calculate_file_checksum(archive_path)
            
            # 압축 파일 무결성 검사
            content_lines = await self._read_archive_content(archive_entry)
            if not content_lines:
                raise Exception("Archive content is empty or corrupted")
            
            self.logger.debug(f"Archive verification successful: {archive_entry.archive_id}")
            
        except Exception as e:
            self.logger.error(f"Archive verification failed: {e}")
            raise
    
    def _save_archive_entry(self, entry: ArchiveEntry):
        """아카이브 엔트리 저장"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO archives 
                (archive_id, original_path, archive_path, storage_tier, created_at,
                 original_size_bytes, compressed_size_bytes, compression_ratio,
                 checksum, metadata, indexed, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.archive_id,
                    entry.original_path,
                    entry.archive_path,
                    entry.storage_tier.value,
                    entry.created_at.isoformat(),
                    entry.original_size_bytes,
                    entry.compressed_size_bytes,
                    entry.compression_ratio,
                    entry.checksum,
                    json.dumps(entry.metadata),
                    entry.indexed,
                    entry.access_count,
                    entry.last_accessed.isoformat() if entry.last_accessed else None
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save archive entry: {e}")
    
    async def search_archives(self, query: str, level: Optional[str] = None,
                            module: Optional[str] = None, start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """아카이브 검색"""
        try:
            if not self.config.enable_indexing:
                return []
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # 검색 조건 구성
                where_conditions = []
                params = []
                
                if query:
                    where_conditions.append("keywords LIKE ?")
                    params.append(f"%{query.lower()}%")
                
                if level:
                    where_conditions.append("level = ?")
                    params.append(level.upper())
                
                if module:
                    where_conditions.append("module = ?")
                    params.append(module)
                
                if start_time:
                    where_conditions.append("timestamp >= ?")
                    params.append(start_time.isoformat())
                
                if end_time:
                    where_conditions.append("timestamp <= ?")
                    params.append(end_time.isoformat())
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # 검색 실행
                cursor = conn.execute(f"""
                SELECT ai.*, a.original_path, a.archive_path, a.storage_tier
                FROM archive_index ai
                JOIN archives a ON ai.archive_id = a.archive_id
                WHERE {where_clause}
                ORDER BY ai.timestamp DESC
                LIMIT ?
                """, params + [limit])
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'archive_id': row[1],
                        'file_path': row[2],
                        'line_number': row[3],
                        'content_hash': row[4],
                        'keywords': json.loads(row[5]) if row[5] else [],
                        'timestamp': row[6],
                        'level': row[7],
                        'module': row[8],
                        'original_path': row[9],
                        'archive_path': row[10],
                        'storage_tier': row[11]
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Archive search failed: {e}")
            return []
    
    async def migrate_storage_tier(self, archive_id: str, target_tier: StorageTier) -> bool:
        """스토리지 계층 마이그레이션"""
        try:
            # 아카이브 정보 조회
            archive_entry = self._get_archive_entry(archive_id)
            if not archive_entry:
                raise Exception(f"Archive not found: {archive_id}")
            
            if archive_entry.storage_tier == target_tier:
                return True  # 이미 목표 계층에 있음
            
            # 새 경로 결정
            new_tier_dir = self.archive_dir / target_tier.value
            new_tier_dir.mkdir(parents=True, exist_ok=True)
            
            old_path = Path(archive_entry.archive_path)
            new_path = new_tier_dir / old_path.name
            
            # 파일 이동
            shutil.move(old_path, new_path)
            
            # 데이터베이스 업데이트
            archive_entry.archive_path = str(new_path)
            archive_entry.storage_tier = target_tier
            self._save_archive_entry(archive_entry)
            
            self.logger.info(f"Migrated archive {archive_id} to {target_tier.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Storage tier migration failed: {e}")
            return False
    
    def _get_archive_entry(self, archive_id: str) -> Optional[ArchiveEntry]:
        """아카이브 엔트리 조회"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT * FROM archives WHERE archive_id = ?", (archive_id,))
                row = cursor.fetchone()
                
                if row:
                    return ArchiveEntry(
                        archive_id=row[0],
                        original_path=row[1],
                        archive_path=row[2],
                        storage_tier=StorageTier(row[3]),
                        created_at=datetime.fromisoformat(row[4]),
                        original_size_bytes=row[5],
                        compressed_size_bytes=row[6],
                        compression_ratio=row[7],
                        checksum=row[8] or "",
                        metadata=json.loads(row[9]) if row[9] else {},
                        indexed=bool(row[10]),
                        access_count=row[11],
                        last_accessed=datetime.fromisoformat(row[12]) if row[12] else None
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get archive entry: {e}")
            return None
    
    def start_auto_cleanup(self):
        """자동 정리 시작"""
        if self._cleanup_active:
            return
        
        self._cleanup_active = True
        
        def cleanup_loop():
            while self._cleanup_active:
                try:
                    # 정리 작업 실행
                    self._perform_cleanup()
                    
                    # 스토리지 계층 마이그레이션
                    self._perform_tier_migration()
                    
                    # 대기
                    for _ in range(self.config.cleanup_interval_hours * 3600):
                        if not self._cleanup_active:
                            break
                        import time
                    time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Cleanup loop error: {e}")
                    time.sleep(3600)  # 1시간 후 재시도
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        self.logger.info("Auto cleanup started")
    
    def stop_auto_cleanup(self):
        """자동 정리 중지"""
        if not self._cleanup_active:
            return
        
        self._cleanup_active = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=10)
        
        self.logger.info("Auto cleanup stopped")
    
    def _perform_cleanup(self):
        """정리 작업 수행"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.delete_after_days)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # 오래된 아카이브 조회
                cursor = conn.execute("""
                SELECT archive_id, archive_path FROM archives 
                WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                old_archives = cursor.fetchall()
                
                # 아카이브 파일 및 데이터 삭제
                for archive_id, archive_path in old_archives:
                    try:
                        # 파일 삭제
                        path = Path(archive_path)
                        if path.exists():
                            path.unlink()
                        
                        # 데이터베이스에서 삭제
                        conn.execute("DELETE FROM archives WHERE archive_id = ?", (archive_id,))
                        conn.execute("DELETE FROM archive_index WHERE archive_id = ?", (archive_id,))
                        conn.execute("DELETE FROM duplicates WHERE archive_id = ?", (archive_id,))
                        
                        self.logger.info(f"Cleaned up old archive: {archive_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup archive {archive_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {e}")
    
    def _perform_tier_migration(self):
        """스토리지 계층 마이그레이션 수행"""
        try:
            if not self.config.enable_tiered_storage:
                return
            
            current_time = datetime.now()
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # 마이그레이션 후보 조회
                cursor = conn.execute("""
                SELECT archive_id, created_at, storage_tier FROM archives 
                WHERE storage_tier != 'frozen'
                """)
                
                for archive_id, created_at_str, current_tier in cursor.fetchall():
                    created_at = datetime.fromisoformat(created_at_str)
                    age_days = (current_time - created_at).days
                    
                    target_tier = None
                    
                    if current_tier == 'hot' and age_days > self.config.hot_storage_days:
                        target_tier = StorageTier.WARM
                    elif current_tier == 'warm' and age_days > self.config.warm_storage_days:
                        target_tier = StorageTier.COLD
                    elif current_tier == 'cold' and age_days > self.config.cold_storage_days:
                        target_tier = StorageTier.FROZEN
                    
                    if target_tier:
                        asyncio.create_task(self.migrate_storage_tier(archive_id, target_tier))
                        
        except Exception as e:
            self.logger.error(f"Tier migration failed: {e}")
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """아카이브 통계 반환"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 기본 통계
                cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(original_size_bytes) as total_original_size,
                    SUM(compressed_size_bytes) as total_compressed_size,
                    AVG(compression_ratio) as avg_compression,
                    storage_tier,
                    COUNT(*) as tier_count
                FROM archives 
                GROUP BY storage_tier
                """)
                
                tier_stats = {}
                total_archives = 0
                total_original_size = 0
                total_compressed_size = 0
                
                for row in cursor.fetchall():
                    tier_stats[row[4]] = row[5]
                    if row[4] == list(tier_stats.keys())[0]:  # 첫 번째 행에서만
                        total_archives = row[0]
                        total_original_size = row[1] or 0
                        total_compressed_size = row[2] or 0
                
                # 전체 통계 다시 계산
                cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(original_size_bytes) as total_original_size,
                    SUM(compressed_size_bytes) as total_compressed_size,
                    AVG(compression_ratio) as avg_compression
                FROM archives
                """)
                
                total_stats = cursor.fetchone()
                
                return {
                    'total_archives': total_stats[0],
                    'total_original_size_bytes': total_stats[1] or 0,
                    'total_compressed_size_bytes': total_stats[2] or 0,
                    'total_original_size_gb': (total_stats[1] or 0) / (1024**3),
                    'total_compressed_size_gb': (total_stats[2] or 0) / (1024**3),
                    'avg_compression_ratio': f"{(total_stats[3] or 0) * 100:.1f}%",
                    'storage_efficiency': f"{((total_stats[1] or 1) - (total_stats[2] or 0)) / (total_stats[1] or 1) * 100:.1f}%",
                    'tier_distribution': tier_stats,
                    'cleanup_active': self._cleanup_active,
                    'indexing_enabled': self.config.enable_indexing,
                    'tiered_storage_enabled': self.config.enable_tiered_storage
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get archive statistics: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_auto_cleanup()
            self._save_configuration()
            
            # 임시 디렉토리 정리
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            self.logger.info("Archive manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Archive manager cleanup failed: {e}")

# 전역 아카이브 관리자
_global_archive_manager = None

def get_archive_manager(config: Optional[ArchiveConfig] = None, config_file: str = None) -> ArchiveManager:
    """전역 아카이브 관리자 반환"""
    global _global_archive_manager
    if _global_archive_manager is None:
        _global_archive_manager = ArchiveManager(
            config=config,
            config_file=config_file or "archive_config.json"
        )
    return _global_archive_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_archive_manager():
        print("🧪 Archive Manager 테스트")
        
        # 설정
        config = ArchiveConfig(
            archive_dir="test_archives",
            temp_dir="test_temp_archive",
            source_patterns=["*.log"],
            policy=ArchivePolicy.TIME_BASED,
            archive_after_days=0,  # 즉시 아카이브
            compression_enabled=True,
            compression_format="gzip",
            enable_indexing=True,
            enable_tiered_storage=True,
            auto_cleanup=False
        )
        
        manager = get_archive_manager(config, "test_archive_config.json")
        
        # 테스트 로그 파일 생성
        test_logs_dir = Path("test_logs")
        test_logs_dir.mkdir(exist_ok=True)
        
        log_file = test_logs_dir / "test.log"
        with open(log_file, 'w') as f:
            f.write("2024-01-01 10:00:00 - INFO - TestModule - Test log message\n")
            f.write("2024-01-01 10:01:00 - ERROR - TestModule - Error occurred\n")
            f.write("2024-01-01 10:02:00 - WARNING - TestModule - Warning message\n")
        
        print("\n1️⃣ 파일 아카이브")
        archive_ids = await manager.archive_files(test_logs_dir, "*.log")
        print(f"  아카이브된 파일 수: {len(archive_ids)}")
        
        print("\n2️⃣ 아카이브 검색")
        search_results = await manager.search_archives("error", limit=10)
        print(f"  검색 결과: {len(search_results)}개")
        for result in search_results:
            print(f"    {result['archive_id']}: {result['level']} - {result['keywords']}")
        
        print("\n3️⃣ 아카이브 통계")
        stats = manager.get_archive_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n4️⃣ 스토리지 계층 마이그레이션")
        if archive_ids:
            migration_success = await manager.migrate_storage_tier(archive_ids[0], StorageTier.COLD)
            print(f"  마이그레이션 성공: {migration_success}")
        
        print("\n🎉 Archive Manager 테스트 완료!")
        
        # 정리
        manager.cleanup()
        
        # 테스트 파일 정리
        import shutil
        for dir_name in ["test_archives", "test_temp_archive", "test_logs"]:
            test_dir = Path(dir_name)
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
        
        test_config = Path("test_archive_config.json")
        if test_config.exists():
            test_config.unlink()
    
    # 테스트 실행
    asyncio.run(test_archive_manager())