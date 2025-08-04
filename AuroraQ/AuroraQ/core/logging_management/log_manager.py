#!/usr/bin/env python3
"""
고급 로그 관리 시스템
P7-1: 로그 관리 시스템 설계
"""

import sys
import os
import logging
import logging.handlers
import asyncio
import threading
import gzip
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, TextIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
import warnings
from collections import defaultdict, deque
import traceback

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)

class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogRotationPolicy(Enum):
    """로그 순환 정책"""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class LogConfig:
    """로그 설정"""
    # 기본 설정
    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    max_file_size_mb: int = 50
    max_backup_count: int = 30
    
    # 순환 정책
    rotation_policy: LogRotationPolicy = LogRotationPolicy.SIZE_BASED
    rotation_time: str = "midnight"  # daily 로테이션 시간
    
    # 포맷 설정
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 필터링
    include_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)
    sensitive_keywords: List[str] = field(default_factory=lambda: ["password", "token", "key", "secret"])
    
    # 성능 설정
    buffer_size: int = 1024
    flush_interval: float = 5.0
    compress_old_logs: bool = True
    
    # 경고/에러 특별 처리
    separate_error_log: bool = True
    error_log_retention_days: int = 90
    
    # 모니터링
    enable_metrics: bool = True
    metrics_interval: int = 300  # 5분

@dataclass
class LogEntry:
    """로그 엔트리"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = 0
    process_id: int = 0
    exception_info: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'logger_name': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'exception_info': self.exception_info,
            'extra_data': self.extra_data
        }

@dataclass
class LogMetrics:
    """로그 메트릭"""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    logs_by_module: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0
    warning_rate: float = 0.0
    avg_logs_per_minute: float = 0.0
    peak_logs_per_minute: float = 0.0
    last_error_time: Optional[datetime] = None
    last_warning_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)

class AdvancedFormatter(logging.Formatter):
    """고급 로그 포매터"""
    
    def __init__(self, config: LogConfig):
        super().__init__(config.format_string, config.date_format)
        self.config = config
    
    def format(self, record):
        # 민감한 정보 마스킹
        original_msg = record.getMessage()
        masked_msg = self._mask_sensitive_info(original_msg)
        record.msg = masked_msg
        record.args = None
        
        # 추가 컨텍스트 정보
        record.thread_id = threading.get_ident()
        record.process_id = os.getpid()
        
        # 예외 정보 포함
        if record.exc_info:
            record.exception_info = self.formatException(record.exc_info)
        
        return super().format(record)
    
    def _mask_sensitive_info(self, message: str) -> str:
        """민감한 정보 마스킹"""
        for keyword in self.config.sensitive_keywords:
            # 키워드 = 값 패턴 마스킹
            pattern = rf'({keyword}\s*[=:]\s*)([^\s,\]}}]+)'
            message = re.sub(pattern, r'\1***', message, flags=re.IGNORECASE)
        return message

class LogManager:
    """고급 로그 관리 시스템"""
    
    def __init__(self, config: Optional[LogConfig] = None, config_file: str = "log_config.json"):
        self.config = config or LogConfig()
        self.config_file = config_file
        
        # 로거 관리
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, List[logging.Handler]] = defaultdict(list)
        
        # 메트릭 및 모니터링
        self.metrics = LogMetrics()
        self.log_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.metrics_history: List[LogMetrics] = []
        
        # 스레드 안전성
        self._lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # 모니터링 제어
        self._monitoring_active = False
        self._monitor_thread = None
        self._buffer_thread = None
        
        # 로그 디렉토리 생성
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 로드 및 초기화
        self._load_configuration()
        self._setup_root_logger()
        
        # 모니터링 시작
        if self.config.enable_metrics:
            self.start_monitoring()
        
        # logger는 초기화 완료 후 별도로 설정
        try:
            self.logger = self.get_logger("LogManager")
            self.logger.info("Advanced log manager initialized")
        except Exception:
            print("Info: Advanced log manager initialized")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 업데이트
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        if key in ['log_level', 'rotation_policy']:
                            # Enum 타입 처리
                            enum_class = LogLevel if key == 'log_level' else LogRotationPolicy
                            setattr(self.config, key, enum_class(value))
                        else:
                            setattr(self.config, key, value)
                
                print(f"Info: Log configuration loaded from {self.config_file}")
        except Exception as e:
            print(f"Warning: Failed to load log configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config_data = {
                'log_level': self.config.log_level.value,
                'log_dir': self.config.log_dir,
                'max_file_size_mb': self.config.max_file_size_mb,
                'max_backup_count': self.config.max_backup_count,
                'rotation_policy': self.config.rotation_policy.value,
                'rotation_time': self.config.rotation_time,
                'format_string': self.config.format_string,
                'date_format': self.config.date_format,
                'include_modules': self.config.include_modules,
                'exclude_modules': self.config.exclude_modules,
                'sensitive_keywords': self.config.sensitive_keywords,
                'buffer_size': self.config.buffer_size,
                'flush_interval': self.config.flush_interval,
                'compress_old_logs': self.config.compress_old_logs,
                'separate_error_log': self.config.separate_error_log,
                'error_log_retention_days': self.config.error_log_retention_days,
                'enable_metrics': self.config.enable_metrics,
                'metrics_interval': self.config.metrics_interval
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Failed to save log configuration: {e}")
    
    def _setup_root_logger(self):
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 고급 포매터
        formatter = AdvancedFormatter(self.config)
        
        # 메인 로그 파일 핸들러
        main_handler = self._create_rotating_handler(
            self.log_dir / "auroaq.log",
            formatter
        )
        root_logger.addHandler(main_handler)
        self.handlers["root"].append(main_handler)
        
        # 에러 전용 로그 파일
        if self.config.separate_error_log:
            error_handler = self._create_rotating_handler(
                self.log_dir / "error.log",
                formatter,
                min_level=logging.ERROR
            )
            root_logger.addHandler(error_handler)
            self.handlers["root"].append(error_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.config.log_level.value))
        root_logger.addHandler(console_handler)
        self.handlers["root"].append(console_handler)
    
    def _create_rotating_handler(self, log_file: Path, formatter: logging.Formatter, 
                                min_level: int = logging.DEBUG) -> logging.Handler:
        """순환 핸들러 생성"""
        if self.config.rotation_policy == LogRotationPolicy.SIZE_BASED:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.max_backup_count,
                encoding='utf-8'
            )
        elif self.config.rotation_policy == LogRotationPolicy.TIME_BASED:
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.max_backup_count,
                encoding='utf-8'
            )
        elif self.config.rotation_policy == LogRotationPolicy.DAILY:
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='D',
                interval=1,
                backupCount=self.config.max_backup_count,
                encoding='utf-8'
            )
        elif self.config.rotation_policy == LogRotationPolicy.WEEKLY:
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='W0',  # Monday
                interval=1,
                backupCount=self.config.max_backup_count,
                encoding='utf-8'
            )
        else:  # MONTHLY
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=30,  # 대략 월별
                backupCount=12,
                encoding='utf-8'
            )
        
        handler.setFormatter(formatter)
        handler.setLevel(min_level)
        
        # 압축 활성화
        if self.config.compress_old_logs:
            handler.rotator = self._compress_rotated_log
        
        return handler
    
    def _compress_rotated_log(self, source: str, dest: str):
        """로테이트된 로그 압축"""
        try:
            with open(source, 'rb') as f_in:
                with gzip.open(f"{dest}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(source)
        except Exception as e:
            print(f"Warning: Failed to compress log file {source}: {e}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """로거 반환"""
        with self._lock:
            if name not in self.loggers:
                logger = logging.getLogger(name)
                
                # 모듈 필터링
                if self.config.include_modules and name not in self.config.include_modules:
                    logger.disabled = True
                elif name in self.config.exclude_modules:
                    logger.disabled = True
                
                # 로그 엔트리 수집기 추가
                collector = LogEntryCollector(self)
                logger.addHandler(collector)
                
                self.loggers[name] = logger
            
            return self.loggers[name]
    
    def add_log_entry(self, entry: LogEntry):
        """로그 엔트리 추가"""
        try:
            with self._metrics_lock:
                # 메트릭 업데이트
                self.metrics.total_logs += 1
                self.metrics.logs_by_level[entry.level.value] += 1
                self.metrics.logs_by_module[entry.module] += 1
                
                if entry.level == LogLevel.ERROR:
                    self.metrics.last_error_time = entry.timestamp
                elif entry.level == LogLevel.WARNING:
                    self.metrics.last_warning_time = entry.timestamp
                
                # 버퍼에 추가
                self.log_buffer.append(entry)
                
        except Exception as e:
            print(f"Warning: Failed to add log entry: {e}")
    
    def search_logs(self, query: str, level: Optional[LogLevel] = None, 
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[LogEntry]:
        """로그 검색"""
        try:
            results = []
            
            # 메모리 버퍼에서 검색
            for entry in self.log_buffer:
                if self._matches_criteria(entry, query, level, start_time, end_time):
                    results.append(entry)
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            print(f"Error: Log search failed: {e}")
            return []
    
    def _matches_criteria(self, entry: LogEntry, query: str, level: Optional[LogLevel],
                         start_time: Optional[datetime], end_time: Optional[datetime]) -> bool:
        """검색 조건 확인"""
        # 텍스트 검색
        if query and query.lower() not in entry.message.lower():
            return False
        
        # 레벨 필터
        if level and entry.level != level:
            return False
        
        # 시간 범위
        if start_time and entry.timestamp < start_time:
            return False
        if end_time and entry.timestamp > end_time:
            return False
        
        return True
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """로그 통계 반환"""
        try:
            with self._metrics_lock:
                # 에러율 계산
                total_logs = max(1, self.metrics.total_logs)
                error_count = self.metrics.logs_by_level.get('ERROR', 0)
                warning_count = self.metrics.logs_by_level.get('WARNING', 0)
                
                self.metrics.error_rate = (error_count / total_logs) * 100
                self.metrics.warning_rate = (warning_count / total_logs) * 100
                
                # 분당 로그 수 계산
                elapsed_minutes = (datetime.now() - self.metrics.start_time).total_seconds() / 60
                if elapsed_minutes > 0:
                    self.metrics.avg_logs_per_minute = self.metrics.total_logs / elapsed_minutes
                
                return {
                    'total_logs': self.metrics.total_logs,
                    'logs_by_level': dict(self.metrics.logs_by_level),
                    'logs_by_module': dict(self.metrics.logs_by_module),
                    'error_rate': f"{self.metrics.error_rate:.2f}%",
                    'warning_rate': f"{self.metrics.warning_rate:.2f}%",
                    'avg_logs_per_minute': f"{self.metrics.avg_logs_per_minute:.1f}",
                    'peak_logs_per_minute': f"{self.metrics.peak_logs_per_minute:.1f}",
                    'last_error_time': self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                    'last_warning_time': self.metrics.last_warning_time.isoformat() if self.metrics.last_warning_time else None,
                    'uptime_minutes': elapsed_minutes,
                    'buffer_size': len(self.log_buffer),
                    'monitoring_active': self._monitoring_active
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    # 메트릭 수집
                    self._collect_metrics()
                    
                    # 로그 파일 정리
                    self._cleanup_old_logs()
                    
                    # 대기
                    for _ in range(self.config.metrics_interval):
                        if not self._monitoring_active:
                            break
                        import time
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"Warning: Monitoring loop error: {e}")
                    time.sleep(10)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"Info: Log monitoring started (interval: {self.config.metrics_interval}s)")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        print("Info: Log monitoring stopped")
    
    def _collect_metrics(self):
        """메트릭 수집"""
        try:
            with self._metrics_lock:
                # 현재 메트릭 스냅샷
                current_metrics = LogMetrics(
                    total_logs=self.metrics.total_logs,
                    logs_by_level=dict(self.metrics.logs_by_level),
                    logs_by_module=dict(self.metrics.logs_by_module),
                    error_rate=self.metrics.error_rate,
                    warning_rate=self.metrics.warning_rate,
                    avg_logs_per_minute=self.metrics.avg_logs_per_minute,
                    peak_logs_per_minute=self.metrics.peak_logs_per_minute,
                    last_error_time=self.metrics.last_error_time,
                    last_warning_time=self.metrics.last_warning_time,
                    start_time=datetime.now()
                )
                
                self.metrics_history.append(current_metrics)
                
                # 히스토리 크기 제한
                if len(self.metrics_history) > 288:  # 24시간 (5분 간격)
                    self.metrics_history.pop(0)
                
        except Exception as e:
            print(f"Warning: Failed to collect metrics: {e}")
    
    def _cleanup_old_logs(self):
        """오래된 로그 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.error_log_retention_days)
            
            for log_file in self.log_dir.glob("*.log*"):
                try:
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        print(f"Info: Cleaned up old log file: {log_file}")
                except Exception as e:
                    print(f"Warning: Failed to cleanup log file {log_file}: {e}")
                    
        except Exception as e:
            print(f"Error: Log cleanup failed: {e}")
    
    def export_logs(self, output_file: str, format: str = "json", 
                   level: Optional[LogLevel] = None,
                   start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None) -> bool:
        """로그 내보내기"""
        try:
            entries = self.search_logs("", level, start_time, end_time, limit=10000)
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([entry.to_dict() for entry in entries], f, indent=2, ensure_ascii=False)
            elif format.lower() == "csv":
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if entries:
                        fieldnames = list(entries[0].to_dict().keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for entry in entries:
                            writer.writerow(entry.to_dict())
            else:
                # Plain text
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        f.write(f"{entry.timestamp.isoformat()} [{entry.level.value}] {entry.logger_name}: {entry.message}\n")
            
            print(f"Info: Exported {len(entries)} log entries to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error: Log export failed: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_monitoring()
            
            # 모든 핸들러 정리
            for handlers in self.handlers.values():
                for handler in handlers:
                    handler.close()
            
            # 설정 저장
            self._save_configuration()
            
            print("Info: Log manager cleanup completed")
            
        except Exception as e:
            print(f"Warning: Log manager cleanup failed: {e}")

class LogEntryCollector(logging.Handler):
    """로그 엔트리 수집기"""
    
    def __init__(self, log_manager: LogManager):
        super().__init__()
        self.log_manager = log_manager
    
    def emit(self, record):
        try:
            # LogRecord를 LogEntry로 변환
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=LogLevel(record.levelname),
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module if hasattr(record, 'module') else "",
                function=record.funcName if hasattr(record, 'funcName') else "",
                line_number=record.lineno if hasattr(record, 'lineno') else 0,
                thread_id=getattr(record, 'thread_id', 0),
                process_id=getattr(record, 'process_id', 0),
                exception_info=getattr(record, 'exception_info', None)
            )
            
            self.log_manager.add_log_entry(entry)
            
        except Exception:
            # 로그 수집 중 에러 발생 시 무시 (무한 루프 방지)
            pass

# 전역 로그 관리자
_global_log_manager = None

def get_log_manager(config: Optional[LogConfig] = None, config_file: str = None) -> LogManager:
    """전역 로그 관리자 반환"""
    global _global_log_manager
    if _global_log_manager is None:
        _global_log_manager = LogManager(
            config=config,
            config_file=config_file or "log_config.json"
        )
    return _global_log_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_log_manager():
        print("🧪 Advanced Log Manager 테스트")
        
        # 설정
        config = LogConfig(
            log_level=LogLevel.DEBUG,
            log_dir="test_logs",
            max_file_size_mb=1,  # 작은 크기로 테스트
            rotation_policy=LogRotationPolicy.SIZE_BASED,
            enable_metrics=True,
            metrics_interval=5
        )
        
        manager = get_log_manager(config, "test_log_config.json")
        
        print("\n1️⃣ 다양한 레벨 로그 생성")
        logger = manager.get_logger("TestModule")
        
        logger.debug("디버그 메시지")
        logger.info("정보 메시지")
        logger.warning("경고 메시지")
        logger.error("에러 메시지")
        logger.critical("심각한 에러 메시지")
        
        # 민감한 정보 테스트
        logger.info("사용자 로그인: password=secret123, token=abc123")
        
        print("\n2️⃣ 로그 통계")
        stats = manager.get_log_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n3️⃣ 로그 검색")
        results = manager.search_logs("에러", LogLevel.ERROR, limit=5)
        print(f"  에러 로그 검색 결과: {len(results)}개")
        for result in results:
            print(f"    {result.timestamp}: {result.message}")
        
        print("\n4️⃣ 로그 내보내기")
        export_success = manager.export_logs("test_logs/export.json", "json")
        print(f"  내보내기 성공: {export_success}")
        
        print("\n5️⃣ 대량 로그 생성 (로테이션 테스트)")
        for i in range(100):
            logger.info(f"대량 로그 메시지 {i+1}")
            if i % 20 == 0:
                await asyncio.sleep(0.1)
        
        await asyncio.sleep(2)
        
        print("\n6️⃣ 최종 통계")
        final_stats = manager.get_log_statistics()
        print(f"  총 로그 수: {final_stats['total_logs']}")
        print(f"  에러율: {final_stats['error_rate']}")
        print(f"  분당 평균 로그: {final_stats['avg_logs_per_minute']}")
        
        print("\n🎉 Advanced Log Manager 테스트 완료!")
        
        # 정리
        manager.cleanup()
        
        # 테스트 파일 정리
        import shutil
        test_dir = Path("test_logs")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        test_config = Path("test_log_config.json")
        if test_config.exists():
            test_config.unlink()
    
    # 테스트 실행
    asyncio.run(test_log_manager())