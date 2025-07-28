#!/usr/bin/env python3
"""
AuroraQ Logging and Backup System
실거래 환경용 로깅 및 백업 체계
"""

import os
import sys
import json
import shutil
import sqlite3
import gzip
import asyncio
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import schedule
import time
import subprocess

@dataclass
class BackupConfig:
    """백업 설정"""
    backup_dir: str = "backups"
    retention_days: int = 30
    compress_after_days: int = 7
    database_backup: bool = True
    log_backup: bool = True
    config_backup: bool = True
    max_backup_size_mb: int = 1000

class EnhancedLogger:
    """향상된 로거 시스템"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 메인 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """로그 핸들러 설정"""
        
        # 1. 파일 핸들러 (일반 로그)
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # 2. 에러 전용 핸들러
        error_file = self.log_dir / f"{self.name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        # 3. 시간 기반 핸들러 (일일 아카이브)
        timed_file = self.log_dir / "daily" / f"{self.name}"
        timed_file.parent.mkdir(exist_ok=True)
        
        timed_handler = logging.handlers.TimedRotatingFileHandler(
            str(timed_file) + ".log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        timed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        timed_handler.setFormatter(timed_formatter)
        timed_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(timed_handler)
        
        # 4. 콘솔 핸들러 (개발용)
        if os.getenv('AURORA_DEBUG', 'false').lower() == 'true':
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """로거 인스턴스 반환"""
        return self.logger

class StructuredLogger:
    """구조화된 로깅 시스템"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """거래 로그"""
        structured_data = {
            "event_type": "trade_executed",
            "timestamp": datetime.now().isoformat(),
            "data": trade_data
        }
        self.logger.info(f"TRADE: {json.dumps(structured_data, default=str)}")
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """시그널 로그"""
        structured_data = {
            "event_type": "signal_generated",
            "timestamp": datetime.now().isoformat(),
            "data": signal_data
        }
        self.logger.info(f"SIGNAL: {json.dumps(structured_data, default=str)}")
    
    def log_risk_event(self, risk_data: Dict[str, Any]):
        """리스크 이벤트 로그"""
        structured_data = {
            "event_type": "risk_event",
            "timestamp": datetime.now().isoformat(),
            "data": risk_data
        }
        self.logger.warning(f"RISK: {json.dumps(structured_data, default=str)}")
    
    def log_error(self, error_data: Dict[str, Any]):
        """에러 로그"""
        structured_data = {
            "event_type": "error",
            "timestamp": datetime.now().isoformat(),
            "data": error_data
        }
        self.logger.error(f"ERROR: {json.dumps(structured_data, default=str)}")
    
    def log_performance(self, perf_data: Dict[str, Any]):
        """성능 로그"""
        structured_data = {
            "event_type": "performance_metric",
            "timestamp": datetime.now().isoformat(),
            "data": perf_data
        }
        self.logger.info(f"PERF: {json.dumps(structured_data, default=str)}")

class BackupManager:
    """백업 관리자"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # 백업 디렉토리 구조 생성
        (self.backup_dir / "daily").mkdir(exist_ok=True)
        (self.backup_dir / "weekly").mkdir(exist_ok=True) 
        (self.backup_dir / "monthly").mkdir(exist_ok=True)
        (self.backup_dir / "emergency").mkdir(exist_ok=True)
        
        self.logger = EnhancedLogger("backup_manager").get_logger()
    
    def backup_databases(self) -> List[str]:
        """데이터베이스 백업"""
        backups_created = []
        
        # SQLite 데이터베이스 찾기
        db_patterns = [
            "SharedCore/data_storage/*.db",
            "SharedCore/data_storage/**/*.db",
            "*.db"
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for pattern in db_patterns:
            for db_file in Path(".").glob(pattern):
                if db_file.is_file():
                    try:
                        # 백업 파일명 생성
                        backup_name = f"{db_file.stem}_{timestamp}.db"
                        backup_path = self.backup_dir / "daily" / backup_name
                        
                        # SQLite 백업 (VACUUM INTO 사용)
                        conn = sqlite3.connect(str(db_file))
                        conn.execute(f"VACUUM INTO '{backup_path}'")
                        conn.close()
                        
                        # 7일 후 압축
                        if (datetime.now().hour == 0):  # 자정에만 압축 체크
                            self._compress_old_backups(self.backup_dir / "daily", 7)
                        
                        backups_created.append(str(backup_path))
                        self.logger.info(f"Database backed up: {db_file} -> {backup_path}")
                        
                    except Exception as e:
                        self.logger.error(f"Database backup failed for {db_file}: {e}")
        
        return backups_created
    
    def backup_logs(self) -> List[str]:
        """로그 파일 백업"""
        backups_created = []
        
        log_dirs = ["logs", "SharedCore/logs", "sentiment-service/logs"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            if log_path.exists():
                try:
                    # 로그 디렉토리 전체 압축
                    backup_name = f"logs_{log_path.name}_{timestamp}.tar.gz"
                    backup_path = self.backup_dir / "daily" / backup_name
                    
                    # tar.gz 압축
                    subprocess.run([
                        "tar", "-czf", str(backup_path), "-C", str(log_path.parent), log_path.name
                    ], check=True)
                    
                    backups_created.append(str(backup_path))
                    self.logger.info(f"Logs backed up: {log_dir} -> {backup_path}")
                    
                except Exception as e:
                    self.logger.error(f"Log backup failed for {log_dir}: {e}")
        
        return backups_created
    
    def backup_configs(self) -> List[str]:
        """설정 파일 백업"""
        backups_created = []
        
        config_patterns = [
            "SharedCore/config/*.json",
            "SharedCore/config/*.yaml",
            "sentiment-service/*.json",
            "sentiment-service/*.yaml",
            "*.json",
            "*.yaml"
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_backup_dir = self.backup_dir / "daily" / f"configs_{timestamp}"
        config_backup_dir.mkdir(exist_ok=True)
        
        for pattern in config_patterns:
            for config_file in Path(".").glob(pattern):
                if config_file.is_file() and config_file.stat().st_size < 10*1024*1024:  # 10MB 제한
                    try:
                        backup_path = config_backup_dir / config_file.name
                        shutil.copy2(config_file, backup_path)
                        backups_created.append(str(backup_path))
                        
                    except Exception as e:
                        self.logger.error(f"Config backup failed for {config_file}: {e}")
        
        if backups_created:
            # 설정 디렉토리 압축
            archive_path = self.backup_dir / "daily" / f"configs_{timestamp}.tar.gz"
            subprocess.run([
                "tar", "-czf", str(archive_path), "-C", str(config_backup_dir.parent), config_backup_dir.name
            ], check=True)
            
            # 임시 디렉토리 삭제
            shutil.rmtree(config_backup_dir)
            self.logger.info(f"Configs backed up: {len(backups_created)} files -> {archive_path}")
            return [str(archive_path)]
        
        return []
    
    def _compress_old_backups(self, backup_dir: Path, days_old: int):
        """오래된 백업 압축"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for backup_file in backup_dir.glob("*.db"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    # gzip 압축
                    compressed_path = backup_file.with_suffix(backup_file.suffix + ".gz")
                    
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    backup_file.unlink()  # 원본 삭제
                    self.logger.info(f"Compressed old backup: {backup_file}")
                    
                except Exception as e:
                    self.logger.error(f"Compression failed for {backup_file}: {e}")
    
    def cleanup_old_backups(self):
        """오래된 백업 정리"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        backup_dirs = [
            self.backup_dir / "daily",
            self.backup_dir / "weekly", 
            self.backup_dir / "monthly"
        ]
        
        total_deleted = 0
        total_size_freed = 0
        
        for backup_dir in backup_dirs:
            if backup_dir.exists():
                for backup_file in backup_dir.iterdir():
                    if backup_file.is_file():
                        file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        
                        if file_time < cutoff_date:
                            try:
                                file_size = backup_file.stat().st_size
                                backup_file.unlink()
                                total_deleted += 1
                                total_size_freed += file_size
                                
                            except Exception as e:
                                self.logger.error(f"Failed to delete old backup {backup_file}: {e}")
        
        if total_deleted > 0:
            self.logger.info(f"Cleaned up {total_deleted} old backups, freed {total_size_freed/1024/1024:.1f}MB")
    
    def emergency_backup(self, reason: str = "manual") -> Dict[str, List[str]]:
        """긴급 백업"""
        self.logger.warning(f"Emergency backup triggered: {reason}")
        
        emergency_dir = self.backup_dir / "emergency" / datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_dir.mkdir(parents=True)
        
        # 원래 백업 디렉토리 임시 변경
        original_daily_dir = self.backup_dir / "daily"
        self.backup_dir = emergency_dir.parent
        (self.backup_dir / "daily").mkdir(exist_ok=True)
        
        try:
            results = {
                "databases": self.backup_databases(),
                "logs": self.backup_logs(),
                "configs": self.backup_configs()
            }
            
            # 백업을 emergency 폴더로 이동
            for backup_list in results.values():
                for backup_path in backup_list:
                    backup_file = Path(backup_path)
                    if backup_file.exists():
                        new_path = emergency_dir / backup_file.name
                        shutil.move(backup_file, new_path)
            
            self.logger.info(f"Emergency backup completed: {emergency_dir}")
            return results
            
        finally:
            # 원래 설정 복원
            self.backup_dir = original_daily_dir.parent
            shutil.rmtree(self.backup_dir / "daily", ignore_errors=True)
    
    def get_backup_status(self) -> Dict[str, Any]:
        """백업 상태 조회"""
        status = {
            "last_backup": None,
            "backup_count": 0,
            "total_size_mb": 0.0,
            "oldest_backup": None,
            "disk_usage": {}
        }
        
        backup_files = []
        for backup_dir in [self.backup_dir / "daily", self.backup_dir / "weekly", self.backup_dir / "monthly"]:
            if backup_dir.exists():
                backup_files.extend(backup_dir.iterdir())
        
        if backup_files:
            # 파일 정보 수집
            file_times = [datetime.fromtimestamp(f.stat().st_mtime) for f in backup_files if f.is_file()]
            file_sizes = [f.stat().st_size for f in backup_files if f.is_file()]
            
            status["backup_count"] = len(file_times)
            status["total_size_mb"] = sum(file_sizes) / (1024 * 1024)
            
            if file_times:
                status["last_backup"] = max(file_times).isoformat()
                status["oldest_backup"] = min(file_times).isoformat()
        
        # 디스크 사용량
        disk_usage = shutil.disk_usage(self.backup_dir)
        status["disk_usage"] = {
            "total_gb": disk_usage.total / (1024**3),
            "used_gb": disk_usage.used / (1024**3),
            "free_gb": disk_usage.free / (1024**3)
        }
        
        return status

class LoggingBackupSystem:
    """로깅 및 백업 통합 시스템"""
    
    def __init__(self, backup_config: Optional[BackupConfig] = None):
        self.backup_config = backup_config or BackupConfig()
        self.backup_manager = BackupManager(self.backup_config)
        
        # 시스템 로거들
        self.loggers = {
            "trading": EnhancedLogger("trading_system"),
            "risk": EnhancedLogger("risk_management"),
            "data": EnhancedLogger("data_collection"),
            "sentiment": EnhancedLogger("sentiment_analysis"),
            "system": EnhancedLogger("system_monitor")
        }
        
        # 구조화된 로거들
        self.structured_loggers = {
            name: StructuredLogger(logger.get_logger()) 
            for name, logger in self.loggers.items()
        }
        
        self.system_logger = self.loggers["system"].get_logger()
        self.system_logger.info("Logging and Backup System initialized")
    
    def get_logger(self, name: str) -> logging.Logger:
        """로거 조회"""
        if name in self.loggers:
            return self.loggers[name].get_logger()
        else:
            # 새 로거 생성
            self.loggers[name] = EnhancedLogger(name)
            self.structured_loggers[name] = StructuredLogger(self.loggers[name].get_logger())
            return self.loggers[name].get_logger()
    
    def get_structured_logger(self, name: str) -> StructuredLogger:
        """구조화된 로거 조회"""
        if name not in self.structured_loggers:
            self.get_logger(name)  # 로거 생성
        return self.structured_loggers[name]
    
    def run_scheduled_backup(self):
        """예약된 백업 실행"""
        try:
            self.system_logger.info("Starting scheduled backup...")
            
            results = {
                "databases": [],
                "logs": [],
                "configs": []
            }
            
            if self.backup_config.database_backup:
                results["databases"] = self.backup_manager.backup_databases()
            
            if self.backup_config.log_backup:
                results["logs"] = self.backup_manager.backup_logs()
            
            if self.backup_config.config_backup:
                results["configs"] = self.backup_manager.backup_configs()
            
            # 오래된 백업 정리
            self.backup_manager.cleanup_old_backups()
            
            total_backups = sum(len(backup_list) for backup_list in results.values())
            self.system_logger.info(f"Scheduled backup completed: {total_backups} items backed up")
            
            return results
            
        except Exception as e:
            self.system_logger.error(f"Scheduled backup failed: {e}")
            return None
    
    def setup_backup_schedule(self):
        """백업 스케줄 설정"""
        # 매일 자정 백업
        schedule.every().day.at("00:00").do(self.run_scheduled_backup)
        
        # 매주 일요일 주간 백업
        schedule.every().sunday.at("01:00").do(self._weekly_backup)
        
        # 매월 1일 월간 백업
        schedule.every().month.do(self._monthly_backup)
        
        self.system_logger.info("Backup schedule configured")
    
    def _weekly_backup(self):
        """주간 백업"""
        try:
            # 일일 백업을 주간 폴더로 복사
            daily_dir = self.backup_manager.backup_dir / "daily"
            weekly_dir = self.backup_manager.backup_dir / "weekly"
            
            timestamp = datetime.now().strftime("%Y%m%d")
            
            for backup_file in daily_dir.glob("*"):
                if backup_file.is_file():
                    weekly_file = weekly_dir / f"weekly_{timestamp}_{backup_file.name}"
                    shutil.copy2(backup_file, weekly_file)
            
            self.system_logger.info("Weekly backup completed")
            
        except Exception as e:
            self.system_logger.error(f"Weekly backup failed: {e}")
    
    def _monthly_backup(self):
        """월간 백업"""
        try:
            # 주간 백업을 월간 폴더로 복사
            weekly_dir = self.backup_manager.backup_dir / "weekly"
            monthly_dir = self.backup_manager.backup_dir / "monthly"
            
            timestamp = datetime.now().strftime("%Y%m")
            
            # 가장 최근 주간 백업 찾기
            weekly_backups = sorted(weekly_dir.glob("weekly_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if weekly_backups:
                latest_weekly = weekly_backups[0]
                monthly_file = monthly_dir / f"monthly_{timestamp}_{latest_weekly.name}"
                shutil.copy2(latest_weekly, monthly_file)
                
                self.system_logger.info(f"Monthly backup completed: {monthly_file}")
            
        except Exception as e:
            self.system_logger.error(f"Monthly backup failed: {e}")
    
    def start_backup_scheduler(self):
        """백업 스케줄러 시작"""
        self.system_logger.info("Starting backup scheduler...")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
            except KeyboardInterrupt:
                self.system_logger.info("Backup scheduler stopped by user")
                break
            except Exception as e:
                self.system_logger.error(f"Backup scheduler error: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "logging": {
                "active_loggers": list(self.loggers.keys()),
                "log_directory": "logs"
            },
            "backup": self.backup_manager.get_backup_status(),
            "schedule": {
                "next_backup": schedule.next_run().isoformat() if schedule.jobs else None,
                "pending_jobs": len(schedule.jobs)
            }
        }

# 전역 인스턴스
logging_backup_system = LoggingBackupSystem()

def get_logger(name: str) -> logging.Logger:
    """전역 로거 조회 함수"""
    return logging_backup_system.get_logger(name)

def get_structured_logger(name: str) -> StructuredLogger:
    """전역 구조화된 로거 조회 함수"""
    return logging_backup_system.get_structured_logger(name)

async def main():
    """메인 실행 (테스트용)"""
    print("🚀 AuroraQ Logging and Backup System")
    print("=" * 50)
    
    # 백업 설정
    config = BackupConfig(
        backup_dir="backups",
        retention_days=30,
        compress_after_days=7
    )
    
    system = LoggingBackupSystem(config)
    
    # 테스트 로깅
    trading_logger = system.get_structured_logger("trading")
    trading_logger.log_trade({
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.001,
        "price": 45000,
        "timestamp": datetime.now().isoformat()
    })
    
    # 백업 실행
    print("\n📦 Running backup test...")
    results = system.run_scheduled_backup()
    
    if results:
        total_backups = sum(len(backup_list) for backup_list in results.values())
        print(f"✅ Backup completed: {total_backups} items backed up")
    else:
        print("❌ Backup failed")
    
    # 상태 조회
    print("\n📊 System status:")
    status = system.get_system_status()
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"\n❌ System failed: {e}")
        import traceback
        traceback.print_exc()