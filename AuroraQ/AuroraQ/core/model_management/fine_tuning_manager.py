#!/usr/bin/env python3
"""
Fine-tuning 관리 시스템
P4: 모델 품질 모니터링 및 Fine-tuning 시스템 구축
"""

import sys
import os
import json
import time
import logging
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import threading
import warnings

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class TuningStrategy(Enum):
    """Fine-tuning 전략"""
    INCREMENTAL = "incremental"  # 점진적 개선
    AGGRESSIVE = "aggressive"    # 적극적 재학습
    CONSERVATIVE = "conservative" # 보수적 조정
    EMERGENCY = "emergency"      # 긴급 복구

class TuningStatus(Enum):
    """Fine-tuning 상태"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelBackupLevel(Enum):
    """모델 백업 레벨"""
    MINIMAL = "minimal"      # 최소 백업
    STANDARD = "standard"    # 표준 백업
    COMPLETE = "complete"    # 완전 백업

@dataclass
class TuningConfig:
    """Fine-tuning 설정"""
    model_type: str
    strategy: TuningStrategy = TuningStrategy.INCREMENTAL
    learning_rate: float = 1e-5
    batch_size: int = 16
    max_epochs: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    backup_level: ModelBackupLevel = ModelBackupLevel.STANDARD
    auto_rollback: bool = True
    quality_threshold: float = 0.8
    max_training_time_hours: int = 24
    data_augmentation: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TuningJob:
    """Fine-tuning 작업"""
    job_id: str
    model_type: str
    config: TuningConfig
    status: TuningStatus = TuningStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_epoch: int = 0
    best_metric: float = 0.0
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    backup_path: Optional[str] = None
    
    def add_log(self, message: str):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        logger.info(f"Job {self.job_id}: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'job_id': self.job_id,
            'model_type': self.model_type,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'logs': self.logs[-10:],  # 최근 10개 로그만
            'error_message': self.error_message,
            'backup_path': self.backup_path,
            'config': asdict(self.config)
        }

@dataclass
class ModelBackup:
    """모델 백업 정보"""
    backup_id: str
    model_type: str
    backup_path: str
    created_at: datetime
    backup_level: ModelBackupLevel
    model_version: str = "unknown"
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'backup_id': self.backup_id,
            'model_type': self.model_type,
            'backup_path': self.backup_path,
            'created_at': self.created_at.isoformat(),
            'backup_level': self.backup_level.value,
            'model_version': self.model_version,
            'quality_metrics': self.quality_metrics,
            'notes': self.notes,
            'size_mb': self.size_mb
        }

class MockTrainer:
    """Mock 학습기 (실제 구현 대신 시뮬레이션)"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_data = []
        
    async def prepare_data(self, data_source: str) -> bool:
        """데이터 준비 시뮬레이션"""
        await asyncio.sleep(0.5)  # 시뮬레이션 지연
        return True
    
    async def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """에포크 학습 시뮬레이션"""
        await asyncio.sleep(1.0)  # 시뮬레이션 지연
        
        # 시뮬레이션된 학습 결과
        base_metric = 0.6 + (epoch * 0.05) + np.random.normal(0, 0.02)
        metric = min(0.95, max(0.3, base_metric))
        
        metrics = {
            'accuracy': metric,
            'loss': 1.0 - metric + np.random.normal(0, 0.01),
            'val_accuracy': metric - 0.02 + np.random.normal(0, 0.01)
        }
        
        return metric, metrics
    
    async def validate(self) -> Dict[str, float]:
        """검증 시뮬레이션"""
        await asyncio.sleep(0.3)
        
        return {
            'accuracy': self.best_metric + np.random.normal(0, 0.01),
            'precision': self.best_metric - 0.01 + np.random.normal(0, 0.01),
            'recall': self.best_metric - 0.02 + np.random.normal(0, 0.01)
        }

class FineTuningManager:
    """Fine-tuning 관리자"""
    
    def __init__(self, models_dir: str = "models", backups_dir: str = "model_backups"):
        self.models_dir = Path(models_dir)
        self.backups_dir = Path(backups_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
        
        # 작업 관리
        self.active_jobs: Dict[str, TuningJob] = {}
        self.job_history: List[TuningJob] = []
        self.backups: Dict[str, ModelBackup] = {}
        
        # 스레드 관리
        self._job_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        
        # 품질 모니터 통합
        self.quality_monitor = None
        
        logger.info(f"Fine-tuning manager initialized (models: {self.models_dir}, backups: {self.backups_dir})")
        
        # 기존 백업 로드
        self._load_existing_backups()
    
    def set_quality_monitor(self, monitor):
        """품질 모니터 설정"""
        self.quality_monitor = monitor
        logger.info("Quality monitor integrated with fine-tuning manager")
    
    def _load_existing_backups(self):
        """기존 백업 로드"""
        try:
            backup_index_file = self.backups_dir / "backup_index.json"
            if backup_index_file.exists():
                with open(backup_index_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                for backup_info in backup_data:
                    backup = ModelBackup(
                        backup_id=backup_info['backup_id'],
                        model_type=backup_info['model_type'],
                        backup_path=backup_info['backup_path'],
                        created_at=datetime.fromisoformat(backup_info['created_at']),
                        backup_level=ModelBackupLevel(backup_info['backup_level']),
                        model_version=backup_info.get('model_version', 'unknown'),
                        quality_metrics=backup_info.get('quality_metrics', {}),
                        notes=backup_info.get('notes', ''),
                        size_mb=backup_info.get('size_mb', 0.0)
                    )
                    self.backups[backup.backup_id] = backup
                
                logger.info(f"Loaded {len(self.backups)} existing backups")
        except Exception as e:
            logger.warning(f"Failed to load existing backups: {e}")
    
    def _save_backup_index(self):
        """백업 인덱스 저장"""
        try:
            backup_index_file = self.backups_dir / "backup_index.json"
            backup_data = [backup.to_dict() for backup in self.backups.values()]
            
            with open(backup_index_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save backup index: {e}")
    
    def create_model_backup(self, model_type: str, model_path: str,
                           backup_level: ModelBackupLevel = ModelBackupLevel.STANDARD,
                           notes: str = "") -> Optional[ModelBackup]:
        """모델 백업 생성"""
        try:
            # 백업 ID 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"{model_type}_backup_{timestamp}"
            
            # 백업 경로
            backup_dir = self.backups_dir / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            source_path = Path(model_path)
            if not source_path.exists():
                logger.warning(f"Model path does not exist: {model_path}")
                # 더미 모델 파일 생성 (테스트용)
                source_path.parent.mkdir(parents=True, exist_ok=True)
                with open(source_path, 'w') as f:
                    f.write(f"# Dummy model file for {model_type}\n")
                    f.write(f"# Created at: {datetime.now().isoformat()}\n")
                logger.info(f"Created dummy model file: {source_path}")
            
            # 파일 복사 또는 시뮬레이션
            backup_file = backup_dir / source_path.name
            if source_path.is_file():
                shutil.copy2(source_path, backup_file)
                size_mb = backup_file.stat().st_size / (1024 * 1024)
            else:
                # 더미 백업 파일 생성
                with open(backup_file, 'w') as f:
                    f.write(f"# Backup of {model_type} model\n")
                    f.write(f"# Original path: {model_path}\n")
                    f.write(f"# Backup level: {backup_level.value}\n")
                    f.write(f"# Created: {datetime.now().isoformat()}\n")
                size_mb = 0.1  # 더미 파일 크기
            
            # 메타데이터 파일 생성
            metadata_file = backup_dir / "metadata.json"
            metadata = {
                'backup_id': backup_id,
                'model_type': model_type,
                'original_path': str(source_path),
                'backup_level': backup_level.value,
                'created_at': datetime.now().isoformat(),
                'notes': notes
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 품질 메트릭 수집 (가능한 경우)
            quality_metrics = {}
            if self.quality_monitor:
                try:
                    from utils.model_quality_monitor import ModelType
                    model_type_enum = ModelType(model_type.lower())
                    quality_metrics = self.quality_monitor.calculate_quality_metrics(model_type_enum, hours=1)
                    quality_metrics = {k.value: v for k, v in quality_metrics.items()}
                except:
                    pass
            
            # 백업 객체 생성
            backup = ModelBackup(
                backup_id=backup_id,
                model_type=model_type,
                backup_path=str(backup_dir),
                created_at=datetime.now(),
                backup_level=backup_level,
                quality_metrics=quality_metrics,
                notes=notes,
                size_mb=size_mb
            )
            
            self.backups[backup_id] = backup
            self._save_backup_index()
            
            logger.info(f"Model backup created: {backup_id} (size: {size_mb:.1f}MB)")
            return backup
            
        except Exception as e:
            logger.error(f"Failed to create model backup: {e}")
            return None
    
    def restore_model_backup(self, backup_id: str, target_path: str) -> bool:
        """모델 백업 복원"""
        try:
            if backup_id not in self.backups:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            backup = self.backups[backup_id]
            backup_dir = Path(backup.backup_path)
            
            if not backup_dir.exists():
                logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # 백업 파일 찾기
            backup_files = list(backup_dir.glob("*.zip")) + list(backup_dir.glob("*.pkl")) + list(backup_dir.glob("*.bin"))
            if not backup_files:
                # 메타데이터나 더미 파일도 복원
                backup_files = list(backup_dir.glob("*"))
                backup_files = [f for f in backup_files if f.name != "metadata.json"]
            
            if not backup_files:
                logger.error(f"No backup files found in {backup_dir}")
                return False
            
            # 대상 경로 준비
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복원
            source_file = backup_files[0]  # 첫 번째 파일 사용
            shutil.copy2(source_file, target_path)
            
            logger.info(f"Model restored from backup {backup_id} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore model backup: {e}")
            return False
    
    def create_tuning_job(self, model_type: str, config: TuningConfig = None) -> str:
        """Fine-tuning 작업 생성"""
        try:
            # 작업 ID 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"tune_{model_type}_{timestamp}"
            
            # 기본 설정 사용
            if config is None:
                config = TuningConfig(model_type=model_type)
            
            # 작업 생성
            job = TuningJob(
                job_id=job_id,
                model_type=model_type,
                config=config
            )
            
            with self._lock:
                self.active_jobs[job_id] = job
            
            job.add_log(f"Fine-tuning job created for {model_type}")
            logger.info(f"Created fine-tuning job: {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create tuning job: {e}")
            raise
    
    async def start_tuning_job(self, job_id: str, data_source: str = None) -> bool:
        """Fine-tuning 작업 시작"""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job not found: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status != TuningStatus.IDLE:
                logger.error(f"Job {job_id} is not in idle status: {job.status}")
                return False
            
            # 백업 생성
            if job.config.backup_level != ModelBackupLevel.MINIMAL:
                model_path = self.models_dir / f"{job.model_type}_model"
                backup = self.create_model_backup(
                    job.model_type,
                    str(model_path),
                    job.config.backup_level,
                    f"Pre-tuning backup for job {job_id}"
                )
                if backup:
                    job.backup_path = backup.backup_path
                    job.add_log(f"Model backup created: {backup.backup_id}")
            
            # 비동기 학습 시작
            job.status = TuningStatus.PREPARING
            job.started_at = datetime.now()
            job.add_log("Starting fine-tuning process")
            
            # 별도 스레드에서 학습 실행
            def run_training():
                asyncio.run(self._execute_training(job_id, data_source))
            
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()
            
            with self._lock:
                self._job_threads[job_id] = thread
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start tuning job {job_id}: {e}")
            return False
    
    async def _execute_training(self, job_id: str, data_source: str = None):
        """학습 실행 (내부 메서드)"""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        
        try:
            # Mock 학습기 초기화
            trainer = MockTrainer(job.config)
            
            # 데이터 준비
            job.status = TuningStatus.PREPARING
            job.add_log("Preparing training data...")
            
            success = await trainer.prepare_data(data_source or "default")
            if not success:
                raise Exception("Failed to prepare training data")
            
            job.progress = 0.1
            job.add_log("Data preparation completed")
            
            # 학습 시작
            job.status = TuningStatus.TRAINING
            job.add_log("Starting training...")
            
            best_metric = 0.0
            patience_counter = 0
            
            for epoch in range(job.config.max_epochs):
                if job.status == TuningStatus.CANCELLED:
                    job.add_log("Training cancelled by user")
                    return
                
                job.current_epoch = epoch + 1
                job.add_log(f"Training epoch {job.current_epoch}/{job.config.max_epochs}")
                
                # 에포크 학습
                metric, metrics = await trainer.train_epoch(epoch)
                
                # 진행률 업데이트
                job.progress = 0.1 + (0.7 * (epoch + 1) / job.config.max_epochs)
                
                # 최고 성능 업데이트
                if metric > best_metric:
                    best_metric = metric
                    job.best_metric = best_metric
                    patience_counter = 0
                    job.add_log(f"New best metric: {best_metric:.4f}")
                else:
                    patience_counter += 1
                
                job.add_log(f"Epoch {job.current_epoch} - Metric: {metric:.4f}, Loss: {metrics.get('loss', 0):.4f}")
                
                # Early stopping 체크
                if patience_counter >= job.config.early_stopping_patience:
                    job.add_log(f"Early stopping triggered (patience: {patience_counter})")
                    break
                
                # 시간 제한 체크
                if job.started_at:
                    elapsed_hours = (datetime.now() - job.started_at).total_seconds() / 3600
                    if elapsed_hours > job.config.max_training_time_hours:
                        job.add_log("Training time limit exceeded")
                        break
            
            # 검증
            job.status = TuningStatus.VALIDATING
            job.add_log("Validating model...")
            
            validation_metrics = await trainer.validate()
            job.progress = 0.9
            
            # 품질 체크
            final_quality = validation_metrics.get('accuracy', best_metric)
            if final_quality < job.config.quality_threshold:
                if job.config.auto_rollback and job.backup_path:
                    job.add_log(f"Quality below threshold ({final_quality:.3f} < {job.config.quality_threshold})")
                    job.add_log("Auto-rollback enabled, restoring backup...")
                    
                    model_path = self.models_dir / f"{job.model_type}_model"
                    if self.restore_model_backup(Path(job.backup_path).name, str(model_path)):
                        job.add_log("Model rollback completed")
                    else:
                        job.add_log("Model rollback failed")
                else:
                    job.add_log(f"Warning: Quality below threshold ({final_quality:.3f})")
            
            # 완료
            job.status = TuningStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.add_log(f"Fine-tuning completed - Final quality: {final_quality:.4f}")
            
            # 품질 모니터에 업데이트 (가능한 경우)
            if self.quality_monitor:
                try:
                    from utils.model_quality_monitor import ModelType
                    model_type_enum = ModelType(job.model_type.lower())
                    self.quality_monitor.record_prediction(
                        model_type=model_type_enum,
                        input_data="Fine-tuning validation",
                        prediction="tuning_completed",
                        confidence=final_quality,
                        metadata={
                            'job_id': job_id,
                            'epochs': job.current_epoch,
                            'final_metric': best_metric
                        }
                    )
                except Exception as e:
                    job.add_log(f"Failed to update quality monitor: {e}")
            
        except Exception as e:
            job.status = TuningStatus.FAILED
            job.error_message = str(e)
            job.add_log(f"Training failed: {e}")
            logger.error(f"Training failed for job {job_id}: {e}")
        
        finally:
            # 작업 완료 후 정리
            with self._lock:
                if job_id in self.active_jobs:
                    completed_job = self.active_jobs.pop(job_id)
                    self.job_history.append(completed_job)
                
                if job_id in self._job_threads:
                    del self._job_threads[job_id]
    
    def cancel_tuning_job(self, job_id: str) -> bool:
        """Fine-tuning 작업 취소"""
        try:
            if job_id not in self.active_jobs:
                logger.error(f"Job not found: {job_id}")
                return False
            
            job = self.active_jobs[job_id]
            
            if job.status in [TuningStatus.COMPLETED, TuningStatus.FAILED, TuningStatus.CANCELLED]:
                logger.warning(f"Job {job_id} is already finished: {job.status}")
                return False
            
            job.status = TuningStatus.CANCELLED
            job.add_log("Job cancellation requested")
            
            logger.info(f"Cancelled tuning job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel tuning job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        try:
            # 활성 작업 확인
            if job_id in self.active_jobs:
                return self.active_jobs[job_id].to_dict()
            
            # 완료된 작업 확인
            for job in self.job_history:
                if job.job_id == job_id:
                    return job.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            return None
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """활성 작업 목록"""
        try:
            with self._lock:
                return [job.to_dict() for job in self.active_jobs.values()]
        except Exception as e:
            logger.error(f"Failed to list active jobs: {e}")
            return []
    
    def list_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """작업 히스토리"""
        try:
            # 최신순으로 정렬
            sorted_history = sorted(self.job_history, 
                                  key=lambda x: x.created_at, reverse=True)
            return [job.to_dict() for job in sorted_history[:limit]]
        except Exception as e:
            logger.error(f"Failed to get job history: {e}")
            return []
    
    def list_backups(self, model_type: str = None) -> List[Dict[str, Any]]:
        """백업 목록"""
        try:
            backups = list(self.backups.values())
            
            if model_type:
                backups = [b for b in backups if b.model_type == model_type]
            
            # 최신순으로 정렬
            backups.sort(key=lambda x: x.created_at, reverse=True)
            
            return [backup.to_dict() for backup in backups]
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def delete_backup(self, backup_id: str) -> bool:
        """백업 삭제"""
        try:
            if backup_id not in self.backups:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            backup = self.backups[backup_id]
            backup_path = Path(backup.backup_path)
            
            # 백업 디렉토리 삭제
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            # 백업 목록에서 제거
            del self.backups[backup_id]
            self._save_backup_index()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            return {
                'active_jobs_count': len(self.active_jobs),
                'total_jobs_history': len(self.job_history),
                'total_backups': len(self.backups),
                'models_dir': str(self.models_dir),
                'backups_dir': str(self.backups_dir),
                'models_dir_size_mb': self._get_directory_size(self.models_dir),
                'backups_dir_size_mb': self._get_directory_size(self.backups_dir),
                'active_job_ids': list(self.active_jobs.keys()),
                'quality_monitor_connected': self.quality_monitor is not None
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def _get_directory_size(self, directory: Path) -> float:
        """디렉토리 크기 계산 (MB)"""
        try:
            if not directory.exists():
                return 0.0
            
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 모든 활성 작업 취소
            for job_id in list(self.active_jobs.keys()):
                self.cancel_tuning_job(job_id)
            
            # 스레드 정리
            for thread in self._job_threads.values():
                if thread.is_alive():
                    thread.join(timeout=5)
            
            self._job_threads.clear()
            
            logger.info("Fine-tuning manager cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 Fine-tuning 관리자
_global_tuning_manager = None

def get_tuning_manager(models_dir: str = None, backups_dir: str = None) -> FineTuningManager:
    """전역 Fine-tuning 관리자 반환"""
    global _global_tuning_manager
    if _global_tuning_manager is None:
        _global_tuning_manager = FineTuningManager(
            models_dir or "models",
            backups_dir or "model_backups"
        )
    return _global_tuning_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_fine_tuning_manager():
        print("🧪 Fine-tuning Manager 테스트")
        
        manager = get_tuning_manager("test_models", "test_backups")
        
        print("\n1️⃣ 모델 백업 생성")
        backup = manager.create_model_backup(
            model_type="finbert",
            model_path="test_models/finbert_model.pkl",
            backup_level=ModelBackupLevel.STANDARD,
            notes="테스트 백업"
        )
        
        if backup:
            print(f"  ✅ 백업 생성: {backup.backup_id}")
        
        print("\n2️⃣ Fine-tuning 작업 생성 및 실행")
        
        # FinBERT 작업
        finbert_config = TuningConfig(
            model_type="finbert",
            strategy=TuningStrategy.INCREMENTAL,
            max_epochs=3,
            quality_threshold=0.75
        )
        
        job_id = manager.create_tuning_job("finbert", finbert_config)
        print(f"  📝 작업 생성: {job_id}")
        
        # 작업 시작
        success = await manager.start_tuning_job(job_id)
        print(f"  🚀 작업 시작: {'성공' if success else '실패'}")
        
        # 진행 상황 모니터링
        for _ in range(10):
            status = manager.get_job_status(job_id)
            if status:
                print(f"  📊 진행률: {status['progress']*100:.1f}% - 상태: {status['status']}")
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    break
            await asyncio.sleep(1)
        
        print("\n3️⃣ 시스템 상태 확인")
        system_status = manager.get_system_status()
        print(f"  📈 활성 작업: {system_status['active_jobs_count']}")
        print(f"  📁 총 백업: {system_status['total_backups']}")
        print(f"  💾 백업 크기: {system_status['backups_dir_size_mb']:.1f}MB")
        
        print("\n4️⃣ 백업 목록 확인")
        backups = manager.list_backups()
        for backup_info in backups:
            print(f"  📦 {backup_info['backup_id']}: {backup_info['model_type']} ({backup_info['size_mb']:.1f}MB)")
        
        print("\n🎉 Fine-tuning Manager 테스트 완료!")
        
        # 정리
        manager.cleanup()
        
        # 테스트 디렉토리 정리
        import shutil
        test_dirs = [Path("test_models"), Path("test_backups")]
        for test_dir in test_dirs:
            if test_dir.exists():
                shutil.rmtree(test_dir)
    
    # 테스트 실행
    asyncio.run(test_fine_tuning_manager())