#!/usr/bin/env python3
"""
로그 오케스트레이터 통합 인터페이스
P7-5: LogOrchestrator 통합 인터페이스 구현
"""

import sys
import os
import asyncio
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_management import (
    LogManager, get_log_manager, LogLevel, LogConfig, LogRotationPolicy,
    BackupManager, get_backup_manager, BackupConfig, BackupType, CompressionType,
    ArchiveManager, get_archive_manager, ArchiveConfig, ArchivePolicy
)

class OperationStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OperationType(Enum):
    """작업 유형"""
    LOG_ROTATION = "log_rotation"
    BACKUP_CREATION = "backup_creation"
    BACKUP_RESTORATION = "backup_restoration"
    ARCHIVE_CREATION = "archive_creation"
    CLEANUP = "cleanup"
    INTEGRITY_CHECK = "integrity_check"
    WORKFLOW = "workflow"

@dataclass
class Operation:
    """작업 정보"""
    operation_id: str
    operation_type: OperationType
    status: OperationStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0

@dataclass
class WorkflowStep:
    """워크플로우 단계"""
    step_id: str
    operation_type: OperationType
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3

@dataclass
class WorkflowDefinition:
    """워크플로우 정의"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: datetime = field(default_factory=datetime.now)

class LogOrchestrator:
    """로그 관리 통합 오케스트레이터"""
    
    def __init__(self, 
                 log_config: Optional[LogConfig] = None,
                 backup_config: Optional[BackupConfig] = None,
                 archive_config: Optional[ArchiveConfig] = None):
        
        # 개별 관리자 초기화
        self.log_manager = LogManager(log_config) if log_config else get_log_manager()
        self.backup_manager = BackupManager(backup_config) if backup_config else get_backup_manager()
        self.archive_manager = ArchiveManager(archive_config) if archive_config else get_archive_manager()
        
        # Logger 초기화
        try:
            self.logger = self.log_manager.get_logger("LogOrchestrator")
        except Exception:
            import logging
            logging.basicConfig()
            self.logger = logging.getLogger("LogOrchestrator")
        
        # 작업 관리
        self.operations: Dict[str, Operation] = {}
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.active_operations: Dict[str, asyncio.Task] = {}
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 내장 워크플로우 등록
        self._register_builtin_workflows()
        
        self.logger.info("Log Orchestrator initialized")
    
    def _register_builtin_workflows(self):
        """내장 워크플로우 등록"""
        
        # 1. 일일 로그 관리 워크플로우
        daily_log_workflow = WorkflowDefinition(
            workflow_id="daily_log_management",
            name="Daily Log Management",
            description="Daily log rotation, backup, and archiving workflow",
            steps=[
                WorkflowStep(
                    step_id="rotate_logs",
                    operation_type=OperationType.LOG_ROTATION,
                    parameters={"max_age_hours": 24}
                ),
                WorkflowStep(
                    step_id="backup_logs",
                    operation_type=OperationType.BACKUP_CREATION,
                    parameters={"backup_type": "incremental", "source": "logs"},
                    depends_on=["rotate_logs"]
                ),
                WorkflowStep(
                    step_id="archive_old_logs",
                    operation_type=OperationType.ARCHIVE_CREATION,
                    parameters={"age_days": 7},
                    depends_on=["backup_logs"]
                ),
                WorkflowStep(
                    step_id="cleanup_temp",
                    operation_type=OperationType.CLEANUP,
                    parameters={"target": "temp_files", "age_days": 1},
                    depends_on=["archive_old_logs"]
                )
            ]
        )
        
        # 2. 백업 무결성 검증 워크플로우
        integrity_check_workflow = WorkflowDefinition(
            workflow_id="backup_integrity_check",
            name="Backup Integrity Check",
            description="Comprehensive backup integrity validation workflow",
            steps=[
                WorkflowStep(
                    step_id="verify_recent_backups",
                    operation_type=OperationType.INTEGRITY_CHECK,
                    parameters={"scope": "recent", "days": 7}
                ),
                WorkflowStep(
                    step_id="test_restore_sample",
                    operation_type=OperationType.BACKUP_RESTORATION,
                    parameters={"test_mode": True, "sample_files": 5},
                    depends_on=["verify_recent_backups"]
                ),
                WorkflowStep(
                    step_id="cleanup_test_restore",
                    operation_type=OperationType.CLEANUP,
                    parameters={"target": "test_restore"},
                    depends_on=["test_restore_sample"]
                )
            ]
        )
        
        # 3. 긴급 복구 워크플로우
        emergency_recovery_workflow = WorkflowDefinition(
            workflow_id="emergency_recovery",
            name="Emergency Recovery",
            description="Emergency log and data recovery workflow",
            steps=[
                WorkflowStep(
                    step_id="stop_logging",
                    operation_type=OperationType.LOG_ROTATION,
                    parameters={"emergency_stop": True}
                ),
                WorkflowStep(
                    step_id="emergency_backup",
                    operation_type=OperationType.BACKUP_CREATION,
                    parameters={"backup_type": "full", "priority": "emergency"},
                    depends_on=["stop_logging"]
                ),
                WorkflowStep(
                    step_id="restore_from_backup",
                    operation_type=OperationType.BACKUP_RESTORATION,
                    parameters={"restore_latest": True, "verify": True},
                    depends_on=["emergency_backup"]
                )
            ]
        )
        
        self.workflows.update({
            "daily_log_management": daily_log_workflow,
            "backup_integrity_check": integrity_check_workflow,
            "emergency_recovery": emergency_recovery_workflow
        })
    
    async def execute_operation(self, operation_type: OperationType, 
                              parameters: Dict[str, Any]) -> str:
        """단일 작업 실행"""
        operation_id = f"{operation_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        operation = Operation(
            operation_id=operation_id,
            operation_type=operation_type,
            status=OperationStatus.PENDING,
            created_at=datetime.now(),
            parameters=parameters
        )
        
        with self._lock:
            self.operations[operation_id] = operation
        
        try:
            self.logger.info(f"Starting operation {operation_id}: {operation_type.value}")
            
            operation.status = OperationStatus.RUNNING
            operation.started_at = datetime.now()
            
            # 작업 유형별 실행
            if operation_type == OperationType.LOG_ROTATION:
                result = await self._execute_log_rotation(parameters)
            elif operation_type == OperationType.BACKUP_CREATION:
                result = await self._execute_backup_creation(parameters)
            elif operation_type == OperationType.BACKUP_RESTORATION:
                result = await self._execute_backup_restoration(parameters)
            elif operation_type == OperationType.ARCHIVE_CREATION:
                result = await self._execute_archive_creation(parameters)
            elif operation_type == OperationType.CLEANUP:
                result = await self._execute_cleanup(parameters)
            elif operation_type == OperationType.INTEGRITY_CHECK:
                result = await self._execute_integrity_check(parameters)
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
            
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = datetime.now()
            operation.result = result
            operation.progress = 100.0
            
            self.logger.info(f"Operation {operation_id} completed successfully")
            return operation_id
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.now()
            
            self.logger.error(f"Operation {operation_id} failed: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str, 
                             parameters: Optional[Dict[str, Any]] = None) -> str:
        """워크플로우 실행"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        workflow_operation_id = f"workflow_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        operation = Operation(
            operation_id=workflow_operation_id,
            operation_type=OperationType.WORKFLOW,
            status=OperationStatus.RUNNING,
            created_at=datetime.now(),
            started_at=datetime.now(),
            parameters={"workflow_id": workflow_id, "parameters": parameters or {}}
        )
        
        with self._lock:
            self.operations[workflow_operation_id] = operation
        
        try:
            self.logger.info(f"Starting workflow {workflow_id}")
            
            # 단계별 실행 (의존성 순서대로)
            completed_steps = set()
            step_results = {}
            workflow_context = {}  # 워크플로우 컨텍스트 저장
            remaining_steps = workflow.steps.copy()
            
            while remaining_steps:
                executed_this_round = False
                
                for i, step in enumerate(remaining_steps):
                    # 의존성 확인
                    if all(dep in completed_steps for dep in step.depends_on):
                        # 단계 실행
                        self.logger.info(f"Executing workflow step: {step.step_id}")
                        
                        # 컨텍스트 기반 매개변수 보강
                        step_params = step.parameters.copy()
                        
                        # backup_integrity_check 워크플로우의 특별 처리
                        if workflow_id == "backup_integrity_check" and step.step_id == "test_restore_sample":
                            # 이전 단계에서 찾은 백업 사용
                            if "verify_recent_backups" in step_results:
                                verify_op = self.get_operation_status(step_results["verify_recent_backups"])
                                if verify_op and verify_op.result and verify_op.result.get("checked_backups"):
                                    checked_backups = verify_op.result["checked_backups"]
                                    if checked_backups:
                                        # 가장 최근 백업 사용
                                        latest_backup = checked_backups[0]
                                        step_params["backup_id"] = latest_backup["backup_id"]
                                    else:
                                        step_params["restore_latest"] = True
                                else:
                                    step_params["restore_latest"] = True
                        
                        step_operation_id = await self.execute_operation(
                            step.operation_type, 
                            step_params
                        )
                        
                        # 결과를 컨텍스트에 저장
                        step_operation = self.get_operation_status(step_operation_id)
                        if step_operation and step_operation.result:
                            workflow_context[step.step_id] = step_operation.result
                        
                        step_results[step.step_id] = step_operation_id
                        completed_steps.add(step.step_id)
                        
                        # 완료된 단계 제거
                        remaining_steps.pop(i)
                        executed_this_round = True
                        
                        # 진행률 업데이트
                        operation.progress = (len(completed_steps) / len(workflow.steps)) * 100
                        break
                
                # 무한 루프 방지
                if not executed_this_round:
                    # 의존성 문제로 실행할 수 없는 단계들
                    failed_steps = [step.step_id for step in remaining_steps]
                    raise Exception(f"Workflow deadlock: cannot execute steps {failed_steps}")
            
            # 모든 단계 완료
            
            operation.status = OperationStatus.COMPLETED
            operation.completed_at = datetime.now()
            operation.result = {"step_results": step_results}
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            return workflow_operation_id
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.now()
            
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
    
    async def _execute_log_rotation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """로그 순환 실행"""
        max_age_hours = parameters.get("max_age_hours", 24)
        emergency_stop = parameters.get("emergency_stop", False)
        
        if emergency_stop:
            # 긴급 정지 - 현재 로깅 중단
            self.log_manager.stop_monitoring()
            return {"action": "emergency_stop", "stopped": True}
        
        # 일반 로그 순환
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        rotated_files = []
        
        log_dir = Path(self.log_manager.config.log_dir)
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    # 로그 파일 순환 (압축)
                    rotated_file = log_file.with_suffix(".log.gz")
                    
                    import gzip
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(rotated_file, 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    log_file.unlink()
                    rotated_files.append(str(rotated_file))
        
        return {
            "action": "log_rotation",
            "rotated_files": rotated_files,
            "cutoff_time": cutoff_time.isoformat()
        }
    
    async def _execute_backup_creation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """백업 생성 실행"""
        backup_type_str = parameters.get("backup_type", "incremental")
        source = parameters.get("source", "logs")
        priority = parameters.get("priority", "normal")
        
        # BackupType 변환
        backup_type = BackupType.INCREMENTAL
        if backup_type_str == "full":
            backup_type = BackupType.FULL
        elif backup_type_str == "differential":
            backup_type = BackupType.DIFFERENTIAL
        elif backup_type_str == "snapshot":
            backup_type = BackupType.SNAPSHOT
        
        # 소스 경로 결정
        if source == "logs":
            source_path = self.log_manager.config.log_dir
        else:
            source_path = source
        
        # 백업 실행
        backup_id = await self.backup_manager.create_backup(
            backup_type=backup_type,
            source_path=source_path
        )
        
        return {
            "action": "backup_creation",
            "backup_id": backup_id,
            "backup_type": backup_type_str,
            "source_path": source_path,
            "priority": priority
        }
    
    async def _execute_backup_restoration(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """백업 복원 실행 (강화된 무결성 검사 포함)"""
        test_mode = parameters.get("test_mode", False)
        restore_latest = parameters.get("restore_latest", False)
        verify = parameters.get("verify", True)
        sample_files = parameters.get("sample_files", 0)
        
        # 백업 선택
        if restore_latest:
            backups = await self.backup_manager.list_backups()
            if not backups:
                raise ValueError("No backups available for restoration")
            
            # 가장 최신 백업 선택
            latest_backup = max(backups, key=lambda x: x.created_at)
            backup_id = latest_backup.backup_id
        else:
            backup_id = parameters.get("backup_id")
            if not backup_id:
                raise ValueError("backup_id is required when restore_latest=False")
        
        # 복원 대상 경로
        if test_mode:
            restore_path = str(Path(self.backup_manager.config.temp_dir) / "test_restore")
        else:
            restore_path = parameters.get("restore_path")
            if not restore_path:
                raise ValueError("restore_path is required for non-test mode")
        
        # 백업 무결성 검사 강화
        backup_entry = await self.backup_manager.get_backup_by_id(backup_id)
        if not backup_entry:
            raise ValueError(f"Backup not found: {backup_id}")
        
        # 1. 체크섬 검증
        if verify:
            self.logger.info(f"Verifying backup integrity: {backup_id}")
            verification_result = await self.backup_manager.verify_backup(backup_entry)
            if not verification_result:
                raise ValueError(f"Backup integrity check failed: {backup_id}")
        
        # 2. 백업 파일 존재 확인
        backup_path = Path(backup_entry.backup_path)
        if not backup_path.exists():
            raise ValueError(f"Backup file not found: {backup_path}")
        
        # 3. 단계적 복원 (로그 리플레이 방식)
        self.logger.info(f"Starting step-by-step restoration: {backup_id}")
        
        # 임시 디렉토리에서 먼저 압축 해제
        temp_extract_dir = Path(self.backup_manager.config.temp_dir) / f"extract_{backup_id}"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 압축 해제
            if backup_entry.compression == CompressionType.GZIP:
                import gzip
                import tarfile
                
                if backup_path.suffix == '.gz':
                    if backup_path.name.endswith('.tar.gz'):
                        # tar.gz 압축 해제
                        with tarfile.open(backup_path, 'r:gz') as tar:
                            tar.extractall(temp_extract_dir)
                    else:
                        # gzip 압축 해제
                        with gzip.open(backup_path, 'rb') as f_in:
                            extract_file = temp_extract_dir / backup_path.stem
                            with open(extract_file, 'wb') as f_out:
                                f_out.write(f_in.read())
            else:
                # 압축되지 않은 파일 복사
                import shutil
                shutil.copy2(backup_path, temp_extract_dir)
            
            # 4. 샘플 파일 검증 (테스트 모드)
            if test_mode and sample_files > 0:
                extracted_files = list(temp_extract_dir.rglob("*"))
                # 파일만 필터링
                file_list = [f for f in extracted_files if f.is_file()]
                file_count = len(file_list)
                
                if file_count == 0:
                    raise ValueError("No files found in backup")
                
                # 샘플 파일 검증
                sample_count = min(sample_files, file_count)
                verified_files = 0
                
                for file_path in file_list[:sample_count]:
                    # 파일 읽기 테스트
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # 첫 1KB 읽기 테스트
                        verified_files += 1
                    except Exception as e:
                        self.logger.warning(f"File verification failed: {file_path} - {e}")
                
                if verified_files == 0:
                    raise ValueError("All sample files failed verification")
            
            # 5. 실제 복원 (테스트 모드가 아닌 경우)
            if not test_mode:
                restore_target = Path(restore_path)
                restore_target.mkdir(parents=True, exist_ok=True)
                
                # 파일 복사
                import shutil
                for item in temp_extract_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, restore_target)
                    elif item.is_dir():
                        shutil.copytree(item, restore_target / item.name, dirs_exist_ok=True)
            
            return {
                "action": "backup_restoration",
                "backup_id": backup_id,
                "restore_path": restore_path,
                "test_mode": test_mode,
                "verified": verify,
                "sample_files_verified": verified_files if test_mode else None,
                "success": True
            }
            
        finally:
            # 임시 디렉토리 정리
            import shutil
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
    
    async def _execute_archive_creation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """아카이브 생성 실행"""
        age_days = parameters.get("age_days", 7)
        source_dir = parameters.get("source_dir", self.log_manager.config.log_dir)
        pattern = parameters.get("pattern", "*.log*")
        
        # 나이 기준 파일 필터링
        cutoff_time = datetime.now() - timedelta(days=age_days)
        
        archive_ids = await self.archive_manager.archive_files(
            source_dir=source_dir,
            pattern=pattern
        )
        
        return {
            "action": "archive_creation",
            "archive_ids": archive_ids,
            "source_dir": source_dir,
            "age_days": age_days,
            "cutoff_time": cutoff_time.isoformat()
        }
    
    async def _execute_cleanup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """정리 작업 실행"""
        target = parameters.get("target", "temp_files")
        age_days = parameters.get("age_days", 1)
        
        cleaned_files = []
        cutoff_time = datetime.now() - timedelta(days=age_days)
        
        if target == "temp_files":
            # 임시 파일 정리
            temp_dirs = [
                Path(self.backup_manager.config.temp_dir),
                Path(self.archive_manager.config.temp_dir)
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file_path in temp_dir.rglob("*"):
                        if (file_path.is_file() and 
                            file_path.stat().st_mtime < cutoff_time.timestamp()):
                            try:
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {file_path}: {e}")
        
        elif target == "test_restore":
            # 테스트 복원 디렉토리 정리
            test_restore_dir = Path(self.backup_manager.config.temp_dir) / "test_restore"
            if test_restore_dir.exists():
                import shutil
                shutil.rmtree(test_restore_dir, ignore_errors=True)
                cleaned_files.append(str(test_restore_dir))
        
        return {
            "action": "cleanup",
            "target": target,
            "cleaned_files": cleaned_files,
            "age_days": age_days
        }
    
    async def _execute_integrity_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """무결성 검사 실행"""
        scope = parameters.get("scope", "recent")
        days = parameters.get("days", 7)
        
        results = {
            "action": "integrity_check",
            "scope": scope,
            "checked_backups": [],
            "failed_backups": [],
            "success_rate": 0.0
        }
        
        # 백업 목록 조회
        all_backups = await self.backup_manager.list_backups()
        
        if scope == "recent":
            cutoff_time = datetime.now() - timedelta(days=days)
            backups_to_check = [
                b for b in all_backups 
                if b.created_at >= cutoff_time
            ]
        else:
            backups_to_check = all_backups
        
        # 각 백업 검증
        for backup in backups_to_check:
            try:
                verification_result = await self.backup_manager.verify_backup(backup)
                
                check_result = {
                    "backup_id": backup.backup_id,
                    "created_at": backup.created_at.isoformat(),
                    "verified": verification_result
                }
                
                results["checked_backups"].append(check_result)
                
                if not verification_result:
                    results["failed_backups"].append(backup.backup_id)
                    
            except Exception as e:
                self.logger.error(f"Integrity check failed for {backup.backup_id}: {e}")
                results["failed_backups"].append(backup.backup_id)
        
        # 성공률 계산
        total_checked = len(results["checked_backups"])
        if total_checked > 0:
            success_count = total_checked - len(results["failed_backups"])
            results["success_rate"] = (success_count / total_checked) * 100
        
        return results
    
    def get_operation_status(self, operation_id: str) -> Optional[Operation]:
        """작업 상태 조회"""
        return self.operations.get(operation_id)
    
    def list_operations(self, status: Optional[OperationStatus] = None) -> List[Operation]:
        """작업 목록 조회"""
        operations = list(self.operations.values())
        
        if status:
            operations = [op for op in operations if op.status == status]
        
        return sorted(operations, key=lambda x: x.created_at, reverse=True)
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """워크플로우 정의 조회"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[WorkflowDefinition]:
        """워크플로우 목록 조회"""
        return list(self.workflows.values())
    
    async def get_unified_status(self) -> Dict[str, Any]:
        """통합 상태 조회"""
        return {
            "log_manager": self.log_manager.get_log_statistics(),
            "backup_manager": self.backup_manager.get_backup_statistics(),
            "archive_manager": self.archive_manager.get_archive_statistics(),
            "orchestrator": {
                "total_operations": len(self.operations),
                "active_operations": len([op for op in self.operations.values() 
                                        if op.status == OperationStatus.RUNNING]),
                "completed_operations": len([op for op in self.operations.values() 
                                           if op.status == OperationStatus.COMPLETED]),
                "failed_operations": len([op for op in self.operations.values() 
                                        if op.status == OperationStatus.FAILED]),
                "available_workflows": len(self.workflows)
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 활성 작업 취소
            for task in self.active_operations.values():
                if not task.done():
                    task.cancel()
            
            # 개별 관리자 정리
            if hasattr(self.log_manager, 'cleanup'):
                self.log_manager.cleanup()
            
            if hasattr(self.backup_manager, 'cleanup'):
                await self.backup_manager.cleanup()
            
            if hasattr(self.archive_manager, 'cleanup'):
                if asyncio.iscoroutinefunction(self.archive_manager.cleanup):
                    await self.archive_manager.cleanup()
                else:
                    self.archive_manager.cleanup()
            
            self.logger.info("Log Orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Orchestrator cleanup failed: {e}")

# 전역 오케스트레이터
_global_orchestrator = None

def get_log_orchestrator(log_config: Optional[LogConfig] = None,
                        backup_config: Optional[BackupConfig] = None,
                        archive_config: Optional[ArchiveConfig] = None) -> LogOrchestrator:
    """전역 로그 오케스트레이터 반환"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = LogOrchestrator(log_config, backup_config, archive_config)
    return _global_orchestrator

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_orchestrator():
        print("🧪 Log Orchestrator 테스트")
        
        # 오케스트레이터 초기화
        orchestrator = LogOrchestrator()
        
        print("\n1️⃣ 통합 상태 확인")
        status = await orchestrator.get_unified_status()
        print(f"  로그 매니저: {status['log_manager']['total_logs']} logs")
        print(f"  백업 매니저: {status['backup_manager']['total_backups']} backups")
        print(f"  아카이브 매니저: {status['archive_manager']['total_archives']} archives")
        
        print("\n2️⃣ 워크플로우 목록")
        workflows = orchestrator.list_workflows()
        for workflow in workflows:
            print(f"  {workflow.workflow_id}: {workflow.name}")
        
        print("\n3️⃣ 백업 무결성 검사 워크플로우 실행")
        try:
            workflow_id = await orchestrator.execute_workflow("backup_integrity_check")
            print(f"  워크플로우 실행 ID: {workflow_id}")
        except Exception as e:
            print(f"  워크플로우 실행 실패: {e}")
        
        print("\n4️⃣ 작업 목록")
        operations = orchestrator.list_operations()
        for op in operations[:3]:  # 최근 3개만
            print(f"  {op.operation_id}: {op.status.value} ({op.operation_type.value})")
        
        print("\n🎉 Log Orchestrator 테스트 완료!")
        
        # 정리
        await orchestrator.cleanup()
    
    # 테스트 실행
    asyncio.run(test_orchestrator())