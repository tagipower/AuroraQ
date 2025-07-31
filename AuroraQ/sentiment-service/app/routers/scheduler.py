#!/usr/bin/env python3
"""
Scheduler Router for AuroraQ Sentiment Service
배치 스케줄러 관리 API 엔드포인트
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import structlog

from ..dependencies import get_batch_scheduler
from ...schedulers.batch_scheduler import ScheduleType, TaskPriority, TaskStatus
from ...models import SuccessResponse, ErrorResponse

logger = structlog.get_logger(__name__)

router = APIRouter()

# Request/Response Models
class TaskScheduleRequest(BaseModel):
    """작업 스케줄링 요청"""
    task_id: str = Field(..., description="작업 ID")
    name: str = Field(..., description="작업 이름")
    schedule_type: ScheduleType = Field(..., description="스케줄 유형")
    schedule_config: Dict[str, Any] = Field(..., description="스케줄 설정")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="우선순위")
    max_runtime: int = Field(3600, description="최대 실행 시간(초)")
    retry_count: int = Field(3, description="재시도 횟수")

class TaskStatusResponse(BaseModel):
    """작업 상태 응답"""
    task_id: str
    name: str
    status: str
    priority: str
    last_run: Optional[str]
    next_run: Optional[str]
    run_count: int
    success_count: int
    error_count: int
    success_rate: float
    avg_runtime: float
    is_running: bool

class SchedulerStatsResponse(BaseModel):
    """스케줄러 통계 응답"""
    success: bool = True
    stats: Dict[str, Any]
    task_details: Dict[str, TaskStatusResponse]

class TaskListResponse(BaseModel):
    """작업 목록 응답"""
    success: bool = True
    tasks: List[TaskStatusResponse]
    total_count: int
    running_count: int
    metadata: Dict[str, Any]

# API Endpoints

@router.get("/stats", response_model=SchedulerStatsResponse)
async def get_scheduler_stats(scheduler=Depends(get_batch_scheduler)):
    """
    스케줄러 통계 조회
    
    배치 스케줄러의 현재 상태와 통계를 반환합니다.
    """
    try:
        logger.info("Retrieving scheduler stats")
        
        # 스케줄러 통계 조회
        stats = scheduler.get_scheduler_stats()
        
        # 작업 상세 정보를 TaskStatusResponse로 변환
        task_details = {}
        for task_id, task_info in stats.get("task_details", {}).items():
            if task_info:
                task_details[task_id] = TaskStatusResponse(**task_info)
        
        # task_details를 stats에서 제거 (중복 방지)
        cleaned_stats = {k: v for k, v in stats.items() if k != "task_details"}
        
        logger.info("Scheduler stats retrieved",
                   scheduled_tasks=cleaned_stats.get("scheduled_tasks_count", 0),
                   running_tasks=cleaned_stats.get("running_tasks_count", 0))
        
        return SchedulerStatsResponse(
            stats=cleaned_stats,
            task_details=task_details
        )
        
    except Exception as e:
        logger.error("Failed to retrieve scheduler stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve scheduler stats: {str(e)}"
        )

@router.get("/tasks", response_model=TaskListResponse)
async def get_scheduled_tasks(
    status_filter: Optional[TaskStatus] = Query(None, description="상태별 필터링"),
    priority_filter: Optional[TaskPriority] = Query(None, description="우선순위별 필터링"),
    scheduler=Depends(get_batch_scheduler)
):
    """
    예약된 작업 목록 조회
    
    현재 스케줄링된 모든 작업들의 목록을 반환합니다.
    """
    try:
        logger.info("Retrieving scheduled tasks",
                   status_filter=status_filter.value if status_filter else None,
                   priority_filter=priority_filter.value if priority_filter else None)
        
        stats = scheduler.get_scheduler_stats()
        task_details = stats.get("task_details", {})
        
        # 작업 목록 생성
        tasks = []
        running_count = 0
        
        for task_id, task_info in task_details.items():
            if not task_info:
                continue
            
            # 필터링
            if status_filter and task_info.get("status") != status_filter.value:
                continue
            
            if priority_filter and task_info.get("priority") != priority_filter.value:
                continue
            
            task_response = TaskStatusResponse(**task_info)
            tasks.append(task_response)
            
            if task_response.is_running:
                running_count += 1
        
        logger.info("Scheduled tasks retrieved",
                   total_count=len(tasks),
                   running_count=running_count)
        
        return TaskListResponse(
            tasks=tasks,
            total_count=len(tasks),
            running_count=running_count,
            metadata={
                "status_filter": status_filter.value if status_filter else None,
                "priority_filter": priority_filter.value if priority_filter else None,
                "scheduler_uptime": stats.get("scheduler_uptime", 0)
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve scheduled tasks", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve scheduled tasks: {str(e)}"
        )

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    scheduler=Depends(get_batch_scheduler)
):
    """
    특정 작업 상태 조회
    
    지정된 작업의 상세 상태를 반환합니다.
    """
    try:
        logger.info("Retrieving task status", task_id=task_id)
        
        task_status = scheduler.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        logger.info("Task status retrieved",
                   task_id=task_id,
                   status=task_status.get("status"),
                   run_count=task_status.get("run_count", 0))
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve task status",
                    task_id=task_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task status: {str(e)}"
        )

@router.post("/tasks/{task_id}/pause")
async def pause_task(
    task_id: str,
    scheduler=Depends(get_batch_scheduler)
):
    """
    작업 일시 정지
    
    지정된 작업을 일시 정지합니다.
    """
    try:
        logger.info("Pausing task", task_id=task_id)
        
        success = await scheduler.pause_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to pause task: {task_id}"
            )
        
        logger.info("Task paused successfully", task_id=task_id)
        
        return SuccessResponse(
            success=True,
            message=f"Task paused: {task_id}",
            data={"task_id": task_id, "action": "paused"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause task",
                    task_id=task_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause task: {str(e)}"
        )

@router.post("/tasks/{task_id}/resume")
async def resume_task(
    task_id: str,
    scheduler=Depends(get_batch_scheduler)
):
    """
    작업 재개
    
    일시 정지된 작업을 재개합니다.
    """
    try:
        logger.info("Resuming task", task_id=task_id)
        
        success = await scheduler.resume_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to resume task: {task_id}"
            )
        
        logger.info("Task resumed successfully", task_id=task_id)
        
        return SuccessResponse(
            success=True,
            message=f"Task resumed: {task_id}",
            data={"task_id": task_id, "action": "resumed"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume task",
                    task_id=task_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume task: {str(e)}"
        )

@router.delete("/tasks/{task_id}")
async def remove_task(
    task_id: str,
    scheduler=Depends(get_batch_scheduler)
):
    """
    작업 제거
    
    스케줄에서 작업을 완전히 제거합니다.
    """
    try:
        logger.info("Removing task", task_id=task_id)
        
        success = await scheduler.remove_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to remove task: {task_id}"
            )
        
        logger.info("Task removed successfully", task_id=task_id)
        
        return SuccessResponse(
            success=True,
            message=f"Task removed: {task_id}",
            data={"task_id": task_id, "action": "removed"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove task",
                    task_id=task_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove task: {str(e)}"
        )

@router.get("/resource-monitor")
async def get_resource_status(scheduler=Depends(get_batch_scheduler)):
    """
    리소스 모니터링 상태
    
    현재 시스템 리소스 사용량과 부하 상태를 반환합니다.
    """
    try:
        logger.info("Retrieving resource monitor status")
        
        resource_load = scheduler.resource_monitor.get_current_load()
        should_throttle = scheduler.resource_monitor.should_throttle()
        
        # CPU, 메모리 히스토리 (최근 10분)
        cpu_history = list(scheduler.resource_monitor.cpu_history)[-10:]
        memory_history = list(scheduler.resource_monitor.memory_history)[-10:]
        
        logger.info("Resource monitor status retrieved",
                   cpu_load=resource_load.get("cpu", 0),
                   memory_load=resource_load.get("memory", 0),
                   should_throttle=should_throttle)
        
        return SuccessResponse(
            success=True,
            message="Resource monitor status",
            data={
                "current_load": resource_load,
                "should_throttle": should_throttle,
                "throttle_thresholds": {
                    "cpu_percent": 80.0,
                    "memory_mb": 1024.0,
                    "max_concurrent_tasks": 5
                },
                "history": {
                    "cpu": cpu_history,
                    "memory": memory_history
                },
                "running_tasks": scheduler.resource_monitor.running_tasks
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve resource monitor status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve resource monitor status: {str(e)}"
        )

@router.get("/schedule-types")
async def get_schedule_types():
    """
    스케줄 유형 및 옵션
    
    사용 가능한 스케줄 유형과 우선순위 옵션을 반환합니다.
    """
    try:
        schedule_types = []
        for schedule_type in ScheduleType:
            schedule_types.append({
                "type": schedule_type.value,
                "name": schedule_type.name,
                "description": f"Schedule type: {schedule_type.value}"
            })
        
        task_priorities = []
        for priority in TaskPriority:
            task_priorities.append({
                "priority": priority.value,
                "name": priority.name,
                "description": f"Priority level: {priority.value}"
            })
        
        task_statuses = []
        for status in TaskStatus:
            task_statuses.append({
                "status": status.value,
                "name": status.name,
                "description": f"Task status: {status.value}"
            })
        
        return SuccessResponse(
            success=True,
            message="Schedule types and options",
            data={
                "schedule_types": schedule_types,
                "task_priorities": task_priorities,
                "task_statuses": task_statuses,
                "schedule_config_examples": {
                    "interval": {
                        "minutes": 5,
                        "hours": 1,
                        "description": "Run every N minutes/hours"
                    },
                    "cron": {
                        "hour": 2,
                        "minute": 0,
                        "description": "Run at specific time daily"
                    }
                }
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve schedule types", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schedule types: {str(e)}"
        )

@router.post("/maintenance")
async def trigger_maintenance(scheduler=Depends(get_batch_scheduler)):
    """
    수동 유지보수 실행
    
    시스템 유지보수 작업을 수동으로 실행합니다.
    """
    try:
        logger.info("Triggering manual maintenance")
        
        # 유지보수 작업 수동 실행
        maintenance_task = scheduler.scheduled_tasks.get("system_maintenance")
        
        if not maintenance_task:
            raise HTTPException(
                status_code=404,
                detail="Maintenance task not found"
            )
        
        # 작업 실행 (백그라운드에서)
        asyncio.create_task(
            scheduler._execute_task(maintenance_task, {})
        )
        
        logger.info("Manual maintenance triggered")
        
        return SuccessResponse(
            success=True,
            message="System maintenance triggered",
            data={
                "task_id": "system_maintenance",
                "triggered_at": datetime.now().isoformat(),
                "execution_mode": "manual"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to trigger maintenance", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger maintenance: {str(e)}"
        )