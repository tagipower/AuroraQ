#!/usr/bin/env python3
"""
Automated Recovery System for AuroraQ
AuroraQ 자동화된 복구 시스템 - 수동 개입 없는 자동 복구 및 지능형 장애 예측
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics
from collections import defaultdict, deque
import aiohttp
import subprocess
import psutil

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    """복구 액션 유형"""
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION = "reset_connection"
    SCALE_RESOURCES = "scale_resources"
    SWITCH_ENDPOINT = "switch_endpoint"
    ROLLBACK_CONFIG = "rollback_config"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    HEALTH_CHECK = "health_check"

class FailurePattern(Enum):
    """장애 패턴 유형"""
    MEMORY_LEAK = "memory_leak"
    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DISK_SPACE_LOW = "disk_space_low"
    HIGH_CPU_USAGE = "high_cpu_usage"
    NETWORK_LATENCY = "network_latency"
    DATABASE_TIMEOUT = "database_timeout"
    API_DEGRADATION = "api_degradation"

class RecoveryStatus(Enum):
    """복구 상태"""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class FailureEvent:
    """장애 이벤트"""
    id: str
    component: str
    failure_type: str
    severity: str
    timestamp: datetime
    description: str
    metrics: Dict[str, Any]
    pattern: Optional[FailurePattern] = None
    predicted: bool = False

@dataclass
class RecoveryPlan:
    """복구 계획"""
    failure_event: FailureEvent
    actions: List[RecoveryAction]
    estimated_time: int  # 예상 복구 시간 (초)
    success_probability: float
    rollback_plan: Optional[List[RecoveryAction]] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class RecoveryResult:
    """복구 결과"""
    plan: RecoveryPlan
    status: RecoveryStatus
    executed_actions: List[RecoveryAction]
    execution_time: float
    success_metrics: Dict[str, Any]
    error_message: Optional[str] = None

class AutomatedRecoverySystem:
    """자동화된 복구 시스템"""
    
    def __init__(self,
                 prediction_window: int = 300,  # 5분 예측 윈도우
                 max_concurrent_recoveries: int = 3,
                 recovery_timeout: int = 300):  # 5분 복구 타임아웃
        """
        초기화
        
        Args:
            prediction_window: 장애 예측 윈도우 (초)
            max_concurrent_recoveries: 최대 동시 복구 수
            recovery_timeout: 복구 타임아웃 (초)
        """
        self.prediction_window = prediction_window
        self.max_concurrent_recoveries = max_concurrent_recoveries
        self.recovery_timeout = recovery_timeout
        
        # 데이터 저장소
        self.failure_history: deque = deque(maxlen=1000)
        self.recovery_history: deque = deque(maxlen=500)
        self.active_recoveries: Dict[str, RecoveryResult] = {}
        
        # 패턴 학습 데이터
        self.failure_patterns: Dict[FailurePattern, Dict[str, Any]] = {}
        self.component_health: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.recovery_success_rates: Dict[RecoveryAction, float] = defaultdict(float)
        
        # 예측 모델 파라미터
        self.prediction_thresholds = {
            FailurePattern.MEMORY_LEAK: {
                "memory_growth_rate": 5.0,  # MB/분
                "memory_threshold": 85.0    # 사용률 %
            },
            FailurePattern.CONNECTION_POOL_EXHAUSTION: {
                "connection_usage": 90.0,   # %
                "connection_growth_rate": 10.0  # 연결/분
            },
            FailurePattern.HIGH_CPU_USAGE: {
                "cpu_threshold": 85.0,      # %
                "duration_threshold": 300   # 초
            },
            FailurePattern.DISK_SPACE_LOW: {
                "disk_threshold": 90.0,     # %
                "growth_rate": 1.0          # GB/시간
            }
        }
        
        # 복구 전략 매핑
        self.recovery_strategies = {
            FailurePattern.MEMORY_LEAK: [
                RecoveryAction.RESTART_SERVICE,
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.SCALE_RESOURCES
            ],
            FailurePattern.CONNECTION_POOL_EXHAUSTION: [
                RecoveryAction.RESET_CONNECTION,
                RecoveryAction.RESTART_SERVICE,
                RecoveryAction.SWITCH_ENDPOINT
            ],
            FailurePattern.HIGH_CPU_USAGE: [
                RecoveryAction.SCALE_RESOURCES,
                RecoveryAction.RESTART_SERVICE,
                RecoveryAction.EMERGENCY_SHUTDOWN
            ],
            FailurePattern.DISK_SPACE_LOW: [
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.SCALE_RESOURCES
            ],
            FailurePattern.RATE_LIMIT_EXCEEDED: [
                RecoveryAction.SWITCH_ENDPOINT,
                RecoveryAction.SCALE_RESOURCES
            ]
        }
        
        # 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 통계
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "prevented_failures": 0,
            "avg_recovery_time": 0.0
        }
        
        logger.info(f"Automated Recovery System initialized with {prediction_window}s prediction window")
    
    async def predict_failures(self) -> List[FailureEvent]:
        """장애 예측"""
        predicted_failures = []
        
        try:
            # 시스템 메트릭 수집
            system_metrics = await self._collect_system_metrics()
            
            # 각 패턴별 예측 수행
            for pattern, thresholds in self.prediction_thresholds.items():
                risk_score = await self._calculate_failure_risk(pattern, system_metrics)
                
                if risk_score > 0.7:  # 70% 이상 위험도
                    failure_event = FailureEvent(
                        id=f"predicted_{pattern.value}_{int(time.time())}",
                        component=self._get_component_for_pattern(pattern),
                        failure_type=pattern.value,
                        severity="predicted",
                        timestamp=datetime.now(),
                        description=f"예측된 {pattern.value} 장애 (위험도: {risk_score:.1%})",
                        metrics=system_metrics,
                        pattern=pattern,
                        predicted=True
                    )
                    
                    predicted_failures.append(failure_event)
                    self.stats["total_predictions"] += 1
                    
                    logger.warning(f"Predicted failure: {pattern.value} (risk: {risk_score:.1%})")
            
            return predicted_failures
            
        except Exception as e:
            logger.error(f"Error predicting failures: {e}")
            return []
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 메트릭
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 메모리 메트릭
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # 디스크 메트릭
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB
            
            # 네트워크 메트릭
            network = psutil.net_io_counters()
            
            # 프로세스 메트릭
            process_count = len(psutil.pids())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                "memory": {
                    "percent": memory_percent,
                    "available_gb": memory_available,
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3)
                },
                "disk": {
                    "percent": disk_percent,
                    "free_gb": disk_free,
                    "total_gb": disk.total / (1024**3)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": {
                    "count": process_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def _calculate_failure_risk(self, pattern: FailurePattern, metrics: Dict[str, Any]) -> float:
        """장애 위험도 계산"""
        if pattern not in self.prediction_thresholds:
            return 0.0
        
        thresholds = self.prediction_thresholds[pattern]
        risk_score = 0.0
        
        try:
            if pattern == FailurePattern.MEMORY_LEAK:
                memory_percent = metrics.get("memory", {}).get("percent", 0)
                memory_threshold = thresholds["memory_threshold"]
                
                if memory_percent > memory_threshold:
                    risk_score = min(1.0, (memory_percent - memory_threshold) / (100 - memory_threshold))
                
                # 메모리 증가율 추가 고려 (히스토리 기반)
                risk_score += self._get_trend_risk("memory_usage", memory_percent) * 0.3
            
            elif pattern == FailurePattern.HIGH_CPU_USAGE:
                cpu_percent = metrics.get("cpu", {}).get("percent", 0)
                cpu_threshold = thresholds["cpu_threshold"]
                
                if cpu_percent > cpu_threshold:
                    risk_score = min(1.0, (cpu_percent - cpu_threshold) / (100 - cpu_threshold))
                
                # CPU 사용률 지속 시간 고려
                risk_score += self._get_duration_risk("cpu_usage", cpu_percent, cpu_threshold) * 0.4
            
            elif pattern == FailurePattern.DISK_SPACE_LOW:
                disk_percent = metrics.get("disk", {}).get("percent", 0)
                disk_threshold = thresholds["disk_threshold"]
                
                if disk_percent > disk_threshold:
                    risk_score = min(1.0, (disk_percent - disk_threshold) / (100 - disk_threshold))
            
            # 과거 패턴 기반 위험도 조정
            risk_score *= self._get_historical_risk_multiplier(pattern)
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating failure risk for {pattern}: {e}")
            return 0.0
    
    def _get_trend_risk(self, metric_name: str, current_value: float) -> float:
        """트렌드 기반 위험도 계산"""
        # 간단한 트렌드 분석 (실제로는 더 정교한 분석 필요)
        if metric_name in self.component_health:
            history = self.component_health[metric_name].get("history", [])
            if len(history) >= 3:
                recent_values = history[-3:]
                if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
                    return 0.5  # 증가 트렌드 감지
        return 0.0
    
    def _get_duration_risk(self, metric_name: str, current_value: float, threshold: float) -> float:
        """지속 시간 기반 위험도 계산"""
        if current_value <= threshold:
            return 0.0
        
        # 임계값 초과 지속 시간 추적
        duration_key = f"{metric_name}_duration"
        if duration_key not in self.component_health:
            self.component_health[duration_key] = {"start_time": time.time()}
            return 0.0
        
        duration = time.time() - self.component_health[duration_key]["start_time"]
        return min(1.0, duration / 300)  # 5분 기준으로 정규화
    
    def _get_historical_risk_multiplier(self, pattern: FailurePattern) -> float:
        """과거 패턴 기반 위험도 승수"""
        if pattern not in self.failure_patterns:
            return 1.0
        
        pattern_data = self.failure_patterns[pattern]
        recent_failures = pattern_data.get("recent_count", 0)
        
        # 최근 장애가 많을수록 위험도 증가
        if recent_failures > 3:
            return 1.5
        elif recent_failures > 1:
            return 1.2
        
        return 1.0
    
    def _get_component_for_pattern(self, pattern: FailurePattern) -> str:
        """패턴에 해당하는 컴포넌트 반환"""
        mapping = {
            FailurePattern.MEMORY_LEAK: "system",
            FailurePattern.CONNECTION_POOL_EXHAUSTION: "network",
            FailurePattern.HIGH_CPU_USAGE: "system",
            FailurePattern.DISK_SPACE_LOW: "storage",
            FailurePattern.RATE_LIMIT_EXCEEDED: "api"
        }
        return mapping.get(pattern, "unknown")
    
    async def create_recovery_plan(self, failure_event: FailureEvent) -> RecoveryPlan:
        """복구 계획 생성"""
        
        # 패턴 기반 복구 액션 선택
        if failure_event.pattern and failure_event.pattern in self.recovery_strategies:
            actions = self.recovery_strategies[failure_event.pattern].copy()
        else:
            # 기본 복구 액션
            actions = [RecoveryAction.HEALTH_CHECK, RecoveryAction.RESTART_SERVICE]
        
        # 성공률 기반 액션 정렬
        actions.sort(key=lambda a: self.recovery_success_rates.get(a, 0.5), reverse=True)
        
        # 예상 시간 계산
        estimated_time = self._estimate_recovery_time(actions)
        
        # 성공 확률 계산
        success_probability = self._calculate_success_probability(actions, failure_event)
        
        # 롤백 계획 생성
        rollback_plan = self._create_rollback_plan(actions)
        
        return RecoveryPlan(
            failure_event=failure_event,
            actions=actions,
            estimated_time=estimated_time,
            success_probability=success_probability,
            rollback_plan=rollback_plan,
            dependencies=self._get_dependencies(failure_event.component)
        )
    
    def _estimate_recovery_time(self, actions: List[RecoveryAction]) -> int:
        """복구 시간 추정"""
        time_estimates = {
            RecoveryAction.HEALTH_CHECK: 10,
            RecoveryAction.CLEAR_CACHE: 30,
            RecoveryAction.RESET_CONNECTION: 20,
            RecoveryAction.RESTART_SERVICE: 60,
            RecoveryAction.SCALE_RESOURCES: 120,
            RecoveryAction.SWITCH_ENDPOINT: 15,
            RecoveryAction.ROLLBACK_CONFIG: 45,
            RecoveryAction.EMERGENCY_SHUTDOWN: 30
        }
        
        return sum(time_estimates.get(action, 60) for action in actions)
    
    def _calculate_success_probability(self, actions: List[RecoveryAction], failure_event: FailureEvent) -> float:
        """성공 확률 계산"""
        base_probability = 0.8  # 기본 성공률
        
        # 액션별 성공률 고려
        action_probabilities = [self.recovery_success_rates.get(action, 0.7) for action in actions]
        combined_probability = 1.0 - (1.0 - base_probability)
        for prob in action_probabilities:
            combined_probability *= prob
        
        # 장애 심각도 고려
        severity_multiplier = {
            "low": 1.0,
            "medium": 0.9,
            "high": 0.8,
            "critical": 0.7,
            "predicted": 0.95  # 예측된 장애는 더 높은 성공률
        }
        
        multiplier = severity_multiplier.get(failure_event.severity, 0.8)
        return min(1.0, combined_probability * multiplier)
    
    def _create_rollback_plan(self, actions: List[RecoveryAction]) -> List[RecoveryAction]:
        """롤백 계획 생성"""
        rollback_actions = []
        
        for action in actions:
            if action == RecoveryAction.RESTART_SERVICE:
                rollback_actions.append(RecoveryAction.HEALTH_CHECK)
            elif action == RecoveryAction.ROLLBACK_CONFIG:
                continue  # 롤백의 롤백은 없음
            elif action == RecoveryAction.SCALE_RESOURCES:
                rollback_actions.append(RecoveryAction.SCALE_RESOURCES)  # 원래 스케일로 복구
        
        return rollback_actions
    
    def _get_dependencies(self, component: str) -> List[str]:
        """컴포넌트 의존성 반환"""
        dependency_map = {
            "news_collector": ["network", "api"],
            "sentiment_analyzer": ["model", "cache"],
            "topic_classifier": ["model"],
            "strategy_selector": ["database", "cache"],
            "system": []
        }
        return dependency_map.get(component, [])
    
    async def execute_recovery(self, plan: RecoveryPlan) -> RecoveryResult:
        """복구 실행"""
        if len(self.active_recoveries) >= self.max_concurrent_recoveries:
            logger.warning("Maximum concurrent recoveries reached, queuing recovery")
            await asyncio.sleep(10)  # 잠시 대기
        
        recovery_id = f"{plan.failure_event.id}_recovery"
        start_time = time.time()
        
        result = RecoveryResult(
            plan=plan,
            status=RecoveryStatus.IN_PROGRESS,
            executed_actions=[],
            execution_time=0,
            success_metrics={}
        )
        
        self.active_recoveries[recovery_id] = result
        
        try:
            logger.info(f"Starting recovery for {plan.failure_event.component}: {plan.failure_event.description}")
            
            # 각 액션 순차 실행
            for action in plan.actions:
                try:
                    await self._execute_action(action, plan.failure_event)
                    result.executed_actions.append(action)
                    
                    # 액션 후 건강성 체크
                    health_score = await self._check_component_health(plan.failure_event.component)
                    if health_score > 0.8:  # 건강 상태 회복
                        logger.info(f"Recovery successful after action: {action.value}")
                        break
                    
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
                    result.error_message = str(e)
                    
                    # 롤백 계획 실행
                    if plan.rollback_plan:
                        await self._execute_rollback(plan.rollback_plan, plan.failure_event)
                    break
            
            # 최종 건강성 체크
            final_health = await self._check_component_health(plan.failure_event.component)
            result.success_metrics["final_health_score"] = final_health
            
            if final_health > 0.8:
                result.status = RecoveryStatus.SUCCESS
                self.stats["successful_recoveries"] += 1
                
                # 성공률 업데이트
                for action in result.executed_actions:
                    current_rate = self.recovery_success_rates.get(action, 0.5)
                    self.recovery_success_rates[action] = min(1.0, current_rate + 0.1)
                    
                if plan.failure_event.predicted:
                    self.stats["prevented_failures"] += 1
            else:
                result.status = RecoveryStatus.FAILED
                
                # 실패율 업데이트
                for action in result.executed_actions:
                    current_rate = self.recovery_success_rates.get(action, 0.5)
                    self.recovery_success_rates[action] = max(0.1, current_rate - 0.05)
            
        except asyncio.TimeoutError:
            result.status = RecoveryStatus.TIMEOUT
            result.error_message = "Recovery timeout"
        except Exception as e:
            result.status = RecoveryStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Recovery execution failed: {e}")
        
        finally:
            result.execution_time = time.time() - start_time
            self.recovery_history.append(result)
            
            if recovery_id in self.active_recoveries:
                del self.active_recoveries[recovery_id]
            
            self.stats["total_recoveries"] += 1
            
            # 평균 복구 시간 업데이트
            total_time = sum(r.execution_time for r in self.recovery_history)
            self.stats["avg_recovery_time"] = total_time / len(self.recovery_history)
        
        logger.info(f"Recovery completed: {result.status.value} in {result.execution_time:.1f}s")
        return result
    
    async def _execute_action(self, action: RecoveryAction, failure_event: FailureEvent):
        """개별 복구 액션 실행"""
        logger.info(f"Executing recovery action: {action.value}")
        
        if action == RecoveryAction.HEALTH_CHECK:
            await self._perform_health_check(failure_event.component)
        
        elif action == RecoveryAction.CLEAR_CACHE:
            await self._clear_cache(failure_event.component)
        
        elif action == RecoveryAction.RESET_CONNECTION:
            await self._reset_connections(failure_event.component)
        
        elif action == RecoveryAction.RESTART_SERVICE:
            await self._restart_service(failure_event.component)
        
        elif action == RecoveryAction.SCALE_RESOURCES:
            await self._scale_resources(failure_event.component)
        
        elif action == RecoveryAction.SWITCH_ENDPOINT:
            await self._switch_endpoint(failure_event.component)
        
        elif action == RecoveryAction.ROLLBACK_CONFIG:
            await self._rollback_config(failure_event.component)
        
        elif action == RecoveryAction.EMERGENCY_SHUTDOWN:
            await self._emergency_shutdown(failure_event.component)
        
        else:
            logger.warning(f"Unknown recovery action: {action}")
    
    async def _perform_health_check(self, component: str):
        """건강성 체크 수행"""
        await asyncio.sleep(1)  # 건강성 체크 시뮬레이션
        logger.info(f"Health check completed for {component}")
    
    async def _clear_cache(self, component: str):
        """캐시 정리"""
        await asyncio.sleep(2)  # 캐시 정리 시뮬레이션
        logger.info(f"Cache cleared for {component}")
    
    async def _reset_connections(self, component: str):
        """연결 리셋"""
        await asyncio.sleep(3)  # 연결 리셋 시뮬레이션
        logger.info(f"Connections reset for {component}")
    
    async def _restart_service(self, component: str):
        """서비스 재시작"""
        await asyncio.sleep(10)  # 서비스 재시작 시뮬레이션
        logger.info(f"Service restarted for {component}")
    
    async def _scale_resources(self, component: str):
        """리소스 스케일링"""
        await asyncio.sleep(15)  # 리소스 스케일링 시뮬레이션
        logger.info(f"Resources scaled for {component}")
    
    async def _switch_endpoint(self, component: str):
        """엔드포인트 전환"""
        await asyncio.sleep(2)  # 엔드포인트 전환 시뮬레이션
        logger.info(f"Endpoint switched for {component}")
    
    async def _rollback_config(self, component: str):
        """설정 롤백"""
        await asyncio.sleep(5)  # 설정 롤백 시뮬레이션
        logger.info(f"Configuration rolled back for {component}")
    
    async def _emergency_shutdown(self, component: str):
        """비상 종료"""
        await asyncio.sleep(3)  # 비상 종료 시뮬레이션
        logger.warning(f"Emergency shutdown executed for {component}")
    
    async def _execute_rollback(self, rollback_plan: List[RecoveryAction], failure_event: FailureEvent):
        """롤백 실행"""
        logger.info("Executing rollback plan")
        
        for action in rollback_plan:
            try:
                await self._execute_action(action, failure_event)
            except Exception as e:
                logger.error(f"Rollback action {action.value} failed: {e}")
    
    async def _check_component_health(self, component: str) -> float:
        """컴포넌트 건강성 체크"""
        # 실제 구현에서는 각 컴포넌트의 실제 상태를 체크
        # 여기서는 시뮬레이션
        await asyncio.sleep(0.5)
        
        # 랜덤 건강성 점수 (실제로는 메트릭 기반)
        import random
        base_score = random.uniform(0.6, 0.95)
        
        # 컴포넌트별 조정
        if component == "system":
            base_score *= 0.9  # 시스템 복구는 더 어려움
        elif component == "network":
            base_score *= 0.95
        
        return base_score
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 예측 및 자동 복구
                asyncio.create_task(self._predictive_recovery_cycle())
                
                # 패턴 학습 업데이트
                self._update_failure_patterns()
                
                time.sleep(60)  # 1분마다 실행
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    async def _predictive_recovery_cycle(self):
        """예측적 복구 사이클"""
        try:
            # 장애 예측
            predicted_failures = await self.predict_failures()
            
            # 예측된 장애에 대한 선제적 복구
            for failure in predicted_failures:
                if failure.pattern in [FailurePattern.MEMORY_LEAK, FailurePattern.HIGH_CPU_USAGE]:
                    # 높은 위험도 패턴에 대해서만 자동 복구
                    plan = await self.create_recovery_plan(failure)
                    if plan.success_probability > 0.8:
                        logger.info(f"Executing preventive recovery for predicted failure: {failure.description}")
                        await self.execute_recovery(plan)
                
        except Exception as e:
            logger.error(f"Predictive recovery cycle error: {e}")
    
    def _update_failure_patterns(self):
        """장애 패턴 학습 업데이트"""
        # 최근 장애 이벤트 분석
        recent_failures = [f for f in self.failure_history if f.timestamp > datetime.now() - timedelta(hours=24)]
        
        # 패턴별 통계 업데이트
        for pattern in FailurePattern:
            pattern_failures = [f for f in recent_failures if f.pattern == pattern]
            
            if pattern not in self.failure_patterns:
                self.failure_patterns[pattern] = {
                    "total_count": 0,
                    "recent_count": 0,
                    "avg_severity": 0.0,
                    "common_components": defaultdict(int)
                }
            
            pattern_data = self.failure_patterns[pattern]
            pattern_data["recent_count"] = len(pattern_failures)
            
            # 컴포넌트별 빈도 업데이트
            for failure in pattern_failures:
                pattern_data["common_components"][failure.component] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "recovery_system_status": "active" if self.monitoring_active else "inactive",
            "active_recoveries": len(self.active_recoveries),
            "failure_patterns_learned": len(self.failure_patterns),
            "statistics": self.stats.copy(),
            "recent_recoveries": len([r for r in self.recovery_history if r.plan.failure_event.timestamp > datetime.now() - timedelta(hours=1)]),
            "success_rate": self.stats["successful_recoveries"] / max(1, self.stats["total_recoveries"]),
            "avg_recovery_time": self.stats["avg_recovery_time"],
            "prediction_accuracy": self.stats["successful_predictions"] / max(1, self.stats["total_predictions"])
        }
    
    def shutdown(self):
        """복구 시스템 종료"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Automated Recovery System shut down")

# 전역 복구 시스템 인스턴스
_recovery_system: Optional[AutomatedRecoverySystem] = None

def get_recovery_system() -> AutomatedRecoverySystem:
    """전역 복구 시스템 반환"""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = AutomatedRecoverySystem()
    return _recovery_system

# 테스트 코드
if __name__ == "__main__":
    async def test_automated_recovery_system():
        """자동화된 복구 시스템 테스트"""
        print("=== Automated Recovery System Test ===\n")
        
        recovery_system = AutomatedRecoverySystem(
            prediction_window=60,  # 1분 예측 윈도우
            max_concurrent_recoveries=2
        )
        
        # 테스트 1: 장애 예측
        print("1. Testing failure prediction...")
        predicted_failures = await recovery_system.predict_failures()
        print(f"   Predicted failures: {len(predicted_failures)}")
        for failure in predicted_failures:
            print(f"   - {failure.pattern.value}: {failure.description}")
        
        # 테스트 2: 복구 계획 생성
        print("\n2. Testing recovery plan creation...")
        test_failure = FailureEvent(
            id="test_failure_001",
            component="news_collector",
            failure_type="connection_timeout",
            severity="high",
            timestamp=datetime.now(),
            description="뉴스 수집기 연결 타임아웃",
            metrics={"response_time": 15.0, "error_rate": 0.8},
            pattern=FailurePattern.CONNECTION_POOL_EXHAUSTION
        )
        
        plan = await recovery_system.create_recovery_plan(test_failure)
        print(f"   Recovery actions: {[a.value for a in plan.actions]}")
        print(f"   Estimated time: {plan.estimated_time}s")
        print(f"   Success probability: {plan.success_probability:.1%}")
        
        # 테스트 3: 복구 실행
        print("\n3. Testing recovery execution...")
        result = await recovery_system.execute_recovery(plan)
        print(f"   Recovery status: {result.status.value}")
        print(f"   Executed actions: {[a.value for a in result.executed_actions]}")
        print(f"   Execution time: {result.execution_time:.1f}s")
        
        # 테스트 4: 시스템 상태
        print("\n4. System status:")
        status = recovery_system.get_system_status()
        print(f"   Total recoveries: {status['statistics']['total_recoveries']}")
        print(f"   Successful recoveries: {status['statistics']['successful_recoveries']}")
        print(f"   Success rate: {status['success_rate']:.1%}")
        print(f"   Prevented failures: {status['statistics']['prevented_failures']}")
        
        recovery_system.shutdown()
        print("\n✅ Automated Recovery System test completed")
    
    # 테스트 실행
    asyncio.run(test_automated_recovery_system())