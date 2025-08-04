#!/usr/bin/env python3
"""
시스템 리소스 관리자
P5: 시스템 리소스 관리 및 최적화
"""

import sys
import os
import psutil
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import gc
import warnings

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """리소스 타입"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class OptimizationAction(Enum):
    """최적화 액션"""
    CLEANUP_MEMORY = "cleanup_memory"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    PAUSE_OPERATIONS = "pause_operations"
    RESTART_SERVICES = "restart_services"
    SCALE_DOWN = "scale_down"
    ALERT_ADMIN = "alert_admin"

@dataclass
class ResourceThreshold:
    """리소스 임계값"""
    warning_level: float = 70.0      # 경고 레벨 (%)
    critical_level: float = 85.0     # 위험 레벨 (%)
    emergency_level: float = 95.0    # 응급 레벨 (%)
    target_level: float = 60.0       # 목표 레벨 (%)

@dataclass
class ResourceMetrics:
    """리소스 메트릭"""
    timestamp: datetime
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    file_descriptors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_percent': self.disk_percent,
            'disk_used_gb': self.disk_used_gb,
            'disk_free_gb': self.disk_free_gb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'process_count': self.process_count,
            'thread_count': self.thread_count,
            'file_descriptors': self.file_descriptors
        }

@dataclass
class ResourceAlert:
    """리소스 알림"""
    timestamp: datetime
    resource_type: ResourceType
    level: AlertLevel
    current_value: float
    threshold_value: float
    message: str
    action_taken: Optional[str] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'resource_type': self.resource_type.value,
            'level': self.level.value,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'action_taken': self.action_taken,
            'resolved': self.resolved
        }

@dataclass
class SystemHealth:
    """시스템 건강도"""
    overall_score: float = 100.0  # 0-100점
    cpu_score: float = 100.0
    memory_score: float = 100.0
    disk_score: float = 100.0
    network_score: float = 100.0
    stability_score: float = 100.0
    status: str = "healthy"
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self):
        """전체 점수 계산"""
        scores = [
            self.cpu_score,
            self.memory_score,
            self.disk_score,
            self.network_score,
            self.stability_score
        ]
        self.overall_score = sum(scores) / len(scores)
        
        # 상태 결정
        if self.overall_score >= 90:
            self.status = "excellent"
        elif self.overall_score >= 75:
            self.status = "good"
        elif self.overall_score >= 60:
            self.status = "fair"
        elif self.overall_score >= 40:
            self.status = "poor"
        else:
            self.status = "critical"

class SystemResourceManager:
    """시스템 리소스 관리자"""
    
    def __init__(self, config_file: str = "resource_manager_config.json"):
        self.config_file = config_file
        
        # 임계값 설정
        self.thresholds = {
            ResourceType.CPU: ResourceThreshold(70.0, 85.0, 95.0, 60.0),
            ResourceType.MEMORY: ResourceThreshold(75.0, 90.0, 98.0, 65.0),
            ResourceType.DISK: ResourceThreshold(80.0, 90.0, 95.0, 70.0),
            ResourceType.NETWORK: ResourceThreshold(70.0, 85.0, 95.0, 60.0)
        }
        
        # 모니터링 데이터
        self.metrics_history: List[ResourceMetrics] = []
        self.alerts_history: List[ResourceAlert] = []
        self.max_history_size = 1000
        
        # 최적화 액션 핸들러
        self.optimization_handlers: Dict[OptimizationAction, Callable] = {}
        self._register_default_handlers()
        
        # 모니터링 제어
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # 통계
        self.stats = {
            "monitoring_start_time": None,
            "total_alerts": 0,
            "critical_alerts": 0,
            "optimizations_performed": 0,
            "memory_cleanups": 0,
            "last_optimization": None
        }
        
        # 설정 로드
        self._load_configuration()
        
        logger.info("System resource manager initialized")
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 임계값 로드
                for resource_name, threshold_config in config.get('thresholds', {}).items():
                    try:
                        resource_type = ResourceType(resource_name)
                        self.thresholds[resource_type] = ResourceThreshold(
                            warning_level=threshold_config.get('warning_level', 70.0),
                            critical_level=threshold_config.get('critical_level', 85.0),
                            emergency_level=threshold_config.get('emergency_level', 95.0),
                            target_level=threshold_config.get('target_level', 60.0)
                        )
                    except ValueError:
                        continue
                
                # 통계 로드
                self.stats.update(config.get('stats', {}))
                
                logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config = {
                'thresholds': {
                    resource.value: {
                        'warning_level': threshold.warning_level,
                        'critical_level': threshold.critical_level,
                        'emergency_level': threshold.emergency_level,
                        'target_level': threshold.target_level
                    }
                    for resource, threshold in self.thresholds.items()
                },
                'stats': self.stats
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _register_default_handlers(self):
        """기본 최적화 핸들러 등록"""
        self.optimization_handlers[OptimizationAction.CLEANUP_MEMORY] = self._cleanup_memory
        self.optimization_handlers[OptimizationAction.REDUCE_BATCH_SIZE] = self._reduce_batch_size
        self.optimization_handlers[OptimizationAction.PAUSE_OPERATIONS] = self._pause_operations
        self.optimization_handlers[OptimizationAction.ALERT_ADMIN] = self._alert_admin
    
    def collect_system_metrics(self) -> ResourceMetrics:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            memory_available_mb = memory.available / 1024 / 1024
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / 1024 / 1024 / 1024
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / 1024 / 1024
            network_recv_mb = network.bytes_recv / 1024 / 1024
            
            # 프로세스 정보
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # 파일 디스크립터 (Unix 계열에서만)
            file_descriptors = 0
            try:
                file_descriptors = current_process.num_fds()
            except (AttributeError, OSError):
                pass
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                thread_count=thread_count,
                file_descriptors=file_descriptors
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return ResourceMetrics(timestamp=datetime.now())
    
    def analyze_metrics(self, metrics: ResourceMetrics) -> List[ResourceAlert]:
        """메트릭 분석 및 알림 생성"""
        alerts = []
        
        try:
            # CPU 분석
            cpu_threshold = self.thresholds[ResourceType.CPU]
            if metrics.cpu_percent >= cpu_threshold.emergency_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.CPU,
                    level=AlertLevel.EMERGENCY,
                    current_value=metrics.cpu_percent,
                    threshold_value=cpu_threshold.emergency_level,
                    message=f"CPU usage extremely high: {metrics.cpu_percent:.1f}%"
                ))
            elif metrics.cpu_percent >= cpu_threshold.critical_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.CPU,
                    level=AlertLevel.CRITICAL,
                    current_value=metrics.cpu_percent,
                    threshold_value=cpu_threshold.critical_level,
                    message=f"CPU usage critical: {metrics.cpu_percent:.1f}%"
                ))
            elif metrics.cpu_percent >= cpu_threshold.warning_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.CPU,
                    level=AlertLevel.WARNING,
                    current_value=metrics.cpu_percent,
                    threshold_value=cpu_threshold.warning_level,
                    message=f"CPU usage high: {metrics.cpu_percent:.1f}%"
                ))
            
            # 메모리 분석
            memory_threshold = self.thresholds[ResourceType.MEMORY]
            if metrics.memory_percent >= memory_threshold.emergency_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.MEMORY,
                    level=AlertLevel.EMERGENCY,
                    current_value=metrics.memory_percent,
                    threshold_value=memory_threshold.emergency_level,
                    message=f"Memory usage extremely high: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f}MB)"
                ))
            elif metrics.memory_percent >= memory_threshold.critical_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.MEMORY,
                    level=AlertLevel.CRITICAL,
                    current_value=metrics.memory_percent,
                    threshold_value=memory_threshold.critical_level,
                    message=f"Memory usage critical: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f}MB)"
                ))
            elif metrics.memory_percent >= memory_threshold.warning_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.MEMORY,
                    level=AlertLevel.WARNING,
                    current_value=metrics.memory_percent,
                    threshold_value=memory_threshold.warning_level,
                    message=f"Memory usage high: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f}MB)"
                ))
            
            # 디스크 분석
            disk_threshold = self.thresholds[ResourceType.DISK]
            if metrics.disk_percent >= disk_threshold.emergency_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.DISK,
                    level=AlertLevel.EMERGENCY,
                    current_value=metrics.disk_percent,
                    threshold_value=disk_threshold.emergency_level,
                    message=f"Disk usage extremely high: {metrics.disk_percent:.1f}% ({metrics.disk_free_gb:.1f}GB free)"
                ))
            elif metrics.disk_percent >= disk_threshold.critical_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.DISK,
                    level=AlertLevel.CRITICAL,
                    current_value=metrics.disk_percent,
                    threshold_value=disk_threshold.critical_level,
                    message=f"Disk usage critical: {metrics.disk_percent:.1f}% ({metrics.disk_free_gb:.1f}GB free)"
                ))
            elif metrics.disk_percent >= disk_threshold.warning_level:
                alerts.append(ResourceAlert(
                    timestamp=metrics.timestamp,
                    resource_type=ResourceType.DISK,
                    level=AlertLevel.WARNING,
                    current_value=metrics.disk_percent,
                    threshold_value=disk_threshold.warning_level,
                    message=f"Disk usage high: {metrics.disk_percent:.1f}% ({metrics.disk_free_gb:.1f}GB free)"
                ))
            
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
        
        return alerts
    
    def process_alerts(self, alerts: List[ResourceAlert]):
        """알림 처리 및 최적화 액션 실행"""
        try:
            for alert in alerts:
                # 알림 히스토리에 추가
                with self._lock:
                    self.alerts_history.append(alert)
                    if len(self.alerts_history) > self.max_history_size:
                        self.alerts_history.pop(0)
                    
                    self.stats["total_alerts"] += 1
                    if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                        self.stats["critical_alerts"] += 1
                
                # 로그 출력
                logger.warning(f"RESOURCE ALERT [{alert.level.value.upper()}]: {alert.message}")
                
                # 자동 최적화 액션 결정 및 실행
                actions = self._determine_optimization_actions(alert)
                for action in actions:
                    try:
                        if action in self.optimization_handlers:
                            success = self.optimization_handlers[action](alert)
                            if success:
                                alert.action_taken = action.value
                                self.stats["optimizations_performed"] += 1
                                self.stats["last_optimization"] = datetime.now().isoformat()
                                logger.info(f"Optimization action executed: {action.value}")
                    except Exception as e:
                        logger.error(f"Failed to execute optimization action {action.value}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
    
    def _determine_optimization_actions(self, alert: ResourceAlert) -> List[OptimizationAction]:
        """최적화 액션 결정"""
        actions = []
        
        try:
            if alert.level == AlertLevel.EMERGENCY:
                # 응급 상황 - 즉시 조치
                if alert.resource_type == ResourceType.MEMORY:
                    actions.extend([
                        OptimizationAction.CLEANUP_MEMORY,
                        OptimizationAction.REDUCE_BATCH_SIZE,
                        OptimizationAction.ALERT_ADMIN
                    ])
                elif alert.resource_type == ResourceType.CPU:
                    actions.extend([
                        OptimizationAction.REDUCE_BATCH_SIZE,
                        OptimizationAction.PAUSE_OPERATIONS,
                        OptimizationAction.ALERT_ADMIN
                    ])
                elif alert.resource_type == ResourceType.DISK:
                    actions.extend([
                        OptimizationAction.CLEANUP_MEMORY,  # 임시 파일 정리
                        OptimizationAction.ALERT_ADMIN
                    ])
                
            elif alert.level == AlertLevel.CRITICAL:
                # 위험 상황 - 적극적 조치
                if alert.resource_type == ResourceType.MEMORY:
                    actions.extend([
                        OptimizationAction.CLEANUP_MEMORY,
                        OptimizationAction.REDUCE_BATCH_SIZE
                    ])
                elif alert.resource_type == ResourceType.CPU:
                    actions.append(OptimizationAction.REDUCE_BATCH_SIZE)
                elif alert.resource_type == ResourceType.DISK:
                    actions.append(OptimizationAction.CLEANUP_MEMORY)
                
            elif alert.level == AlertLevel.WARNING:
                # 경고 상황 - 예방적 조치
                if alert.resource_type == ResourceType.MEMORY:
                    actions.append(OptimizationAction.CLEANUP_MEMORY)
                
        except Exception as e:
            logger.error(f"Failed to determine optimization actions: {e}")
        
        return actions
    
    def _cleanup_memory(self, alert: ResourceAlert) -> bool:
        """메모리 정리"""
        try:
            # 가비지 컬렉션 강제 실행
            collected = gc.collect()
            
            # 통계 업데이트
            self.stats["memory_cleanups"] += 1
            
            logger.info(f"Memory cleanup completed: {collected} objects collected")
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _reduce_batch_size(self, alert: ResourceAlert) -> bool:
        """배치 크기 감소"""
        try:
            # 동적 배치 관리자와 연동
            try:
                from core.performance.dynamic_batch_manager import get_batch_manager
                batch_manager = get_batch_manager()
                
                # 현재 배치 크기의 50% 감소
                current_size = batch_manager.get_current_batch_size()
                target_size = max(batch_manager.config.min_batch_size, int(current_size * 0.5))
                
                batch_manager.force_batch_size(target_size, "resource_manager_optimization")
                
                logger.info(f"Batch size reduced: {current_size} -> {target_size}")
                return True
                
            except ImportError:
                logger.warning("Dynamic batch manager not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reduce batch size: {e}")
            return False
    
    def _pause_operations(self, alert: ResourceAlert) -> bool:
        """작업 일시 중단"""
        try:
            # 실제 구현에서는 시스템의 주요 작업들을 일시 중단
            logger.warning("Operations pause requested - implement in production")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause operations: {e}")
            return False
    
    def _alert_admin(self, alert: ResourceAlert) -> bool:
        """관리자 알림"""
        try:
            alert_message = (
                f"SYSTEM RESOURCE ALERT\\n"
                f"Level: {alert.level.value.upper()}\\n"
                f"Resource: {alert.resource_type.value}\\n"
                f"Current: {alert.current_value:.1f}%\\n"
                f"Threshold: {alert.threshold_value:.1f}%\\n"
                f"Message: {alert.message}\\n"
                f"Timestamp: {alert.timestamp.isoformat()}"
            )
            
            # 실제 구현에서는 이메일, Slack 등으로 알림
            logger.critical(f"ADMIN ALERT: {alert_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send admin alert: {e}")
            return False
    
    def calculate_system_health(self) -> SystemHealth:
        """시스템 건강도 계산"""
        try:
            if not self.metrics_history:
                return SystemHealth()
            
            # 최근 메트릭 분석 (최근 10개)
            recent_metrics = self.metrics_history[-10:]
            
            health = SystemHealth()
            
            # CPU 점수
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            cpu_threshold = self.thresholds[ResourceType.CPU]
            health.cpu_score = max(0, 100 - (avg_cpu - cpu_threshold.target_level) * 2)
            
            # 메모리 점수
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            memory_threshold = self.thresholds[ResourceType.MEMORY]
            health.memory_score = max(0, 100 - (avg_memory - memory_threshold.target_level) * 2)
            
            # 디스크 점수
            avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
            disk_threshold = self.thresholds[ResourceType.DISK]
            health.disk_score = max(0, 100 - (avg_disk - disk_threshold.target_level) * 1.5)
            
            # 네트워크 점수 (간단히 100점으로 설정)
            health.network_score = 100.0
            
            # 안정성 점수 (최근 알림 빈도 기반)
            recent_alerts = [a for a in self.alerts_history 
                           if a.timestamp > datetime.now() - timedelta(hours=1)]
            critical_alerts = [a for a in recent_alerts 
                             if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
            
            health.stability_score = max(0, 100 - len(critical_alerts) * 10 - len(recent_alerts) * 2)
            
            # 전체 점수 계산
            health.calculate_overall_score()
            
            # 권장사항 생성
            if health.cpu_score < 70:
                health.recommendations.append("CPU 사용률이 높습니다. 프로세스 최적화를 고려하세요.")
            if health.memory_score < 70:
                health.recommendations.append("메모리 사용량이 많습니다. 메모리 정리를 실행하세요.")
            if health.disk_score < 70:
                health.recommendations.append("디스크 공간이 부족합니다. 불필요한 파일을 정리하세요.")
            if health.stability_score < 70:
                health.recommendations.append("시스템이 불안정합니다. 리소스 모니터링을 강화하세요.")
            
            if not health.recommendations:
                health.recommendations.append("시스템이 안정적으로 작동 중입니다.")
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to calculate system health: {e}")
            return SystemHealth(overall_score=0, status="error")
    
    def start_monitoring(self, interval_seconds: int = 30):
        """리소스 모니터링 시작"""
        if self._monitoring_active:
            logger.warning("Resource monitoring already active")
            return
        
        self._monitoring_active = True
        self.stats["monitoring_start_time"] = datetime.now().isoformat()
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    # 메트릭 수집
                    metrics = self.collect_system_metrics()
                    
                    # 히스토리에 추가
                    with self._lock:
                        self.metrics_history.append(metrics)
                        if len(self.metrics_history) > self.max_history_size:
                            self.metrics_history.pop(0)
                    
                    # 알림 분석
                    alerts = self.analyze_metrics(metrics)
                    
                    # 알림 처리
                    if alerts:
                        self.process_alerts(alerts)
                    
                    # 설정 저장 (주기적으로)
                    if len(self.metrics_history) % 20 == 0:
                        self._save_configuration()
                    
                    # 대기
                    for _ in range(interval_seconds):
                        if not self._monitoring_active:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(10)  # 에러 시 10초 후 재시도
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Resource monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """리소스 모니터링 중지"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        self._save_configuration()
        logger.info("Resource monitoring stopped")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """상태 요약"""
        try:
            current_metrics = self.collect_system_metrics()
            health = self.calculate_system_health()
            
            # 최근 알림 (1시간 이내)
            recent_alerts = [
                a for a in self.alerts_history 
                if a.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self._monitoring_active,
                'current_metrics': current_metrics.to_dict(),
                'system_health': {
                    'overall_score': health.overall_score,
                    'status': health.status,
                    'cpu_score': health.cpu_score,
                    'memory_score': health.memory_score,
                    'disk_score': health.disk_score,
                    'stability_score': health.stability_score,
                    'recommendations': health.recommendations
                },
                'recent_alerts': len(recent_alerts),
                'critical_alerts_24h': len([
                    a for a in self.alerts_history 
                    if a.timestamp > datetime.now() - timedelta(hours=24) and
                    a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
                ]),
                'stats': self.stats,
                'thresholds': {
                    resource.value: {
                        'warning': threshold.warning_level,
                        'critical': threshold.critical_level,
                        'emergency': threshold.emergency_level,
                        'target': threshold.target_level
                    }
                    for resource, threshold in self.thresholds.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get status summary: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_monitoring()
            
            # 메모리 정리
            self.metrics_history.clear()
            self.alerts_history.clear()
            
            logger.info("System resource manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 리소스 관리자
_global_resource_manager = None

def get_resource_manager(config_file: str = None) -> SystemResourceManager:
    """전역 리소스 관리자 반환"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = SystemResourceManager(
            config_file or "resource_manager_config.json"
        )
    return _global_resource_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_resource_manager():
        print("🧪 System Resource Manager 테스트")
        
        manager = get_resource_manager("test_resource_config.json")
        
        print("\n1️⃣ 시스템 메트릭 수집")
        metrics = manager.collect_system_metrics()
        print(f"  CPU: {metrics.cpu_percent:.1f}%")
        print(f"  메모리: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f}MB)")
        print(f"  디스크: {metrics.disk_percent:.1f}% ({metrics.disk_free_gb:.1f}GB 여유)")
        
        print("\n2️⃣ 알림 분석")
        alerts = manager.analyze_metrics(metrics)
        for alert in alerts:
            print(f"  ⚠️ {alert.level.value}: {alert.message}")
        
        print("\n3️⃣ 시스템 건강도")
        health = manager.calculate_system_health()
        print(f"  전체 점수: {health.overall_score:.1f}")
        print(f"  상태: {health.status}")
        print(f"  권장사항: {health.recommendations[0] if health.recommendations else 'None'}")
        
        print("\n4️⃣ 모니터링 시작 (5초)")
        manager.start_monitoring(interval_seconds=2)
        await asyncio.sleep(5)
        
        print("\n5️⃣ 상태 요약")
        summary = manager.get_status_summary()
        print(f"  모니터링 활성: {summary['monitoring_active']}")
        print(f"  최근 알림: {summary['recent_alerts']}개")
        print(f"  시스템 상태: {summary['system_health']['status']}")
        
        print("\n🎉 System Resource Manager 테스트 완료!")
        
        # 정리
        manager.cleanup()
        
        # 테스트 파일 정리
        test_file = Path("test_resource_config.json")
        if test_file.exists():
            test_file.unlink()
    
    # 테스트 실행
    asyncio.run(test_resource_manager())