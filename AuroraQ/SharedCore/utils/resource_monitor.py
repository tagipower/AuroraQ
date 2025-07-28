"""
리소스 모니터링 및 최적화 유틸리티
"""

import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """리소스 사용량 정보"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: datetime


class ResourceMonitor:
    """
    시스템 리소스 모니터링 및 최적화 제안
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.usage_history = []
        self.thresholds = {
            'cpu_warning': 70.0,      # CPU 70% 이상시 경고
            'cpu_critical': 85.0,     # CPU 85% 이상시 위험
            'memory_warning': 75.0,   # 메모리 75% 이상시 경고
            'memory_critical': 90.0,  # 메모리 90% 이상시 위험
        }
        
    def get_current_usage(self) -> ResourceUsage:
        """현재 리소스 사용량 조회"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        usage = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_io_read_mb=(disk_io.read_bytes / 1024 / 1024) if disk_io else 0,
            disk_io_write_mb=(disk_io.write_bytes / 1024 / 1024) if disk_io else 0,
            network_sent_mb=(network_io.bytes_sent / 1024 / 1024) if network_io else 0,
            network_recv_mb=(network_io.bytes_recv / 1024 / 1024) if network_io else 0,
            timestamp=datetime.now()
        )
        
        # 히스토리 업데이트
        self.usage_history.append(usage)
        
        # 최대 100개 항목만 유지
        if len(self.usage_history) > 100:
            self.usage_history.pop(0)
            
        return usage
    
    def check_resource_health(self) -> Dict[str, str]:
        """리소스 상태 체크"""
        usage = self.get_current_usage()
        health_status = {}
        
        # CPU 상태
        if usage.cpu_percent >= self.thresholds['cpu_critical']:
            health_status['cpu'] = 'critical'
        elif usage.cpu_percent >= self.thresholds['cpu_warning']:
            health_status['cpu'] = 'warning'
        else:
            health_status['cpu'] = 'healthy'
            
        # 메모리 상태
        if usage.memory_percent >= self.thresholds['memory_critical']:
            health_status['memory'] = 'critical'
        elif usage.memory_percent >= self.thresholds['memory_warning']:
            health_status['memory'] = 'warning'
        else:
            health_status['memory'] = 'healthy'
            
        return health_status
    
    def get_optimization_suggestions(self, mode: str = "aurora") -> Dict[str, list]:
        """모드별 최적화 제안"""
        health = self.check_resource_health()
        suggestions = {
            'immediate': [],    # 즉시 적용 가능
            'configuration': [],  # 설정 변경 필요
            'infrastructure': []  # 인프라 변경 필요
        }
        
        # CPU 최적화
        if health['cpu'] in ['warning', 'critical']:
            if mode == "aurora":
                suggestions['immediate'].extend([
                    "감정분석 배치 크기 축소 (32 → 16)",
                    "캐시 TTL 증가로 API 호출 줄이기",
                    "불필요한 로깅 레벨 조정"
                ])
                suggestions['configuration'].extend([
                    "실시간 데이터 수집 주기 증가 (1분 → 2분)",
                    "PPO 추론 주기 최적화"
                ])
            elif mode == "macro":
                suggestions['immediate'].extend([
                    "TFT 추론 배치 크기 축소",
                    "포트폴리오 최적화 주기 증가"
                ])
                
        # 메모리 최적화
        if health['memory'] in ['warning', 'critical']:
            suggestions['immediate'].extend([
                "메모리 캐시 크기 제한",
                "사용하지 않는 데이터 수집기 비활성화",
                "가비지 컬렉션 강제 실행"
            ])
            suggestions['configuration'].extend([
                "Redis 외부 캐시 사용 권장",
                "데이터 보존 기간 단축"
            ])
            
        # 인프라 제안
        if health['cpu'] == 'critical' or health['memory'] == 'critical':
            suggestions['infrastructure'].extend([
                "CPU 코어 추가 또는 메모리 증설 검토",
                "로드 밸런서를 통한 부하 분산",
                "캐싱 서버 분리"
            ])
            
        return suggestions
    
    def get_resource_trend(self, hours: int = 24) -> Dict[str, float]:
        """리소스 사용량 트렌드 분석"""
        if len(self.usage_history) < 2:
            return {}
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_usage = [u for u in self.usage_history if u.timestamp >= cutoff_time]
        
        if len(recent_usage) < 2:
            return {}
            
        # 평균 및 최대값 계산
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        max_cpu = max(u.cpu_percent for u in recent_usage)
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        max_memory = max(u.memory_percent for u in recent_usage)
        
        return {
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'sample_count': len(recent_usage)
        }
    
    def should_enable_optimization_mode(self) -> bool:
        """최적화 모드 활성화 여부 판단"""
        usage = self.get_current_usage()
        
        # CPU 또는 메모리가 임계치를 넘으면 최적화 모드 활성화
        return (
            usage.cpu_percent >= self.thresholds['cpu_warning'] or
            usage.memory_percent >= self.thresholds['memory_warning']
        )
    
    def log_resource_status(self):
        """현재 리소스 상태 로깅"""
        usage = self.get_current_usage()
        health = self.check_resource_health()
        
        logger.info(f"📊 Resource Status: CPU {usage.cpu_percent:.1f}% | "
                   f"Memory {usage.memory_percent:.1f}% ({usage.memory_used_mb:.0f}MB)")
        
        # 경고/위험 상태시 추가 로깅
        for resource, status in health.items():
            if status != 'healthy':
                logger.warning(f"⚠️ {resource.upper()} {status.upper()}: "
                             f"Consider optimization measures")


# 전역 리소스 모니터 인스턴스
_resource_monitor = None

def get_resource_monitor() -> ResourceMonitor:
    """전역 리소스 모니터 인스턴스 반환"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor