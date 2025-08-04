"""
AuroraQ VPS 통합 로깅 시스템
4가지 로그 범주 통합 관리 패키지
"""

from .unified_log_manager import (
    UnifiedLogManager,
    LogCategory,
    LogLevel,
    LogEntry,
    RetentionPolicy,
    StorageConfig,
    LoggingAdapter,
    create_vps_log_manager
)

# VPS 통합 로깅을 위한 알리아스 함수
def get_vps_log_integrator(base_dir: str = "/app/logs") -> UnifiedLogManager:
    """VPS 로그 통합기 생성 (create_vps_log_manager의 알리아스)"""
    return create_vps_log_manager(base_dir)

__version__ = "1.0.0"
__all__ = [
    "UnifiedLogManager",
    "LogCategory", 
    "LogLevel",
    "LogEntry",
    "RetentionPolicy",
    "StorageConfig",
    "LoggingAdapter",
    "create_vps_log_manager",
    "get_vps_log_integrator"
]