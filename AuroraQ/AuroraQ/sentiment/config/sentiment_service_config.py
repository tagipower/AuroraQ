#!/usr/bin/env python3
"""
Sentiment Service Configuration for VPS Deployment
VPS 환경에 최적화된 설정 관리
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

class DeploymentMode(Enum):
    """배포 모드"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    VPS = "vps"

class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "auroaq_sentiment"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5  # VPS에서 연결 수 제한
    max_overflow: int = 3
    pool_timeout: int = 30
    ssl_mode: str = "prefer"

@dataclass
class RedisConfig:
    """Redis 설정"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10  # VPS 최적화
    socket_timeout: int = 5
    health_check_interval: int = 30

@dataclass
class APIConfig:
    """API 키 설정"""
    newsapi_key: Optional[str] = None
    finnhub_key: Optional[str] = None
    openai_key: Optional[str] = None
    
    def __post_init__(self):
        """환경변수에서 키 로드"""
        self.newsapi_key = self.newsapi_key or os.getenv("NEWSAPI_KEY")
        self.finnhub_key = self.finnhub_key or os.getenv("FINNHUB_KEY")
        self.openai_key = self.openai_key or os.getenv("OPENAI_KEY")

@dataclass
class VPSResourceLimits:
    """VPS 리소스 제한"""
    max_memory_mb: int = 3072  # 3GB
    max_cpu_cores: int = 2
    max_concurrent_requests: int = 3
    max_batch_size: int = 8
    request_timeout: int = 10
    max_retries: int = 2
    thread_pool_workers: int = 2

@dataclass
class CollectorConfig:
    """수집기 설정"""
    enabled_sources: List[str] = field(default_factory=lambda: [
        "google_news", "yahoo_finance", "newsapi", "finnhub"
    ])
    collection_interval: int = 300  # 5분
    max_items_per_source: int = 15  # VPS에서 제한
    duplicate_threshold: float = 0.8
    content_max_length: int = 2000
    hours_back: int = 24
    
    # Rate limiting (VPS 친화적)
    rate_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "newsapi": {"requests_per_hour": 100, "concurrent": 2},
        "finnhub": {"requests_per_hour": 60, "concurrent": 1},
        "google_news": {"requests_per_minute": 30, "concurrent": 2},
        "yahoo_finance": {"requests_per_minute": 50, "concurrent": 2}
    })

@dataclass
class ProcessorConfig:
    """프로세서 설정"""
    finbert_model: str = "ProsusAI/finbert"
    batch_interval: int = 900  # 15분
    initial_batch_size: int = 6  # VPS에서 감소
    max_batch_size: int = 12
    min_batch_size: int = 2
    max_sequence_length: int = 256  # 512에서 감소
    
    # 메모리 관리
    memory_threshold_mb: int = 2048  # 2GB
    cpu_threshold_percent: int = 80
    gc_interval: int = 300  # 5분마다 가비지 컬렉션
    
    # 이벤트 감지
    event_detection_interval: int = 600  # 10분
    impact_threshold: float = 7.0
    urgency_levels: Dict[str, int] = field(default_factory=lambda: {
        "immediate": 0,    # 즉시
        "high": 300,       # 5분
        "normal": 900,     # 15분
        "low": 1800        # 30분
    })

@dataclass
class FusionConfig:
    """융합 설정"""
    cache_size: int = 1000
    cache_ttl: int = 300  # 5분
    enable_adaptive_weights: bool = True
    confidence_threshold: float = 0.5  # VPS에서 낮춤
    quality_threshold: float = 0.4
    outlier_z_threshold: float = 2.5
    
    # 소스별 가중치
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "finbert": 0.6,
        "keyword": 0.4,
        "technical": 0.3,
        "social": 0.2,
        "news": 0.5
    })

@dataclass
class SchedulerConfig:
    """스케줄러 설정"""
    max_concurrent_tasks: int = 2  # VPS에 맞게 감소
    resource_check_interval: int = 30
    adaptive_scheduling: bool = True
    task_timeout: int = 900  # 15분
    
    # 작업별 우선순위 및 간격
    task_intervals: Dict[str, int] = field(default_factory=lambda: {
        "news_collection": 300,      # 5분
        "finbert_batch": 900,        # 15분
        "event_detection": 600,      # 10분
        "trading_signals_live": 180, # 3분
        "trading_signals_paper": 120, # 2분
        "cache_cleanup": 1800,       # 30분
        "system_maintenance": 86400   # 24시간
    })

@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "/var/log/auroaq/sentiment_service.log"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True

@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    enable_health_check: bool = True
    health_check_port: int = 8080
    metrics_port: int = 8081
    
    # 알림 설정
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 85.0,
        "memory_mb": 2560.0,  # 2.5GB
        "disk_percent": 85.0,
        "error_rate": 5.0     # 5%
    })
    
    # Webhook URLs
    discord_webhook: Optional[str] = None
    slack_webhook: Optional[str] = None

@dataclass
class SentimentServiceConfig:
    """전체 서비스 설정"""
    deployment_mode: DeploymentMode = DeploymentMode.VPS
    service_name: str = "auroaq-sentiment-service"
    version: str = "2.0.0"
    
    # 하위 설정들
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api_keys: APIConfig = field(default_factory=APIConfig)
    vps_limits: VPSResourceLimits = field(default_factory=VPSResourceLimits)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 기타 설정
    enable_debug: bool = False
    startup_delay: int = 30  # 30초 지연 시작
    graceful_shutdown_timeout: int = 60
    
    def __post_init__(self):
        """설정 후처리"""
        # 환경변수 기반 설정 오버라이드
        self._load_from_environment()
        
        # 배포 모드별 설정 조정
        self._adjust_for_deployment_mode()
        
        # 설정 검증
        self._validate_configuration()
    
    def _load_from_environment(self):
        """환경변수에서 설정 로드"""
        # 배포 모드
        if os.getenv("DEPLOYMENT_MODE"):
            try:
                self.deployment_mode = DeploymentMode(os.getenv("DEPLOYMENT_MODE"))
            except ValueError:
                pass
        
        # 디버그 모드
        self.enable_debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # 데이터베이스
        if os.getenv("DB_HOST"):
            self.database.host = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            self.database.port = int(os.getenv("DB_PORT"))
        if os.getenv("DB_NAME"):
            self.database.database = os.getenv("DB_NAME")
        if os.getenv("DB_USER"):
            self.database.username = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"):
            self.database.password = os.getenv("DB_PASSWORD")
        
        # Redis
        if os.getenv("REDIS_HOST"):
            self.redis.host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.redis.port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            self.redis.password = os.getenv("REDIS_PASSWORD")
        
        # 모니터링
        if os.getenv("DISCORD_WEBHOOK"):
            self.monitoring.discord_webhook = os.getenv("DISCORD_WEBHOOK")
        if os.getenv("SLACK_WEBHOOK"):
            self.monitoring.slack_webhook = os.getenv("SLACK_WEBHOOK")
    
    def _adjust_for_deployment_mode(self):
        """배포 모드별 설정 조정"""
        if self.deployment_mode == DeploymentMode.DEVELOPMENT:
            # 개발 모드: 더 많은 리소스, 짧은 간격
            self.logging.level = LogLevel.DEBUG
            self.collector.collection_interval = 60
            self.processor.batch_interval = 300
            self.vps_limits.max_concurrent_requests = 5
            
        elif self.deployment_mode == DeploymentMode.VPS:
            # VPS 모드: 리소스 절약, 안정성 우선
            self.logging.level = LogLevel.INFO
            self.vps_limits.max_memory_mb = 3072
            self.vps_limits.max_concurrent_requests = 3
            self.processor.initial_batch_size = 6
            self.processor.max_batch_size = 12
            
        elif self.deployment_mode == DeploymentMode.PRODUCTION:
            # 프로덕션 모드: 성능 최적화, 에러 최소화
            self.logging.level = LogLevel.WARNING
            self.processor.max_retries = 3
            self.collector.max_items_per_source = 20
    
    def _validate_configuration(self):
        """설정 검증"""
        warnings = []
        
        # API 키 체크
        if not self.api_keys.newsapi_key:
            warnings.append("NewsAPI key not configured")
        if not self.api_keys.finnhub_key:
            warnings.append("Finnhub key not configured")
        
        # 메모리 제한 체크
        if self.vps_limits.max_memory_mb < 1024:
            warnings.append("Memory limit too low (< 1GB)")
        
        # 배치 크기 체크
        if self.processor.max_batch_size > 20:
            warnings.append("Batch size may be too large for VPS")
        
        # 동시 요청 수 체크
        if self.vps_limits.max_concurrent_requests > 5:
            warnings.append("Concurrent requests may be too high for VPS")
        
        if warnings and self.enable_debug:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "deployment_mode": self.deployment_mode.value,
            "service_name": self.service_name,
            "version": self.version,
            "enable_debug": self.enable_debug,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "pool_size": self.database.pool_size
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "max_connections": self.redis.max_connections
            },
            "vps_limits": {
                "max_memory_mb": self.vps_limits.max_memory_mb,
                "max_cpu_cores": self.vps_limits.max_cpu_cores,
                "max_concurrent_requests": self.vps_limits.max_concurrent_requests,
                "max_batch_size": self.vps_limits.max_batch_size
            },
            "collector": {
                "enabled_sources": self.collector.enabled_sources,
                "collection_interval": self.collector.collection_interval,
                "max_items_per_source": self.collector.max_items_per_source
            },
            "processor": {
                "batch_interval": self.processor.batch_interval,
                "initial_batch_size": self.processor.initial_batch_size,
                "max_batch_size": self.processor.max_batch_size,
                "memory_threshold_mb": self.processor.memory_threshold_mb
            },
            "fusion": {
                "cache_size": self.fusion.cache_size,
                "cache_ttl": self.fusion.cache_ttl,
                "confidence_threshold": self.fusion.confidence_threshold
            },
            "scheduler": {
                "max_concurrent_tasks": self.scheduler.max_concurrent_tasks,
                "adaptive_scheduling": self.scheduler.adaptive_scheduling
            },
            "logging": {
                "level": self.logging.level.value,
                "file_path": self.logging.file_path,
                "enable_console": self.logging.enable_console
            },
            "monitoring": {
                "enable_health_check": self.monitoring.enable_health_check,
                "health_check_port": self.monitoring.health_check_port,
                "alert_thresholds": self.monitoring.alert_thresholds
            }
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SentimentServiceConfig':
        """파일에서 설정 로드"""
        import json
        import yaml
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # 딕셔너리에서 설정 객체 생성 (간단한 구현)
            config = cls()
            
            # 필요한 필드들만 업데이트
            if 'deployment_mode' in config_data:
                config.deployment_mode = DeploymentMode(config_data['deployment_mode'])
            
            if 'database' in config_data:
                db_config = config_data['database']
                config.database.host = db_config.get('host', config.database.host)
                config.database.port = db_config.get('port', config.database.port)
                config.database.database = db_config.get('database', config.database.database)
            
            return config
            
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            return cls()  # 기본 설정 반환

# 전역 설정 인스턴스
config = SentimentServiceConfig()

def get_config() -> SentimentServiceConfig:
    """전역 설정 인스턴스 반환"""
    return config

def reload_config(config_path: Optional[str] = None) -> SentimentServiceConfig:
    """설정 재로드"""
    global config
    
    if config_path:
        config = SentimentServiceConfig.from_file(config_path)
    else:
        config = SentimentServiceConfig()
    
    return config

# 설정 검증 함수
def validate_vps_environment() -> List[str]:
    """VPS 환경 검증"""
    issues = []
    
    # 메모리 체크
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        if available_memory < config.vps_limits.max_memory_mb:
            issues.append(f"Available memory ({available_memory:.0f}MB) is less than configured limit ({config.vps_limits.max_memory_mb}MB)")
    except ImportError:
        issues.append("psutil not available for memory checking")
    
    # CPU 체크
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        if cpu_count < config.vps_limits.max_cpu_cores:
            issues.append(f"Available CPU cores ({cpu_count}) is less than configured ({config.vps_limits.max_cpu_cores})")
    except ImportError:
        pass
    
    # 디스크 공간 체크
    try:
        import shutil
        disk_usage = shutil.disk_usage('/')
        free_gb = disk_usage.free / 1024 / 1024 / 1024
        if free_gb < 5:  # 5GB 미만
            issues.append(f"Low disk space: {free_gb:.1f}GB remaining")
    except Exception:
        pass
    
    return issues

if __name__ == "__main__":
    # 설정 테스트
    print("=== Sentiment Service Configuration Test ===")
    
    config = SentimentServiceConfig()
    
    print(f"Deployment Mode: {config.deployment_mode.value}")
    print(f"Service Name: {config.service_name}")
    print(f"Version: {config.version}")
    print(f"Debug Mode: {config.enable_debug}")
    
    print(f"\nVPS Limits:")
    print(f"  Max Memory: {config.vps_limits.max_memory_mb}MB")
    print(f"  Max CPU Cores: {config.vps_limits.max_cpu_cores}")
    print(f"  Max Concurrent Requests: {config.vps_limits.max_concurrent_requests}")
    print(f"  Max Batch Size: {config.vps_limits.max_batch_size}")
    
    print(f"\nProcessor Config:")
    print(f"  Batch Interval: {config.processor.batch_interval}s")
    print(f"  Initial Batch Size: {config.processor.initial_batch_size}")
    print(f"  Memory Threshold: {config.processor.memory_threshold_mb}MB")
    
    print(f"\nEnvironment Validation:")
    issues = validate_vps_environment()
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"  ✅ Environment looks good")
    
    print(f"\nConfiguration Summary:")
    config_dict = config.to_dict()
    import json
    print(json.dumps(config_dict, indent=2))