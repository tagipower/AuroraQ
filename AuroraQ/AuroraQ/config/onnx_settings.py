#!/usr/bin/env python3
"""
ONNX 센티먼트 서비스 설정
VPS 최적화 및 20분 배치 처리 전용
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator

class ONNXSettings(BaseSettings):
    """ONNX 센티먼트 서비스 설정 (VPS 최적화)"""
    
    # 애플리케이션 기본 설정
    app_name: str = "AuroraQ ONNX Sentiment Service"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # ONNX 모델 설정
    onnx_model_path: str = Field(
        default="/app/models/finbert.onnx",
        description="ONNX 모델 파일 경로"
    )
    model_cache_dir: str = Field(
        default="/app/cache",
        description="모델 캐시 디렉토리"
    )
    tokenizer_name: str = Field(
        default="ProsusAI/finbert",
        description="토크나이저 모델명"
    )
    
    # VPS 성능 최적화 설정 (48GB RAM 기준)
    onnx_threads: int = Field(
        default=6,
        ge=1,
        le=16,
        description="ONNX Runtime 스레드 수"
    )
    max_sequence_length: int = Field(
        default=512,
        ge=128,
        le=1024,
        description="최대 시퀀스 길이"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=32,
        description="기본 배치 크기"
    )
    max_batch_size: int = Field(
        default=400,
        ge=10,
        le=1000,
        description="최대 배치 크기"
    )
    memory_limit_gb: float = Field(
        default=10.0,
        ge=4.0,
        le=32.0,
        description="메모리 사용 제한 (GB)"
    )
    cpu_limit_cores: float = Field(
        default=6.0,
        ge=1.0,
        le=16.0,
        description="CPU 코어 제한"
    )
    
    # 20분 배치 스케줄러 설정
    batch_interval_minutes: int = Field(
        default=20,
        ge=5,
        le=60,
        description="배치 실행 간격 (분)"
    )
    enable_batch_scheduler: bool = Field(
        default=True,
        description="배치 스케줄러 활성화"
    )
    scheduler_auto_start: bool = Field(
        default=True,
        description="스케줄러 자동 시작"
    )
    
    # Redis 캐시 설정
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis 연결 URL"
    )
    redis_password: Optional[str] = None
    redis_db: int = Field(default=0, ge=0, le=15)
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="캐시 TTL (초)"
    )
    batch_result_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="배치 결과 TTL (시간)"
    )
    
    # 데이터 수집 설정
    news_collection_enabled: bool = Field(
        default=True,
        description="뉴스 수집 활성화"
    )
    news_sources: List[str] = Field(
        default=["google_news", "newsapi"],
        description="뉴스 소스 목록"
    )
    news_collection_limit: int = Field(
        default=400,
        ge=10,
        le=1000,
        description="뉴스 수집 최대 개수"
    )
    
    # API 서버 설정
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Uvicorn 워커 수 (VPS에서는 1 권장)"
    )
    
    # 보안 설정
    api_keys: List[str] = Field(
        default=[],
        description="허용된 API 키 목록"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS 허용 오리진"
    )
    trusted_hosts: List[str] = Field(
        default=["*"],
        description="신뢰할 수 있는 호스트"
    )
    
    # 로깅 설정
    log_level: str = Field(
        default="INFO",
        regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    log_format: str = Field(
        default="json",
        regex="^(json|text)$"
    )
    log_file: Optional[str] = Field(
        default="/app/logs/onnx_sentiment.log",
        description="로그 파일 경로"
    )
    
    # 통합 로깅 설정
    enable_unified_logging: bool = Field(
        default=True,
        description="통합 로그 관리자 사용"
    )
    unified_log_dir: str = Field(
        default="/app/logs/unified",
        description="통합 로그 디렉토리"
    )
    
    # 모니터링 설정
    enable_metrics: bool = Field(
        default=True,
        description="Prometheus 메트릭 활성화"
    )
    metrics_port: int = Field(
        default=8002,
        ge=8001,
        le=9000
    )
    
    # 개발/테스트 설정
    model_warmup: bool = Field(
        default=True,
        description="서비스 시작시 모델 웜업"
    )
    enable_test_endpoints: bool = Field(
        default=False,
        description="테스트 엔드포인트 활성화"
    )
    
    @validator('batch_interval_minutes')
    def validate_batch_interval(cls, v):
        """배치 간격 검증"""
        if v not in [5, 10, 15, 20, 30, 60]:
            raise ValueError("Batch interval should be 5, 10, 15, 20, 30, or 60 minutes")
        return v
    
    @validator('onnx_model_path')
    def validate_model_path(cls, v):
        """모델 경로 검증"""
        if not v or not v.endswith('.onnx'):
            raise ValueError("ONNX model path must end with .onnx")
        return v
    
    @property
    def batch_interval_seconds(self) -> int:
        """배치 간격을 초 단위로 반환"""
        return self.batch_interval_minutes * 60
    
    @property
    def cache_ttl_hours(self) -> float:
        """캐시 TTL을 시간 단위로 반환"""
        return self.cache_ttl_seconds / 3600
    
    @property
    def memory_limit_bytes(self) -> int:
        """메모리 제한을 바이트 단위로 반환"""
        return int(self.memory_limit_gb * 1024 * 1024 * 1024)
    
    def get_redis_config(self) -> dict:
        """Redis 연결 설정 반환"""
        config = {
            "url": self.redis_url,
            "db": self.redis_db,
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
            "retry_on_timeout": True,
            "health_check_interval": 30
        }
        
        if self.redis_password:
            config["password"] = self.redis_password
            
        return config
    
    def get_onnx_config(self) -> dict:
        """ONNX 런타임 설정 반환"""
        return {
            "model_path": self.onnx_model_path,
            "max_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "num_threads": self.onnx_threads,
            "providers": ["CPUExecutionProvider"],
            "session_options": {
                "intra_op_num_threads": self.onnx_threads,
                "inter_op_num_threads": 1,
                "graph_optimization_level": "all"
            }
        }
    
    def get_scheduler_config(self) -> dict:
        """배치 스케줄러 설정 반환"""
        return {
            "batch_interval": self.batch_interval_seconds,
            "max_batch_size": self.max_batch_size,
            "redis_ttl": self.cache_ttl_seconds,
            "enable_fusion": True,
            "auto_start": self.scheduler_auto_start
        }
    
    class Config:
        env_file = ".env"
        env_prefix = "ONNX_"
        case_sensitive = False

# 전역 설정 인스턴스
_settings = None

def get_onnx_settings() -> ONNXSettings:
    """ONNX 설정 인스턴스 반환 (싱글톤)"""
    global _settings
    if _settings is None:
        _settings = ONNXSettings()
    return _settings