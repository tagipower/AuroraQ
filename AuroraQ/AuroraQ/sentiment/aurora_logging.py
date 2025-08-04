#!/usr/bin/env python3
"""
Simple logging module for AuroraQ
간단한 로깅 모듈
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class LogCategory(Enum):
    """로그 카테고리"""
    GENERAL = "general"
    SENTIMENT = "sentiment"
    TRADING = "trading"
    SYSTEM = "system"

class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SimpleLogIntegrator:
    """간단한 로그 통합기"""
    
    def __init__(self):
        self.logger = logging.getLogger("auroaq")
        
    def get_logger(self, name: str):
        """로거 반환"""
        return logging.getLogger(f"auroaq.{name}")
    
    async def log_onnx_inference(self, **kwargs):
        """ONNX 추론 로깅"""
        self.logger.info(f"ONNX inference: {kwargs}")
    
    async def log_batch_processing(self, **kwargs):
        """배치 처리 로깅"""
        self.logger.info(f"Batch processing: {kwargs}")
    
    async def log_security_event(self, **kwargs):
        """보안 이벤트 로깅"""
        self.logger.warning(f"Security event: {kwargs}")
    
    async def log_system_metrics(self, **kwargs):
        """시스템 메트릭 로깅"""
        self.logger.info(f"System metrics: {kwargs}")

_log_integrator = None

def get_vps_log_integrator():
    """VPS 로그 통합기 반환"""
    global _log_integrator
    if _log_integrator is None:
        _log_integrator = SimpleLogIntegrator()
    return _log_integrator

def get_logger(name: str):
    """간편한 로거 반환 함수"""
    return logging.getLogger(f"auroaq.{name}")