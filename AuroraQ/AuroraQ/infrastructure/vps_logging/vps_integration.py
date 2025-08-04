#!/usr/bin/env python3
"""
VPS Deployment 통합 로깅 어댑터
기존 VPS deployment 시스템과 통합 로그 관리자 연결
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 통합 로그 관리자 임포트
from .unified_log_manager import (
    UnifiedLogManager, LogCategory, LogLevel, 
    LoggingAdapter, create_vps_log_manager
)

# VPS deployment 기존 모듈들과 호환성을 위한 임포트 처리
try:
    # VPS sentiment service 모듈들
    from ..config.onnx_settings import get_onnx_settings
    VPS_SETTINGS_AVAILABLE = True
except ImportError:
    VPS_SETTINGS_AVAILABLE = False

class VPSLogIntegrator:
    """VPS Deployment 로깅 통합기"""
    
    def __init__(self, 
                 base_log_dir: str = "/app/logs",
                 enable_unified_logging: bool = True):
        """
        VPS 로깅 통합기 초기화
        
        Args:
            base_log_dir: 로그 기본 디렉토리
            enable_unified_logging: 통합 로깅 활성화
        """
        self.base_log_dir = Path(base_log_dir)
        self.enable_unified_logging = enable_unified_logging
        
        # 통합 로그 관리자
        if enable_unified_logging:
            self.unified_manager = create_vps_log_manager(str(self.base_log_dir))
            self._start_unified_logging()
        else:
            self.unified_manager = None
        
        # 컴포넌트별 어댑터
        self.adapters: Dict[str, LoggingAdapter] = {}
        
        # VPS 설정 로드
        self.vps_config = self._load_vps_config()
        
        # 기존 로거들 재설정
        self._integrate_existing_loggers()
    
    def _load_vps_config(self) -> Dict[str, Any]:
        """VPS 설정 로드"""
        config = {
            "log_level": "INFO",
            "log_format": "json",
            "log_file": str(self.base_log_dir / "vps_integration.log")
        }
        
        if VPS_SETTINGS_AVAILABLE:
            try:
                settings = get_onnx_settings()
                config.update({
                    "log_level": settings.log_level,
                    "log_format": settings.log_format,
                    "log_file": settings.log_file or config["log_file"]
                })
            except Exception as e:
                print(f"Warning: Could not load VPS settings: {e}")
        
        return config
    
    def _start_unified_logging(self):
        """통합 로깅 백그라운드 작업 시작"""
        if self.unified_manager:
            try:
                # 이벤트 루프가 실행 중인지 확인
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 실행 중인 루프에 태스크 추가
                    asyncio.create_task(self.unified_manager.start_background_tasks())
                else:
                    # 새 루프에서 실행
                    asyncio.run(self.unified_manager.start_background_tasks())
            except RuntimeError:
                # 이벤트 루프가 없는 경우 나중에 시작
                pass
    
    def _integrate_existing_loggers(self):
        """기존 로거들과 통합"""
        # 주요 VPS 컴포넌트들
        components = [
            "onnx_sentiment",
            "metrics_router", 
            "fusion_manager",
            "keyword_scorer",
            "batch_scheduler",
            "realtime_engine",
            "dashboard"
        ]
        
        for component in components:
            self.adapters[component] = self.get_logger(component)
    
    def get_logger(self, component_name: str) -> LoggingAdapter:
        """컴포넌트별 로거 어댑터 반환"""
        if component_name not in self.adapters:
            if self.unified_manager:
                self.adapters[component_name] = LoggingAdapter(
                    self.unified_manager, component_name
                )
            else:
                # 폴백: 표준 로거 사용
                logger = logging.getLogger(component_name)
                self.adapters[component_name] = StandardLoggerAdapter(logger)
        
        return self.adapters[component_name]
    
    async def log_onnx_inference(self, 
                                text: str,
                                inference_time: float,
                                confidence: float,
                                model_version: str = "finbert_onnx",
                                **metadata):
        """ONNX 추론 로깅"""
        if self.unified_manager:
            await self.unified_manager.log(
                category=LogCategory.TRAINING,
                level=LogLevel.INFO,
                component="onnx_inference",
                event_type="model_prediction",
                message=f"ONNX inference completed: {confidence:.4f} confidence",
                metadata={
                    "text_length": len(text),
                    "inference_time_ms": inference_time * 1000,
                    "confidence": confidence,
                    "model_version": model_version,
                    **metadata
                }
            )
    
    async def log_batch_processing(self,
                                  batch_size: int,
                                  processing_time: float,
                                  success_count: int,
                                  error_count: int,
                                  **metadata):
        """배치 처리 로깅"""
        if self.unified_manager:
            level = LogLevel.ERROR if error_count > 0 else LogLevel.INFO
            category = LogCategory.SUMMARY if success_count > 0 else LogCategory.RAW
            
            await self.unified_manager.log(
                category=category,
                level=level,
                component="batch_processor",
                event_type="batch_completed",
                message=f"Batch processed: {success_count}/{batch_size} success",
                metadata={
                    "batch_size": batch_size,
                    "processing_time_s": processing_time,
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": success_count / batch_size if batch_size > 0 else 0,
                    **metadata
                }
            )
    
    async def log_system_metrics(self,
                                component: str,
                                metrics: Dict[str, float],
                                **metadata):
        """시스템 메트릭 로깅"""
        if self.unified_manager:
            await self.unified_manager.log(
                category=LogCategory.SUMMARY,
                level=LogLevel.INFO,
                component=component,
                event_type="system_metrics",
                message=f"System metrics recorded for {component}",
                metadata={
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
            )
    
    async def log_security_event(self,
                                event_type: str,
                                severity: str,
                                description: str,
                                **metadata):
        """보안 이벤트 로깅 (Tagged 범주)"""
        if self.unified_manager:
            level_mapping = {
                "low": LogLevel.INFO,
                "medium": LogLevel.WARNING,
                "high": LogLevel.ERROR,
                "critical": LogLevel.CRITICAL
            }
            
            await self.unified_manager.log(
                category=LogCategory.TAGGED,
                level=level_mapping.get(severity, LogLevel.WARNING),
                component="security",
                event_type=event_type,
                message=description,
                tags=["security", severity, event_type],
                metadata=metadata
            )
    
    async def log_api_request(self,
                             method: str,
                             path: str,
                             status_code: int,
                             response_time: float,
                             **metadata):
        """API 요청 로깅"""
        if self.unified_manager:
            level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
            
            await self.unified_manager.log(
                category=LogCategory.RAW,
                level=level,
                component="api",
                event_type="http_request",
                message=f"{method} {path} -> {status_code}",
                metadata={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "response_time_ms": response_time * 1000,
                    **metadata
                }
            )
    
    async def log_strategy_performance(self,
                                     strategy_name: str,
                                     performance_metrics: Dict[str, Any],
                                     **metadata):
        """전략 성과 로깅 (Profit Factor 포함)"""
        if self.unified_manager:
            # Profit Factor 표시 처리
            pf = performance_metrics.get('profit_factor', 0)
            pf_display = '∞' if pf == float('inf') else f"{pf:.3f}"
            
            message = (f"Strategy Performance - {strategy_name}: "
                      f"PF={pf_display}, WR={performance_metrics.get('win_rate', 0):.1%}, "
                      f"Trades={performance_metrics.get('total_trades', 0)}")
            
            await self.unified_manager.log(
                category=LogCategory.SUMMARY,
                level=LogLevel.INFO,
                component="strategy_performance",
                event_type="performance_update",
                message=message,
                metadata={
                    "strategy_name": strategy_name,
                    "profit_factor": pf,
                    "profit_factor_display": pf_display,
                    "win_rate": performance_metrics.get('win_rate', 0),
                    "total_trades": performance_metrics.get('total_trades', 0),
                    "total_profit": performance_metrics.get('total_profit', 0),
                    "total_loss": performance_metrics.get('total_loss', 0),
                    "avg_win": performance_metrics.get('avg_win', 0),
                    "avg_loss": performance_metrics.get('avg_loss', 0),
                    "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0),
                    "total_pnl": performance_metrics.get('total_pnl', 0),
                    **metadata
                },
                tags=["strategy", "performance", "profit_factor"]
            )
    
    async def log_strategy_signal(self,
                                 strategy_name: str,
                                 signal_data: Dict[str, Any],
                                 selection_metadata: Dict[str, Any] = None,
                                 **metadata):
        """전략 신호 로깅 (점수 및 Profit Factor 포함)"""
        if self.unified_manager:
            selection_meta = selection_metadata or {}
            performance_info = signal_data.get('performance_info', {})
            
            # Profit Factor 표시 처리
            pf = performance_info.get('profit_factor', 0)
            if pf == 'Infinity':
                pf_display = '∞'
            else:
                pf_display = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
            
            message = (f"Strategy Signal - {strategy_name}: "
                      f"{signal_data.get('action', 'UNKNOWN')} "
                      f"(Score: {selection_meta.get('final_score', 0):.3f}, "
                      f"PF: {pf_display})")
            
            await self.unified_manager.log(
                category=LogCategory.RAW,
                level=LogLevel.INFO,
                component="strategy_signal",
                event_type="signal_generated",
                message=message,
                metadata={
                    "strategy_name": strategy_name,
                    "action": signal_data.get('action'),
                    "strength": signal_data.get('strength', 0),
                    "price": signal_data.get('price', 0),
                    "final_score": selection_meta.get('final_score', 0),
                    "composite_score": selection_meta.get('score_breakdown', {}).get('composite_score', 0),
                    "confidence": selection_meta.get('score_breakdown', {}).get('confidence', 0),
                    "performance_score": selection_meta.get('score_breakdown', {}).get('performance_score', 0),
                    "profit_factor": pf,
                    "profit_factor_display": pf_display,
                    "win_rate": performance_info.get('win_rate', 0),
                    "total_trades": performance_info.get('total_trades', 0),
                    "sharpe_ratio": performance_info.get('sharpe_ratio', 0),
                    **metadata
                },
                tags=["strategy", "signal", "profit_factor", signal_data.get('action', '').lower()]
            )
    
    async def log_trading_result(self,
                               strategy_name: str,
                               pnl: float,
                               trade_duration: float = None,
                               **metadata):
        """거래 결과 로깅 (Profit Factor 영향)"""
        if self.unified_manager:
            result_type = "profit" if pnl > 0 else "loss"
            level = LogLevel.INFO if pnl >= 0 else LogLevel.WARNING
            
            message = (f"Trade Result - {strategy_name}: "
                      f"{'+' if pnl >= 0 else ''}{pnl:.4f} "
                      f"({'Win' if pnl > 0 else 'Loss'})")
            
            await self.unified_manager.log(
                category=LogCategory.TRAINING,  # 학습 데이터로 활용
                level=level,
                component="trading_results",
                event_type="trade_completed",
                message=message,
                metadata={
                    "strategy_name": strategy_name,
                    "pnl": pnl,
                    "result_type": result_type,
                    "trade_duration_minutes": trade_duration,
                    "is_profitable": pnl > 0,
                    **metadata
                },
                tags=["trading", "result", result_type, strategy_name.lower()]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """통합 로깅 통계"""
        stats = {
            "integration_enabled": self.enable_unified_logging,
            "adapters_count": len(self.adapters),
            "base_log_dir": str(self.base_log_dir),
            "vps_config": self.vps_config
        }
        
        if self.unified_manager:
            stats.update(self.unified_manager.get_stats())
        
        return stats
    
    async def shutdown(self):
        """안전한 종료"""
        if self.unified_manager:
            await self.unified_manager.shutdown()

class StandardLoggerAdapter:
    """표준 로거 어댑터 (폴백용)"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

# 전역 통합기 인스턴스
_vps_integrator: Optional[VPSLogIntegrator] = None

def get_vps_log_integrator(base_log_dir: str = "/app/logs") -> VPSLogIntegrator:
    """VPS 로깅 통합기 싱글톤 인스턴스"""
    global _vps_integrator
    
    if _vps_integrator is None:
        _vps_integrator = VPSLogIntegrator(base_log_dir)
    
    return _vps_integrator

def setup_vps_logging(base_log_dir: str = "/app/logs",
                     enable_unified: bool = True) -> VPSLogIntegrator:
    """VPS 로깅 시스템 설정"""
    global _vps_integrator
    
    _vps_integrator = VPSLogIntegrator(
        base_log_dir=base_log_dir,
        enable_unified_logging=enable_unified
    )
    
    return _vps_integrator

# 편의 함수들
async def log_onnx_event(text: str, confidence: float, inference_time: float, **kwargs):
    """ONNX 이벤트 로깅 편의 함수"""
    integrator = get_vps_log_integrator()
    await integrator.log_onnx_inference(text, inference_time, confidence, **kwargs)

async def log_batch_event(batch_size: int, processing_time: float, 
                         success_count: int, error_count: int, **kwargs):
    """배치 이벤트 로깅 편의 함수"""
    integrator = get_vps_log_integrator()
    await integrator.log_batch_processing(batch_size, processing_time, 
                                        success_count, error_count, **kwargs)

async def log_security_alert(event_type: str, severity: str, description: str, **kwargs):
    """보안 알림 로깅 편의 함수"""
    integrator = get_vps_log_integrator()
    await integrator.log_security_event(event_type, severity, description, **kwargs)

async def log_strategy_perf(strategy_name: str, performance_metrics: Dict[str, Any], **kwargs):
    """전략 성과 로깅 편의 함수 (Profit Factor 포함)"""
    integrator = get_vps_log_integrator()
    await integrator.log_strategy_performance(strategy_name, performance_metrics, **kwargs)

async def log_strategy_sig(strategy_name: str, signal_data: Dict[str, Any], 
                          selection_metadata: Dict[str, Any] = None, **kwargs):
    """전략 신호 로깅 편의 함수 (Profit Factor 포함)"""
    integrator = get_vps_log_integrator()
    await integrator.log_strategy_signal(strategy_name, signal_data, selection_metadata, **kwargs)

async def log_trade_result(strategy_name: str, pnl: float, trade_duration: float = None, **kwargs):
    """거래 결과 로깅 편의 함수 (Profit Factor 영향)"""
    integrator = get_vps_log_integrator()
    await integrator.log_trading_result(strategy_name, pnl, trade_duration, **kwargs)

if __name__ == "__main__":
    # 테스트 실행
    async def test_integration():
        integrator = setup_vps_logging("/tmp/test_logs")
        
        # 테스트 로그들
        await log_onnx_event("Test news", 0.85, 0.150)
        await log_batch_event(100, 5.2, 98, 2)
        await log_security_alert("auth_failure", "medium", "Failed login attempt")
        
        print("Integration stats:", integrator.get_stats())
        
        await integrator.shutdown()
    
    asyncio.run(test_integration())