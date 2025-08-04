"""
VPS Sentiment Service API Package
9패널 대시보드용 메트릭 API 모듈
"""

from .metrics_router import router as metrics_router

__all__ = ["metrics_router"]