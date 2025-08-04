"""
AuroraQ VPS Deployment Package
VPS 환경 최적화된 실전매매 시스템
"""

__version__ = "3.0.0"
__author__ = "AuroraQ VPS Team"

# 패키지 정보
VPS_DEPLOYMENT_INFO = {
    "name": "AuroraQ VPS Deployment",
    "version": __version__,
    "description": "48GB VPS 최적화 실전매매 시스템",
    "components": [
        "trading",
        "vps_logging", 
        "sentiment_service",
        "dashboard",
        "monitoring"
    ]
}