"""
VPS Sentiment Service Processors Package
데이터 처리 및 융합 분석 모듈
"""

from .advanced_fusion_manager_vps import (
    VPSAdvancedFusionManager,
    RefinedFeatureSet,
    MarketPrediction,
    EventImpactAnalysis,
    StrategyPerformance,
    AnomalyDetection,
    analyze_fusion_vps,
    get_vps_fusion_stats,
    cleanup_vps_fusion
)

__all__ = [
    "VPSAdvancedFusionManager",
    "RefinedFeatureSet",
    "MarketPrediction",
    "EventImpactAnalysis", 
    "StrategyPerformance",
    "AnomalyDetection",
    "analyze_fusion_vps",
    "get_vps_fusion_stats",
    "cleanup_vps_fusion"
]