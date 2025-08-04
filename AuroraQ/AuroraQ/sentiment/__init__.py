"""
VPS-Optimized Sentiment Service
9패널 대시보드용 메트릭 생성 서비스

ONNX 통합 및 VPS 최적화된 센티먼트 분석 패키지
"""

from .models.advanced_keyword_scorer_vps import (
    VPSAdvancedKeywordScorer,
    MultiModalSentiment,
    AdvancedFeatures,
    EmotionalState,
    MarketRegime,
    analyze_sentiment_vps,
    get_vps_performance_stats,
    clear_vps_cache
)

from .processors.advanced_fusion_manager_vps import (
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

__version__ = "2.0.0"
__author__ = "AuroraQ VPS Team"

# VPS 패키지 정보
VPS_INFO = {
    "optimized_for": "48GB VPS",
    "memory_limit": "4GB",
    "cache_ttl": "30min",
    "onnx_integration": True,
    "batch_processing": True,
    "dashboard_panels": 9
}

__all__ = [
    # Keyword Scorer
    "VPSAdvancedKeywordScorer",
    "MultiModalSentiment", 
    "AdvancedFeatures",
    "EmotionalState",
    "MarketRegime",
    "analyze_sentiment_vps",
    "get_vps_performance_stats",
    "clear_vps_cache",
    
    # Fusion Manager
    "VPSAdvancedFusionManager",
    "RefinedFeatureSet",
    "MarketPrediction", 
    "EventImpactAnalysis",
    "StrategyPerformance",
    "AnomalyDetection",
    "analyze_fusion_vps",
    "get_vps_fusion_stats", 
    "cleanup_vps_fusion",
    
    # Package Info
    "VPS_INFO"
]