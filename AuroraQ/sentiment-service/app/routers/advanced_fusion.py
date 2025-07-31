# app/routers/advanced_fusion.py
"""Advanced Sentiment Fusion API endpoints"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import structlog

from models import (
    FusionRequest,
    FusionResponse,
    ErrorResponse
)
# 의존성 import 오류 예방
try:
    from app.dependencies import get_advanced_fusion_manager, get_cache_manager
except ImportError:
    try:
        from ..dependencies import get_advanced_fusion_manager, get_cache_manager
    except ImportError:
        # Mock functions for testing
        def get_advanced_fusion_manager():
            return None
        def get_cache_manager():
            return None

# 로깅 및 설정 import 처리
try:
    from ..utils.logging_config import get_logger, log_function_call
    from ..config.settings import settings
except ImportError:
    try:
        from utils.logging_config import get_logger, log_function_call
        from config.settings import settings
    except ImportError:
        try:
            from config.settings import settings
            import structlog
            logger = structlog.get_logger(__name__)
            def get_logger(name): return logger
            def log_function_call(*args, **kwargs): return {}
        except ImportError:
            # 기본 대안
            import structlog
            logger = structlog.get_logger(__name__)
            def get_logger(name): return logger
            def log_function_call(*args, **kwargs): return {}
            
            # settings 기본값
            class Settings:
                debug = False
                cache_ttl = 300
            settings = Settings()

# 로컬 임포트 (실제 환경에서 사용)
# from ..processors.advanced_fusion_manager import AdvancedFusionManager, RefinedFeatureSet
# from ..models.advanced_keyword_scorer import AdvancedKeywordScorer

# logger 초기화
try:
    logger = get_logger(__name__)
except:
    import structlog
    logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/advanced/{symbol}", response_model=Dict[str, Any])
async def advanced_fusion_analysis(
    symbol: str,
    background_tasks: BackgroundTasks,
    text: Optional[str] = None,
    include_market_data: bool = True,
    include_social_data: bool = True,
    force_refresh: bool = False,
    advanced_fusion_manager=Depends(get_advanced_fusion_manager),
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Advanced multimodal sentiment fusion analysis
    
    - **symbol**: Trading symbol (e.g., BTCUSDT, ETH, BTC)
    - **text**: Optional text for analysis (uses latest news if not provided)
    - **include_market_data**: Include price and volume data in analysis
    - **include_social_data**: Include social media data in analysis
    - **force_refresh**: Force refresh of cached data
    
    Returns comprehensive analysis including:
    - Multimodal sentiment scores
    - ML refined predictions
    - Event impact analysis
    - Strategy performance metrics
    - Anomaly detection results
    - Advanced feature sets for ML/PPO training
    """
    start_time = time.time()
    
    try:
        # Generate cache key
        cache_key_data = {
            "symbol": symbol,
            "text": text or "auto",
            "market_data": include_market_data,
            "social_data": include_social_data,
            "timestamp": int(time.time() / 300) * 300  # 5분 단위 캐싱
        }
        cache_key = f"advanced_fusion:{hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()}"
        
        # Try to get from cache first (simplified mock)
        # 실제 구현에서는 cache_manager 사용
        # if not force_refresh:
        #     cached_result = await cache_manager.get_content(cache_key)
        #     if cached_result:
        #         logger.info("Cache hit for advanced fusion analysis", cache_key=cache_key)
        #         return cached_result
        
        # Prepare input data
        if not text:
            # Get latest news for the symbol (mock implementation)
            text = f"Latest market analysis for {symbol} shows mixed signals with institutional interest growing"
        
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Mock market data (실제 구현에서는 실제 시장 데이터 사용)
        market_data = None
        if include_market_data:
            import numpy as np
            market_data = {
                "price_data": {
                    "prices": [45000 + i * 10 + np.random.normal(0, 50) for i in range(200)]
                },
                "volume_data": {
                    "volumes": [1000000 + i * 1000 + np.random.normal(0, 50000) for i in range(200)]
                }
            }
        
        # Mock social data (실제 구현에서는 실제 소셜 데이터 사용)
        social_data = None
        if include_social_data:
            social_data = {
                "twitter": {"likes": 1500, "retweets": 300, "replies": 150},
                "reddit": {"upvotes": 250, "comments": 45},
                "retweet_velocity": 50.5,
                "cross_platform_spread": 0.7,
                "influencer_amplification": 1.5
            }
        
        # Perform advanced fusion analysis
        # 실제 구현에서는 다음 라인의 주석을 해제하고 mock 데이터 제거
        # refined_features = await advanced_fusion_manager.advanced_fusion_analysis(
        #     content_hash=content_hash,
        #     text=text,
        #     market_data=market_data,
        #     social_data=social_data,
        #     force_refresh=force_refresh
        # )
        
        # Mock refined features (실제 구현에서 제거)
        import numpy as np
        refined_features = {
            "fusion_score": np.random.uniform(-1, 1),
            "ml_refined_prediction": {
                "direction": np.random.choice(["bullish", "bearish", "neutral"]),
                "probability": np.random.uniform(0.3, 0.9),
                "volatility_forecast": np.random.uniform(0.1, 0.5),
                "confidence_level": np.random.choice(["very_high", "high", "medium", "low"]),
                "trend_strength": np.random.uniform(0.0, 1.0),
                "momentum_persistence": np.random.uniform(0.3, 0.9),
                "regime_stability": np.random.uniform(0.4, 0.8),
                "black_swan_risk": np.random.uniform(0.01, 0.1),
                "model_ensemble_size": 3,
                "prediction_uncertainty": np.random.uniform(0.1, 0.4),
                "feature_importance": {
                    "text_sentiment": np.random.uniform(0.1, 0.3),
                    "price_action": np.random.uniform(0.1, 0.3),
                    "volume_analysis": np.random.uniform(0.1, 0.3),
                    "social_signals": np.random.uniform(0.1, 0.3)
                }
            },
            "event_impact": {
                "event_type": np.random.choice(["news", "regulatory", "technical", "social"]),
                "impact_score": np.random.uniform(0.3, 0.9),
                "lag_estimate": np.random.uniform(10, 120),
                "duration_estimate": np.random.uniform(6, 48),
                "market_spillover": np.random.uniform(0.2, 0.8),
                "cross_asset_correlation": np.random.uniform(0.1, 0.7),
                "volatility_impact": np.random.uniform(0.1, 0.6),
                "prediction_accuracy": np.random.uniform(0.6, 0.9)
            },
            "strategy_performance": {
                "strategy_name": "AuroraQ_Advanced_Strategy",
                "roi": np.random.uniform(-0.1, 0.3),
                "sharpe_ratio": np.random.uniform(0.5, 2.5),
                "max_drawdown": np.random.uniform(0.05, 0.25),
                "win_rate": np.random.uniform(0.45, 0.75),
                "var_95": np.random.uniform(0.01, 0.05),
                "expected_shortfall": np.random.uniform(0.015, 0.08),
                "beta": np.random.uniform(0.8, 1.2),
                "alpha": np.random.uniform(-0.05, 0.15)
            },
            "anomaly_detection": {
                "anomaly_flag": np.random.choice([True, False], p=[0.3, 0.7]),
                "anomaly_score": np.random.uniform(0.0, 1.0),
                "anomaly_type": np.random.choice(["price", "volume", "sentiment", "correlation"]),
                "severity": np.random.choice(["low", "medium", "high", "critical"]),
                "event_tag": np.random.choice(["whale_movement", "news_spike", "technical_breakout", None]),
                "recommended_action": np.random.choice(["monitor", "investigate", "alert", "hedge"]),
                "confidence": np.random.uniform(0.6, 0.95)
            },
            "advanced_features": {
                "multimodal_sentiment": {
                    "text_sentiment": np.random.uniform(-1, 1),
                    "price_action_sentiment": np.random.uniform(-1, 1),
                    "volume_sentiment": np.random.uniform(-1, 1),
                    "social_engagement": np.random.uniform(0, 1)
                },
                "temporal_features": {
                    "momentum_1h": np.random.uniform(-1, 1),
                    "momentum_4h": np.random.uniform(-1, 1),
                    "momentum_24h": np.random.uniform(-1, 1),
                    "volatility_regime_score": np.random.uniform(0, 1)
                },
                "network_features": {
                    "viral_score": np.random.uniform(0, 1),
                    "network_effect": np.random.uniform(0, 1),
                    "herding_behavior": np.random.uniform(0, 1)
                },
                "risk_features": {
                    "black_swan_probability": np.random.uniform(0, 0.1),
                    "tail_risk": np.random.uniform(0, 0.2),
                    "panic_indicator": np.random.uniform(0, 1)
                },
                "emotional_ai": {
                    "emotional_state_score": np.random.uniform(-1, 1)
                },
                "prediction_meta": {
                    "confidence_calibration": np.random.uniform(0.5, 1.0),
                    "prediction_stability": np.random.uniform(0.3, 0.9),
                    "ensemble_agreement": np.random.uniform(0.6, 1.0)
                }
            },
            "feature_quality_score": np.random.uniform(0.7, 0.95),
            "completeness_ratio": np.random.uniform(0.8, 1.0),
            "processing_metadata": {
                "symbol": symbol,
                "text_length": len(text),
                "market_data_included": include_market_data,
                "social_data_included": include_social_data,
                "content_hash": content_hash
            }
        }
        
        # Calculate overall confidence
        overall_confidence = (
            refined_features["ml_refined_prediction"]["probability"] * 0.4 +
            refined_features["event_impact"]["prediction_accuracy"] * 0.3 +
            refined_features["feature_quality_score"] * 0.3
        )
        
        # Determine market outlook
        ml_direction = refined_features["ml_refined_prediction"]["direction"]
        fusion_score = refined_features["fusion_score"]
        
        if ml_direction == "bullish" and fusion_score > 0.2:
            market_outlook = "strong_bullish"
        elif ml_direction == "bullish" or fusion_score > 0.1:
            market_outlook = "bullish"
        elif ml_direction == "bearish" and fusion_score < -0.2:
            market_outlook = "strong_bearish"
        elif ml_direction == "bearish" or fusion_score < -0.1:
            market_outlook = "bearish"
        else:
            market_outlook = "neutral"
        
        # Prepare response
        result = {
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time,
            "cache_hit": False,
            
            # Core analysis results
            "fusion_score": refined_features["fusion_score"],
            "market_outlook": market_outlook,
            "overall_confidence": overall_confidence,
            
            # ML predictions
            "ml_prediction": refined_features["ml_refined_prediction"],
            
            # Event analysis
            "event_impact": refined_features["event_impact"],
            
            # Strategy performance
            "strategy_performance": refined_features["strategy_performance"],
            
            # Anomaly detection
            "anomaly_detection": refined_features["anomaly_detection"],
            
            # Advanced features for ML/PPO
            "advanced_features": refined_features["advanced_features"],
            
            # Quality metrics
            "feature_quality_score": refined_features["feature_quality_score"],
            "completeness_ratio": refined_features["completeness_ratio"],
            
            # Metadata
            "metadata": refined_features["processing_metadata"],
            
            # API metadata
            "api_version": "v3.0",
            "feature_version": "advanced_v2.0"
        }
        
        # Cache result in background (mock implementation)
        # 실제 구현에서는 다음과 같이 구현:
        # background_tasks.add_task(
        #     cache_manager.store_content,
        #     cache_key,
        #     json.dumps({k: v for k, v in result.items() if k not in ['processing_time', 'cache_hit']}),
        #     ttl=300
        # )
        
        # Log performance
        logger.info(
            "Advanced sentiment fusion completed",
            **log_function_call(
                "advanced_fusion_analysis",
                symbol=symbol,
                market_outlook=market_outlook,
                fusion_score=refined_features["fusion_score"],
                overall_confidence=overall_confidence,
                processing_time=result["processing_time"],
                feature_quality=refined_features["feature_quality_score"]
            )
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Advanced sentiment fusion failed",
            error=str(e),
            symbol=symbol,
            text_length=len(text) if text else 0
        )
        raise HTTPException(
            status_code=500,
            detail=f"Advanced sentiment fusion failed: {str(e)}"
        )

@router.get("/features/{symbol}")
async def get_advanced_features(
    symbol: str,
    feature_types: List[str] = Query(default=["all"]),
    format: str = Query(default="json", regex="^(json|numpy|pandas)$"),
    advanced_fusion_manager=Depends(get_advanced_fusion_manager)
) -> Dict[str, Any]:
    """
    Get advanced features for ML/PPO training
    
    - **symbol**: Trading symbol
    - **feature_types**: Types of features to include (multimodal, temporal, network, risk, emotional, prediction_meta, or all)
    - **format**: Output format (json, numpy, pandas)
    
    Returns structured feature sets optimized for machine learning training.
    """
    try:
        # Mock implementation (실제 구현에서는 advanced_fusion_manager 사용)
        import numpy as np
        
        available_feature_types = ["multimodal", "temporal", "network", "risk", "emotional", "prediction_meta"]
        
        if "all" in feature_types:
            selected_types = available_feature_types
        else:
            selected_types = [ft for ft in feature_types if ft in available_feature_types]
        
        features = {}
        
        if "multimodal" in selected_types:
            features["multimodal_sentiment"] = {
                "text_sentiment": np.random.uniform(-1, 1),
                "price_action_sentiment": np.random.uniform(-1, 1),
                "volume_sentiment": np.random.uniform(-1, 1),
                "social_engagement": np.random.uniform(0, 1)
            }
        
        if "temporal" in selected_types:
            features["temporal_features"] = {
                "momentum_1h": np.random.uniform(-1, 1),
                "momentum_4h": np.random.uniform(-1, 1),
                "momentum_24h": np.random.uniform(-1, 1),
                "volatility_regime_score": np.random.uniform(0, 1),
                "returns_series": np.random.normal(0, 0.02, 100).tolist(),
                "volume_series": np.random.lognormal(13, 0.5, 100).tolist()
            }
        
        if "network" in selected_types:
            features["network_features"] = {
                "viral_score": np.random.uniform(0, 1),
                "network_effect": np.random.uniform(0, 1),
                "herding_behavior": np.random.uniform(0, 1),
                "information_diffusion_rate": np.random.uniform(0, 1),
                "network_centrality": np.random.uniform(0, 1)
            }
        
        if "risk" in selected_types:
            features["risk_features"] = {
                "black_swan_probability": np.random.uniform(0, 0.1),
                "tail_risk": np.random.uniform(0, 0.2),
                "panic_indicator": np.random.uniform(0, 1),
                "volatility_clustering": np.random.uniform(0, 1),
                "correlation_breakdown": np.random.uniform(0, 1)
            }
        
        if "emotional" in selected_types:
            features["emotional_ai"] = {
                "emotional_state_score": np.random.uniform(-1, 1),
                "fear_greed_index": np.random.uniform(0, 100),
                "market_psychology_state": np.random.choice(["fear", "greed", "neutral"]),
                "sentiment_velocity": np.random.uniform(-0.5, 0.5),
                "emotional_persistence": np.random.uniform(0, 1)
            }
        
        if "prediction_meta" in selected_types:
            features["prediction_meta"] = {
                "confidence_calibration": np.random.uniform(0.5, 1.0),
                "prediction_stability": np.random.uniform(0.3, 0.9),
                "ensemble_agreement": np.random.uniform(0.6, 1.0),
                "model_uncertainty": np.random.uniform(0.1, 0.4),
                "concept_drift_score": np.random.uniform(0, 0.3)
            }
        
        # Format conversion
        if format == "numpy":
            # Convert to numpy-friendly format
            numpy_features = {}
            for feature_type, feature_data in features.items():
                if isinstance(feature_data, dict):
                    numpy_features[feature_type] = {
                        k: v if isinstance(v, list) else [v] for k, v in feature_data.items()
                    }
                else:
                    numpy_features[feature_type] = feature_data
            features = numpy_features
        
        elif format == "pandas":
            # Convert to pandas-friendly format (flattened)
            flattened_features = {}
            for feature_type, feature_data in features.items():
                if isinstance(feature_data, dict):
                    for key, value in feature_data.items():
                        if isinstance(value, list):
                            # For time series data, take the last value
                            flattened_features[f"{feature_type}_{key}_latest"] = value[-1] if value else 0.0
                            flattened_features[f"{feature_type}_{key}_mean"] = np.mean(value) if value else 0.0
                            flattened_features[f"{feature_type}_{key}_std"] = np.std(value) if len(value) > 1 else 0.0
                        else:
                            flattened_features[f"{feature_type}_{key}"] = value
                else:
                    flattened_features[feature_type] = feature_data
            features = flattened_features
        
        result = {
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "feature_types": selected_types,
            "format": format,
            "features": features,
            "feature_count": len(features),
            "metadata": {
                "api_version": "v3.0",
                "feature_extraction_method": "advanced_multimodal",
                "quality_score": np.random.uniform(0.8, 0.95)
            }
        }
        
        logger.info(
            "Advanced features extracted",
            symbol=symbol,
            feature_types=selected_types,
            feature_count=len(features),
            format=format
        )
        
        return result
        
    except Exception as e:
        logger.error("Advanced feature extraction failed", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Advanced feature extraction failed: {str(e)}"
        )

@router.get("/insights/{symbol}")
async def get_ai_insights(
    symbol: str,
    insight_types: List[str] = Query(default=["pattern", "risk", "opportunity"]),
    confidence_threshold: float = Query(default=0.6, ge=0.0, le=1.0),
    max_insights: int = Query(default=5, ge=1, le=20),
    advanced_fusion_manager=Depends(get_advanced_fusion_manager)
) -> Dict[str, Any]:
    """
    Get AI-generated insights and recommendations
    
    - **symbol**: Trading symbol
    - **insight_types**: Types of insights (pattern, risk, opportunity, sentiment, market_structure)
    - **confidence_threshold**: Minimum confidence level for insights
    - **max_insights**: Maximum number of insights to return
    
    Returns AI-generated insights with confidence scores and actionable recommendations.
    """
    try:
        # Mock AI insights (실제 구현에서는 advanced_fusion_manager 사용)
        import numpy as np
        
        available_types = ["pattern", "risk", "opportunity", "sentiment", "market_structure"]
        selected_types = [t for t in insight_types if t in available_types]
        
        insights = []
        
        for insight_type in selected_types:
            confidence = np.random.uniform(confidence_threshold, 1.0)
            
            if insight_type == "pattern":
                insight = {
                    "type": "pattern_recognition",
                    "confidence": confidence,
                    "title": "Bullish Divergence Detected",
                    "message": f"RSI showing bullish divergence while {symbol} price continues downtrend. Historical patterns suggest potential reversal within 2-4 trading sessions.",
                    "recommendations": [
                        "Monitor for volume confirmation on next upward move",
                        "Consider scaled entry approach rather than single large position",
                        "Set stop loss below recent swing low"
                    ],
                    "evidence": {
                        "rsi_divergence_strength": np.random.uniform(0.6, 0.9),
                        "price_trend_duration": np.random.randint(5, 15),
                        "volume_pattern": "decreasing_on_declines",
                        "support_level": 42500,
                        "resistance_level": 47000
                    },
                    "time_horizon": "2-4 days",
                    "risk_level": "medium"
                }
            
            elif insight_type == "risk":
                insight = {
                    "type": "risk_assessment",
                    "confidence": confidence,
                    "title": "Elevated Correlation Risk",
                    "message": f"Cross-asset correlations have increased significantly. {symbol} showing high correlation with equity markets, reducing diversification benefits.",
                    "recommendations": [
                        "Consider reducing position size during high correlation periods",
                        "Implement alternative hedging strategies",
                        "Monitor traditional market movements more closely"
                    ],
                    "evidence": {
                        "correlation_with_sp500": np.random.uniform(0.7, 0.9),
                        "correlation_increase": np.random.uniform(0.2, 0.4),
                        "volatility_clustering": True,
                        "risk_metrics": {
                            "var_95": np.random.uniform(0.03, 0.08),
                            "expected_shortfall": np.random.uniform(0.05, 0.12)
                        }
                    },
                    "time_horizon": "1-2 weeks",
                    "risk_level": "high"
                }
            
            elif insight_type == "opportunity":
                insight = {
                    "type": "opportunity_identification",
                    "confidence": confidence,
                    "title": "Institutional Accumulation Pattern",
                    "message": f"Large block trades and decreased exchange outflows suggest institutional accumulation in {symbol}. This typically precedes significant price movements.",
                    "recommendations": [
                        "Position for potential upside breakout",
                        "Monitor on-chain metrics for continuation signals",
                        "Consider increasing allocation if risk tolerance permits"
                    ],
                    "evidence": {
                        "large_transaction_volume": np.random.uniform(1.5, 3.0),  # Multiple of normal
                        "exchange_outflow_ratio": np.random.uniform(0.15, 0.35),
                        "whale_accumulation_score": np.random.uniform(0.7, 0.95),
                        "institutional_interest_index": np.random.uniform(0.6, 0.9)
                    },
                    "time_horizon": "1-4 weeks",
                    "risk_level": "medium"
                }
            
            elif insight_type == "sentiment":
                insight = {
                    "type": "sentiment_analysis",
                    "confidence": confidence,
                    "title": "Sentiment-Price Divergence",
                    "message": f"Social sentiment for {symbol} remains highly positive while price action shows weakness. This divergence often leads to sentiment-driven price moves.",
                    "recommendations": [
                        "Monitor for sentiment exhaustion signals",
                        "Prepare for potential sentiment-driven volatility",
                        "Consider contrarian positioning if divergence persists"
                    ],
                    "evidence": {
                        "social_sentiment_score": np.random.uniform(0.6, 0.9),
                        "price_momentum": np.random.uniform(-0.3, -0.1),
                        "sentiment_persistence": np.random.uniform(5, 15),  # Days
                        "viral_content_engagement": np.random.uniform(0.7, 0.95)
                    },
                    "time_horizon": "3-7 days",
                    "risk_level": "medium"
                }
            
            elif insight_type == "market_structure":
                insight = {
                    "type": "market_microstructure",
                    "confidence": confidence,
                    "title": "Liquidity Concentration Analysis",
                    "message": f"Order book analysis shows liquidity concentration at key levels. {symbol} may experience increased volatility around these levels.",
                    "recommendations": [
                        "Use limit orders near identified liquidity zones",
                        "Avoid market orders during low liquidity periods",
                        "Monitor depth changes for early trend signals"
                    ],
                    "evidence": {
                        "bid_ask_spread": np.random.uniform(0.1, 0.5),  # Percentage
                        "market_depth_ratio": np.random.uniform(0.3, 0.8),
                        "liquidity_zones": [42000, 45000, 47500],
                        "order_flow_imbalance": np.random.uniform(-0.3, 0.3)
                    },
                    "time_horizon": "intraday",
                    "risk_level": "low"
                }
            
            insights.append(insight)
            
            if len(insights) >= max_insights:
                break
        
        # Sort by confidence and limit results
        insights = sorted(insights, key=lambda x: x["confidence"], reverse=True)[:max_insights]
        
        # Calculate aggregate metrics
        avg_confidence = np.mean([insight["confidence"] for insight in insights])
        risk_distribution = {}
        for insight in insights:
            risk_level = insight["risk_level"]
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        result = {
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "insights": insights,
            "summary": {
                "total_insights": len(insights),
                "average_confidence": avg_confidence,
                "risk_distribution": risk_distribution,
                "confidence_threshold": confidence_threshold,
                "insight_types_requested": selected_types
            },
            "metadata": {
                "api_version": "v3.0",
                "ai_model_version": "advanced_v2.0",
                "generation_method": "multimodal_ensemble"
            }
        }
        
        logger.info(
            "AI insights generated",
            symbol=symbol,
            insight_count=len(insights),
            average_confidence=avg_confidence,
            insight_types=selected_types
        )
        
        return result
        
    except Exception as e:
        logger.error("AI insight generation failed", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"AI insight generation failed: {str(e)}"
        )

@router.get("/performance/strategy")
async def get_strategy_performance_analysis(
    strategy_name: str = Query(default="AuroraQ_Advanced"),
    time_period: str = Query(default="30d", regex="^(1d|7d|30d|90d|1y|all)$"),
    include_drawdown_analysis: bool = True,
    include_risk_metrics: bool = True,
    advanced_fusion_manager=Depends(get_advanced_fusion_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive strategy performance analysis
    
    - **strategy_name**: Name of the trading strategy
    - **time_period**: Time period for analysis (1d, 7d, 30d, 90d, 1y, all)
    - **include_drawdown_analysis**: Include drawdown periods analysis
    - **include_risk_metrics**: Include advanced risk metrics
    
    Returns detailed strategy performance metrics and analysis.
    """
    try:
        # Mock strategy performance analysis (실제 구현에서는 advanced_fusion_manager 사용)
        import numpy as np
        
        # Generate mock performance data
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365, "all": 730}[time_period]
        
        # Basic performance metrics
        total_return = np.random.uniform(-0.2, 0.8) if days > 30 else np.random.uniform(-0.1, 0.3)
        annualized_return = total_return * (365 / days) if days < 365 else total_return
        volatility = np.random.uniform(0.15, 0.45)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        performance_data = {
            "strategy_name": strategy_name,
            "analysis_period": time_period,
            "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
            "end_date": datetime.now().isoformat(),
            
            "returns": {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "monthly_returns": [np.random.uniform(-0.15, 0.15) for _ in range(min(days//30, 12))],
                "best_month": np.random.uniform(0.1, 0.4),
                "worst_month": np.random.uniform(-0.3, -0.05),
                "positive_months": np.random.randint(6, 11) if days >= 365 else np.random.randint(1, days//30 + 1)
            },
            
            "risk_metrics": {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sharpe_ratio * np.random.uniform(1.1, 1.4),
                "calmar_ratio": np.random.uniform(0.5, 2.0),
                "max_drawdown": np.random.uniform(0.05, 0.3),
                "var_95": np.random.uniform(0.02, 0.08),
                "expected_shortfall": np.random.uniform(0.03, 0.12),
                "beta": np.random.uniform(0.7, 1.3),
                "alpha": np.random.uniform(-0.05, 0.15)
            },
            
            "trading_stats": {
                "total_trades": np.random.randint(50, 500),
                "win_rate": np.random.uniform(0.45, 0.75),
                "avg_win": np.random.uniform(0.02, 0.08),
                "avg_loss": np.random.uniform(-0.06, -0.015),
                "profit_factor": np.random.uniform(1.1, 2.5),
                "avg_holding_period": np.random.uniform(2, 14),  # Days
                "max_consecutive_wins": np.random.randint(3, 12),
                "max_consecutive_losses": np.random.randint(2, 8)
            }
        }
        
        # Optional drawdown analysis
        if include_drawdown_analysis:
            performance_data["drawdown_analysis"] = {
                "current_drawdown": np.random.uniform(0.0, 0.15),
                "max_drawdown": performance_data["risk_metrics"]["max_drawdown"],
                "avg_drawdown": np.random.uniform(0.02, 0.08),
                "drawdown_periods": [
                    {
                        "start_date": (datetime.now() - timedelta(days=np.random.randint(10, 100))).isoformat(),
                        "end_date": (datetime.now() - timedelta(days=np.random.randint(1, 50))).isoformat(),
                        "depth": np.random.uniform(0.05, 0.2),
                        "duration_days": np.random.randint(3, 30),
                        "recovery_days": np.random.randint(5, 45)
                    } for _ in range(3)
                ],
                "underwater_curve": [np.random.uniform(-0.2, 0) for _ in range(min(days, 100))]
            }
        
        # Optional advanced risk metrics
        if include_risk_metrics:
            performance_data["advanced_risk"] = {
                "tail_ratio": np.random.uniform(0.5, 1.5),
                "skewness": np.random.uniform(-1.0, 1.0),
                "kurtosis": np.random.uniform(0, 5),
                "downside_deviation": np.random.uniform(0.1, 0.3),
                "upside_capture": np.random.uniform(0.8, 1.3),
                "downside_capture": np.random.uniform(0.7, 1.2),
                "information_ratio": np.random.uniform(-0.5, 1.5),
                "tracking_error": np.random.uniform(0.05, 0.25),
                "treynor_ratio": np.random.uniform(0.05, 0.25)
            }
        
        # Performance attribution
        performance_data["attribution"] = {
            "factor_contributions": {
                "sentiment_alpha": np.random.uniform(-0.03, 0.08),
                "momentum_factor": np.random.uniform(-0.02, 0.06),
                "mean_reversion": np.random.uniform(-0.02, 0.04),
                "volatility_timing": np.random.uniform(-0.02, 0.05),
                "event_capture": np.random.uniform(-0.01, 0.03)
            },
            "sector_allocation": {
                "crypto_major": np.random.uniform(0.4, 0.8),
                "crypto_alt": np.random.uniform(0.1, 0.4),
                "cash": np.random.uniform(0.1, 0.3)
            },
            "top_contributing_positions": [
                {"symbol": "BTC", "contribution": np.random.uniform(0.02, 0.15)},
                {"symbol": "ETH", "contribution": np.random.uniform(0.01, 0.08)},
                {"symbol": "SOL", "contribution": np.random.uniform(-0.02, 0.05)}
            ]
        }
        
        # Current portfolio state
        performance_data["current_portfolio"] = {
            "total_value": 100000 * (1 + total_return),  # Assuming $100k start
            "positions": {
                "BTC": np.random.uniform(0.3, 0.6),
                "ETH": np.random.uniform(0.2, 0.4),
                "Others": np.random.uniform(0.1, 0.3),
                "Cash": np.random.uniform(0.05, 0.2)
            },
            "leverage": np.random.uniform(1.0, 2.0),
            "margin_usage": np.random.uniform(0.1, 0.6)
        }
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_type": "comprehensive_strategy_performance",
            "performance_data": performance_data,
            "metadata": {
                "api_version": "v3.0",
                "calculation_method": "advanced_portfolio_analytics",
                "benchmark": "HODL_BTC",
                "data_quality_score": np.random.uniform(0.85, 0.98)
            }
        }
        
        logger.info(
            "Strategy performance analysis completed",
            strategy_name=strategy_name,
            time_period=time_period,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio
        )
        
        return result
        
    except Exception as e:
        logger.error("Strategy performance analysis failed", strategy_name=strategy_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Strategy performance analysis failed: {str(e)}"
        )

@router.get("/health/advanced")
async def advanced_fusion_health_check(
    advanced_fusion_manager=Depends(get_advanced_fusion_manager)
):
    """Advanced fusion service health check with detailed diagnostics"""
    try:
        # Mock health check (실제 구현에서는 advanced_fusion_manager 사용)
        import numpy as np
        
        # Simulate component health checks
        components_health = {
            "advanced_keyword_scorer": {
                "status": "healthy",
                "response_time_ms": np.random.uniform(10, 50),
                "accuracy": np.random.uniform(0.8, 0.95),
                "uptime": "99.9%"
            },
            "ml_prediction_engine": {
                "status": "healthy",
                "ensemble_size": 3,
                "model_accuracy": np.random.uniform(0.75, 0.9),
                "last_training": (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat()
            },
            "event_impact_analyzer": {
                "status": "healthy",
                "events_processed": np.random.randint(100, 1000),
                "avg_processing_time": np.random.uniform(50, 200)
            },
            "anomaly_detector": {
                "status": "healthy",
                "detection_rate": np.random.uniform(0.02, 0.1),
                "false_positive_rate": np.random.uniform(0.01, 0.05)
            },
            "network_analyzer": {
                "status": "healthy",
                "data_sources": ["twitter", "reddit", "telegram"],
                "update_frequency": "real-time"
            }
        }
        
        # Overall system metrics (mock implementation)
        try:
            import psutil
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": 45.2,
                "active_connections": np.random.randint(50, 200),
                "cache_hit_rate": np.random.uniform(0.8, 0.95),
                "average_response_time": np.random.uniform(100, 500)
            }
        except ImportError:
            # psutil이 없는 경우 mock 데이터 사용
            system_metrics = {
                "cpu_usage": 25.5,
                "memory_usage": 45.2,
                "disk_usage": 65.8,
                "active_connections": 125,
                "cache_hit_rate": 0.87,
                "average_response_time": 180.5
            }
        
        # Determine overall health
        unhealthy_components = [
            name for name, health in components_health.items() 
            if health.get("status") != "healthy"
        ]
        
        overall_status = "healthy" if not unhealthy_components else "degraded"
        
        if system_metrics["memory_usage"] > 90 or system_metrics["cpu_usage"] > 90:
            overall_status = "stressed"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "advanced_sentiment_fusion",
            "version": "v3.0",
            "components": components_health,
            "system_metrics": system_metrics,
            "diagnostics": {
                "total_components": len(components_health),
                "healthy_components": len([c for c in components_health.values() if c.get("status") == "healthy"]),
                "last_health_check": datetime.utcnow().isoformat(),
                "uptime_seconds": np.random.randint(3600, 86400)
            },
            "recommendations": [
                "System operating within normal parameters" if overall_status == "healthy"
                else "Monitor resource usage and consider scaling"
            ]
        }
        
    except Exception as e:
        logger.error("Advanced fusion health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "advanced_sentiment_fusion",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )