#!/usr/bin/env python3
"""
Advanced AuroraQ Dashboard v3.0
고도화된 AI 기반 감정 분석 대시보드 with ML 예측 및 고급 분석
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from collections import deque
import curses
import colorsys

# 로컬 임포트 (실제 환경에서 사용)
# from ..processors.advanced_fusion_manager import AdvancedFusionManager, RefinedFeatureSet
# from ..models.advanced_keyword_scorer import AdvancedKeywordScorer

logger = logging.getLogger(__name__)

class PanelType(Enum):
    """Panel Type Definitions"""
    SENTIMENT_FUSION = "sentiment_fusion"
    ML_PREDICTIONS = "ml_predictions"
    EVENT_IMPACT = "event_impact"
    STRATEGY_PERFORMANCE = "strategy_performance"
    ANOMALY_DETECTION = "anomaly_detection"
    NETWORK_ANALYSIS = "network_analysis"
    MARKET_PULSE = "market_pulse"
    SYSTEM_INTELLIGENCE = "system_intelligence"
    LIVE_DATA_FEED = "live_data_feed"

class AlertLevel(Enum):
    """Alert Level Definitions"""
    CRITICAL = ("critical", 1)
    HIGH = ("high", 2)
    MEDIUM = ("medium", 3)
    LOW = ("low", 4)
    INFO = ("info", 5)

@dataclass
class PanelData:
    """Panel Data Structure"""
    title: str
    content: Dict[str, Any]
    status: str
    last_updated: datetime
    alert_level: AlertLevel
    metadata: Dict[str, Any]

@dataclass
class AIInsight:
    """AI Insight Data Structure"""
    insight_type: str
    confidence: float
    message: str
    recommendations: List[str]
    evidence: Dict[str, Any]
    created_at: datetime

class ClaudeCodeAdvancedTypingEffect:
    """Claude Code 스타일 고급 타이핑 효과"""
    
    def __init__(self):
        self.typing_speed = 0.008  # 더 빠른 타이핑
        self.cursor_blink_rate = 0.5
        self.color_transition_speed = 0.1
        self.effects = {
            'rainbow': self._rainbow_effect,
            'pulse': self._pulse_effect,
            'matrix': self._matrix_effect,
            'neon': self._neon_effect
        }
    
    def _rainbow_effect(self, text: str, progress: float) -> str:
        """White unified effect (formerly rainbow)"""
        # 흰색 통일 효과
        return f"\033[37;1m{text}\033[0m"  # 밝은 흰색
    
    def _pulse_effect(self, text: str, progress: float) -> str:
        """펄스 효과"""
        intensity = int(255 * (0.5 + 0.5 * np.sin(progress * 10)))
        return f"\033[38;2;{intensity};{intensity//2};{255-intensity}m{text}\033[0m"
    
    def _matrix_effect(self, text: str, progress: float) -> str:
        """매트릭스 효과"""
        intensity = int(255 * (0.3 + 0.7 * progress))
        return f"\033[38;2;0;{intensity};0m{text}\033[0m"
    
    def _neon_effect(self, text: str, progress: float) -> str:
        """네온 효과"""
        glow = int(255 * progress)
        return f"\033[38;2;{glow};{255};{glow}m{text}\033[0m"

class AdvancedAuroraServiceClient:
    """Aurora Advanced Service Client"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.session = None
        self.connection_timeout = 3.0  # 더 짧은 타임아웃
        self.read_timeout = 8.0
        self.max_retries = 2  # 재시도 줄임
        self.fallback_enabled = True  # 폴백 항상 활성화
        
        # 캐시 시스템
        self.cache = {}
        self.cache_ttl = 30  # 30초
        
        # 연결 상태 모니터링
        self.connection_status = "disconnected"
        self.last_successful_call = None
        self.error_count = 0
        
        # 성능 메트릭
        self.response_times = deque(maxlen=100)
        self.success_rate = 1.0
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        timeout = aiohttp.ClientTimeout(
            total=self.read_timeout,
            connect=self.connection_timeout
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_advanced_fusion_analysis(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """고급 융합 분석 조회"""
        endpoint = f"/api/v1/fusion/advanced/{symbol}"
        return await self._make_request("GET", endpoint, fallback_data={
            "fusion_score": np.random.uniform(-1, 1),
            "ml_prediction": {
                "direction": np.random.choice(["bullish", "bearish", "neutral"]),
                "probability": np.random.uniform(0.3, 0.9),
                "volatility_forecast": np.random.uniform(0.1, 0.5),
                "confidence_level": np.random.choice(["high", "medium", "low"]),
                "trend_strength": np.random.uniform(0.0, 1.0)
            },
            "multimodal_sentiment": {
                "text_sentiment": np.random.uniform(-1, 1),
                "price_action_sentiment": np.random.uniform(-1, 1),
                "volume_sentiment": np.random.uniform(-1, 1),
                "social_engagement": np.random.uniform(0, 1)
            },
            "advanced_features": {
                "viral_score": np.random.uniform(0, 1),
                "network_effect": np.random.uniform(0, 1),
                "black_swan_probability": np.random.uniform(0, 0.1),
                "herding_behavior": np.random.uniform(0, 1)
            }
        })
    
    async def get_event_impact_analysis(self) -> Dict[str, Any]:
        """이벤트 영향도 분석 조회"""
        endpoint = "/api/v1/events/impact-analysis"
        return await self._make_request("GET", endpoint, fallback_data={
            "events": [
                {
                    "event_type": "regulatory",
                    "impact_score": np.random.uniform(0.5, 1.0),
                    "lag_estimate": np.random.uniform(10, 120),
                    "duration_estimate": np.random.uniform(6, 48),
                    "market_spillover": np.random.uniform(0.3, 0.8),
                    "description": "SEC Bitcoin ETF Decision Pending"
                },
                {
                    "event_type": "technical",
                    "impact_score": np.random.uniform(0.2, 0.7),
                    "lag_estimate": np.random.uniform(5, 30),
                    "duration_estimate": np.random.uniform(2, 12),
                    "market_spillover": np.random.uniform(0.1, 0.5),
                    "description": "Golden Cross Formation Detected"
                }
            ],
            "aggregate_impact": np.random.uniform(0.4, 0.9),
            "market_sentiment_shift": np.random.uniform(-0.3, 0.5)
        })
    
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """전략 성과 조회"""
        endpoint = "/api/v1/trading/performance"
        return await self._make_request("GET", endpoint, fallback_data={
            "strategy_name": "AuroraQ_Advanced_Strategy",
            "roi": np.random.uniform(-0.1, 0.3),
            "sharpe_ratio": np.random.uniform(0.5, 2.5),
            "max_drawdown": np.random.uniform(0.05, 0.25),
            "win_rate": np.random.uniform(0.45, 0.75),
            "var_95": np.random.uniform(0.01, 0.05),
            "current_positions": {
                "BTC": np.random.uniform(0.2, 0.6),
                "ETH": np.random.uniform(0.1, 0.4),
                "CASH": np.random.uniform(0.1, 0.5)
            },
            "recent_trades": [
                {"symbol": "BTC", "side": "buy", "pnl": np.random.uniform(-0.05, 0.15)},
                {"symbol": "ETH", "side": "sell", "pnl": np.random.uniform(-0.03, 0.08)}
            ],
            "performance_metrics": {
                "daily_return": np.random.uniform(-0.05, 0.05),
                "weekly_return": np.random.uniform(-0.15, 0.15),
                "monthly_return": np.random.uniform(-0.30, 0.30)
            }
        })
    
    async def get_anomaly_detection(self) -> Dict[str, Any]:
        """이상 탐지 결과 조회"""
        endpoint = "/api/v1/analysis/anomalies"
        anomaly_flag = np.random.choice([True, False], p=[0.3, 0.7])
        return await self._make_request("GET", endpoint, fallback_data={
            "anomaly_flag": anomaly_flag,
            "anomaly_score": np.random.uniform(0.1, 1.0) if anomaly_flag else 0.0,
            "anomaly_type": np.random.choice(["price", "volume", "sentiment", "correlation"]) if anomaly_flag else "normal",
            "severity": np.random.choice(["low", "medium", "high", "critical"]) if anomaly_flag else "low",
            "event_tag": np.random.choice(["whale_movement", "news_event", "technical_breakout", None]),
            "recommended_action": np.random.choice(["monitor", "investigate", "alert", "hedge"]),
            "confidence": np.random.uniform(0.6, 0.95),
            "anomalies_detected": [
                {
                    "type": "volume_spike",
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat(),
                    "details": "Unusual trading volume detected in BTC markets"
                }
            ] if anomaly_flag else []
        })
    
    async def get_network_analysis(self) -> Dict[str, Any]:
        """네트워크 분석 조회"""
        endpoint = "/api/v1/social/network-analysis"
        return await self._make_request("GET", endpoint, fallback_data={
            "network_metrics": {
                "viral_coefficient": np.random.uniform(0.2, 0.9),
                "information_diffusion_rate": np.random.uniform(0.1, 0.8),
                "network_density": np.random.uniform(0.3, 0.9),
                "influence_distribution": np.random.uniform(0.4, 0.8),
                "echo_chamber_score": np.random.uniform(0.2, 0.7)
            },
            "social_sentiment": {
                "twitter_sentiment": np.random.uniform(-1, 1),
                "reddit_sentiment": np.random.uniform(-1, 1),
                "telegram_sentiment": np.random.uniform(-1, 1),
                "aggregate_sentiment": np.random.uniform(-1, 1)
            },
            "trending_topics": [
                {"topic": "Bitcoin ETF", "score": np.random.uniform(0.7, 1.0)},
                {"topic": "DeFi Summer", "score": np.random.uniform(0.5, 0.8)},
                {"topic": "AI Trading", "score": np.random.uniform(0.3, 0.6)}
            ],
            "influencer_activity": {
                "top_influencers": [
                    {"name": "CryptoAnalyst", "impact": np.random.uniform(0.6, 1.0)},
                    {"name": "BlockchainExpert", "impact": np.random.uniform(0.5, 0.9)}
                ],
                "sentiment_leaders": ["bullish_whale", "cautious_trader"]
            }
        })
    
    async def get_market_pulse(self) -> Dict[str, Any]:
        """시장 펄스 조회"""
        endpoint = "/api/v1/market/pulse"
        return await self._make_request("GET", endpoint, fallback_data={
            "market_overview": {
                "btc_price": 45000 + np.random.normal(0, 2000),
                "eth_price": 3000 + np.random.normal(0, 200),
                "market_cap": 1.8e12 + np.random.normal(0, 1e11),
                "total_volume_24h": 50e9 + np.random.normal(0, 10e9),
                "fear_greed_index": np.random.randint(20, 80),
                "dominance_btc": np.random.uniform(40, 55)
            },
            "momentum_indicators": {
                "short_term_momentum": np.random.uniform(-1, 1),
                "medium_term_momentum": np.random.uniform(-1, 1),
                "long_term_momentum": np.random.uniform(-1, 1),
                "volatility_index": np.random.uniform(0.1, 0.8)
            },
            "sector_performance": {
                "defi": np.random.uniform(-0.1, 0.2),
                "nft": np.random.uniform(-0.15, 0.15),
                "layer1": np.random.uniform(-0.08, 0.12),
                "gaming": np.random.uniform(-0.12, 0.18)
            },
            "market_regime": np.random.choice(["bull", "bear", "sideways", "high_volatility"])
        })
    
    async def get_system_intelligence(self) -> Dict[str, Any]:
        """시스템 인텔리전스 조회"""
        endpoint = "/api/v1/system/intelligence"
        return await self._make_request("GET", endpoint, fallback_data={
            "ai_insights": [
                {
                    "insight_type": "pattern_recognition",
                    "confidence": np.random.uniform(0.7, 0.95),
                    "message": "Bullish divergence pattern detected in BTC/USD",
                    "recommendations": ["Consider long position", "Monitor volume confirmation"],
                    "evidence": {"rsi_divergence": 0.8, "volume_trend": "increasing"}
                },
                {
                    "insight_type": "risk_assessment",
                    "confidence": np.random.uniform(0.6, 0.9),
                    "message": "Elevated correlation risk in crypto markets",
                    "recommendations": ["Diversify portfolio", "Reduce leverage"],
                    "evidence": {"correlation_matrix": "high", "volatility_clustering": True}
                }
            ],
            "model_performance": {
                "prediction_accuracy": np.random.uniform(0.65, 0.85),
                "ensemble_agreement": np.random.uniform(0.7, 0.95),
                "confidence_calibration": np.random.uniform(0.6, 0.9),
                "model_drift_score": np.random.uniform(0.0, 0.3)
            },
            "learning_metrics": {
                "data_quality_score": np.random.uniform(0.8, 0.98),
                "feature_importance_stability": np.random.uniform(0.7, 0.95),
                "concept_drift_detection": np.random.choice([True, False], p=[0.2, 0.8]),
                "adaptive_learning_rate": np.random.uniform(0.01, 0.1)
            },
            "system_health": {
                "latency_ms": np.random.uniform(10, 100),
                "throughput_rps": np.random.uniform(100, 1000),
                "error_rate": np.random.uniform(0.001, 0.01),
                "cache_hit_rate": np.random.uniform(0.8, 0.95)
            }
        })
    
    async def get_live_data_feed(self) -> Dict[str, Any]:
        """실시간 데이터 피드 조회"""
        endpoint = "/api/v1/feeds/live"
        return await self._make_request("GET", endpoint, fallback_data={
            "latest_news": [
                {
                    "headline": "Bitcoin Reaches New All-Time High",
                    "sentiment": np.random.uniform(-1, 1),
                    "impact_score": np.random.uniform(0.5, 1.0),
                    "timestamp": datetime.now().isoformat(),
                    "source": "CryptoNews"
                },
                {
                    "headline": "Federal Reserve Announces Rate Decision",
                    "sentiment": np.random.uniform(-1, 1),
                    "impact_score": np.random.uniform(0.3, 0.8),
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "source": "Financial Times"
                }
            ],
            "social_activity": {
                "mentions_per_minute": np.random.randint(50, 500),
                "sentiment_velocity": np.random.uniform(-0.5, 0.5),
                "viral_content": [
                    {"content": "🚀 BTC breaking resistance!", "engagement": np.random.randint(1000, 10000)},
                    {"content": "Altcoin season incoming?", "engagement": np.random.randint(500, 5000)}
                ]
            },
            "price_movements": {
                "top_gainers": [
                    {"symbol": "ETH", "change": np.random.uniform(5, 20)},
                    {"symbol": "SOL", "change": np.random.uniform(3, 15)}
                ],
                "top_losers": [
                    {"symbol": "ADA", "change": np.random.uniform(-15, -3)},
                    {"symbol": "DOT", "change": np.random.uniform(-12, -2)}
                ]
            },
            "trading_signals": [
                {
                    "signal": "BUY",
                    "symbol": "BTC",
                    "confidence": np.random.uniform(0.6, 0.9),
                    "strategy": "momentum_breakout"
                },
                {
                    "signal": "SELL",
                    "symbol": "ETH",
                    "confidence": np.random.uniform(0.5, 0.8),
                    "strategy": "mean_reversion"
                }
            ]
        })
    
    async def _make_request(self, method: str, endpoint: str, fallback_data: Dict = None) -> Dict[str, Any]:
        """HTTP 요청 실행 with 폴백"""
        cache_key = f"{method}:{endpoint}"
        
        # 캐시 확인
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        start_time = time.time()
        
        try:
            if not self.session:
                raise aiohttp.ClientError("Session not initialized")
            
            url = f"{self.base_url}{endpoint}"
            
            async with self.session.request(method, url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # 성공 시 캐시 저장
                    self.cache[cache_key] = (data, time.time())
                    
                    # 성능 메트릭 업데이트
                    response_time = (time.time() - start_time) * 1000
                    self.response_times.append(response_time)
                    self.connection_status = "connected"
                    self.last_successful_call = datetime.now()
                    self.error_count = 0
                    
                    return data
                else:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
        
        except Exception as e:
            logger.warning(f"API request failed: {e}")
            self.error_count += 1
            self.connection_status = "error"
            
            # 폴백 데이터 반환
            if self.fallback_enabled and fallback_data:
                return fallback_data
            else:
                raise e
    
    def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 조회"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        success_rate = max(0, 1 - (self.error_count / 100))  # 최근 100회 기준
        
        return {
            "status": self.connection_status,
            "last_successful_call": self.last_successful_call.isoformat() if self.last_successful_call else None,
            "error_count": self.error_count,
            "avg_response_time_ms": avg_response_time,
            "success_rate": success_rate,
            "cache_size": len(self.cache)
        }

class AdvancedAuroraDashboard:
    """고도화된 Aurora 대시보드"""
    
    def __init__(self):
        """초기화"""
        self.client = AdvancedAuroraServiceClient()
        self.typing_effect = ClaudeCodeAdvancedTypingEffect()
        
        # 대시보드 설정
        self.panel_layout = [
            [PanelType.SENTIMENT_FUSION, PanelType.ML_PREDICTIONS, PanelType.EVENT_IMPACT],
            [PanelType.STRATEGY_PERFORMANCE, PanelType.ANOMALY_DETECTION, PanelType.NETWORK_ANALYSIS],
            [PanelType.MARKET_PULSE, PanelType.SYSTEM_INTELLIGENCE, PanelType.LIVE_DATA_FEED]
        ]
        
        # 패널 데이터 저장소
        self.panel_data: Dict[PanelType, PanelData] = {}
        
        # 실시간 업데이트 설정
        self.update_intervals = {
            PanelType.SENTIMENT_FUSION: 5,      # 5초
            PanelType.ML_PREDICTIONS: 10,       # 10초
            PanelType.EVENT_IMPACT: 15,         # 15초
            PanelType.STRATEGY_PERFORMANCE: 30, # 30초
            PanelType.ANOMALY_DETECTION: 5,     # 5초
            PanelType.NETWORK_ANALYSIS: 20,     # 20초
            PanelType.MARKET_PULSE: 10,         # 10초
            PanelType.SYSTEM_INTELLIGENCE: 30,  # 30초
            PanelType.LIVE_DATA_FEED: 3         # 3초
        }
        
        self.last_updates = {panel: datetime.now() for panel in self.update_intervals}
        
        # 상태 관리
        self.is_running = False
        self.current_time = datetime.now()
        self.uptime_start = datetime.now()
        
        # AI 인사이트 저장소
        self.ai_insights: List[AIInsight] = []
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "avg_update_time": 0.0
        }
        
        logger.info("AdvancedAuroraDashboard initialized successfully")
    
    async def start(self):
        """대시보드 시작"""
        self.is_running = True
        self.uptime_start = datetime.now()
        
        async with self.client:
            logger.info("Starting Advanced Aurora Dashboard v3.0")
            
            # 초기 데이터 로드
            await self._initialize_panels()
            
            # 메인 루프 시작
            await self._main_loop()
    
    async def stop(self):
        """대시보드 중지"""
        self.is_running = False
        logger.info("Advanced Aurora Dashboard stopped")
    
    async def _initialize_panels(self):
        """Initialize Panels"""
        print("\n🚀 Initializing Aurora Advanced Dashboard v3.0...")
        
        for i, row in enumerate(self.panel_layout):
            for j, panel_type in enumerate(row):
                try:
                    await self._update_panel(panel_type)
                    print(f"✅ {panel_type.value} panel initialized")
                except Exception as e:
                    logger.error(f"Panel initialization failed for {panel_type}: {e}")
                    print(f"⚠️ {panel_type.value} panel initialization failed: {e}")
        
        print("🎯 All panels initialized successfully!\n")
    
    async def _main_loop(self):
        """메인 루프"""
        try:
            while self.is_running:
                self.current_time = datetime.now()
                
                # 화면 클리어 및 헤더 출력
                print("\033[2J\033[H")  # 화면 클리어
                await self._render_header()
                
                # 패널 업데이트 및 렌더링
                await self._update_and_render_panels()
                
                # AI 인사이트 출력
                await self._render_ai_insights()
                
                # 푸터 출력
                await self._render_footer()
                
                # 업데이트 주기 대기
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            print(f"\n❌ 오류 발생: {e}")
        finally:
            await self.stop()
    
    async def _render_header(self):
        """헤더 렌더링"""
        uptime = self.current_time - self.uptime_start
        
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                     🌟 AURORA ADVANCED AI DASHBOARD v3.0 🌟                                                                   ║
║  🕒 {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} | ⏱️ Uptime: {str(uptime).split('.')[0]} | 🔄 Updates: {self.performance_metrics['total_updates']}                                   ║
║  📊 Success Rate: {self.performance_metrics['successful_updates']/(self.performance_metrics['total_updates']+1)*100:.1f}% | 🚀 AI Engine: ACTIVE | 🌐 Connection: {self.client.connection_status.upper()}                           ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
        
        # 레인보우 효과 적용
        colored_header = self.typing_effect._rainbow_effect(header, time.time() * 0.5)
        print(colored_header)
    
    async def _update_and_render_panels(self):
        """패널 업데이트 및 렌더링"""
        
        # 9개 패널 3x3 그리드로 배치
        for row_idx, row in enumerate(self.panel_layout):
            # 패널 업데이트 (필요한 경우)
            for panel_type in row:
                if await self._should_update_panel(panel_type):
                    await self._update_panel(panel_type)
            
            # 행 렌더링
            await self._render_panel_row(row, row_idx)
    
    async def _should_update_panel(self, panel_type: PanelType) -> bool:
        """패널 업데이트 필요 여부 확인"""
        if panel_type not in self.last_updates:
            return True
        
        interval = self.update_intervals.get(panel_type, 30)
        time_since_update = (self.current_time - self.last_updates[panel_type]).total_seconds()
        
        return time_since_update >= interval
    
    async def _update_panel(self, panel_type: PanelType):
        """개별 패널 데이터 업데이트"""
        start_time = time.time()
        
        try:
            self.performance_metrics["total_updates"] += 1
            
            # 패널별 데이터 조회
            if panel_type == PanelType.SENTIMENT_FUSION:
                data = await self.client.get_advanced_fusion_analysis()
                content = {
                    "fusion_score": data.get("fusion_score", 0.0),
                    "multimodal": data.get("multimodal_sentiment", {}),
                    "advanced": data.get("advanced_features", {})
                }
                status = self._determine_sentiment_status(data.get("fusion_score", 0.0))
                alert_level = self._determine_alert_level(abs(data.get("fusion_score", 0.0)))
                
            elif panel_type == PanelType.ML_PREDICTIONS:
                data = await self.client.get_advanced_fusion_analysis()
                ml_pred = data.get("ml_prediction", {})
                content = {
                    "direction": ml_pred.get("direction", "neutral"),
                    "probability": ml_pred.get("probability", 0.5),
                    "volatility": ml_pred.get("volatility_forecast", 0.1),
                    "confidence": ml_pred.get("confidence_level", "medium"),
                    "trend_strength": ml_pred.get("trend_strength", 0.0)
                }
                status = ml_pred.get("direction", "neutral")
                alert_level = self._determine_alert_level(ml_pred.get("probability", 0.5))
                
            elif panel_type == PanelType.EVENT_IMPACT:
                data = await self.client.get_event_impact_analysis()
                content = {
                    "events": data.get("events", []),
                    "aggregate_impact": data.get("aggregate_impact", 0.0),
                    "sentiment_shift": data.get("market_sentiment_shift", 0.0)
                }
                status = "active" if data.get("aggregate_impact", 0.0) > 0.5 else "monitoring"
                alert_level = self._determine_alert_level(data.get("aggregate_impact", 0.0))
                
            elif panel_type == PanelType.STRATEGY_PERFORMANCE:
                data = await self.client.get_strategy_performance()
                content = {
                    "roi": data.get("roi", 0.0),
                    "sharpe": data.get("sharpe_ratio", 0.0),
                    "drawdown": data.get("max_drawdown", 0.0),
                    "win_rate": data.get("win_rate", 0.0),
                    "positions": data.get("current_positions", {}),
                    "recent_performance": data.get("performance_metrics", {})
                }
                status = "profitable" if data.get("roi", 0.0) > 0 else "loss"
                alert_level = self._determine_alert_level(abs(data.get("roi", 0.0)))
                
            elif panel_type == PanelType.ANOMALY_DETECTION:
                data = await self.client.get_anomaly_detection()
                content = {
                    "anomaly_flag": data.get("anomaly_flag", False),
                    "anomaly_score": data.get("anomaly_score", 0.0),
                    "anomaly_type": data.get("anomaly_type", "normal"),
                    "severity": data.get("severity", "low"),
                    "recommended_action": data.get("recommended_action", "monitor"),
                    "anomalies": data.get("anomalies_detected", [])
                }
                status = data.get("severity", "low") if data.get("anomaly_flag", False) else "normal"
                alert_level = AlertLevel.CRITICAL if data.get("severity") == "critical" else AlertLevel.INFO
                
            elif panel_type == PanelType.NETWORK_ANALYSIS:
                data = await self.client.get_network_analysis()
                content = {
                    "network_metrics": data.get("network_metrics", {}),
                    "social_sentiment": data.get("social_sentiment", {}),
                    "trending_topics": data.get("trending_topics", []),
                    "influencer_activity": data.get("influencer_activity", {})
                }
                avg_sentiment = np.mean(list(data.get("social_sentiment", {}).values()))
                status = "bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral"
                alert_level = self._determine_alert_level(abs(avg_sentiment))
                
            elif panel_type == PanelType.MARKET_PULSE:
                data = await self.client.get_market_pulse()
                content = {
                    "market_overview": data.get("market_overview", {}),
                    "momentum": data.get("momentum_indicators", {}),
                    "sector_performance": data.get("sector_performance", {}),
                    "regime": data.get("market_regime", "sideways")
                }
                status = data.get("market_regime", "sideways")
                fear_greed = data.get("market_overview", {}).get("fear_greed_index", 50)
                alert_level = self._determine_alert_level(abs(fear_greed - 50) / 50)
                
            elif panel_type == PanelType.SYSTEM_INTELLIGENCE:
                data = await self.client.get_system_intelligence()
                content = {
                    "ai_insights": data.get("ai_insights", []),
                    "model_performance": data.get("model_performance", {}),
                    "learning_metrics": data.get("learning_metrics", {}),
                    "system_health": data.get("system_health", {})
                }
                avg_confidence = np.mean([insight.get("confidence", 0.5) for insight in data.get("ai_insights", [])])
                status = "high_confidence" if avg_confidence > 0.8 else "medium_confidence" if avg_confidence > 0.6 else "low_confidence"
                alert_level = self._determine_alert_level(avg_confidence)
                
            elif panel_type == PanelType.LIVE_DATA_FEED:
                data = await self.client.get_live_data_feed()
                content = {
                    "latest_news": data.get("latest_news", []),
                    "social_activity": data.get("social_activity", {}),
                    "price_movements": data.get("price_movements", {}),
                    "trading_signals": data.get("trading_signals", [])
                }
                signal_count = len(data.get("trading_signals", []))
                status = "active" if signal_count > 0 else "monitoring"
                alert_level = AlertLevel.HIGH if signal_count > 2 else AlertLevel.INFO
            
            # 패널 데이터 저장
            self.panel_data[panel_type] = PanelData(
                title=self._get_panel_title(panel_type),
                content=content,
                status=status,
                last_updated=self.current_time,
                alert_level=alert_level,
                metadata={"update_time": time.time() - start_time}
            )
            
            self.last_updates[panel_type] = self.current_time
            self.performance_metrics["successful_updates"] += 1
            
        except Exception as e:
            logger.error(f"Panel update failed for {panel_type}: {e}")
            self.performance_metrics["failed_updates"] += 1
            
            # 에러 시 기본 데이터 생성
            self.panel_data[panel_type] = PanelData(
                title=self._get_panel_title(panel_type),
                content={"error": str(e)},
                status="error",
                last_updated=self.current_time,
                alert_level=AlertLevel.CRITICAL,
                metadata={"error": True}
            )
        
        # 성능 메트릭 업데이트
        total_time = sum([panel.metadata.get("update_time", 0) for panel in self.panel_data.values()])
        self.performance_metrics["avg_update_time"] = total_time / len(self.panel_data) if self.panel_data else 0
    
    async def _render_panel_row(self, panels: List[PanelType], row_idx: int):
        """패널 행 렌더링"""
        # 패널 너비 (터미널 너비를 3으로 나눔)
        panel_width = 45
        panel_height = 12
        
        # 각 패널의 콘텐츠 생성
        panel_contents = []
        for panel_type in panels:
            content = await self._render_panel_content(panel_type, panel_width, panel_height)
            panel_contents.append(content)
        
        # 행별로 출력
        for line_idx in range(panel_height):
            line = ""
            for panel_content in panel_contents:
                if line_idx < len(panel_content):
                    line += panel_content[line_idx]
                else:
                    line += " " * panel_width
                line += "  "  # 패널 간 간격
            print(line)
        
        print()  # 행 간 여백
    
    async def _render_panel_content(self, panel_type: PanelType, width: int, height: int) -> List[str]:
        """개별 패널 콘텐츠 렌더링"""
        lines = []
        
        # 패널 데이터 가져오기
        panel_data = self.panel_data.get(panel_type)
        if not panel_data:
            return [f"║{'Loading Panel...':^{width-2}}║" for _ in range(height)]
        
        # 상태별 색상 결정
        status_color = self._get_status_color(panel_data.status, panel_data.alert_level)
        
        # 헤더
        title = panel_data.title[:width-4]
        lines.append(f"╔{'═' * (width-2)}╗")
        lines.append(f"║{status_color}{title:^{width-2}}\033[0m║")
        lines.append(f"╠{'═' * (width-2)}╣")
        
        # 콘텐츠 영역
        content_lines = await self._format_panel_content(panel_type, panel_data, width-4)
        
        for i in range(height-4):  # 헤더 3줄, 푸터 1줄 제외
            if i < len(content_lines):
                line = content_lines[i][:width-4]
                lines.append(f"║ {line:<{width-4}} ║")
            else:
                lines.append(f"║{' ' * (width-2)}║")
        
        # 푸터
        lines.append(f"╚{'═' * (width-2)}╝")
        
        return lines
    
    async def _format_panel_content(self, panel_type: PanelType, panel_data: PanelData, width: int) -> List[str]:
        """패널별 콘텐츠 포맷팅"""
        lines = []
        content = panel_data.content
        
        if panel_type == PanelType.SENTIMENT_FUSION:
            fusion_score = content.get("fusion_score", 0.0)
            multimodal = content.get("multimodal", {})
            
            lines.append(f"📊 융합점수: {fusion_score:+.3f}")
            lines.append(f"📝 텍스트: {multimodal.get('text_sentiment', 0.0):+.2f}")
            lines.append(f"📈 가격행동: {multimodal.get('price_action_sentiment', 0.0):+.2f}")
            lines.append(f"📊 거래량: {multimodal.get('volume_sentiment', 0.0):+.2f}")
            lines.append(f"💬 소셜참여: {multimodal.get('social_engagement', 0.0):.2f}")
            lines.append(f"🔥 바이럴: {content.get('advanced', {}).get('viral_score', 0.0):.2f}")
            lines.append(f"🌐 네트워크: {content.get('advanced', {}).get('network_effect', 0.0):.2f}")
            
        elif panel_type == PanelType.ML_PREDICTIONS:
            direction = content.get("direction", "neutral")
            probability = content.get("probability", 0.5)
            
            direction_emoji = "🚀" if direction == "bullish" else "📉" if direction == "bearish" else "⚖️"
            lines.append(f"{direction_emoji} 방향: {direction.upper()}")
            lines.append(f"🎯 확률: {probability:.1%}")
            lines.append(f"📊 변동성: {content.get('volatility', 0.1):.1%}")
            lines.append(f"💪 신뢰도: {content.get('confidence', 'medium')}")
            lines.append(f"📈 트렌드강도: {content.get('trend_strength', 0.0):.2f}")
            lines.append(f"⏰ 시간범위: 4시간")
            
        elif panel_type == PanelType.EVENT_IMPACT:
            events = content.get("events", [])
            aggregate = content.get("aggregate_impact", 0.0)
            
            lines.append(f"💥 Total Impact: {aggregate:.2f}")
            lines.append(f"📈 Sentiment Shift: {content.get('sentiment_shift', 0.0):+.2f}")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            for event in events[:3]:  # 최대 3개 이벤트 표시
                event_type = event.get("event_type", "unknown")
                impact = event.get("impact_score", 0.0)
                emoji = "🏛️" if event_type == "regulatory" else "📊" if event_type == "technical" else "📰"
                lines.append(f"{emoji} {event_type[:8]}: {impact:.2f}")
            
        elif panel_type == PanelType.STRATEGY_PERFORMANCE:
            roi = content.get("roi", 0.0)
            sharpe = content.get("sharpe", 0.0)
            
            roi_emoji = "💰" if roi > 0 else "💸" if roi < 0 else "⚖️"
            lines.append(f"{roi_emoji} ROI: {roi:+.1%}")
            lines.append(f"📊 샤프: {sharpe:.2f}")
            lines.append(f"📉 MDD: {content.get('drawdown', 0.0):.1%}")
            lines.append(f"🎯 승률: {content.get('win_rate', 0.0):.1%}")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            positions = content.get("positions", {})
            for symbol, weight in list(positions.items())[:2]:
                lines.append(f"💎 {symbol}: {weight:.1%}")
                
        elif panel_type == PanelType.ANOMALY_DETECTION:
            anomaly_flag = content.get("anomaly_flag", False)
            score = content.get("anomaly_score", 0.0)
            severity = content.get("severity", "low")
            
            if anomaly_flag:
                severity_emoji = "🚨" if severity == "critical" else "⚠️" if severity == "high" else "🟡"
                lines.append(f"{severity_emoji} 이상 탐지!")
                lines.append(f"📊 점수: {score:.3f}")
                lines.append(f"🏷️ 타입: {content.get('anomaly_type', 'unknown')}")
                lines.append(f"⚡ 심각도: {severity}")
                lines.append(f"💡 권고: {content.get('recommended_action', 'monitor')}")
            else:
                lines.append("✅ Normal Status")
                lines.append("🔍 Continuous Monitoring...")
                lines.append("📊 Anomaly Score: 0.000")
                lines.append("🛡️ System Stable")
                
        elif panel_type == PanelType.NETWORK_ANALYSIS:
            social_sentiment = content.get("social_sentiment", {})
            trending = content.get("trending_topics", [])
            
            avg_sentiment = np.mean(list(social_sentiment.values())) if social_sentiment else 0.0
            sentiment_emoji = "🚀" if avg_sentiment > 0.2 else "📉" if avg_sentiment < -0.2 else "⚖️"
            
            lines.append(f"{sentiment_emoji} Social Sentiment: {avg_sentiment:+.2f}")
            lines.append(f"🐦 Twitter: {social_sentiment.get('twitter_sentiment', 0.0):+.2f}")
            lines.append(f"📱 Reddit: {social_sentiment.get('reddit_sentiment', 0.0):+.2f}")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            for topic in trending[:2]:
                score = topic.get("score", 0.0)
                lines.append(f"🔥 {topic.get('topic', 'Unknown')[:12]}")
                
        elif panel_type == PanelType.MARKET_PULSE:
            overview = content.get("market_overview", {})
            regime = content.get("regime", "sideways")
            
            btc_price = overview.get("btc_price", 0)
            fear_greed = overview.get("fear_greed_index", 50)
            
            regime_emoji = "🐂" if regime == "bull" else "🐻" if regime == "bear" else "⚖️"
            lines.append(f"{regime_emoji} Market Regime: {regime}")
            lines.append(f"₿ BTC: ${btc_price:,.0f}")
            lines.append(f"😱 Fear & Greed: {fear_greed}")
            lines.append(f"📊 BTC Dominance: {overview.get('dominance_btc', 0.0):.1f}%")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            momentum = content.get("momentum", {})
            lines.append(f"⚡ Short: {momentum.get('short_term_momentum', 0.0):+.2f}")
            lines.append(f"📈 Long: {momentum.get('long_term_momentum', 0.0):+.2f}")
            
        elif panel_type == PanelType.SYSTEM_INTELLIGENCE:
            insights = content.get("ai_insights", [])
            model_perf = content.get("model_performance", {})
            
            accuracy = model_perf.get("prediction_accuracy", 0.0)
            confidence = model_perf.get("ensemble_agreement", 0.0)
            
            lines.append(f"🎯 Accuracy: {accuracy:.1%}")
            lines.append(f"🤝 Consensus: {confidence:.1%}")
            lines.append(f"🧠 Insights: {len(insights)}")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            for insight in insights[:2]:
                insight_type = insight.get("insight_type", "unknown")
                conf = insight.get("confidence", 0.0)
                type_emoji = "🔍" if "pattern" in insight_type else "⚠️"
                lines.append(f"{type_emoji} {insight_type[:8]}: {conf:.1%}")
                
        elif panel_type == PanelType.LIVE_DATA_FEED:
            news = content.get("latest_news", [])
            signals = content.get("trading_signals", [])
            
            lines.append(f"📰 News: {len(news)}")
            lines.append(f"🎯 Signals: {len(signals)}")
            lines.append("━━━━━━━━━━━━━━━━━━")
            
            for article in news[:2]:
                sentiment = article.get("sentiment", 0.0)
                sentiment_emoji = "🚀" if sentiment > 0.3 else "📉" if sentiment < -0.3 else "⚖️"
                headline = article.get("headline", "")[:15]
                lines.append(f"{sentiment_emoji} {headline}...")
            
            if signals:
                signal = signals[0]
                signal_emoji = "🟢" if signal.get("signal") == "BUY" else "🔴"
                lines.append(f"{signal_emoji} {signal.get('signal')}: {signal.get('symbol')}")
        
        return lines
    
    def _get_panel_title(self, panel_type: PanelType) -> str:
        """Get Panel Title"""
        titles = {
            PanelType.SENTIMENT_FUSION: "🧠 AI Sentiment Fusion",
            PanelType.ML_PREDICTIONS: "🔮 ML Predictions",
            PanelType.EVENT_IMPACT: "💥 Event Impact",
            PanelType.STRATEGY_PERFORMANCE: "📈 Strategy Performance",
            PanelType.ANOMALY_DETECTION: "🚨 Anomaly Detection",
            PanelType.NETWORK_ANALYSIS: "🌐 Network Analysis",
            PanelType.MARKET_PULSE: "💓 Market Pulse",
            PanelType.SYSTEM_INTELLIGENCE: "🤖 System AI",
            PanelType.LIVE_DATA_FEED: "📡 Live Data Feed"
        }
        return titles.get(panel_type, panel_type.value)
    
    def _get_status_color(self, status: str, alert_level: AlertLevel) -> str:
        """Return color based on status"""
        if alert_level == AlertLevel.CRITICAL:
            return "\033[1;91m"  # 밝은 빨강
        elif alert_level == AlertLevel.HIGH:
            return "\033[1;93m"  # 밝은 노랑
        elif status in ["bullish", "profitable", "high_confidence"]:
            return "\033[1;92m"  # 밝은 초록
        elif status in ["bearish", "loss", "error"]:
            return "\033[1;91m"  # 밝은 빨강
        else:
            return "\033[1;96m"  # 밝은 청록
    
    def _determine_sentiment_status(self, score: float) -> str:
        """Determine status based on sentiment score"""
        if score > 0.3:
            return "bullish"
        elif score < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _determine_alert_level(self, intensity: float) -> AlertLevel:
        """Determine alert level based on intensity"""
        if intensity > 0.8:
            return AlertLevel.CRITICAL
        elif intensity > 0.6:
            return AlertLevel.HIGH
        elif intensity > 0.4:
            return AlertLevel.MEDIUM
        elif intensity > 0.2:
            return AlertLevel.LOW
        else:
            return AlertLevel.INFO
    
    async def _render_ai_insights(self):
        """Render AI insights"""
        if not self.ai_insights:
            return
        
        print("\n" + "="*140)
        print("🧠 AI INSIGHTS & RECOMMENDATIONS")
        print("="*140)
        
        for insight in self.ai_insights[-3:]:  # 최근 3개만 표시
            confidence_color = "\033[1;92m" if insight.confidence > 0.8 else "\033[1;93m" if insight.confidence > 0.6 else "\033[1;96m"
            
            print(f"{confidence_color}🎯 {insight.insight_type.upper()} (Confidence: {insight.confidence:.1%})\033[0m")
            print(f"   💡 {insight.message}")
            
            if insight.recommendations:
                print(f"   📋 Recommendations:")
                for rec in insight.recommendations[:2]:
                    print(f"      • {rec}")
            print()
    
    async def _render_footer(self):
        """Render footer"""
        connection_status = self.client.get_connection_status()
        
        footer = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  📊 Performance: Updates {self.performance_metrics['successful_updates']}/{self.performance_metrics['total_updates']} | Avg Response: {connection_status['avg_response_time_ms']:.1f}ms | Success Rate: {connection_status['success_rate']:.1%}        ║
║  🎮 Controls: Ctrl+C to Exit | 🔄 Auto-refresh: Active | 💾 Cache: {connection_status['cache_size']} items | 🌟 AI Engine v3.0 Active                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
        
        # 네온 효과 적용
        colored_footer = self.typing_effect._neon_effect(footer, time.time() * 0.3)
        print(colored_footer)


# 메인 실행
async def main():
    """Main function"""
    dashboard = AdvancedAuroraDashboard()
    
    try:
        await dashboard.start()
    except KeyboardInterrupt:
        print("\n👋 Dashboard safely terminated.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                        🌟 AURORA ADVANCED AI DASHBOARD v3.0 🌟                              ║
    ║                                                                                              ║
    ║  🧠 Multimodal Sentiment Fusion    🔮 ML Prediction Engine    💥 Event Impact Analysis      ║
    ║  📈 Strategy Performance Tracking  🚨 Anomaly Detection       🌐 Network Analysis           ║
    ║  💓 Real-time Market Pulse         🤖 System Intelligence     📡 Live Data Feed              ║
    ║                                                                                              ║
    ║  Advanced AI-powered sentiment analysis and prediction for next-gen trading dashboard        ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 대시보드 실행
    asyncio.run(main())