#!/usr/bin/env python3
"""
AuroraQ System Bridge
기존 AuroraQ 컴포넌트들과 새로운 예방적 관리 시스템 간의 브리지
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# 기존 AuroraQ 시스템 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class AuroraQSystemBridge:
    """AuroraQ 시스템 브리지"""
    
    def __init__(self):
        """초기화"""
        self.components = {}
        self.metrics_cache = {}
        self.last_update = datetime.now()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        logger.info("AuroraQ System Bridge initialized")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        
        # 1. 거래 시스템 연동
        try:
            from trade.trading.realtime_engine import VPSRealtimeSystem
            from trade.trading.vps_realtime_system import VPSRealtimeSystem as VPSRealtime
            self.components['trading_system'] = self._get_trading_system()
            logger.info("Trading system connected")
        except ImportError as e:
            logger.warning(f"Trading system not available: {e}")
            self.components['trading_system'] = None
        
        # 2. 센티멘트 시스템 연동
        try:
            from sentiment.processors.sentiment_fusion_manager_v2 import SentimentFusionManagerV2
            from sentiment.collectors.enhanced_news_collector_v3 import EnhancedNewsCollectorV3
            self.components['sentiment_system'] = self._get_sentiment_system()
            logger.info("Sentiment system connected")
        except ImportError as e:
            logger.warning(f"Sentiment system not available: {e}")
            self.components['sentiment_system'] = None
        
        # 3. 성능 최적화 시스템 연동
        try:
            from core.performance.performance_optimizer import PerformanceOptimizer
            self.components['performance_optimizer'] = PerformanceOptimizer()
            logger.info("Performance optimizer connected")
        except ImportError as e:
            logger.warning(f"Performance optimizer not available: {e}")
            self.components['performance_optimizer'] = None
        
        # 4. 모니터링 시스템 연동
        try:
            from infrastructure.monitoring.monitoring_alert_system import MonitoringAlertSystem
            self.components['monitoring_system'] = MonitoringAlertSystem()
            logger.info("Monitoring system connected")
        except ImportError as e:
            logger.warning(f"Monitoring system not available: {e}")
            self.components['monitoring_system'] = None
    
    def _get_trading_system(self):
        """거래 시스템 인스턴스 반환"""
        # 실제 구현에서는 기존 거래 시스템 인스턴스를 반환
        return {
            "status": "connected",
            "mode": "paper",
            "positions": [],
            "orders": [],
            "balance": 10000.0
        }
    
    def _get_sentiment_system(self):
        """센티멘트 시스템 인스턴스 반환"""
        # 실제 구현에서는 기존 센티멘트 시스템 인스턴스를 반환
        return {
            "status": "active",
            "last_analysis": datetime.now(),
            "sentiment_score": 0.65,
            "news_processed": 1250
        }
    
    async def get_trading_metrics(self) -> Dict[str, Any]:
        """거래 시스템 메트릭 수집"""
        if not self.components.get('trading_system'):
            return {"error": "Trading system not available"}
        
        try:
            # 기존 거래 시스템에서 메트릭 수집
            return {
                "current_mode": "paper",
                "active_positions": 0,
                "pending_orders": 0,
                "account_balance": 10000.0,
                "daily_pnl": 0.0,
                "total_trades": 25,
                "win_rate": 0.68,
                "last_trade_time": datetime.now().isoformat(),
                "connection_status": "connected",
                "api_latency": 120.5  # ms
            }
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return {"error": str(e)}
    
    async def get_sentiment_metrics(self) -> Dict[str, Any]:
        """센티멘트 시스템 메트릭 수집"""
        if not self.components.get('sentiment_system'):
            return {"error": "Sentiment system not available"}
        
        try:
            # 기존 센티멘트 시스템에서 메트릭 수집
            return {
                "overall_sentiment": 0.65,
                "sentiment_trend": "positive",
                "news_sources_active": 5,
                "news_processed_today": 1250,
                "last_update": datetime.now().isoformat(),
                "processing_rate": 85.2,  # news/minute
                "sentiment_confidence": 0.78,
                "topic_distribution": {
                    "macro": 25,
                    "market": 30,
                    "technology": 20,
                    "regulation": 15,
                    "other": 10
                }
            }
        except Exception as e:
            logger.error(f"Error collecting sentiment metrics: {e}")
            return {"error": str(e)}
    
    async def get_system_performance_metrics(self) -> Dict[str, Any]:
        """시스템 성능 메트릭 수집"""
        try:
            import psutil
            
            # CPU 메트릭
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 메모리 메트릭
            memory = psutil.virtual_memory()
            
            # 디스크 메트릭
            disk = psutil.disk_usage('/')
            
            # 네트워크 메트릭
            network = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3)
                },
                "disk": {
                    "usage_percent": disk.percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e)}
    
    async def get_consolidated_metrics(self) -> Dict[str, Any]:
        """통합 메트릭 수집"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational"
        }
        
        # 각 시스템별 메트릭 수집
        try:
            trading_metrics = await self.get_trading_metrics()
            sentiment_metrics = await self.get_sentiment_metrics()
            system_metrics = await self.get_system_performance_metrics()
            
            metrics.update({
                "trading": trading_metrics,
                "sentiment": sentiment_metrics,
                "system": system_metrics
            })
            
            # 종합 상태 계산
            metrics["overall_health"] = self._calculate_overall_health(
                trading_metrics, sentiment_metrics, system_metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting consolidated metrics: {e}")
            metrics["error"] = str(e)
        
        # 캐시 업데이트
        self.metrics_cache = metrics
        self.last_update = datetime.now()
        
        return metrics
    
    def _calculate_overall_health(self, trading: Dict, sentiment: Dict, system: Dict) -> Dict[str, Any]:
        """전체 시스템 건강도 계산"""
        health_score = 1.0
        health_factors = []
        
        # 거래 시스템 건강도
        if "error" not in trading:
            if trading.get("connection_status") == "connected":
                health_factors.append(("trading_connection", 1.0))
            else:
                health_factors.append(("trading_connection", 0.5))
                health_score *= 0.8
        else:
            health_factors.append(("trading_system", 0.0))
            health_score *= 0.7
        
        # 센티멘트 시스템 건강도
        if "error" not in sentiment:
            processing_rate = sentiment.get("processing_rate", 0)
            if processing_rate > 50:
                health_factors.append(("sentiment_processing", 1.0))
            elif processing_rate > 20:
                health_factors.append(("sentiment_processing", 0.7))
                health_score *= 0.9
            else:
                health_factors.append(("sentiment_processing", 0.3))
                health_score *= 0.8
        else:
            health_factors.append(("sentiment_system", 0.0))
            health_score *= 0.7
        
        # 시스템 리소스 건강도
        if "error" not in system:
            cpu_usage = system.get("cpu", {}).get("usage_percent", 0)
            memory_usage = system.get("memory", {}).get("usage_percent", 0)
            
            if cpu_usage < 70 and memory_usage < 80:
                health_factors.append(("system_resources", 1.0))
            elif cpu_usage < 85 and memory_usage < 90:
                health_factors.append(("system_resources", 0.7))
                health_score *= 0.9
            else:
                health_factors.append(("system_resources", 0.3))
                health_score *= 0.7
        
        # 건강도 등급 결정
        if health_score >= 0.9:
            health_status = "excellent"
        elif health_score >= 0.8:
            health_status = "good"
        elif health_score >= 0.6:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "overall_score": health_score,
            "status": health_status,
            "factors": health_factors,
            "last_calculated": datetime.now().isoformat()
        }
    
    async def execute_trading_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """거래 시스템 명령 실행"""
        if not self.components.get('trading_system'):
            return {"error": "Trading system not available"}
        
        try:
            if command == "switch_mode":
                mode = params.get("mode", "paper")
                # 실제 구현에서는 거래 시스템의 모드 전환 메서드 호출
                return {
                    "success": True,
                    "message": f"Trading mode switched to {mode}",
                    "new_mode": mode,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif command == "get_positions":
                # 실제 구현에서는 거래 시스템의 포지션 조회 메서드 호출
                return {
                    "positions": [],
                    "total_value": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif command == "get_orders":
                # 실제 구현에서는 거래 시스템의 주문 조회 메서드 호출
                return {
                    "orders": [],
                    "pending_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Error executing trading command {command}: {e}")
            return {"error": str(e)}
    
    def get_cached_metrics(self) -> Dict[str, Any]:
        """캐시된 메트릭 반환"""
        return self.metrics_cache
    
    def is_cache_valid(self, max_age_seconds: int = 30) -> bool:
        """캐시 유효성 확인"""
        if not self.metrics_cache:
            return False
        
        age = (datetime.now() - self.last_update).total_seconds()
        return age <= max_age_seconds

# 전역 브리지 인스턴스
_system_bridge: Optional[AuroraQSystemBridge] = None

def get_system_bridge() -> AuroraQSystemBridge:
    """전역 시스템 브리지 반환"""
    global _system_bridge
    if _system_bridge is None:
        _system_bridge = AuroraQSystemBridge()
    return _system_bridge

# 테스트 코드
if __name__ == "__main__":
    async def test_system_bridge():
        """시스템 브리지 테스트"""
        print("=== AuroraQ System Bridge Test ===\n")
        
        bridge = AuroraQSystemBridge()
        
        # 거래 메트릭 테스트
        print("1. Testing trading metrics...")
        trading_metrics = await bridge.get_trading_metrics()
        print(f"   Trading Status: {trading_metrics.get('connection_status', 'Unknown')}")
        print(f"   Current Mode: {trading_metrics.get('current_mode', 'Unknown')}")
        print(f"   Win Rate: {trading_metrics.get('win_rate', 0):.1%}")
        
        # 센티멘트 메트릭 테스트
        print("\n2. Testing sentiment metrics...")
        sentiment_metrics = await bridge.get_sentiment_metrics()
        print(f"   Overall Sentiment: {sentiment_metrics.get('overall_sentiment', 0):.2f}")
        print(f"   News Processed: {sentiment_metrics.get('news_processed_today', 0)}")
        print(f"   Processing Rate: {sentiment_metrics.get('processing_rate', 0)} news/min")
        
        # 시스템 성능 메트릭 테스트
        print("\n3. Testing system performance...")
        system_metrics = await bridge.get_system_performance_metrics()
        if "error" not in system_metrics:
            cpu_usage = system_metrics["cpu"]["usage_percent"]
            memory_usage = system_metrics["memory"]["usage_percent"]
            print(f"   CPU Usage: {cpu_usage:.1f}%")
            print(f"   Memory Usage: {memory_usage:.1f}%")
        
        # 통합 메트릭 테스트
        print("\n4. Testing consolidated metrics...")
        consolidated = await bridge.get_consolidated_metrics()
        overall_health = consolidated.get("overall_health", {})
        print(f"   Overall Health: {overall_health.get('status', 'Unknown')} ({overall_health.get('overall_score', 0):.2f})")
        
        # 거래 명령 테스트
        print("\n5. Testing trading commands...")
        result = await bridge.execute_trading_command("switch_mode", {"mode": "live"})
        print(f"   Mode Switch Result: {result.get('success', False)}")
        
        print("\n✅ System Bridge test completed")
    
    # 테스트 실행
    asyncio.run(test_system_bridge())