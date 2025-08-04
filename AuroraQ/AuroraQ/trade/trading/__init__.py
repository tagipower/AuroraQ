"""
AuroraQ VPS 실전매매 시스템
Production 환경에서 검증된 실전매매 모듈을 VPS에 최적화

VPS 최적화된 매매 엔진:
- 통합 로깅 시스템 연동
- ONNX 센티먼트 분석 통합  
- PPO + 룰 기반 하이브리드 전략
- 실시간 리스크 관리
- 메모리 효율적 포지션 관리
"""

# VPS 최적화 실전매매 모듈들
try:
    from trading.vps_realtime_system import VPSRealtimeSystem, VPSTradingConfig
    from trading.vps_position_manager import VPSPositionManager, VPSPosition, VPSTradingLimits
    from trading.vps_order_manager import VPSOrderManager, VPSOrder, OrderType, OrderStatus, OrderSide
    from trading.vps_strategy_adapter import VPSStrategyAdapter
    from trading.vps_market_data import VPSMarketDataProvider, VPSMarketDataPoint
except ImportError as e:
    # 상대 임포트 실패 시 로깅만 하고 계속 진행
    print(f"Warning: Some VPS trading modules could not be imported: {e}")
    # 기본 클래스들 정의
    class VPSRealtimeSystem: pass
    class VPSTradingConfig: pass
    class VPSPositionManager: pass
    class VPSPosition: pass
    class VPSTradingLimits: pass
    class VPSOrderManager: pass
    class VPSOrder: pass
    class OrderType: pass
    class OrderStatus: pass
    class OrderSide: pass
    class VPSStrategyAdapter: pass
    class VPSMarketDataProvider: pass
    class VPSMarketDataPoint: pass

# 기존 모듈들 (호환성 유지)
try:
    from trading.realtime_engine import (
        AuroraQTradingEngine,
        MarketDataHandler,
        SentimentAnalyzer,
        PPOAgent,
        RuleBasedEngine,
        RiskManager,
        OrderExecutor,
        PositionManager,
        TradingSignal,
        Position,
        TradingStats,
        MarketData
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

__version__ = "3.0.0"
__author__ = "AuroraQ VPS Team"

# VPS Trading module info
VPS_TRADING_INFO = {
    "optimized_for": "48GB VPS",
    "memory_usage": "2-3GB",
    "update_interval": "30 seconds", 
    "unified_logging": True,
    "risk_management": True,
    "paper_trading": True,
    "live_trading": True,
    "websocket_port": 8002,
    "legacy_compatible": LEGACY_AVAILABLE
}

# 메인 export (VPS 최적화 버전)
__all__ = [
    # VPS 최적화 모듈들
    "VPSRealtimeSystem",
    "VPSTradingConfig", 
    "VPSPositionManager",
    "VPSPosition",
    "VPSTradingLimits",
    "VPSOrderManager",
    "VPSOrder",
    "OrderType",
    "OrderStatus", 
    "OrderSide",
    "VPSStrategyAdapter",
    "VPSMarketDataProvider",
    "VPSMarketDataPoint",
    "VPS_TRADING_INFO"
]

# 기존 모듈들 (호환성)
if LEGACY_AVAILABLE:
    __all__.extend([
        "AuroraQTradingEngine",
        "MarketDataHandler",
        "SentimentAnalyzer", 
        "PPOAgent",
        "RuleBasedEngine",
        "RiskManager",
        "OrderExecutor",
        "PositionManager",
        "TradingSignal",
        "Position",
        "TradingStats",
        "MarketData"
    ])

# 편의 함수들
def create_vps_trading_system(config=None):
    """VPS 최적화 실전매매 시스템 생성"""
    from trading.vps_realtime_system import create_vps_trading_system
    return create_vps_trading_system(config)

def create_paper_trading_config():
    """페이퍼 트레이딩 설정 생성"""
    return VPSTradingConfig(
        mode="paper",
        symbol="BTCUSDT",
        enable_sentiment=True,
        enable_unified_logging=True,
        max_position_size=0.05,  # 5% 포지션
        max_daily_trades=5
    )

def create_live_trading_config():
    """실제 거래 설정 생성"""
    return VPSTradingConfig(
        mode="live",
        symbol="BTCUSDT", 
        enable_sentiment=True,
        enable_unified_logging=True,
        max_position_size=0.02,  # 2% 포지션 (보수적)
        max_daily_trades=3,
        risk_tolerance="conservative"
    )