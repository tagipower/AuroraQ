#!/usr/bin/env python3
"""
AuroraQ 독립 실행 런처
최소 의존성으로 AuroraQ만 실행하는 전용 런처
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent.parent))

# AuroraQ 전용 imports
from SharedCore.data_layer.unified_data_provider import UnifiedDataProvider
from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator
from AuroraQ.agent import AuroraQAgent, AuroraQConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuroraQStandalone:
    """
    AuroraQ 독립 실행 클래스
    최소 리소스로 AuroraQ만 운영
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.data_provider = None
        self.sentiment_aggregator = None
        self.aurora_agent = None
        
    async def initialize(self):
        """AuroraQ 단독 초기화"""
        logger.info("🎯 Initializing AuroraQ Standalone Mode...")
        
        try:
            # 1. 암호화폐 전용 데이터 프로바이더 초기화
            self.data_provider = UnifiedDataProvider(
                use_crypto=True,    # 암호화폐만
                use_macro=False     # 거시경제 데이터 비활성화
            )
            await self.data_provider.connect()
            
            # 2. 감정분석 엔진 초기화
            self.sentiment_aggregator = SentimentAggregator()
            
            # 3. AuroraQ Agent 초기화
            config = AuroraQConfig(
                initial_capital=100000.0,
                mode="simulation",
                max_position_size=0.2,
                risk_per_trade=0.02
            )
            
            self.aurora_agent = AuroraQAgent(
                config=config,
                data_provider=self.data_provider,
                sentiment_aggregator=self.sentiment_aggregator
            )
            
            await self.aurora_agent.initialize()
            
            logger.info("✅ AuroraQ Standalone initialized successfully")
            logger.info("📊 Active modules: Crypto data, Sentiment analysis, PPO+Rules strategies")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AuroraQ Standalone: {e}")
            raise
    
    async def run_trading(self):
        """실시간 트레이딩 실행"""
        logger.info("🚀 Starting AuroraQ Trading...")
        
        if not self.aurora_agent:
            await self.initialize()
        
        try:
            await self.aurora_agent.start_trading()
        except KeyboardInterrupt:
            logger.info("⏹️ Trading stopped by user")
            await self.aurora_agent.stop_trading()
    
    async def run_backtest(self, start_date: str, end_date: str):
        """백테스트 실행"""
        logger.info(f"📈 Running AuroraQ backtest: {start_date} to {end_date}")
        
        if not self.aurora_agent:
            await self.initialize()
        
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        result = await self.aurora_agent.run_backtest(start_dt, end_dt)
        
        # 결과 출력
        logger.info("🎯 AuroraQ Backtest Results:")
        logger.info(f"  💰 Total Return: {result.get('total_return', 0):.2%}")
        logger.info(f"  📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  📉 Max Drawdown: {result.get('max_drawdown', 0):.2%}")
        logger.info(f"  🎲 Win Rate: {result.get('win_rate', 0):.1%}")
        logger.info(f"  🔄 Total Trades: {result.get('total_trades', 0)}")
        
        return result
    
    async def get_status(self):
        """현재 상태 조회"""
        if not self.aurora_agent:
            return {"status": "not_initialized"}
        
        # 포트폴리오 상태
        portfolio = await self.aurora_agent.get_portfolio_status()
        
        # 최근 거래 정보
        recent_trades = await self.aurora_agent.get_recent_trades(limit=5)
        
        # 성과 지표
        performance = await self.aurora_agent.get_performance_metrics()
        
        return {
            "status": "running",
            "portfolio": portfolio,
            "recent_trades": recent_trades,
            "performance": performance,
            "loaded_modules": self.data_provider.loaded_data_types if self.data_provider else []
        }
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 Shutting down AuroraQ Standalone...")
        
        if self.aurora_agent:
            await self.aurora_agent.stop_trading()
        
        if self.data_provider:
            await self.data_provider.close()
        
        logger.info("✅ AuroraQ Standalone shutdown complete")


async def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(description="AuroraQ Standalone Runner")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "status"],
        default="live",
        help="실행 모드"
    )
    parser.add_argument("--start-date", help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--config", help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # AuroraQ 독립 실행
    runner = AuroraQStandalone(config_path=args.config)
    
    try:
        if args.mode == "live":
            await runner.run_trading()
        elif args.mode == "backtest":
            if not args.start_date or not args.end_date:
                logger.error("백테스트 모드에는 --start-date와 --end-date가 필요합니다")
                return
            await runner.run_backtest(args.start_date, args.end_date)
        elif args.mode == "status":
            await runner.initialize()
            status = await runner.get_status()
            logger.info(f"AuroraQ Status: {status}")
            
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중지됨")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    # Windows 환경 지원
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())