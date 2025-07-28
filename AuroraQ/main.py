#!/usr/bin/env python3
"""
QuantumAI 메인 런처 v2.0
모드별 선택적 모듈 로딩으로 리소스 최적화
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

# 공통 모듈 (항상 로드)
from SharedCore.data_layer.unified_data_provider import UnifiedDataProvider
from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator

# 모드별 선택적 import는 런타임에 수행

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumAIOrchestrator:
    """
    QuantumAI 오케스트레이터 v2.0
    모드별 선택적 모듈 초기화로 리소스 최적화
    """
    
    def __init__(self, mode: str = "both"):
        self.mode = mode
        self.data_provider = None
        self.sentiment_aggregator = None
        
        # 모드별 모듈
        self.aurora_agent = None
        self.macro_tft = None
        self.portfolio_optimizer = None
        
        # 로드된 모듈 추적
        self.loaded_modules = set()
        
    async def initialize(self):
        """모드별 선택적 시스템 초기화"""
        logger.info(f"🚀 Initializing QuantumAI System (Mode: {self.mode})...")
        
        try:
            # 1. SharedCore 초기화 (모드별 최적화)
            use_macro = self.mode in ["macro", "both", "backtest"]
            use_crypto = self.mode in ["aurora", "both", "backtest"]
            
            self.data_provider = UnifiedDataProvider(
                use_macro=use_macro,
                use_crypto=use_crypto
            )
            await self.data_provider.connect()
            
            # 감정분석은 AuroraQ 사용시에만 초기화
            if use_crypto:
                self.sentiment_aggregator = SentimentAggregator()
                self.loaded_modules.add("sentiment")
            
            # 2. 모드별 모듈 초기화
            if self.mode in ["aurora", "both", "backtest"]:
                await self._initialize_aurora()
                
            if self.mode in ["macro", "both"]:
                await self._initialize_macro()
            
            logger.info(f"✅ QuantumAI System initialized successfully")
            logger.info(f"📦 Loaded modules: {', '.join(self.loaded_modules)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize QuantumAI: {e}")
            raise
    
    async def _initialize_aurora(self):
        """AuroraQ 모듈 초기화"""
        logger.info("🎯 Initializing AuroraQ Agent...")
        
        # 동적 import
        from AuroraQ.agent import AuroraQAgent, AuroraQConfig
        
        aurora_config = AuroraQConfig(
            initial_capital=100000.0,
            mode="simulation"
        )
        
        self.aurora_agent = AuroraQAgent(
            config=aurora_config,
            data_provider=self.data_provider,
            sentiment_aggregator=self.sentiment_aggregator
        )
        
        await self.aurora_agent.initialize()
        self.loaded_modules.add("aurora")
    
    async def _initialize_macro(self):
        """MacroQ 모듈 초기화"""
        logger.info("📊 Initializing MacroQ System...")
        
        # 동적 import
        from MacroQ.core.tft_engine.lightweight_tft import LightweightTFT, TFTConfig
        from MacroQ.portfolio.optimizer import PortfolioOptimizer
        
        tft_config = TFTConfig(
            num_assets=5,
            prediction_horizons=[7, 30, 90]
        )
        
        self.macro_tft = LightweightTFT(tft_config)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.loaded_modules.add("macro")
    
    async def run_aurora_only(self):
        """AuroraQ만 실행"""
        logger.info("🎯 Starting AuroraQ Agent...")
        
        if not self.aurora_agent:
            await self.initialize()
            
        await self.aurora_agent.start_trading()
    
    async def run_macro_only(self):
        """MacroQ만 실행"""
        logger.info("📊 Starting MacroQ System...")
        
        if not self.macro_tft:
            await self.initialize()
            
        # 임시 데모
        logger.info("MacroQ TFT model ready for predictions")
        logger.info("Portfolio optimizer ready")
        
        # 실제로는 여기서 포트폴리오 최적화 루프 실행
        await asyncio.sleep(5)
        logger.info("MacroQ demo completed")
    
    async def run_both(self):
        """두 Agent 동시 실행"""
        logger.info("🚀 Starting Both AuroraQ and MacroQ...")
        
        await self.initialize()
        
        # 병렬 실행
        aurora_task = asyncio.create_task(self.aurora_agent.start_trading())
        macro_task = asyncio.create_task(self._run_macro_loop())
        
        try:
            await asyncio.gather(aurora_task, macro_task)
        except KeyboardInterrupt:
            logger.info("⏹️ Shutting down...")
            await self.aurora_agent.stop_trading()
            aurora_task.cancel()
            macro_task.cancel()
    
    async def _run_macro_loop(self):
        """MacroQ 실행 루프"""
        while True:
            try:
                logger.info("MacroQ: Portfolio optimization cycle...")
                
                # 여기서 실제 포트폴리오 최적화 수행
                await asyncio.sleep(3600)  # 1시간마다
                
            except Exception as e:
                logger.error(f"MacroQ error: {e}")
                await asyncio.sleep(300)  # 5분 후 재시도
    
    async def run_backtest(self, start_date: str, end_date: str):
        """백테스트 실행"""
        logger.info(f"📈 Running backtest from {start_date} to {end_date}")
        
        await self.initialize()
        
        # AuroraQ 백테스트
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        result = await self.aurora_agent.run_backtest(start_dt, end_dt)
        
        logger.info("Backtest Results:")
        logger.info(f"  Total Return: {result['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {result['max_drawdown']:.2%}")
        
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 Shutting down QuantumAI...")
        
        if self.aurora_agent:
            await self.aurora_agent.stop_trading()
            
        if self.data_provider:
            await self.data_provider.close()
            
        logger.info("✅ QuantumAI shutdown complete")


async def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(description="QuantumAI - AI Agent Trading System")
    parser.add_argument(
        "--mode",
        choices=["aurora", "macro", "both", "backtest"],
        default="both",
        help="실행 모드 선택"
    )
    parser.add_argument("--start-date", help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="백테스트 종료일 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # 모드별 오케스트레이터 초기화
    orchestrator = QuantumAIOrchestrator(mode=args.mode)
    
    try:
        # 모드별 실행
        if args.mode == "aurora":
            await orchestrator.run_aurora_only()
        elif args.mode == "macro":
            await orchestrator.run_macro_only()
        elif args.mode == "both":
            await orchestrator.run_both()
        elif args.mode == "backtest":
            if not args.start_date or not args.end_date:
                logger.error("백테스트 모드에는 --start-date와 --end-date가 필요합니다")
                return
            await orchestrator.run_backtest(args.start_date, args.end_date)
            
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중지됨")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Windows에서 ProactorEventLoop 사용
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())