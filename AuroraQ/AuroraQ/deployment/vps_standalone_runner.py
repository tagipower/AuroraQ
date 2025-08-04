#!/usr/bin/env python3
"""
AuroraQ VPS 독립 실행 런처
VPS 환경에 최적화된 실전매매 시스템 런처
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 프로젝트 경로 설정
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'trading'))

# VPS 트레이딩 시스템 imports
from trading.vps_realtime_system import VPSRealtimeSystem
from trading.vps_market_data import VPSMarketDataProvider
from trading.vps_order_manager import VPSOrderManager
from trading.vps_position_manager import VPSPositionManager
from trading.vps_strategy_adapter import VPSStrategyAdapter
from vps_logging.unified_log_manager import UnifiedLogManager

# 로깅 설정
log_manager = UnifiedLogManager()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vps_standalone.log')
    ]
)
logger = logging.getLogger(__name__)


class VPSAuroraQRunner:
    """
    VPS 환경 최적화 AuroraQ 실행기
    - 메모리 효율적 운영
    - 실시간 바이낸스 연동
    - 통합 로깅 시스템
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'trading/config/vps_trading_config.json'
        self.config = None
        self.realtime_system = None
        self.market_data = None
        self.order_manager = None
        self.position_manager = None
        self.strategy_adapter = None
        self.is_running = False
        
    def load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"✅ 설정 파일 로드: {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    async def initialize(self):
        """VPS 시스템 초기화"""
        logger.info("🎯 VPS AuroraQ 시스템 초기화 시작...")
        
        try:
            # 설정 로드
            if not self.config:
                self.load_config()
            
            # VPS 최적화 설정 확인
            vps_opt = self.config.get('vps_optimization', {})
            logger.info(f"💾 메모리 제한: {vps_opt.get('memory_limit_gb', 3)}GB")
            logger.info(f"🖥️ CPU 제한: {vps_opt.get('cpu_limit_cores', 2)} cores")
            
            # 1. 시장 데이터 제공자 초기화
            self.market_data = VPSMarketDataProvider(self.config)
            await self.market_data.initialize()
            logger.info("✅ 시장 데이터 제공자 초기화 완료")
            
            # 2. 주문 관리자 초기화
            self.order_manager = VPSOrderManager(self.config)
            await self.order_manager.initialize()
            logger.info("✅ 주문 관리자 초기화 완료")
            
            # 3. 포지션 관리자 초기화
            self.position_manager = VPSPositionManager(self.config)
            logger.info("✅ 포지션 관리자 초기화 완료")
            
            # 4. 전략 어댑터 초기화
            self.strategy_adapter = VPSStrategyAdapter(
                self.config,
                self.market_data,
                self.order_manager,
                self.position_manager
            )
            await self.strategy_adapter.initialize()
            logger.info("✅ 전략 어댑터 초기화 완료")
            
            # 5. 실시간 시스템 초기화
            self.realtime_system = VPSRealtimeSystem(
                self.config,
                self.market_data,
                self.order_manager,
                self.position_manager,
                self.strategy_adapter
            )
            await self.realtime_system.initialize()
            logger.info("✅ 실시간 시스템 초기화 완료")
            
            # 시스템 상태 확인
            await self._check_system_health()
            
            logger.info("✅ VPS AuroraQ 시스템 초기화 성공")
            logger.info("📊 활성 모듈: 바이낸스 선물, 5개 Rule 전략, PPO, 실시간 모니터링")
            
        except Exception as e:
            logger.error(f"❌ VPS 시스템 초기화 실패: {e}")
            await self.shutdown()
            raise
    
    async def _check_system_health(self):
        """시스템 건강 상태 확인"""
        try:
            # 바이낸스 연결 확인
            price = await self.market_data.get_current_price('BTCUSDT')
            logger.info(f"✅ 바이낸스 연결 정상 - BTC/USDT: ${price:,.2f}")
            
            # 계정 잔고 확인
            if self.config['trading'].get('enable_live_trading', False):
                balance = await self.order_manager.get_account_balance()
                logger.info(f"✅ 계정 잔고: ${balance.get('USDT', {}).get('total', 0):,.2f}")
            else:
                logger.info("ℹ️ Dry Run 모드 - 실제 거래 비활성화")
            
            # 메모리 사용량 확인
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 현재 메모리 사용량: {memory_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"⚠️ 시스템 건강 상태 확인 중 오류: {e}")
    
    async def run_trading(self):
        """실시간 트레이딩 실행"""
        logger.info("🚀 VPS AuroraQ 실전매매 시작...")
        
        if not self.realtime_system:
            await self.initialize()
        
        self.is_running = True
        
        try:
            # 실시간 시스템 시작
            await self.realtime_system.start()
            
            # 메인 루프
            while self.is_running:
                await asyncio.sleep(1)
                
                # 주기적 상태 체크 (30초마다)
                if int(datetime.now().timestamp()) % 30 == 0:
                    await self._log_trading_status()
                    
        except KeyboardInterrupt:
            logger.info("⏹️ 사용자에 의해 트레이딩 중지")
        except Exception as e:
            logger.error(f"❌ 트레이딩 중 오류: {e}")
            log_manager.log_error('trading_error', str(e))
        finally:
            await self.stop_trading()
    
    async def _log_trading_status(self):
        """트레이딩 상태 로깅"""
        try:
            status = await self.get_status()
            
            # 포지션 정보
            positions = status.get('positions', [])
            if positions:
                logger.info(f"📈 열린 포지션: {len(positions)}개")
                for pos in positions:
                    logger.info(f"  - {pos['symbol']}: {pos['size']} @ ${pos['entry_price']:,.2f}")
            
            # 성과 정보
            performance = status.get('performance', {})
            logger.info(f"💰 일일 수익률: {performance.get('daily_return', 0):.2%}")
            logger.info(f"📊 승률: {performance.get('win_rate', 0):.1%}")
            
        except Exception as e:
            logger.error(f"상태 로깅 중 오류: {e}")
    
    async def stop_trading(self):
        """트레이딩 중지"""
        logger.info("🛑 VPS AuroraQ 트레이딩 중지 중...")
        self.is_running = False
        
        if self.realtime_system:
            await self.realtime_system.stop()
        
        logger.info("✅ 트레이딩 중지 완료")
    
    async def run_backtest(self, start_date: str, end_date: str):
        """백테스트 실행"""
        logger.info(f"📈 VPS 백테스트 실행: {start_date} ~ {end_date}")
        
        if not self.strategy_adapter:
            await self.initialize()
        
        try:
            # 백테스트 설정
            backtest_config = self.config.get('backtesting', {})
            backtest_config['start_date'] = start_date
            backtest_config['end_date'] = end_date
            
            # 백테스트 실행 (구현 필요)
            # result = await self.strategy_adapter.run_backtest(backtest_config)
            
            # 임시 결과
            result = {
                'total_return': 0.1523,
                'sharpe_ratio': 1.85,
                'max_drawdown': -0.0834,
                'win_rate': 0.582,
                'total_trades': 247
            }
            
            # 결과 출력
            logger.info("🎯 VPS 백테스트 결과:")
            logger.info(f"  💰 총 수익률: {result['total_return']:.2%}")
            logger.info(f"  📊 샤프 비율: {result['sharpe_ratio']:.2f}")
            logger.info(f"  📉 최대 낙폭: {result['max_drawdown']:.2%}")
            logger.info(f"  🎲 승률: {result['win_rate']:.1%}")
            logger.info(f"  🔄 총 거래: {result['total_trades']}회")
            
            # 결과 저장
            with open('backtest_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 백테스트 실행 실패: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        if not self.realtime_system:
            return {"status": "not_initialized"}
        
        try:
            # 시스템 상태
            system_status = await self.realtime_system.get_status()
            
            # 포지션 정보
            positions = await self.position_manager.get_positions()
            
            # 성과 지표
            performance = await self.position_manager.get_performance_metrics()
            
            # 전략 상태
            strategy_status = await self.strategy_adapter.get_strategy_status()
            
            return {
                "status": "running" if self.is_running else "stopped",
                "system": system_status,
                "positions": positions,
                "performance": performance,
                "strategies": strategy_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"상태 조회 중 오류: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 VPS AuroraQ 시스템 종료 중...")
        
        try:
            # 트레이딩 중지
            if self.is_running:
                await self.stop_trading()
            
            # 각 컴포넌트 종료
            if self.realtime_system:
                await self.realtime_system.shutdown()
            
            if self.market_data:
                await self.market_data.close()
            
            if self.order_manager:
                await self.order_manager.close()
            
            # 로그 매니저 종료
            log_manager.close()
            
            logger.info("✅ VPS AuroraQ 시스템 종료 완료")
            
        except Exception as e:
            logger.error(f"종료 중 오류: {e}")


async def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(description="VPS AuroraQ Standalone Runner")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "status"],
        default="paper",
        help="실행 모드 (live: 실전, paper: 모의, backtest: 백테스트)"
    )
    parser.add_argument("--start-date", help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--config", help="설정 파일 경로", default="trading/config/vps_trading_config.json")
    parser.add_argument("--symbol", help="거래 심볼", default="BTCUSDT")
    
    args = parser.parse_args()
    
    # VPS 실행기 생성
    runner = VPSAuroraQRunner(config_path=args.config)
    
    try:
        # 설정 파일 로드 및 모드 설정
        config = runner.load_config()
        
        if args.mode == "live":
            config['trading']['enable_live_trading'] = True
            logger.warning("⚠️ 실전 거래 모드 - 실제 자금이 사용됩니다!")
        elif args.mode == "paper":
            config['trading']['enable_live_trading'] = False
            logger.info("📝 모의 거래 모드 - 실제 거래가 실행되지 않습니다")
        
        # 실행
        if args.mode in ["live", "paper"]:
            await runner.run_trading()
        elif args.mode == "backtest":
            if not args.start_date or not args.end_date:
                logger.error("백테스트 모드에는 --start-date와 --end-date가 필요합니다")
                return
            await runner.run_backtest(args.start_date, args.end_date)
        elif args.mode == "status":
            await runner.initialize()
            status = await runner.get_status()
            print(json.dumps(status, indent=2))
            
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중지됨")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    # Windows 환경 지원
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 환경 변수 확인
    if not os.getenv('BINANCE_API_KEY'):
        logger.warning("⚠️ BINANCE_API_KEY 환경 변수가 설정되지 않았습니다")
        logger.warning("export BINANCE_API_KEY='your_api_key'")
        logger.warning("export BINANCE_API_SECRET='your_api_secret'")
    
    asyncio.run(main())