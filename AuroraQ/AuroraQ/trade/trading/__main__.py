#!/usr/bin/env python3
"""
AuroraQ VPS 실전매매 시스템 메인 진입점
Production-ready trading system optimized for VPS deployment
"""

import sys
import os
import asyncio
import signal
import json
import logging
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading.vps_realtime_system import VPSRealtimeSystem, VPSTradingConfig
from vps_logging import get_vps_log_integrator

class TradingSystemLauncher:
    """VPS 실전매매 시스템 런처"""
    
    def __init__(self):
        self.trading_system: Optional[VPSRealtimeSystem] = None
        self.config: Optional[VPSTradingConfig] = None
        self.log_integrator = None
        self.logger = None
        self.shutdown_event = asyncio.Event()
        
    def setup_logging(self):
        """로깅 시스템 설정"""
        try:
            # 통합 로깅 시스템 활성화
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("trading_launcher")
            
            # 콘솔 로깅도 설정
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # 루트 로거에 핸들러 추가
            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)
            
            return True
            
        except Exception as e:
            print(f"로깅 시스템 설정 실패: {e}")
            return False
    
    def load_config(self) -> VPSTradingConfig:
        """설정 파일 로드"""
        try:
            # 환경 변수에서 설정
            trading_mode = os.getenv('TRADING_MODE', 'paper')
            enable_sentiment = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true').lower() == 'true'
            enable_logging = os.getenv('ENABLE_UNIFIED_LOGGING', 'true').lower() == 'true'
            
            # JSON 설정 파일 로드 (있는 경우)
            config_path = PROJECT_ROOT / 'trading' / 'config' / 'vps_trading_config.json'
            config_data = {}
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                if self.logger:
                    self.logger.info(f"설정 파일 로드됨: {config_path}")
            
            # VPSTradingConfig 생성
            self.config = VPSTradingConfig(
                mode=trading_mode,
                symbol=config_data.get('trading', {}).get('symbols', ['BTCUSDT'])[0],
                enable_sentiment=enable_sentiment,
                enable_unified_logging=enable_logging,
                
                # 리스크 관리 설정
                max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.05')),
                max_daily_trades=int(os.getenv('VPS_MAX_DAILY_TRADES', '10')),
                emergency_stop_loss=float(os.getenv('EMERGENCY_STOP_LOSS', '0.05')),
                
                # VPS 최적화 설정
                vps_memory_limit=os.getenv('VPS_MEMORY_LIMIT', '3G'),
                update_interval=config_data.get('trading', {}).get('update_interval_seconds', 30),
                
                # API 설정
                api_host="0.0.0.0",
                api_port=int(os.getenv('TRADING_API_PORT', '8004')),
                websocket_port=int(os.getenv('TRADING_WEBSOCKET_PORT', '8003'))
            )
            
            if self.logger:
                self.logger.info(f"실전매매 설정 로드 완료: {trading_mode} 모드")
            
            return self.config
            
        except Exception as e:
            error_msg = f"설정 로드 실패: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            if self.logger:
                self.logger.info(f"종료 시그널 수신: {signum}")
            else:
                print(f"종료 시그널 수신: {signum}")
            
            # 비동기 종료 이벤트 설정
            if not self.shutdown_event.is_set():
                self.shutdown_event.set()
        
        # SIGINT (Ctrl+C), SIGTERM 처리
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows에서는 SIGBREAK도 처리
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def initialize_trading_system(self) -> bool:
        """실전매매 시스템 초기화"""
        try:
            if self.logger:
                self.logger.info("VPS 실전매매 시스템 초기화 중...")
            
            # 실전매매 시스템 생성
            self.trading_system = VPSRealtimeSystem(self.config)
            
            # 시스템 초기화
            success = await self.trading_system.initialize()
            
            if success:
                if self.logger:
                    self.logger.info("실전매매 시스템 초기화 완료")
                return True
            else:
                if self.logger:
                    self.logger.error("실전매매 시스템 초기화 실패")
                return False
                
        except Exception as e:
            error_msg = f"실전매매 시스템 초기화 중 오류: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return False
    
    async def run_trading_system(self):
        """실전매매 시스템 실행"""
        try:
            if self.logger:
                self.logger.info("VPS 실전매매 시스템 시작...")
            
            # 실전매매 시스템 시작
            trading_task = asyncio.create_task(self.trading_system.start())
            
            # 종료 이벤트 대기
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())
            
            # 두 태스크 중 하나가 완료될 때까지 대기
            done, pending = await asyncio.wait(
                [trading_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 남은 태스크 취소
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            if self.logger:
                self.logger.info("실전매매 시스템 종료 중...")
            
        except Exception as e:
            error_msg = f"실전매매 시스템 실행 중 오류: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise
    
    async def shutdown_trading_system(self):
        """실전매매 시스템 종료"""
        try:
            if self.trading_system:
                if self.logger:
                    self.logger.info("실전매매 시스템 정리 중...")
                
                await self.trading_system.shutdown()
                
                if self.logger:
                    self.logger.info("실전매매 시스템 종료 완료")
            
        except Exception as e:
            error_msg = f"실전매매 시스템 종료 중 오류: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
    
    async def main(self):
        """메인 실행 함수"""
        try:
            # 1. 로깅 시스템 설정
            if not self.setup_logging():
                print("로깅 시스템 설정 실패 - 기본 로깅으로 계속...")
            
            # 2. 시그널 핸들러 설정
            self.setup_signal_handlers()
            
            # 3. 설정 로드
            config = self.load_config()
            
            # 4. 실전매매 시스템 초기화
            if not await self.initialize_trading_system():
                if self.logger:
                    self.logger.error("시스템 초기화 실패 - 종료")
                return 1
            
            # 5. 시스템 실행
            await self.run_trading_system()
            
            return 0
            
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("사용자 중단 요청")
            else:
                print("사용자 중단 요청")
            return 0
            
        except Exception as e:
            error_msg = f"시스템 실행 중 예상치 못한 오류: {e}"
            if self.logger:
                self.logger.critical(error_msg)
            else:
                print(error_msg)
            return 1
            
        finally:
            # 정리 작업
            await self.shutdown_trading_system()

def main():
    """진입점 함수"""
    launcher = TradingSystemLauncher()
    
    try:
        # 이벤트 루프 실행
        return asyncio.run(launcher.main())
        
    except Exception as e:
        print(f"치명적 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())