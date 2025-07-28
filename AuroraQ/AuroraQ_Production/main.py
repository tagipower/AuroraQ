#!/usr/bin/env python3
"""
AuroraQ Production - 메인 실행 파일
실시간 하이브리드 거래 시스템 시작점
"""

import os
import sys
import argparse
import time
from datetime import datetime

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core import RealtimeHybridSystem, TradingConfig
from utils import get_logger, ConfigManager, setup_logging
from sentiment import NewsCollector, SentimentScorer

logger = get_logger("AuroraQ_Main")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="AuroraQ Production - 실시간 하이브리드 거래 시스템")
    
    parser.add_argument('--mode', choices=['live', 'test', 'demo'], default='live',
                       help='실행 모드 (live: 실거래, test: 테스트, demo: 데모)')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로')
    
    parser.add_argument('--duration', type=int, default=0,
                       help='실행 시간 (분, 0=무제한)')
    
    parser.add_argument('--sentiment', action='store_true',
                       help='센티멘트 분석 활성화')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='로그 레벨')
    
    return parser.parse_args()

def setup_system(config_path: str, mode: str, enable_sentiment: bool) -> RealtimeHybridSystem:
    """시스템 설정 및 초기화"""
    
    # 설정 로드
    config_manager = ConfigManager(config_path)
    app_config = config_manager.get_config()
    
    # 거래 설정 생성
    trading_config = TradingConfig(
        rule_strategies=app_config.strategy.rule_strategies,
        enable_ppo=app_config.strategy.enable_ppo,
        hybrid_mode=app_config.strategy.hybrid_mode,
        execution_strategy=app_config.strategy.execution_strategy,
        risk_tolerance=app_config.strategy.risk_tolerance,
        max_position_size=app_config.trading.max_position_size,
        emergency_stop_loss=app_config.trading.emergency_stop_loss,
        max_daily_trades=app_config.trading.max_daily_trades,
        update_interval_seconds=app_config.trading.update_interval_seconds,
        lookback_periods=app_config.trading.lookback_periods,
        min_data_points=app_config.trading.min_data_points,
        enable_notifications=app_config.notifications.enable_notifications,
        notification_channels=app_config.notifications.channels
    )
    
    # 모드별 설정 조정
    if mode == 'demo':
        trading_config.max_position_size = 0.01  # 작은 포지션
        trading_config.max_daily_trades = 3
        trading_config.min_data_points = 10
    elif mode == 'test':
        trading_config.max_position_size = 0.05
        trading_config.max_daily_trades = 5
        trading_config.min_data_points = 20
    
    # 실시간 시스템 생성
    system = RealtimeHybridSystem(trading_config)
    
    # 센티멘트 분석 설정
    if enable_sentiment:
        logger.info("센티멘트 분석 모듈 활성화")
        system.sentiment_collector = NewsCollector()
        system.sentiment_scorer = SentimentScorer()
    
    logger.info(f"시스템 초기화 완료 - 모드: {mode}")
    return system

def run_live_mode(system: RealtimeHybridSystem, duration: int = 0):
    """실거래 모드 실행"""
    logger.info("=== 실거래 모드 시작 ===")
    logger.warning("주의: 실제 자금이 투입됩니다!")
    
    # 사용자 확인
    confirm = input("실거래를 시작하시겠습니까? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        logger.info("실거래 모드 취소됨")
        return
    
    try:
        if system.start():
            logger.info("실거래 시스템 실행 중...")
            
            start_time = time.time()
            while system.is_running:
                # 지속 시간 체크
                if duration > 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration:
                        logger.info(f"설정된 실행 시간 완료: {duration}분")
                        break
                
                time.sleep(1)
        else:
            logger.error("시스템 시작 실패")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의한 시스템 중지")
    except Exception as e:
        logger.error(f"실거래 모드 오류: {e}")
    finally:
        system.stop()

def run_test_mode(system: RealtimeHybridSystem, duration: int = 5):
    """테스트 모드 실행"""
    logger.info("=== 테스트 모드 시작 ===")
    
    if duration == 0:
        duration = 5  # 기본 5분
    
    try:
        if system.start():
            logger.info(f"테스트 시스템 실행 중... ({duration}분)")
            
            start_time = time.time()
            while system.is_running:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= duration:
                    logger.info(f"테스트 시간 완료: {duration}분")
                    break
                
                time.sleep(1)
        else:
            logger.error("시스템 시작 실패")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의한 테스트 중지")
    except Exception as e:
        logger.error(f"테스트 모드 오류: {e}")
    finally:
        system.stop()

def run_demo_mode(system: RealtimeHybridSystem, duration: int = 2):
    """데모 모드 실행"""
    logger.info("=== 데모 모드 시작 ===")
    
    if duration == 0:
        duration = 2  # 기본 2분
    
    try:
        if system.start():
            logger.info(f"데모 시스템 실행 중... ({duration}분)")
            
            start_time = time.time()
            while system.is_running:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= duration:
                    logger.info(f"데모 시간 완료: {duration}분")
                    break
                
                time.sleep(1)
        else:
            logger.error("시스템 시작 실패")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의한 데모 중지")
    except Exception as e:
        logger.error(f"데모 모드 오류: {e}")
    finally:
        system.stop()

def print_performance_report(system: RealtimeHybridSystem):
    """성과 리포트 출력"""
    logger.info("\n" + "="*50)
    logger.info("🎯 최종 성과 리포트")
    logger.info("="*50)
    
    try:
        report = system.get_performance_report()
        
        logger.info(f"📊 신호 생성: {report.get('total_signals', 0)}개")
        logger.info(f"⚡ 실행된 거래: {report.get('executed_trades', 0)}개")
        logger.info(f"📈 신호 실행률: {report.get('signal_execution_rate', 0)*100:.1f}%")
        
        completed_trades = report.get('total_completed_trades', 0)
        if completed_trades > 0:
            win_rate = report.get('win_rate', 0) * 100
            avg_pnl = report.get('avg_pnl_pct', 0) * 100
            
            logger.info(f"✅ 완료된 거래: {completed_trades}개")
            logger.info(f"🏆 승률: {win_rate:.1f}%")
            logger.info(f"💰 평균 수익률: {avg_pnl:.2f}%")
        
        current_pos = report.get('current_position', 0)
        if current_pos != 0:
            logger.info(f"📍 현재 포지션: {current_pos:.4f}")
        else:
            logger.info("📍 현재 포지션: 없음")
    
    except Exception as e:
        logger.error(f"성과 리포트 생성 오류: {e}")
    
    logger.info("="*50)

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 로깅 설정
    setup_logging(level=args.log_level)
    
    logger.info("🚀 AuroraQ Production 시작")
    logger.info(f"실행 모드: {args.mode}")
    logger.info(f"설정 파일: {args.config}")
    
    try:
        # 시스템 초기화
        system = setup_system(args.config, args.mode, args.sentiment)
        
        # 모드별 실행
        if args.mode == 'live':
            run_live_mode(system, args.duration)
        elif args.mode == 'test':
            run_test_mode(system, args.duration)
        elif args.mode == 'demo':
            run_demo_mode(system, args.duration)
        
        # 성과 리포트 출력
        print_performance_report(system)
    
    except Exception as e:
        logger.error(f"시스템 실행 오류: {e}")
        return 1
    
    logger.info("🎯 AuroraQ Production 종료")
    return 0

if __name__ == "__main__":
    sys.exit(main())