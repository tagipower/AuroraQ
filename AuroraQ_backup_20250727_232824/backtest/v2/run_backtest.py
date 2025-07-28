#!/usr/bin/env python3
"""
AuroraQ 백테스트 실행기 v2
실제 전략들을 사용한 백테스트 시스템
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

try:
    from backtest.v2.layers.controller_layer import BacktestController, BacktestOrchestrator, BacktestMode
    logger.info("백테스트 시스템 임포트 성공")
except ImportError as e:
    logger.error(f"백테스트 시스템 임포트 실패: {e}")
    sys.exit(1)


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="AuroraQ 백테스트 시스템 v2")
    
    # 필수 인수
    parser.add_argument(
        '--price-data', 
        required=True,
        help='가격 데이터 CSV 파일 경로'
    )
    
    # 선택적 인수
    parser.add_argument(
        '--sentiment-data',
        help='감정 데이터 CSV 파일 경로'
    )
    
    parser.add_argument(
        '--start-date',
        help='백테스트 시작 날짜 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        help='백테스트 종료 날짜 (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=1000000,
        help='초기 자본 (기본값: 1,000,000)'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='데이터 윈도우 크기 (기본값: 100)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['normal', 'exploration', 'validation', 'walk_forward'],
        default='normal',
        help='백테스트 모드 (기본값: normal)'
    )
    
    parser.add_argument(
        '--enable-multiframe',
        action='store_true',
        default=True,
        help='다중 타임프레임 활성화'
    )
    
    parser.add_argument(
        '--enable-exploration',
        action='store_true',
        help='탐색 모드 활성화'
    )
    
    parser.add_argument(
        '--cache-size',
        type=int,
        default=1000,
        help='캐시 크기 (기본값: 1000)'
    )
    
    parser.add_argument(
        '--enable-ppo',
        action='store_true',
        help='PPO 에이전트 활성화'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports/backtest',
        help='출력 디렉토리 (기본값: reports/backtest)'
    )
    
    parser.add_argument(
        '--indicators',
        nargs='+',
        default=[
            "sma_20", "sma_50", "ema_12", "ema_26", 
            "rsi", "macd", "macd_line", "macd_signal", "macd_hist",
            "bbands", "bollinger", "bb_upper", "bb_middle", "bb_lower",
            "atr", "adx", "volatility"
        ],
        help='사용할 지표 목록'
    )
    
    return parser.parse_args()


def validate_files(args):
    """파일 존재 확인"""
    if not os.path.exists(args.price_data):
        raise FileNotFoundError(f"가격 데이터 파일을 찾을 수 없습니다: {args.price_data}")
    
    if args.sentiment_data and not os.path.exists(args.sentiment_data):
        raise FileNotFoundError(f"감정 데이터 파일을 찾을 수 없습니다: {args.sentiment_data}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)


def run_single_backtest(args):
    """단일 백테스트 실행"""
    logger.info("=== AuroraQ 백테스트 시작 ===")
    logger.info(f"가격 데이터: {args.price_data}")
    logger.info(f"감정 데이터: {args.sentiment_data}")
    logger.info(f"초기 자본: {args.initial_capital:,}")
    logger.info(f"모드: {args.mode}")
    logger.info(f"윈도우 크기: {args.window_size}")
    
    # 컨트롤러 생성
    controller = BacktestController(
        initial_capital=args.initial_capital,
        mode=args.mode,
        enable_multiframe=args.enable_multiframe,
        enable_exploration=args.enable_exploration,
        cache_size=args.cache_size
    )
    
    # 전략 시스템 초기화
    logger.info("전략 시스템 초기화 중...")
    controller.initialize_strategies(
        sentiment_file=args.sentiment_data,
        enable_ppo=args.enable_ppo
    )
    
    # 백테스트 실행
    logger.info("백테스트 실행 중...")
    result = controller.run_backtest(
        price_data_path=args.price_data,
        sentiment_data_path=args.sentiment_data,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        indicators=args.indicators
    )
    
    # 결과 출력
    if result["success"]:
        logger.info("=== 백테스트 완료 ===")
        
        # 성능 통계
        stats = result["stats"]
        logger.info(f"실행 시간: {stats['execution_time']:.2f}초")
        logger.info(f"총 신호: {stats['total_signals']}")
        logger.info(f"실행된 거래: {stats['executed_trades']}")
        logger.info(f"탐색 거래: {stats['exploration_trades']}")
        logger.info(f"평균 처리 시간: {sum(stats['processing_time'])/len(stats['processing_time'])*1000:.2f}ms")
        
        # 최고 전략
        best_strategy = result["metrics"]["best_strategy"]
        best_metrics = result["metrics"]["best_metrics"]
        
        if best_strategy:
            logger.info(f"\n=== 최고 성과 전략: {best_strategy} ===")
            logger.info(f"ROI: {best_metrics.roi:.2%}")
            logger.info(f"샤프 비율: {best_metrics.sharpe_ratio:.2f}")
            logger.info(f"승률: {best_metrics.win_rate:.1%}")
            logger.info(f"최대 드로우다운: {best_metrics.max_drawdown:.2%}")
            logger.info(f"종합 점수: {best_metrics.composite_score:.3f}")
            logger.info(f"총 거래: {best_metrics.total_trades}")
        
        # 보고서 파일
        reports = result["reports"]
        logger.info(f"\n=== 생성된 보고서 ===")
        for report_type, file_path in reports.items():
            logger.info(f"{report_type}: {file_path}")
        
        # 캐시 통계
        cache_stats = stats.get("cache_stats", {})
        if cache_stats:
            logger.info(f"\n=== 캐시 통계 ===")
            logger.info(f"히트율: {cache_stats.get('hit_rate', 0):.1%}")
            logger.info(f"캐시 크기: {cache_stats.get('size', 0)}")
        
    else:
        logger.error(f"백테스트 실패: {result['error']}")
        return False
    
    return True


def run_multiple_backtests():
    """여러 백테스트 실행 예시"""
    # 다양한 설정으로 백테스트 실행
    configurations = [
        {
            "name": "Normal_Mode",
            "price_data_path": "data/sample_price_data.csv",
            "mode": "normal",
            "initial_capital": 1000000,
            "window_size": 100
        },
        {
            "name": "Exploration_Mode", 
            "price_data_path": "data/sample_price_data.csv",
            "mode": "exploration",
            "enable_exploration": True,
            "initial_capital": 1000000,
            "window_size": 100
        }
    ]
    
    orchestrator = BacktestOrchestrator(n_workers=2)
    results = orchestrator.run_multiple_backtests(configurations, parallel=True)
    
    for i, result in enumerate(results):
        config = configurations[i]
        logger.info(f"\n=== {config['name']} 결과 ===")
        if result["success"]:
            best_strategy = result["metrics"]["best_strategy"]
            best_metrics = result["metrics"]["best_metrics"]
            logger.info(f"최고 전략: {best_strategy}")
            logger.info(f"ROI: {best_metrics.roi:.2%}")
            logger.info(f"샤프 비율: {best_metrics.sharpe_ratio:.2f}")
        else:
            logger.error(f"실패: {result['error']}")


def main():
    """메인 함수"""
    try:
        args = parse_arguments()
        validate_files(args)
        
        success = run_single_backtest(args)
        
        if success:
            logger.info("백테스트가 성공적으로 완료되었습니다.")
            sys.exit(0)
        else:
            logger.error("백테스트가 실패했습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()