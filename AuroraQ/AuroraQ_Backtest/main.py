#!/usr/bin/env python3
"""
AuroraQ Backtest 메인 실행 파일
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from core import BacktestEngine, BacktestConfig
from utils import DataManager, get_logger
from reports import ReportGenerator


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="AuroraQ Backtest - 고성능 백테스팅 프레임워크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 백테스트
  python main.py --symbol BTC-USD --strategy ma --start 2023-01-01
  
  # 파라미터 최적화
  python main.py --symbol BTC-USD --strategy rsi --optimize
  
  # 여러 전략 비교
  python main.py --symbol BTC-USD --strategy ma,rsi,bb --compare
        """
    )
    
    # 데이터 관련 옵션
    data_group = parser.add_argument_group('데이터 옵션')
    data_group.add_argument('--symbol', type=str, default='BTC-USD',
                           help='거래 심볼 (기본값: BTC-USD)')
    data_group.add_argument('--source', type=str, default='yahoo',
                           choices=['yahoo', 'binance', 'csv'],
                           help='데이터 소스 (기본값: yahoo)')
    data_group.add_argument('--start', type=str, default='2023-01-01',
                           help='시작일 (YYYY-MM-DD)')
    data_group.add_argument('--end', type=str,
                           help='종료일 (YYYY-MM-DD, 기본값: 오늘)')
    data_group.add_argument('--interval', type=str, default='1d',
                           choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                           help='시간 간격 (기본값: 1d)')
    
    # 백테스트 설정
    backtest_group = parser.add_argument_group('백테스트 설정')
    backtest_group.add_argument('--capital', type=float, default=100000,
                               help='초기 자본 (기본값: 100000)')
    backtest_group.add_argument('--commission', type=float, default=0.001,
                               help='수수료 (기본값: 0.001)')
    backtest_group.add_argument('--slippage', type=float, default=0.0005,
                               help='슬리피지 (기본값: 0.0005)')
    
    # 전략 관련 옵션
    strategy_group = parser.add_argument_group('전략 옵션')
    strategy_group.add_argument('--strategy', type=str, default='ma',
                               help='전략 이름 또는 여러 전략 (쉼표로 구분)')
    strategy_group.add_argument('--params', type=str,
                               help='전략 파라미터 (JSON 형식)')
    
    # 실행 모드
    mode_group = parser.add_argument_group('실행 모드')
    mode_group.add_argument('--optimize', action='store_true',
                           help='파라미터 최적화 실행')
    mode_group.add_argument('--compare', action='store_true',
                           help='여러 전략 비교')
    mode_group.add_argument('--walk-forward', action='store_true',
                           help='Walk-forward 분석')
    
    # 출력 옵션
    output_group = parser.add_argument_group('출력 옵션')
    output_group.add_argument('--report', action='store_true',
                             help='상세 리포트 생성')
    output_group.add_argument('--charts', action='store_true',
                             help='차트 생성')
    output_group.add_argument('--output-dir', type=str, default='results',
                             help='결과 저장 디렉토리')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='상세 로그 출력')
    
    return parser.parse_args()


def get_strategy_class(strategy_name):
    """전략 이름으로 클래스 가져오기"""
    strategy_map = {
        'ma': 'SimpleMovingAverageStrategy',
        'rsi': 'RSIStrategy', 
        'bb': 'BollingerBandsStrategy',
        'momentum': 'MomentumStrategy',
        'mean_reversion': 'MeanReversionStrategy'
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"지원하지 않는 전략: {strategy_name}")
    
    # 동적으로 전략 클래스 임포트
    module_name = f"strategies.{strategy_name}_strategy"
    class_name = strategy_map[strategy_name]
    
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        # 예제 전략으로 대체
        return create_example_strategy(strategy_name)


def create_example_strategy(strategy_name):
    """예제 전략 생성"""
    from core.trade_executor import OrderSignal
    from abc import ABC, abstractmethod
    
    class ExampleStrategy(ABC):
        def __init__(self, **params):
            self.params = params
            self.name = strategy_name
            self.history = []
        
        def generate_signal(self, market_data, position, equity):
            self.history.append(market_data.close)
            
            if strategy_name == 'ma':
                return self._ma_signal(market_data, position)
            elif strategy_name == 'rsi':
                return self._rsi_signal(market_data, position)
            else:
                return OrderSignal('hold', 0)
        
        def _ma_signal(self, market_data, position):
            if len(self.history) < 50:
                return OrderSignal('hold', 0)
                
            short_ma = sum(self.history[-20:]) / 20
            long_ma = sum(self.history[-50:]) / 50
            
            if short_ma > long_ma and position['is_flat']:
                return OrderSignal('buy', 0.5, confidence=0.7)
            elif short_ma < long_ma and position['is_long']:
                return OrderSignal('sell', 1.0, confidence=0.7)
            
            return OrderSignal('hold', 0)
        
        def _rsi_signal(self, market_data, position):
            if len(self.history) < 15:
                return OrderSignal('hold', 0)
            
            # 간단한 RSI 계산
            deltas = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
            gains = [d for d in deltas[-14:] if d > 0]
            losses = [-d for d in deltas[-14:] if d < 0]
            
            avg_gain = sum(gains) / 14 if gains else 0
            avg_loss = sum(losses) / 14 if losses else 0.001
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            if rsi < 30 and position['is_flat']:
                return OrderSignal('buy', 0.3, confidence=0.6)
            elif rsi > 70 and position['is_long']:
                return OrderSignal('sell', 1.0, confidence=0.6)
            
            return OrderSignal('hold', 0)
    
    return ExampleStrategy


def run_single_backtest(args, logger):
    """단일 백테스트 실행"""
    # 데이터 로딩
    logger.info(f"데이터 로딩: {args.symbol} ({args.source})")
    data_manager = DataManager()
    
    try:
        data = data_manager.load_data(
            symbol=args.symbol,
            source=args.source,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval
        )
        logger.info(f"데이터 로딩 완료: {len(data)} 개 레코드")
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {e}")
        return None
    
    # 백테스트 설정
    config = BacktestConfig(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    # 전략 생성
    strategy_class = get_strategy_class(args.strategy)
    strategy = strategy_class()
    
    # 백테스트 실행
    logger.info("백테스트 실행 시작")
    engine = BacktestEngine(config)
    
    try:
        result = engine.run(strategy, data)
        logger.info("백테스트 완료")
        return result
    except Exception as e:
        logger.error(f"백테스트 실행 실패: {e}")
        return None


def run_optimization(args, logger):
    """파라미터 최적화 실행"""
    logger.info("파라미터 최적화 시작")
    
    # 기본 파라미터 그리드
    param_grids = {
        'ma': {
            'short_window': [10, 15, 20],
            'long_window': [30, 40, 50]
        },
        'rsi': {
            'rsi_period': [10, 14, 21],
            'oversold': [25, 30, 35],
            'overbought': [65, 70, 75]
        }
    }
    
    param_grid = param_grids.get(args.strategy, {})
    if not param_grid:
        logger.error(f"전략 {args.strategy}에 대한 최적화 파라미터가 정의되지 않음")
        return None
    
    # 데이터 로딩
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=args.symbol,
        source=args.source,
        start_date=args.start,
        end_date=args.end
    )
    
    # 최적화 실행
    config = BacktestConfig(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    engine = BacktestEngine(config)
    strategy_class = get_strategy_class(args.strategy)
    
    try:
        best_params, best_result = engine.optimize(
            strategy_class,
            param_grid,
            data,
            metric='sharpe_ratio'
        )
        
        logger.info(f"최적화 완료: {best_params}")
        return best_result
    except Exception as e:
        logger.error(f"최적화 실패: {e}")
        return None


def run_comparison(args, logger):
    """여러 전략 비교"""
    strategies = args.strategy.split(',')
    logger.info(f"전략 비교: {strategies}")
    
    # 데이터 로딩
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=args.symbol,
        source=args.source,
        start_date=args.start,
        end_date=args.end
    )
    
    # 각 전략 실행
    config = BacktestConfig(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    engine = BacktestEngine(config)
    results = {}
    
    for strategy_name in strategies:
        try:
            strategy_class = get_strategy_class(strategy_name.strip())
            strategy = strategy_class()
            
            result = engine.run(strategy, data)
            results[strategy_name] = result
            
            logger.info(f"{strategy_name} 완료: 수익률 {result.total_return:.2%}")
        except Exception as e:
            logger.error(f"{strategy_name} 실패: {e}")
    
    return results


def print_results(result, logger):
    """결과 출력"""
    if result is None:
        return
    
    print("\n" + "="*60)
    print("백테스트 결과")
    print("="*60)
    print(result.summary())


def generate_reports(result, args, logger):
    """리포트 생성"""
    if not args.report or result is None:
        return
    
    logger.info("리포트 생성 시작")
    
    # 출력 디렉토리 생성
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 리포트 생성
    report_gen = ReportGenerator(str(output_path))
    
    try:
        if isinstance(result, dict):
            # 여러 전략 비교 리포트
            report_path = report_gen.compare_strategies(result)
        else:
            # 단일 전략 리포트
            report_path = report_gen.generate_comprehensive_report(
                result,
                strategy_name=args.strategy,
                include_charts=args.charts
            )
        
        logger.info(f"리포트 생성 완료: {report_path}")
        print(f"\n📊 리포트가 생성되었습니다: {report_path}")
        
    except Exception as e:
        logger.error(f"리포트 생성 실패: {e}")


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 로거 설정
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("AuroraQ_Backtest", level=log_level)
    
    logger.info("AuroraQ Backtest 시작")
    logger.info(f"설정: {vars(args)}")
    
    # 실행 모드에 따른 분기
    result = None
    
    try:
        if args.optimize:
            result = run_optimization(args, logger)
        elif args.compare:
            result = run_comparison(args, logger)
        else:
            result = run_single_backtest(args, logger)
        
        # 결과 출력
        print_results(result, logger)
        
        # 리포트 생성
        generate_reports(result, args, logger)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        return 1
    
    logger.info("AuroraQ Backtest 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())