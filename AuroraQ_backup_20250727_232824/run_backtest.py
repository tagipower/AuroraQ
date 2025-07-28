#!/usr/bin/env python3
"""
AuroraQ 백테스트 실행기
실제 전략과 연동하여 백테스트 실행
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backtest.v2.layers.controller_layer import BacktestController, BacktestOrchestrator, BacktestMode


def create_backtest_config(args):
    """명령행 인수로부터 백테스트 설정 생성"""
    config = {
        "name": args.name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "price_data_path": args.price_data,
        "sentiment_data_path": args.sentiment_data,
        "initial_capital": args.capital,
        "mode": args.mode,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_size": args.window_size,
        "enable_multiframe": not args.disable_multiframe,
        "enable_exploration": args.exploration,
        "enable_ppo": not args.disable_ppo,
        "cache_size": args.cache_size,
        "indicators": args.indicators.split(',') if args.indicators else None
    }
    return config


def run_single_backtest(config):
    """단일 백테스트 실행"""
    print(f"🚀 백테스트 시작: {config['name']}")
    print(f"📊 설정:")
    print(f"  - 자본: {config['initial_capital']:,}원")
    print(f"  - 모드: {config['mode']}")
    print(f"  - 기간: {config.get('start_date', 'all')} ~ {config.get('end_date', 'all')}")
    print(f"  - 윈도우: {config['window_size']}")
    print(f"  - 다중프레임: {config['enable_multiframe']}")
    print(f"  - 탐색모드: {config['enable_exploration']}")
    
    # 컨트롤러 생성
    controller = BacktestController(
        initial_capital=config["initial_capital"],
        mode=config["mode"],
        enable_multiframe=config["enable_multiframe"],
        enable_exploration=config["enable_exploration"],
        cache_size=config["cache_size"]
    )
    
    # 전략 초기화
    print("🔧 전략 시스템 초기화...")
    controller.initialize_strategies(
        sentiment_file=config.get("sentiment_data_path"),
        enable_ppo=config["enable_ppo"]
    )
    
    # 백테스트 실행
    print("⚡ 백테스트 실행 중...")
    result = controller.run_backtest(
        price_data_path=config["price_data_path"],
        sentiment_data_path=config.get("sentiment_data_path"),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        window_size=config["window_size"],
        indicators=config.get("indicators")
    )
    
    if result["success"]:
        print("✅ 백테스트 완료!")
        print_results(result)
        save_results(config["name"], result)
    else:
        print(f"❌ 백테스트 실패: {result['error']}")
    
    return result


def run_multiple_backtests(configs, parallel=True):
    """다중 백테스트 실행"""
    print(f"🚀 다중 백테스트 시작: {len(configs)}개")
    
    orchestrator = BacktestOrchestrator(n_workers=4)
    results = orchestrator.run_multiple_backtests(configs, parallel=parallel)
    
    print("✅ 다중 백테스트 완료!")
    
    # 결과 요약
    success_count = sum(1 for r in results if r.get("success"))
    print(f"📊 성공: {success_count}/{len(results)}")
    
    return results


def run_walk_forward(config, n_windows=10, train_ratio=0.8):
    """워크포워드 분석 실행"""
    print(f"🚀 워크포워드 분석 시작: {n_windows}개 윈도우")
    
    orchestrator = BacktestOrchestrator()
    result = orchestrator.walk_forward_analysis(
        base_config=config,
        n_windows=n_windows,
        train_ratio=train_ratio
    )
    
    print("✅ 워크포워드 분석 완료!")
    print_walk_forward_results(result)
    
    return result


def print_results(result):
    """결과 출력"""
    stats = result["stats"]
    metrics = result["metrics"]
    
    print(f"\n📈 백테스트 결과:")
    print(f"  - 실행 시간: {stats['execution_time']:.2f}초")
    print(f"  - 총 신호: {stats['total_signals']}")
    print(f"  - 실행 거래: {stats['executed_trades']}")
    print(f"  - 탐색 거래: {stats.get('exploration_trades', 0)}")
    print(f"  - 캐시 히트율: {stats['cache_stats']['hit_rate']:.2%}")
    
    if metrics.get("best_strategy"):
        best = metrics["best_metrics"]
        print(f"\n🏆 최고 전략: {metrics['best_strategy']}")
        print(f"  - ROI: {best.roi:.2%}")
        print(f"  - 승률: {best.win_rate:.2%}")
        print(f"  - 샤프비율: {best.sharpe_ratio:.3f}")
        print(f"  - 최대낙폭: {best.max_drawdown:.2%}")
        print(f"  - 종합점수: {best.composite_score:.3f}")


def print_walk_forward_results(result):
    """워크포워드 결과 출력"""
    stats = result["statistics"]
    
    print(f"\n📊 워크포워드 분석 결과:")
    print(f"  - 훈련 평균 ROI: {stats['train_avg_roi']:.2%}")
    print(f"  - 테스트 평균 ROI: {stats['test_avg_roi']:.2%}")
    print(f"  - 효율성 비율: {stats['efficiency_ratio']:.3f}")
    print(f"  - 일관성: {stats['consistency']:.3f}")


def save_results(name, result):
    """결과 저장"""
    # 결과 디렉토리 생성
    os.makedirs("reports/backtest", exist_ok=True)
    
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/backtest/{name}_{timestamp}_result.json"
    
    # JSON 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 결과 저장: {filename}")


def load_config_file(config_file):
    """설정 파일 로드"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="AuroraQ 백테스트 실행기")
    
    # 기본 설정
    parser.add_argument("--name", type=str, help="백테스트 이름")
    parser.add_argument("--price-data", type=str, required=True, help="가격 데이터 파일 경로")
    parser.add_argument("--sentiment-data", type=str, help="감정 데이터 파일 경로")
    parser.add_argument("--capital", type=float, default=1000000, help="초기 자본 (기본: 1,000,000)")
    
    # 모드 설정
    parser.add_argument("--mode", type=str, choices=[
        BacktestMode.NORMAL, 
        BacktestMode.EXPLORATION, 
        BacktestMode.VALIDATION,
        BacktestMode.WALK_FORWARD
    ], default=BacktestMode.NORMAL, help="백테스트 모드")
    
    # 날짜 설정
    parser.add_argument("--start-date", type=str, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="종료 날짜 (YYYY-MM-DD)")
    
    # 백테스트 설정
    parser.add_argument("--window-size", type=int, default=100, help="데이터 윈도우 크기")
    parser.add_argument("--cache-size", type=int, default=1000, help="캐시 크기")
    parser.add_argument("--indicators", type=str, help="사용할 지표 (콤마 구분)")
    
    # 플래그 설정
    parser.add_argument("--disable-multiframe", action="store_true", help="다중 타임프레임 비활성화")
    parser.add_argument("--exploration", action="store_true", help="탐색 모드 활성화")
    parser.add_argument("--disable-ppo", action="store_true", help="PPO 비활성화")
    
    # 실행 모드
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--multiple", action="store_true", help="다중 백테스트 실행")
    parser.add_argument("--walk-forward", action="store_true", help="워크포워드 분석")
    parser.add_argument("--no-parallel", action="store_true", help="병렬 처리 비활성화")
    
    # 워크포워드 설정
    parser.add_argument("--wf-windows", type=int, default=10, help="워크포워드 윈도우 수")
    parser.add_argument("--wf-train-ratio", type=float, default=0.8, help="훈련 데이터 비율")
    
    args = parser.parse_args()
    
    try:
        if args.config:
            # 설정 파일에서 로드
            if args.multiple:
                configs = load_config_file(args.config)
                run_multiple_backtests(configs, parallel=not args.no_parallel)
            else:
                config = load_config_file(args.config)
                if args.walk_forward:
                    run_walk_forward(config, args.wf_windows, args.wf_train_ratio)
                else:
                    run_single_backtest(config)
        else:
            # 명령행 인수에서 설정 생성
            config = create_backtest_config(args)
            
            if args.walk_forward:
                run_walk_forward(config, args.wf_windows, args.wf_train_ratio)
            else:
                run_single_backtest(config)
                
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())