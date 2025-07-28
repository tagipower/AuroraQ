# report_loop.py - 리포트 자동 실행 통합 루프 (인자 오류 방지 및 파일 체크 포함)
import os
import traceback
from report.train_report_generator import TrainReportGenerator
from report.strategy_report_generator import StrategyReportGenerator
from report.sentiment_report_generator import SentimentReportGenerator
from report.mab_score_report_generator import MABReportGenerator
from report.backtest_report_generator import BacktestReportGenerator


def run_all_reports():
    print("📊 [REPORT LOOP] 리포트 자동 실행 시작")

    report_configs = {
        TrainReportGenerator: {
            "eval_log_path": "logs/eval_log.csv",
            "model_path": "models/ppo_latest.zip"
        },
        StrategyReportGenerator: {
            "yaml_path": "config/strategy_score_log.yaml"
        },
        SentimentReportGenerator: {},
        MABReportGenerator: {},
        BacktestReportGenerator: {},
    }

    for report_cls, kwargs in report_configs.items():
        try:
            report = report_cls(**kwargs)
            report.run()
            print(f"✅ {report_cls.__name__} 완료")
        except FileNotFoundError as e:
            print(f"⚠️ {report_cls.__name__} → 파일 없음으로 스킵: {e}")
        except Exception as e:
            print(f"❌ {report_cls.__name__} 오류: {e}")
            traceback.print_exc()

    print("🎯 [REPORT LOOP] 전체 리포트 완료")


if __name__ == "__main__":
    run_all_reports()
