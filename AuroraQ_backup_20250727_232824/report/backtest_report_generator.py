import os
import pandas as pd
import json
import glob
import yaml
from report.report_base import ReportBase

class BacktestReportGenerator(ReportBase):
    def __init__(self,
                 csv_path="report/backtest/backtest_results.csv",
                 score_path="strategy_scores.json",
                 output_dir="report/backtest/"):
        super().__init__(csv_path=csv_path, report_dir=output_dir, title="📊 백테스트 전략 평가 리포트")
        self.score_path = score_path
        self.param_html = ""

    def load_data(self):
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if df.empty:
                print("⚠️ 백테스트 결과가 비어있습니다.")
                return

            # 전략별 메트릭 집계 (Sharpe, ROI %, WinRate, Profit Factor, MDD)
            grouped = df.groupby('strategy').agg({
                'sharpe': 'mean',
                'roi': lambda x: (1 + x).prod() - 1,  # 누적 수익률
                'win_rate': 'mean',
                'mdd': 'min',
                'profit_factor': 'mean',
                'composite_score': 'mean'
            }).reset_index()
            grouped['roi'] = grouped['roi'] * 100  # 퍼센트 표시

        else:
            grouped = pd.DataFrame(columns=['strategy', 'sharpe', 'roi', 'win_rate', 'mdd', 'profit_factor', 'composite_score'])

        self.df = grouped
        self.param_html = self.generate_param_html(grouped["strategy"].tolist())

    def generate_param_html(self, strategies):
        param_dir = "logs/strategy_params"
        html = "<h2>⚙️ 전략별 사용 파라미터</h2>"
        for strat in strategies:
            files = sorted(glob.glob(f"{param_dir}/{strat}_*.yaml"), reverse=True)
            if not files:
                html += f"<h4>{strat}</h4><p><i>📭 파라미터 기록 없음</i></p>"
                continue
            with open(files[0], 'r') as f:
                params = yaml.safe_load(f)
                param_str = yaml.dump(params, allow_unicode=True)
                html += f"<h4>{strat}</h4><pre>{param_str}</pre><br>"
        return html

    def generate_html(self):
        html = f"<html><head><meta charset='utf-8'><title>{self.title}</title></head><body>"
        html += f"<h1>{self.title}</h1>"
        if not self.df.empty:
            html += self.df.to_html(index=False, border=1)
        else:
            html += "<p>📭 결과 데이터가 없습니다.</p>"
        html += self.param_html
        html += "</body></html>"

        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"📁 HTML 리포트 저장 완료: {self.html_path}")

    def run(self):
        self.load_data()
        self.generate_html()

def generate_strategy_score_report(
    csv_path="report/backtest_results.csv",
    score_path="strategy_scores.json",
    output_dir="report/backtest/"
) -> pd.DataFrame:
    reporter = BacktestReportGenerator(csv_path, score_path, output_dir)
    reporter.run()
    return reporter.df

def generate_backtest_report(results, output_dir="report/backtest/"):
    os.makedirs(output_dir, exist_ok=True)
    temp_csv = os.path.join(output_dir, "backtest_results_temp.csv")
    pd.DataFrame(results).to_csv(temp_csv, index=False)

    reporter = BacktestReportGenerator(csv_path=temp_csv, output_dir=output_dir)
    reporter.run()
    return reporter.df

if __name__ == "__main__":
    report = BacktestReportGenerator()
    report.run()
