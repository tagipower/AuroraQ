# mab_score_report_generator.py
import os
import yaml
import glob
import pandas as pd
import matplotlib.pyplot as plt
from report.report_base import ReportBase


def load_latest_param(strategy_name):
    param_dir = "logs/strategy_params"
    files = sorted(glob.glob(f"{param_dir}/{strategy_name}_*.yaml"), reverse=True)
    if not files:
        return None
    with open(files[0], "r") as f:
        return yaml.safe_load(f)


class MABReportGenerator(ReportBase):
    def __init__(self, csv_path="logs/mab_score_log.csv", output_dir="report/mab/"):
        super().__init__(csv_path=csv_path, report_dir=output_dir, title="📊 MAB 전략 선택 리포트")
        self.param_html = ""

    def load_data(self):
        df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
        summary = df.groupby("strategy").agg(
            count=("reward", "count"),
            avg_reward=("reward", "mean"),
            max_reward=("reward", "max"),
            min_reward=("reward", "min"),
            last_used=("timestamp", "max")
        ).reset_index().sort_values(by="avg_reward", ascending=False)
        self.df = summary
        self.df.to_csv(self.csv_path.replace(".csv", "_summary.csv"), index=False, encoding="utf-8-sig")

        # ✅ 전략 파라미터 HTML 생성
        self.param_html = self.generate_param_html(summary["strategy"].tolist())

    def generate_param_html(self, strategies):
        html = "<h3>⚙️ 전략별 사용 파라미터</h3>"
        for strat in strategies:
            params = load_latest_param(strat)
            if params:
                param_str = yaml.dump(params, allow_unicode=True)
                html += f"<h4>{strat}</h4><pre>{param_str}</pre><br>"
            else:
                html += f"<h4>{strat}</h4><p><i>📭 파라미터 기록 없음</i></p>"
        return html

    def generate_plot(self):
        plt.figure(figsize=(10, 5))
        bars = plt.bar(self.df["strategy"], self.df["avg_reward"], color="lightgreen")
        plt.title("MAB 전략별 평균 보상")
        plt.xlabel("Strategy")
        plt.ylabel("Average Reward")
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom')

    def generate_html(self):
        html = f"""
        <html>
        <head><meta charset="utf-8"><title>{self.title}</title></head>
        <body>
            <h2>{self.title}</h2>
            <p><strong>총 전략 수:</strong> {len(self.df)}</p>
            <img src="{os.path.basename(self.image_path)}" width="700"><br><br>
            <h3>📋 전략 요약 (평균 보상 기준)</h3>
            {self.df.to_html(index=False)}
            {self.param_html}
        </body>
        </html>
        """
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html)

    def run(self):
        self.run_report()


# 사용 예시
if __name__ == "__main__":
    report = MABReportGenerator()
    report.run()
