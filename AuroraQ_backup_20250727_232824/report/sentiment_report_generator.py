# sentiment_report_generator.py (확장 버전)

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.helpers import ensure_dir  # 폴더 없으면 생성

class SentimentReportGenerator:
    def __init__(self, sentiment_path="data/sentiment", report_path="report/sentiment"):
        self.sentiment_path = sentiment_path
        self.report_path = report_path
        self.image_path = os.path.join(report_path, "images")
        ensure_dir(self.image_path)

    def get_latest_csv(self):
        csv_files = [
            os.path.join(self.sentiment_path, f)
            for f in os.listdir(self.sentiment_path)
            if f.endswith(".csv")
        ]
        if not csv_files:
            print("❌ 감정 점수 CSV 파일 없음.")
            return None
        return max(csv_files, key=os.path.getmtime)

    def plot_bar_distribution(self, df, column, title, filename_suffix):
        plt.figure(figsize=(6, 4))
        df[column].value_counts().plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        image_file = os.path.join(self.image_path, f"{filename_suffix}.png")
        plt.savefig(image_file)
        plt.close()
        return image_file

    def run(self, filepath=None):
        if filepath is None:
            filepath = self.get_latest_csv()
            if not filepath:
                return

        df = pd.read_csv(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # ✅ 감정 라벨 분포 시각화
        label_img = self.plot_bar_distribution(df, "label", "Sentiment Label Distribution", f"label_{timestamp}")

        # ✅ 시나리오 태그 분포 시각화
        scenario_img = self.plot_bar_distribution(df, "scenario_tag", "Scenario Tag Distribution", f"scenario_{timestamp}")

        # ✅ HTML 리포트 생성
        html_path = os.path.join(self.report_path, f"sentiment_{timestamp}_report.html")
        html_content = f"""
        <html>
        <head><meta charset='utf-8'><title>Sentiment Report</title></head>
        <body>
        <h2>Sentiment Report ({timestamp})</h2>

        <h3>📊 Sentiment Label</h3>
        <img src='images/label_{timestamp}.png' width='600'><br><br>

        <h3>🧭 Scenario Tags</h3>
        <img src='images/scenario_{timestamp}.png' width='600'><br><br>

        <h3>📰 Sample Data</h3>
        {df[['title', 'label', 'confidence', 'sentiment_score', 'scenario_tag']].head(20).to_html(index=False)}
        </body>
        </html>
        """

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ 리포트 생성 완료: {html_path}")

# ✅ 외부 호출용 인터페이스
def run_sentiment_report(filepath=None):
    SentimentReportGenerator().run(filepath=filepath)
