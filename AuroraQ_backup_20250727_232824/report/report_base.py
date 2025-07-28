# report_base.py - 리포트 생성 공통 기반 모듈
import os
import pandas as pd
import matplotlib.pyplot as plt

class ReportBase:
    def __init__(self, csv_path, report_dir, title="Report"):
        self.csv_path = csv_path
        self.report_dir = report_dir
        self.title = title
        self.df = None

        os.makedirs(self.report_dir, exist_ok=True)
        self.image_path = os.path.join(self.report_dir, "plot.png")
        self.html_path = os.path.join(self.report_dir, "report.html")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        except Exception as e:
            print(f"❌ CSV 로딩 실패: {e}")
            self.df = None

    def generate_plot(self):
        raise NotImplementedError("generate_plot()는 하위 클래스에서 구현해야 합니다.")

    def generate_html(self):
        html = f"""
        <html>
        <head><title>{self.title}</title></head>
        <body>
            <h2>{self.title}</h2>
            <img src="{os.path.basename(self.image_path)}" width="800">
        </body>
        </html>
        """
        with open(self.html_path, "w") as f:
            f.write(html)

    def save_plot(self):
        plt.tight_layout()
        plt.savefig(self.image_path)
        plt.close()

    def run_report(self):
        self.load_data()
        if self.df is None or self.df.empty:
            print("⚠️ 데이터 없음. 리포트 생성 생략.")
            return

        self.generate_plot()
        self.save_plot()
        self.generate_html()
        print(f"✅ 리포트 저장 완료: {self.html_path}")
