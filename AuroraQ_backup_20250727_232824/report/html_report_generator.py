import os
import yaml
import glob
from datetime import datetime

def load_recent_params(strategy_name, max_count=5):
    param_dir = "logs/strategy_params"
    files = sorted(glob.glob(f"{param_dir}/{strategy_name}_*.yaml"), reverse=True)[:max_count]
    param_data = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                ts = os.path.basename(path).split("_")[-1].replace(".yaml", "")
                param_data.append((ts, y))
        except Exception:
            continue
    return param_data

def generate_html_report(strategy_name, result, output_dir="report/live/"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"live_report_{timestamp}.html")

    pnl = result.get("pnl_pct", "N/A")
    pnl_str = f"{pnl}%" if isinstance(pnl, (int, float)) else pnl
    pnl_color = "green" if isinstance(pnl, (int, float)) and pnl > 0 else "red"

    # ✅ 최근 파라미터 5개 불러오기
    recent_params = load_recent_params(strategy_name)

    if recent_params:
        table_html = "<h4>📘 최근 파라미터 변화 이력</h4>\n<table border='1' cellpadding='5'><thead><tr><th>파일타임스탬프</th><th>주요 파라미터</th></tr></thead><tbody>"
        for ts, y in recent_params:
            keys_to_show = ["adx_threshold", "ema_span", "sma_period", "bb_window"]
            summary = ", ".join([f"{k}: {v}" for k, v in y.items() if k in keys_to_show])
            table_html += f"<tr><td>{ts}</td><td><code>{summary}</code></td></tr>\n"
        table_html += "</tbody></table>\n"
    else:
        table_html = "<p><i>📭 파라미터 기록 없음</i></p>"

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Live Strategy Report</title></head>
    <body>
        <h2>📊 실시간 전략 리포트</h2>
        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr><th>전략 이름</th><td>{strategy_name}</td></tr>
            <tr><th>실행 시각</th><td>{timestamp}</td></tr>
            <tr><th>신호</th><td>{result.get('action', '-')}</td></tr>
            <tr><th>수익률</th><td><span style="color:{pnl_color};">{pnl_str}</span></td></tr>
            <tr><th>레버리지</th><td>{result.get('leverage', 'N/A')}x</td></tr>
        </table>
        {table_html}
    </body>
    </html>
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    return file_path
