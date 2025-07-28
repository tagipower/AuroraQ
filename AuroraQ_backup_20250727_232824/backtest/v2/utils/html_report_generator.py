"""
HTML Report Generator for Backtest Results
백테스트 결과를 시각적인 HTML 리포트로 생성
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Charts will be disabled.")

from utils.logger import get_logger
from .strategy_scores_generator import StrategyScoresGenerator

logger = get_logger("HTMLReportGenerator")


class HTMLReportGenerator:
    """HTML 백테스트 리포트 생성기"""
    
    def __init__(self, output_dir: str = "reports/html"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # CSS 스타일
        self.css_style = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #4a5568;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }
        .metric-label {
            color: #718096;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .positive { color: #48bb78; }
        .negative { color: #f56565; }
        .neutral { color: #ed8936; }
        .strategy-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .strategy-table th,
        .strategy-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
        }
        .strategy-table th {
            background-color: #4a5568;
            color: white;
            font-weight: 600;
        }
        .strategy-table tr:nth-child(even) {
            background-color: #f7fafc;
        }
        .grade-A { background-color: #c6f6d5; color: #22543d; }
        .grade-B { background-color: #fed7e2; color: #744210; }
        .grade-C { background-color: #feebc8; color: #7c2d12; }
        .grade-D { background-color: #fed7d7; color: #742a2a; }
        .grade-F { background-color: #feb2b2; color: #742a2a; }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .summary-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #edf2f7;
            border-radius: 8px;
            flex: 1;
            min-width: 150px;
        }
        .performance-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }
        .footer {
            text-align: center;
            color: #718096;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #e2e8f0;
        }
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            .summary-stats {
                flex-direction: column;
            }
        }
        </style>
        """
    
    def generate_report(self, backtest_results: Dict[str, Any], 
                       strategy_scores: Optional[Dict[str, Any]] = None,
                       title: str = "Backtest Report") -> str:
        """HTML 리포트 생성"""
        
        # 전략 점수가 없으면 생성
        if strategy_scores is None:
            scores_generator = StrategyScoresGenerator()
            scores = scores_generator.generate_scores(backtest_results)
            strategy_scores = {name: score.__dict__ for name, score in scores.items()}
        
        # HTML 구성 요소들
        html_parts = []
        html_parts.append(self._generate_html_header(title))
        html_parts.append(self._generate_overview_section(backtest_results))
        html_parts.append(self._generate_strategy_comparison_section(backtest_results, strategy_scores))
        html_parts.append(self._generate_performance_charts_section(backtest_results))
        html_parts.append(self._generate_detailed_metrics_section(backtest_results))
        html_parts.append(self._generate_execution_stats_section(backtest_results))
        html_parts.append(self._generate_footer())
        
        # HTML 파일 생성
        html_content = '\\n'.join(html_parts)
        timestamp = datetime.now()
        filename = f"backtest_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML 리포트 생성 완료: {filepath}")
        return filepath
    
    def _generate_html_header(self, title: str) -> str:
        """HTML 헤더 생성"""
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {self.css_style}
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
    
    def _generate_overview_section(self, results: Dict[str, Any]) -> str:
        """개요 섹션 생성"""
        stats = results.get('stats', {})
        metrics = results.get('metrics', {})
        best_strategy = metrics.get('best_strategy', 'Unknown')
        
        total_signals = stats.get('total_signals', 0)
        executed_trades = stats.get('executed_trades', 0)
        execution_time = stats.get('execution_time', 0)
        
        # 실행률 계산
        execution_rate = (executed_trades / total_signals * 100) if total_signals > 0 else 0
        
        return f"""
        <div class="section">
            <h2>📊 Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_signals:,}</div>
                    <div class="metric-label">Total Signals</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{executed_trades:,}</div>
                    <div class="metric-label">Executed Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{execution_rate:.1f}%</div>
                    <div class="metric-label">Execution Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{execution_time:.1f}s</div>
                    <div class="metric-label">Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_strategy}</div>
                    <div class="metric-label">Best Strategy</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_strategy_comparison_section(self, results: Dict[str, Any], 
                                            strategy_scores: Dict[str, Any]) -> str:
        """전략 비교 섹션 생성"""
        comparison_data = results.get('metrics', {}).get('comparison', {})
        
        if not comparison_data:
            return "<div class=\"section\"><h2>📈 Strategy Comparison</h2><p>No comparison data available.</p></div>"
        
        # 테이블 헤더
        headers = list(comparison_data.keys())
        strategies = list(comparison_data[headers[0]].values()) if headers else []
        
        table_html = """
        <div class="section">
            <h2>📈 Strategy Comparison</h2>
            <table class="strategy-table">
                <thead>
                    <tr>
        """
        
        for header in headers:
            table_html += f"<th>{header}</th>"
        if strategy_scores:
            table_html += "<th>Grade</th><th>Risk Score</th><th>Reward Score</th>"
        table_html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # 테이블 데이터
        if strategies:
            num_strategies = len(strategies)
            for i in range(num_strategies):
                table_html += "<tr>"
                strategy_name = None
                
                for header in headers:
                    value = comparison_data[header].get(i, 'N/A')
                    if header == 'Strategy':
                        strategy_name = value
                    
                    # 값에 따른 색상 적용
                    css_class = ""
                    if header in ['ROI', 'Sharpe', 'Win Rate', 'Profit Factor']:
                        try:
                            numeric_value = float(str(value).replace('%', ''))
                            if numeric_value > 0:
                                css_class = "positive"
                            elif numeric_value < 0:
                                css_class = "negative"
                            else:
                                css_class = "neutral"
                        except:
                            pass
                    
                    table_html += f'<td class="{css_class}">{value}</td>'
                
                # 전략 점수 추가
                if strategy_scores and strategy_name in strategy_scores:
                    score_data = strategy_scores[strategy_name]
                    grade = score_data.get('overall_grade', 'N/A')
                    risk_score = score_data.get('risk_score', 0)
                    reward_score = score_data.get('reward_score', 0)
                    
                    grade_class = f"grade-{grade[0]}" if grade != 'N/A' else ""
                    table_html += f'<td class="{grade_class}">{grade}</td>'
                    table_html += f'<td>{risk_score:.1f}</td>'
                    table_html += f'<td>{reward_score:.1f}</td>'
                elif strategy_scores:
                    table_html += "<td>N/A</td><td>N/A</td><td>N/A</td>"
                
                table_html += "</tr>"
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        return table_html
    
    def _generate_performance_charts_section(self, results: Dict[str, Any]) -> str:
        """성능 차트 섹션 생성"""
        if not PLOTTING_AVAILABLE:
            return """
            <div class="section">
                <h2>📊 Performance Charts</h2>
                <p>Charts not available (matplotlib not installed)</p>
            </div>
            """
        
        charts_html = """
        <div class="section">
            <h2>📊 Performance Charts</h2>
        """
        
        # 전략별 수익률 비교 차트
        roi_chart = self._create_roi_comparison_chart(results)
        if roi_chart:
            charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{roi_chart}" alt="ROI Comparison"></div>'
        
        # 리스크-수익 분산도
        risk_return_chart = self._create_risk_return_scatter(results)
        if risk_return_chart:
            charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{risk_return_chart}" alt="Risk-Return Scatter"></div>'
        
        charts_html += "</div>"
        return charts_html
    
    def _create_roi_comparison_chart(self, results: Dict[str, Any]) -> Optional[str]:
        """ROI 비교 차트 생성"""
        try:
            comparison_data = results.get('metrics', {}).get('comparison', {})
            if not comparison_data or 'Strategy' not in comparison_data or 'ROI' not in comparison_data:
                return None
            
            strategies = list(comparison_data['Strategy'].values())
            rois = [float(str(roi).replace('%', '')) for roi in comparison_data['ROI'].values()]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#48bb78' if roi > 0 else '#f56565' for roi in rois]
            bars = ax.bar(strategies, rois, color=colors, alpha=0.7)
            
            ax.set_title('Strategy ROI Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('ROI (%)', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 값 라벨 추가
            for bar, roi in zip(bars, rois):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                       f'{roi:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Base64 인코딩
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"ROI 차트 생성 실패: {e}")
            return None
    
    def _create_risk_return_scatter(self, results: Dict[str, Any]) -> Optional[str]:
        """리스크-수익 분산도 생성"""
        try:
            all_metrics = results.get('metrics', {}).get('all_metrics', {})
            if not all_metrics:
                return None
            
            strategies = []
            returns = []
            risks = []
            
            for strategy_name, metrics in all_metrics.items():
                strategies.append(strategy_name)
                returns.append(float(metrics.roi) * 100)
                risks.append(float(metrics.max_drawdown) * 100)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(risks, returns, s=100, alpha=0.7, c=returns, 
                               cmap='RdYlGn', edgecolors='black', linewidth=1)
            
            # 전략 이름 라벨
            for i, strategy in enumerate(strategies):
                ax.annotate(strategy, (risks[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel('Max Drawdown (%)', fontsize=12)
            ax.set_ylabel('ROI (%)', fontsize=12)
            ax.set_title('Risk vs Return Analysis', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # 컬러바
            cbar = plt.colorbar(scatter)
            cbar.set_label('ROI (%)', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            # Base64 인코딩
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"리스크-수익 차트 생성 실패: {e}")
            return None
    
    def _generate_detailed_metrics_section(self, results: Dict[str, Any]) -> str:
        """상세 메트릭 섹션 생성"""
        all_metrics = results.get('metrics', {}).get('all_metrics', {})
        
        if not all_metrics:
            return "<div class=\"section\"><h2>📋 Detailed Metrics</h2><p>No detailed metrics available.</p></div>"
        
        html = """
        <div class="section">
            <h2>📋 Detailed Metrics</h2>
        """
        
        for strategy_name, metrics in all_metrics.items():
            html += f"""
            <h3>{strategy_name}</h3>
            <div class="summary-stats">
                <div class="stat-item">
                    <strong>Sharpe Ratio</strong><br>
                    {float(metrics.sharpe_ratio):.3f}
                </div>
                <div class="stat-item">
                    <strong>Sortino Ratio</strong><br>
                    {float(metrics.sortino_ratio):.3f}
                </div>
                <div class="stat-item">
                    <strong>Calmar Ratio</strong><br>
                    {float(metrics.calmar_ratio):.3f}
                </div>
                <div class="stat-item">
                    <strong>Volatility</strong><br>
                    {float(metrics.volatility):.3f}
                </div>
                <div class="stat-item">
                    <strong>Avg Win</strong><br>
                    ${float(metrics.avg_win):.2f}
                </div>
                <div class="stat-item">
                    <strong>Avg Loss</strong><br>
                    ${float(metrics.avg_loss):.2f}
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_execution_stats_section(self, results: Dict[str, Any]) -> str:
        """실행 통계 섹션 생성"""
        stats = results.get('stats', {})
        cache_stats = stats.get('cache_stats', {})
        
        return f"""
        <div class="section">
            <h2>⚙️ Execution Statistics</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <strong>Cache Hit Rate</strong><br>
                    {cache_stats.get('hit_rate', 0):.3%}
                </div>
                <div class="stat-item">
                    <strong>Cache Size</strong><br>
                    {cache_stats.get('size', 0):,}
                </div>
                <div class="stat-item">
                    <strong>Processing Time</strong><br>
                    {stats.get('execution_time', 0):.1f}s
                </div>
                <div class="stat-item">
                    <strong>Avg Step Time</strong><br>
                    {(sum(stats.get('processing_time', [0])) / len(stats.get('processing_time', [1]))):.3f}s
                </div>
            </div>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """푸터 생성"""
        return f"""
            <div class="footer">
                <p>Generated by AuroraQ Backtest System v2.0</p>
                <p>Report created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """


def generate_html_report(backtest_results: Dict[str, Any], 
                        output_dir: str = "reports/html",
                        title: str = "Backtest Report") -> str:
    """HTML 리포트 생성 (편의 함수)"""
    generator = HTMLReportGenerator(output_dir)
    return generator.generate_report(backtest_results, title=title)


if __name__ == "__main__":
    # 테스트용 더미 데이터
    dummy_results = {
        'success': True,
        'metrics': {
            'all_metrics': {
                'TestStrategy': type('obj', (object,), {
                    'total_return': 0.15,
                    'roi': 0.15,
                    'sharpe_ratio': 1.2,
                    'sortino_ratio': 1.5,
                    'calmar_ratio': 0.8,
                    'max_drawdown': 0.08,
                    'volatility': 0.12,
                    'avg_win': 100.0,
                    'avg_loss': 80.0
                })()
            },
            'comparison': {
                'Strategy': {0: 'TestStrategy'},
                'ROI': {0: '15.00%'},
                'Sharpe': {0: '1.20'}
            }
        },
        'stats': {
            'total_signals': 1000,
            'executed_trades': 250,
            'execution_time': 45.5,
            'cache_stats': {
                'hit_rate': 0.75,
                'size': 500
            },
            'processing_time': [0.02] * 1000
        }
    }
    
    filepath = generate_html_report(dummy_results)
    print(f"테스트 HTML 리포트 생성: {filepath}")