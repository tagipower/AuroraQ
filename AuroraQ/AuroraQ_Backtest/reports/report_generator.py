#!/usr/bin/env python3
"""
리포트 생성기 - 백테스트 결과를 종합 리포트로 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
from dataclasses import asdict

from .visualizer import BacktestVisualizer


class ReportGenerator:
    """백테스트 리포트 생성기"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.visualizer = BacktestVisualizer()
        
    def generate_comprehensive_report(self,
                                    backtest_result,
                                    strategy_name: str = "Strategy",
                                    include_charts: bool = True) -> str:
        """종합 백테스트 리포트 생성"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{strategy_name}_report_{timestamp}"
        
        # 리포트 디렉토리 생성
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        # 1. HTML 리포트 생성
        html_path = report_dir / f"{report_name}.html"
        self._generate_html_report(backtest_result, strategy_name, html_path)
        
        # 2. JSON 데이터 저장
        json_path = report_dir / f"{report_name}_data.json"
        self._save_json_data(backtest_result, json_path)
        
        # 3. CSV 데이터 저장
        csv_dir = report_dir / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        self._save_csv_data(backtest_result, csv_dir)
        
        # 4. 차트 생성
        if include_charts:
            charts_dir = report_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            self._generate_charts(backtest_result, charts_dir)
        
        # 5. 요약 리포트 생성
        summary_path = report_dir / f"{report_name}_summary.txt"
        self._generate_summary_report(backtest_result, strategy_name, summary_path)
        
        return str(report_dir)
    
    def _generate_html_report(self, result, strategy_name: str, output_path: Path):
        """HTML 리포트 생성"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{strategy_name} 백테스트 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #7f8c8d; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        .chart-placeholder {{
            background-color: #ecf0f1;
            height: 300px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px 0;
            font-style: italic;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {strategy_name} 백테스트 리포트</h1>
        
        <div class="section">
            <h2>📈 핵심 성과 지표</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">총 수익률</div>
                    <div class="metric-value {'positive' if result.total_return >= 0 else 'negative'}">
                        {result.total_return:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">연환산 수익률</div>
                    <div class="metric-value {'positive' if result.annualized_return >= 0 else 'negative'}">
                        {result.annualized_return:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">최대 낙폭</div>
                    <div class="metric-value negative">
                        {result.max_drawdown:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">샤프 비율</div>
                    <div class="metric-value {'positive' if result.sharpe_ratio >= 1 else 'neutral'}">
                        {result.sharpe_ratio:.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">승률</div>
                    <div class="metric-value {'positive' if result.win_rate >= 0.5 else 'neutral'}">
                        {result.win_rate:.1%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">총 거래</div>
                    <div class="metric-value neutral">
                        {result.total_trades}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 상세 성과 분석</h2>
            <table>
                <tr>
                    <th>지표</th>
                    <th>값</th>
                    <th>설명</th>
                </tr>
                <tr>
                    <td>백테스트 기간</td>
                    <td>{result.start_date.date()} ~ {result.end_date.date()}</td>
                    <td>분석 대상 기간</td>
                </tr>
                <tr>
                    <td>초기 자본</td>
                    <td>${result.initial_capital:,.0f}</td>
                    <td>백테스트 시작 자본</td>
                </tr>
                <tr>
                    <td>최종 자본</td>
                    <td>${result.final_capital:,.0f}</td>
                    <td>백테스트 종료 자본</td>
                </tr>
                <tr>
                    <td>변동성 (연환산)</td>
                    <td>{result.volatility:.2%}</td>
                    <td>수익률의 표준편차</td>
                </tr>
                <tr>
                    <td>소르티노 비율</td>
                    <td>{result.sortino_ratio:.2f}</td>
                    <td>하방 리스크 대비 수익률</td>
                </tr>
                <tr>
                    <td>칼마 비율</td>
                    <td>{result.calmar_ratio:.2f}</td>
                    <td>최대 낙폭 대비 수익률</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>💹 거래 통계</h2>
            <table>
                <tr>
                    <th>지표</th>
                    <th>값</th>
                </tr>
                <tr>
                    <td>총 거래 수</td>
                    <td>{result.total_trades}</td>
                </tr>
                <tr>
                    <td>승리한 거래</td>
                    <td>{result.winning_trades}</td>
                </tr>
                <tr>
                    <td>손실 거래</td>
                    <td>{result.losing_trades}</td>
                </tr>
                <tr>
                    <td>승률</td>
                    <td>{result.win_rate:.1%}</td>
                </tr>
                <tr>
                    <td>평균 수익</td>
                    <td>{result.avg_win:.2%}</td>
                </tr>
                <tr>
                    <td>평균 손실</td>
                    <td>{result.avg_loss:.2%}</td>
                </tr>
                <tr>
                    <td>손익비</td>
                    <td>{result.profit_factor:.2f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 차트 분석</h2>
            <div class="chart-placeholder">
                자산가치 곡선 차트 (charts/equity_curve.png 참조)
            </div>
            <div class="chart-placeholder">
                수익률 분포 차트 (charts/returns_distribution.png 참조)
            </div>
            <div class="chart-placeholder">
                거래 분석 차트 (charts/trade_analysis.png 참조)
            </div>
        </div>
        
        <div class="section">
            <h2>💡 결론 및 권장사항</h2>
            {self._generate_recommendations(result)}
        </div>
        
        <div class="section">
            <p><small>리포트 생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_recommendations(self, result) -> str:
        """성과 기반 권장사항 생성"""
        
        recommendations = []
        
        # 수익률 분석
        if result.total_return > 0.2:
            recommendations.append("✅ <strong>우수한 수익률:</strong> 전략이 높은 수익률을 달성했습니다.")
        elif result.total_return > 0:
            recommendations.append("📈 <strong>양의 수익률:</strong> 전략이 수익을 창출했으나 개선 여지가 있습니다.")
        else:
            recommendations.append("⚠️ <strong>손실 발생:</strong> 전략 재검토가 필요합니다.")
        
        # 샤프 비율 분석
        if result.sharpe_ratio > 1.5:
            recommendations.append("✅ <strong>우수한 리스크 조정 수익률:</strong> 샤프 비율이 1.5 이상입니다.")
        elif result.sharpe_ratio > 1:
            recommendations.append("📊 <strong>양호한 리스크 조정 수익률:</strong> 샤프 비율이 1 이상입니다.")
        else:
            recommendations.append("⚠️ <strong>리스크 대비 수익률 개선 필요:</strong> 샤프 비율이 1 미만입니다.")
        
        # 최대 낙폭 분석
        if result.max_drawdown < 0.1:
            recommendations.append("✅ <strong>낮은 최대 낙폭:</strong> 리스크 관리가 우수합니다.")
        elif result.max_drawdown < 0.2:
            recommendations.append("📉 <strong>적절한 최대 낙폭:</strong> 리스크 관리가 양호합니다.")
        else:
            recommendations.append("⚠️ <strong>높은 최대 낙폭:</strong> 리스크 관리 강화가 필요합니다.")
        
        # 승률 분석
        if result.win_rate > 0.6:
            recommendations.append("✅ <strong>높은 승률:</strong> 거래 정확도가 우수합니다.")
        elif result.win_rate > 0.5:
            recommendations.append("📈 <strong>양호한 승률:</strong> 거래 정확도가 평균 이상입니다.")
        else:
            recommendations.append("⚠️ <strong>낮은 승률:</strong> 거래 정확도 개선이 필요합니다.")
        
        # 거래 빈도 분석
        if result.total_trades < 10:
            recommendations.append("📊 <strong>낮은 거래 빈도:</strong> 더 많은 기회를 포착할 수 있는지 검토해보세요.")
        elif result.total_trades > 1000:
            recommendations.append("⚡ <strong>높은 거래 빈도:</strong> 거래 비용이 성과에 미치는 영향을 확인해보세요.")
        
        return "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"
    
    def _save_json_data(self, result, output_path: Path):
        """JSON 형태로 데이터 저장"""
        
        # 결과를 딕셔너리로 변환
        result_dict = {
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'max_drawdown': result.max_drawdown,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'profit_factor': result.profit_factor
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    def _save_csv_data(self, result, output_dir: Path):
        """CSV 형태로 데이터 저장"""
        
        # 자산가치 곡선
        if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
            equity_df = result.equity_curve.to_frame('equity')
            equity_df.to_csv(output_dir / 'equity_curve.csv')
        
        # 수익률
        if hasattr(result, 'returns') and not result.returns.empty:
            returns_df = result.returns.to_frame('returns')
            returns_df.to_csv(output_dir / 'returns.csv')
        
        # 포지션
        if hasattr(result, 'positions') and not result.positions.empty:
            result.positions.to_csv(output_dir / 'positions.csv')
        
        # 거래 내역
        if hasattr(result, 'trades') and not result.trades.empty:
            result.trades.to_csv(output_dir / 'trades.csv', index=False)
        
        # 월별 수익률
        if hasattr(result, 'monthly_returns') and not result.monthly_returns.empty:
            monthly_df = result.monthly_returns.to_frame('monthly_returns')
            monthly_df.to_csv(output_dir / 'monthly_returns.csv')
    
    def _generate_charts(self, result, output_dir: Path):
        """차트 생성 및 저장"""
        
        try:
            # 1. 자산가치 곡선
            if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
                drawdown = getattr(result, 'drawdown_series', None)
                fig = self.visualizer.plot_equity_curve(
                    result.equity_curve,
                    drawdown=drawdown,
                    save_path=output_dir / 'equity_curve.png'
                )
                fig.close()
            
            # 2. 수익률 분포
            if hasattr(result, 'returns') and not result.returns.empty:
                fig = self.visualizer.plot_returns_distribution(
                    result.returns,
                    save_path=output_dir / 'returns_distribution.png'
                )
                fig.close()
            
            # 3. 거래 분석
            if hasattr(result, 'trades') and not result.trades.empty:
                fig = self.visualizer.plot_trade_analysis(
                    result.trades,
                    result.equity_curve,
                    save_path=output_dir / 'trade_analysis.png'
                )
                fig.close()
            
            # 4. 성과 지표
            metrics = {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'monthly_returns': getattr(result, 'monthly_returns', pd.Series()),
                'yearly_returns': getattr(result, 'yearly_returns', pd.Series()),
                'rolling_sharpe': getattr(result, 'rolling_sharpe', pd.Series())
            }
            
            fig = self.visualizer.plot_performance_metrics(
                metrics,
                save_path=output_dir / 'performance_metrics.png'
            )
            fig.close()
            
            # 5. 종합 대시보드
            fig = self.visualizer.create_dashboard(
                result.equity_curve,
                result.returns,
                result.trades,
                metrics,
                save_path=output_dir / 'dashboard.png'
            )
            fig.close()
            
        except Exception as e:
            print(f"차트 생성 중 오류 발생: {e}")
    
    def _generate_summary_report(self, result, strategy_name: str, output_path: Path):
        """간단한 텍스트 요약 리포트"""
        
        summary = f"""
{strategy_name} 백테스트 결과 요약
{'=' * 50}

백테스트 기간: {result.start_date.date()} ~ {result.end_date.date()}
백테스트 일수: {(result.end_date - result.start_date).days}일

[자본 현황]
초기 자본: ${result.initial_capital:,.0f}
최종 자본: ${result.final_capital:,.0f}
순손익: ${result.final_capital - result.initial_capital:,.0f}

[수익률 지표]
총 수익률: {result.total_return:.2%}
연환산 수익률: {result.annualized_return:.2%}

[리스크 지표]
최대 낙폭: {result.max_drawdown:.2%}
변동성 (연환산): {result.volatility:.2%}
샤프 비율: {result.sharpe_ratio:.2f}
소르티노 비율: {result.sortino_ratio:.2f}
칼마 비율: {result.calmar_ratio:.2f}

[거래 통계]
총 거래 수: {result.total_trades}
승리한 거래: {result.winning_trades}
손실 거래: {result.losing_trades}
승률: {result.win_rate:.1%}
평균 수익: {result.avg_win:.2%}
평균 손실: {result.avg_loss:.2%}
손익비: {result.profit_factor:.2f}

[성과 평가]
{self._get_performance_grade(result)}

리포트 생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    
    def _get_performance_grade(self, result) -> str:
        """성과 등급 평가"""
        
        score = 0
        
        # 수익률 점수 (40점)
        if result.total_return > 0.3:
            score += 40
        elif result.total_return > 0.1:
            score += 30
        elif result.total_return > 0:
            score += 20
        else:
            score += 0
        
        # 샤프 비율 점수 (30점)
        if result.sharpe_ratio > 2:
            score += 30
        elif result.sharpe_ratio > 1:
            score += 20
        elif result.sharpe_ratio > 0:
            score += 10
        else:
            score += 0
        
        # 최대 낙폭 점수 (20점)
        if result.max_drawdown < 0.05:
            score += 20
        elif result.max_drawdown < 0.1:
            score += 15
        elif result.max_drawdown < 0.2:
            score += 10
        else:
            score += 0
        
        # 승률 점수 (10점)
        if result.win_rate > 0.6:
            score += 10
        elif result.win_rate > 0.5:
            score += 5
        else:
            score += 0
        
        # 등급 부여
        if score >= 90:
            return f"A등급 ({score}점) - 탁월한 성과"
        elif score >= 80:
            return f"B등급 ({score}점) - 우수한 성과"
        elif score >= 70:
            return f"C등급 ({score}점) - 양호한 성과"
        elif score >= 60:
            return f"D등급 ({score}점) - 보통 성과"
        else:
            return f"F등급 ({score}점) - 개선 필요"
    
    def compare_strategies(self,
                         results: Dict[str, any],
                         save_path: Optional[str] = None) -> str:
        """여러 전략 비교 리포트"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"strategy_comparison_{timestamp}"
        
        if save_path:
            output_dir = Path(save_path)
        else:
            output_dir = self.output_dir / report_name
        
        output_dir.mkdir(exist_ok=True)
        
        # 비교 테이블 생성
        comparison_data = []
        for strategy_name, result in results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Annualized Return': f"{result.annualized_return:.2%}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Win Rate': f"{result.win_rate:.1%}",
                'Total Trades': result.total_trades
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # CSV 저장
        comparison_df.to_csv(output_dir / 'strategy_comparison.csv', index=False)
        
        # HTML 리포트 생성
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>전략 비교 리포트</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .best {{ background-color: #d4edda; }}
        .worst {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>전략 비교 리포트</h1>
    <h2>성과 비교</h2>
    {comparison_df.to_html(index=False, table_id='comparison', classes='table')}
    <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>
"""
        
        with open(output_dir / 'comparison_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_dir)