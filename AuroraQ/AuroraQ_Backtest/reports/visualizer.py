#!/usr/bin/env python3
"""
백테스트 시각화 - 차트 및 그래프 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class BacktestVisualizer:
    """백테스트 결과 시각화"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.color_scheme = {
            'profit': '#2E8B57',    # SeaGreen
            'loss': '#DC143C',      # Crimson
            'neutral': '#4169E1',   # RoyalBlue
            'background': '#F5F5F5', # WhiteSmoke
            'grid': '#D3D3D3'       # LightGray
        }
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_equity_curve(self,
                         equity_curve: pd.Series,
                         benchmark: Optional[pd.Series] = None,
                         drawdown: Optional[pd.Series] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """자산가치 곡선 및 낙폭 차트"""
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        
        # 1. 자산가치 곡선
        ax1 = axes[0]
        
        # 초기 자본 대비 퍼센트로 변환
        initial_value = equity_curve.iloc[0]
        equity_pct = (equity_curve / initial_value - 1) * 100
        
        ax1.plot(equity_curve.index, equity_pct, 
                linewidth=2, color=self.color_scheme['profit'], 
                label='Portfolio')
        
        # 벤치마크 (있는 경우)
        if benchmark is not None:
            benchmark_initial = benchmark.iloc[0]
            benchmark_pct = (benchmark / benchmark_initial - 1) * 100
            ax1.plot(benchmark.index, benchmark_pct,
                    linewidth=1.5, color=self.color_scheme['neutral'],
                    linestyle='--', alpha=0.7, label='Benchmark')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Portfolio Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 낙폭 차트
        ax2 = axes[1]
        
        if drawdown is not None:
            drawdown_pct = drawdown * 100
            ax2.fill_between(drawdown.index, drawdown_pct, 0,
                           color=self.color_scheme['loss'], alpha=0.7)
            ax2.plot(drawdown.index, drawdown_pct,
                    color=self.color_scheme['loss'], linewidth=1)
        
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 날짜 포맷팅
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_returns_distribution(self,
                                returns: pd.Series,
                                save_path: Optional[str] = None) -> plt.Figure:
        """수익률 분포 차트"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 히스토그램
        ax1 = axes[0, 0]
        returns_pct = returns * 100
        
        ax1.hist(returns_pct, bins=50, alpha=0.7, color=self.color_scheme['neutral'])
        ax1.axvline(returns_pct.mean(), color=self.color_scheme['profit'], 
                   linestyle='--', linewidth=2, label=f'Mean: {returns_pct.mean():.2f}%')
        ax1.axvline(returns_pct.median(), color=self.color_scheme['loss'], 
                   linestyle='--', linewidth=2, label=f'Median: {returns_pct.median():.2f}%')
        
        ax1.set_title('Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 박스플롯
        ax3 = axes[1, 0]
        
        # 월별 수익률
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        if len(monthly_returns) > 1:
            box_data = [monthly_returns.values]
            ax3.boxplot(box_data, labels=['Monthly Returns'])
            ax3.set_title('Monthly Returns Box Plot', fontweight='bold')
            ax3.set_ylabel('Monthly Returns (%)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 롤링 샤프 비율
        ax4 = axes[1, 1]
        
        if len(returns) >= 30:
            rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252)
            ax4.plot(rolling_sharpe.index, rolling_sharpe, 
                    color=self.color_scheme['neutral'], linewidth=1.5)
            ax4.axhline(y=1, color=self.color_scheme['profit'], 
                       linestyle='--', alpha=0.7, label='Sharpe = 1')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax4.set_title('Rolling Sharpe Ratio (30-day)', fontweight='bold')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(self,
                          trades: pd.DataFrame,
                          equity_curve: pd.Series,
                          save_path: Optional[str] = None) -> plt.Figure:
        """거래 분석 차트"""
        
        if trades.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No trades to display', 
                   ha='center', va='center', fontsize=16)
            ax.set_title('Trade Analysis')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 거래 표시가 있는 가격 차트
        ax1 = axes[0, 0]
        
        # 가격 데이터가 있다면 표시
        if hasattr(equity_curve, 'index'):
            ax1.plot(equity_curve.index, equity_curve, 
                    color='black', linewidth=1, alpha=0.7, label='Equity')
            
            # 거래 포인트 표시
            if 'timestamp' in trades.columns:
                buy_trades = trades[trades['side'] == 'buy']
                sell_trades = trades[trades['side'] == 'sell']
                
                for _, trade in buy_trades.iterrows():
                    if trade['timestamp'] in equity_curve.index:
                        equity_value = equity_curve.loc[trade['timestamp']]
                        ax1.scatter(trade['timestamp'], equity_value, 
                                  color=self.color_scheme['profit'], 
                                  marker='^', s=50, alpha=0.8)
                
                for _, trade in sell_trades.iterrows():
                    if trade['timestamp'] in equity_curve.index:
                        equity_value = equity_curve.loc[trade['timestamp']]
                        ax1.scatter(trade['timestamp'], equity_value, 
                                  color=self.color_scheme['loss'], 
                                  marker='v', s=50, alpha=0.8)
        
        ax1.set_title('Trades on Equity Curve', fontweight='bold')
        ax1.set_ylabel('Equity Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 거래 크기 분포
        ax2 = axes[0, 1]
        
        if 'size' in trades.columns:
            ax2.hist(trades['size'], bins=20, alpha=0.7, 
                    color=self.color_scheme['neutral'])
            ax2.set_title('Trade Size Distribution', fontweight='bold')
            ax2.set_xlabel('Trade Size')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # 3. 시간별 거래 패턴
        ax3 = axes[1, 0]
        
        if 'timestamp' in trades.columns:
            trades_copy = trades.copy()
            trades_copy['hour'] = pd.to_datetime(trades_copy['timestamp']).dt.hour
            hourly_counts = trades_copy['hour'].value_counts().sort_index()
            
            ax3.bar(hourly_counts.index, hourly_counts.values, 
                   color=self.color_scheme['neutral'], alpha=0.7)
            ax3.set_title('Trading Pattern by Hour', fontweight='bold')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Number of Trades')
            ax3.grid(True, alpha=0.3)
        
        # 4. 수수료 누적
        ax4 = axes[1, 1]
        
        if 'commission' in trades.columns:
            cumulative_commission = trades['commission'].cumsum()
            ax4.plot(range(len(cumulative_commission)), cumulative_commission,
                    color=self.color_scheme['loss'], linewidth=2)
            ax4.set_title('Cumulative Commission', fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative Commission')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self,
                               metrics: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        """성과 지표 시각화"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 주요 수익률 지표
        ax1 = axes[0, 0]
        
        returns_metrics = {
            'Total Return': metrics.get('total_return', 0) * 100,
            'Annualized Return': metrics.get('annualized_return', 0) * 100,
            'Max Drawdown': -metrics.get('max_drawdown', 0) * 100
        }
        
        colors = [self.color_scheme['profit'] if v >= 0 else self.color_scheme['loss'] 
                 for v in returns_metrics.values()]
        
        bars = ax1.bar(returns_metrics.keys(), returns_metrics.values(), color=colors)
        ax1.set_title('Return Metrics (%)', fontweight='bold')
        ax1.set_ylabel('Percentage (%)')
        ax1.grid(True, alpha=0.3)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, returns_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. 리스크 지표
        ax2 = axes[0, 1]
        
        risk_metrics = {
            'Volatility': metrics.get('volatility', 0) * 100,
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0)
        }
        
        bars = ax2.bar(risk_metrics.keys(), risk_metrics.values(), 
                      color=self.color_scheme['neutral'])
        ax2.set_title('Risk Metrics', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, risk_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. 거래 통계
        ax3 = axes[0, 2]
        
        trade_stats = {
            'Win Rate': metrics.get('win_rate', 0) * 100,
            'Total Trades': metrics.get('total_trades', 0)
        }
        
        if trade_stats['Total Trades'] > 0:
            # 파이 차트로 승/패 비율 표시
            winning = metrics.get('winning_trades', 0)
            losing = metrics.get('losing_trades', 0)
            
            if winning + losing > 0:
                labels = ['Winning', 'Losing']
                sizes = [winning, losing]
                colors = [self.color_scheme['profit'], self.color_scheme['loss']]
                
                ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90)
                ax3.set_title(f'Trade Results\n(Total: {winning + losing})', fontweight='bold')
        
        # 4. 월별 수익률 히트맵
        ax4 = axes[1, 0]
        
        if 'monthly_returns' in metrics and not metrics['monthly_returns'].empty:
            monthly_returns = metrics['monthly_returns']
            
            # 연도와 월로 분리
            monthly_data = monthly_returns.to_frame('return')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            # 피벗 테이블 생성
            pivot_table = monthly_data.pivot(index='year', columns='month', values='return')
            
            # 히트맵 그리기
            sns.heatmap(pivot_table * 100, annot=True, fmt='.1f', 
                       cmap='RdYlGn', center=0, ax=ax4,
                       cbar_kws={'label': 'Monthly Return (%)'})
            ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Year')
        
        # 5. 연도별 수익률
        ax5 = axes[1, 1]
        
        if 'yearly_returns' in metrics and not metrics['yearly_returns'].empty:
            yearly_returns = metrics['yearly_returns'] * 100
            
            colors = [self.color_scheme['profit'] if v >= 0 else self.color_scheme['loss'] 
                     for v in yearly_returns]
            
            bars = ax5.bar(yearly_returns.index.year, yearly_returns.values, color=colors)
            ax5.set_title('Annual Returns', fontweight='bold')
            ax5.set_ylabel('Return (%)')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, yearly_returns.values):
                ax5.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (1 if value >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 6. 롤링 지표
        ax6 = axes[1, 2]
        
        if 'rolling_sharpe' in metrics and not metrics['rolling_sharpe'].empty:
            rolling_sharpe = metrics['rolling_sharpe']
            ax6.plot(rolling_sharpe.index, rolling_sharpe, 
                    color=self.color_scheme['neutral'], linewidth=1.5)
            ax6.axhline(y=1, color=self.color_scheme['profit'], 
                       linestyle='--', alpha=0.7, label='Good (1.0)')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax6.set_title('Rolling Sharpe Ratio', fontweight='bold')
            ax6.set_ylabel('Sharpe Ratio')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self,
                        equity_curve: pd.Series,
                        returns: pd.Series,
                        trades: pd.DataFrame,
                        metrics: Dict,
                        save_path: Optional[str] = None) -> plt.Figure:
        """종합 대시보드 생성"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 레이아웃 설정
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 자산가치 곡선 (상단 전체)
        ax1 = fig.add_subplot(gs[0, :])
        
        initial_value = equity_curve.iloc[0]
        equity_pct = (equity_curve / initial_value - 1) * 100
        
        ax1.plot(equity_curve.index, equity_pct, linewidth=2, 
                color=self.color_scheme['profit'])
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 수익률 분포
        ax2 = fig.add_subplot(gs[1, 0])
        returns_pct = returns * 100
        ax2.hist(returns_pct, bins=30, alpha=0.7, color=self.color_scheme['neutral'])
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Returns (%)')
        
        # 3. 낙폭
        ax3 = fig.add_subplot(gs[1, 1])
        if 'drawdown_series' in metrics:
            drawdown = metrics['drawdown_series'] * 100
            ax3.fill_between(drawdown.index, drawdown, 0, 
                           color=self.color_scheme['loss'], alpha=0.7)
            ax3.set_title('Drawdown')
            ax3.set_ylabel('Drawdown (%)')
        
        # 4. 주요 지표 표
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.axis('off')
        
        # 지표 테이블
        metrics_table = [
            ['Total Return', f"{metrics.get('total_return', 0):.2%}"],
            ['Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1%}"],
            ['Total Trades', f"{metrics.get('total_trades', 0)}"]
        ]
        
        table = ax4.table(cellText=metrics_table,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 5. 월별 수익률
        ax5 = fig.add_subplot(gs[2, :2])
        if 'monthly_returns' in metrics and not metrics['monthly_returns'].empty:
            monthly_returns = metrics['monthly_returns'] * 100
            colors = [self.color_scheme['profit'] if v >= 0 else self.color_scheme['loss'] 
                     for v in monthly_returns]
            ax5.bar(range(len(monthly_returns)), monthly_returns, color=colors)
            ax5.set_title('Monthly Returns')
            ax5.set_ylabel('Return (%)')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 6. 거래 패턴
        ax6 = fig.add_subplot(gs[2, 2:])
        if not trades.empty and 'timestamp' in trades.columns:
            trade_counts = trades.groupby(trades['timestamp'].dt.month).size()
            ax6.bar(trade_counts.index, trade_counts.values, 
                   color=self.color_scheme['neutral'])
            ax6.set_title('Trades by Month')
            ax6.set_xlabel('Month')
            ax6.set_ylabel('Number of Trades')
        
        # 7. 리스크 지표 (하단)
        ax7 = fig.add_subplot(gs[3, :])
        
        risk_categories = ['Return', 'Risk', 'Efficiency']
        return_score = min(100, max(0, (metrics.get('total_return', 0) + 0.5) * 100))
        risk_score = min(100, max(0, (1 - metrics.get('max_drawdown', 1)) * 100))
        efficiency_score = min(100, max(0, (metrics.get('sharpe_ratio', 0) + 2) * 25))
        
        scores = [return_score, risk_score, efficiency_score]
        colors = [self.color_scheme['profit'], self.color_scheme['neutral'], self.color_scheme['loss']]
        
        bars = ax7.barh(risk_categories, scores, color=colors)
        ax7.set_title('Performance Score (0-100)', fontweight='bold')
        ax7.set_xlabel('Score')
        ax7.set_xlim(0, 100)
        
        for bar, score in zip(bars, scores):
            ax7.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{score:.0f}', va='center')
        
        plt.suptitle('Backtest Results Dashboard', fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig