"""
경량화된 CSV 백테스트 리포터
시각화 없이 CSV 파일로만 결과를 저장하는 효율적인 리포팅 시스템
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.standardized_metrics import StandardizedMetrics, MetricResult
from utils.logger import get_logger

logger = get_logger("LightweightCSVReporter")


class LightweightCSVReporter:
    """
    경량 CSV 리포트 생성기
    - 메모리 효율적인 스트리밍 방식
    - 표준화된 메트릭 사용
    - 다중 전략 비교 지원
    """
    
    def __init__(self, output_dir: str = "report/backtest"):
        """
        Args:
            output_dir: 리포트 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calculator = StandardizedMetrics()
        
        # 리포트 타임스탬프
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_summary_report(self,
                              backtest_results: Dict[str, Any],
                              filename_prefix: str = "backtest_summary") -> str:
        """
        백테스트 요약 리포트 생성
        
        Args:
            backtest_results: 백테스트 결과 딕셔너리
            filename_prefix: 파일명 접두사
            
        Returns:
            생성된 CSV 파일 경로
        """
        filename = f"{filename_prefix}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        try:
            # 전략별 메트릭 계산
            strategy_metrics = {}
            for strategy_name, data in backtest_results.items():
                if 'trades' in data:
                    metrics = self.metrics_calculator.calculate_metrics(
                        trades=data['trades'],
                        initial_capital=data.get('initial_capital', 1000000)
                    )
                    strategy_metrics[strategy_name] = metrics
            
            # CSV 작성
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                # 헤더 정의
                headers = [
                    'Strategy', 'Total_Trades', 'Win_Rate', 'ROI', 'Sharpe_Ratio',
                    'Max_Drawdown', 'Profit_Factor', 'Calmar_Ratio', 'Expectancy',
                    'Avg_Win', 'Avg_Loss', 'Win_Loss_Ratio', 'MAB_Score',
                    'Composite_Score', 'Consistency_Score', 'Recovery_Factor'
                ]
                
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                # 각 전략별 행 작성
                for strategy_name, metrics in strategy_metrics.items():
                    row = {
                        'Strategy': strategy_name,
                        'Total_Trades': metrics.total_trades,
                        'Win_Rate': f"{metrics.win_rate:.4f}",
                        'ROI': f"{metrics.roi:.4f}",
                        'Sharpe_Ratio': f"{metrics.sharpe_ratio:.4f}",
                        'Max_Drawdown': f"{metrics.max_drawdown:.4f}",
                        'Profit_Factor': f"{metrics.profit_factor:.4f}",
                        'Calmar_Ratio': f"{metrics.calmar_ratio:.4f}",
                        'Expectancy': f"{metrics.expectancy:.2f}",
                        'Avg_Win': f"{metrics.avg_win:.2f}",
                        'Avg_Loss': f"{metrics.avg_loss:.2f}",
                        'Win_Loss_Ratio': f"{metrics.avg_win_loss_ratio:.4f}",
                        'MAB_Score': f"{metrics.mab_reward_score:.4f}",
                        'Composite_Score': f"{metrics.composite_score:.4f}",
                        'Consistency_Score': f"{metrics.consistency_score:.4f}",
                        'Recovery_Factor': f"{metrics.recovery_factor:.4f}"
                    }
                    writer.writerow(row)
            
            logger.info(f"요약 리포트 생성 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"요약 리포트 생성 실패: {e}")
            raise
    
    def generate_trades_report(self,
                             trades: List[Dict[str, Any]],
                             strategy_name: str,
                             filename_prefix: str = "trades_detail") -> str:
        """
        거래 상세 리포트 생성
        
        Args:
            trades: 거래 기록 리스트
            strategy_name: 전략 이름
            filename_prefix: 파일명 접두사
            
        Returns:
            생성된 CSV 파일 경로
        """
        filename = f"{filename_prefix}_{strategy_name}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if not trades:
                    f.write("No trades executed\n")
                    return str(filepath)
                
                # 필드명 추출
                fieldnames = list(trades[0].keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # 거래 기록 작성
                for trade in trades:
                    # 타임스탬프 포맷팅
                    if 'timestamp' in trade:
                        trade = trade.copy()
                        if isinstance(trade['timestamp'], datetime):
                            trade['timestamp'] = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow(trade)
            
            logger.info(f"거래 상세 리포트 생성 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"거래 상세 리포트 생성 실패: {e}")
            raise
    
    def generate_comparison_report(self,
                                 strategy_results: Dict[str, Dict[str, Any]],
                                 filename_prefix: str = "strategy_comparison") -> str:
        """
        전략 비교 리포트 생성
        
        Args:
            strategy_results: {전략명: 결과} 딕셔너리
            filename_prefix: 파일명 접두사
            
        Returns:
            생성된 CSV 파일 경로
        """
        filename = f"{filename_prefix}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        try:
            # 비교 데이터 준비
            comparison_data = []
            
            for strategy_name, results in strategy_results.items():
                if 'trades' not in results:
                    continue
                
                metrics = self.metrics_calculator.calculate_metrics(
                    trades=results['trades'],
                    initial_capital=results.get('initial_capital', 1000000)
                )
                
                # 추가 통계
                trade_df = pd.DataFrame(results['trades'])
                
                row_data = {
                    'Strategy': strategy_name,
                    'Period_Start': results.get('start_date', ''),
                    'Period_End': results.get('end_date', ''),
                    'Initial_Capital': results.get('initial_capital', 1000000),
                    'Final_Capital': results.get('final_capital', 0),
                    'Total_PnL': results.get('total_pnl', 0),
                    'Total_Trades': metrics.total_trades,
                    'Win_Rate': metrics.win_rate,
                    'ROI': metrics.roi,
                    'Annual_Return': metrics.roi * (252 / len(trade_df)) if len(trade_df) > 0 else 0,
                    'Sharpe_Ratio': metrics.sharpe_ratio,
                    'Sortino_Ratio': metrics.sortino_ratio,
                    'Max_Drawdown': metrics.max_drawdown,
                    'Calmar_Ratio': metrics.calmar_ratio,
                    'Profit_Factor': metrics.profit_factor,
                    'Expectancy': metrics.expectancy,
                    'Avg_Trade_PnL': trade_df['pnl'].mean() if 'pnl' in trade_df else 0,
                    'Std_Trade_PnL': trade_df['pnl'].std() if 'pnl' in trade_df else 0,
                    'Best_Trade': trade_df['pnl'].max() if 'pnl' in trade_df else 0,
                    'Worst_Trade': trade_df['pnl'].min() if 'pnl' in trade_df else 0,
                    'Avg_Holding_Hours': metrics.avg_holding_time_hours,
                    'MAB_Score': metrics.mab_reward_score,
                    'Composite_Score': metrics.composite_score,
                    'VaR_95': metrics.var_95,
                    'CVaR_95': metrics.cvar_95,
                    'Consistency_Score': metrics.consistency_score,
                    'Max_Consecutive_Wins': metrics.max_consecutive_wins,
                    'Max_Consecutive_Losses': metrics.max_consecutive_losses
                }
                
                comparison_data.append(row_data)
            
            # CSV 작성
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df.to_csv(filepath, index=False, float_format='%.6f')
                
                # 순위 추가 (Composite Score 기준)
                df_sorted = df.sort_values('Composite_Score', ascending=False)
                df_sorted['Rank'] = range(1, len(df_sorted) + 1)
                
                # 순위가 포함된 버전도 저장
                ranked_filename = f"{filename_prefix}_ranked_{self.timestamp}.csv"
                ranked_filepath = self.output_dir / ranked_filename
                df_sorted.to_csv(ranked_filepath, index=False, float_format='%.6f')
                
                logger.info(f"전략 비교 리포트 생성 완료: {filepath}, {ranked_filepath}")
                return str(ranked_filepath)
            else:
                logger.warning("비교할 전략 데이터가 없습니다")
                return ""
                
        except Exception as e:
            logger.error(f"비교 리포트 생성 실패: {e}")
            raise
    
    def generate_period_analysis(self,
                               trades: List[Dict[str, Any]],
                               period: str = "daily",
                               filename_prefix: str = "period_analysis") -> str:
        """
        기간별 분석 리포트 생성
        
        Args:
            trades: 거래 기록
            period: 분석 기간 (daily, weekly, monthly)
            filename_prefix: 파일명 접두사
            
        Returns:
            생성된 CSV 파일 경로
        """
        filename = f"{filename_prefix}_{period}_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        try:
            if not trades:
                logger.warning("거래 기록이 없습니다")
                return ""
            
            # DataFrame 변환
            df = pd.DataFrame(trades)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.error("timestamp 컬럼이 없습니다")
                return ""
            
            # 기간별 그룹핑
            if period == "daily":
                df['period'] = df['timestamp'].dt.date
            elif period == "weekly":
                df['period'] = df['timestamp'].dt.to_period('W')
            elif period == "monthly":
                df['period'] = df['timestamp'].dt.to_period('M')
            else:
                df['period'] = df['timestamp'].dt.date
            
            # 기간별 집계
            period_stats = []
            
            for period_val, group in df.groupby('period'):
                stats = {
                    'Period': str(period_val),
                    'Trades': len(group),
                    'Total_PnL': group['pnl'].sum() if 'pnl' in group else 0,
                    'Win_Rate': (group['pnl'] > 0).mean() if 'pnl' in group else 0,
                    'Avg_PnL': group['pnl'].mean() if 'pnl' in group else 0,
                    'Max_PnL': group['pnl'].max() if 'pnl' in group else 0,
                    'Min_PnL': group['pnl'].min() if 'pnl' in group else 0,
                    'Std_PnL': group['pnl'].std() if 'pnl' in group else 0
                }
                period_stats.append(stats)
            
            # CSV 작성
            period_df = pd.DataFrame(period_stats)
            period_df.to_csv(filepath, index=False, float_format='%.6f')
            
            logger.info(f"기간별 분석 리포트 생성 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"기간별 분석 리포트 생성 실패: {e}")
            raise
    
    def generate_quick_summary(self,
                             metrics: MetricResult,
                             strategy_name: str,
                             output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        빠른 요약 생성 (딕셔너리 또는 간단한 텍스트)
        
        Args:
            metrics: 계산된 메트릭
            strategy_name: 전략 이름
            output_file: 출력 파일 경로 (선택)
            
        Returns:
            요약 딕셔너리
        """
        summary = {
            'strategy': strategy_name,
            'performance': {
                'roi': f"{metrics.roi:.2%}",
                'sharpe': f"{metrics.sharpe_ratio:.2f}",
                'max_dd': f"{metrics.max_drawdown:.2%}",
                'win_rate': f"{metrics.win_rate:.2%}"
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'avg_win_loss': f"{metrics.avg_win_loss_ratio:.2f}"
            },
            'scores': {
                'mab_score': f"{metrics.mab_reward_score:.3f}",
                'composite': f"{metrics.composite_score:.3f}",
                'consistency': f"{metrics.consistency_score:.3f}"
            }
        }
        
        if output_file:
            # JSON으로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def append_to_master_log(self,
                           strategy_name: str,
                           metrics: MetricResult,
                           master_file: str = "master_backtest_log.csv"):
        """
        마스터 로그 파일에 결과 추가 (누적 기록)
        
        Args:
            strategy_name: 전략 이름
            metrics: 메트릭 결과
            master_file: 마스터 로그 파일명
        """
        filepath = self.output_dir / master_file
        
        # 파일 존재 여부 확인
        file_exists = filepath.exists()
        
        try:
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                headers = [
                    'Timestamp', 'Strategy', 'ROI', 'Sharpe', 'Max_DD',
                    'Win_Rate', 'Total_Trades', 'MAB_Score', 'Composite_Score'
                ]
                
                writer = csv.DictWriter(f, fieldnames=headers)
                
                # 헤더 작성 (새 파일인 경우)
                if not file_exists:
                    writer.writeheader()
                
                # 데이터 추가
                row = {
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Strategy': strategy_name,
                    'ROI': f"{metrics.roi:.4f}",
                    'Sharpe': f"{metrics.sharpe_ratio:.4f}",
                    'Max_DD': f"{metrics.max_drawdown:.4f}",
                    'Win_Rate': f"{metrics.win_rate:.4f}",
                    'Total_Trades': metrics.total_trades,
                    'MAB_Score': f"{metrics.mab_reward_score:.4f}",
                    'Composite_Score': f"{metrics.composite_score:.4f}"
                }
                writer.writerow(row)
                
            logger.info(f"마스터 로그 업데이트: {strategy_name}")
            
        except Exception as e:
            logger.error(f"마스터 로그 업데이트 실패: {e}")


# 전역 리포터 인스턴스
_reporter = None


def get_csv_reporter(output_dir: Optional[str] = None) -> LightweightCSVReporter:
    """전역 CSV 리포터 인스턴스 반환"""
    global _reporter
    if _reporter is None or output_dir:
        _reporter = LightweightCSVReporter(output_dir or "report/backtest")
    return _reporter