"""
평가 계층 (Evaluation Layer)
- StandardizedMetrics: ROI, Sharpe, Sortino, Profit Factor
- 표본 수 기반 가중치로 전략 점수 산출
- Trade Logs 및 CSV/JSON 기반 보고서 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import csv
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """거래 로그"""
    timestamp: datetime
    strategy: str
    signal_action: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_time: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    signal_confidence: float = 0.0
    position_size: float = 0.0
    regime: str = ""
    volatility: float = 0.0
    sentiment_score: float = 0.0


@dataclass
class PerformanceMetrics:
    """성과 메트릭"""
    # 수익성 메트릭
    total_return: float = 0.0
    roi: float = 0.0
    annualized_return: float = 0.0
    
    # 리스크 조정 수익률
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 리스크 메트릭
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # 거래 통계
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    
    # 일관성 메트릭
    consistency_score: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # 효율성 메트릭
    avg_holding_time_hours: float = 0.0
    avg_trades_per_day: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # 샘플 기반 신뢰도
    sample_size: int = 0
    confidence_level: float = 0.0
    
    # 종합 점수
    composite_score: float = 0.0
    weighted_score: float = 0.0


class SampleWeightCalculator:
    """표본 수 기반 가중치 계산"""
    
    def __init__(self,
                 min_samples: int = 30,
                 optimal_samples: int = 100,
                 confidence_threshold: float = 0.95):
        """
        Args:
            min_samples: 최소 표본 수
            optimal_samples: 최적 표본 수
            confidence_threshold: 신뢰도 임계값
        """
        self.min_samples = min_samples
        self.optimal_samples = optimal_samples
        self.confidence_threshold = confidence_threshold
    
    def calculate_weight(self, sample_size: int) -> float:
        """표본 크기에 따른 가중치 계산"""
        if sample_size < self.min_samples:
            # 최소 표본 미달 시 페널티
            return sample_size / self.min_samples * 0.5
        
        if sample_size >= self.optimal_samples:
            return 1.0
        
        # 선형 보간
        weight = 0.5 + 0.5 * (sample_size - self.min_samples) / (self.optimal_samples - self.min_samples)
        return weight
    
    def calculate_confidence(self, sample_size: int, win_rate: float) -> float:
        """표본 크기와 승률 기반 신뢰도 계산"""
        if sample_size == 0:
            return 0.0
        
        # Wilson Score Interval (근사)
        z = 1.96  # 95% 신뢰수준
        p = win_rate
        n = sample_size
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        
        lower_bound = max(0, center - margin)
        
        # 신뢰도 = 하한값이 0.5를 넘을 확률
        if lower_bound > 0.5:
            confidence = min(1.0, (lower_bound - 0.5) * 2)
        else:
            confidence = lower_bound
        
        return confidence


class MetricsCalculator:
    """메트릭 계산기"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 무위험 수익률 (연율)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = risk_free_rate / 252
    
    def calculate_all_metrics(self,
                            trades: List[TradeLog],
                            initial_capital: float = 1000000,
                            total_days: Optional[int] = None) -> PerformanceMetrics:
        """모든 메트릭 계산"""
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        # 거래를 DataFrame으로 변환
        df = pd.DataFrame([asdict(t) for t in trades])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 기본 통계
        metrics.total_trades = len(trades)
        metrics.sample_size = len(trades)
        
        # PnL 계산
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = initial_capital + df['cumulative_pnl']
        
        # 수익성 메트릭
        total_pnl = df['pnl'].sum()
        metrics.total_return = total_pnl
        
        # ROI 계산 안정화 - 오버플로 방지 및 소수점 값으로 변환
        if initial_capital > 0:
            roi_raw = total_pnl / initial_capital
            # 극단값 제한 (-100% ~ +1000%)
            metrics.roi = float(np.clip(roi_raw, -1.0, 10.0))
        else:
            metrics.roi = 0.0
        
        # 기간 계산
        if total_days is None:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            total_days = max(1, date_range)
        
        # 연율화 수익률 계산 안정화
        if total_days > 0:
            try:
                # 지수 연산 오버플로 방지
                exponent = 365 / total_days
                base = 1 + metrics.roi
                
                # 극단값 처리
                if base <= 0:
                    metrics.annualized_return = -1.0  # 완전 손실
                elif exponent > 100:  # 매우 짧은 기간
                    metrics.annualized_return = metrics.roi  # 단순 복사
                else:
                    annualized_raw = base ** exponent - 1
                    # 극단값 제한 (-100% ~ +1000%)
                    metrics.annualized_return = float(np.clip(annualized_raw, -1.0, 10.0))
            except (OverflowError, ValueError):
                # 연산 오류 시 ROI를 기본값으로 사용
                metrics.annualized_return = metrics.roi
        
        # 일별 수익률 계산
        daily_returns = self._calculate_daily_returns(df, initial_capital)
        
        # 리스크 조정 수익률
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        
        # 드로우다운
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calculate_drawdown(df)
        
        # Calmar Ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        
        # 변동성
        metrics.volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # 거래 통계
        self._calculate_trade_statistics(df, metrics)
        
        # VaR와 CVaR
        if len(daily_returns) > 5:
            metrics.var_95 = np.percentile(daily_returns, 5)
            tail_returns = daily_returns[daily_returns <= metrics.var_95]
            metrics.cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else metrics.var_95
        
        # 일관성 메트릭
        self._calculate_consistency_metrics(df, metrics)
        
        # 효율성 메트릭
        metrics.avg_holding_time_hours = df['holding_time'].mean() / 3600 if 'holding_time' in df else 0
        metrics.avg_trades_per_day = metrics.total_trades / total_days if total_days > 0 else 0
        metrics.total_commission = df['commission'].sum()
        metrics.total_slippage = df['slippage'].sum()
        
        # 종합 점수
        metrics.composite_score = self._calculate_composite_score(metrics)
        
        return metrics
    
    def _calculate_daily_returns(self, df: pd.DataFrame, initial_capital: float) -> pd.Series:
        """일별 수익률 계산"""
        df['date'] = df['timestamp'].dt.date
        daily_pnl = df.groupby('date')['pnl'].sum()
        daily_returns = daily_pnl / initial_capital
        return daily_returns
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """샤프 비율 계산"""
        if len(daily_returns) < 2:
            return 0.0
        
        excess_returns = daily_returns - self.daily_risk_free
        
        if daily_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series) -> float:
        """소르티노 비율 계산"""
        if len(daily_returns) < 2:
            return 0.0
        
        excess_returns = daily_returns - self.daily_risk_free
        downside_returns = daily_returns[daily_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> Tuple[float, int]:
        """최대 드로우다운 계산"""
        running_max = df['equity'].expanding().max()
        drawdown = (df['equity'] - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # 드로우다운 기간 계산
        dd_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if dd_start is None:
                    dd_start = i
                current_dd_duration = i - dd_start
            else:
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                dd_start = None
                current_dd_duration = 0
        
        # 일수로 변환 (대략적)
        max_dd_days = max_dd_duration * (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days / len(df)
        
        return max_drawdown, int(max_dd_days)
    
    def _calculate_trade_statistics(self, df: pd.DataFrame, metrics: PerformanceMetrics):
        """거래 통계 계산"""
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        if len(winning_trades) > 0:
            metrics.avg_win = winning_trades['pnl'].mean()
            total_wins = winning_trades['pnl'].sum()
        else:
            metrics.avg_win = 0
            total_wins = 0
        
        if len(losing_trades) > 0:
            metrics.avg_loss = abs(losing_trades['pnl'].mean())
            total_losses = abs(losing_trades['pnl'].sum())
        else:
            metrics.avg_loss = 0
            total_losses = 0
        
        # Win/Loss Ratio
        if metrics.avg_loss > 0:
            metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss
        
        # Profit Factor
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        
        # Expectancy
        if metrics.total_trades > 0:
            metrics.expectancy = df['pnl'].mean()
    
    def _calculate_consistency_metrics(self, df: pd.DataFrame, metrics: PerformanceMetrics):
        """일관성 메트릭 계산"""
        # 연속 승/패 계산
        pnl_signs = (df['pnl'] > 0).astype(int)
        
        # 최대 연속 승리
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for sign in pnl_signs:
            if sign == 1:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak
        
        # 일관성 점수 (수익 분산 기반)
        if metrics.total_trades > 5:
            pnl_std = df['pnl'].std()
            pnl_mean = abs(df['pnl'].mean())
            if pnl_mean > 0:
                cv = pnl_std / pnl_mean  # 변동계수
                metrics.consistency_score = max(0, min(1, 1 - cv / 2))
    
    def _calculate_composite_score(self, metrics: PerformanceMetrics) -> float:
        """종합 점수 계산"""
        weights = {
            'roi': 0.20,
            'sharpe': 0.15,
            'win_rate': 0.15,
            'profit_factor': 0.10,
            'consistency': 0.15,
            'drawdown': 0.15,
            'expectancy': 0.10
        }
        
        scores = {}
        
        # ROI (연 20% 기준) - 오버플로 안전 처리
        try:
            roi_score_raw = (metrics.roi + 0.1) / 0.3
            scores['roi'] = float(min(1.0, max(0.0, roi_score_raw)))
        except (ValueError, OverflowError):
            scores['roi'] = 0.0
        
        # Sharpe (2.0 기준) - 안전 처리
        try:
            sharpe_score_raw = (metrics.sharpe_ratio + 1.0) / 3.0
            scores['sharpe'] = float(min(1.0, max(0.0, sharpe_score_raw)))
        except (ValueError, OverflowError):
            scores['sharpe'] = 0.0
        
        # Win Rate (60% 기준)
        scores['win_rate'] = min(1.0, metrics.win_rate / 0.6)
        
        # Profit Factor (2.0 기준) - 안전 처리
        try:
            if metrics.profit_factor > 0 and not np.isinf(metrics.profit_factor):
                pf_score_raw = metrics.profit_factor / 2.0
                scores['profit_factor'] = float(min(1.0, max(0.0, pf_score_raw)))
            else:
                scores['profit_factor'] = 0.0
        except (ValueError, OverflowError):
            scores['profit_factor'] = 0.0
        
        # Consistency
        scores['consistency'] = metrics.consistency_score
        
        # Drawdown (10% 기준, 역수)
        scores['drawdown'] = max(0.0, 1.0 - metrics.max_drawdown / 0.1)
        
        # Expectancy (정규화) - 안전 처리
        try:
            if not np.isnan(metrics.expectancy) and not np.isinf(metrics.expectancy):
                if metrics.expectancy > 0:
                    expectancy_score_raw = metrics.expectancy / 1000
                    scores['expectancy'] = float(min(1.0, max(0.0, expectancy_score_raw)))
                else:
                    expectancy_score_raw = 0.5 + metrics.expectancy / 1000
                    scores['expectancy'] = float(max(0.0, min(1.0, expectancy_score_raw)))
            else:
                scores['expectancy'] = 0.5
        except (ValueError, OverflowError):
            scores['expectancy'] = 0.5
        
        # 가중 평균 - 안전 처리
        try:
            composite = sum(scores[k] * weights[k] for k in weights if k in scores)
            # 극단값 제한 (0.0 ~ 1.0)
            composite = float(max(0.0, min(1.0, composite)))
        except (ValueError, OverflowError):
            composite = 0.0
        
        return composite


class ReportGenerator:
    """보고서 생성기"""
    
    def __init__(self, output_dir: str = "reports/backtest"):
        """
        Args:
            output_dir: 출력 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_csv_report(self,
                          trades: List[TradeLog],
                          metrics: PerformanceMetrics,
                          strategy_name: str,
                          timestamp: Optional[datetime] = None) -> str:
        """CSV 보고서 생성"""
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = f"{strategy_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_trades.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # 거래 기록 저장
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if trades:
                fieldnames = list(asdict(trades[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for trade in trades:
                    writer.writerow(asdict(trade))
        
        # 메트릭 요약 저장
        metrics_filename = f"{strategy_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_metrics.csv"
        metrics_filepath = os.path.join(self.output_dir, metrics_filename)
        
        with open(metrics_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in asdict(metrics).items():
                writer.writerow([key, value])
        
        logger.info(f"CSV 보고서 생성: {filepath}, {metrics_filepath}")
        return filepath
    
    def generate_json_report(self,
                           trades: List[TradeLog],
                           metrics: PerformanceMetrics,
                           strategy_name: str,
                           additional_data: Optional[Dict[str, Any]] = None,
                           timestamp: Optional[datetime] = None) -> str:
        """JSON 보고서 생성"""
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = f"{strategy_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_report.json"
        filepath = os.path.join(self.output_dir, filename)
        
        report = {
            "strategy": strategy_name,
            "timestamp": timestamp.isoformat(),
            "metrics": asdict(metrics),
            "trades": [asdict(t) for t in trades],
            "summary": {
                "total_trades": metrics.total_trades,
                "roi": f"{metrics.roi:.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "win_rate": f"{metrics.win_rate:.2%}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "composite_score": f"{metrics.composite_score:.3f}"
            }
        }
        
        if additional_data:
            report["additional_data"] = additional_data
        
        # JSON 직렬화 가능하도록 변환
        def convert_types(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=convert_types, ensure_ascii=False)
        
        logger.info(f"JSON 보고서 생성: {filepath}")
        return filepath


class MetricsEvaluator:
    """
    평가 계층 메인 클래스
    모든 평가 관련 컴포넌트 통합
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 min_sample_size: int = 30):
        """
        Args:
            initial_capital: 초기 자본
            min_sample_size: 최소 표본 크기
        """
        self.initial_capital = initial_capital
        self.min_sample_size = min_sample_size
        
        # 컴포넌트 초기화
        self.calculator = MetricsCalculator()
        self.weight_calculator = SampleWeightCalculator(min_samples=min_sample_size)
        self.report_generator = ReportGenerator()
        
        # 거래 로그 저장
        self.trade_logs: Dict[str, List[TradeLog]] = defaultdict(list)
        self.strategy_metrics: Dict[str, PerformanceMetrics] = {}
    
    def add_trade(self,
                 strategy: str,
                 trade_data: Dict[str, Any]):
        """거래 추가"""
        trade_log = TradeLog(
            timestamp=trade_data.get("timestamp", datetime.now()),
            strategy=strategy,
            signal_action=trade_data.get("signal_action", ""),
            entry_price=trade_data.get("entry_price", 0),
            exit_price=trade_data.get("exit_price"),
            quantity=trade_data.get("quantity", 0),
            pnl=trade_data.get("pnl", 0),
            pnl_pct=trade_data.get("pnl_pct", 0),
            holding_time=trade_data.get("holding_time", 0),
            commission=trade_data.get("commission", 0),
            slippage=trade_data.get("slippage", 0),
            market_impact=trade_data.get("market_impact", 0),
            signal_confidence=trade_data.get("signal_confidence", 0),
            position_size=trade_data.get("position_size", 0),
            regime=trade_data.get("regime", ""),
            volatility=trade_data.get("volatility", 0),
            sentiment_score=trade_data.get("sentiment_score", 0)
        )
        
        self.trade_logs[strategy].append(trade_log)
    
    def evaluate_strategy(self,
                        strategy: str,
                        total_days: Optional[int] = None) -> PerformanceMetrics:
        """전략 평가"""
        trades = self.trade_logs.get(strategy, [])
        
        if not trades:
            return PerformanceMetrics()
        
        # 메트릭 계산
        metrics = self.calculator.calculate_all_metrics(
            trades, self.initial_capital, total_days
        )
        
        # 표본 기반 가중치
        sample_weight = self.weight_calculator.calculate_weight(len(trades))
        confidence = self.weight_calculator.calculate_confidence(
            len(trades), metrics.win_rate
        )
        
        metrics.confidence_level = confidence
        metrics.weighted_score = metrics.composite_score * sample_weight
        
        # 저장
        self.strategy_metrics[strategy] = metrics
        
        return metrics
    
    def compare_strategies(self) -> pd.DataFrame:
        """전략 비교"""
        if not self.strategy_metrics:
            return pd.DataFrame()
        
        comparison_data = []
        
        for strategy, metrics in self.strategy_metrics.items():
            comparison_data.append({
                "Strategy": strategy,
                "Trades": metrics.total_trades,
                "ROI": f"{metrics.roi:.2%}",
                "Sharpe": f"{metrics.sharpe_ratio:.2f}",
                "Win Rate": f"{metrics.win_rate:.2%}",
                "Profit Factor": f"{metrics.profit_factor:.2f}",
                "Max DD": f"{metrics.max_drawdown:.2%}",
                "Composite Score": f"{metrics.composite_score:.3f}",
                "Weighted Score": f"{metrics.weighted_score:.3f}",
                "Confidence": f"{metrics.confidence_level:.2%}"
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Weighted Score", ascending=False)
        
        return df
    
    def generate_reports(self, 
                       strategy: Optional[str] = None,
                       format: str = "both") -> Dict[str, str]:
        """보고서 생성"""
        reports = {}
        
        strategies = [strategy] if strategy else list(self.trade_logs.keys())
        
        for strat in strategies:
            if strat not in self.trade_logs:
                continue
            
            trades = self.trade_logs[strat]
            metrics = self.strategy_metrics.get(strat, PerformanceMetrics())
            
            if format in ["csv", "both"]:
                csv_path = self.report_generator.generate_csv_report(
                    trades, metrics, strat
                )
                reports[f"{strat}_csv"] = csv_path
            
            if format in ["json", "both"]:
                json_path = self.report_generator.generate_json_report(
                    trades, metrics, strat
                )
                reports[f"{strat}_json"] = json_path
        
        # 비교 보고서
        if len(strategies) > 1:
            comparison_df = self.compare_strategies()
            comparison_path = os.path.join(
                self.report_generator.output_dir,
                f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            comparison_df.to_csv(comparison_path, index=False)
            reports["comparison"] = comparison_path
        
        return reports
    
    def get_best_strategy(self) -> Tuple[str, PerformanceMetrics]:
        """최고 성과 전략 반환"""
        if not self.strategy_metrics:
            return "", PerformanceMetrics()
        
        best_strategy = max(
            self.strategy_metrics.items(),
            key=lambda x: x[1].weighted_score
        )
        
        return best_strategy