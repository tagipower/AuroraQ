"""
표준화된 메트릭 계산 시스템
모든 전략에서 일관된 성과 측정을 위한 통합 메트릭
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """거래 기록 표준 형식"""
    timestamp: datetime
    action: str  # BUY, SELL
    price: float
    quantity: float = 1.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_time: Optional[float] = None  # seconds
    entry_reason: str = ""
    exit_reason: str = ""
    strategy: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_time": self.holding_time,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
            "strategy": self.strategy
        }


@dataclass
class MetricResult:
    """메트릭 계산 결과"""
    # 기본 메트릭
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # 거래 통계
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    
    # 거래 빈도
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trades_per_day: float = 0.0
    avg_holding_time_hours: float = 0.0
    
    # 리스크 메트릭
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    downside_deviation: float = 0.0
    ulcer_index: float = 0.0
    
    # 일관성 메트릭
    consistency_score: float = 0.0  # 0~1
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    
    # MAB 최적화 메트릭
    mab_reward_score: float = 0.0  # 통합 보상 점수
    risk_adjusted_return: float = 0.0
    
    # 종합 점수
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """딕셔너리 변환"""
        return {k: v for k, v in self.__dict__.items()}


class StandardizedMetrics:
    """
    표준화된 메트릭 계산 클래스
    - 모든 전략에 동일한 메트릭 적용
    - MAB 최적화를 위한 통합 점수 제공
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 무위험 수익률 (연율)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = risk_free_rate / 252
    
    def calculate_metrics(self,
                         trades: List[Dict[str, Any]],
                         initial_capital: float = 1000000,
                         price_data: Optional[pd.DataFrame] = None) -> MetricResult:
        """
        표준화된 메트릭 계산
        
        Args:
            trades: 거래 기록 리스트
            initial_capital: 초기 자본
            price_data: 가격 데이터 (선택)
            
        Returns:
            MetricResult 객체
        """
        if not trades:
            return MetricResult()
        
        # DataFrame 변환
        df = pd.DataFrame(trades)
        
        # 기본 계산
        result = MetricResult()
        
        # 수익률 계산
        self._calculate_returns(df, initial_capital, result)
        
        # 거래 통계
        self._calculate_trade_stats(df, result)
        
        # 리스크 메트릭
        self._calculate_risk_metrics(df, result)
        
        # 일관성 메트릭
        self._calculate_consistency_metrics(df, result)
        
        # MAB 최적화 점수
        self._calculate_mab_score(result)
        
        # 종합 점수
        self._calculate_composite_score(result)
        
        return result
    
    def _calculate_returns(self, 
                          df: pd.DataFrame,
                          initial_capital: float,
                          result: MetricResult):
        """수익률 관련 메트릭 계산"""
        if 'pnl' not in df.columns:
            return
        
        # 누적 PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = initial_capital + df['cumulative_pnl']
        
        # ROI
        total_pnl = df['pnl'].sum()
        result.roi = total_pnl / initial_capital
        
        # 일별 수익률 (근사치)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            daily_pnl = df.groupby('date')['pnl'].sum()
            daily_returns = daily_pnl / initial_capital
            
            # Sharpe Ratio
            if len(daily_returns) > 1:
                excess_returns = daily_returns - self.daily_risk_free
                if daily_returns.std() > 0:
                    result.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
            
            # Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    result.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
        
        # Maximum Drawdown
        running_max = df['equity'].expanding().max()
        drawdown = (df['equity'] - running_max) / running_max
        result.max_drawdown = abs(drawdown.min())
        
        # Calmar Ratio
        if result.max_drawdown > 0 and len(df) > 0:
            annual_return = result.roi * (252 / len(df))  # 근사치
            result.calmar_ratio = annual_return / result.max_drawdown
    
    def _calculate_trade_stats(self, df: pd.DataFrame, result: MetricResult):
        """거래 통계 계산"""
        result.total_trades = len(df)
        
        if 'pnl' in df.columns:
            winning_trades = df[df['pnl'] > 0]
            losing_trades = df[df['pnl'] < 0]
            
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            
            # Win Rate
            if result.total_trades > 0:
                result.win_rate = result.winning_trades / result.total_trades
            
            # Average Win/Loss
            if len(winning_trades) > 0:
                result.avg_win = winning_trades['pnl'].mean()
            
            if len(losing_trades) > 0:
                result.avg_loss = abs(losing_trades['pnl'].mean())
            
            # Win/Loss Ratio
            if result.avg_loss > 0:
                result.avg_win_loss_ratio = result.avg_win / result.avg_loss
            
            # Profit Factor
            total_wins = winning_trades['pnl'].sum()
            total_losses = abs(losing_trades['pnl'].sum())
            if total_losses > 0:
                result.profit_factor = total_wins / total_losses
            
            # Expectancy
            if result.total_trades > 0:
                result.expectancy = df['pnl'].mean()
        
        # 거래 빈도
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_span = (df['timestamp'].max() - df['timestamp'].min()).days
            if time_span > 0:
                result.avg_trades_per_day = result.total_trades / time_span
        
        # 평균 보유 시간
        if 'holding_time' in df.columns:
            avg_holding_seconds = df['holding_time'].mean()
            result.avg_holding_time_hours = avg_holding_seconds / 3600
    
    def _calculate_risk_metrics(self, df: pd.DataFrame, result: MetricResult):
        """리스크 메트릭 계산"""
        if 'pnl' not in df.columns or len(df) < 5:
            return
        
        returns = df['pnl'].values
        
        # VaR (95% 신뢰수준)
        result.var_95 = np.percentile(returns, 5)
        
        # CVaR (Conditional VaR)
        var_threshold = result.var_95
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) > 0:
            result.cvar_95 = tail_losses.mean()
        
        # Downside Deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            result.downside_deviation = negative_returns.std()
        
        # Ulcer Index (간단한 버전)
        if 'equity' in df.columns:
            rolling_max = df['equity'].expanding().max()
            drawdown_squared = ((df['equity'] - rolling_max) / rolling_max) ** 2
            result.ulcer_index = np.sqrt(drawdown_squared.mean())
    
    def _calculate_consistency_metrics(self, df: pd.DataFrame, result: MetricResult):
        """일관성 메트릭 계산"""
        if 'pnl' not in df.columns:
            return
        
        # 연속 승/패
        pnl_signs = (df['pnl'] > 0).astype(int)
        
        # 최대 연속 승리
        result.max_consecutive_wins = self._max_consecutive(pnl_signs, 1)
        
        # 최대 연속 패배
        result.max_consecutive_losses = self._max_consecutive(pnl_signs, 0)
        
        # 일관성 점수 (수익 분포의 안정성)
        if result.total_trades > 5:
            # 표준편차가 작을수록 일관성 높음
            pnl_std = df['pnl'].std()
            pnl_mean = abs(df['pnl'].mean())
            if pnl_mean > 0:
                cv = pnl_std / pnl_mean  # 변동계수
                result.consistency_score = max(0, min(1, 1 - cv / 2))
        
        # Recovery Factor
        if result.max_drawdown > 0 and 'cumulative_pnl' in df.columns:
            final_pnl = df['cumulative_pnl'].iloc[-1]
            if final_pnl > 0:
                result.recovery_factor = final_pnl / (result.max_drawdown * df['equity'].iloc[0])
    
    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """최대 연속 횟수 계산"""
        groups = (series != value).cumsum()
        consecutive = series[series == value].groupby(groups).size()
        return consecutive.max() if len(consecutive) > 0 else 0
    
    def _calculate_mab_score(self, result: MetricResult):
        """MAB 최적화를 위한 통합 보상 점수 계산"""
        # 각 요소별 가중치
        weights = {
            'roi': 0.20,
            'sharpe': 0.15,
            'win_rate': 0.15,
            'profit_factor': 0.10,
            'consistency': 0.15,
            'drawdown': 0.15,
            'expectancy': 0.10
        }
        
        # 정규화된 점수 계산
        scores = {}
        
        # ROI (연 20% 기준)
        scores['roi'] = min(1.0, max(0.0, (result.roi + 0.1) / 0.3))
        
        # Sharpe (2.0 기준)
        scores['sharpe'] = min(1.0, max(0.0, (result.sharpe_ratio + 1.0) / 3.0))
        
        # Win Rate (60% 기준)
        scores['win_rate'] = min(1.0, result.win_rate / 0.6)
        
        # Profit Factor (2.0 기준)
        scores['profit_factor'] = min(1.0, result.profit_factor / 2.0) if result.profit_factor > 0 else 0
        
        # Consistency
        scores['consistency'] = result.consistency_score
        
        # Drawdown (10% 기준, 역수)
        scores['drawdown'] = max(0.0, 1.0 - result.max_drawdown / 0.1)
        
        # Expectancy (정규화)
        if result.expectancy > 0:
            scores['expectancy'] = min(1.0, result.expectancy / 1000)  # 1000원 기준
        else:
            scores['expectancy'] = 0.5 + min(0.5, result.expectancy / 1000)
        
        # 가중 평균
        result.mab_reward_score = sum(scores[k] * weights[k] for k in weights)
        
        # Risk-Adjusted Return
        if result.downside_deviation > 0:
            result.risk_adjusted_return = result.roi / result.downside_deviation
        else:
            result.risk_adjusted_return = result.roi
    
    def _calculate_composite_score(self, result: MetricResult):
        """종합 점수 계산"""
        # MAB 점수를 기반으로 추가 조정
        base_score = result.mab_reward_score
        
        # 추가 조정 요소
        adjustments = 0.0
        
        # 거래 빈도 조정 (하루 1~5회가 이상적)
        if 1 <= result.avg_trades_per_day <= 5:
            adjustments += 0.05
        elif result.avg_trades_per_day > 10:
            adjustments -= 0.05
        
        # 보유 시간 조정 (2~8시간이 이상적)
        if 2 <= result.avg_holding_time_hours <= 8:
            adjustments += 0.05
        elif result.avg_holding_time_hours > 24:
            adjustments -= 0.05
        
        # Recovery Factor 보너스
        if result.recovery_factor > 3:
            adjustments += 0.05
        
        # 최종 점수
        result.composite_score = max(0.0, min(1.0, base_score + adjustments))
    
    def calculate_incremental_metrics(self,
                                    previous_metrics: MetricResult,
                                    new_trade: TradeRecord,
                                    all_trades: List[Dict[str, Any]]) -> MetricResult:
        """
        증분 메트릭 계산 (효율성을 위해)
        
        Args:
            previous_metrics: 이전 메트릭
            new_trade: 새로운 거래
            all_trades: 전체 거래 기록
            
        Returns:
            업데이트된 MetricResult
        """
        # 간단한 구현 - 실제로는 더 효율적으로 구현 가능
        all_trades.append(new_trade.to_dict())
        return self.calculate_metrics(all_trades)
    
    def get_mab_reward(self, metrics: MetricResult) -> float:
        """MAB 시스템을 위한 보상 값 반환"""
        return metrics.mab_reward_score


# 전역 인스턴스
_metrics_calculator = StandardizedMetrics()


def get_metrics_calculator() -> StandardizedMetrics:
    """전역 메트릭 계산기 반환"""
    return _metrics_calculator


def calculate_strategy_metrics(trades: List[Dict[str, Any]],
                             initial_capital: float = 1000000) -> Dict[str, float]:
    """
    전략 메트릭 계산 헬퍼 함수
    
    Returns:
        메트릭 딕셔너리
    """
    calculator = get_metrics_calculator()
    result = calculator.calculate_metrics(trades, initial_capital)
    return result.to_dict()