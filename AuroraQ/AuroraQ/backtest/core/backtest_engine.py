#!/usr/bin/env python3
"""
백테스트 엔진 - 거래 전략 시뮬레이션의 핵심
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings

from .portfolio import Portfolio
from .trade_executor import TradeExecutor
from .market_simulator import MarketSimulator
from ..utils.performance_metrics import PerformanceAnalyzer
from ..utils.logger import get_logger
from ....SharedCore.utils.logger import get_logger as shared_logger

warnings.filterwarnings('ignore')
logger = get_logger("BacktestEngine")


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    min_order_size: float = 0.0001
    max_position_size: float = 0.95  # 자본의 95%
    enable_short: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02  # 연 2%
    

@dataclass
class BacktestResult:
    """백테스트 결과"""
    # 기본 정보
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # 수익률 지표
    total_return: float
    annualized_return: float
    
    # 리스크 지표
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 시계열 데이터
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    
    # 추가 분석
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    yearly_returns: pd.Series = field(default_factory=pd.Series)
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        """결과 요약"""
        return f"""
=== 백테스트 결과 요약 ===
기간: {self.start_date.date()} ~ {self.end_date.date()}
초기 자본: ${self.initial_capital:,.0f}
최종 자본: ${self.final_capital:,.0f}

[수익률]
총 수익률: {self.total_return:.2%}
연환산 수익률: {self.annualized_return:.2%}

[리스크]
최대 낙폭: {self.max_drawdown:.2%}
변동성: {self.volatility:.2%}
샤프 비율: {self.sharpe_ratio:.2f}
소르티노 비율: {self.sortino_ratio:.2f}

[거래 통계]
총 거래: {self.total_trades}
승률: {self.win_rate:.1%}
평균 수익: {self.avg_win:.2%}
평균 손실: {self.avg_loss:.2%}
손익비: {self.profit_factor:.2f}
========================
"""


class BacktestEngine:
    """고성능 백테스트 엔진"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.portfolio = None
        self.trade_executor = None
        self.market_simulator = None
        self.performance_analyzer = PerformanceAnalyzer()
        
    def run(self, 
            strategy: Any,
            data: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            progress_callback: Optional[callable] = None) -> BacktestResult:
        """
        백테스트 실행
        
        Args:
            strategy: 거래 전략 객체
            data: OHLCV 데이터프레임
            start_date: 시작일
            end_date: 종료일
            progress_callback: 진행률 콜백
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        logger.info(f"백테스트 시작: {strategy.__class__.__name__}")
        
        # 데이터 준비
        data = self._prepare_data(data, start_date, end_date)
        
        # 컴포넌트 초기화
        self._initialize_components(data)
        
        # 백테스트 실행
        equity_curve = []
        positions = []
        trades = []
        
        total_bars = len(data)
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # 시장 데이터 업데이트
            market_data = self.market_simulator.update(timestamp, row)
            
            # 전략 신호 생성
            signal = strategy.generate_signal(
                market_data,
                self.portfolio.get_position(),
                self.portfolio.get_equity()
            )
            
            # 주문 실행
            if signal and signal['action'] != 'hold':
                trade = self.trade_executor.execute(
                    signal,
                    market_data,
                    self.portfolio
                )
                if trade:
                    trades.append(trade)
            
            # 포트폴리오 업데이트
            self.portfolio.update(market_data['close'])
            
            # 기록
            equity_curve.append({
                'timestamp': timestamp,
                'equity': self.portfolio.get_equity(),
                'cash': self.portfolio.cash,
                'position_value': self.portfolio.get_position_value()
            })
            
            positions.append({
                'timestamp': timestamp,
                'position': self.portfolio.position,
                'avg_price': self.portfolio.avg_price
            })
            
            # 진행률 업데이트
            if progress_callback and i % 100 == 0:
                progress = (i + 1) / total_bars
                progress_callback(progress)
        
        # 결과 분석
        result = self._analyze_results(
            equity_curve,
            positions,
            trades,
            data
        )
        
        logger.info(f"백테스트 완료: 수익률 {result.total_return:.2%}")
        
        return result
    
    def run_multiple(self,
                    strategies: List[Any],
                    data: pd.DataFrame,
                    parallel: bool = True) -> Dict[str, BacktestResult]:
        """
        여러 전략 동시 백테스트
        
        Args:
            strategies: 전략 리스트
            data: OHLCV 데이터
            parallel: 병렬 처리 여부
            
        Returns:
            전략별 결과 딕셔너리
        """
        results = {}
        
        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.run, strategy, data): strategy
                    for strategy in strategies
                }
                
                for future in futures:
                    strategy = futures[future]
                    result = future.result()
                    results[strategy.__class__.__name__] = result
        else:
            for strategy in strategies:
                result = self.run(strategy, data)
                results[strategy.__class__.__name__] = result
        
        return results
    
    def optimize(self,
                strategy_class: type,
                param_grid: Dict[str, List],
                data: pd.DataFrame,
                metric: str = 'sharpe_ratio') -> Tuple[Dict, BacktestResult]:
        """
        파라미터 최적화
        
        Args:
            strategy_class: 전략 클래스
            param_grid: 파라미터 그리드
            data: OHLCV 데이터
            metric: 최적화 지표
            
        Returns:
            최적 파라미터와 결과
        """
        logger.info(f"파라미터 최적화 시작: {len(param_grid)} 조합")
        
        best_params = None
        best_result = None
        best_metric = -np.inf
        
        # 파라미터 조합 생성
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            # 전략 인스턴스 생성
            strategy = strategy_class(**params)
            
            # 백테스트 실행
            result = self.run(strategy, data)
            
            # 지표 확인
            metric_value = getattr(result, metric)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params
                best_result = result
        
        logger.info(f"최적화 완료: {metric}={best_metric:.4f}")
        
        return best_params, best_result
    
    def walk_forward_analysis(self,
                            strategy: Any,
                            data: pd.DataFrame,
                            train_period: int,
                            test_period: int,
                            step: int) -> List[BacktestResult]:
        """
        Walk-forward 분석
        
        Args:
            strategy: 거래 전략
            data: OHLCV 데이터
            train_period: 훈련 기간 (일)
            test_period: 테스트 기간 (일)
            step: 이동 간격 (일)
            
        Returns:
            기간별 백테스트 결과
        """
        results = []
        
        for i in range(0, len(data) - train_period - test_period, step):
            # 훈련 데이터
            train_data = data.iloc[i:i+train_period]
            
            # 테스트 데이터
            test_start = i + train_period
            test_end = test_start + test_period
            test_data = data.iloc[test_start:test_end]
            
            # 전략 훈련 (해당되는 경우)
            if hasattr(strategy, 'train'):
                strategy.train(train_data)
            
            # 백테스트
            result = self.run(strategy, test_data)
            results.append(result)
        
        return results
    
    def _prepare_data(self,
                     data: pd.DataFrame,
                     start_date: Optional[str],
                     end_date: Optional[str]) -> pd.DataFrame:
        """데이터 전처리"""
        # 날짜 인덱스 확인
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            else:
                raise ValueError("데이터에 timestamp 컬럼이 필요합니다")
        
        # 날짜 필터링
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        # 데이터 정제
        data = data.dropna()
        data = data.sort_index()
        
        return data
    
    def _initialize_components(self, data: pd.DataFrame):
        """컴포넌트 초기화"""
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            commission=self.config.commission
        )
        
        self.trade_executor = TradeExecutor(
            commission=self.config.commission,
            slippage=self.config.slippage,
            min_order_size=self.config.min_order_size
        )
        
        self.market_simulator = MarketSimulator(
            data=data,
            slippage=self.config.slippage
        )
    
    def _analyze_results(self,
                        equity_curve: List[Dict],
                        positions: List[Dict],
                        trades: List[Dict],
                        data: pd.DataFrame) -> BacktestResult:
        """결과 분석"""
        # DataFrame 변환
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        positions_df = pd.DataFrame(positions).set_index('timestamp')
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # 수익률 계산
        returns = equity_df['equity'].pct_change().dropna()
        
        # 성과 지표 계산
        metrics = self.performance_analyzer.calculate_metrics(
            equity_curve=equity_df['equity'],
            returns=returns,
            trades=trades_df,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # 결과 생성
        result = BacktestResult(
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.config.initial_capital,
            final_capital=equity_df['equity'].iloc[-1],
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            total_trades=len(trades_df),
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            profit_factor=metrics['profit_factor'],
            equity_curve=equity_df['equity'],
            returns=returns,
            positions=positions_df,
            trades=trades_df,
            monthly_returns=metrics.get('monthly_returns', pd.Series()),
            yearly_returns=metrics.get('yearly_returns', pd.Series()),
            rolling_sharpe=metrics.get('rolling_sharpe', pd.Series())
        )
        
        return result
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """파라미터 조합 생성"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations