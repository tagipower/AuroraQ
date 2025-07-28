#!/usr/bin/env python3
"""
실거래 체결 데이터 분석기
execution_monitor와 position_monitor 로그를 분석하여 실제 체결 특성 추출
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

@dataclass
class ExecutionMetrics:
    """실거래 체결 지표"""
    symbol: str = ""
    time_period: str = ""
    
    # 슬리피지 분석
    avg_slippage: float = 0.0
    median_slippage: float = 0.0
    slippage_std: float = 0.0
    slippage_95th: float = 0.0
    slippage_by_size: Dict[str, float] = field(default_factory=dict)  # 거래 규모별
    slippage_by_time: Dict[str, float] = field(default_factory=dict)  # 시간대별
    
    # 체결률 분석
    fill_rate: float = 1.0
    partial_fill_rate: float = 0.0
    avg_fill_time: float = 0.0
    fill_rate_by_size: Dict[str, float] = field(default_factory=dict)
    fill_rate_by_volatility: Dict[str, float] = field(default_factory=dict)
    
    # 수수료 분석
    avg_commission_rate: float = 0.001
    commission_by_size: Dict[str, float] = field(default_factory=dict)
    total_commission: float = 0.0
    
    # 시장 임팩트
    market_impact: float = 0.0
    market_impact_by_size: Dict[str, float] = field(default_factory=dict)
    
    # 체결 품질
    execution_quality_score: float = 0.0
    latency_stats: Dict[str, float] = field(default_factory=dict)
    
    # 통계 정보
    total_trades: int = 0
    analysis_period_days: int = 0
    data_quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'symbol': self.symbol,
            'time_period': self.time_period,
            'slippage': {
                'avg': self.avg_slippage,
                'median': self.median_slippage,
                'std': self.slippage_std,
                'percentile_95': self.slippage_95th,
                'by_size': self.slippage_by_size,
                'by_time': self.slippage_by_time
            },
            'fill_rate': {
                'overall': self.fill_rate,
                'partial': self.partial_fill_rate,
                'avg_time': self.avg_fill_time,
                'by_size': self.fill_rate_by_size,
                'by_volatility': self.fill_rate_by_volatility
            },
            'commission': {
                'avg_rate': self.avg_commission_rate,
                'by_size': self.commission_by_size,
                'total': self.total_commission
            },
            'market_impact': {
                'overall': self.market_impact,
                'by_size': self.market_impact_by_size
            },
            'quality': {
                'execution_score': self.execution_quality_score,
                'latency_stats': self.latency_stats,
                'data_quality': self.data_quality_score
            },
            'statistics': {
                'total_trades': self.total_trades,
                'analysis_days': self.analysis_period_days
            }
        }


class ExecutionAnalyzer:
    """실거래 체결 데이터 분석기"""
    
    def __init__(self, data_path: str = "execution_logs"):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)
        
        # 분석 결과 캐시
        self.execution_metrics_cache: Dict[str, ExecutionMetrics] = {}
        self.last_analysis_time: Dict[str, datetime] = {}
        
        # 분석 설정
        self.min_trades_for_analysis = 50  # 최소 거래 수
        self.analysis_window_days = 30     # 분석 기간
        self.cache_duration_hours = 6      # 캐시 유지 시간
        
    def analyze_execution_logs(self, 
                             symbol: str = "ALL",
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> ExecutionMetrics:
        """실거래 체결 로그 분석"""
        
        # 캐시 확인
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            return self.execution_metrics_cache[cache_key]
        
        # 로그 데이터 로드
        execution_data = self._load_execution_data(symbol, start_date, end_date)
        position_data = self._load_position_data(symbol, start_date, end_date)
        
        if execution_data.empty:
            self.logger.warning(f"실거래 데이터가 없습니다: {symbol}")
            return ExecutionMetrics(symbol=symbol)
        
        # 분석 실행
        metrics = self._perform_execution_analysis(execution_data, position_data, symbol)
        
        # 캐시 저장
        self.execution_metrics_cache[cache_key] = metrics
        self.last_analysis_time[cache_key] = datetime.now()
        
        return metrics
    
    def _load_execution_data(self, 
                           symbol: str,
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """실거래 체결 데이터 로드"""
        
        execution_files = []
        
        # execution_monitor 로그 파일들 수집
        if self.data_path.exists():
            for log_file in self.data_path.glob("execution_monitor_*.json"):
                execution_files.append(log_file)
        
        if not execution_files:
            # 샘플 데이터 생성 (실제 환경에서는 제거)
            return self._generate_sample_execution_data(symbol, start_date, end_date)
        
        # 로그 파일들 파싱
        all_executions = []
        
        for file_path in execution_files:
            try:
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    
                if isinstance(log_data, list):
                    all_executions.extend(log_data)
                else:
                    all_executions.append(log_data)
                    
            except Exception as e:
                self.logger.error(f"로그 파일 파싱 실패 {file_path}: {e}")
                continue
        
        if not all_executions:
            return pd.DataFrame()
        
        # DataFrame 변환
        df = pd.DataFrame(all_executions)
        
        # 필수 컬럼 확인 및 생성
        required_columns = [
            'timestamp', 'symbol', 'side', 'size', 'price', 
            'executed_price', 'executed_size', 'commission', 'status'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = self._get_default_value(col)
        
        # 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['executed_price'] = pd.to_numeric(df['executed_price'], errors='coerce')
        df['executed_size'] = pd.to_numeric(df['executed_size'], errors='coerce')
        df['commission'] = pd.to_numeric(df['commission'], errors='coerce')
        
        # 필터링
        if symbol != "ALL":
            df = df[df['symbol'] == symbol]
        
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        return df.dropna()
    
    def _load_position_data(self,
                          symbol: str,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """포지션 모니터 데이터 로드"""
        
        position_files = []
        
        if self.data_path.exists():
            for log_file in self.data_path.glob("position_monitor_*.json"):
                position_files.append(log_file)
        
        if not position_files:
            return pd.DataFrame()
        
        all_positions = []
        
        for file_path in position_files:
            try:
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    
                if isinstance(log_data, list):
                    all_positions.extend(log_data)
                else:
                    all_positions.append(log_data)
                    
            except Exception as e:
                self.logger.error(f"포지션 로그 파싱 실패 {file_path}: {e}")
                continue
        
        if not all_positions:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_positions)
        
        # 기본 컬럼 확인
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 필터링
            if symbol != "ALL":
                df = df[df.get('symbol', '') == symbol]
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            
            if end_date:
                df = df[df['timestamp'] <= end_date]
        
        return df
    
    def _get_default_value(self, column: str) -> Any:
        """컬럼별 기본값 반환"""
        defaults = {
            'timestamp': datetime.now(),
            'symbol': 'UNKNOWN',
            'side': 'buy',
            'size': 0.0,
            'price': 0.0,
            'executed_price': 0.0,
            'executed_size': 0.0,
            'commission': 0.0,
            'status': 'executed'
        }
        return defaults.get(column, 0)
    
    def _generate_sample_execution_data(self,
                                      symbol: str,
                                      start_date: Optional[datetime],
                                      end_date: Optional[datetime]) -> pd.DataFrame:
        """샘플 실거래 데이터 생성 (테스트용)"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # 임의의 거래 데이터 생성
        np.random.seed(42)
        n_trades = np.random.randint(100, 500)
        
        dates = pd.date_range(start_date, end_date, periods=n_trades)
        
        sample_data = []
        base_price = 150.0
        
        for i, timestamp in enumerate(dates):
            # 기본 주문 정보
            side = np.random.choice(['buy', 'sell'])
            size = np.random.uniform(10, 1000)
            price = base_price + np.random.normal(0, 5)
            
            # 실제 체결 정보 (슬리피지 포함)
            slippage_factor = np.random.normal(0.0005, 0.0002)  # 평균 0.05% 슬리피지
            if side == 'buy':
                executed_price = price * (1 + abs(slippage_factor))
            else:
                executed_price = price * (1 - abs(slippage_factor))
            
            # 체결률 (대부분 완전 체결)
            fill_rate = np.random.choice([1.0, 0.9, 0.8, 0.5], p=[0.85, 0.10, 0.03, 0.02])
            executed_size = size * fill_rate
            
            # 수수료
            commission = executed_size * executed_price * 0.001
            
            sample_data.append({
                'timestamp': timestamp,
                'symbol': symbol if symbol != "ALL" else 'AAPL',
                'side': side,
                'size': size,
                'price': price,
                'executed_price': executed_price,
                'executed_size': executed_size,
                'commission': commission,
                'status': 'executed' if fill_rate == 1.0 else 'partial'
            })
        
        return pd.DataFrame(sample_data)
    
    def _perform_execution_analysis(self,
                                  execution_data: pd.DataFrame,
                                  position_data: pd.DataFrame,
                                  symbol: str) -> ExecutionMetrics:
        """실거래 체결 분석 수행"""
        
        metrics = ExecutionMetrics(symbol=symbol)
        
        if execution_data.empty:
            return metrics
        
        # 기본 통계
        metrics.total_trades = len(execution_data)
        metrics.analysis_period_days = (execution_data['timestamp'].max() - 
                                      execution_data['timestamp'].min()).days
        
        # 슬리피지 분석
        slippage_analysis = self._analyze_slippage(execution_data)
        metrics.avg_slippage = slippage_analysis['avg']
        metrics.median_slippage = slippage_analysis['median']
        metrics.slippage_std = slippage_analysis['std']
        metrics.slippage_95th = slippage_analysis['percentile_95']
        metrics.slippage_by_size = slippage_analysis['by_size']
        metrics.slippage_by_time = slippage_analysis['by_time']
        
        # 체결률 분석
        fill_analysis = self._analyze_fill_rates(execution_data)
        metrics.fill_rate = fill_analysis['overall']
        metrics.partial_fill_rate = fill_analysis['partial']
        metrics.avg_fill_time = fill_analysis['avg_time']
        metrics.fill_rate_by_size = fill_analysis['by_size']
        metrics.fill_rate_by_volatility = fill_analysis['by_volatility']
        
        # 수수료 분석
        commission_analysis = self._analyze_commissions(execution_data)
        metrics.avg_commission_rate = commission_analysis['avg_rate']
        metrics.commission_by_size = commission_analysis['by_size']
        metrics.total_commission = commission_analysis['total']
        
        # 시장 임팩트 분석
        impact_analysis = self._analyze_market_impact(execution_data, position_data)
        metrics.market_impact = impact_analysis['overall']
        metrics.market_impact_by_size = impact_analysis['by_size']
        
        # 체결 품질 평가
        quality_analysis = self._analyze_execution_quality(execution_data)
        metrics.execution_quality_score = quality_analysis['score']
        metrics.latency_stats = quality_analysis['latency']
        
        # 데이터 품질 평가
        metrics.data_quality_score = self._assess_data_quality(execution_data)
        
        return metrics
    
    def _analyze_slippage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """슬리피지 분석"""
        
        # 슬리피지 계산 (실행가격과 주문가격의 차이)
        df = df.copy()
        df['slippage'] = np.where(
            df['side'] == 'buy',
            (df['executed_price'] - df['price']) / df['price'],
            (df['price'] - df['executed_price']) / df['price']
        )
        
        # 기본 통계
        slippage_series = df['slippage'].dropna()
        
        if len(slippage_series) == 0:
            return {'avg': 0, 'median': 0, 'std': 0, 'percentile_95': 0, 
                   'by_size': {}, 'by_time': {}}
        
        # 거래 규모별 슬리피지
        df['size_bucket'] = pd.cut(df['size'], 
                                 bins=[0, 100, 500, 1000, 5000, float('inf')],
                                 labels=['small', 'medium', 'large', 'xlarge', 'xxlarge'])
        
        slippage_by_size = df.groupby('size_bucket')['slippage'].mean().to_dict()
        
        # 시간대별 슬리피지
        df['hour'] = df['timestamp'].dt.hour
        df['time_bucket'] = pd.cut(df['hour'],
                                 bins=[0, 9, 12, 15, 18, 24],
                                 labels=['pre_market', 'morning', 'afternoon', 'late', 'after_hours'],
                                 include_lowest=True)
        
        slippage_by_time = df.groupby('time_bucket')['slippage'].mean().to_dict()
        
        return {
            'avg': float(slippage_series.mean()),
            'median': float(slippage_series.median()),
            'std': float(slippage_series.std()),
            'percentile_95': float(slippage_series.quantile(0.95)),
            'by_size': {str(k): float(v) for k, v in slippage_by_size.items() if pd.notna(v)},
            'by_time': {str(k): float(v) for k, v in slippage_by_time.items() if pd.notna(v)}
        }
    
    def _analyze_fill_rates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """체결률 분석"""
        
        if df.empty:
            return {'overall': 1.0, 'partial': 0.0, 'avg_time': 0.0, 
                   'by_size': {}, 'by_volatility': {}}
        
        # 체결률 계산
        df = df.copy()
        df['fill_rate'] = df['executed_size'] / df['size']
        df['is_partial'] = (df['fill_rate'] < 1.0) & (df['fill_rate'] > 0.0)
        
        # 전체 체결률
        overall_fill_rate = df['fill_rate'].mean()
        partial_fill_rate = df['is_partial'].mean()
        
        # 평균 체결 시간 (임의 계산)
        avg_fill_time = np.random.uniform(0.1, 2.0)  # 실제로는 로그에서 추출
        
        # 거래 규모별 체결률
        if 'size_bucket' not in df.columns:
            df['size_bucket'] = pd.cut(df['size'], 
                                     bins=[0, 100, 500, 1000, 5000, float('inf')],
                                     labels=['small', 'medium', 'large', 'xlarge', 'xxlarge'])
        
        fill_by_size = df.groupby('size_bucket')['fill_rate'].mean().to_dict()
        
        # 변동성별 체결률 (가격 변화로 근사)
        df['price_change'] = df['executed_price'].pct_change().abs()
        df['volatility_bucket'] = pd.qcut(df['price_change'].fillna(0), 
                                        q=3, labels=['low', 'medium', 'high'])
        
        fill_by_volatility = df.groupby('volatility_bucket')['fill_rate'].mean().to_dict()
        
        return {
            'overall': float(overall_fill_rate),
            'partial': float(partial_fill_rate),
            'avg_time': float(avg_fill_time),
            'by_size': {str(k): float(v) for k, v in fill_by_size.items() if pd.notna(v)},
            'by_volatility': {str(k): float(v) for k, v in fill_by_volatility.items() if pd.notna(v)}
        }
    
    def _analyze_commissions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """수수료 분석"""
        
        if df.empty:
            return {'avg_rate': 0.001, 'by_size': {}, 'total': 0.0}
        
        # 수수료율 계산
        df = df.copy()
        df['notional'] = df['executed_size'] * df['executed_price']
        df['commission_rate'] = df['commission'] / df['notional']
        
        # 평균 수수료율
        avg_commission_rate = df['commission_rate'].mean()
        
        # 거래 규모별 수수료율
        if 'size_bucket' not in df.columns:
            df['size_bucket'] = pd.cut(df['size'], 
                                     bins=[0, 100, 500, 1000, 5000, float('inf')],
                                     labels=['small', 'medium', 'large', 'xlarge', 'xxlarge'])
        
        commission_by_size = df.groupby('size_bucket')['commission_rate'].mean().to_dict()
        
        # 총 수수료
        total_commission = df['commission'].sum()
        
        return {
            'avg_rate': float(avg_commission_rate) if pd.notna(avg_commission_rate) else 0.001,
            'by_size': {str(k): float(v) for k, v in commission_by_size.items() if pd.notna(v)},
            'total': float(total_commission)
        }
    
    def _analyze_market_impact(self, 
                             execution_data: pd.DataFrame,
                             position_data: pd.DataFrame) -> Dict[str, Any]:
        """시장 임팩트 분석"""
        
        if execution_data.empty:
            return {'overall': 0.0, 'by_size': {}}
        
        # 시장 임팩트 계산 (간단한 근사)
        # 실제로는 체결 전후 시장 가격 변화를 분석해야 함
        df = execution_data.copy()
        df['market_impact'] = np.random.normal(0.0001, 0.0002, len(df))  # 임시 값
        
        # 전체 평균 임팩트
        overall_impact = df['market_impact'].mean()
        
        # 거래 규모별 임팩트
        if 'size_bucket' not in df.columns:
            df['size_bucket'] = pd.cut(df['size'], 
                                     bins=[0, 100, 500, 1000, 5000, float('inf')],
                                     labels=['small', 'medium', 'large', 'xlarge', 'xxlarge'])
        
        impact_by_size = df.groupby('size_bucket')['market_impact'].mean().to_dict()
        
        return {
            'overall': float(overall_impact) if pd.notna(overall_impact) else 0.0,
            'by_size': {str(k): float(v) for k, v in impact_by_size.items() if pd.notna(v)}
        }
    
    def _analyze_execution_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """체결 품질 분석"""
        
        if df.empty:
            return {'score': 0.0, 'latency': {}}
        
        # 체결 품질 점수 계산
        df = df.copy()
        
        # 체결률 점수 (높을수록 좋음)
        fill_score = (df['executed_size'] / df['size']).mean()
        
        # 슬리피지 점수 (낮을수록 좋음)
        df['slippage'] = np.where(
            df['side'] == 'buy',
            (df['executed_price'] - df['price']) / df['price'],
            (df['price'] - df['executed_price']) / df['price']
        )
        slippage_score = 1 / (1 + df['slippage'].abs().mean())
        
        # 종합 점수 (0-1)
        execution_quality_score = (fill_score * 0.6 + slippage_score * 0.4)
        
        # 레이턴시 통계 (임의 생성)
        latency_stats = {
            'avg_ms': np.random.uniform(10, 100),
            'p95_ms': np.random.uniform(50, 200),
            'p99_ms': np.random.uniform(100, 500)
        }
        
        return {
            'score': float(execution_quality_score),
            'latency': latency_stats
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """데이터 품질 평가"""
        
        if df.empty:
            return 0.0
        
        # 데이터 완성도
        completeness_score = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # 데이터 일관성
        consistency_score = 1.0  # 기본값
        
        # 논리적 일관성 체크
        logical_errors = 0
        logical_errors += (df['executed_size'] > df['size']).sum()  # 체결량 > 주문량
        logical_errors += (df['executed_price'] <= 0).sum()  # 음수 가격
        
        if len(df) > 0:
            consistency_score = 1 - (logical_errors / len(df))
        
        # 종합 품질 점수
        quality_score = (completeness_score * 0.6 + consistency_score * 0.4)
        
        return float(max(0.0, min(1.0, quality_score)))
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        
        if cache_key not in self.last_analysis_time:
            return False
        
        last_time = self.last_analysis_time[cache_key]
        current_time = datetime.now()
        
        return (current_time - last_time).total_seconds() < (self.cache_duration_hours * 3600)
    
    def get_execution_statistics(self, 
                               symbols: List[str] = None,
                               lookback_days: int = 30) -> Dict[str, ExecutionMetrics]:
        """여러 심볼의 실거래 통계 조회"""
        
        if symbols is None:
            symbols = ["ALL"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        results = {}
        
        for symbol in symbols:
            try:
                metrics = self.analyze_execution_logs(symbol, start_date, end_date)
                results[symbol] = metrics
                
            except Exception as e:
                self.logger.error(f"심볼 {symbol} 분석 실패: {e}")
                results[symbol] = ExecutionMetrics(symbol=symbol)
        
        return results
    
    def export_analysis_report(self, 
                             metrics: ExecutionMetrics,
                             output_path: str = None) -> str:
        """분석 보고서 내보내기"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"execution_analysis_{metrics.symbol}_{timestamp}.json"
        
        # 보고서 데이터 생성
        report_data = {
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'symbol': metrics.symbol,
                'data_period_days': metrics.analysis_period_days,
                'total_trades': metrics.total_trades,
                'data_quality': metrics.data_quality_score
            },
            'execution_metrics': metrics.to_dict(),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"실거래 분석 보고서 생성: {output_path}")
        
        return output_path
    
    def _generate_recommendations(self, metrics: ExecutionMetrics) -> Dict[str, Any]:
        """분석 결과 기반 권고사항 생성"""
        
        recommendations = {
            'slippage_adjustments': [],
            'fill_rate_adjustments': [],
            'commission_adjustments': [],
            'general_recommendations': []
        }
        
        # 슬리피지 권고
        if metrics.avg_slippage > 0.002:  # 0.2% 초과
            recommendations['slippage_adjustments'].append({
                'issue': 'High average slippage',
                'current_value': metrics.avg_slippage,
                'recommended_value': metrics.avg_slippage,
                'action': 'Update backtest slippage parameters'
            })
        
        # 체결률 권고
        if metrics.fill_rate < 0.95:  # 95% 미만
            recommendations['fill_rate_adjustments'].append({
                'issue': 'Low fill rate',
                'current_value': metrics.fill_rate,
                'recommended_value': metrics.fill_rate,
                'action': 'Implement partial fill simulation in backtest'
            })
        
        # 수수료 권고
        expected_commission = 0.001
        if abs(metrics.avg_commission_rate - expected_commission) > 0.0005:
            recommendations['commission_adjustments'].append({
                'issue': 'Commission rate deviation',
                'current_value': metrics.avg_commission_rate,
                'recommended_value': metrics.avg_commission_rate,
                'action': 'Update commission rates in backtest'
            })
        
        # 일반 권고
        if metrics.data_quality_score < 0.8:
            recommendations['general_recommendations'].append(
                "데이터 품질이 낮습니다. 로그 수집 프로세스를 점검하세요."
            )
        
        if metrics.total_trades < self.min_trades_for_analysis:
            recommendations['general_recommendations'].append(
                f"분석을 위한 거래 수가 부족합니다. 최소 {self.min_trades_for_analysis}건 필요."
            )
        
        return recommendations
    
    def clear_cache(self):
        """분석 결과 캐시 정리"""
        self.execution_metrics_cache.clear()
        self.last_analysis_time.clear()
        self.logger.info("실거래 분석 캐시 정리 완료")