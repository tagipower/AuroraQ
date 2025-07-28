#!/usr/bin/env python3
"""
리스크 관리 데이터 모델
VaR, CVaR, MDD 등 고급 리스크 지표 정의
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """알림 타입"""
    VAR_BREACH = "var_breach"
    CVAR_BREACH = "cvar_breach"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    VOLATILITY_SPIKE = "volatility_spike"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    
    # VaR 설정
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_lookback_period: int = 252  # 1년
    var_limit_pct: float = 0.05  # 일일 5% VaR 한도
    
    # CVaR 설정
    cvar_confidence_level: float = 0.95
    cvar_limit_pct: float = 0.08  # 일일 8% CVaR 한도
    
    # 낙폭 관리
    max_drawdown_limit: float = 0.15  # 15% 최대 낙폭
    drawdown_alert_threshold: float = 0.10  # 10% 낙폭 경고
    drawdown_position_reduction: float = 0.5  # 낙폭 시 50% 포지션 축소
    
    # 포지션 집중도 관리
    max_single_position_pct: float = 0.20  # 단일 포지션 20% 한도
    max_sector_concentration: float = 0.40  # 섹터별 40% 한도
    
    # 상관관계 관리
    max_correlation_threshold: float = 0.7  # 포지션 간 최대 70% 상관관계
    correlation_lookback_period: int = 60  # 상관관계 계산 기간
    
    # 변동성 관리
    volatility_threshold_multiplier: float = 2.0  # 평균 변동성의 2배
    volatility_lookback_period: int = 30  # 변동성 계산 기간
    
    # 유동성 관리
    min_liquidity_ratio: float = 0.1  # 최소 10% 현금 비율
    liquidity_buffer_pct: float = 0.05  # 5% 유동성 버퍼
    
    # 스트레스 테스트
    stress_test_scenarios: List[float] = field(default_factory=lambda: [-0.1, -0.2, -0.3])  # 10%, 20%, 30% 하락
    
    # 리밸런싱
    rebalancing_threshold: float = 0.05  # 5% 편차 시 리밸런싱
    auto_rebalancing_enabled: bool = True


@dataclass
class RiskMetrics:
    """리스크 지표"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # VaR 지표
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    var_95_pct: float = 0.0  # 자본 대비 95% VaR 비율
    var_99_pct: float = 0.0  # 자본 대비 99% VaR 비율
    
    # CVaR 지표
    cvar_95: float = 0.0  # 95% CVaR
    cvar_99: float = 0.0  # 99% CVaR
    cvar_95_pct: float = 0.0  # 자본 대비 95% CVaR 비율
    
    # 낙폭 지표
    current_drawdown: float = 0.0  # 현재 낙폭
    max_drawdown: float = 0.0  # 최대 낙폭
    drawdown_duration: int = 0  # 낙폭 지속 일수
    underwater_curve: List[float] = field(default_factory=list)  # 낙폭 곡선
    
    # 변동성 지표
    realized_volatility: float = 0.0  # 실현 변동성 (연환산)
    rolling_volatility: float = 0.0  # 롤링 변동성
    volatility_ratio: float = 0.0  # 현재/평균 변동성 비율
    
    # 집중도 지표
    position_concentration: Dict[str, float] = field(default_factory=dict)  # 포지션별 집중도
    sector_concentration: Dict[str, float] = field(default_factory=dict)  # 섹터별 집중도
    herfindahl_index: float = 0.0  # 허핀달 지수 (집중도 측정)
    
    # 상관관계 지표
    avg_correlation: float = 0.0  # 평균 상관관계
    max_correlation: float = 0.0  # 최대 상관관계
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # 유동성 지표
    cash_ratio: float = 0.0  # 현금 비율
    liquidity_score: float = 0.0  # 유동성 점수
    
    # 레버리지 지표
    gross_leverage: float = 0.0  # 총 레버리지
    net_leverage: float = 0.0  # 순 레버리지
    
    # 스트레스 테스트 결과
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    # 종합 리스크 점수
    overall_risk_score: float = 0.0  # 0-100 점수
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'var_95': self.var_95,
            'var_99': self.var_99,
            'var_95_pct': self.var_95_pct,
            'var_99_pct': self.var_99_pct,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'cvar_95_pct': self.cvar_95_pct,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'realized_volatility': self.realized_volatility,
            'rolling_volatility': self.rolling_volatility,
            'volatility_ratio': self.volatility_ratio,
            'position_concentration': self.position_concentration,
            'sector_concentration': self.sector_concentration,
            'herfindahl_index': self.herfindahl_index,
            'avg_correlation': self.avg_correlation,
            'max_correlation': self.max_correlation,
            'cash_ratio': self.cash_ratio,
            'liquidity_score': self.liquidity_score,
            'gross_leverage': self.gross_leverage,
            'net_leverage': self.net_leverage,
            'stress_test_results': self.stress_test_results,
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level.value
        }


@dataclass
class RiskAlert:
    """리스크 알림"""
    alert_id: str = field(default_factory=lambda: f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    timestamp: datetime = field(default_factory=datetime.now)
    alert_type: AlertType = AlertType.VAR_BREACH
    risk_level: RiskLevel = RiskLevel.MEDIUM
    title: str = ""
    description: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    affected_positions: List[str] = field(default_factory=list)
    is_resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'title': self.title,
            'description': self.description,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'recommended_actions': self.recommended_actions,
            'affected_positions': self.affected_positions,
            'is_resolved': self.is_resolved,
            'resolved_timestamp': self.resolved_timestamp.isoformat() if self.resolved_timestamp else None
        }


@dataclass
class PortfolioSnapshot:
    """포트폴리오 스냅샷"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_equity: float = 0.0
    cash: float = 0.0
    positions: Dict[str, Any] = field(default_factory=dict)
    returns: List[float] = field(default_factory=list)
    prices: Dict[str, float] = field(default_factory=dict)
    
    def get_portfolio_returns(self, lookback_periods: int = 252) -> np.ndarray:
        """포트폴리오 수익률 반환"""
        if len(self.returns) < 2:
            return np.array([])
        
        recent_returns = self.returns[-lookback_periods:] if len(self.returns) > lookback_periods else self.returns
        return np.array(recent_returns)
    
    def get_position_weights(self) -> Dict[str, float]:
        """포지션 가중치 계산"""
        if self.total_equity <= 0:
            return {}
        
        weights = {}
        for symbol, position_data in self.positions.items():
            market_value = position_data.get('market_value', 0)
            weights[symbol] = market_value / self.total_equity
        
        return weights


class RiskBudget:
    """리스크 예산 관리"""
    
    def __init__(self, total_risk_budget: float = 0.02):  # 일일 2% 리스크 예산
        self.total_risk_budget = total_risk_budget
        self.allocated_risk: Dict[str, float] = {}
        self.remaining_budget = total_risk_budget
    
    def allocate_risk(self, strategy_id: str, risk_amount: float) -> bool:
        """리스크 할당"""
        if risk_amount > self.remaining_budget:
            return False
        
        self.allocated_risk[strategy_id] = self.allocated_risk.get(strategy_id, 0) + risk_amount
        self.remaining_budget -= risk_amount
        return True
    
    def release_risk(self, strategy_id: str, risk_amount: float):
        """리스크 해제"""
        if strategy_id in self.allocated_risk:
            released = min(risk_amount, self.allocated_risk[strategy_id])
            self.allocated_risk[strategy_id] -= released
            self.remaining_budget += released
            
            if self.allocated_risk[strategy_id] <= 0:
                del self.allocated_risk[strategy_id]
    
    def get_utilization(self) -> float:
        """리스크 예산 사용률"""
        used_budget = self.total_risk_budget - self.remaining_budget
        return used_budget / self.total_risk_budget if self.total_risk_budget > 0 else 0
    
    def reset_daily_budget(self):
        """일일 리스크 예산 초기화"""
        self.allocated_risk.clear()
        self.remaining_budget = self.total_risk_budget


class RiskScenario:
    """리스크 시나리오"""
    
    def __init__(self, name: str, description: str, probability: float):
        self.name = name
        self.description = description
        self.probability = probability  # 0-1
        self.market_shocks: Dict[str, float] = {}  # 심볼별 충격
        self.correlation_changes: Dict[Tuple[str, str], float] = {}  # 상관관계 변화
        self.volatility_multipliers: Dict[str, float] = {}  # 변동성 배수
    
    def add_market_shock(self, symbol: str, shock_pct: float):
        """시장 충격 추가"""
        self.market_shocks[symbol] = shock_pct
    
    def add_correlation_change(self, symbol1: str, symbol2: str, new_correlation: float):
        """상관관계 변화 추가"""
        self.correlation_changes[(symbol1, symbol2)] = new_correlation
    
    def add_volatility_change(self, symbol: str, multiplier: float):
        """변동성 변화 추가"""
        self.volatility_multipliers[symbol] = multiplier
    
    def apply_to_portfolio(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """포트폴리오에 시나리오 적용"""
        scenario_results = {}
        
        for symbol, position_data in portfolio_snapshot.positions.items():
            current_value = position_data.get('market_value', 0)
            
            # 시장 충격 적용
            shock = self.market_shocks.get(symbol, 0)
            shocked_value = current_value * (1 + shock)
            
            scenario_results[symbol] = shocked_value - current_value
        
        return scenario_results