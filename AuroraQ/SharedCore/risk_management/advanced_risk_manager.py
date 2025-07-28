#!/usr/bin/env python3
"""
고도화된 리스크 관리자
VaR, CVaR, MDD 기반 동적 리스크 관리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import warnings

from .var_calculator import VaRCalculator
from .risk_models import RiskConfig, RiskMetrics, RiskAlert, RiskLevel, AlertType, PortfolioSnapshot, RiskBudget
from ..position_management.unified_position_manager import UnifiedPositionManager

warnings.filterwarnings('ignore')


class AdvancedRiskManager:
    """고도화된 리스크 관리자"""
    
    def __init__(self, 
                 position_manager: Optional[UnifiedPositionManager] = None,
                 config: Optional[RiskConfig] = None):
        
        self.position_manager = position_manager
        self.config = config or RiskConfig()
        self.var_calculator = VaRCalculator(confidence_levels=self.config.var_confidence_levels)
        
        # 리스크 상태 추적
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        self.active_alerts: List[RiskAlert] = []
        self.risk_budget = RiskBudget()
        
        # 리스크 콜백 및 이벤트
        self.risk_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # 포지션 관리자와 연동
        if self.position_manager:
            self.position_manager.add_risk_callback(self._on_trade_executed)
    
    def add_risk_callback(self, callback: Callable[[RiskMetrics, List[RiskAlert]], None]):
        """리스크 콜백 추가"""
        self.risk_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[List[RiskAlert]], None]):
        """긴급 상황 콜백 추가"""
        self.emergency_callbacks.append(callback)
    
    def update_portfolio_snapshot(self, 
                                 total_equity: float,
                                 cash: float,
                                 positions: Dict[str, Any],
                                 prices: Dict[str, float]) -> PortfolioSnapshot:
        """포트폴리오 스냅샷 업데이트"""
        
        # 수익률 계산
        returns = []
        if len(self.portfolio_snapshots) > 0:
            prev_equity = self.portfolio_snapshots[-1].total_equity
            if prev_equity > 0:
                return_rate = (total_equity - prev_equity) / prev_equity
                returns = self.portfolio_snapshots[-1].returns + [return_rate]
                # 최대 1년치 수익률만 유지
                if len(returns) > 252:
                    returns = returns[-252:]
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_equity=total_equity,
            cash=cash,
            positions=positions,
            returns=returns,
            prices=prices
        )
        
        self.portfolio_snapshots.append(snapshot)
        
        # 메모리 관리 (최근 1000개만 유지)
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-500:]
        
        return snapshot
    
    def calculate_risk_metrics(self, portfolio_snapshot: Optional[PortfolioSnapshot] = None) -> RiskMetrics:
        """포괄적 리스크 지표 계산"""
        
        if portfolio_snapshot is None and self.portfolio_snapshots:
            portfolio_snapshot = self.portfolio_snapshots[-1]
        
        if portfolio_snapshot is None:
            return RiskMetrics()
        
        metrics = RiskMetrics(timestamp=datetime.now())
        
        # 1. VaR/CVaR 계산
        returns = portfolio_snapshot.get_portfolio_returns(self.config.var_lookback_period)
        if len(returns) >= 30:
            var_results = self._calculate_var_metrics(returns, portfolio_snapshot.total_equity)
            metrics.var_95 = var_results['var_95']
            metrics.var_99 = var_results['var_99']
            metrics.var_95_pct = var_results['var_95_pct']
            metrics.var_99_pct = var_results['var_99_pct']
            metrics.cvar_95 = var_results['cvar_95']
            metrics.cvar_99 = var_results['cvar_99']
            metrics.cvar_95_pct = var_results['cvar_95_pct']
        
        # 2. 낙폭 계산
        drawdown_metrics = self._calculate_drawdown_metrics()
        metrics.current_drawdown = drawdown_metrics['current_drawdown']
        metrics.max_drawdown = drawdown_metrics['max_drawdown']
        metrics.drawdown_duration = drawdown_metrics['drawdown_duration']
        metrics.underwater_curve = drawdown_metrics['underwater_curve']
        
        # 3. 변동성 계산
        volatility_metrics = self._calculate_volatility_metrics(returns)
        metrics.realized_volatility = volatility_metrics['realized_volatility']
        metrics.rolling_volatility = volatility_metrics['rolling_volatility']
        metrics.volatility_ratio = volatility_metrics['volatility_ratio']
        
        # 4. 집중도 지표
        concentration_metrics = self._calculate_concentration_metrics(portfolio_snapshot)
        metrics.position_concentration = concentration_metrics['position_concentration']
        metrics.sector_concentration = concentration_metrics['sector_concentration']
        metrics.herfindahl_index = concentration_metrics['herfindahl_index']
        
        # 5. 상관관계 분석
        if len(returns) >= self.config.correlation_lookback_period:
            correlation_metrics = self._calculate_correlation_metrics(portfolio_snapshot)
            metrics.avg_correlation = correlation_metrics['avg_correlation']
            metrics.max_correlation = correlation_metrics['max_correlation']
            metrics.correlation_matrix = correlation_metrics['correlation_matrix']
        
        # 6. 유동성 지표
        liquidity_metrics = self._calculate_liquidity_metrics(portfolio_snapshot)
        metrics.cash_ratio = liquidity_metrics['cash_ratio']
        metrics.liquidity_score = liquidity_metrics['liquidity_score']
        
        # 7. 레버리지 지표
        leverage_metrics = self._calculate_leverage_metrics(portfolio_snapshot)
        metrics.gross_leverage = leverage_metrics['gross_leverage']
        metrics.net_leverage = leverage_metrics['net_leverage']
        
        # 8. 스트레스 테스트
        stress_results = self._run_stress_tests(returns, portfolio_snapshot)
        metrics.stress_test_results = stress_results
        
        # 9. 종합 리스크 점수 계산
        overall_score = self._calculate_overall_risk_score(metrics)
        metrics.overall_risk_score = overall_score['score']
        metrics.risk_level = overall_score['level']
        
        # 리스크 지표 히스토리 저장
        self.risk_metrics_history.append(metrics)
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-500:]
        
        return metrics
    
    def _calculate_var_metrics(self, returns: np.ndarray, portfolio_value: float) -> Dict[str, float]:
        """VaR/CVaR 지표 계산"""
        results = {}
        
        for confidence_level in self.config.var_confidence_levels:
            var_result = self.var_calculator.calculate_var(
                returns, 
                method='historical',
                confidence_level=confidence_level,
                portfolio_value=portfolio_value
            )
            
            if confidence_level == 0.95:
                results.update({
                    'var_95': var_result['var'],
                    'var_95_pct': var_result['var_pct'],
                    'cvar_95': var_result['cvar'],
                    'cvar_95_pct': var_result['cvar_pct']
                })
            elif confidence_level == 0.99:
                results.update({
                    'var_99': var_result['var'],
                    'var_99_pct': var_result['var_pct'],
                    'cvar_99': var_result['cvar']
                })
        
        return results
    
    def _calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """낙폭 지표 계산"""
        if len(self.portfolio_snapshots) < 2:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'drawdown_duration': 0,
                'underwater_curve': []
            }
        
        # 자산가치 추출
        equity_values = [snapshot.total_equity for snapshot in self.portfolio_snapshots]
        equity_series = pd.Series(equity_values)
        
        # 누적 최대값 계산
        peak_series = equity_series.expanding().max()
        
        # 낙폭 계산
        drawdown_series = (equity_series - peak_series) / peak_series
        
        current_drawdown = abs(drawdown_series.iloc[-1])
        max_drawdown = abs(drawdown_series.min())
        
        # 낙폭 지속 기간 계산
        drawdown_duration = 0
        for i in range(len(drawdown_series) - 1, -1, -1):
            if drawdown_series.iloc[i] < -0.001:  # 0.1% 이상 낙폭
                drawdown_duration += 1
            else:
                break
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'underwater_curve': drawdown_series.tolist()
        }
    
    def _calculate_volatility_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """변동성 지표 계산"""
        if len(returns) < 2:
            return {
                'realized_volatility': 0.0,
                'rolling_volatility': 0.0,
                'volatility_ratio': 1.0
            }
        
        # 실현 변동성 (연환산)
        realized_vol = np.std(returns) * np.sqrt(252)
        
        # 롤링 변동성 (최근 30일)
        recent_period = min(30, len(returns))
        recent_returns = returns[-recent_period:]
        rolling_vol = np.std(recent_returns) * np.sqrt(252)
        
        # 변동성 비율 (현재/평균)
        vol_ratio = rolling_vol / realized_vol if realized_vol > 0 else 1.0
        
        return {
            'realized_volatility': realized_vol,
            'rolling_volatility': rolling_vol,
            'volatility_ratio': vol_ratio
        }
    
    def _calculate_concentration_metrics(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """집중도 지표 계산"""
        if portfolio_snapshot.total_equity <= 0:
            return {
                'position_concentration': {},
                'sector_concentration': {},
                'herfindahl_index': 0.0
            }
        
        # 포지션별 집중도
        position_weights = portfolio_snapshot.get_position_weights()
        
        # 허핀달 지수 계산
        herfindahl = sum(weight**2 for weight in position_weights.values())
        
        # 섹터별 집중도 (간단한 예시 - 실제로는 종목별 섹터 정보 필요)
        sector_concentration = {}
        for symbol, weight in position_weights.items():
            # 예시: 심볼 기반 간단한 섹터 분류
            sector = self._get_sector_from_symbol(symbol)
            sector_concentration[sector] = sector_concentration.get(sector, 0) + weight
        
        return {
            'position_concentration': position_weights,
            'sector_concentration': sector_concentration,
            'herfindahl_index': herfindahl
        }
    
    def _calculate_correlation_metrics(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """상관관계 지표 계산"""
        if len(portfolio_snapshot.positions) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_matrix': pd.DataFrame()
            }
        
        # 실제 구현에서는 각 포지션의 가격 히스토리가 필요
        # 여기서는 예시로 간단한 계산
        symbols = list(portfolio_snapshot.positions.keys())
        correlation_matrix = pd.DataFrame(
            np.random.rand(len(symbols), len(symbols)),
            index=symbols,
            columns=symbols
        )
        np.fill_diagonal(correlation_matrix.values, 1.0)
        
        # 상관관계 통계
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.stack()
        
        avg_correlation = correlations.mean() if len(correlations) > 0 else 0.0
        max_correlation = correlations.max() if len(correlations) > 0 else 0.0
        
        return {
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'correlation_matrix': correlation_matrix
        }
    
    def _calculate_liquidity_metrics(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """유동성 지표 계산"""
        if portfolio_snapshot.total_equity <= 0:
            return {'cash_ratio': 0.0, 'liquidity_score': 0.0}
        
        cash_ratio = portfolio_snapshot.cash / portfolio_snapshot.total_equity
        
        # 유동성 점수 (현금 비율 + 포지션 유동성)
        # 실제로는 각 자산의 유동성 정보가 필요
        liquidity_score = min(1.0, cash_ratio + 0.5)  # 간단한 예시
        
        return {
            'cash_ratio': cash_ratio,
            'liquidity_score': liquidity_score
        }
    
    def _calculate_leverage_metrics(self, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """레버리지 지표 계산"""
        if portfolio_snapshot.total_equity <= 0:
            return {'gross_leverage': 0.0, 'net_leverage': 0.0}
        
        total_position_value = sum(
            pos_data.get('market_value', 0) 
            for pos_data in portfolio_snapshot.positions.values()
        )
        
        gross_leverage = total_position_value / portfolio_snapshot.total_equity
        net_leverage = gross_leverage  # 간단한 예시 (실제로는 롱/숏 구분 필요)
        
        return {
            'gross_leverage': gross_leverage,
            'net_leverage': net_leverage
        }
    
    def _run_stress_tests(self, returns: np.ndarray, portfolio_snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """스트레스 테스트 실행"""
        if len(returns) < 10:
            return {}
        
        stress_results = self.var_calculator.stress_test_var(
            returns, 
            self.config.stress_test_scenarios
        )
        
        return stress_results.get('stress_results', {})
    
    def _calculate_overall_risk_score(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """종합 리스크 점수 계산"""
        score = 0.0
        weights = {
            'var': 0.25,
            'drawdown': 0.25,
            'concentration': 0.20,
            'leverage': 0.15,
            'volatility': 0.15
        }
        
        # VaR 점수 (0-25점)
        var_score = min(25, metrics.var_95_pct * 500) if metrics.var_95_pct > 0 else 0
        score += var_score * weights['var']
        
        # 낙폭 점수 (0-25점)
        drawdown_score = min(25, metrics.current_drawdown * 100)
        score += drawdown_score * weights['drawdown']
        
        # 집중도 점수 (0-20점)
        concentration_score = min(20, metrics.herfindahl_index * 100)
        score += concentration_score * weights['concentration']
        
        # 레버리지 점수 (0-15점)
        leverage_score = min(15, max(0, (metrics.gross_leverage - 1) * 10))
        score += leverage_score * weights['leverage']
        
        # 변동성 점수 (0-15점)
        volatility_score = min(15, max(0, (metrics.volatility_ratio - 1) * 30))
        score += volatility_score * weights['volatility']
        
        # 리스크 레벨 결정
        if score <= 25:
            level = RiskLevel.LOW
        elif score <= 50:
            level = RiskLevel.MEDIUM
        elif score <= 75:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return {'score': score, 'level': level}
    
    def _get_sector_from_symbol(self, symbol: str) -> str:
        """심볼에서 섹터 추정 (간단한 예시)"""
        # 실제로는 외부 데이터 소스에서 섹터 정보를 가져와야 함
        if symbol.startswith(('AAPL', 'MSFT', 'GOOGL')):
            return 'Technology'
        elif symbol.startswith(('JPM', 'BAC', 'WFC')):
            return 'Financial'
        else:
            return 'Other'
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """리스크 한도 체크 및 알림 생성"""
        alerts = []
        
        # 1. VaR 한도 체크
        if metrics.var_95_pct > self.config.var_limit_pct:
            alert = RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH,
                title="VaR 한도 초과",
                description=f"95% VaR {metrics.var_95_pct:.2%}가 한도 {self.config.var_limit_pct:.2%}를 초과했습니다.",
                current_value=metrics.var_95_pct,
                threshold_value=self.config.var_limit_pct,
                recommended_actions=[
                    "포지션 크기 축소",
                    "고위험 자산 매도",
                    "헷지 포지션 추가"
                ]
            )
            alerts.append(alert)
        
        # 2. CVaR 한도 체크
        if metrics.cvar_95_pct > self.config.cvar_limit_pct:
            alert = RiskAlert(
                alert_type=AlertType.CVAR_BREACH,
                risk_level=RiskLevel.HIGH,
                title="CVaR 한도 초과",
                description=f"95% CVaR {metrics.cvar_95_pct:.2%}가 한도 {self.config.cvar_limit_pct:.2%}를 초과했습니다.",
                current_value=metrics.cvar_95_pct,
                threshold_value=self.config.cvar_limit_pct,
                recommended_actions=[
                    "즉시 포지션 축소",
                    "스톱로스 설정",
                    "포트폴리오 재조정"
                ]
            )
            alerts.append(alert)
        
        # 3. 낙폭 한도 체크
        if metrics.current_drawdown > self.config.drawdown_alert_threshold:
            risk_level = RiskLevel.CRITICAL if metrics.current_drawdown > self.config.max_drawdown_limit else RiskLevel.HIGH
            
            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN_LIMIT,
                risk_level=risk_level,
                title="낙폭 경고",
                description=f"현재 낙폭 {metrics.current_drawdown:.2%}이 경고 수준을 초과했습니다.",
                current_value=metrics.current_drawdown,
                threshold_value=self.config.drawdown_alert_threshold,
                recommended_actions=[
                    "포지션 크기 축소",
                    "손절매 실행",
                    "리밸런싱 실행"
                ]
            )
            alerts.append(alert)
        
        # 4. 집중도 체크
        max_position_concentration = max(metrics.position_concentration.values()) if metrics.position_concentration else 0
        if max_position_concentration > self.config.max_single_position_pct:
            alert = RiskAlert(
                alert_type=AlertType.CONCENTRATION_RISK,
                risk_level=RiskLevel.MEDIUM,
                title="포지션 집중도 초과",
                description=f"단일 포지션 집중도 {max_position_concentration:.2%}가 한도를 초과했습니다.",
                current_value=max_position_concentration,
                threshold_value=self.config.max_single_position_pct,
                recommended_actions=[
                    "집중 포지션 분산",
                    "리밸런싱 실행"
                ]
            )
            alerts.append(alert)
        
        # 5. 상관관계 체크
        if metrics.max_correlation > self.config.max_correlation_threshold:
            alert = RiskAlert(
                alert_type=AlertType.CORRELATION_RISK,
                risk_level=RiskLevel.MEDIUM,
                title="높은 상관관계 위험",
                description=f"최대 상관관계 {metrics.max_correlation:.2f}이 한도를 초과했습니다.",
                current_value=metrics.max_correlation,
                threshold_value=self.config.max_correlation_threshold,
                recommended_actions=[
                    "상관관계 낮은 자산 추가",
                    "포트폴리오 다각화"
                ]
            )
            alerts.append(alert)
        
        # 6. 변동성 급등 체크
        if metrics.volatility_ratio > self.config.volatility_threshold_multiplier:
            alert = RiskAlert(
                alert_type=AlertType.VOLATILITY_SPIKE,
                risk_level=RiskLevel.MEDIUM,
                title="변동성 급등",
                description=f"변동성 비율 {metrics.volatility_ratio:.2f}이 임계값을 초과했습니다.",
                current_value=metrics.volatility_ratio,
                threshold_value=self.config.volatility_threshold_multiplier,
                recommended_actions=[
                    "포지션 크기 조정",
                    "변동성 헷지"
                ]
            )
            alerts.append(alert)
        
        # 7. 유동성 체크
        if metrics.cash_ratio < self.config.min_liquidity_ratio:
            alert = RiskAlert(
                alert_type=AlertType.LIQUIDITY_RISK,
                risk_level=RiskLevel.MEDIUM,
                title="유동성 부족",
                description=f"현금 비율 {metrics.cash_ratio:.2%}이 최소 요구사항 미달입니다.",
                current_value=metrics.cash_ratio,
                threshold_value=self.config.min_liquidity_ratio,
                recommended_actions=[
                    "일부 포지션 청산",
                    "현금 확보"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    def get_position_sizing_recommendation(self, 
                                        symbol: str,
                                        current_price: float,
                                        signal_confidence: float = 0.5) -> Dict[str, Any]:
        """VaR 기반 동적 포지션 사이징"""
        
        if not self.portfolio_snapshots:
            return {'recommended_size': 0.0, 'reason': 'No portfolio data available'}
        
        current_snapshot = self.portfolio_snapshots[-1]
        current_metrics = self.calculate_risk_metrics(current_snapshot)
        
        # 기본 포지션 크기 계산
        available_capital = current_snapshot.total_equity * 0.95  # 95% 활용
        base_position_size = available_capital * 0.1  # 기본 10%
        
        # VaR 기반 조정
        var_adjustment = 1.0
        if current_metrics.var_95_pct > 0:
            target_var_pct = self.config.var_limit_pct * 0.8  # 한도의 80%까지 활용
            var_utilization = current_metrics.var_95_pct / target_var_pct
            var_adjustment = max(0.1, 1.0 - var_utilization)
        
        # 낙폭 기반 조정
        drawdown_adjustment = 1.0
        if current_metrics.current_drawdown > self.config.drawdown_alert_threshold:
            reduction_factor = self.config.drawdown_position_reduction
            drawdown_adjustment = 1.0 - reduction_factor
        
        # 신호 신뢰도 기반 조정
        confidence_adjustment = 0.5 + (signal_confidence * 0.5)
        
        # 변동성 기반 조정
        volatility_adjustment = 1.0
        if current_metrics.volatility_ratio > 1.5:
            volatility_adjustment = 1.0 / current_metrics.volatility_ratio
        
        # 최종 포지션 크기 계산
        final_adjustment = var_adjustment * drawdown_adjustment * confidence_adjustment * volatility_adjustment
        recommended_size = (base_position_size * final_adjustment) / current_price
        
        return {
            'recommended_size': recommended_size,
            'base_size': base_position_size / current_price,
            'adjustments': {
                'var_adjustment': var_adjustment,
                'drawdown_adjustment': drawdown_adjustment,
                'confidence_adjustment': confidence_adjustment,
                'volatility_adjustment': volatility_adjustment,
                'final_adjustment': final_adjustment
            },
            'risk_metrics': current_metrics,
            'reason': f'VaR-based dynamic sizing with {final_adjustment:.2f} adjustment factor'
        }
    
    def should_reduce_positions(self, metrics: RiskMetrics) -> Tuple[bool, str, float]:
        """포지션 축소 필요성 판단"""
        
        # 1. MDD 기반 축소
        if metrics.current_drawdown > self.config.max_drawdown_limit:
            reduction_pct = self.config.drawdown_position_reduction
            return True, f"최대 낙폭 {metrics.current_drawdown:.2%} 도달", reduction_pct
        
        # 2. VaR 한도 초과
        if metrics.var_95_pct > self.config.var_limit_pct:
            excess_ratio = metrics.var_95_pct / self.config.var_limit_pct
            reduction_pct = min(0.5, (excess_ratio - 1) * 0.3)
            return True, f"VaR 한도 초과 {excess_ratio:.2f}배", reduction_pct
        
        # 3. CVaR 한도 초과
        if metrics.cvar_95_pct > self.config.cvar_limit_pct:
            excess_ratio = metrics.cvar_95_pct / self.config.cvar_limit_pct
            reduction_pct = min(0.6, (excess_ratio - 1) * 0.4)
            return True, f"CVaR 한도 초과 {excess_ratio:.2f}배", reduction_pct
        
        # 4. 종합 리스크 점수 기반
        if metrics.overall_risk_score > 80:
            reduction_pct = (metrics.overall_risk_score - 80) / 100 * 0.3
            return True, f"종합 리스크 점수 {metrics.overall_risk_score:.1f} 과도", reduction_pct
        
        return False, "정상 범위", 0.0
    
    def execute_emergency_procedures(self, alerts: List[RiskAlert]):
        """긴급 상황 대응 절차"""
        critical_alerts = [alert for alert in alerts if alert.risk_level == RiskLevel.CRITICAL]
        
        if not critical_alerts:
            return
        
        self.logger.critical(f"EMERGENCY: {len(critical_alerts)} critical risk alerts triggered")
        
        # 긴급 포지션 축소
        if self.position_manager:
            emergency_reduction = 0.7  # 70% 축소
            
            for symbol in list(self.position_manager.positions.keys()):
                try:
                    position = self.position_manager.positions[symbol]
                    reduce_size = abs(position.size) * emergency_reduction
                    
                    self.position_manager.close_position(
                        symbol, 
                        size=reduce_size,
                        reason="Emergency risk management"
                    )
                    
                    self.logger.warning(f"Emergency reduction: {symbol} by {reduce_size}")
                    
                except Exception as e:
                    self.logger.error(f"Emergency reduction failed for {symbol}: {e}")
        
        # 긴급 콜백 실행
        for callback in self.emergency_callbacks:
            try:
                callback(critical_alerts)
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {e}")
    
    def _on_trade_executed(self, risk_data: Dict[str, Any]):
        """거래 실행 시 호출되는 리스크 콜백"""
        try:
            # 포트폴리오 스냅샷 업데이트
            if self.position_manager:
                equity = self.position_manager.get_equity()
                cash = self.position_manager.cash
                positions = {
                    symbol: pos.get_position_info() 
                    for symbol, pos in self.position_manager.positions.items()
                }
                prices = {
                    symbol: pos.state.current_price 
                    for symbol, pos in self.position_manager.positions.items()
                }
                
                snapshot = self.update_portfolio_snapshot(equity, cash, positions, prices)
                
                # 리스크 지표 계산
                metrics = self.calculate_risk_metrics(snapshot)
                
                # 리스크 한도 체크
                alerts = self.check_risk_limits(metrics)
                
                # 새로운 알림만 추가
                new_alerts = []
                for alert in alerts:
                    if not any(existing.alert_type == alert.alert_type and 
                             existing.current_value == alert.current_value 
                             for existing in self.active_alerts):
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)
                
                # 긴급 상황 체크
                critical_alerts = [alert for alert in new_alerts if alert.risk_level == RiskLevel.CRITICAL]
                if critical_alerts:
                    self.execute_emergency_procedures(critical_alerts)
                
                # 리스크 콜백 실행
                for callback in self.risk_callbacks:
                    try:
                        callback(metrics, new_alerts)
                    except Exception as e:
                        self.logger.error(f"Risk callback failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Risk management callback failed: {e}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """리스크 대시보드 데이터"""
        if not self.risk_metrics_history:
            return {}
        
        current_metrics = self.risk_metrics_history[-1]
        
        dashboard = {
            'current_metrics': current_metrics.to_dict(),
            'active_alerts': [alert.to_dict() for alert in self.active_alerts if not alert.is_resolved],
            'risk_budget_utilization': self.risk_budget.get_utilization(),
            'emergency_warnings': []
        }
        
        # 포지션 축소 권고 체크
        should_reduce, reason, reduction_pct = self.should_reduce_positions(current_metrics)
        if should_reduce:
            dashboard['position_reduction_recommendation'] = {
                'should_reduce': True,
                'reason': reason,
                'reduction_percentage': reduction_pct
            }
        
        # 최근 성과 트렌드
        if len(self.risk_metrics_history) >= 10:
            recent_metrics = self.risk_metrics_history[-10:]
            dashboard['trends'] = {
                'var_trend': [m.var_95_pct for m in recent_metrics],
                'drawdown_trend': [m.current_drawdown for m in recent_metrics],
                'risk_score_trend': [m.overall_risk_score for m in recent_metrics]
            }
        
        return dashboard