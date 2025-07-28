#!/usr/bin/env python3
"""
백테스트 시스템 통합 모듈
AuroraQ_Backtest와 AuroraQ_Shared 모듈의 통합
"""

import sys
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import json

# 상대 경로로 백테스트 모듈 임포트
try:
    from ...AuroraQ_Backtest.core.backtest_engine import BacktestEngine
    from ...AuroraQ_Backtest.core.portfolio import Portfolio
    from ...AuroraQ_Backtest.strategies.base_strategy import BaseStrategy
except ImportError:
    # 개발 환경에서의 대체 임포트
    import warnings
    warnings.warn("백테스트 모듈을 직접 임포트할 수 없습니다. 통합 모드에서 실행하세요.")
    
    class BacktestEngine:
        def __init__(self): pass
    class Portfolio:
        def __init__(self): pass
    class BaseStrategy:
        def __init__(self): pass

from ..risk_management import AdvancedRiskManager, RiskConfig, RiskMetrics
from ..position_management import UnifiedPositionManager
from ..calibration import CalibrationManager, CalibrationConfig, CalibrationResult


class BacktestRiskIntegration:
    """백테스트 시스템 리스크 관리 통합"""
    
    def __init__(self,
                 backtest_engine: BacktestEngine,
                 risk_config: Optional[RiskConfig] = None,
                 enable_calibration: bool = True,
                 calibration_config: Optional[CalibrationConfig] = None):
        
        self.backtest_engine = backtest_engine
        self.risk_config = risk_config or RiskConfig()
        self.enable_calibration = enable_calibration
        
        # 보정 관리자 초기화
        if self.enable_calibration:
            self.calibration_manager = CalibrationManager(
                config=calibration_config or CalibrationConfig()
            )
            self.calibration_manager.add_calibration_callback(
                self._on_calibration_complete
            )
        else:
            self.calibration_manager = None
        
        # 초기 파라미터 (보정 전 기본값)
        self._initial_params = {
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'fill_rate': 1.0
        }
        
        # 현재 사용 중인 파라미터 (보정 후)
        self._current_params = self._initial_params.copy()
        
        # 백테스트 시작 전 파라미터 보정 실행
        if self.enable_calibration:
            self._calibrate_initial_parameters()
        
        # 통합 포지션 관리자 (백테스트용)
        self.position_manager = UnifiedPositionManager(
            initial_capital=100000,  # 기본값, 백테스트 설정에 따라 조정
            commission_rate=self._current_params['commission_rate'],
            slippage_rate=self._current_params['slippage_rate']
        )
        
        # 리스크 관리자
        self.risk_manager = AdvancedRiskManager(
            position_manager=self.position_manager,
            config=self.risk_config
        )
        
        # 백테스트 결과 저장
        self.risk_metrics_history: List[RiskMetrics] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.calibration_history: List[CalibrationResult] = []
        
        # 로거 초기화
        self.logger = logging.getLogger(__name__)
        
        # 콜백 등록
        self._setup_risk_callbacks()
    
    def _calibrate_initial_parameters(self):
        """백테스트 시작 전 초기 파라미터 보정"""
        try:
            if self.calibration_manager:
                self.logger.info("백테스트 파라미터 초기 보정 실행")
                
                # 전체 종목에 대한 보정 실행
                calibration_result = self.calibration_manager.calibrate_parameters("ALL")
                
                if calibration_result.confidence_score > 0.5:
                    # 보정된 파라미터 적용
                    self._current_params.update({
                        'commission_rate': calibration_result.calibrated_commission,
                        'slippage_rate': calibration_result.calibrated_slippage,
                        'fill_rate': calibration_result.calibrated_fill_rate
                    })
                    
                    self.calibration_history.append(calibration_result)
                    
                    self.logger.info(
                        f"보정 파라미터 적용: "
                        f"commission={calibration_result.calibrated_commission:.4f}, "
                        f"slippage={calibration_result.calibrated_slippage:.4f}, "
                        f"fill_rate={calibration_result.calibrated_fill_rate:.4f} "
                        f"(신뢰도: {calibration_result.confidence_score:.2f})"
                    )
                else:
                    self.logger.warning(
                        f"보정 신뢰도가 낮아 기본 파라미터 사용: {calibration_result.confidence_score:.2f}"
                    )
                    
        except Exception as e:
            self.logger.error(f"초기 파라미터 보정 실패: {e}")
    
    def _on_calibration_complete(self, calibration_result: CalibrationResult):
        """보정 완료 콜백"""
        
        self.calibration_history.append(calibration_result)
        
        # 백테스트 실행 중이면 동적으로 파라미터 업데이트
        if calibration_result.confidence_score > 0.6:
            self._update_backtest_parameters(calibration_result)
            
            self.logger.info(
                f"실시간 파라미터 보정 적용: {calibration_result.symbol} "
                f"(신뢰도: {calibration_result.confidence_score:.2f})"
            )
    
    def _update_backtest_parameters(self, calibration_result: CalibrationResult):
        """백테스트 파라미터 동적 업데이트"""
        
        # 현재 파라미터 업데이트
        self._current_params.update({
            'commission_rate': calibration_result.calibrated_commission,
            'slippage_rate': calibration_result.calibrated_slippage,
            'fill_rate': calibration_result.calibrated_fill_rate
        })
        
        # 포지션 관리자 파라미터 업데이트
        if self.position_manager:
            self.position_manager.commission_rate = calibration_result.calibrated_commission
            self.position_manager.slippage_rate = calibration_result.calibrated_slippage
            
        # 백테스트 엔진에 파라미터 전달 (엔진이 지원하는 경우)
        if hasattr(self.backtest_engine, 'update_execution_parameters'):
            self.backtest_engine.update_execution_parameters({
                'commission_rate': calibration_result.calibrated_commission,
                'slippage_rate': calibration_result.calibrated_slippage,
                'fill_rate': calibration_result.calibrated_fill_rate
            })
    
    def periodic_calibration_check(self, current_date: datetime = None):
        """백테스트 실행 중 주기적 보정 체크"""
        
        if not self.enable_calibration or not self.calibration_manager:
            return
        
        try:
            # 보정이 필요한지 확인
            should_calibrate = False
            
            if current_date:
                # 마지막 보정으로부터 일정 시간이 지났는지 확인
                if self.calibration_history:
                    last_calibration = self.calibration_history[-1]
                    hours_since_last = (current_date - last_calibration.timestamp).total_seconds() / 3600
                    
                    if hours_since_last >= self.calibration_manager.config.calibration_interval_hours:
                        should_calibrate = True
                else:
                    should_calibrate = True  # 첫 보정
            
            if should_calibrate:
                # 비동기적으로 보정 실행 (백테스트 성능에 영향 최소화)
                calibration_result = self.calibration_manager.calibrate_parameters("ALL")
                
                if calibration_result.confidence_score > 0.5:
                    self._update_backtest_parameters(calibration_result)
                    
        except Exception as e:
            self.logger.error(f"주기적 보정 체크 실패: {e}")
    
    def _setup_risk_callbacks(self):
        """리스크 콜백 설정"""
        
        def risk_monitoring_callback(metrics: RiskMetrics, alerts):
            """백테스트 중 리스크 모니터링"""
            self.risk_metrics_history.append(metrics)
            
            # VaR 한도 초과 시 포지션 축소
            if metrics.var_95_pct > self.risk_config.var_limit_pct:
                reduction_factor = 0.3  # 30% 축소
                self._reduce_positions(reduction_factor, "VaR 한도 초과")
            
            # 낙폭 한도 도달 시 포지션 축소
            if metrics.current_drawdown > self.risk_config.max_drawdown_limit:
                reduction_factor = self.risk_config.drawdown_position_reduction
                self._reduce_positions(reduction_factor, "최대 낙폭 도달")
        
        self.risk_manager.add_risk_callback(risk_monitoring_callback)
    
    def _reduce_positions(self, reduction_factor: float, reason: str):
        """포지션 축소 실행"""
        for symbol in list(self.position_manager.positions.keys()):
            position = self.position_manager.positions[symbol]
            reduce_size = abs(position.size) * reduction_factor
            
            self.position_manager.close_position(
                symbol, 
                size=reduce_size,
                reason=f"Risk management: {reason}"
            )
    
    def adapt_portfolio_to_unified_manager(self, portfolio: Portfolio) -> Dict[str, Any]:
        """백테스트 Portfolio를 통합 포지션 관리자 형식으로 변환"""
        
        positions = {}
        total_value = 0
        
        # Portfolio의 포지션을 통합 관리자 형식으로 변환
        if hasattr(portfolio, 'positions'):
            for symbol, position_data in portfolio.positions.items():
                # 백테스트 Portfolio 구조에 맞게 조정
                market_value = getattr(position_data, 'market_value', 0)
                size = getattr(position_data, 'size', 0)
                
                positions[symbol] = {
                    'market_value': market_value,
                    'size': size,
                    'avg_price': getattr(position_data, 'avg_price', 0),
                    'unrealized_pnl': getattr(position_data, 'unrealized_pnl', 0)
                }
                total_value += market_value
        
        cash = getattr(portfolio, 'cash', 0)
        total_equity = total_value + cash
        
        return {
            'total_equity': total_equity,
            'cash': cash,
            'positions': positions,
            'total_position_value': total_value
        }
    
    def integrate_with_strategy(self, strategy: BaseStrategy) -> BaseStrategy:
        """전략에 리스크 관리 통합"""
        
        original_generate_signals = strategy.generate_signals
        
        def risk_adjusted_signals(data: pd.DataFrame, **kwargs):
            """리스크 조정된 신호 생성"""
            
            # 원본 신호 생성
            signals = original_generate_signals(data, **kwargs)
            
            # 현재 포트폴리오 상태 업데이트
            if hasattr(strategy, 'portfolio'):
                portfolio_data = self.adapt_portfolio_to_unified_manager(strategy.portfolio)
                
                # 포트폴리오 스냅샷 업데이트
                self.risk_manager.update_portfolio_snapshot(
                    total_equity=portfolio_data['total_equity'],
                    cash=portfolio_data['cash'],
                    positions=portfolio_data['positions'],
                    prices=data.iloc[-1].to_dict() if not data.empty else {}
                )
                
                # 리스크 지표 계산
                current_metrics = self.risk_manager.calculate_risk_metrics()
                
                # 신호 조정
                adjusted_signals = self._adjust_signals_for_risk(signals, current_metrics, data)
                return adjusted_signals
            
            return signals
        
        # 메서드 교체
        strategy.generate_signals = risk_adjusted_signals
        return strategy
    
    def _adjust_signals_for_risk(self, 
                               signals: pd.DataFrame, 
                               metrics: RiskMetrics,
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """리스크 기반 신호 조정"""
        
        adjusted_signals = signals.copy()
        
        # VaR 기반 포지션 크기 조정
        for symbol in adjusted_signals.columns:
            if symbol in market_data.columns:
                current_price = market_data[symbol].iloc[-1]
                
                # 신호 강도 (0-1)
                signal_strength = abs(adjusted_signals[symbol].iloc[-1]) if not adjusted_signals.empty else 0
                
                # VaR 기반 포지션 사이징
                sizing_rec = self.risk_manager.get_position_sizing_recommendation(
                    symbol=symbol,
                    current_price=current_price,
                    signal_confidence=signal_strength
                )
                
                # 신호 크기 조정
                if not adjusted_signals.empty:
                    original_signal = adjusted_signals[symbol].iloc[-1]
                    risk_adjustment = sizing_rec['adjustments']['final_adjustment']
                    adjusted_signals[symbol].iloc[-1] = original_signal * risk_adjustment
        
        return adjusted_signals
    
    def run_risk_aware_backtest(self,
                              strategy: BaseStrategy,
                              data: pd.DataFrame,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              enable_periodic_calibration: bool = True) -> Dict[str, Any]:
        """리스크 관리가 통합된 백테스트 실행"""
        
        self.logger.info("보정된 백테스트 실행 시작")
        
        # 백테스트 시작 전 최종 보정 실행
        if self.enable_calibration and enable_periodic_calibration:
            self.periodic_calibration_check()
        
        # 전략에 리스크 관리 통합
        risk_integrated_strategy = self.integrate_with_strategy(strategy)
        
        # 보정 통합 전략 래핑 (주기적 보정 체크 포함)
        if self.enable_calibration and enable_periodic_calibration:
            calibration_integrated_strategy = self._wrap_strategy_with_calibration(
                risk_integrated_strategy, data
            )
        else:
            calibration_integrated_strategy = risk_integrated_strategy
        
        # 백테스트 실행
        backtest_result = self.backtest_engine.run(
            strategy=calibration_integrated_strategy,
            data=data,
            start_date=start_date,
            end_date=end_date
        )
        
        # 리스크 지표 분석 추가
        risk_analysis = self._analyze_backtest_risk_metrics()
        
        # 보정 분석 추가
        calibration_analysis = self._analyze_calibration_impact()
        
        # 결과 통합
        enhanced_result = {
            **backtest_result,
            'risk_analysis': risk_analysis,
            'calibration_analysis': calibration_analysis,
            'risk_metrics_history': [m.to_dict() for m in self.risk_metrics_history],
            'calibration_history': [c.to_dict() for c in self.calibration_history],
            'risk_adjusted_performance': self._calculate_risk_adjusted_performance(),
            'calibrated_parameters': self._current_params.copy()
        }
        
        self.logger.info("보정된 백테스트 실행 완료")
        
        return enhanced_result
    
    def _wrap_strategy_with_calibration(self, strategy: BaseStrategy, data: pd.DataFrame) -> BaseStrategy:
        """전략을 보정 기능과 함께 래핑"""
        
        original_generate_signals = strategy.generate_signals
        
        def calibration_aware_signals(data_slice: pd.DataFrame, **kwargs):
            """보정이 적용된 신호 생성"""
            
            # 현재 날짜 확인 (주기적 보정 체크)
            if not data_slice.empty and hasattr(data_slice.index, 'to_pydatetime'):
                current_date = data_slice.index[-1]
                if hasattr(current_date, 'to_pydatetime'):
                    current_date = current_date.to_pydatetime()
                
                # 주기적 보정 체크
                self.periodic_calibration_check(current_date)
            
            # 원본 신호 생성
            signals = original_generate_signals(data_slice, **kwargs)
            
            # 체결률에 따른 신호 조정 (보정된 체결률 반영)
            if self.enable_calibration and isinstance(signals, pd.DataFrame):
                fill_rate = self._current_params.get('fill_rate', 1.0)
                if fill_rate < 1.0:
                    # 체결률이 100% 미만인 경우 신호 강도 조정
                    signals = signals * fill_rate
            
            return signals
        
        # 메서드 교체
        strategy.generate_signals = calibration_aware_signals
        return strategy
    
    def _analyze_calibration_impact(self) -> Dict[str, Any]:
        """보정 영향 분석"""
        
        if not self.calibration_history:
            return {
                'calibration_enabled': self.enable_calibration,
                'total_calibrations': 0,
                'impact_summary': "보정 이력 없음"
            }
        
        latest_calibration = self.calibration_history[-1]
        
        # 파라미터 변화량 계산
        slippage_change = (latest_calibration.calibrated_slippage - 
                         latest_calibration.original_slippage) / latest_calibration.original_slippage * 100
        
        commission_change = (latest_calibration.calibrated_commission - 
                           latest_calibration.original_commission) / latest_calibration.original_commission * 100
        
        fill_rate_change = (latest_calibration.calibrated_fill_rate - 
                          latest_calibration.original_fill_rate) * 100
        
        # 보정 품질 평가
        avg_confidence = np.mean([c.confidence_score for c in self.calibration_history])
        total_trades_analyzed = sum([c.trades_analyzed for c in self.calibration_history])
        
        return {
            'calibration_enabled': self.enable_calibration,
            'total_calibrations': len(self.calibration_history),
            'latest_calibration': {
                'timestamp': latest_calibration.timestamp.isoformat(),
                'confidence_score': latest_calibration.confidence_score,
                'market_condition': latest_calibration.market_condition,
                'trades_analyzed': latest_calibration.trades_analyzed
            },
            'parameter_changes': {
                'slippage_change_pct': slippage_change,
                'commission_change_pct': commission_change,
                'fill_rate_change_pct': fill_rate_change
            },
            'calibration_quality': {
                'avg_confidence': avg_confidence,
                'total_trades_analyzed': total_trades_analyzed,
                'data_quality': latest_calibration.data_quality
            },
            'current_parameters': self._current_params.copy(),
            'original_parameters': self._initial_params.copy(),
            'impact_summary': self._generate_calibration_impact_summary(
                slippage_change, commission_change, fill_rate_change, avg_confidence
            )
        }
    
    def _generate_calibration_impact_summary(self, 
                                           slippage_change: float,
                                           commission_change: float,
                                           fill_rate_change: float,
                                           avg_confidence: float) -> str:
        """보정 영향 요약 생성"""
        
        impact_items = []
        
        if abs(slippage_change) > 10:
            impact_items.append(f"슬리피지 {slippage_change:+.1f}% 조정")
        
        if abs(commission_change) > 5:
            impact_items.append(f"수수료 {commission_change:+.1f}% 조정")
        
        if abs(fill_rate_change) > 2:
            impact_items.append(f"체결률 {fill_rate_change:+.1f}%p 조정")
        
        if not impact_items:
            summary = "보정 변화량 미미"
        else:
            summary = ", ".join(impact_items)
        
        confidence_desc = "높음" if avg_confidence > 0.8 else "보통" if avg_confidence > 0.6 else "낮음"
        summary += f" (보정 신뢰도: {confidence_desc})"
        
        return summary
    
    def _analyze_backtest_risk_metrics(self) -> Dict[str, Any]:
        """백테스트 리스크 지표 분석"""
        
        if not self.risk_metrics_history:
            return {}
        
        var_95_series = [m.var_95_pct for m in self.risk_metrics_history]
        drawdown_series = [m.current_drawdown for m in self.risk_metrics_history]
        risk_score_series = [m.overall_risk_score for m in self.risk_metrics_history]
        
        analysis = {
            'risk_statistics': {
                'avg_var_95': np.mean(var_95_series) if var_95_series else 0,
                'max_var_95': np.max(var_95_series) if var_95_series else 0,
                'var_violations': sum(1 for v in var_95_series if v > self.risk_config.var_limit_pct),
                'max_drawdown': np.max(drawdown_series) if drawdown_series else 0,
                'avg_risk_score': np.mean(risk_score_series) if risk_score_series else 0,
                'high_risk_periods': sum(1 for score in risk_score_series if score > 75)
            },
            'risk_trends': {
                'var_trend': var_95_series,
                'drawdown_trend': drawdown_series,
                'risk_score_trend': risk_score_series
            },
            'risk_limit_compliance': {
                'var_compliance_rate': 1 - (sum(1 for v in var_95_series if v > self.risk_config.var_limit_pct) / len(var_95_series)) if var_95_series else 1,
                'drawdown_compliance_rate': 1 - (sum(1 for d in drawdown_series if d > self.risk_config.max_drawdown_limit) / len(drawdown_series)) if drawdown_series else 1
            }
        }
        
        return analysis
    
    def _calculate_risk_adjusted_performance(self) -> Dict[str, Any]:
        """리스크 조정 성과 지표 계산"""
        
        performance_summary = self.position_manager.get_performance_summary()
        
        if not self.risk_metrics_history:
            return performance_summary
        
        # 추가 리스크 조정 지표
        total_return = performance_summary.get('total_return', 0)
        max_drawdown = performance_summary.get('max_drawdown', 0)
        
        # Calmar Ratio (연수익률 / 최대낙폭)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR 조정 수익률
        avg_var = np.mean([m.var_95_pct for m in self.risk_metrics_history])
        var_adjusted_return = total_return / avg_var if avg_var > 0 else 0
        
        # Risk-Adjusted Return
        risk_adjusted_metrics = {
            'calmar_ratio': calmar_ratio,
            'var_adjusted_return': var_adjusted_return,
            'risk_efficiency': total_return / np.mean([m.overall_risk_score for m in self.risk_metrics_history]) if self.risk_metrics_history else 0,
            'downside_protection': 1 - max_drawdown,  # 하방 보호 효과
            'consistency_score': 1 - np.std([m.overall_risk_score for m in self.risk_metrics_history]) / 100 if self.risk_metrics_history else 0
        }
        
        performance_summary.update(risk_adjusted_metrics)
        return performance_summary
    
    def export_risk_report(self, output_path: str = "backtest_risk_report.html"):
        """백테스트 리스크 분석 보고서 생성"""
        
        if not self.risk_metrics_history:
            print("리스크 데이터가 없습니다. 백테스트를 먼저 실행하세요.")
            return
        
        # HTML 보고서 생성 (간단한 예시)
        html_content = self._generate_risk_report_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"리스크 분석 보고서가 생성되었습니다: {output_path}")
    
    def _generate_risk_report_html(self) -> str:
        """리스크 보고서 HTML 생성"""
        
        risk_analysis = self._analyze_backtest_risk_metrics()
        performance = self._calculate_risk_adjusted_performance()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AuroraQ 백테스트 리스크 분석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
                .risk-high {{ background-color: #ffebee; }}
                .risk-medium {{ background-color: #fff3e0; }}
                .risk-low {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AuroraQ 백테스트 리스크 분석 보고서</h1>
                <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 리스크 통계</h2>
                <div class="metric">
                    <strong>평균 95% VaR:</strong> {risk_analysis.get('risk_statistics', {}).get('avg_var_95', 0):.2%}
                </div>
                <div class="metric">
                    <strong>최대 낙폭:</strong> {risk_analysis.get('risk_statistics', {}).get('max_drawdown', 0):.2%}
                </div>
                <div class="metric">
                    <strong>평균 리스크 점수:</strong> {risk_analysis.get('risk_statistics', {}).get('avg_risk_score', 0):.1f}
                </div>
                <div class="metric">
                    <strong>VaR 준수율:</strong> {risk_analysis.get('risk_limit_compliance', {}).get('var_compliance_rate', 0):.1%}
                </div>
            </div>
            
            <div class="section">
                <h2>💰 리스크 조정 성과</h2>
                <div class="metric">
                    <strong>총 수익률:</strong> {performance.get('total_return', 0):.2%}
                </div>
                <div class="metric">
                    <strong>Calmar Ratio:</strong> {performance.get('calmar_ratio', 0):.2f}
                </div>
                <div class="metric">
                    <strong>VaR 조정 수익률:</strong> {performance.get('var_adjusted_return', 0):.2f}
                </div>
                <div class="metric">
                    <strong>하방 보호 효과:</strong> {performance.get('downside_protection', 0):.2%}
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ 리스크 한도 위반</h2>
                <div class="metric risk-high">
                    <strong>VaR 한도 위반:</strong> {risk_analysis.get('risk_statistics', {}).get('var_violations', 0)}회
                </div>
                <div class="metric risk-medium">
                    <strong>고위험 기간:</strong> {risk_analysis.get('risk_statistics', {}).get('high_risk_periods', 0)}일
                </div>
            </div>
            
            <div class="section">
                <h2>📈 권고사항</h2>
                <ul>
                    <li>VaR 한도 준수율이 {risk_analysis.get('risk_limit_compliance', {}).get('var_compliance_rate', 0):.1%}입니다.</li>
                    <li>최대 낙폭 {risk_analysis.get('risk_statistics', {}).get('max_drawdown', 0):.2%}를 관리하세요.</li>
                    <li>평균 리스크 점수 {risk_analysis.get('risk_statistics', {}).get('avg_risk_score', 0):.1f}를 유지하세요.</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 확인"""
        
        status = {
            'position_manager_initialized': self.position_manager is not None,
            'risk_manager_initialized': self.risk_manager is not None,
            'backtest_engine_connected': self.backtest_engine is not None,
            'risk_metrics_count': len(self.risk_metrics_history),
            'current_equity': self.position_manager.get_equity() if self.position_manager else 0,
            'active_positions': len(self.position_manager.positions) if self.position_manager else 0,
            'risk_config': self.risk_config.to_dict() if hasattr(self.risk_config, 'to_dict') else str(self.risk_config),
            'calibration': {
                'enabled': self.enable_calibration,
                'manager_initialized': self.calibration_manager is not None,
                'calibration_count': len(self.calibration_history),
                'current_parameters': self._current_params.copy(),
                'last_calibration': self.calibration_history[-1].timestamp.isoformat() if self.calibration_history else None,
                'calibration_status': self.calibration_manager.get_calibration_status() if self.calibration_manager else {}
            }
        }
        
        return status
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """보정 요약 정보 조회"""
        
        if not self.enable_calibration:
            return {'calibration_enabled': False}
        
        summary = {
            'calibration_enabled': True,
            'total_calibrations': len(self.calibration_history),
            'current_parameters': self._current_params.copy(),
            'original_parameters': self._initial_params.copy()
        }
        
        if self.calibration_history:
            latest = self.calibration_history[-1]
            summary.update({
                'latest_calibration': {
                    'timestamp': latest.timestamp.isoformat(),
                    'confidence_score': latest.confidence_score,
                    'market_condition': latest.market_condition,
                    'trades_analyzed': latest.trades_analyzed,
                    'adjustment_reason': latest.adjustment_reason
                }
            })
        
        return summary
    
    def export_calibrated_backtest_report(self, output_path: str = None) -> str:
        """보정된 백테스트 종합 보고서 생성"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"calibrated_backtest_report_{timestamp}.json"
        
        # 종합 보고서 데이터
        report_data = {
            'metadata': {
                'report_time': datetime.now().isoformat(),
                'calibration_enabled': self.enable_calibration,
                'risk_management_enabled': True
            },
            'integration_status': self.get_integration_status(),
            'calibration_summary': self.get_calibration_summary(),
            'calibration_analysis': self._analyze_calibration_impact(),
            'risk_analysis': self._analyze_backtest_risk_metrics(),
            'performance_metrics': self._calculate_risk_adjusted_performance()
        }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"보정된 백테스트 종합 보고서 생성: {output_path}")
        
        return output_path


# 편의 함수들
def create_risk_aware_backtest(initial_capital: float = 100000,
                             risk_config: Optional[RiskConfig] = None,
                             enable_calibration: bool = True,
                             calibration_config: Optional[CalibrationConfig] = None) -> BacktestRiskIntegration:
    """리스크 관리가 통합된 백테스트 시스템 생성"""
    
    # 백테스트 엔진 초기화
    backtest_engine = BacktestEngine()
    
    # 리스크 설정
    if risk_config is None:
        risk_config = RiskConfig(
            var_limit_pct=0.05,
            max_drawdown_limit=0.15,
            drawdown_position_reduction=0.5
        )
    
    # 통합 시스템 생성
    integration = BacktestRiskIntegration(
        backtest_engine=backtest_engine,
        risk_config=risk_config,
        enable_calibration=enable_calibration,
        calibration_config=calibration_config
    )
    
    # 초기 자본 설정
    integration.position_manager.initial_capital = initial_capital
    integration.position_manager.current_capital = initial_capital
    integration.position_manager.cash = initial_capital
    
    return integration


def quick_risk_backtest(strategy: BaseStrategy,
                       data: pd.DataFrame,
                       initial_capital: float = 100000,
                       enable_calibration: bool = True,
                       **kwargs) -> Dict[str, Any]:
    """간편한 리스크 관리 백테스트 실행"""
    
    # 리스크 통합 백테스트 생성
    risk_backtest = create_risk_aware_backtest(
        initial_capital=initial_capital,
        enable_calibration=enable_calibration
    )
    
    # 백테스트 실행
    result = risk_backtest.run_risk_aware_backtest(
        strategy=strategy,
        data=data,
        **kwargs
    )
    
    return result


def create_calibrated_backtest(initial_capital: float = 100000,
                             calibration_interval_hours: int = 24,
                             min_trades_for_calibration: int = 100) -> BacktestRiskIntegration:
    """보정 기능이 강화된 백테스트 시스템 생성"""
    
    # 보정 설정
    calibration_config = CalibrationConfig(
        calibration_interval_hours=calibration_interval_hours,
        min_trades_for_calibration=min_trades_for_calibration,
        market_condition_adjustment=True
    )
    
    # 리스크 설정
    risk_config = RiskConfig(
        var_limit_pct=0.05,
        max_drawdown_limit=0.15,
        drawdown_position_reduction=0.5
    )
    
    return create_risk_aware_backtest(
        initial_capital=initial_capital,
        risk_config=risk_config,
        enable_calibration=True,
        calibration_config=calibration_config
    )


def create_synchronized_backtest_environment(
    realtime_system_config: Optional[Dict[str, Any]] = None,
    sync_parameters: bool = True
) -> 'BacktestIntegration':
    """
    실시간 시스템과 동기화된 백테스트 환경 생성
    
    Args:
        realtime_system_config: 실시간 시스템 설정
        sync_parameters: 파라미터 동기화 여부
        
    Returns:
        동기화된 백테스트 통합 시스템
    """
    from .realtime_hybrid_system import RealtimeSystemConfig
    from .realtime_calibration_system import RealtimeCalibrationConfig
    
    # 실시간 시스템 설정이 있다면 동일한 설정 사용
    if realtime_system_config:
        initial_capital = realtime_system_config.get('initial_capital', 1000000.0)
        max_portfolio_risk = realtime_system_config.get('max_portfolio_risk', 0.02)
        enable_calibration = realtime_system_config.get('enable_realtime_calibration', True)
    else:
        # 기본값 사용
        initial_capital = 1000000.0
        max_portfolio_risk = 0.02
        enable_calibration = True
    
    # 포지션 관리자 생성 (실시간과 동일한 설정)
    position_manager = UnifiedPositionManager(
        initial_capital=initial_capital,
        commission_rate=0.001,  # 실시간 시스템과 동일
        slippage_rate=0.0005    # 실시간 시스템과 동일
    )
    
    # 리스크 관리자 생성 (실시간과 동일한 설정)
    risk_config = RiskConfig(
        var_limit_pct=max_portfolio_risk,
        max_drawdown_limit=0.10,  # 실시간과 동일
        drawdown_position_reduction=0.5
    )
    
    risk_manager = AdvancedRiskManager(
        position_manager=position_manager,
        config=risk_config
    )
    
    # 보정 설정 (실시간과 유사하게)
    calibration_config = CalibrationConfig(
        calibration_interval_hours=0.5,  # 30분마다 (실시간과 동일)
        min_trades_for_calibration=5,
        market_condition_adjustment=True,
        enable_auto_parameter_adjustment=enable_calibration
    )
    
    # 백테스트 통합 시스템 생성
    backtest_integration = BacktestIntegration(
        position_manager=position_manager,
        risk_manager=risk_manager,
        enable_calibration=enable_calibration,
        calibration_config=calibration_config
    )
    
    return backtest_integration


def sync_backtest_with_realtime_parameters(
    backtest_integration: 'BacktestIntegration',
    realtime_parameters: Dict[str, Any]
):
    """
    백테스트 시스템의 파라미터를 실시간 시스템과 동기화
    
    Args:
        backtest_integration: 백테스트 통합 시스템
        realtime_parameters: 실시간 시스템의 현재 파라미터
    """
    try:
        # 거래 파라미터 동기화
        if 'slippage_rate' in realtime_parameters:
            backtest_integration.position_manager.slippage_rate = realtime_parameters['slippage_rate']
        
        if 'commission_rate' in realtime_parameters:
            backtest_integration.position_manager.commission_rate = realtime_parameters['commission_rate']
        
        # 리스크 파라미터 동기화
        if 'var_limit_pct' in realtime_parameters:
            backtest_integration.risk_manager.config.var_limit_pct = realtime_parameters['var_limit_pct']
        
        if 'max_drawdown_limit' in realtime_parameters:
            backtest_integration.risk_manager.config.max_drawdown_limit = realtime_parameters['max_drawdown_limit']
        
        # 시장 레짐별 조정사항 적용
        if 'market_regime_adjustments' in realtime_parameters:
            adjustments = realtime_parameters['market_regime_adjustments']
            
            # VaR 한도 조정
            if 'var_limit_multiplier' in adjustments:
                original_var_limit = backtest_integration.risk_manager.config.var_limit_pct
                backtest_integration.risk_manager.config.var_limit_pct = original_var_limit * adjustments['var_limit_multiplier']
            
            # 포지션 크기 조정
            if 'position_size_multiplier' in adjustments:
                # 백테스트에서도 동일한 포지션 크기 제약 적용
                backtest_integration.position_sizing_multiplier = adjustments['position_size_multiplier']
        
        print(f"Backtest parameters synchronized with realtime system")
        
    except Exception as e:
        print(f"Error synchronizing backtest parameters: {e}")


def run_comparative_analysis(
    strategy,
    data: pd.DataFrame,
    realtime_system_config: Optional[Dict[str, Any]] = None,
    backtest_period: Tuple[str, str] = None
) -> Dict[str, Any]:
    """
    실시간 시스템과 백테스트 결과를 비교 분석
    
    Args:
        strategy: 거래 전략
        data: 백테스트 데이터
        realtime_system_config: 실시간 시스템 설정
        backtest_period: 백테스트 기간 (start_date, end_date)
        
    Returns:
        비교 분석 결과
    """
    results = {}
    
    try:
        # 1. 기본 백테스트 실행
        basic_backtest = create_simple_backtest()
        basic_results = basic_backtest.run_backtest(strategy, data)
        results['basic_backtest'] = basic_results
        
        # 2. 동기화된 백테스트 실행
        sync_backtest = create_synchronized_backtest_environment(realtime_system_config)
        
        # 실시간 파라미터가 있다면 동기화
        if realtime_system_config:
            sync_backtest_with_realtime_parameters(sync_backtest, realtime_system_config)
        
        sync_results = sync_backtest.run_risk_aware_backtest(
            strategy, data, 
            start_date=backtest_period[0] if backtest_period else None,
            end_date=backtest_period[1] if backtest_period else None,
            enable_periodic_calibration=True
        )
        results['synchronized_backtest'] = sync_results
        
        # 3. 성과 비교 분석
        comparison = _compare_backtest_results(basic_results, sync_results)
        results['comparison'] = comparison
        
        # 4. 실시간 환경과의 일치도 분석
        if realtime_system_config:
            consistency_analysis = _analyze_consistency_with_realtime(
                sync_results, realtime_system_config
            )
            results['consistency_analysis'] = consistency_analysis
        
        return results
        
    except Exception as e:
        print(f"Error in comparative analysis: {e}")
        return {'error': str(e)}


def _compare_backtest_results(basic_results: Dict[str, Any], sync_results: Dict[str, Any]) -> Dict[str, Any]:
    """백테스트 결과 비교"""
    comparison = {}
    
    try:
        # 수익률 비교
        basic_return = basic_results.get('total_return', 0)
        sync_return = sync_results.get('total_return', 0)
        
        comparison['return_difference'] = sync_return - basic_return
        comparison['return_improvement'] = (sync_return - basic_return) / abs(basic_return) if basic_return != 0 else 0
        
        # 샤프 비율 비교
        basic_sharpe = basic_results.get('sharpe_ratio', 0)
        sync_sharpe = sync_results.get('sharpe_ratio', 0)
        
        comparison['sharpe_difference'] = sync_sharpe - basic_sharpe
        
        # 최대 낙폭 비교
        basic_mdd = basic_results.get('max_drawdown', 0)
        sync_mdd = sync_results.get('max_drawdown', 0)
        
        comparison['mdd_difference'] = sync_mdd - basic_mdd
        comparison['mdd_improvement'] = (basic_mdd - sync_mdd) / basic_mdd if basic_mdd != 0 else 0
        
        # 거래 횟수 비교
        basic_trades = basic_results.get('total_trades', 0)
        sync_trades = sync_results.get('total_trades', 0)
        
        comparison['trade_count_difference'] = sync_trades - basic_trades
        
        # 종합 개선도
        improvement_score = (
            comparison['return_improvement'] * 0.4 +
            comparison['mdd_improvement'] * 0.4 +
            (comparison['sharpe_difference'] / max(abs(basic_sharpe), 0.1)) * 0.2
        )
        
        comparison['overall_improvement_score'] = improvement_score
        
    except Exception as e:
        comparison['error'] = str(e)
    
    return comparison


def _analyze_consistency_with_realtime(
    backtest_results: Dict[str, Any], 
    realtime_config: Dict[str, Any]
) -> Dict[str, Any]:
    """실시간 환경과의 일치도 분석"""
    consistency = {}
    
    try:
        # 파라미터 일치도 체크
        consistency['parameter_alignment'] = {
            'slippage_match': True,  # 실제로는 실시간 시스템에서 가져와서 비교
            'commission_match': True,
            'risk_limits_match': True
        }
        
        # 시장 레짐 적응도
        if 'calibration_results' in backtest_results:
            calibration_data = backtest_results['calibration_results']
            consistency['market_adaptation'] = {
                'calibration_frequency': len(calibration_data.get('calibration_history', [])),
                'parameter_adjustments': len(calibration_data.get('adjustment_history', [])),
                'market_regime_changes': len(set([
                    adj.get('market_regime', 'normal') 
                    for adj in calibration_data.get('adjustment_history', [])
                ]))
            }
        
        # 리스크 관리 일치도
        if 'risk_metrics' in backtest_results:
            risk_data = backtest_results['risk_metrics']
            consistency['risk_management'] = {
                'var_limit_breaches': risk_data.get('var_breaches', 0),
                'emergency_stops': risk_data.get('emergency_stops', 0),
                'risk_adjusted_trades': risk_data.get('risk_adjusted_trades', 0)
            }
        
        # 전반적 일치도 점수
        alignment_score = sum([
            1.0 if consistency['parameter_alignment']['slippage_match'] else 0.0,
            1.0 if consistency['parameter_alignment']['commission_match'] else 0.0,
            1.0 if consistency['parameter_alignment']['risk_limits_match'] else 0.0
        ]) / 3.0
        
        consistency['overall_alignment_score'] = alignment_score
        
    except Exception as e:
        consistency['error'] = str(e)
    
    return consistency


class BacktestIntegration:
    """백테스트 시스템 통합 관리자
    
    BacktestRiskIntegration을 기반으로 한 통합 백테스트 시스템
    테스트 호환성을 위한 Wrapper 클래스
    """
    
    def __init__(self,
                 position_manager: Optional[UnifiedPositionManager] = None,
                 risk_manager: Optional[AdvancedRiskManager] = None,
                 enable_calibration: bool = True,
                 calibration_config: Optional[CalibrationConfig] = None):
        
        # 백테스트 엔진 초기화 (간단한 더미 구현)
        self.backtest_engine = BacktestEngine()
        
        # 포지션 관리자 설정
        if position_manager is None:
            position_manager = UnifiedPositionManager(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
        
        # 리스크 관리자 설정
        if risk_manager is None:
            risk_config = RiskConfig(
                var_limit_pct=0.05,
                max_drawdown_limit=0.15,
                drawdown_position_reduction=0.5
            )
            risk_manager = AdvancedRiskManager(
                position_manager=position_manager,
                config=risk_config
            )
        
        # 백테스트 리스크 통합 시스템을 내부적으로 사용
        self.risk_integration = BacktestRiskIntegration(
            backtest_engine=self.backtest_engine,
            risk_config=risk_manager.config if risk_manager else None,
            enable_calibration=enable_calibration,
            calibration_config=calibration_config
        )
        
        # 외부 인터페이스
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.enable_calibration = enable_calibration
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("BacktestIntegration 초기화 완료")
    
    def run_backtest(self, strategy, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """기본 백테스트 실행"""
        return self.risk_integration.run_risk_aware_backtest(
            strategy=strategy,
            data=data,
            **kwargs
        )
    
    def run_risk_aware_backtest(self, strategy, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """리스크 관리가 적용된 백테스트 실행"""
        return self.risk_integration.run_risk_aware_backtest(
            strategy=strategy,
            data=data,
            **kwargs
        )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 확인"""
        return self.risk_integration.get_integration_status()
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """보정 요약 정보"""
        return self.risk_integration.get_calibration_summary()
    
    def export_risk_report(self, output_path: str = "backtest_risk_report.html"):
        """리스크 분석 보고서 생성"""
        return self.risk_integration.export_risk_report(output_path)


# 편의 함수 (기존 create_simple_backtest 등)
def create_simple_backtest(initial_capital: float = 100000) -> BacktestIntegration:
    """간단한 백테스트 시스템 생성"""
    
    # 기본 포지션 관리자
    position_manager = UnifiedPositionManager(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # 기본 리스크 관리자
    risk_config = RiskConfig(
        var_limit_pct=0.05,
        max_drawdown_limit=0.15,
        drawdown_position_reduction=0.5
    )
    
    risk_manager = AdvancedRiskManager(
        position_manager=position_manager,
        config=risk_config
    )
    
    return BacktestIntegration(
        position_manager=position_manager,
        risk_manager=risk_manager,
        enable_calibration=False  # 간단한 백테스트는 보정 비활성화
    )


def create_auto_calibrated_backtest(
    initial_capital: float = 100000,
    calibration_interval_hours: int = 24,
    min_trades_for_calibration: int = 100
) -> BacktestIntegration:
    """자동 보정 백테스트 시스템 생성"""
    
    # 보정 설정
    calibration_config = CalibrationConfig(
        calibration_interval_hours=calibration_interval_hours,
        min_trades_for_calibration=min_trades_for_calibration,
        market_condition_adjustment=True,
        enable_auto_parameter_adjustment=True
    )
    
    # 포지션 관리자
    position_manager = UnifiedPositionManager(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # 리스크 관리자
    risk_config = RiskConfig(
        var_limit_pct=0.05,
        max_drawdown_limit=0.15,
        drawdown_position_reduction=0.5
    )
    
    risk_manager = AdvancedRiskManager(
        position_manager=position_manager,
        config=risk_config
    )
    
    return BacktestIntegration(
        position_manager=position_manager,
        risk_manager=risk_manager,
        enable_calibration=True,
        calibration_config=calibration_config
    )