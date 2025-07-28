#!/usr/bin/env python3
"""
실거래 데이터 기반 백테스트 보정 관리자
실거래 분석 결과를 백테스트 환경에 자동으로 반영
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import asyncio
from pathlib import Path

from .execution_analyzer import ExecutionAnalyzer, ExecutionMetrics
from .market_condition_detector import MarketConditionDetector

@dataclass
class CalibrationConfig:
    """보정 설정"""
    
    # 보정 주기
    calibration_interval_hours: int = 24  # 24시간마다 보정
    min_trades_for_calibration: int = 100  # 최소 거래 수
    calibration_window_days: int = 30     # 보정용 데이터 기간
    
    # 보정 가중치
    slippage_weight: float = 0.7          # 실거래 슬리피지 반영 비율
    fill_rate_weight: float = 0.8         # 실거래 체결률 반영 비율
    commission_weight: float = 1.0        # 실거래 수수료 반영 비율
    
    # 보정 한계값
    max_slippage_adjustment: float = 0.005  # 최대 슬리피지 조정 (0.5%)
    min_fill_rate: float = 0.7             # 최소 체결률
    max_commission_rate: float = 0.01      # 최대 수수료율 (1%)
    
    # 시장 상황별 보정
    market_condition_adjustment: bool = True
    volatility_adjustment_factor: float = 1.5
    
    # 백테스트 적용
    apply_to_backtest: bool = True
    backup_original_params: bool = True


@dataclass 
class CalibrationResult:
    """보정 결과"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    
    # 보정 전 파라미터
    original_slippage: float = 0.0
    original_commission: float = 0.0
    original_fill_rate: float = 1.0
    
    # 보정 후 파라미터
    calibrated_slippage: float = 0.0
    calibrated_commission: float = 0.0
    calibrated_fill_rate: float = 1.0
    
    # 보정 근거
    execution_metrics: Optional[ExecutionMetrics] = None
    market_condition: str = "normal"
    confidence_score: float = 0.0
    
    # 보정 통계
    trades_analyzed: int = 0
    data_quality: float = 1.0
    adjustment_reason: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'original_params': {
                'slippage': self.original_slippage,
                'commission': self.original_commission,
                'fill_rate': self.original_fill_rate
            },
            'calibrated_params': {
                'slippage': self.calibrated_slippage,
                'commission': self.calibrated_commission,
                'fill_rate': self.calibrated_fill_rate
            },
            'analysis': {
                'market_condition': self.market_condition,
                'confidence_score': self.confidence_score,
                'trades_analyzed': self.trades_analyzed,
                'data_quality': self.data_quality,
                'adjustment_reason': self.adjustment_reason
            }
        }


class CalibrationManager:
    """실거래 데이터 기반 백테스트 보정 관리자"""
    
    def __init__(self, 
                 config: Optional[CalibrationConfig] = None,
                 execution_analyzer: Optional[ExecutionAnalyzer] = None):
        
        self.config = config or CalibrationConfig()
        self.execution_analyzer = execution_analyzer or ExecutionAnalyzer()
        self.market_detector = MarketConditionDetector()
        
        self.logger = logging.getLogger(__name__)
        
        # 보정 이력
        self.calibration_history: List[CalibrationResult] = []
        self.last_calibration_time: Dict[str, datetime] = {}
        
        # 보정 콜백
        self.calibration_callbacks: List[Callable] = []
        
        # 자동 보정 태스크
        self.auto_calibration_task: Optional[asyncio.Task] = None
        self.is_auto_calibrating = False
        
    def add_calibration_callback(self, callback: Callable[[CalibrationResult], None]):
        """보정 완료 콜백 추가"""
        self.calibration_callbacks.append(callback)
    
    def calibrate_parameters(self, 
                           symbol: str = "ALL",
                           force_calibration: bool = False) -> CalibrationResult:
        """파라미터 보정 실행"""
        
        # 보정 필요성 확인
        if not force_calibration and not self._should_calibrate(symbol):
            # 기존 보정 결과 반환
            return self._get_latest_calibration(symbol)
        
        self.logger.info(f"파라미터 보정 시작: {symbol}")
        
        # 실거래 데이터 분석
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.calibration_window_days)
        
        execution_metrics = self.execution_analyzer.analyze_execution_logs(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # 시장 상황 분석
        market_condition = self.market_detector.detect_current_condition(symbol)
        
        # 보정 실행
        calibration_result = self._perform_calibration(
            symbol, execution_metrics, market_condition
        )
        
        # 보정 이력 저장
        self.calibration_history.append(calibration_result)
        self.last_calibration_time[symbol] = datetime.now()
        
        # 콜백 실행
        self._trigger_calibration_callbacks(calibration_result)
        
        self.logger.info(f"파라미터 보정 완료: {symbol}")
        
        return calibration_result
    
    def _should_calibrate(self, symbol: str) -> bool:
        """보정 필요성 판단"""
        
        # 강제 보정 조건
        if symbol not in self.last_calibration_time:
            return True
        
        # 시간 기반 보정
        last_time = self.last_calibration_time[symbol]
        hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
        
        if hours_since_last >= self.config.calibration_interval_hours:
            return True
        
        # 시장 상황 변화 기반 보정
        if self.config.market_condition_adjustment:
            current_condition = self.market_detector.detect_current_condition(symbol)
            last_calibration = self._get_latest_calibration(symbol)
            
            if (last_calibration and 
                current_condition != last_calibration.market_condition and
                current_condition in ['high_volatility', 'low_liquidity']):
                return True
        
        return False
    
    def _get_latest_calibration(self, symbol: str) -> Optional[CalibrationResult]:
        """최신 보정 결과 조회"""
        
        symbol_calibrations = [
            cal for cal in self.calibration_history 
            if cal.symbol == symbol or symbol == "ALL"
        ]
        
        if not symbol_calibrations:
            return None
        
        return max(symbol_calibrations, key=lambda x: x.timestamp)
    
    def _perform_calibration(self,
                           symbol: str,
                           execution_metrics: ExecutionMetrics,
                           market_condition: str) -> CalibrationResult:
        """보정 수행"""
        
        # 기본 파라미터 (백테스트 기본값)
        original_slippage = 0.0005   # 0.05%
        original_commission = 0.001  # 0.1%
        original_fill_rate = 1.0     # 100%
        
        result = CalibrationResult(
            symbol=symbol,
            original_slippage=original_slippage,
            original_commission=original_commission,
            original_fill_rate=original_fill_rate,
            market_condition=market_condition,
            execution_metrics=execution_metrics
        )
        
        # 데이터 품질 확인
        if execution_metrics.total_trades < self.config.min_trades_for_calibration:
            self.logger.warning(f"보정용 거래 데이터 부족: {execution_metrics.total_trades}")
            result.confidence_score = 0.3
            result.adjustment_reason.append("Insufficient trade data")
            return self._apply_minimal_adjustment(result)
        
        result.trades_analyzed = execution_metrics.total_trades
        result.data_quality = execution_metrics.data_quality_score
        
        # 슬리피지 보정
        calibrated_slippage = self._calibrate_slippage(
            original_slippage, execution_metrics, market_condition
        )
        
        # 수수료 보정
        calibrated_commission = self._calibrate_commission(
            original_commission, execution_metrics
        )
        
        # 체결률 보정
        calibrated_fill_rate = self._calibrate_fill_rate(
            original_fill_rate, execution_metrics, market_condition
        )
        
        # 보정 결과 적용
        result.calibrated_slippage = calibrated_slippage
        result.calibrated_commission = calibrated_commission
        result.calibrated_fill_rate = calibrated_fill_rate
        
        # 신뢰도 점수 계산
        result.confidence_score = self._calculate_confidence_score(
            execution_metrics, market_condition
        )
        
        return result
    
    def _calibrate_slippage(self,
                          original: float,
                          metrics: ExecutionMetrics,
                          market_condition: str) -> float:
        """슬리피지 보정"""
        
        if metrics.avg_slippage <= 0:
            return original
        
        # 실거래 슬리피지 반영
        real_slippage = metrics.avg_slippage
        
        # 가중 평균
        weight = self.config.slippage_weight
        calibrated = original * (1 - weight) + real_slippage * weight
        
        # 시장 상황별 조정
        if self.config.market_condition_adjustment:
            adjustment_factor = self._get_market_adjustment_factor(market_condition)
            calibrated *= adjustment_factor
        
        # 한계값 적용
        max_adjustment = original + self.config.max_slippage_adjustment
        calibrated = min(calibrated, max_adjustment)
        
        return max(0, calibrated)
    
    def _calibrate_commission(self,
                            original: float,
                            metrics: ExecutionMetrics) -> float:
        """수수료 보정"""
        
        if metrics.avg_commission_rate <= 0:
            return original
        
        # 실거래 수수료율 반영
        real_commission = metrics.avg_commission_rate
        
        # 가중 평균
        weight = self.config.commission_weight
        calibrated = original * (1 - weight) + real_commission * weight
        
        # 한계값 적용
        calibrated = min(calibrated, self.config.max_commission_rate)
        
        return max(0, calibrated)
    
    def _calibrate_fill_rate(self,
                           original: float,
                           metrics: ExecutionMetrics,
                           market_condition: str) -> float:
        """체결률 보정"""
        
        if metrics.fill_rate <= 0:
            return original
        
        # 실거래 체결률 반영
        real_fill_rate = metrics.fill_rate
        
        # 가중 평균
        weight = self.config.fill_rate_weight
        calibrated = original * (1 - weight) + real_fill_rate * weight
        
        # 시장 상황별 조정
        if self.config.market_condition_adjustment:
            if market_condition in ['high_volatility', 'low_liquidity']:
                calibrated *= 0.95  # 5% 감소
        
        # 한계값 적용
        calibrated = max(calibrated, self.config.min_fill_rate)
        
        return min(1.0, calibrated)
    
    def _get_market_adjustment_factor(self, market_condition: str) -> float:
        """시장 상황별 조정 팩터"""
        
        adjustments = {
            'normal': 1.0,
            'high_volatility': self.config.volatility_adjustment_factor,
            'low_liquidity': 1.3,
            'market_stress': 2.0,
            'after_hours': 1.2
        }
        
        return adjustments.get(market_condition, 1.0)
    
    def _calculate_confidence_score(self,
                                  metrics: ExecutionMetrics,
                                  market_condition: str) -> float:
        """보정 신뢰도 점수 계산"""
        
        # 데이터 품질 점수
        data_score = metrics.data_quality_score
        
        # 데이터 양 점수
        trade_score = min(1.0, metrics.total_trades / 500)  # 500건 기준
        
        # 시장 상황 점수
        market_scores = {
            'normal': 1.0,
            'high_volatility': 0.8,
            'low_liquidity': 0.7,
            'market_stress': 0.6,
            'after_hours': 0.8
        }
        market_score = market_scores.get(market_condition, 0.9)
        
        # 통계적 유의성 점수
        stat_score = 1.0 if metrics.total_trades >= 100 else 0.5
        
        # 종합 점수
        confidence = (data_score * 0.3 + 
                     trade_score * 0.3 + 
                     market_score * 0.2 + 
                     stat_score * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    def _apply_minimal_adjustment(self, result: CalibrationResult) -> CalibrationResult:
        """최소 보정 적용 (데이터 부족 시)"""
        
        # 시장 상황만 고려한 최소 조정
        market_factor = self._get_market_adjustment_factor(result.market_condition)
        
        result.calibrated_slippage = result.original_slippage * market_factor
        result.calibrated_commission = result.original_commission
        result.calibrated_fill_rate = result.original_fill_rate
        
        if result.market_condition != 'normal':
            result.calibrated_fill_rate *= 0.98  # 2% 감소
        
        result.adjustment_reason.append("Minimal adjustment due to insufficient data")
        
        return result
    
    def _trigger_calibration_callbacks(self, result: CalibrationResult):
        """보정 콜백 실행"""
        
        for callback in self.calibration_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"보정 콜백 실행 실패: {e}")
    
    async def start_auto_calibration(self, symbols: List[str] = None):
        """자동 보정 시작"""
        
        if self.is_auto_calibrating:
            self.logger.warning("자동 보정이 이미 실행 중입니다.")
            return
        
        if symbols is None:
            symbols = ["ALL"]
        
        self.is_auto_calibrating = True
        self.auto_calibration_task = asyncio.create_task(
            self._auto_calibration_loop(symbols)
        )
        
        self.logger.info(f"자동 보정 시작: {symbols}")
    
    async def stop_auto_calibration(self):
        """자동 보정 중단"""
        
        if not self.is_auto_calibrating:
            return
        
        self.is_auto_calibrating = False
        
        if self.auto_calibration_task:
            self.auto_calibration_task.cancel()
            try:
                await self.auto_calibration_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("자동 보정 중단")
    
    async def _auto_calibration_loop(self, symbols: List[str]):
        """자동 보정 루프"""
        
        try:
            while self.is_auto_calibrating:
                # 각 심볼별 보정 실행
                for symbol in symbols:
                    if not self.is_auto_calibrating:
                        break
                    
                    try:
                        result = self.calibrate_parameters(symbol)
                        self.logger.info(
                            f"자동 보정 완료: {symbol} "
                            f"(신뢰도: {result.confidence_score:.2f})"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"자동 보정 실패 {symbol}: {e}")
                
                # 다음 보정까지 대기
                await asyncio.sleep(3600)  # 1시간 대기
                
        except asyncio.CancelledError:
            self.logger.info("자동 보정 루프 종료")
        except Exception as e:
            self.logger.error(f"자동 보정 루프 오류: {e}")
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """보정 상태 조회"""
        
        status = {
            'auto_calibration_active': self.is_auto_calibrating,
            'total_calibrations': len(self.calibration_history),
            'symbols_calibrated': list(self.last_calibration_time.keys()),
            'last_calibrations': {}
        }
        
        # 심볼별 최신 보정 정보
        for symbol in self.last_calibration_time.keys():
            latest = self._get_latest_calibration(symbol)
            if latest:
                status['last_calibrations'][symbol] = {
                    'timestamp': latest.timestamp.isoformat(),
                    'confidence': latest.confidence_score,
                    'trades_analyzed': latest.trades_analyzed,
                    'market_condition': latest.market_condition
                }
        
        return status
    
    def export_calibration_report(self, 
                                symbol: str = "ALL",
                                output_path: str = None) -> str:
        """보정 보고서 생성"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"calibration_report_{symbol}_{timestamp}.json"
        
        # 해당 심볼의 보정 이력
        symbol_history = [
            cal for cal in self.calibration_history 
            if cal.symbol == symbol or symbol == "ALL"
        ]
        
        if not symbol_history:
            self.logger.warning(f"보정 이력이 없습니다: {symbol}")
            return ""
        
        # 보고서 데이터 생성
        report_data = {
            'metadata': {
                'report_time': datetime.now().isoformat(),
                'symbol': symbol,
                'total_calibrations': len(symbol_history),
                'config': self._config_to_dict()
            },
            'calibration_history': [cal.to_dict() for cal in symbol_history],
            'summary': self._generate_calibration_summary(symbol_history),
            'recommendations': self._generate_calibration_recommendations(symbol_history)
        }
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"보정 보고서 생성: {output_path}")
        
        return output_path
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'calibration_interval_hours': self.config.calibration_interval_hours,
            'min_trades_for_calibration': self.config.min_trades_for_calibration,
            'calibration_window_days': self.config.calibration_window_days,
            'slippage_weight': self.config.slippage_weight,
            'fill_rate_weight': self.config.fill_rate_weight,
            'commission_weight': self.config.commission_weight,
            'market_condition_adjustment': self.config.market_condition_adjustment
        }
    
    def _generate_calibration_summary(self, 
                                    history: List[CalibrationResult]) -> Dict[str, Any]:
        """보정 요약 통계"""
        
        if not history:
            return {}
        
        # 최신 보정 결과
        latest = max(history, key=lambda x: x.timestamp)
        
        # 평균 조정량
        slippage_adjustments = [
            abs(cal.calibrated_slippage - cal.original_slippage) 
            for cal in history
        ]
        
        commission_adjustments = [
            abs(cal.calibrated_commission - cal.original_commission)
            for cal in history
        ]
        
        fill_rate_adjustments = [
            abs(cal.calibrated_fill_rate - cal.original_fill_rate)
            for cal in history
        ]
        
        return {
            'latest_calibration': latest.to_dict(),
            'average_adjustments': {
                'slippage': np.mean(slippage_adjustments) if slippage_adjustments else 0,
                'commission': np.mean(commission_adjustments) if commission_adjustments else 0,
                'fill_rate': np.mean(fill_rate_adjustments) if fill_rate_adjustments else 0
            },
            'confidence_stats': {
                'avg_confidence': np.mean([cal.confidence_score for cal in history]),
                'min_confidence': min([cal.confidence_score for cal in history]),
                'max_confidence': max([cal.confidence_score for cal in history])
            },
            'market_conditions': {
                condition: len([cal for cal in history if cal.market_condition == condition])
                for condition in set([cal.market_condition for cal in history])
            }
        }
    
    def _generate_calibration_recommendations(self,
                                            history: List[CalibrationResult]) -> List[str]:
        """보정 권고사항 생성"""
        
        recommendations = []
        
        if not history:
            return recommendations
        
        latest = max(history, key=lambda x: x.timestamp)
        
        # 신뢰도 기반 권고
        if latest.confidence_score < 0.6:
            recommendations.append(
                "보정 신뢰도가 낮습니다. 더 많은 실거래 데이터 수집이 필요합니다."
            )
        
        # 조정량 기반 권고
        if latest.calibrated_slippage > latest.original_slippage * 2:
            recommendations.append(
                "슬리피지가 크게 증가했습니다. 거래 전략이나 실행 방식을 검토하세요."
            )
        
        if latest.calibrated_fill_rate < 0.9:
            recommendations.append(
                "체결률이 낮습니다. 주문 크기나 타이밍을 조정하는 것을 고려하세요."
            )
        
        # 시장 상황 기반 권고
        if latest.market_condition in ['high_volatility', 'market_stress']:
            recommendations.append(
                f"현재 시장 상황({latest.market_condition})에서는 더 보수적인 전략을 고려하세요."
            )
        
        # 데이터 품질 기반 권고
        if latest.data_quality < 0.8:
            recommendations.append(
                "실거래 데이터 품질이 낮습니다. 로그 수집 과정을 점검하세요."
            )
        
        return recommendations
    
    def reset_calibration_history(self, symbol: str = None):
        """보정 이력 초기화"""
        
        if symbol is None:
            # 전체 초기화
            self.calibration_history.clear()
            self.last_calibration_time.clear()
            self.logger.info("전체 보정 이력 초기화")
        else:
            # 특정 심볼만 초기화
            self.calibration_history = [
                cal for cal in self.calibration_history if cal.symbol != symbol
            ]
            if symbol in self.last_calibration_time:
                del self.last_calibration_time[symbol]
            self.logger.info(f"심볼 {symbol} 보정 이력 초기화")
    
    def get_current_parameters(self, symbol: str = "ALL") -> Dict[str, float]:
        """현재 보정된 파라미터 조회"""
        
        latest = self._get_latest_calibration(symbol)
        
        if latest is None:
            # 기본값 반환
            return {
                'slippage': 0.0005,
                'commission': 0.001,
                'fill_rate': 1.0
            }
        
        return {
            'slippage': latest.calibrated_slippage,
            'commission': latest.calibrated_commission,
            'fill_rate': latest.calibrated_fill_rate
        }