#!/usr/bin/env python3
"""
실시간 하이브리드 거래 시스템
AuroraQ Production 메인 거래 엔진
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from threading import Thread, Event

# 로컬 임포트를 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .market_data import MarketDataProvider, MarketDataPoint
from .position_manager import PositionManager, TradingLimits
from ..utils.logger import get_logger

logger = get_logger("RealtimeHybridSystem")

@dataclass
class TradingConfig:
    """거래 설정"""
    # 전략 설정
    rule_strategies: List[str] = field(default_factory=lambda: ["RuleStrategyA"])
    enable_ppo: bool = True
    hybrid_mode: str = "ensemble"
    execution_strategy: str = "market"
    risk_tolerance: str = "moderate"
    
    # 실시간 설정
    update_interval_seconds: int = 60
    max_position_size: float = 0.1
    emergency_stop_loss: float = 0.05
    max_daily_trades: int = 10
    
    # 데이터 설정
    lookback_periods: int = 100
    min_data_points: int = 50
    
    # 알림 설정
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["console", "file"])

class RealtimeHybridSystem:
    """실시간 하이브리드 거래 시스템"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.is_running = False
        self.stop_event = Event()
        
        # 컴포넌트 초기화
        self.market_data_provider = MarketDataProvider("simulation")
        
        # 거래 한도 설정
        trading_limits = TradingLimits(
            max_position_size=config.max_position_size,
            max_daily_trades=config.max_daily_trades,
            emergency_stop_loss=config.emergency_stop_loss
        )
        self.position_manager = PositionManager(trading_limits)
        
        # 하이브리드 백테스트 시스템 (지연 로딩)
        self.hybrid_system = None
        
        # 데이터 버퍼
        self.price_buffer = []
        self.max_buffer_size = config.lookback_periods
        
        # 통계
        self.total_signals = 0
        self.executed_trades = 0
        self.last_update_time = None
        
        # 최적 설정 로드
        self.optimal_config = None
        self._load_optimal_configuration()
        
        # 센티멘트 분석 (선택적)
        self.sentiment_collector = None
        self.sentiment_scorer = None
    
    def _load_optimal_configuration(self):
        """최적 설정 로드"""
        try:
            # 최신 최적화 결과 찾기
            results_dir = os.path.join(parent_dir, "optimization", "results")
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.startswith("optimal_combinations_")]
                if result_files:
                    latest_file = sorted(result_files)[-1]
                    with open(os.path.join(results_dir, latest_file), 'r', encoding='utf-8') as f:
                        optimization_data = json.load(f)
                    
                    best_combo = optimization_data.get("best_combination")
                    if best_combo:
                        self.optimal_config = best_combo
                        logger.info(f"최적 설정 로드: {best_combo['hybrid_mode']}/{best_combo['execution_strategy']}")
                        logger.info(f"최적 가중치: {best_combo['weights']}")
                        
                        # 설정 업데이트
                        self.config.hybrid_mode = best_combo['hybrid_mode']
                        self.config.execution_strategy = best_combo['execution_strategy']
                    
        except Exception as e:
            logger.warning(f"최적 설정 로드 실패: {e}")
    
    def initialize(self):
        """시스템 초기화"""
        try:
            # 하이브리드 백테스트 시스템 초기화 (지연 로딩)
            self._initialize_hybrid_system()
            
            # 데이터 구독
            self.market_data_provider.subscribe(self._on_market_data)
            
            logger.info("실시간 하이브리드 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            return False
    
    def _initialize_hybrid_system(self):
        """하이브리드 시스템 초기화 (import 지연)"""
        try:
            # 필요한 모듈 동적 임포트
            sys.path.append(os.path.dirname(parent_dir))
            from backtest.complete_hybrid_backtest import CompleteHybridBacktestSystem
            
            self.hybrid_system = CompleteHybridBacktestSystem(
                rule_strategies=self.config.rule_strategies,
                enable_ppo=self.config.enable_ppo,
                hybrid_mode=self.config.hybrid_mode,
                execution_strategy=self.config.execution_strategy,
                risk_tolerance=self.config.risk_tolerance
            )
            
            logger.info("하이브리드 백테스트 시스템 로드 완료")
            
        except ImportError as e:
            logger.error(f"하이브리드 시스템 로드 실패: {e}")
            # 백테스트 시스템 없이도 기본 동작은 가능
            self.hybrid_system = None
    
    def start(self):
        """시스템 시작"""
        if not self.initialize():
            return False
        
        logger.info("=== 실시간 하이브리드 거래 시스템 시작 ===")
        logger.info(f"설정: {self.config}")
        
        self.is_running = True
        
        # 시장 데이터 스트림 시작
        self.market_data_provider.start_stream()
        
        # 메인 루프 시작
        self._main_loop()
        
        return True
    
    def stop(self):
        """시스템 중지"""
        logger.info("시스템 중지 요청됨")
        self.is_running = False
        self.stop_event.set()
        self.market_data_provider.stop_stream()
        
        # 열린 포지션 청산
        if self.position_manager.current_position != 0:
            logger.info("시스템 종료 시 포지션 강제 청산")
            if self.price_buffer:
                last_price = self.price_buffer[-1]['close']
                self.position_manager.close_position(last_price, "system_shutdown")
    
    def _main_loop(self):
        """메인 실행 루프"""
        try:
            while self.is_running and not self.stop_event.is_set():
                # 긴급 손절 체크
                if self.price_buffer and self.position_manager.current_position != 0:
                    current_price = self.price_buffer[-1]['close']
                    if self.position_manager.check_stop_loss(current_price):
                        self.position_manager.close_position(current_price, "stop_loss")
                
                # 센티멘트 분석 업데이트 (5분마다)
                if len(self.price_buffer) % 300 == 0 and self.sentiment_collector:
                    self._update_sentiment_analysis()
                
                # 상태 출력 (1분마다)
                if len(self.price_buffer) % 60 == 0 and self.price_buffer:
                    self._print_status()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 시스템 중지")
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}")
        finally:
            self.stop()
    
    def _on_market_data(self, data_point: MarketDataPoint):
        """시장 데이터 수신 처리"""
        try:
            # 데이터 버퍼에 추가
            row_data = data_point.to_dataframe_row()
            self.price_buffer.append(row_data)
            
            # 버퍼 크기 제한
            if len(self.price_buffer) > self.max_buffer_size:
                self.price_buffer.pop(0)
            
            # 충분한 데이터가 있을 때만 신호 생성
            if len(self.price_buffer) >= self.config.min_data_points:
                self._process_trading_signal(data_point)
                
        except Exception as e:
            logger.error(f"시장 데이터 처리 오류: {e}")
    
    def _process_trading_signal(self, data_point: MarketDataPoint):
        """거래 신호 처리"""
        try:
            if not self.hybrid_system:
                logger.debug("하이브리드 시스템이 로드되지 않음")
                return
            
            # DataFrame 생성
            df = pd.DataFrame(self.price_buffer)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 하이브리드 시스템으로 신호 생성
            result = self.hybrid_system.run_complete_backtest(
                df, 
                start_index=len(df)-2, 
                max_iterations=1
            )
            
            self.total_signals += 1
            
            # 결과 분석
            enhanced_results = result.get('enhanced_results', [])
            if enhanced_results:
                latest_result = enhanced_results[-1]
                
                if latest_result.get('trading_attempted', False):
                    execution_result = latest_result.get('execution_result', {})
                    
                    if execution_result.get('executed', False):
                        self._execute_trade(data_point, latest_result)
                    else:
                        logger.debug(f"거래 실행 실패: {execution_result.get('reason', '알 수 없음')}")
                
        except Exception as e:
            logger.error(f"거래 신호 처리 오류: {e}")
    
    def _execute_trade(self, data_point: MarketDataPoint, signal_result: Dict[str, Any]):
        """실제 거래 실행"""
        try:
            # 신호 정보 추출
            hybrid_summary = signal_result.get('hybrid_summary', {})
            action = hybrid_summary.get('action', 'HOLD')
            confidence = hybrid_summary.get('confidence', 0.0)
            
            if action == 'HOLD':
                return
            
            # 센티멘트 가중치 적용 (선택적)
            if self.sentiment_scorer:
                sentiment_signal = self._get_sentiment_signal()
                confidence = self._apply_sentiment_weight(confidence, sentiment_signal)
            
            # 포지션 크기 계산
            base_size = self.config.max_position_size
            adjusted_size = base_size * confidence
            
            if action == 'SELL':
                adjusted_size = -adjusted_size
            
            current_price = data_point.price
            
            # 기존 포지션이 있는 경우 청산
            if self.position_manager.current_position != 0:
                current_sign = np.sign(self.position_manager.current_position)
                signal_sign = np.sign(adjusted_size)
                
                if current_sign != signal_sign:
                    self.position_manager.close_position(current_price, "reverse_signal")
            
            # 새 포지션 개설
            signal_info = {
                'action': action,
                'confidence': confidence,
                'signal_result': signal_result,
                'strategy': 'hybrid'
            }
            
            if self.position_manager.open_position(adjusted_size, current_price, signal_info):
                self.executed_trades += 1
                logger.info(f"거래 실행: {action} {abs(adjusted_size):.4f} @ {current_price:.2f}")
                
                # 알림 발송
                self._send_notification(f"거래 실행: {action} {abs(adjusted_size):.4f} @ {current_price:.2f}")
            
        except Exception as e:
            logger.error(f"거래 실행 오류: {e}")
    
    def _update_sentiment_analysis(self):
        """센티멘트 분석 업데이트"""
        try:
            if not self.sentiment_collector:
                return
            
            # 뉴스 수집
            news_items = self.sentiment_collector.collect_crypto_news(hours_back=6)
            
            if news_items and self.sentiment_scorer:
                # 센티멘트 점수 계산
                sentiment_score = self.sentiment_scorer.calculate_market_sentiment(news_items, "crypto")
                logger.info(f"센티멘트 업데이트: {sentiment_score.sentiment_label.value} "
                           f"(점수: {sentiment_score.overall_score:.3f})")
        
        except Exception as e:
            logger.error(f"센티멘트 분석 오류: {e}")
    
    def _get_sentiment_signal(self) -> Dict[str, Any]:
        """센티멘트 신호 조회"""
        if not self.sentiment_scorer or not self.sentiment_scorer.sentiment_history:
            return {"signal": "HOLD", "confidence": 0.5}
        
        latest_sentiment = self.sentiment_scorer.sentiment_history[-1]
        return self.sentiment_scorer.get_trading_signal(latest_sentiment)
    
    def _apply_sentiment_weight(self, base_confidence: float, sentiment_signal: Dict[str, Any]) -> float:
        """센티멘트 가중치 적용"""
        sentiment_confidence = sentiment_signal.get('confidence', 0.5)
        sentiment_direction = sentiment_signal.get('signal', 'HOLD')
        
        # 센티멘트 방향이 일치하면 신뢰도 증가, 반대면 감소
        if sentiment_direction in ['BUY', 'STRONG_BUY']:
            return min(1.0, base_confidence * (1 + sentiment_confidence * 0.2))
        elif sentiment_direction in ['SELL', 'STRONG_SELL']:
            return min(1.0, base_confidence * (1 + sentiment_confidence * 0.2))
        else:
            return base_confidence * 0.9  # 중립적 센티멘트는 약간 감소
    
    def _print_status(self):
        """시스템 상태 출력"""
        if not self.price_buffer:
            return
        
        current_price = self.price_buffer[-1]['close']
        current_time = datetime.now()
        
        # 포지션 정보
        position_info = "No Position"
        if self.position_manager.current_position != 0:
            pnl_pct = self.position_manager.get_current_pnl_pct(current_price)
            position_info = f"Position: {self.position_manager.current_position:.4f} @ {self.position_manager.entry_price:.2f} (PnL: {pnl_pct:.2%})"
        
        # 통계
        signal_rate = self.executed_trades / max(self.total_signals, 1) * 100
        
        status = f"""
=== 실시간 거래 시스템 상태 ({current_time.strftime('%H:%M:%S')}) ===
현재 가격: {current_price:.2f}
{position_info}
총 신호: {self.total_signals}, 실행된 거래: {self.executed_trades} (실행률: {signal_rate:.1f}%)
일일 거래 수: {self.position_manager.daily_trade_count}/{self.config.max_daily_trades}
데이터 버퍼: {len(self.price_buffer)}/{self.max_buffer_size}
"""
        logger.info(status)
    
    def _send_notification(self, message: str):
        """알림 발송"""
        if not self.config.enable_notifications:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        for channel in self.config.notification_channels:
            if channel == "console":
                print(f"🔔 {formatted_message}")
            elif channel == "file":
                log_path = os.path.join(parent_dir, "logs", "notifications.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{formatted_message}\n")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성과 리포트 생성"""
        position_history = self.position_manager.position_history
        total_trades = len([p for p in position_history if p['action'] == 'close'])
        
        if total_trades == 0:
            return {
                "total_signals": self.total_signals,
                "executed_trades": self.executed_trades,
                "signal_execution_rate": self.executed_trades / max(self.total_signals, 1),
                "total_completed_trades": 0,
                "message": "아직 완료된 거래가 없습니다"
            }
        
        # 성과 계산
        performance_summary = self.position_manager.get_performance_summary()
        
        return {
            "total_signals": self.total_signals,
            "executed_trades": self.executed_trades,
            "signal_execution_rate": self.executed_trades / max(self.total_signals, 1),
            "current_position": self.position_manager.current_position,
            "daily_trade_count": self.position_manager.daily_trade_count,
            "position_history": position_history,
            **performance_summary
        }