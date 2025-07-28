#!/usr/bin/env python3
"""
실거래 모니터링 시스템
실시간으로 거래 데이터를 수집하고 체결 특성을 추적
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
import threading
from queue import Queue


@dataclass
class TradeRecord:
    """거래 기록"""
    timestamp: datetime
    trade_id: str
    symbol: str
    side: str  # buy, sell
    order_type: str  # market, limit
    
    # 주문 정보
    order_size: float
    order_price: float
    order_time: datetime
    
    # 체결 정보
    executed_size: float
    executed_price: float
    execution_time: datetime
    
    # 비용 정보
    commission: float
    slippage: float
    market_impact: float
    
    # 상태 정보
    status: str  # executed, partial, cancelled
    latency_ms: float
    
    # 시장 정보
    market_condition: str
    volatility: float
    spread: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'order': {
                'size': self.order_size,
                'price': self.order_price,
                'time': self.order_time.isoformat()
            },
            'execution': {
                'size': self.executed_size,
                'price': self.executed_price,
                'time': self.execution_time.isoformat()
            },
            'costs': {
                'commission': self.commission,
                'slippage': self.slippage,
                'market_impact': self.market_impact
            },
            'status': {
                'status': self.status,
                'latency_ms': self.latency_ms
            },
            'market': {
                'condition': self.market_condition,
                'volatility': self.volatility,
                'spread': self.spread,
                'volume': self.volume
            }
        }


@dataclass
class MonitoringStats:
    """모니터링 통계"""
    start_time: datetime = field(default_factory=datetime.now)
    
    # 거래 통계
    total_trades: int = 0
    successful_trades: int = 0
    partial_fills: int = 0
    cancelled_trades: int = 0
    
    # 체결 통계
    avg_fill_rate: float = 1.0
    avg_execution_time: float = 0.0
    avg_slippage: float = 0.0
    avg_commission_rate: float = 0.0
    
    # 시장 임팩트
    avg_market_impact: float = 0.0
    
    # 최근 성과
    recent_fill_rate: float = 1.0
    recent_slippage: float = 0.0
    
    # 데이터 품질
    data_collection_rate: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_stats(self, trades: List[TradeRecord]):
        """통계 업데이트"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        self.successful_trades = len([t for t in trades if t.status == 'executed'])
        self.partial_fills = len([t for t in trades if t.status == 'partial'])
        self.cancelled_trades = len([t for t in trades if t.status == 'cancelled'])
        
        executed_trades = [t for t in trades if t.status in ['executed', 'partial']]
        
        if executed_trades:
            self.avg_fill_rate = np.mean([t.executed_size / t.order_size for t in executed_trades])
            self.avg_execution_time = np.mean([t.latency_ms for t in executed_trades])
            self.avg_slippage = np.mean([t.slippage for t in executed_trades])
            
            # 수수료율 계산
            commission_rates = [
                t.commission / (t.executed_size * t.executed_price) 
                for t in executed_trades 
                if t.executed_size * t.executed_price > 0
            ]
            self.avg_commission_rate = np.mean(commission_rates) if commission_rates else 0
            
            self.avg_market_impact = np.mean([t.market_impact for t in executed_trades])
            
            # 최근 성과 (최근 20% 거래)
            recent_count = max(1, len(executed_trades) // 5)
            recent_trades = executed_trades[-recent_count:]
            
            self.recent_fill_rate = np.mean([t.executed_size / t.order_size for t in recent_trades])
            self.recent_slippage = np.mean([t.slippage for t in recent_trades])
        
        self.last_update = datetime.now()


class RealTradeMonitor:
    """실거래 모니터링 시스템"""
    
    def __init__(self, 
                 log_directory: str = "execution_logs",
                 buffer_size: int = 1000):
        
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)
        
        # 데이터 저장
        self.trade_records: List[TradeRecord] = []
        self.trade_buffer = Queue(maxsize=buffer_size)
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_stats = MonitoringStats()
        
        # 콜백 시스템
        self.trade_callbacks: List[Callable] = []
        self.stats_callbacks: List[Callable] = []
        
        # 비동기 작업
        self.monitoring_task: Optional[asyncio.Task] = None
        self.logging_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 설정
        self.log_flush_interval = 60  # 60초마다 로그 플러시
        self.stats_update_interval = 30  # 30초마다 통계 업데이트
        self.max_records_memory = 10000  # 메모리 최대 기록 수
        
    def add_trade_callback(self, callback: Callable[[TradeRecord], None]):
        """거래 콜백 추가"""
        self.trade_callbacks.append(callback)
    
    def add_stats_callback(self, callback: Callable[[MonitoringStats], None]):
        """통계 콜백 추가"""
        self.stats_callbacks.append(callback)
    
    def record_trade(self, trade_record: TradeRecord):
        """거래 기록"""
        
        # 버퍼에 추가
        try:
            self.trade_buffer.put_nowait(trade_record)
        except:
            # 버퍼가 가득 찬 경우 가장 오래된 기록 제거
            try:
                self.trade_buffer.get_nowait()
                self.trade_buffer.put_nowait(trade_record)
            except:
                pass
        
        # 메모리에 추가
        self.trade_records.append(trade_record)
        
        # 메모리 관리
        if len(self.trade_records) > self.max_records_memory:
            self.trade_records = self.trade_records[-self.max_records_memory//2:]
        
        # 콜백 실행
        self._trigger_trade_callbacks(trade_record)
        
        # 통계 업데이트 (최근 거래 기준)
        if len(self.trade_records) % 10 == 0:  # 10건마다 업데이트
            self._update_monitoring_stats()
    
    def record_order_execution(self,
                             trade_id: str,
                             symbol: str,
                             side: str,
                             order_size: float,
                             order_price: float,
                             executed_size: float,
                             executed_price: float,
                             commission: float,
                             order_time: datetime = None,
                             execution_time: datetime = None) -> TradeRecord:
        """주문 체결 기록 (간편 인터페이스)"""
        
        if order_time is None:
            order_time = datetime.now() - timedelta(seconds=1)
        if execution_time is None:
            execution_time = datetime.now()
        
        # 슬리피지 계산
        if side.lower() == 'buy':
            slippage = (executed_price - order_price) / order_price if order_price > 0 else 0
        else:
            slippage = (order_price - executed_price) / order_price if order_price > 0 else 0
        
        # 상태 결정
        if executed_size >= order_size * 0.99:  # 99% 이상 체결
            status = 'executed'
        elif executed_size > 0:
            status = 'partial'
        else:
            status = 'cancelled'
        
        # 레이턴시 계산
        latency_ms = (execution_time - order_time).total_seconds() * 1000
        
        # 시장 정보 (시뮬레이션)
        market_condition = self._detect_market_condition(symbol)
        volatility = np.random.uniform(0.15, 0.35)
        spread = np.random.uniform(0.0005, 0.003)
        volume = np.random.uniform(500000, 2000000)
        
        # 시장 임팩트 추정
        market_impact = self._estimate_market_impact(order_size, executed_price, volume)
        
        trade_record = TradeRecord(
            timestamp=execution_time,
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            order_type='market',  # 기본값
            order_size=order_size,
            order_price=order_price,
            order_time=order_time,
            executed_size=executed_size,
            executed_price=executed_price,
            execution_time=execution_time,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            status=status,
            latency_ms=latency_ms,
            market_condition=market_condition,
            volatility=volatility,
            spread=spread,
            volume=volume
        )
        
        self.record_trade(trade_record)
        return trade_record
    
    def _detect_market_condition(self, symbol: str) -> str:
        """시장 상황 감지 (간단한 구현)"""
        # 실제로는 MarketConditionDetector를 사용
        conditions = ['normal', 'high_volatility', 'low_liquidity', 'after_hours']
        weights = [0.7, 0.15, 0.1, 0.05]
        return np.random.choice(conditions, p=weights)
    
    def _estimate_market_impact(self, order_size: float, price: float, volume: float) -> float:
        """시장 임팩트 추정"""
        # 간단한 시장 임팩트 모델
        order_value = order_size * price
        participation_rate = order_value / (volume * price)
        
        # 참여율에 비례하는 임팩트
        impact = participation_rate * 0.01  # 1% 최대 임팩트
        return min(impact, 0.005)  # 최대 0.5% 제한
    
    async def start_monitoring(self):
        """모니터링 시작"""
        
        if self.is_monitoring:
            self.logger.warning("모니터링이 이미 실행 중입니다.")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # 로깅 스레드 시작
        self.logging_thread = threading.Thread(target=self._logging_worker)
        self.logging_thread.start()
        
        # 비동기 모니터링 태스크 시작
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("실거래 모니터링 시작")
    
    async def stop_monitoring(self):
        """모니터링 중단"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        # 로깅 스레드 종료
        if self.logging_thread:
            self.logging_thread.join(timeout=5)
        
        # 비동기 태스크 종료
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 남은 데이터 플러시
        self._flush_logs()
        
        self.logger.info("실거래 모니터링 중단")
    
    def _logging_worker(self):
        """로깅 워커 스레드"""
        
        while not self.stop_event.is_set():
            try:
                # 정기적으로 로그 플러시
                self._flush_logs()
                
                # 대기
                if self.stop_event.wait(self.log_flush_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"로깅 워커 오류: {e}")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        
        try:
            while self.is_monitoring:
                # 정기적으로 통계 업데이트
                self._update_monitoring_stats()
                
                # 대기
                await asyncio.sleep(self.stats_update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("모니터링 루프 종료")
        except Exception as e:
            self.logger.error(f"모니터링 루프 오류: {e}")
    
    def _flush_logs(self):
        """로그 파일에 기록 플러시"""
        
        # 버퍼에서 모든 기록 추출
        records_to_log = []
        
        while not self.trade_buffer.empty():
            try:
                record = self.trade_buffer.get_nowait()
                records_to_log.append(record)
            except:
                break
        
        if not records_to_log:
            return
        
        # 파일에 기록
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_directory / f"execution_monitor_{timestamp}.json"
        
        try:
            # 기존 데이터 로드
            existing_data = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            
            # 새 데이터 추가
            new_data = [record.to_dict() for record in records_to_log]
            all_data = existing_data + new_data
            
            # 파일에 저장
            with open(log_file, 'w') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"{len(records_to_log)}개 거래 기록을 {log_file}에 저장")
            
        except Exception as e:
            self.logger.error(f"로그 파일 저장 실패: {e}")
    
    def _update_monitoring_stats(self):
        """모니터링 통계 업데이트"""
        
        # 최근 1시간 거래 기준으로 통계 계산
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_trades = [
            trade for trade in self.trade_records 
            if trade.timestamp >= cutoff_time
        ]
        
        self.monitoring_stats.update_stats(recent_trades)
        
        # 통계 콜백 실행
        self._trigger_stats_callbacks(self.monitoring_stats)
    
    def _trigger_trade_callbacks(self, trade_record: TradeRecord):
        """거래 콜백 실행"""
        
        for callback in self.trade_callbacks:
            try:
                callback(trade_record)
            except Exception as e:
                self.logger.error(f"거래 콜백 실행 실패: {e}")
    
    def _trigger_stats_callbacks(self, stats: MonitoringStats):
        """통계 콜백 실행"""
        
        for callback in self.stats_callbacks:
            try:
                callback(stats)
            except Exception as e:
                self.logger.error(f"통계 콜백 실행 실패: {e}")
    
    def get_recent_trades(self, 
                         symbol: str = None,
                         hours: int = 1) -> List[TradeRecord]:
        """최근 거래 조회"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_trades = [
            trade for trade in self.trade_records 
            if trade.timestamp >= cutoff_time
        ]
        
        if symbol:
            recent_trades = [
                trade for trade in recent_trades 
                if trade.symbol == symbol
            ]
        
        return recent_trades
    
    def get_monitoring_stats(self) -> MonitoringStats:
        """현재 모니터링 통계 조회"""
        return self.monitoring_stats
    
    def get_execution_summary(self, 
                            symbol: str = None,
                            hours: int = 24) -> Dict[str, Any]:
        """체결 요약 통계"""
        
        recent_trades = self.get_recent_trades(symbol, hours)
        
        if not recent_trades:
            return {}
        
        executed_trades = [t for t in recent_trades if t.status in ['executed', 'partial']]
        
        summary = {
            'period_hours': hours,
            'symbol': symbol or 'ALL',
            'total_trades': len(recent_trades),
            'executed_trades': len(executed_trades),
            'success_rate': len(executed_trades) / len(recent_trades) if recent_trades else 0,
        }
        
        if executed_trades:
            # 체결 통계
            fill_rates = [t.executed_size / t.order_size for t in executed_trades]
            slippages = [t.slippage for t in executed_trades]
            latencies = [t.latency_ms for t in executed_trades]
            
            summary.update({
                'execution_stats': {
                    'avg_fill_rate': np.mean(fill_rates),
                    'median_fill_rate': np.median(fill_rates),
                    'avg_slippage': np.mean(slippages),
                    'median_slippage': np.median(slippages),
                    'slippage_95th': np.percentile(slippages, 95),
                    'avg_latency_ms': np.mean(latencies),
                    'latency_95th_ms': np.percentile(latencies, 95)
                },
                'cost_analysis': {
                    'total_commission': sum(t.commission for t in executed_trades),
                    'avg_commission_rate': np.mean([
                        t.commission / (t.executed_size * t.executed_price)
                        for t in executed_trades 
                        if t.executed_size * t.executed_price > 0
                    ]),
                    'total_market_impact': sum(t.market_impact for t in executed_trades),
                    'avg_market_impact': np.mean([t.market_impact for t in executed_trades])
                }
            })
            
            # 시장 상황별 분석
            condition_groups = {}
            for trade in executed_trades:
                condition = trade.market_condition
                if condition not in condition_groups:
                    condition_groups[condition] = []
                condition_groups[condition].append(trade)
            
            condition_stats = {}
            for condition, trades in condition_groups.items():
                condition_stats[condition] = {
                    'count': len(trades),
                    'avg_fill_rate': np.mean([t.executed_size / t.order_size for t in trades]),
                    'avg_slippage': np.mean([t.slippage for t in trades])
                }
            
            summary['market_condition_analysis'] = condition_stats
        
        return summary
    
    def export_execution_log(self,
                           symbol: str = None,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           output_path: str = None) -> str:
        """체결 로그 내보내기"""
        
        # 필터링
        trades_to_export = self.trade_records.copy()
        
        if symbol:
            trades_to_export = [t for t in trades_to_export if t.symbol == symbol]
        
        if start_date:
            trades_to_export = [t for t in trades_to_export if t.timestamp >= start_date]
        
        if end_date:
            trades_to_export = [t for t in trades_to_export if t.timestamp <= end_date]
        
        if not trades_to_export:
            self.logger.warning("내보낼 거래 데이터가 없습니다.")
            return ""
        
        # 출력 파일명 생성
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol_suffix = f"_{symbol}" if symbol else ""
            output_path = f"execution_export{symbol_suffix}_{timestamp}.json"
        
        # 데이터 내보내기
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'symbol_filter': symbol,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'total_records': len(trades_to_export)
            },
            'trades': [trade.to_dict() for trade in trades_to_export],
            'summary': self.get_execution_summary(symbol, 
                                                 hours=(end_date - start_date).total_seconds() / 3600 
                                                 if start_date and end_date else 24)
        }
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"체결 로그 내보내기 완료: {output_path}")
        
        return output_path
    
    def clear_old_records(self, days: int = 7):
        """오래된 기록 정리"""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.trade_records)
        self.trade_records = [
            trade for trade in self.trade_records 
            if trade.timestamp >= cutoff_time
        ]
        
        cleaned_count = original_count - len(self.trade_records)
        
        self.logger.info(f"오래된 거래 기록 {cleaned_count}개 정리 완료")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 조회"""
        
        return {
            'is_monitoring': self.is_monitoring,
            'total_records': len(self.trade_records),
            'buffer_size': self.trade_buffer.qsize(),
            'last_update': self.monitoring_stats.last_update.isoformat(),
            'monitoring_stats': {
                'total_trades': self.monitoring_stats.total_trades,
                'avg_fill_rate': self.monitoring_stats.avg_fill_rate,
                'avg_slippage': self.monitoring_stats.avg_slippage,
                'avg_commission_rate': self.monitoring_stats.avg_commission_rate,
                'data_collection_rate': self.monitoring_stats.data_collection_rate
            }
        }