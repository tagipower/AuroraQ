#!/usr/bin/env python3
"""
Rule 전략 로깅 시스템
RuleStrategyA~E의 점수, 선택률, 성과를 .csv/.json 형태로 자동 저장
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

@dataclass
class RuleStrategyRecord:
    """Rule 전략 기록"""
    timestamp: str
    strategy_name: str
    action: str
    strength: float
    strategy_score: float
    confidence: float
    market_outcome: Optional[float] = None
    selected: bool = False
    selection_rank: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    
class RuleStrategyLogger:
    """Rule 전략 로깅 시스템"""
    
    def __init__(self, log_base_dir: str = None):
        # 로그 디렉토리 설정
        if log_base_dir is None:
            log_base_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "logs"
            )
        
        self.log_dir = Path(log_base_dir)
        self.summary_dir = self.log_dir / "summary_logs"
        self.metrics_dir = self.log_dir / "metrics"
        
        # 디렉토리 생성
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로 설정
        today = datetime.now().strftime("%Y%m%d")
        self.csv_file = self.summary_dir / f"rule_strategies_{today}.csv"
        self.json_file = self.metrics_dir / f"rule_metrics_{today}.json"
        self.daily_summary_file = self.summary_dir / f"rule_daily_summary_{today}.json"
        
        # CSV 헤더 초기화
        self._init_csv_file()
        
        # 메모리 버퍼
        self.record_buffer: List[RuleStrategyRecord] = []
        self.buffer_limit = 50
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
    def _init_csv_file(self) -> None:
        """CSV 파일 초기화 (헤더 생성)"""
        if not self.csv_file.exists():
            headers = [
                'timestamp', 'strategy_name', 'action', 'strength', 'strategy_score',
                'confidence', 'market_outcome', 'selected', 'selection_rank',
                'total_trades', 'win_rate', 'profit_factor', 'total_pnl'
            ]
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_strategy_signal(self, strategy_name: str, action: str, strength: float,
                           strategy_score: float, confidence: float,
                           market_outcome: Optional[float] = None, selected: bool = False,
                           selection_rank: int = 0, total_trades: int = 0,
                           win_rate: float = 0.0, profit_factor: float = 0.0,
                           total_pnl: float = 0.0) -> None:
        """Rule 전략 신호 기록 추가"""
        try:
            record = RuleStrategyRecord(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                action=action,
                strength=strength,
                strategy_score=strategy_score,
                confidence=confidence,
                market_outcome=market_outcome,
                selected=selected,
                selection_rank=selection_rank,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_pnl=total_pnl
            )
            
            # 버퍼에 추가
            self.record_buffer.append(record)
            
            # 즉시 CSV에 기록
            self._write_to_csv(record)
            
            # 버퍼가 가득 차면 JSON 업데이트
            if len(self.record_buffer) >= self.buffer_limit:
                self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"Rule 전략 기록 실패: {e}")
    
    def _write_to_csv(self, record: RuleStrategyRecord) -> None:
        """CSV 파일에 기록 추가"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.timestamp,
                    record.strategy_name,
                    record.action,
                    record.strength,
                    record.strategy_score,
                    record.confidence,
                    record.market_outcome,
                    record.selected,
                    record.selection_rank,
                    record.total_trades,
                    record.win_rate,
                    record.profit_factor,
                    record.total_pnl
                ])
        except Exception as e:
            self.logger.error(f"CSV 기록 실패: {e}")
    
    def _flush_buffer(self) -> None:
        """버퍼를 JSON 파일로 플러시"""
        try:
            if not self.record_buffer:
                return
            
            # 기존 JSON 데이터 로드
            existing_data = []
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 새 데이터 추가
            new_data = [asdict(record) for record in self.record_buffer]
            existing_data.extend(new_data)
            
            # JSON 파일 업데이트
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            # 버퍼 클리어
            self.record_buffer.clear()
            
            # 일일 요약 업데이트
            self._update_daily_summary()
            
        except Exception as e:
            self.logger.error(f"JSON 플러시 실패: {e}")
    
    def _update_daily_summary(self) -> None:
        """일일 요약 통계 업데이트"""
        try:
            if not self.json_file.exists():
                return
            
            # 데이터 로드
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return
            
            df = pd.DataFrame(data)
            
            # 오늘 데이터만 필터링
            today = datetime.now().strftime("%Y-%m-%d")
            df['date'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d")
            today_data = df[df['date'] == today]
            
            if len(today_data) == 0:
                return
            
            # 전략별 통계 계산
            strategy_stats = {}
            for strategy in today_data['strategy_name'].unique():
                strategy_data = today_data[today_data['strategy_name'] == strategy]
                
                strategy_stats[strategy] = {
                    'total_signals': len(strategy_data),
                    'selected_count': int(strategy_data['selected'].sum()),
                    'selection_rate': float(strategy_data['selected'].mean()),
                    'avg_strategy_score': float(strategy_data['strategy_score'].mean()),
                    'avg_confidence': float(strategy_data['confidence'].mean()),
                    'avg_strength': float(strategy_data['strength'].mean()),
                    'action_distribution': strategy_data['action'].value_counts().to_dict(),
                    'avg_market_outcome': float(strategy_data['market_outcome'].dropna().mean()) if not strategy_data['market_outcome'].dropna().empty else 0.0,
                    'latest_total_trades': int(strategy_data['total_trades'].iloc[-1]) if len(strategy_data) > 0 else 0,
                    'latest_win_rate': float(strategy_data['win_rate'].iloc[-1]) if len(strategy_data) > 0 else 0.0,
                    'latest_profit_factor': float(strategy_data['profit_factor'].iloc[-1]) if len(strategy_data) > 0 else 0.0,
                    'latest_total_pnl': float(strategy_data['total_pnl'].iloc[-1]) if len(strategy_data) > 0 else 0.0
                }
            
            # 전체 통계
            summary = {
                'date': today,
                'total_signals': len(today_data),
                'total_selections': int(today_data['selected'].sum()),
                'overall_selection_rate': float(today_data['selected'].mean()),
                'strategy_count': len(today_data['strategy_name'].unique()),
                'strategies': strategy_stats,
                'top_performer': max(strategy_stats.keys(), key=lambda x: strategy_stats[x]['avg_strategy_score']) if strategy_stats else None,
                'most_selected': max(strategy_stats.keys(), key=lambda x: strategy_stats[x]['selection_rate']) if strategy_stats else None,
                'last_updated': datetime.now().isoformat()
            }
            
            # 일일 요약 저장
            with open(self.daily_summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"일일 요약 업데이트 실패: {e}")
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """일일 요약 통계 조회"""
        try:
            if self.daily_summary_file.exists():
                with open(self.daily_summary_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"일일 요약 조회 실패: {e}")
            return {}
    
    def get_strategy_performance(self, strategy_name: str, days: int = 7) -> Dict[str, Any]:
        """특정 전략의 성과 조회"""
        try:
            # 최근 N일간의 데이터 수집
            cutoff_date = datetime.now() - timedelta(days=days)
            
            all_data = []
            for json_file in self.metrics_dir.glob("rule_metrics_*.json"):
                try:
                    file_date_str = json_file.stem.split('_')[-1]
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date >= cutoff_date:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            all_data.extend(data)
                except Exception:
                    continue
            
            if not all_data:
                return {}
            
            df = pd.DataFrame(all_data)
            strategy_data = df[df['strategy_name'] == strategy_name]
            
            if len(strategy_data) == 0:
                return {}
            
            return {
                'strategy_name': strategy_name,
                'period_days': days,
                'total_signals': len(strategy_data),
                'selections': int(strategy_data['selected'].sum()),
                'selection_rate': float(strategy_data['selected'].mean()),
                'avg_score': float(strategy_data['strategy_score'].mean()),
                'avg_confidence': float(strategy_data['confidence'].mean()),
                'avg_strength': float(strategy_data['strength'].mean()),
                'action_distribution': strategy_data['action'].value_counts().to_dict(),
                'performance_trend': {
                    'win_rate_trend': strategy_data['win_rate'].tolist()[-10:],
                    'profit_factor_trend': strategy_data['profit_factor'].tolist()[-10:],
                    'score_trend': strategy_data['strategy_score'].tolist()[-10:]
                }
            }
            
        except Exception as e:
            self.logger.error(f"전략 성과 조회 실패: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """오래된 로그 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # CSV 파일 정리
            for csv_file in self.summary_dir.glob("rule_strategies_*.csv"):
                file_date_str = csv_file.stem.split('_')[-1]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        csv_file.unlink()
                        self.logger.info(f"오래된 Rule CSV 파일 삭제: {csv_file}")
                except ValueError:
                    continue
            
            # JSON 파일 정리
            for json_file in self.metrics_dir.glob("rule_metrics_*.json"):
                file_date_str = json_file.stem.split('_')[-1]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        json_file.unlink()
                        self.logger.info(f"오래된 Rule JSON 파일 삭제: {json_file}")
                except ValueError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Rule 로그 정리 실패: {e}")
    
    def __del__(self):
        """소멸자: 버퍼 플러시"""
        try:
            if hasattr(self, 'record_buffer') and self.record_buffer:
                self._flush_buffer()
        except Exception:
            pass

# 전역 로거 인스턴스
_rule_logger_instance = None

def get_rule_strategy_logger() -> RuleStrategyLogger:
    """Rule 전략 로거 싱글톤 인스턴스 반환"""
    global _rule_logger_instance
    if _rule_logger_instance is None:
        _rule_logger_instance = RuleStrategyLogger()
    return _rule_logger_instance