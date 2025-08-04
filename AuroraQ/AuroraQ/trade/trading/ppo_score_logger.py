#!/usr/bin/env python3
"""
PPO 전략 점수 로깅 시스템
전략 점수, 선택률, 성과를 .csv/.json 형태로 자동 저장
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
class PPOScoreRecord:
    """PPO 점수 기록"""
    timestamp: str
    strategy_score: float
    confidence: float
    action: str
    market_outcome: Optional[float] = None
    final_reward: Optional[float] = None
    selected: bool = False
    selection_rank: int = 0
    total_predictions: int = 0
    success_rate: float = 0.0
    
class PPOScoreLogger:
    """PPO 전략 점수 로깅 시스템"""
    
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
        self.csv_file = self.summary_dir / f"ppo_scores_{today}.csv"
        self.json_file = self.metrics_dir / f"ppo_metrics_{today}.json"
        self.daily_summary_file = self.summary_dir / f"ppo_daily_summary_{today}.json"
        
        # CSV 헤더 초기화
        self._init_csv_file()
        
        # 메모리 버퍼
        self.score_buffer: List[PPOScoreRecord] = []
        self.buffer_limit = 100
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
    def _init_csv_file(self) -> None:
        """CSV 파일 초기화 (헤더 생성)"""
        if not self.csv_file.exists():
            headers = [
                'timestamp', 'strategy_score', 'confidence', 'action',
                'market_outcome', 'final_reward', 'selected', 'selection_rank',
                'total_predictions', 'success_rate'
            ]
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_score(self, strategy_score: float, confidence: float, action: str,
                  market_outcome: Optional[float] = None, final_reward: Optional[float] = None,
                  selected: bool = False, selection_rank: int = 0,
                  total_predictions: int = 0, success_rate: float = 0.0) -> None:
        """PPO 점수 기록 추가"""
        try:
            record = PPOScoreRecord(
                timestamp=datetime.now().isoformat(),
                strategy_score=strategy_score,
                confidence=confidence,
                action=action,
                market_outcome=market_outcome,
                final_reward=final_reward,
                selected=selected,
                selection_rank=selection_rank,
                total_predictions=total_predictions,
                success_rate=success_rate
            )
            
            # 버퍼에 추가
            self.score_buffer.append(record)
            
            # 즉시 CSV에 기록
            self._write_to_csv(record)
            
            # 버퍼가 가득 차면 JSON 업데이트
            if len(self.score_buffer) >= self.buffer_limit:
                self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"PPO 점수 기록 실패: {e}")
    
    def _write_to_csv(self, record: PPOScoreRecord) -> None:
        """CSV 파일에 기록 추가"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.timestamp,
                    record.strategy_score,
                    record.confidence,
                    record.action,
                    record.market_outcome,
                    record.final_reward,
                    record.selected,
                    record.selection_rank,
                    record.total_predictions,
                    record.success_rate
                ])
        except Exception as e:
            self.logger.error(f"CSV 기록 실패: {e}")
    
    def _flush_buffer(self) -> None:
        """버퍼를 JSON 파일로 플러시"""
        try:
            if not self.score_buffer:
                return
            
            # 기존 JSON 데이터 로드
            existing_data = []
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 새 데이터 추가
            new_data = [asdict(record) for record in self.score_buffer]
            existing_data.extend(new_data)
            
            # JSON 파일 업데이트
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            # 버퍼 클리어
            self.score_buffer.clear()
            
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
            
            # 통계 계산
            summary = {
                'date': today,
                'total_predictions': len(today_data),
                'selected_count': int(today_data['selected'].sum()),
                'selection_rate': float(today_data['selected'].mean()),
                'avg_strategy_score': float(today_data['strategy_score'].mean()),
                'avg_confidence': float(today_data['confidence'].mean()),
                'action_distribution': today_data['action'].value_counts().to_dict(),
                'avg_market_outcome': float(today_data['market_outcome'].dropna().mean()) if not today_data['market_outcome'].dropna().empty else 0.0,
                'avg_final_reward': float(today_data['final_reward'].dropna().mean()) if not today_data['final_reward'].dropna().empty else 0.0,
                'success_rate_trend': today_data['success_rate'].tolist()[-10:],  # 최근 10개
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
    
    def get_recent_scores(self, limit: int = 50) -> List[Dict[str, Any]]:
        """최근 점수 기록 조회"""
        try:
            if self.json_file.exists():
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data[-limit:] if data else []
            return []
        except Exception as e:
            self.logger.error(f"최근 점수 조회 실패: {e}")
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """오래된 로그 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # CSV 파일 정리
            for csv_file in self.summary_dir.glob("ppo_scores_*.csv"):
                file_date_str = csv_file.stem.split('_')[-1]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        csv_file.unlink()
                        self.logger.info(f"오래된 CSV 파일 삭제: {csv_file}")
                except ValueError:
                    continue
            
            # JSON 파일 정리
            for json_file in self.metrics_dir.glob("ppo_metrics_*.json"):
                file_date_str = json_file.stem.split('_')[-1]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        json_file.unlink()
                        self.logger.info(f"오래된 JSON 파일 삭제: {json_file}")
                except ValueError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"로그 정리 실패: {e}")
    
    def export_weekly_report(self) -> Dict[str, Any]:
        """주간 리포트 생성"""
        try:
            # 최근 7일 데이터 수집
            week_ago = datetime.now() - timedelta(days=7)
            
            all_data = []
            for json_file in self.metrics_dir.glob("ppo_metrics_*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                except Exception:
                    continue
            
            if not all_data:
                return {}
            
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 최근 7일 데이터 필터링
            recent_data = df[df['timestamp'] >= week_ago]
            
            if len(recent_data) == 0:
                return {}
            
            # 주간 통계 계산
            report = {
                'period': {
                    'start': week_ago.strftime("%Y-%m-%d"),
                    'end': datetime.now().strftime("%Y-%m-%d")
                },
                'total_predictions': len(recent_data),
                'total_selections': int(recent_data['selected'].sum()),
                'overall_selection_rate': float(recent_data['selected'].mean()),
                'avg_strategy_score': float(recent_data['strategy_score'].mean()),
                'avg_confidence': float(recent_data['confidence'].mean()),
                'action_distribution': recent_data['action'].value_counts().to_dict(),
                'daily_breakdown': {},
                'performance_metrics': {
                    'avg_market_outcome': float(recent_data['market_outcome'].dropna().mean()) if not recent_data['market_outcome'].dropna().empty else 0.0,
                    'avg_final_reward': float(recent_data['final_reward'].dropna().mean()) if not recent_data['final_reward'].dropna().empty else 0.0,
                    'positive_outcomes': int((recent_data['market_outcome'] > 0).sum()),
                    'negative_outcomes': int((recent_data['market_outcome'] < 0).sum())
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # 일별 분석
            daily_stats = recent_data.groupby(recent_data['timestamp'].dt.date).agg({
                'strategy_score': 'mean',
                'confidence': 'mean',
                'selected': ['sum', 'count'],
                'market_outcome': 'mean'
            }).round(3)
            
            for date, stats in daily_stats.iterrows():
                report['daily_breakdown'][str(date)] = {
                    'avg_score': float(stats[('strategy_score', 'mean')]),
                    'avg_confidence': float(stats[('confidence', 'mean')]),
                    'selections': int(stats[('selected', 'sum')]),
                    'total_predictions': int(stats[('selected', 'count')]),
                    'selection_rate': float(stats[('selected', 'sum')] / stats[('selected', 'count')]),
                    'avg_outcome': float(stats[('market_outcome', 'mean')]) if not pd.isna(stats[('market_outcome', 'mean')]) else 0.0
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"주간 리포트 생성 실패: {e}")
            return {}
    
    def __del__(self):
        """소멸자: 버퍼 플러시"""
        try:
            if hasattr(self, 'score_buffer') and self.score_buffer:
                self._flush_buffer()
        except Exception:
            pass

# 전역 로거 인스턴스
_ppo_logger_instance = None

def get_ppo_score_logger() -> PPOScoreLogger:
    """PPO 점수 로거 싱글톤 인스턴스 반환"""
    global _ppo_logger_instance
    if _ppo_logger_instance is None:
        _ppo_logger_instance = PPOScoreLogger()
    return _ppo_logger_instance