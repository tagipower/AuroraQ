#!/usr/bin/env python3
"""
모델 품질 모니터링 시스템
P4: 모델 품질 모니터링 및 Fine-tuning 시스템 구축
"""

import sys
import os
import json
import time
import logging
import asyncio
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import threading
import warnings

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """모델 타입"""
    FINBERT = "finbert"
    PPO = "ppo"
    CUSTOM = "custom"

class QualityMetric(Enum):
    """품질 메트릭 타입"""
    ACCURACY = "accuracy"
    PRECISION = "precision" 
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFIDENCE = "confidence"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    PREDICTION_DRIFT = "prediction_drift"
    BIAS_SCORE = "bias_score"
    STABILITY = "stability"

class QualityStatus(Enum):
    """품질 상태"""
    EXCELLENT = "excellent"  # 95%+
    GOOD = "good"           # 85-95%
    ACCEPTABLE = "acceptable" # 70-85%
    POOR = "poor"           # 50-70%
    CRITICAL = "critical"   # <50%

@dataclass
class QualityThreshold:
    """품질 임계값 설정"""
    excellent: float = 0.95
    good: float = 0.85
    acceptable: float = 0.70
    poor: float = 0.50
    
    def get_status(self, value: float) -> QualityStatus:
        """값에 따른 상태 반환"""
        if value >= self.excellent:
            return QualityStatus.EXCELLENT
        elif value >= self.good:
            return QualityStatus.GOOD
        elif value >= self.acceptable:
            return QualityStatus.ACCEPTABLE
        elif value >= self.poor:
            return QualityStatus.POOR
        else:
            return QualityStatus.CRITICAL

@dataclass
class ModelPrediction:
    """모델 예측 결과"""
    model_type: ModelType
    timestamp: datetime
    input_data: str  # 입력 데이터 요약
    prediction: Any  # 예측 결과
    confidence: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'model_type': self.model_type.value,
            'timestamp': self.timestamp.isoformat(),
            'input_data': self.input_data,
            'prediction': str(self.prediction),
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb,
            'metadata': self.metadata
        }

@dataclass
class QualityReport:
    """품질 보고서"""
    model_type: ModelType
    timestamp: datetime
    metrics: Dict[QualityMetric, float]
    status: QualityStatus
    recommendations: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'model_type': self.model_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metrics': {k.value: v for k, v in self.metrics.items()},
            'status': self.status.value,
            'recommendations': self.recommendations,
            'issues': self.issues,
            'summary': self.summary
        }

class ModelQualityDatabase:
    """모델 품질 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "model_quality.db"):
        self.db_path = db_path
        self._init_database()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        input_data TEXT,
                        prediction TEXT,
                        confidence REAL,
                        latency_ms REAL,
                        memory_mb REAL,
                        metadata TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS quality_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        status TEXT NOT NULL,
                        recommendations TEXT,
                        issues TEXT,
                        summary TEXT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_predictions_model_time 
                    ON predictions(model_type, timestamp);
                    
                    CREATE INDEX IF NOT EXISTS idx_reports_model_time 
                    ON quality_reports(model_type, timestamp);
                """)
                logger.info(f"Model quality database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_prediction(self, prediction: ModelPrediction) -> bool:
        """예측 결과 저장"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO predictions 
                        (model_type, timestamp, input_data, prediction, confidence, 
                         latency_ms, memory_mb, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prediction.model_type.value,
                        prediction.timestamp.isoformat(),
                        prediction.input_data,
                        str(prediction.prediction),
                        prediction.confidence,
                        prediction.latency_ms,
                        prediction.memory_mb,
                        json.dumps(prediction.metadata)
                    ))
                return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def save_quality_report(self, report: QualityReport) -> bool:
        """품질 보고서 저장"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO quality_reports 
                        (model_type, timestamp, metrics, status, recommendations, issues, summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        report.model_type.value,
                        report.timestamp.isoformat(),
                        json.dumps({k.value: v for k, v in report.metrics.items()}),
                        report.status.value,
                        json.dumps(report.recommendations),
                        json.dumps(report.issues),
                        report.summary
                    ))
                return True
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
            return False
    
    def get_recent_predictions(self, model_type: ModelType, 
                             hours: int = 24, limit: int = 1000) -> List[Dict[str, Any]]:
        """최근 예측 결과 조회"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                    
                    cursor = conn.execute("""
                        SELECT * FROM predictions 
                        WHERE model_type = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (model_type.value, cutoff_time, limit))
                    
                    results = []
                    for row in cursor:
                        result = dict(row)
                        result['metadata'] = json.loads(result['metadata'] or '{}')
                        results.append(result)
                    
                    return results
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []
    
    def get_quality_history(self, model_type: ModelType, 
                           days: int = 7) -> List[Dict[str, Any]]:
        """품질 히스토리 조회"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
                    
                    cursor = conn.execute("""
                        SELECT * FROM quality_reports 
                        WHERE model_type = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (model_type.value, cutoff_time))
                    
                    results = []
                    for row in cursor:
                        result = dict(row)
                        result['metrics'] = json.loads(result['metrics'])
                        result['recommendations'] = json.loads(result['recommendations'] or '[]')
                        result['issues'] = json.loads(result['issues'] or '[]')
                        results.append(result)
                    
                    return results
        except Exception as e:
            logger.error(f"Failed to get quality history: {e}")
            return []

class ModelQualityMonitor:
    """모델 품질 모니터링 시스템"""
    
    def __init__(self, db_path: str = "model_quality.db"):
        self.db = ModelQualityDatabase(db_path)
        self.predictions_cache = defaultdict(lambda: deque(maxlen=1000))
        self.thresholds = {
            ModelType.FINBERT: {
                QualityMetric.ACCURACY: QualityThreshold(0.90, 0.80, 0.70, 0.60),
                QualityMetric.CONFIDENCE: QualityThreshold(0.85, 0.75, 0.65, 0.50),
                QualityMetric.LATENCY: QualityThreshold(100, 200, 500, 1000),  # ms (낮을수록 좋음)
                QualityMetric.MEMORY_USAGE: QualityThreshold(500, 1000, 2000, 4000),  # MB (낮을수록 좋음)
            },
            ModelType.PPO: {
                QualityMetric.ACCURACY: QualityThreshold(0.85, 0.75, 0.65, 0.55),
                QualityMetric.CONFIDENCE: QualityThreshold(0.80, 0.70, 0.60, 0.50),
                QualityMetric.STABILITY: QualityThreshold(0.90, 0.80, 0.70, 0.60),
                QualityMetric.PREDICTION_DRIFT: QualityThreshold(0.05, 0.10, 0.20, 0.30),  # 낮을수록 좋음
            }
        }
        
        # 품질 모니터링 스레드
        self._monitoring_active = False
        self._monitor_thread = None
        
        logger.info("Model quality monitor initialized")
    
    def record_prediction(self, model_type: ModelType, input_data: str, 
                         prediction: Any, confidence: float = 0.0,
                         latency_ms: float = 0.0, memory_mb: float = 0.0,
                         metadata: Dict[str, Any] = None) -> bool:
        """예측 결과 기록"""
        try:
            prediction_record = ModelPrediction(
                model_type=model_type,
                timestamp=datetime.now(),
                input_data=input_data[:200] if isinstance(input_data, str) else str(input_data)[:200],
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                metadata=metadata or {}
            )
            
            # 캐시에 추가
            self.predictions_cache[model_type].append(prediction_record)
            
            # 데이터베이스에 저장
            return self.db.save_prediction(prediction_record)
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
            return False
    
    def calculate_quality_metrics(self, model_type: ModelType, 
                                hours: int = 24) -> Dict[QualityMetric, float]:
        """품질 메트릭 계산"""
        metrics = {}
        
        try:
            # 최근 예측 데이터 가져오기
            predictions = self.db.get_recent_predictions(model_type, hours)
            
            if not predictions:
                logger.warning(f"No predictions found for {model_type.value} in last {hours} hours")
                return metrics
            
            # 기본 메트릭 계산
            confidences = [p['confidence'] for p in predictions if p['confidence'] > 0]
            latencies = [p['latency_ms'] for p in predictions if p['latency_ms'] > 0]
            memories = [p['memory_mb'] for p in predictions if p['memory_mb'] > 0]
            
            if confidences:
                metrics[QualityMetric.CONFIDENCE] = np.mean(confidences)
            
            if latencies:
                metrics[QualityMetric.LATENCY] = np.mean(latencies)
            
            if memories:
                metrics[QualityMetric.MEMORY_USAGE] = np.mean(memories)
            
            # 안정성 계산 (confidence 변동성)
            if len(confidences) > 1:
                confidence_std = np.std(confidences)
                metrics[QualityMetric.STABILITY] = max(0, 1.0 - (confidence_std * 2))
            
            # 예측 드리프트 계산 (시간에 따른 변화)
            if len(confidences) >= 10:
                # 최근 절반과 이전 절반 비교
                mid_point = len(confidences) // 2
                recent_avg = np.mean(confidences[:mid_point])
                older_avg = np.mean(confidences[mid_point:])
                drift = abs(recent_avg - older_avg) / max(older_avg, 0.01)
                metrics[QualityMetric.PREDICTION_DRIFT] = drift
            
            # 모델별 특화 메트릭
            if model_type == ModelType.FINBERT:
                # FinBERT 특화 메트릭 (감정 분석 정확도 추정)
                positive_ratio = sum(1 for p in predictions 
                                   if 'sentiment' in str(p['prediction']).lower() 
                                   and 'positive' in str(p['prediction']).lower()) / len(predictions)
                # 균형있는 예측인지 확인 (0.2-0.8 범위가 건강함)
                balance_score = 1.0 - abs(positive_ratio - 0.5) * 2
                metrics[QualityMetric.ACCURACY] = max(0.5, balance_score * np.mean(confidences) if confidences else 0.5)
                
            elif model_type == ModelType.PPO:
                # PPO 특화 메트릭 (액션 분포와 신뢰도)
                action_counts = defaultdict(int)
                for p in predictions:
                    pred_str = str(p['prediction']).upper()
                    if 'BUY' in pred_str:
                        action_counts['BUY'] += 1
                    elif 'SELL' in pred_str:
                        action_counts['SELL'] += 1
                    else:
                        action_counts['HOLD'] += 1
                
                # 액션 다양성 점수 (너무 편향되지 않은 것이 좋음)
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    action_ratios = [count / total_actions for count in action_counts.values()]
                    diversity_score = 1.0 - max(action_ratios) + 0.3  # 최대 70% 편향까지 허용
                    metrics[QualityMetric.ACCURACY] = min(1.0, diversity_score)
            
            logger.debug(f"Calculated {len(metrics)} metrics for {model_type.value}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            return metrics
    
    def generate_quality_report(self, model_type: ModelType, 
                              hours: int = 24) -> QualityReport:
        """품질 보고서 생성"""
        try:
            # 메트릭 계산
            metrics = self.calculate_quality_metrics(model_type, hours)
            
            if not metrics:
                return QualityReport(
                    model_type=model_type,
                    timestamp=datetime.now(),
                    metrics={},
                    status=QualityStatus.CRITICAL,
                    issues=["No data available for quality assessment"],
                    summary="Insufficient data for quality analysis"
                )
            
            # 전체 상태 결정
            thresholds = self.thresholds.get(model_type, {})
            status_scores = []
            issues = []
            recommendations = []
            
            for metric, value in metrics.items():
                threshold = thresholds.get(metric)
                if threshold:
                    # latency와 memory는 낮을수록 좋음
                    if metric in [QualityMetric.LATENCY, QualityMetric.MEMORY_USAGE, QualityMetric.PREDICTION_DRIFT]:
                        # 역산 점수 계산
                        if value <= threshold.excellent:
                            status = QualityStatus.EXCELLENT
                            score = 1.0
                        elif value <= threshold.good:
                            status = QualityStatus.GOOD  
                            score = 0.8
                        elif value <= threshold.acceptable:
                            status = QualityStatus.ACCEPTABLE
                            score = 0.6
                        elif value <= threshold.poor:
                            status = QualityStatus.POOR
                            score = 0.4
                        else:
                            status = QualityStatus.CRITICAL
                            score = 0.2
                    else:
                        status = threshold.get_status(value)
                        score = value
                    
                    status_scores.append(score)
                    
                    # 문제점과 권장사항 생성
                    if status == QualityStatus.CRITICAL:
                        issues.append(f"{metric.value} is critical: {value:.3f}")
                        recommendations.append(f"Urgent: Address {metric.value} issues")
                    elif status == QualityStatus.POOR:
                        issues.append(f"{metric.value} is poor: {value:.3f}")
                        recommendations.append(f"Improve {metric.value}")
            
            # 전체 상태 결정 (평균 점수 기반)
            if status_scores:
                avg_score = np.mean(status_scores)
                overall_threshold = QualityThreshold()
                overall_status = overall_threshold.get_status(avg_score)
            else:
                overall_status = QualityStatus.CRITICAL
                avg_score = 0.0
            
            # 권장사항 추가
            if not recommendations:
                if overall_status == QualityStatus.EXCELLENT:
                    recommendations.append("Model performance is excellent. Continue monitoring.")
                elif overall_status == QualityStatus.GOOD:
                    recommendations.append("Model performance is good. Monitor for consistency.")
                else:
                    recommendations.append("Consider model retraining or fine-tuning.")
            
            # 요약 생성
            prediction_count = len(self.db.get_recent_predictions(model_type, hours))
            summary = (f"{model_type.value} model quality: {overall_status.value} "
                      f"(score: {avg_score:.2f}) based on {prediction_count} predictions "
                      f"in last {hours} hours")
            
            report = QualityReport(
                model_type=model_type,
                timestamp=datetime.now(),
                metrics=metrics,
                status=overall_status,
                recommendations=recommendations,
                issues=issues,
                summary=summary
            )
            
            # 보고서 저장
            self.db.save_quality_report(report)
            
            logger.info(f"Quality report generated for {model_type.value}: {overall_status.value}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return QualityReport(
                model_type=model_type,
                timestamp=datetime.now(),
                metrics={},
                status=QualityStatus.CRITICAL,
                issues=[f"Error generating report: {str(e)}"],
                summary="Failed to generate quality report"
            )
    
    def get_model_status(self, model_type: ModelType) -> Dict[str, Any]:
        """모델 상태 조회"""
        try:
            # 최신 품질 보고서 조회
            recent_reports = self.db.get_quality_history(model_type, days=1)
            latest_report = recent_reports[0] if recent_reports else None
            
            # 예측 통계
            recent_predictions = self.db.get_recent_predictions(model_type, hours=24)
            prediction_count_24h = len(recent_predictions)
            prediction_count_1h = len([p for p in recent_predictions 
                                     if datetime.fromisoformat(p['timestamp']) > 
                                     datetime.now() - timedelta(hours=1)])
            
            status = {
                'model_type': model_type.value,
                'timestamp': datetime.now().isoformat(),
                'prediction_count_24h': prediction_count_24h,
                'prediction_count_1h': prediction_count_1h,
                'latest_report': latest_report,
                'monitoring_active': self._monitoring_active,
                'cache_size': len(self.predictions_cache[model_type])
            }
            
            if recent_predictions:
                latest_prediction = recent_predictions[0]
                status['latest_prediction_time'] = latest_prediction['timestamp']
                status['latest_confidence'] = latest_prediction['confidence']
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {
                'model_type': model_type.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_monitoring(self, interval_minutes: int = 60):
        """품질 모니터링 시작"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    # 각 모델 타입에 대해 품질 보고서 생성
                    for model_type in [ModelType.FINBERT, ModelType.PPO]:
                        report = self.generate_quality_report(model_type)
                        logger.info(f"Generated quality report for {model_type.value}: {report.status.value}")
                    
                    # 다음 실행까지 대기
                    for _ in range(interval_minutes * 60):
                        if not self._monitoring_active:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # 에러 시 1분 후 재시도
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Quality monitoring started (interval: {interval_minutes} minutes)")
    
    def stop_monitoring(self):
        """품질 모니터링 중지"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Quality monitoring stopped")
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_monitoring()
        self.predictions_cache.clear()
        logger.info("Model quality monitor cleanup completed")

# 전역 모니터 인스턴스
_global_monitor = None

def get_quality_monitor(db_path: str = None) -> ModelQualityMonitor:
    """전역 품질 모니터 반환"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ModelQualityMonitor(db_path or "model_quality.db")
    return _global_monitor

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    import random
    
    async def test_quality_monitor():
        print("🧪 Model Quality Monitor 테스트")
        
        monitor = get_quality_monitor("test_quality.db")
        
        print("\n1️⃣ 예측 데이터 시뮬레이션")
        
        # FinBERT 예측 시뮬레이션
        for i in range(50):
            confidence = random.uniform(0.6, 0.95)
            sentiment = "positive" if random.random() > 0.4 else "negative"
            
            monitor.record_prediction(
                model_type=ModelType.FINBERT,
                input_data=f"Test news {i+1}: market analysis",
                prediction=f"sentiment: {sentiment}",
                confidence=confidence,
                latency_ms=random.uniform(50, 200),
                memory_mb=random.uniform(200, 800),
                metadata={"test_run": True}
            )
        
        # PPO 예측 시뮬레이션
        actions = ["BUY", "SELL", "HOLD"]
        for i in range(30):
            action = random.choice(actions)
            confidence = random.uniform(0.5, 0.9)
            
            monitor.record_prediction(
                model_type=ModelType.PPO,
                input_data=f"Market state {i+1}",
                prediction=action,
                confidence=confidence,
                latency_ms=random.uniform(80, 300),
                memory_mb=random.uniform(300, 1200),
                metadata={"action": action, "test_run": True}
            )
        
        print(f"  ✅ 80개 예측 기록 완료")
        
        print("\n2️⃣ 품질 보고서 생성")
        
        # FinBERT 품질 보고서
        finbert_report = monitor.generate_quality_report(ModelType.FINBERT)
        print(f"  📊 FinBERT 상태: {finbert_report.status.value}")
        print(f"  📈 메트릭 수: {len(finbert_report.metrics)}")
        if finbert_report.recommendations:
            print(f"  💡 권장사항: {finbert_report.recommendations[0]}")
        
        # PPO 품질 보고서  
        ppo_report = monitor.generate_quality_report(ModelType.PPO)
        print(f"  📊 PPO 상태: {ppo_report.status.value}")
        print(f"  📈 메트릭 수: {len(ppo_report.metrics)}")
        if ppo_report.recommendations:
            print(f"  💡 권장사항: {ppo_report.recommendations[0]}")
        
        print("\n3️⃣ 모델 상태 조회")
        
        finbert_status = monitor.get_model_status(ModelType.FINBERT)
        ppo_status = monitor.get_model_status(ModelType.PPO)
        
        print(f"  📋 FinBERT: {finbert_status['prediction_count_24h']}개 예측 (24h)")
        print(f"  📋 PPO: {ppo_status['prediction_count_24h']}개 예측 (24h)")
        
        print("\n🎉 Model Quality Monitor 테스트 완료!")
        
        # 정리
        monitor.cleanup()
        
        # 테스트 DB 파일 삭제
        test_db_path = Path("test_quality.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    # 테스트 실행
    asyncio.run(test_quality_monitor())