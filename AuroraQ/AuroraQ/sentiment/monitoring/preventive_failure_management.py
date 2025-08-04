#!/usr/bin/env python3
"""
Preventive Failure Management System for AuroraQ
AuroraQ 예방적 장애 관리 시스템 - 사후 대응에서 선제적 예방으로 전환
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics
from collections import defaultdict, deque
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PreventionStrategy(Enum):
    """예방 전략 유형"""
    RESOURCE_PREALLOCATION = "resource_preallocation"
    LOAD_BALANCING = "load_balancing"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMITING = "rate_limiting"
    HEALTH_MONITORING = "health_monitoring"
    PREDICTIVE_SCALING = "predictive_scaling"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    ANOMALY_DETECTION = "anomaly_detection"

class RiskLevel(Enum):
    """위험 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PreventionAction(Enum):
    """예방 액션"""
    INCREASE_RESOURCES = "increase_resources"
    DECREASE_LOAD = "decrease_load"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    ALERT_OPERATORS = "alert_operators"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    BACKUP_DATA = "backup_data"
    OPTIMIZE_QUERIES = "optimize_queries"

@dataclass
class RiskAssessment:
    """위험 평가"""
    component: str
    risk_level: RiskLevel
    risk_score: float
    contributing_factors: List[str]
    predicted_failure_time: Optional[datetime] = None
    confidence: float = 0.0
    recommended_actions: List[PreventionAction] = field(default_factory=list)

@dataclass
class PreventiveAction:
    """예방적 조치"""
    id: str
    strategy: PreventionStrategy
    action: PreventionAction
    component: str
    description: str
    scheduled_time: datetime
    estimated_duration: int  # 분
    priority: int
    success_probability: float
    cost_estimate: float  # 상대적 비용

@dataclass
class PreventionResult:
    """예방 결과"""
    action: PreventiveAction
    executed_at: datetime
    success: bool
    execution_time: float
    prevented_incidents: int
    metrics_improvement: Dict[str, float]
    error_message: Optional[str] = None

class PreventiveFailureManagement:
    """예방적 장애 관리 시스템"""
    
    def __init__(self,
                 assessment_interval: int = 300,  # 5분마다 위험 평가
                 prediction_horizon: int = 3600,  # 1시간 예측 범위
                 history_window: int = 86400):    # 24시간 히스토리
        """
        초기화
        
        Args:
            assessment_interval: 위험 평가 간격 (초)
            prediction_horizon: 예측 범위 (초)
            history_window: 히스토리 윈도우 (초)
        """
        self.assessment_interval = assessment_interval
        self.prediction_horizon = prediction_horizon
        self.history_window = history_window
        
        # 데이터 저장소
        self.risk_assessments: deque = deque(maxlen=1000)
        self.prevention_history: deque = deque(maxlen=1000)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.active_preventions: Dict[str, PreventiveAction] = {}
        
        # 예측 모델
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # 임계값 및 설정
        self.risk_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.9
        }
        
        # 컴포넌트별 중요도
        self.component_criticality = {
            "news_collector": 0.9,
            "sentiment_analyzer": 0.8,
            "topic_classifier": 0.7,
            "strategy_selector": 0.95,
            "system": 1.0,
            "database": 0.9,
            "cache": 0.6
        }
        
        # 예방 전략 매핑
        self.prevention_strategies = {
            "high_cpu_usage": [
                PreventionStrategy.PREDICTIVE_SCALING,
                PreventionStrategy.LOAD_BALANCING,
                PreventionStrategy.RESOURCE_PREALLOCATION
            ],
            "memory_leak": [
                PreventionStrategy.HEALTH_MONITORING,
                PreventionStrategy.MAINTENANCE_SCHEDULING,
                PreventionStrategy.ANOMALY_DETECTION
            ],
            "connection_exhaustion": [
                PreventionStrategy.CIRCUIT_BREAKER,
                PreventionStrategy.RATE_LIMITING,
                PreventionStrategy.LOAD_BALANCING
            ],
            "disk_space": [
                PreventionStrategy.MAINTENANCE_SCHEDULING,
                PreventionStrategy.HEALTH_MONITORING
            ],
            "api_degradation": [
                PreventionStrategy.CIRCUIT_BREAKER,
                PreventionStrategy.RATE_LIMITING,
                PreventionStrategy.HEALTH_MONITORING
            ]
        }
        
        # 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 통계
        self.stats = {
            "total_assessments": 0,
            "preventions_executed": 0,
            "incidents_prevented": 0,
            "false_positives": 0,
            "cost_savings": 0.0,
            "avg_risk_score": 0.0
        }
        
        logger.info(f"Preventive Failure Management initialized with {assessment_interval}s assessment interval")
    
    async def assess_risks(self) -> List[RiskAssessment]:
        """시스템 위험 평가"""
        assessments = []
        
        try:
            # 각 컴포넌트별 위험 평가
            components = ["news_collector", "sentiment_analyzer", "topic_classifier", 
                         "strategy_selector", "system", "database", "cache"]
            
            for component in components:
                assessment = await self._assess_component_risk(component)
                if assessment:
                    assessments.append(assessment)
                    self.stats["total_assessments"] += 1
            
            # 전체 위험 점수 업데이트
            if assessments:
                avg_risk = statistics.mean([a.risk_score for a in assessments])
                self.stats["avg_risk_score"] = avg_risk
            
            # 고위험 평가 저장
            high_risk_assessments = [a for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            for assessment in high_risk_assessments:
                self.risk_assessments.append(assessment)
            
            return assessments
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return []
    
    async def _assess_component_risk(self, component: str) -> Optional[RiskAssessment]:
        """개별 컴포넌트 위험 평가"""
        try:
            # 메트릭 수집
            metrics = await self._collect_component_metrics(component)
            if not metrics:
                return None
            
            # 위험 요소 분석
            risk_factors = []
            risk_score = 0.0
            
            # 1. 트렌드 분석
            trend_risk = self._analyze_metric_trends(component, metrics)
            if trend_risk > 0.3:
                risk_factors.append(f"degrading_trend ({trend_risk:.2f})")
                risk_score += trend_risk * 0.3
            
            # 2. 이상 탐지  
            anomaly_risk = self._detect_anomalies(component, metrics)
            if anomaly_risk > 0.3:
                risk_factors.append(f"anomaly_detected ({anomaly_risk:.2f})")
                risk_score += anomaly_risk * 0.25
            
            # 3. 임계값 접근
            threshold_risk = self._check_threshold_proximity(component, metrics)
            if threshold_risk > 0.3:
                risk_factors.append(f"threshold_proximity ({threshold_risk:.2f})")
                risk_score += threshold_risk * 0.25
            
            # 4. 히스토리 기반 예측
            prediction_risk = await self._predict_failure_probability(component, metrics)
            if prediction_risk > 0.3:
                risk_factors.append(f"failure_prediction ({prediction_risk:.2f})")
                risk_score += prediction_risk * 0.2
            
            # 컴포넌트 중요도 가중치 적용
            criticality = self.component_criticality.get(component, 0.5)
            risk_score *= criticality
            
            # 위험 수준 결정
            risk_level = self._determine_risk_level(risk_score)
            
            # 예측된 장애 시간 계산
            predicted_time = None
            confidence = 0.0
            if risk_score > 0.6:
                predicted_time, confidence = self._predict_failure_time(component, metrics, risk_score)
            
            # 권장 조치 생성
            recommended_actions = self._generate_prevention_recommendations(component, risk_factors, risk_score)
            
            return RiskAssessment(
                component=component,
                risk_level=risk_level,
                risk_score=risk_score,
                contributing_factors=risk_factors,
                predicted_failure_time=predicted_time,
                confidence=confidence,
                recommended_actions=recommended_actions
            )
            
        except Exception as e:
            logger.error(f"Error assessing risk for {component}: {e}")
            return None
    
    async def _collect_component_metrics(self, component: str) -> Dict[str, Any]:
        """컴포넌트 메트릭 수집"""
        # 실제 구현에서는 각 컴포넌트의 실제 메트릭을 수집
        # 여기서는 시뮬레이션 데이터
        import random
        
        base_metrics = {
            "timestamp": datetime.now(),
            "response_time": random.uniform(0.1, 2.0),
            "error_rate": random.uniform(0.0, 0.1),
            "throughput": random.uniform(100, 1000),
            "cpu_usage": random.uniform(20, 90),
            "memory_usage": random.uniform(30, 85),
            "connection_count": random.randint(10, 100)
        }
        
        # 컴포넌트별 특별 메트릭
        if component == "news_collector":
            base_metrics.update({
                "sources_available": random.randint(3, 5),
                "collection_rate": random.uniform(50, 200),
                "api_quota_remaining": random.uniform(0.2, 1.0)
            })
        elif component == "database":
            base_metrics.update({
                "connection_pool_usage": random.uniform(0.3, 0.9),
                "query_time": random.uniform(0.01, 0.5),
                "disk_io": random.uniform(10, 100)
            })
        
        # 메트릭 히스토리에 저장
        self.metric_history[component].append(base_metrics)
        
        return base_metrics
    
    def _analyze_metric_trends(self, component: str, current_metrics: Dict[str, Any]) -> float:
        """메트릭 트렌드 분석"""
        history = list(self.metric_history[component])
        if len(history) < 5:
            return 0.0
        
        # 최근 메트릭들의 트렌드 분석
        recent_metrics = history[-10:]  # 최근 10개
        
        risk_score = 0.0
        
        # 응답 시간 트렌드
        response_times = [m.get("response_time", 0) for m in recent_metrics]
        if len(response_times) >= 3:
            trend = self._calculate_trend(response_times)
            if trend > 0.1:  # 응답 시간 증가 트렌드
                risk_score += min(0.5, trend * 2)
        
        # 에러율 트렌드
        error_rates = [m.get("error_rate", 0) for m in recent_metrics]
        if len(error_rates) >= 3:
            trend = self._calculate_trend(error_rates)
            if trend > 0.01:  # 에러율 증가 트렌드
                risk_score += min(0.4, trend * 10)
        
        # CPU/메모리 사용률 트렌드
        cpu_usage = [m.get("cpu_usage", 0) for m in recent_metrics]
        memory_usage = [m.get("memory_usage", 0) for m in recent_metrics]
        
        for usage_data in [cpu_usage, memory_usage]:
            if len(usage_data) >= 3:
                trend = self._calculate_trend(usage_data)
                if trend > 2.0:  # 사용률 증가 트렌드
                    risk_score += min(0.3, trend / 20)
        
        return min(1.0, risk_score)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """트렌드 계산 (선형 회귀 기울기)"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(x, y)
            
            return float(model.coef_[0])
        except:
            return 0.0
    
    def _detect_anomalies(self, component: str, current_metrics: Dict[str, Any]) -> float:
        """이상 탐지"""
        history = list(self.metric_history[component])
        if len(history) < 10:
            return 0.0
        
        try:
            # 이상 탐지 모델 학습/업데이트
            if component not in self.anomaly_detectors:
                self.anomaly_detectors[component] = IsolationForest(
                    contamination=0.1, 
                    random_state=42
                )
            
            detector = self.anomaly_detectors[component]
            
            # 히스토리 데이터로 학습
            feature_data = []
            for metrics in history[-50:]:  # 최근 50개 데이터포인트
                features = [
                    metrics.get("response_time", 0),
                    metrics.get("error_rate", 0),
                    metrics.get("cpu_usage", 0),
                    metrics.get("memory_usage", 0),
                    metrics.get("throughput", 0)
                ]
                feature_data.append(features)
            
            if len(feature_data) >= 10:
                detector.fit(feature_data)
                
                # 현재 메트릭 이상 점수 계산
                current_features = [[
                    current_metrics.get("response_time", 0),
                    current_metrics.get("error_rate", 0),
                    current_metrics.get("cpu_usage", 0),
                    current_metrics.get("memory_usage", 0),
                    current_metrics.get("throughput", 0)
                ]]
                
                anomaly_score = detector.decision_function(current_features)[0]
                # 점수를 0-1 범위로 정규화 (낮을수록 이상)
                normalized_score = max(0, -anomaly_score / 2)
                
                return min(1.0, normalized_score)
            
        except Exception as e:
            logger.debug(f"Anomaly detection error for {component}: {e}")
        
        return 0.0
    
    def _check_threshold_proximity(self, component: str, metrics: Dict[str, Any]) -> float:
        """임계값 접근도 체크"""
        risk_score = 0.0
        
        # 임계값 정의
        thresholds = {
            "response_time": 2.0,
            "error_rate": 0.05,  # 5%
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "connection_pool_usage": 0.9
        }
        
        for metric, threshold in thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                proximity = value / threshold
                
                if proximity > 0.8:  # 80% 이상 임계값에 접근
                    risk_addition = min(0.4, (proximity - 0.8) * 2)
                    risk_score += risk_addition
        
        return min(1.0, risk_score)
    
    async def _predict_failure_probability(self, component: str, metrics: Dict[str, Any]) -> float:
        """장애 확률 예측"""
        # 간단한 확률적 예측 모델
        # 실제로는 더 정교한 ML 모델 사용
        
        base_probability = 0.0
        
        # 메트릭 기반 확률 계산
        response_time = metrics.get("response_time", 0)
        error_rate = metrics.get("error_rate", 0)
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)
        
        # 응답 시간 기반
        if response_time > 1.5:
            base_probability += min(0.3, (response_time - 1.5) * 0.2)
        
        # 에러율 기반
        if error_rate > 0.02:
            base_probability += min(0.4, error_rate * 10)
        
        # 리소스 사용률 기반
        resource_pressure = (cpu_usage + memory_usage) / 200
        if resource_pressure > 0.7:
            base_probability += min(0.3, (resource_pressure - 0.7) * 1.0)
        
        # 과거 장애 패턴 기반 조정
        failure_history_multiplier = self._get_failure_history_multiplier(component)
        
        return min(1.0, base_probability * failure_history_multiplier)
    
    def _get_failure_history_multiplier(self, component: str) -> float:
        """과거 장애 이력 기반 승수"""
        # 실제 구현에서는 과거 장애 데이터베이스 조회
        # 여기서는 시뮬레이션
        
        # 컴포넌트별 기본 안정성
        base_stability = {
            "news_collector": 0.9,
            "sentiment_analyzer": 0.95,
            "topic_classifier": 0.98,
            "strategy_selector": 0.9,
            "system": 0.85,
            "database": 0.92,
            "cache": 0.95
        }
        
        stability = base_stability.get(component, 0.9)
        return 1.0 / stability  # 안정성이 낮을수록 높은 승수
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """위험 수준 결정"""
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _predict_failure_time(self, component: str, metrics: Dict[str, Any], risk_score: float) -> Tuple[Optional[datetime], float]:
        """장애 발생 시간 예측"""
        if risk_score < 0.6:
            return None, 0.0
        
        # 위험 점수 기반 시간 예측
        # 높은 위험일수록 더 빠른 장애 예측
        hours_until_failure = max(1, int(24 * (1 - risk_score)))
        predicted_time = datetime.now() + timedelta(hours=hours_until_failure)
        
        # 신뢰도는 위험 점수와 메트릭 품질에 기반
        confidence = min(0.9, risk_score * 0.8)
        
        return predicted_time, confidence
    
    def _generate_prevention_recommendations(self, component: str, risk_factors: List[str], risk_score: float) -> List[PreventionAction]:
        """예방 조치 권장사항 생성"""
        recommendations = []
        
        # 위험 요소별 권장 조치
        for factor in risk_factors:
            if "trend" in factor or "cpu" in factor or "memory" in factor:
                recommendations.extend([
                    PreventionAction.INCREASE_RESOURCES,
                    PreventionAction.OPTIMIZE_QUERIES
                ])
            elif "anomaly" in factor:
                recommendations.extend([
                    PreventionAction.ALERT_OPERATORS,
                    PreventionAction.SCHEDULE_MAINTENANCE
                ])
            elif "threshold" in factor:
                recommendations.extend([
                    PreventionAction.ADJUST_THRESHOLDS,
                    PreventionAction.ENABLE_CIRCUIT_BREAKER
                ])
            elif "prediction" in factor:
                recommendations.extend([
                    PreventionAction.BACKUP_DATA,
                    PreventionAction.DECREASE_LOAD
                ])
        
        # 중복 제거 및 우선순위 정렬
        unique_recommendations = list(set(recommendations))
        
        # 위험 수준에 따른 우선순위 조정
        if risk_score > 0.8:
            priority_actions = [
                PreventionAction.ALERT_OPERATORS,
                PreventionAction.BACKUP_DATA,
                PreventionAction.ENABLE_CIRCUIT_BREAKER
            ]
            unique_recommendations = priority_actions + [a for a in unique_recommendations if a not in priority_actions]
        
        return unique_recommendations[:5]  # 상위 5개만 반환
    
    async def create_preventive_actions(self, risk_assessments: List[RiskAssessment]) -> List[PreventiveAction]:
        """예방적 조치 계획 생성"""
        actions = []
        
        # 고위험 평가만 처리
        high_risk_assessments = [a for a in risk_assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        for assessment in high_risk_assessments:
            for i, action_type in enumerate(assessment.recommended_actions):
                action_id = f"{assessment.component}_{action_type.value}_{int(time.time())}"
                
                # 전략 결정
                strategy = self._determine_prevention_strategy(assessment.component, action_type)
                
                # 스케줄링
                urgency_hours = 1 if assessment.risk_level == RiskLevel.CRITICAL else 4
                scheduled_time = datetime.now() + timedelta(hours=urgency_hours)
                
                # 비용 및 성공률 추정
                cost = self._estimate_action_cost(action_type)
                success_prob = self._estimate_success_probability(action_type, assessment)
                
                preventive_action = PreventiveAction(
                    id=action_id,
                    strategy=strategy,
                    action=action_type,
                    component=assessment.component,
                    description=self._generate_action_description(action_type, assessment),
                    scheduled_time=scheduled_time,
                    estimated_duration=self._estimate_duration(action_type),
                    priority=10 - i,  # 첫 번째 권장 조치가 높은 우선순위
                    success_probability=success_prob,
                    cost_estimate=cost
                )
                
                actions.append(preventive_action)
        
        # 우선순위 및 비용으로 정렬
        actions.sort(key=lambda a: (a.priority, -a.success_probability, a.cost_estimate), reverse=True)
        
        return actions
    
    def _determine_prevention_strategy(self, component: str, action: PreventionAction) -> PreventionStrategy:
        """예방 전략 결정"""
        strategy_mapping = {
            PreventionAction.INCREASE_RESOURCES: PreventionStrategy.PREDICTIVE_SCALING,
            PreventionAction.DECREASE_LOAD: PreventionStrategy.LOAD_BALANCING,
            PreventionAction.ENABLE_CIRCUIT_BREAKER: PreventionStrategy.CIRCUIT_BREAKER,
            PreventionAction.SCHEDULE_MAINTENANCE: PreventionStrategy.MAINTENANCE_SCHEDULING,
            PreventionAction.ALERT_OPERATORS: PreventionStrategy.HEALTH_MONITORING,
            PreventionAction.ADJUST_THRESHOLDS: PreventionStrategy.HEALTH_MONITORING,
            PreventionAction.BACKUP_DATA: PreventionStrategy.MAINTENANCE_SCHEDULING,
            PreventionAction.OPTIMIZE_QUERIES: PreventionStrategy.PREDICTIVE_SCALING
        }
        
        return strategy_mapping.get(action, PreventionStrategy.HEALTH_MONITORING)
    
    def _estimate_action_cost(self, action: PreventionAction) -> float:
        """액션 비용 추정 (0-1 스케일)"""
        cost_mapping = {
            PreventionAction.ALERT_OPERATORS: 0.1,
            PreventionAction.ADJUST_THRESHOLDS: 0.2,
            PreventionAction.ENABLE_CIRCUIT_BREAKER: 0.3,
            PreventionAction.OPTIMIZE_QUERIES: 0.4,
            PreventionAction.DECREASE_LOAD: 0.5,
            PreventionAction.BACKUP_DATA: 0.6,
            PreventionAction.SCHEDULE_MAINTENANCE: 0.7,
            PreventionAction.INCREASE_RESOURCES: 0.8
        }
        
        return cost_mapping.get(action, 0.5)
    
    def _estimate_success_probability(self, action: PreventionAction, assessment: RiskAssessment) -> float:
        """성공 확률 추정"""
        base_probability = {
            PreventionAction.ALERT_OPERATORS: 0.9,
            PreventionAction.ADJUST_THRESHOLDS: 0.8,
            PreventionAction.ENABLE_CIRCUIT_BREAKER: 0.85,
            PreventionAction.OPTIMIZE_QUERIES: 0.75,
            PreventionAction.DECREASE_LOAD: 0.8,
            PreventionAction.BACKUP_DATA: 0.95,
            PreventionAction.SCHEDULE_MAINTENANCE: 0.9,
            PreventionAction.INCREASE_RESOURCES: 0.85
        }
        
        probability = base_probability.get(action, 0.7)
        
        # 위험 수준에 따른 조정
        if assessment.risk_level == RiskLevel.CRITICAL:
            probability *= 0.9  # 매우 위험한 상황에서는 성공률 약간 감소
        elif assessment.risk_level == RiskLevel.LOW:
            probability *= 1.1  # 저위험 상황에서는 성공률 증가
        
        return min(1.0, probability)
    
    def _generate_action_description(self, action: PreventionAction, assessment: RiskAssessment) -> str:
        """액션 설명 생성"""
        descriptions = {
            PreventionAction.INCREASE_RESOURCES: f"{assessment.component} 리소스 증설",
            PreventionAction.DECREASE_LOAD: f"{assessment.component} 부하 감소",
            PreventionAction.ENABLE_CIRCUIT_BREAKER: f"{assessment.component} 서킷 브레이커 활성화",
            PreventionAction.SCHEDULE_MAINTENANCE: f"{assessment.component} 예방 정비 스케줄링",
            PreventionAction.ALERT_OPERATORS: f"{assessment.component} 운영진 알림 발송",
            PreventionAction.ADJUST_THRESHOLDS: f"{assessment.component} 임계값 조정",
            PreventionAction.BACKUP_DATA: f"{assessment.component} 데이터 백업 실행",
            PreventionAction.OPTIMIZE_QUERIES: f"{assessment.component} 쿼리 최적화"
        }
        
        base_desc = descriptions.get(action, f"{assessment.component}에 대한 예방 조치")
        risk_desc = f" (위험도: {assessment.risk_score:.1%}, 수준: {assessment.risk_level.value})"
        
        return base_desc + risk_desc
    
    def _estimate_duration(self, action: PreventionAction) -> int:
        """액션 소요 시간 추정 (분)"""
        duration_mapping = {
            PreventionAction.ALERT_OPERATORS: 5,
            PreventionAction.ADJUST_THRESHOLDS: 15,
            PreventionAction.ENABLE_CIRCUIT_BREAKER: 10,
            PreventionAction.OPTIMIZE_QUERIES: 30,
            PreventionAction.DECREASE_LOAD: 20,
            PreventionAction.BACKUP_DATA: 45,
            PreventionAction.SCHEDULE_MAINTENANCE: 60,
            PreventionAction.INCREASE_RESOURCES: 30
        }
        
        return duration_mapping.get(action, 30)
    
    async def execute_preventive_action(self, action: PreventiveAction) -> PreventionResult:
        """예방적 조치 실행"""
        start_time = time.time()
        
        result = PreventionResult(
            action=action,
            executed_at=datetime.now(),
            success=False,
            execution_time=0,
            prevented_incidents=0,
            metrics_improvement={}
        )
        
        try:
            logger.info(f"Executing preventive action: {action.description}")
            
            # 액션별 실행 로직
            success = await self._execute_specific_action(action)
            
            if success:
                result.success = True
                result.prevented_incidents = 1 if action.action in [
                    PreventionAction.INCREASE_RESOURCES,
                    PreventionAction.ENABLE_CIRCUIT_BREAKER,
                    PreventionAction.SCHEDULE_MAINTENANCE
                ] else 0
                
                # 메트릭 개선 측정
                result.metrics_improvement = await self._measure_improvement(action)
                
                self.stats["preventions_executed"] += 1
                self.stats["incidents_prevented"] += result.prevented_incidents
                
                # 비용 절감 계산 (예상 장애 비용 - 예방 비용)
                estimated_savings = self._calculate_cost_savings(action)
                self.stats["cost_savings"] += estimated_savings
                
                logger.info(f"Preventive action completed successfully: {action.description}")
            else:
                result.error_message = "Action execution failed"
                self.stats["false_positives"] += 1
                
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Preventive action failed: {action.description} - {e}")
        
        finally:
            result.execution_time = time.time() - start_time
            self.prevention_history.append(result)
        
        return result
    
    async def _execute_specific_action(self, action: PreventiveAction) -> bool:
        """특정 액션 실행"""
        # 실제 구현에서는 각 액션의 실제 로직 구현
        # 여기서는 시뮬레이션
        
        await asyncio.sleep(action.estimated_duration / 60)  # 분을 초로 변환
        
        # 성공 확률에 따른 결과 결정
        import random
        return random.random() < action.success_probability
    
    async def _measure_improvement(self, action: PreventiveAction) -> Dict[str, float]:
        """메트릭 개선 측정"""
        # 액션 후 메트릭 개선 시뮬레이션
        improvements = {}
        
        if action.action == PreventionAction.INCREASE_RESOURCES:
            improvements = {
                "cpu_usage_reduction": 15.0,
                "memory_usage_reduction": 10.0,
                "response_time_improvement": 0.3
            }
        elif action.action == PreventionAction.ENABLE_CIRCUIT_BREAKER:
            improvements = {
                "error_rate_reduction": 0.02,
                "availability_improvement": 0.05
            }
        elif action.action == PreventionAction.OPTIMIZE_QUERIES:
            improvements = {
                "response_time_improvement": 0.5,
                "throughput_increase": 20.0
            }
        
        return improvements
    
    def _calculate_cost_savings(self, action: PreventiveAction) -> float:
        """비용 절감 계산"""
        # 예상 장애 비용 - 예방 조치 비용
        estimated_incident_cost = 1000.0  # 기본 장애 비용
        prevention_cost = action.cost_estimate * 100  # 예방 비용
        
        # 컴포넌트 중요도에 따른 장애 비용 조정
        criticality = self.component_criticality.get(action.component, 0.5)
        adjusted_incident_cost = estimated_incident_cost * criticality
        
        return max(0, adjusted_incident_cost - prevention_cost)
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 정기적인 위험 평가 및 예방 조치
                asyncio.create_task(self._preventive_maintenance_cycle())
                
                time.sleep(self.assessment_interval)
                
            except Exception as e:
                logger.error(f"Preventive monitoring loop error: {e}")
                time.sleep(self.assessment_interval)
    
    async def _preventive_maintenance_cycle(self):
        """예방적 유지보수 사이클"""
        try:
            # 위험 평가
            risk_assessments = await self.assess_risks()
            
            # 고위험 상황에 대한 예방 조치 생성
            preventive_actions = await self.create_preventive_actions(risk_assessments)
            
            # 즉시 실행이 필요한 액션들 실행
            immediate_actions = [a for a in preventive_actions if a.priority >= 8]
            
            for action in immediate_actions:
                await self.execute_preventive_action(action)
                
        except Exception as e:
            logger.error(f"Preventive maintenance cycle error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        recent_assessments = [a for a in self.risk_assessments if a.predicted_failure_time and a.predicted_failure_time > datetime.now()]
        
        return {
            "prevention_system_status": "active" if self.monitoring_active else "inactive",
            "active_risks": len(recent_assessments),
            "high_risk_components": [a.component for a in recent_assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]],
            "statistics": self.stats.copy(),
            "avg_risk_score": self.stats["avg_risk_score"],
            "prevention_success_rate": self.stats["preventions_executed"] / max(1, self.stats["total_assessments"]) * 100,
            "cost_savings": self.stats["cost_savings"],
            "incidents_prevented": self.stats["incidents_prevented"]
        }
    
    def shutdown(self):
        """예방적 장애 관리 시스템 종료"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Preventive Failure Management System shut down")

# 전역 예방 시스템 인스턴스
_prevention_system: Optional[PreventiveFailureManagement] = None

def get_prevention_system() -> PreventiveFailureManagement:
    """전역 예방 시스템 반환"""
    global _prevention_system
    if _prevention_system is None:
        _prevention_system = PreventiveFailureManagement()
    return _prevention_system

# 테스트 코드
if __name__ == "__main__":
    async def test_preventive_failure_management():
        """예방적 장애 관리 시스템 테스트"""
        print("=== Preventive Failure Management System Test ===\n")
        
        prevention_system = PreventiveFailureManagement(
            assessment_interval=30,  # 30초 평가 간격
            prediction_horizon=1800  # 30분 예측 범위
        )
        
        # 테스트 1: 위험 평가
        print("1. Testing risk assessment...")
        assessments = await prevention_system.assess_risks()
        print(f"   Risk assessments: {len(assessments)}")
        for assessment in assessments:
            print(f"   - {assessment.component}: {assessment.risk_level.value} ({assessment.risk_score:.2f})")
            if assessment.predicted_failure_time:
                print(f"     Predicted failure: {assessment.predicted_failure_time.strftime('%Y-%m-%d %H:%M:%S')} (confidence: {assessment.confidence:.1%})")
        
        # 테스트 2: 예방 조치 계획
        print("\n2. Testing preventive action planning...")
        if assessments:
            actions = await prevention_system.create_preventive_actions(assessments)
            print(f"   Preventive actions planned: {len(actions)}")
            for action in actions[:3]:  # 상위 3개만 표시
                print(f"   - {action.description}")
                print(f"     Priority: {action.priority}, Success probability: {action.success_probability:.1%}")
                print(f"     Scheduled: {action.scheduled_time.strftime('%H:%M:%S')}, Duration: {action.estimated_duration}min")
            
            # 테스트 3: 예방 조치 실행
            if actions:
                print("\n3. Testing preventive action execution...")
                test_action = actions[0]
                result = await prevention_system.execute_preventive_action(test_action)
                print(f"   Action: {test_action.description}")
                print(f"   Result: {result.success}")
                print(f"   Execution time: {result.execution_time:.1f}s")
                print(f"   Prevented incidents: {result.prevented_incidents}")
                if result.metrics_improvement:
                    print(f"   Improvements: {result.metrics_improvement}")
        
        # 테스트 4: 시스템 상태
        print("\n4. System status:")
        status = prevention_system.get_system_status()
        print(f"   Active risks: {status['active_risks']}")
        print(f"   High risk components: {status['high_risk_components']}")
        print(f"   Total assessments: {status['statistics']['total_assessments']}")
        print(f"   Preventions executed: {status['statistics']['preventions_executed']}")
        print(f"   Incidents prevented: {status['statistics']['incidents_prevented']}")
        print(f"   Cost savings: ${status['cost_savings']:.2f}")
        
        prevention_system.shutdown()
        print("\n✅ Preventive Failure Management System test completed")
    
    # 테스트 실행
    asyncio.run(test_preventive_failure_management())