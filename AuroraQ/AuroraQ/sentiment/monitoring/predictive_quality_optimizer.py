#!/usr/bin/env python3
"""
Predictive Quality Optimizer for AuroraQ
AuroraQ 예측적 품질 최적화기 - 데이터 품질 향상 및 예방적 장애 관리
"""

import asyncio
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import threading
import json

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """품질 메트릭 유형"""
    COMPLETENESS = "completeness"       # 데이터 완전성
    ACCURACY = "accuracy"              # 정확성
    CONSISTENCY = "consistency"        # 일관성
    TIMELINESS = "timeliness"         # 시의성
    VALIDITY = "validity"             # 유효성
    UNIQUENESS = "uniqueness"         # 고유성

class QualityIssue(Enum):
    """품질 문제 유형"""
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    OUTDATED_DATA = "outdated_data"
    DUPLICATE_DATA = "duplicate_data"
    INCONSISTENT_DATA = "inconsistent_data"
    LOW_CONFIDENCE = "low_confidence"

@dataclass
class QualityScore:
    """품질 점수"""
    overall: float
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    validity: float
    uniqueness: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityRule:
    """품질 규칙"""
    name: str
    metric: QualityMetric
    condition: Callable[[Any], bool]
    weight: float
    threshold: float
    description: str

@dataclass
class QualityImprovement:
    """품질 개선 조치"""
    issue: QualityIssue
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int
    expected_improvement: float
    description: str

class PredictiveQualityOptimizer:
    """예측적 품질 최적화기"""
    
    def __init__(self, 
                 target_quality: float = 0.8,
                 history_size: int = 1000,
                 prediction_window: int = 300):
        """
        초기화
        
        Args:
            target_quality: 목표 품질 점수 (80%)
            history_size: 품질 히스토리 크기
            prediction_window: 예측 윈도우 (초)
        """
        self.target_quality = target_quality
        self.history_size = history_size
        self.prediction_window = prediction_window
        
        # 품질 히스토리
        self.quality_history: deque = deque(maxlen=history_size)
        self.component_quality: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 품질 규칙
        self.quality_rules: List[QualityRule] = []
        self._initialize_quality_rules()
        
        # 개선 조치
        self.improvement_actions: Dict[QualityIssue, QualityImprovement] = {}
        self._initialize_improvement_actions()
        
        # 예측 모델 파라미터
        self.prediction_weights = {
            "trend": 0.4,      # 트렌드 가중치
            "seasonal": 0.3,    # 계절성 가중치
            "recent": 0.3       # 최근 데이터 가중치
        }
        
        # 모니터링
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 통계
        self.stats = {
            "total_assessments": 0,
            "quality_improvements": 0,
            "predicted_issues": 0,
            "prevented_issues": 0
        }
        
        logger.info(f"Predictive Quality Optimizer initialized with target quality: {target_quality:.1%}")
    
    def _initialize_quality_rules(self):
        """품질 규칙 초기화"""
        
        # 완전성 규칙
        self.quality_rules.extend([
            QualityRule(
                name="required_fields_present",
                metric=QualityMetric.COMPLETENESS,
                condition=lambda data: all(key in data for key in ["title", "content", "url"]),
                weight=0.3,
                threshold=0.95,
                description="필수 필드 존재 여부"
            ),
            QualityRule(
                name="non_empty_content",
                metric=QualityMetric.COMPLETENESS,
                condition=lambda data: len(str(data.get("content", ""))) > 10,
                weight=0.2,
                threshold=0.9,
                description="콘텐츠 비어있지 않음"
            )
        ])
        
        # 정확성 규칙
        self.quality_rules.extend([
            QualityRule(
                name="valid_url_format",
                metric=QualityMetric.ACCURACY,
                condition=lambda data: str(data.get("url", "")).startswith(("http://", "https://")),
                weight=0.15,
                threshold=0.95,
                description="유효한 URL 형식"
            ),
            QualityRule(
                name="reasonable_content_length",
                metric=QualityMetric.ACCURACY,
                condition=lambda data: 50 <= len(str(data.get("content", ""))) <= 10000,
                weight=0.1,
                threshold=0.8,
                description="적절한 콘텐츠 길이"
            )
        ])
        
        # 시의성 규칙
        self.quality_rules.extend([
            QualityRule(
                name="recent_published_date",
                metric=QualityMetric.TIMELINESS,
                condition=self._is_recently_published,
                weight=0.2,
                threshold=0.7,
                description="최근 발행된 뉴스"
            )
        ])
        
        # 유효성 규칙
        self.quality_rules.extend([
            QualityRule(
                name="valid_sentiment_range",
                metric=QualityMetric.VALIDITY,
                condition=lambda data: -1.0 <= data.get("sentiment", 0) <= 1.0,
                weight=0.05,
                threshold=0.99,
                description="유효한 센티먼트 범위"
            )
        ])
    
    def _initialize_improvement_actions(self):
        """품질 개선 조치 초기화"""
        
        self.improvement_actions = {
            QualityIssue.MISSING_DATA: QualityImprovement(
                issue=QualityIssue.MISSING_DATA,
                action=self._fix_missing_data,
                priority=1,
                expected_improvement=0.2,
                description="누락된 데이터 보완"
            ),
            
            QualityIssue.INVALID_FORMAT: QualityImprovement(
                issue=QualityIssue.INVALID_FORMAT,
                action=self._fix_invalid_format,
                priority=2,
                expected_improvement=0.15,
                description="잘못된 형식 수정"
            ),
            
            QualityIssue.OUTDATED_DATA: QualityImprovement(
                issue=QualityIssue.OUTDATED_DATA,
                action=self._fix_outdated_data,
                priority=3,
                expected_improvement=0.1,
                description="오래된 데이터 처리"
            ),
            
            QualityIssue.DUPLICATE_DATA: QualityImprovement(
                issue=QualityIssue.DUPLICATE_DATA,
                action=self._fix_duplicate_data,
                priority=2,
                expected_improvement=0.05,
                description="중복 데이터 제거"
            ),
            
            QualityIssue.INCONSISTENT_DATA: QualityImprovement(
                issue=QualityIssue.INCONSISTENT_DATA,
                action=self._fix_inconsistent_data,
                priority=2,
                expected_improvement=0.1,
                description="일관성 없는 데이터 정규화"
            )
        }
    
    def _is_recently_published(self, data: Dict[str, Any]) -> bool:
        """최근 발행 여부 확인"""
        published_at = data.get("published_at")
        if not published_at:
            return False
        
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except:
                return False
        
        age = datetime.now() - published_at.replace(tzinfo=None)
        return age.total_seconds() < 86400  # 24시간 이내
    
    async def assess_quality(self, 
                           data: Dict[str, Any], 
                           component: str = "unknown") -> QualityScore:
        """
        데이터 품질 평가
        
        Args:
            data: 평가할 데이터
            component: 컴포넌트 이름
            
        Returns:
            품질 점수
        """
        start_time = time.time()
        
        # 각 메트릭별 점수 계산
        metric_scores = defaultdict(list)
        
        for rule in self.quality_rules:
            try:
                if rule.condition(data):
                    score = 1.0
                else:
                    score = 0.0
                
                metric_scores[rule.metric].append(score * rule.weight)
                
            except Exception as e:
                logger.debug(f"Quality rule '{rule.name}' failed: {e}")
                metric_scores[rule.metric].append(0.0)
        
        # 메트릭별 가중 평균
        completeness = self._calculate_weighted_average(metric_scores[QualityMetric.COMPLETENESS], 0.8)
        accuracy = self._calculate_weighted_average(metric_scores[QualityMetric.ACCURACY], 0.8)
        consistency = self._calculate_weighted_average(metric_scores[QualityMetric.CONSISTENCY], 0.9)
        timeliness = self._calculate_weighted_average(metric_scores[QualityMetric.TIMELINESS], 0.7)
        validity = self._calculate_weighted_average(metric_scores[QualityMetric.VALIDITY], 0.9)
        uniqueness = self._calculate_weighted_average(metric_scores[QualityMetric.UNIQUENESS], 0.8)
        
        # 전체 품질 점수 (가중 평균)
        overall = (
            completeness * 0.25 +
            accuracy * 0.25 +
            consistency * 0.15 +
            timeliness * 0.15 +
            validity * 0.15 +
            uniqueness * 0.05
        )
        
        quality_score = QualityScore(
            overall=overall,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness
        )
        
        # 히스토리 업데이트
        self.quality_history.append({
            "timestamp": datetime.now(),
            "component": component,
            "quality": quality_score,
            "data_size": len(str(data))
        })
        
        self.component_quality[component].append(quality_score)
        
        # 통계 업데이트
        self.stats["total_assessments"] += 1
        
        processing_time = time.time() - start_time
        logger.debug(f"Quality assessment completed in {processing_time:.3f}s: {overall:.3f}")
        
        return quality_score
    
    def _calculate_weighted_average(self, scores: List[float], default: float) -> float:
        """가중 평균 계산"""
        if not scores:
            return default
        return min(sum(scores) / len(scores), 1.0)
    
    async def improve_quality(self, 
                            data: Dict[str, Any], 
                            quality_score: QualityScore) -> Dict[str, Any]:
        """
        데이터 품질 개선
        
        Args:
            data: 개선할 데이터
            quality_score: 현재 품질 점수
            
        Returns:
            개선된 데이터
        """
        if quality_score.overall >= self.target_quality:
            return data  # 이미 목표 달성
        
        improved_data = data.copy()
        improvements_applied = []
        
        # 품질 문제 식별
        issues = self._identify_quality_issues(data, quality_score)
        
        # 우선순위 순으로 개선 조치 적용
        sorted_issues = sorted(issues, key=lambda x: self.improvement_actions[x].priority)
        
        for issue in sorted_issues:
            if issue in self.improvement_actions:
                improvement = self.improvement_actions[issue]
                
                try:
                    logger.info(f"Applying quality improvement: {improvement.description}")
                    improved_data = await improvement.action(improved_data)
                    improvements_applied.append(issue.value)
                    
                    # 개선 후 품질 재평가
                    new_quality = await self.assess_quality(improved_data)
                    if new_quality.overall >= self.target_quality:
                        break  # 목표 달성시 중단
                        
                except Exception as e:
                    logger.error(f"Quality improvement failed for {issue.value}: {e}")
        
        if improvements_applied:
            self.stats["quality_improvements"] += 1
            logger.info(f"Quality improvements applied: {improvements_applied}")
        
        return improved_data
    
    def _identify_quality_issues(self, 
                               data: Dict[str, Any], 
                               quality_score: QualityScore) -> List[QualityIssue]:
        """품질 문제 식별"""
        issues = []
        
        # 완전성 문제
        if quality_score.completeness < 0.8:
            if not data.get("title") or not data.get("content"):
                issues.append(QualityIssue.MISSING_DATA)
        
        # 정확성 문제
        if quality_score.accuracy < 0.8:
            url = data.get("url", "")
            if not url.startswith(("http://", "https://")):
                issues.append(QualityIssue.INVALID_FORMAT)
        
        # 시의성 문제
        if quality_score.timeliness < 0.7:
            issues.append(QualityIssue.OUTDATED_DATA)
        
        # 일관성 문제
        if quality_score.consistency < 0.8:
            issues.append(QualityIssue.INCONSISTENT_DATA)
        
        return issues
    
    # 품질 개선 액션들
    async def _fix_missing_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """누락된 데이터 보완"""
        improved = data.copy()
        
        # 제목 누락 처리
        if not improved.get("title"):
            content = improved.get("content", "")
            if content:
                # 첫 문장을 제목으로 사용
                first_sentence = content.split(".")[0][:100]
                improved["title"] = f"[Generated] {first_sentence}..."
            else:
                improved["title"] = "[Title Not Available]"
        
        # 콘텐츠 누락 처리
        if not improved.get("content"):
            title = improved.get("title", "")
            improved["content"] = f"[Content unavailable for: {title}]"
        
        # URL 누락 처리
        if not improved.get("url"):
            improved["url"] = "https://example.com/unavailable"
        
        # 발행일 누락 처리
        if not improved.get("published_at"):
            improved["published_at"] = datetime.now().isoformat()
        
        logger.debug("Fixed missing data fields")
        return improved
    
    async def _fix_invalid_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """잘못된 형식 수정"""
        improved = data.copy()
        
        # URL 형식 수정
        url = improved.get("url", "")
        if url and not url.startswith(("http://", "https://")):
            if url.startswith("www."):
                improved["url"] = f"https://{url}"
            elif "." in url:
                improved["url"] = f"https://{url}"
            else:
                improved["url"] = "https://example.com/corrected"
        
        # 센티먼트 범위 수정
        sentiment = improved.get("sentiment")
        if sentiment is not None:
            if sentiment > 1.0:
                improved["sentiment"] = 1.0
            elif sentiment < -1.0:
                improved["sentiment"] = -1.0
        
        # 제목/콘텐츠 길이 조정
        title = improved.get("title", "")
        if len(title) > 200:
            improved["title"] = title[:197] + "..."
        
        content = improved.get("content", "")
        if len(content) > 10000:
            improved["content"] = content[:9997] + "..."
        elif len(content) < 50 and content:
            improved["content"] = content + " [Content truncated or incomplete]"
        
        logger.debug("Fixed invalid format issues")
        return improved
    
    async def _fix_outdated_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """오래된 데이터 처리"""
        improved = data.copy()
        
        # 발행일이 너무 오래된 경우 플래그 추가
        published_at = improved.get("published_at")
        if published_at:
            try:
                if isinstance(published_at, str):
                    pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                else:
                    pub_date = published_at
                
                age_days = (datetime.now() - pub_date.replace(tzinfo=None)).days
                
                if age_days > 7:
                    improved["data_age_warning"] = f"Data is {age_days} days old"
                    improved["relevance_score"] = max(0.1, 1.0 - (age_days / 30))
                
            except Exception as e:
                logger.debug(f"Error processing publication date: {e}")
        
        logger.debug("Processed outdated data")
        return improved
    
    async def _fix_duplicate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """중복 데이터 제거"""
        improved = data.copy()
        
        # 중복 마커 추가 (실제 중복 제거는 상위 시스템에서 처리)
        content = improved.get("content", "")
        title = improved.get("title", "")
        
        # 간단한 중복 감지 (제목과 내용의 유사성)
        if title and content and title.lower() in content.lower():
            improved["potential_duplicate"] = True
            improved["duplicate_confidence"] = 0.8
        
        logger.debug("Processed potential duplicate data")
        return improved
    
    async def _fix_inconsistent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """일관성 없는 데이터 정규화"""
        improved = data.copy()
        
        # 소스 이름 정규화
        source = improved.get("source", "")
        if source:
            source_mapping = {
                "google": "google_news",
                "yahoo": "yahoo_finance",
                "reuters.com": "reuters",
                "bloomberg.com": "bloomberg"
            }
            normalized_source = source_mapping.get(source.lower(), source)
            improved["source"] = normalized_source
        
        # 카테고리 정규화
        category = improved.get("category", "")
        if category:
            category_mapping = {
                "crypto": "cryptocurrency",
                "btc": "bitcoin",
                "eth": "ethereum",
                "finance": "financial",
                "tech": "technology"
            }
            normalized_category = category_mapping.get(category.lower(), category)
            improved["category"] = normalized_category
        
        # 텍스트 정규화
        for field in ["title", "content"]:
            text = improved.get(field, "")
            if text:
                # 기본적인 텍스트 정리
                text = text.strip()
                text = " ".join(text.split())  # 중복 공백 제거
                improved[field] = text
        
        logger.debug("Normalized inconsistent data")
        return improved
    
    def predict_quality_trend(self, 
                            component: str = None, 
                            forecast_minutes: int = 30) -> Dict[str, Any]:
        """
        품질 트렌드 예측
        
        Args:
            component: 특정 컴포넌트 (None이면 전체)
            forecast_minutes: 예측 시간 (분)
            
        Returns:
            예측 결과
        """
        if component:
            history = list(self.component_quality[component])
        else:
            history = [entry["quality"] for entry in self.quality_history]
        
        if len(history) < 5:
            return {
                "predicted_quality": self.target_quality,
                "confidence": 0.1,
                "trend": "insufficient_data",
                "recommended_actions": ["Collect more data for prediction"]
            }
        
        # 최근 품질 점수들
        recent_scores = [q.overall for q in history[-20:]]  # 최근 20개
        
        # 트렌드 계산
        if len(recent_scores) >= 3:
            trend_slope = self._calculate_trend_slope(recent_scores)
            
            # 예측
            current_quality = recent_scores[-1]
            predicted_change = trend_slope * (forecast_minutes / 10)  # 10분당 변화율
            predicted_quality = max(0.0, min(1.0, current_quality + predicted_change))
        else:
            trend_slope = 0
            predicted_quality = recent_scores[-1] if recent_scores else 0.5
        
        # 신뢰도 계산
        quality_variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.1
        confidence = max(0.1, min(0.9, 1.0 - quality_variance))
        
        # 트렌드 분류
        if trend_slope > 0.01:
            trend = "improving"
        elif trend_slope < -0.01:
            trend = "degrading"
        else:
            trend = "stable"
        
        # 권장 조치
        recommended_actions = self._generate_trend_recommendations(
            predicted_quality, trend, component
        )
        
        prediction = {
            "predicted_quality": predicted_quality,
            "current_quality": recent_scores[-1] if recent_scores else 0,
            "confidence": confidence,
            "trend": trend,
            "trend_slope": trend_slope,
            "forecast_minutes": forecast_minutes,
            "recommended_actions": recommended_actions,
            "component": component or "all"
        }
        
        # 예측 통계 업데이트
        if predicted_quality < self.target_quality:
            self.stats["predicted_issues"] += 1
        
        return prediction
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """트렌드 기울기 계산 (단순 선형 회귀)"""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        
        # 선형 회귀 계수 계산
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        slope = numerator / denominator
        return slope
    
    def _generate_trend_recommendations(self, 
                                      predicted_quality: float,
                                      trend: str, 
                                      component: Optional[str]) -> List[str]:
        """트렌드 기반 권장 조치 생성"""
        recommendations = []
        
        if predicted_quality < self.target_quality:
            recommendations.append(f"품질이 목표치 미달 예상 ({predicted_quality:.1%} < {self.target_quality:.1%})")
            
            if trend == "degrading":
                recommendations.append("품질 악화 트렌드 감지 - 즉시 개선 조치 필요")
                recommendations.append("주요 컴포넌트의 오류율 점검")
                recommendations.append("데이터 소스 및 처리 파이프라인 검토")
            
            if component:
                recommendations.append(f"{component} 컴포넌트에 집중적인 품질 개선 적용")
        
        elif trend == "improving":
            recommendations.append("품질 개선 트렌드 유지")
            recommendations.append("현재 개선 조치 지속")
        
        else:  # stable
            recommendations.append("품질 안정 상태 유지")
            recommendations.append("정기적인 모니터링 지속")
        
        return recommendations
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 전체 품질 트렌드 예측
                prediction = self.predict_quality_trend(forecast_minutes=30)
                
                # 위험 상황 감지
                if prediction["predicted_quality"] < 0.6:  # 심각한 품질 저하 예상
                    logger.warning(f"Critical quality degradation predicted: {prediction['predicted_quality']:.1%}")
                    self.stats["predicted_issues"] += 1
                
                # 컴포넌트별 예측
                for component in self.component_quality.keys():
                    comp_prediction = self.predict_quality_trend(component, 15)
                    if comp_prediction["trend"] == "degrading":
                        logger.warning(f"Quality degradation trend detected for {component}")
                
                time.sleep(180)  # 3분마다 예측
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(180)
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """품질 대시보드 데이터 반환"""
        current_time = datetime.now()
        
        # 전체 품질 통계
        if self.quality_history:
            recent_qualities = [entry["quality"].overall for entry in list(self.quality_history)[-50:]]
            current_quality = recent_qualities[-1] if recent_qualities else 0
            avg_quality = statistics.mean(recent_qualities) if recent_qualities else 0
            quality_trend = self.predict_quality_trend(forecast_minutes=15)
        else:
            current_quality = 0
            avg_quality = 0
            quality_trend = {"trend": "no_data", "predicted_quality": 0}
        
        # 컴포넌트별 품질
        component_status = {}
        for component, qualities in self.component_quality.items():
            if qualities:
                recent_scores = [q.overall for q in list(qualities)[-10:]]
                component_status[component] = {
                    "current_quality": recent_scores[-1] if recent_scores else 0,
                    "avg_quality": statistics.mean(recent_scores) if recent_scores else 0,
                    "trend": self.predict_quality_trend(component, 10)["trend"],
                    "total_assessments": len(qualities)
                }
        
        return {
            "overall_status": {
                "current_quality": current_quality,
                "average_quality": avg_quality,
                "target_quality": self.target_quality,
                "quality_status": "GOOD" if current_quality >= self.target_quality else "NEEDS_IMPROVEMENT",
                "trend": quality_trend["trend"],
                "predicted_quality": quality_trend["predicted_quality"]
            },
            "component_status": component_status,
            "statistics": self.stats,
            "quality_distribution": self._get_quality_distribution(),
            "recent_improvements": self._get_recent_improvements(),
            "timestamp": current_time.isoformat()
        }
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """품질 분포 반환"""
        if not self.quality_history:
            return {}
        
        recent_qualities = [entry["quality"].overall for entry in list(self.quality_history)[-100:]]
        
        distribution = {
            "excellent": sum(1 for q in recent_qualities if q >= 0.9),
            "good": sum(1 for q in recent_qualities if 0.8 <= q < 0.9),
            "adequate": sum(1 for q in recent_qualities if 0.7 <= q < 0.8),
            "poor": sum(1 for q in recent_qualities if 0.5 <= q < 0.7),
            "critical": sum(1 for q in recent_qualities if q < 0.5)
        }
        
        return distribution
    
    def _get_recent_improvements(self) -> List[Dict[str, Any]]:
        """최근 개선 사항 반환"""
        # 최근 1시간 내 개선 사항
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_improvements = []
        for entry in reversed(list(self.quality_history)):
            if entry["timestamp"] < cutoff_time:
                break
            
            if entry["quality"].overall >= self.target_quality:
                recent_improvements.append({
                    "timestamp": entry["timestamp"].isoformat(),
                    "component": entry["component"],
                    "quality": entry["quality"].overall,
                    "improvement_type": "quality_target_achieved"
                })
        
        return recent_improvements[-10:]  # 최근 10개만
    
    def shutdown(self):
        """품질 최적화기 종료"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Predictive Quality Optimizer shut down")

# 전역 품질 최적화기 인스턴스
_quality_optimizer: Optional[PredictiveQualityOptimizer] = None

def get_quality_optimizer() -> PredictiveQualityOptimizer:
    """전역 품질 최적화기 반환"""
    global _quality_optimizer
    if _quality_optimizer is None:
        _quality_optimizer = PredictiveQualityOptimizer()
    return _quality_optimizer

# 테스트 코드
if __name__ == "__main__":
    async def test_predictive_quality_optimizer():
        """예측적 품질 최적화기 테스트"""
        print("=== Predictive Quality Optimizer Test ===\n")
        
        optimizer = PredictiveQualityOptimizer(target_quality=0.8)
        
        # 테스트 데이터들
        test_data = [
            {
                "title": "Federal Reserve announces interest rate decision",
                "content": "The Federal Reserve has decided to raise interest rates by 0.25% to combat inflation...",
                "url": "https://example.com/fed-decision",
                "published_at": datetime.now().isoformat(),
                "sentiment": 0.1
            },
            {
                "title": "",  # 누락된 제목
                "content": "Some content without title",
                "url": "invalid-url",  # 잘못된 URL
                "published_at": (datetime.now() - timedelta(days=30)).isoformat(),  # 오래된 데이터
                "sentiment": 1.5  # 잘못된 범위
            },
            {
                "title": "Bitcoin price analysis",
                "content": "Bitcoin shows strong momentum...",
                "url": "https://crypto-news.com/bitcoin-analysis",
                "published_at": datetime.now().isoformat(),
                "sentiment": 0.7
            }
        ]
        
        print("1. Quality Assessment Tests:")
        for i, data in enumerate(test_data, 1):
            quality = await optimizer.assess_quality(data, f"test_component_{i}")
            print(f"   Data {i}: Overall={quality.overall:.3f}, "
                  f"Completeness={quality.completeness:.3f}, "
                  f"Accuracy={quality.accuracy:.3f}, "
                  f"Timeliness={quality.timeliness:.3f}")
        
        print("\n2. Quality Improvement Tests:")
        low_quality_data = test_data[1]  # 품질이 낮은 데이터
        initial_quality = await optimizer.assess_quality(low_quality_data, "improvement_test")
        print(f"   Initial quality: {initial_quality.overall:.3f}")
        
        improved_data = await optimizer.improve_quality(low_quality_data, initial_quality)
        improved_quality = await optimizer.assess_quality(improved_data, "improvement_test")
        print(f"   Improved quality: {improved_quality.overall:.3f}")
        print(f"   Improvements: {improved_data.get('title', 'N/A')[:50]}...")
        
        print("\n3. Quality Trend Prediction:")
        # 더 많은 데이터로 히스토리 생성
        for i in range(10):
            quality_level = 0.9 - (i * 0.05)  # 점진적 품질 저하 시뮬레이션
            test_item = {
                "title": f"Test item {i}",
                "content": "Test content",
                "url": "https://example.com",
                "published_at": datetime.now().isoformat(),
                "sentiment": quality_level
            }
            await optimizer.assess_quality(test_item, "trend_test")
        
        prediction = optimizer.predict_quality_trend("trend_test", 30)
        print(f"   Current trend: {prediction['trend']}")
        print(f"   Predicted quality: {prediction['predicted_quality']:.3f}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        
        print("\n4. Dashboard Data:")
        dashboard = optimizer.get_quality_dashboard_data()
        overall = dashboard["overall_status"]
        print(f"   Overall Quality: {overall['current_quality']:.3f} ({overall['quality_status']})")
        print(f"   Trend: {overall['trend']}")
        print(f"   Total Assessments: {dashboard['statistics']['total_assessments']}")
        print(f"   Quality Improvements: {dashboard['statistics']['quality_improvements']}")
        
        optimizer.shutdown()
        print("\n✅ Predictive Quality Optimizer test completed")
    
    # 테스트 실행
    asyncio.run(test_predictive_quality_optimizer())