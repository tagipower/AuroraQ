#!/usr/bin/env python3
"""
동적 배치 크기 관리자
P1-3: 성능 튜닝 - 시스템 리소스에 따른 배치 크기 동적 조정
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import gc
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceState(Enum):
    """시스템 리소스 상태"""
    OPTIMAL = "optimal"
    STRESSED = "stressed"
    CRITICAL = "critical"
    RECOVERING = "recovering"

@dataclass
class BatchConfig:
    """배치 설정"""
    initial_batch_size: int = 64
    min_batch_size: int = 8
    max_batch_size: int = 256
    target_memory_mb: float = 1500.0
    max_memory_mb: float = 2500.0
    target_cpu_percent: float = 70.0
    max_cpu_percent: float = 85.0
    target_processing_time_s: float = 2.0
    max_processing_time_s: float = 5.0
    adjustment_factor: float = 0.15  # 15% 조정

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    processing_time_s: float = 0.0
    items_processed: int = 0
    success_rate: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class DynamicBatchManager:
    """동적 배치 크기 관리자"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.current_state = ResourceState.OPTIMAL
        
        # 성능 히스토리
        self.metrics_history = []
        self.max_history_size = 100
        
        # 조정 통계
        self.adjustment_stats = {
            "total_adjustments": 0,
            "increases": 0,
            "decreases": 0,
            "last_adjustment": None,
            "state_transitions": 0
        }
        
        # 안정성 제어
        self.last_adjustment_time = 0
        self.min_adjustment_interval_s = 5.0  # 최소 5초 간격
        self.consecutive_good_batches = 0
        self.consecutive_bad_batches = 0
        
        logger.info(f"Dynamic batch manager initialized: initial={self.current_batch_size}, "
                   f"range=({self.config.min_batch_size}-{self.config.max_batch_size})")
    
    def get_current_batch_size(self) -> int:
        """현재 배치 크기 반환"""
        return self.current_batch_size
    
    def collect_system_metrics(self) -> PerformanceMetrics:
        """시스템 메트릭 수집"""
        try:
            # 메모리 사용량
            memory_info = psutil.virtual_memory()
            memory_used_mb = (memory_info.total - memory_info.available) / 1024 / 1024
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            metrics = PerformanceMetrics(
                memory_usage_mb=memory_used_mb,
                cpu_percent=cpu_percent,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return PerformanceMetrics()
    
    def update_batch_performance(self, processing_time: float, items_processed: int, 
                               success_rate: float = 1.0, custom_metrics: Dict[str, Any] = None):
        """배치 성능 업데이트"""
        try:
            # 시스템 메트릭 수집
            system_metrics = self.collect_system_metrics()
            
            # 성능 메트릭 생성
            metrics = PerformanceMetrics(
                memory_usage_mb=system_metrics.memory_usage_mb,
                cpu_percent=system_metrics.cpu_percent,
                processing_time_s=processing_time,
                items_processed=items_processed,
                success_rate=success_rate
            )
            
            # 히스토리에 추가
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            # 리소스 상태 평가
            new_state = self._evaluate_resource_state(metrics)
            if new_state != self.current_state:
                logger.info(f"Resource state changed: {self.current_state.value} -> {new_state.value}")
                self.current_state = new_state
                self.adjustment_stats["state_transitions"] += 1
            
            # 배치 크기 조정
            self._adjust_batch_size(metrics)
            
            logger.debug(f"Batch performance updated: size={self.current_batch_size}, "
                        f"time={processing_time:.2f}s, items={items_processed}, "
                        f"memory={metrics.memory_usage_mb:.1f}MB, cpu={metrics.cpu_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to update batch performance: {e}")
    
    def _evaluate_resource_state(self, metrics: PerformanceMetrics) -> ResourceState:
        """리소스 상태 평가"""
        
        # Critical 상태 검사
        if (metrics.memory_usage_mb > self.config.max_memory_mb or
            metrics.cpu_percent > self.config.max_cpu_percent or
            metrics.processing_time_s > self.config.max_processing_time_s):
            return ResourceState.CRITICAL
        
        # Stressed 상태 검사
        stress_score = 0
        if metrics.memory_usage_mb > self.config.target_memory_mb:
            stress_score += 1
        if metrics.cpu_percent > self.config.target_cpu_percent:
            stress_score += 1
        if metrics.processing_time_s > self.config.target_processing_time_s:
            stress_score += 1
        
        if stress_score >= 2:
            return ResourceState.STRESSED
        
        # Recovering 상태 검사 (이전 상태가 CRITICAL/STRESSED이고 현재 개선됨)
        if (self.current_state in [ResourceState.CRITICAL, ResourceState.STRESSED] and
            stress_score <= 1):
            return ResourceState.RECOVERING
        
        return ResourceState.OPTIMAL
    
    def _adjust_batch_size(self, metrics: PerformanceMetrics):
        """배치 크기 조정"""
        current_time = time.time()
        
        # 최소 간격 체크
        if current_time - self.last_adjustment_time < self.min_adjustment_interval_s:
            return
        
        old_size = self.current_batch_size
        adjustment_made = False
        
        # 상태별 조정 전략
        if self.current_state == ResourceState.CRITICAL:
            # 즉시 큰 폭으로 감소
            reduction = max(8, int(self.current_batch_size * 0.3))
            self.current_batch_size = max(self.config.min_batch_size, 
                                        self.current_batch_size - reduction)
            adjustment_made = True
            self.consecutive_bad_batches += 1
            self.consecutive_good_batches = 0
            
        elif self.current_state == ResourceState.STRESSED:
            # 중간 폭으로 감소
            reduction = max(2, int(self.current_batch_size * self.config.adjustment_factor))
            self.current_batch_size = max(self.config.min_batch_size,
                                        self.current_batch_size - reduction)
            adjustment_made = True
            self.consecutive_bad_batches += 1
            self.consecutive_good_batches = 0
            
        elif self.current_state == ResourceState.OPTIMAL:
            # 안정적으로 증가
            self.consecutive_good_batches += 1
            self.consecutive_bad_batches = 0
            
            # 연속으로 좋은 성능이면 증가
            if self.consecutive_good_batches >= 3:
                increase = max(1, int(self.current_batch_size * (self.config.adjustment_factor / 2)))
                self.current_batch_size = min(self.config.max_batch_size,
                                            self.current_batch_size + increase)
                adjustment_made = True
                self.consecutive_good_batches = 0
                
        elif self.current_state == ResourceState.RECOVERING:
            # 회복 중이므로 신중하게 조정
            self.consecutive_good_batches += 1
            if self.consecutive_good_batches >= 5:  # 더 오래 기다림
                increase = 1  # 아주 작게 증가
                self.current_batch_size = min(self.config.max_batch_size,
                                            self.current_batch_size + increase)
                adjustment_made = True
                self.consecutive_good_batches = 0
        
        # 조정 통계 업데이트
        if adjustment_made:
            self.last_adjustment_time = current_time
            self.adjustment_stats["total_adjustments"] += 1
            self.adjustment_stats["last_adjustment"] = datetime.now().isoformat()
            
            if self.current_batch_size > old_size:
                self.adjustment_stats["increases"] += 1
                logger.info(f"Batch size increased: {old_size} -> {self.current_batch_size} "
                           f"(state: {self.current_state.value})")
            elif self.current_batch_size < old_size:
                self.adjustment_stats["decreases"] += 1
                logger.info(f"Batch size decreased: {old_size} -> {self.current_batch_size} "
                           f"(state: {self.current_state.value})")
    
    def force_batch_size(self, size: int, reason: str = "manual"):
        """배치 크기 강제 설정"""
        old_size = self.current_batch_size
        self.current_batch_size = max(self.config.min_batch_size, 
                                    min(self.config.max_batch_size, size))
        
        logger.info(f"Batch size forced: {old_size} -> {self.current_batch_size} (reason: {reason})")
        self.adjustment_stats["total_adjustments"] += 1
        self.adjustment_stats["last_adjustment"] = datetime.now().isoformat()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # 최근 10개
        
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_time = sum(m.processing_time_s for m in recent_metrics) / len(recent_metrics)
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "current_batch_size": self.current_batch_size,
            "current_state": self.current_state.value,
            "performance": {
                "avg_memory_mb": round(avg_memory, 1),
                "avg_cpu_percent": round(avg_cpu, 1),
                "avg_processing_time_s": round(avg_time, 2),
                "avg_success_rate": round(avg_success_rate, 3)
            },
            "config": {
                "min_batch_size": self.config.min_batch_size,
                "max_batch_size": self.config.max_batch_size,
                "target_memory_mb": self.config.target_memory_mb,
                "target_cpu_percent": self.config.target_cpu_percent
            },
            "adjustment_stats": self.adjustment_stats.copy(),
            "consecutive_good_batches": self.consecutive_good_batches,
            "consecutive_bad_batches": self.consecutive_bad_batches,
            "metrics_count": len(self.metrics_history)
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 반환"""
        recommendations = []
        
        if not self.metrics_history:
            return ["데이터 수집 중... 권장사항은 나중에 제공됩니다."]
        
        summary = self.get_performance_summary()
        perf = summary["performance"]
        
        # 메모리 기반 권장사항
        if perf["avg_memory_mb"] > self.config.max_memory_mb:
            recommendations.append(f"메모리 사용량이 높습니다 ({perf['avg_memory_mb']:.1f}MB). "
                                 f"최대 배치 크기를 {self.config.max_batch_size // 2}로 낮춰보세요.")
        
        # CPU 기반 권장사항  
        if perf["avg_cpu_percent"] > self.config.max_cpu_percent:
            recommendations.append(f"CPU 사용률이 높습니다 ({perf['avg_cpu_percent']:.1f}%). "
                                 f"처리 간격을 늘리거나 배치 크기를 줄여보세요.")
        
        # 처리 시간 기반 권장사항
        if perf["avg_processing_time_s"] > self.config.max_processing_time_s:
            recommendations.append(f"처리 시간이 깁니다 ({perf['avg_processing_time_s']:.2f}s). "
                                 f"배치 크기를 줄이거나 알고리즘을 최적화하세요.")
        
        # 성공률 기반 권장사항
        if perf["avg_success_rate"] < 0.95:
            recommendations.append(f"성공률이 낮습니다 ({perf['avg_success_rate']:.1%}). "
                                 f"에러 처리 로직을 점검하세요.")
        
        # 조정 빈도 기반 권장사항
        total_adjustments = self.adjustment_stats["total_adjustments"]
        if total_adjustments > 20:
            decrease_ratio = self.adjustment_stats["decreases"] / total_adjustments
            if decrease_ratio > 0.7:
                recommendations.append("배치 크기가 자주 감소합니다. 초기 배치 크기를 낮춰보세요.")
        
        if not recommendations:
            recommendations.append("현재 성능이 양호합니다. 계속 모니터링하세요.")
        
        return recommendations
    
    async def cleanup_resources(self):
        """리소스 정리"""
        try:
            # 강제 가비지 컬렉션
            gc.collect()
            
            # 메모리 사용량 확인
            memory_info = psutil.virtual_memory()
            memory_used_mb = (memory_info.total - memory_info.available) / 1024 / 1024
            
            logger.info(f"Resource cleanup completed. Memory usage: {memory_used_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

# 전역 배치 관리자 인스턴스
_global_batch_manager = None

def get_batch_manager(config: BatchConfig = None) -> DynamicBatchManager:
    """전역 배치 관리자 반환"""
    global _global_batch_manager
    if _global_batch_manager is None:
        _global_batch_manager = DynamicBatchManager(config)
    return _global_batch_manager

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    import random
    
    async def test_dynamic_batch_manager():
        print("🧪 Dynamic Batch Manager 테스트")
        
        # 커스텀 설정으로 매니저 생성
        config = BatchConfig(
            initial_batch_size=32,
            min_batch_size=4,
            max_batch_size=128,
            target_memory_mb=1000.0,
            max_memory_mb=1500.0
        )
        
        manager = DynamicBatchManager(config)
        
        # 시뮬레이션된 배치 처리
        for i in range(20):
            # 랜덤한 성능 메트릭 시뮬레이션
            processing_time = random.uniform(0.5, 6.0)
            items_processed = manager.get_current_batch_size()
            success_rate = random.uniform(0.85, 1.0)
            
            print(f"\n배치 {i+1}: size={manager.get_current_batch_size()}, "
                  f"time={processing_time:.2f}s, items={items_processed}")
            
            # 성능 업데이트
            manager.update_batch_performance(processing_time, items_processed, success_rate)
            
            # 요약 정보
            if i % 5 == 4:  # 5번마다 요약
                summary = manager.get_performance_summary()
                print(f"  📊 상태: {summary['current_state']}")
                print(f"  📈 조정 횟수: {summary['adjustment_stats']['total_adjustments']}")
                
            await asyncio.sleep(0.1)
        
        # 최종 요약
        print("\n🎯 최종 성능 요약:")
        summary = manager.get_performance_summary()
        print(f"  현재 배치 크기: {summary['current_batch_size']}")
        print(f"  현재 상태: {summary['current_state']}")
        print(f"  평균 메모리: {summary['performance']['avg_memory_mb']:.1f}MB")
        print(f"  평균 CPU: {summary['performance']['avg_cpu_percent']:.1f}%")
        print(f"  총 조정 횟수: {summary['adjustment_stats']['total_adjustments']}")
        
        # 권장사항
        print("\n💡 최적화 권장사항:")
        recommendations = manager.get_optimization_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # 리소스 정리
        await manager.cleanup_resources()
    
    # 테스트 실행
    asyncio.run(test_dynamic_batch_manager())