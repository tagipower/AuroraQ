#!/usr/bin/env python3
"""
VPS Deployment 에러 핸들링 및 복구 시스템
자동 복구, 서킷 브레이커, 재시도 로직, 장애 복구
"""

import asyncio
import functools
import json
import logging
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Type
import threading
import sys
import os


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """복구 액션"""
    RETRY = "retry"
    RESTART = "restart"
    ROLLBACK = "rollback"
    NOTIFY = "notify"
    IGNORE = "ignore"
    ESCALATE = "escalate"


class CircuitBreakerState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ErrorPattern:
    """에러 패턴 정의"""
    
    def __init__(self, 
                 error_type: Type[Exception],
                 severity: ErrorSeverity,
                 recovery_action: RecoveryAction,
                 max_retries: int = 3,
                 backoff_multiplier: float = 2.0,
                 description: str = ""):
        self.error_type = error_type
        self.severity = severity
        self.recovery_action = recovery_action
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.description = description


class CircuitBreaker:
    """서킷 브레이커 구현"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def __call__(self, func: Callable) -> Callable:
        """데코레이터로 사용"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_async(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _call_async(self, func: Callable, *args, **kwargs):
        """비동기 함수 호출"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker half-open, attempting reset")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _call_sync(self, func: Callable, *args, **kwargs):
        """동기 함수 호출"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker half-open, attempting reset")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """재시도 여부 확인"""
        return (self.last_failure_time and
                time.time() - self.last_failure_time >= self.timeout_seconds)
    
    def _on_success(self):
        """성공 시 처리"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
        self.failure_count = 0
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class RetryPolicy:
    """재시도 정책"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
    def calculate_delay(self, attempt: int) -> float:
        """재시도 지연 시간 계산"""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50%-100% 범위
        
        return delay


class ErrorRecoverySystem:
    """에러 복구 시스템"""
    
    def __init__(self):
        self.error_patterns = {}
        self.error_history = deque(maxlen=1000)
        self.recovery_actions = {}
        self.circuit_breakers = {}
        self.retry_policies = {}
        
        self.logger = logging.getLogger(__name__)
        
        # 기본 에러 패턴 등록
        self._register_default_patterns()
        
        # 기본 복구 액션 등록
        self._register_default_recovery_actions()
    
    def _register_default_patterns(self):
        """기본 에러 패턴 등록"""
        patterns = [
            ErrorPattern(
                error_type=ConnectionError,
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.RETRY,
                max_retries=5,
                description="네트워크 연결 오류"
            ),
            ErrorPattern(
                error_type=TimeoutError,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                max_retries=3,
                description="타임아웃 오류"
            ),
            ErrorPattern(
                error_type=MemoryError,
                severity=ErrorSeverity.CRITICAL,
                recovery_action=RecoveryAction.RESTART,
                max_retries=1,
                description="메모리 부족 오류"
            ),
            ErrorPattern(
                error_type=ValueError,
                severity=ErrorSeverity.LOW,
                recovery_action=RecoveryAction.NOTIFY,
                max_retries=0,
                description="잘못된 값 오류"
            ),
            ErrorPattern(
                error_type=KeyError,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                max_retries=2,
                description="키 오류"
            )
        ]
        
        for pattern in patterns:
            self.register_error_pattern(pattern)
    
    def _register_default_recovery_actions(self):
        """기본 복구 액션 등록"""
        self.recovery_actions[RecoveryAction.RETRY] = self._retry_action
        self.recovery_actions[RecoveryAction.RESTART] = self._restart_action
        self.recovery_actions[RecoveryAction.ROLLBACK] = self._rollback_action
        self.recovery_actions[RecoveryAction.NOTIFY] = self._notify_action
        self.recovery_actions[RecoveryAction.IGNORE] = self._ignore_action
        self.recovery_actions[RecoveryAction.ESCALATE] = self._escalate_action
    
    def register_error_pattern(self, pattern: ErrorPattern):
        """에러 패턴 등록"""
        self.error_patterns[pattern.error_type] = pattern
        self.logger.info(f"에러 패턴 등록: {pattern.error_type.__name__} -> {pattern.recovery_action.value}")
    
    def register_recovery_action(self, action: RecoveryAction, handler: Callable):
        """복구 액션 등록"""
        self.recovery_actions[action] = handler
        self.logger.info(f"복구 액션 등록: {action.value}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에러 처리"""
        context = context or {}
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'severity': ErrorSeverity.MEDIUM.value,
            'recovery_action': RecoveryAction.NOTIFY.value,
            'recovery_result': None
        }
        
        # 에러 패턴 매칭
        pattern = self._match_error_pattern(error)
        if pattern:
            error_info['severity'] = pattern.severity.value
            error_info['recovery_action'] = pattern.recovery_action.value
            error_info['pattern_description'] = pattern.description
            
            # 복구 액션 실행
            try:
                recovery_result = self._execute_recovery_action(pattern, error, context)
                error_info['recovery_result'] = recovery_result
            except Exception as recovery_error:
                self.logger.error(f"복구 액션 실행 실패: {recovery_error}")
                error_info['recovery_error'] = str(recovery_error)
        
        # 에러 히스토리에 기록
        self.error_history.append(error_info)
        
        # 로깅
        self._log_error(error_info)
        
        return error_info
    
    def _match_error_pattern(self, error: Exception) -> Optional[ErrorPattern]:
        """에러 패턴 매칭"""
        error_type = type(error)
        
        # 정확한 타입 매칭
        if error_type in self.error_patterns:
            return self.error_patterns[error_type]
        
        # 상위 클래스 매칭
        for pattern_type, pattern in self.error_patterns.items():
            if isinstance(error, pattern_type):
                return pattern
        
        return None
    
    def _execute_recovery_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Any:
        """복구 액션 실행"""
        action = pattern.recovery_action
        
        if action in self.recovery_actions:
            handler = self.recovery_actions[action]
            return handler(pattern, error, context)
        else:
            self.logger.warning(f"알 수 없는 복구 액션: {action}")
            return None
    
    def _retry_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """재시도 액션"""
        max_retries = pattern.max_retries
        backoff_multiplier = pattern.backoff_multiplier
        
        retry_result = {
            'action': 'retry',
            'max_retries': max_retries,
            'backoff_multiplier': backoff_multiplier,
            'message': f"최대 {max_retries}회 재시도 예정"
        }
        
        self.logger.info(f"재시도 액션: {error} (최대 {max_retries}회)")
        return retry_result
    
    def _restart_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """재시작 액션"""
        restart_result = {
            'action': 'restart',
            'message': '시스템 재시작 권장',
            'critical': True
        }
        
        self.logger.critical(f"재시작 액션: {error}")
        
        # 실제 재시작은 외부에서 처리
        # 여기서는 권장사항만 반환
        
        return restart_result
    
    def _rollback_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """롤백 액션"""
        rollback_result = {
            'action': 'rollback',
            'message': '이전 상태로 롤백 권장',
            'context': context
        }
        
        self.logger.warning(f"롤백 액션: {error}")
        return rollback_result
    
    def _notify_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """알림 액션"""
        notify_result = {
            'action': 'notify',
            'message': f'에러 발생: {error}',
            'severity': pattern.severity.value
        }
        
        self.logger.info(f"알림 액션: {error}")
        return notify_result
    
    def _ignore_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """무시 액션"""
        ignore_result = {
            'action': 'ignore',
            'message': '에러 무시됨'
        }
        
        self.logger.debug(f"무시 액션: {error}")
        return ignore_result
    
    def _escalate_action(self, pattern: ErrorPattern, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """에스컬레이션 액션"""
        escalate_result = {
            'action': 'escalate',
            'message': '상위 레벨로 에스컬레이션',
            'severity': pattern.severity.value,
            'requires_attention': True
        }
        
        self.logger.error(f"에스컬레이션 액션: {error}")
        return escalate_result
    
    def _log_error(self, error_info: Dict[str, Any]):
        """에러 로깅"""
        severity = error_info['severity']
        
        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(f"CRITICAL ERROR: {error_info['error_message']}")
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(f"HIGH ERROR: {error_info['error_message']}")
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(f"MEDIUM ERROR: {error_info['error_message']}")
        else:
            self.logger.info(f"LOW ERROR: {error_info['error_message']}")
    
    def with_error_handling(self, context: Dict[str, Any] = None):
        """에러 핸들링 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_info = self.handle_error(e, context)
                    
                    # 심각한 에러는 재발생
                    if error_info['severity'] in [ErrorSeverity.CRITICAL.value, ErrorSeverity.HIGH.value]:
                        raise
                    
                    return None
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_info = self.handle_error(e, context)
                    
                    # 심각한 에러는 재발생
                    if error_info['severity'] in [ErrorSeverity.CRITICAL.value, ErrorSeverity.HIGH.value]:
                        raise
                    
                    return None
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def with_retry(self, max_attempts: int = 3, base_delay: float = 1.0):
        """재시도 데코레이터"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                policy = RetryPolicy(max_attempts=max_attempts, base_delay=base_delay)
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            self.handle_error(e, {'function': func.__name__, 'attempt': attempt + 1})
                            raise
                        
                        delay = policy.calculate_delay(attempt)
                        self.logger.warning(f"재시도 {attempt + 1}/{max_attempts} (지연: {delay:.2f}s): {e}")
                        await asyncio.sleep(delay)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                policy = RetryPolicy(max_attempts=max_attempts, base_delay=base_delay)
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            self.handle_error(e, {'function': func.__name__, 'attempt': attempt + 1})
                            raise
                        
                        delay = policy.calculate_delay(attempt)
                        self.logger.warning(f"재시도 {attempt + 1}/{max_attempts} (지연: {delay:.2f}s): {e}")
                        time.sleep(delay)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def error_context(self, context: Dict[str, Any] = None):
        """에러 컨텍스트 매니저"""
        try:
            yield
        except Exception as e:
            self.handle_error(e, context)
            raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계"""
        if not self.error_history:
            return {"message": "No error data available"}
        
        errors = list(self.error_history)
        
        # 에러 타입별 통계
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        recovery_actions = defaultdict(int)
        
        for error in errors:
            error_types[error['error_type']] += 1
            severity_counts[error['severity']] += 1
            recovery_actions[error['recovery_action']] += 1
        
        # 시간대별 통계 (최근 24시간)
        now = datetime.now()
        recent_errors = [e for e in errors 
                        if datetime.fromisoformat(e['timestamp']) > now - timedelta(hours=24)]
        
        return {
            'total_errors': len(errors),
            'recent_errors_24h': len(recent_errors),
            'error_types': dict(error_types),
            'severity_distribution': dict(severity_counts),
            'recovery_actions_used': dict(recovery_actions),
            'error_rate_per_hour': len(recent_errors) / 24 if recent_errors else 0,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            'patterns_registered': len(self.error_patterns)
        }
    
    def export_error_report(self, filename: str = None) -> str:
        """에러 리포트 내보내기"""
        if filename is None:
            filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'error_history': list(self.error_history),
            'registered_patterns': {
                pattern_type.__name__: {
                    'severity': pattern.severity.value,
                    'recovery_action': pattern.recovery_action.value,
                    'max_retries': pattern.max_retries,
                    'description': pattern.description
                }
                for pattern_type, pattern in self.error_patterns.items()
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"에러 리포트 저장: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"에러 리포트 저장 실패: {e}")
            return ""


# 전역 에러 복구 시스템
global_error_recovery = ErrorRecoverySystem()


# 편의 함수들
def handle_errors(context: Dict[str, Any] = None):
    """에러 핸들링 데코레이터 (편의 함수)"""
    return global_error_recovery.with_error_handling(context)


def retry(max_attempts: int = 3, base_delay: float = 1.0):
    """재시도 데코레이터 (편의 함수)"""
    return global_error_recovery.with_retry(max_attempts, base_delay)


def circuit_breaker(failure_threshold: int = 5, timeout_seconds: int = 60):
    """서킷 브레이커 데코레이터 (편의 함수)"""
    return CircuitBreaker(failure_threshold, timeout_seconds)


def error_context(context: Dict[str, Any] = None):
    """에러 컨텍스트 (편의 함수)"""
    return global_error_recovery.error_context(context)


# 사용 예시
if __name__ == "__main__":
    
    # 에러 핸들링 예시
    @handle_errors(context={'module': 'example'})
    @retry(max_attempts=3)
    def example_function_with_errors():
        import random
        if random.random() < 0.7:  # 70% 확률로 실패
            raise ConnectionError("네트워크 연결 실패")
        return "성공!"
    
    @circuit_breaker(failure_threshold=3, timeout_seconds=30)
    async def example_async_function():
        import random
        if random.random() < 0.5:  # 50% 확률로 실패
            raise TimeoutError("타임아웃 발생")
        return "비동기 성공!"
    
    # 테스트 실행
    async def test_error_recovery_system():
        print("🚨 에러 복구 시스템 테스트")
        
        # 동기 함수 테스트
        for i in range(5):
            try:
                result = example_function_with_errors()
                print(f"동기 함수 결과 {i+1}: {result}")
            except Exception as e:
                print(f"동기 함수 에러 {i+1}: {e}")
        
        # 비동기 함수 테스트
        for i in range(3):
            try:
                result = await example_async_function()
                print(f"비동기 함수 결과 {i+1}: {result}")
            except Exception as e:
                print(f"비동기 함수 에러 {i+1}: {e}")
        
        # 에러 통계 출력
        stats = global_error_recovery.get_error_statistics()
        print(f"\n📊 에러 통계:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 에러 리포트 생성
        report_file = global_error_recovery.export_error_report()
        print(f"\n📄 에러 리포트: {report_file}")
    
    # 테스트 실행
    asyncio.run(test_error_recovery_system())