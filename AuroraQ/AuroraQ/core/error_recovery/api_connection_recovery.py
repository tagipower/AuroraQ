#!/usr/bin/env python3
"""
API 연결 실패 복구 메커니즘
P1-2: 에러 복구 - API 연결 전용 fallback 및 복구 시스템
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 기존 에러 복구 시스템 통합
try:
    from utils.error_recovery_system import (
        global_error_recovery, 
        ErrorSeverity, 
        RecoveryAction,
        ErrorPattern,
        CircuitBreaker,
        RetryPolicy
    )
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False

class APIConnectionState(Enum):
    """API 연결 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILING = "failing"
    OFFLINE = "offline"
    RECOVERING = "recovering"

class APIRecoveryStrategy(Enum):
    """API 복구 전략"""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_API = "fallback_api"
    CACHED_DATA = "cached_data"
    GRACEFUL_DEGRADATION = "graceful_degradation"

@dataclass
class APIEndpoint:
    """API 엔드포인트 정보"""
    name: str
    url: str
    priority: int = 1  # 1=primary, 2=secondary, 3=fallback
    timeout_seconds: float = 30.0
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    health_check_interval: int = 60
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class APIConnectionMetrics:
    """API 연결 메트릭"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    current_state: APIConnectionState = APIConnectionState.HEALTHY

class APIConnectionRecovery:
    """API 연결 복구 관리자"""
    
    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.metrics: Dict[str, APIConnectionMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.session_pool: Dict[str, aiohttp.ClientSession] = {}
        self.recovery_strategies: Dict[str, List[APIRecoveryStrategy]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # 전역 설정
        self.max_concurrent_connections = 10
        self.default_timeout = 30.0
        self.health_check_enabled = True
        self.health_check_interval = 60  # 초
        
        # 기본 복구 전략 등록
        self._register_default_recovery_strategies()
        
        # 에러 복구 시스템 통합
        if ERROR_RECOVERY_AVAILABLE:
            self._integrate_with_error_recovery_system()
    
    def _register_default_recovery_strategies(self):
        """기본 복구 전략 등록"""
        default_strategies = [
            APIRecoveryStrategy.IMMEDIATE_RETRY,
            APIRecoveryStrategy.EXPONENTIAL_BACKOFF,
            APIRecoveryStrategy.FALLBACK_API,
            APIRecoveryStrategy.CACHED_DATA,
            APIRecoveryStrategy.GRACEFUL_DEGRADATION
        ]
        
        # 모든 API에 기본 전략 적용
        self.default_recovery_strategies = default_strategies
    
    def _integrate_with_error_recovery_system(self):
        """기존 에러 복구 시스템과 통합"""
        if not ERROR_RECOVERY_AVAILABLE:
            return
        
        # API 연결 관련 에러 패턴 추가
        api_error_patterns = [
            ErrorPattern(
                error_type=aiohttp.ClientConnectorError,
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.RETRY,
                max_retries=3,
                description="API 연결 오류"
            ),
            ErrorPattern(
                error_type=aiohttp.ClientTimeout,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                max_retries=2,
                description="API 타임아웃"
            ),
            ErrorPattern(
                error_type=aiohttp.ClientResponseError,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                max_retries=2,
                description="API 응답 오류"
            )
        ]
        
        for pattern in api_error_patterns:
            global_error_recovery.register_error_pattern(pattern)
    
    def register_api_endpoint(self, endpoint: APIEndpoint):
        """API 엔드포인트 등록"""
        self.endpoints[endpoint.name] = endpoint
        self.metrics[endpoint.name] = APIConnectionMetrics()
        
        # 서킷 브레이커 설정
        self.circuit_breakers[endpoint.name] = CircuitBreaker(
            failure_threshold=endpoint.circuit_breaker_threshold,
            timeout_seconds=60,
            expected_exception=Exception
        )
        
        # 복구 전략 설정
        self.recovery_strategies[endpoint.name] = self.default_recovery_strategies.copy()
        
        self.logger.info(f"API 엔드포인트 등록: {endpoint.name} ({endpoint.url})")
    
    async def get_healthy_session(self, endpoint_name: str) -> Optional[aiohttp.ClientSession]:
        """건강한 세션 반환"""
        if endpoint_name not in self.session_pool:
            endpoint = self.endpoints.get(endpoint_name)
            if not endpoint:
                return None
            
            # 새 세션 생성
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            session = aiohttp.ClientSession(timeout=timeout)
            self.session_pool[endpoint_name] = session
        
        return self.session_pool[endpoint_name]
    
    async def make_request_with_recovery(self, 
                                       endpoint_name: str, 
                                       method: str = "GET",
                                       url_suffix: str = "",
                                       params: Dict[str, Any] = None,
                                       data: Dict[str, Any] = None,
                                       headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """복구 메커니즘이 적용된 API 요청"""
        
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint or not endpoint.is_active:
            self.logger.warning(f"비활성 엔드포인트: {endpoint_name}")
            return None
        
        metrics = self.metrics[endpoint_name]
        
        # 복구 전략 실행
        strategies = self.recovery_strategies.get(endpoint_name, self.default_recovery_strategies)
        
        for strategy in strategies:
            try:
                result = await self._execute_recovery_strategy(
                    strategy, endpoint_name, method, url_suffix, params, data, headers
                )
                
                if result:
                    # 성공 메트릭 업데이트
                    await self._update_success_metrics(endpoint_name, result)
                    return result
                    
            except Exception as e:
                self.logger.warning(f"복구 전략 {strategy.value} 실패 ({endpoint_name}): {e}")
                continue
        
        # 모든 전략 실패
        await self._update_failure_metrics(endpoint_name)
        self.logger.error(f"모든 복구 전략 실패: {endpoint_name}")
        return None
    
    async def _execute_recovery_strategy(self,
                                       strategy: APIRecoveryStrategy,
                                       endpoint_name: str,
                                       method: str,
                                       url_suffix: str,
                                       params: Dict[str, Any],
                                       data: Dict[str, Any], 
                                       headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """개별 복구 전략 실행"""
        
        if strategy == APIRecoveryStrategy.IMMEDIATE_RETRY:
            return await self._immediate_retry_strategy(
                endpoint_name, method, url_suffix, params, data, headers
            )
        
        elif strategy == APIRecoveryStrategy.EXPONENTIAL_BACKOFF:
            return await self._exponential_backoff_strategy(
                endpoint_name, method, url_suffix, params, data, headers
            )
        
        elif strategy == APIRecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_strategy(
                endpoint_name, method, url_suffix, params, data, headers
            )
        
        elif strategy == APIRecoveryStrategy.FALLBACK_API:
            return await self._fallback_api_strategy(
                endpoint_name, method, url_suffix, params, data, headers
            )
        
        elif strategy == APIRecoveryStrategy.CACHED_DATA:
            return await self._cached_data_strategy(endpoint_name)
        
        elif strategy == APIRecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_strategy(endpoint_name)
        
        return None
    
    async def _immediate_retry_strategy(self, endpoint_name: str, method: str, url_suffix: str,
                                      params: Dict[str, Any], data: Dict[str, Any], 
                                      headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """즉시 재시도 전략"""
        endpoint = self.endpoints[endpoint_name]
        
        for attempt in range(endpoint.retry_count):
            try:
                session = await self.get_healthy_session(endpoint_name)
                if not session:
                    continue
                
                full_url = endpoint.url + url_suffix
                
                async with session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.debug(f"즉시 재시도 성공: {endpoint_name} (시도 {attempt + 1})")
                        return result
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
            except Exception as e:
                if attempt == endpoint.retry_count - 1:
                    raise
                
                self.logger.debug(f"즉시 재시도 실패: {endpoint_name} (시도 {attempt + 1}): {e}")
                await asyncio.sleep(0.1)  # 100ms 대기
        
        return None
    
    async def _exponential_backoff_strategy(self, endpoint_name: str, method: str, url_suffix: str,
                                          params: Dict[str, Any], data: Dict[str, Any], 
                                          headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """지수 백오프 재시도 전략"""
        endpoint = self.endpoints[endpoint_name]
        retry_policy = RetryPolicy(
            max_attempts=endpoint.retry_count,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True
        )
        
        for attempt in range(retry_policy.max_attempts):
            try:
                session = await self.get_healthy_session(endpoint_name)
                if not session:
                    continue
                
                full_url = endpoint.url + url_suffix
                
                async with session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.debug(f"지수 백오프 성공: {endpoint_name} (시도 {attempt + 1})")
                        return result
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
            except Exception as e:
                if attempt == retry_policy.max_attempts - 1:
                    raise
                
                delay = retry_policy.calculate_delay(attempt)
                self.logger.debug(f"지수 백오프 재시도: {endpoint_name} (시도 {attempt + 1}, 지연 {delay:.1f}s): {e}")
                await asyncio.sleep(delay)
        
        return None
    
    async def _circuit_breaker_strategy(self, endpoint_name: str, method: str, url_suffix: str,
                                      params: Dict[str, Any], data: Dict[str, Any], 
                                      headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """서킷 브레이커 전략"""
        circuit_breaker = self.circuit_breakers[endpoint_name]
        
        @circuit_breaker
        async def protected_request():
            session = await self.get_healthy_session(endpoint_name)
            if not session:
                raise Exception("세션을 생성할 수 없음")
            
            endpoint = self.endpoints[endpoint_name]
            full_url = endpoint.url + url_suffix
            
            async with session.request(
                method=method,
                url=full_url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
        
        try:
            result = await protected_request()
            self.logger.debug(f"서킷 브레이커 성공: {endpoint_name}")
            return result
        except Exception as e:
            self.logger.debug(f"서킷 브레이커 실패: {endpoint_name}: {e}")
            raise
    
    async def _fallback_api_strategy(self, endpoint_name: str, method: str, url_suffix: str,
                                   params: Dict[str, Any], data: Dict[str, Any], 
                                   headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """대체 API 전략"""
        # 우선순위가 낮은 대체 엔드포인트 찾기
        primary_endpoint = self.endpoints[endpoint_name]
        fallback_endpoints = [
            ep for ep in self.endpoints.values() 
            if ep.priority > primary_endpoint.priority and ep.is_active
        ]
        
        # 우선순위 정렬
        fallback_endpoints.sort(key=lambda x: x.priority)
        
        for fallback_endpoint in fallback_endpoints:
            try:
                session = await self.get_healthy_session(fallback_endpoint.name)
                if not session:
                    continue
                
                full_url = fallback_endpoint.url + url_suffix
                
                async with session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.warning(f"대체 API 사용: {fallback_endpoint.name} (원본: {endpoint_name})")
                        
                        # 결과에 fallback 정보 추가
                        if isinstance(result, dict):
                            result['_fallback_info'] = {
                                'original_endpoint': endpoint_name,
                                'fallback_endpoint': fallback_endpoint.name,
                                'fallback_time': datetime.now().isoformat()
                            }
                        
                        return result
                        
            except Exception as e:
                self.logger.debug(f"대체 API 실패: {fallback_endpoint.name}: {e}")
                continue
        
        return None
    
    async def _cached_data_strategy(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """캐시된 데이터 전략"""
        # 이 메서드는 실제 구현에서는 Redis나 메모리 캐시와 연동
        # 여기서는 간단한 예시 구현
        cache_key = f"api_cache_{endpoint_name}"
        
        # 메타데이터에서 마지막 성공 결과 확인
        metrics = self.metrics[endpoint_name]
        if hasattr(metrics, 'last_success_data'):
            cached_data = getattr(metrics, 'last_success_data')
            if cached_data:
                self.logger.warning(f"캐시된 데이터 사용: {endpoint_name}")
                
                # 캐시 데이터에 fallback 정보 추가
                if isinstance(cached_data, dict):
                    cached_data['_fallback_info'] = {
                        'source': 'cached_data',
                        'cache_time': datetime.now().isoformat(),
                        'warning': 'This is cached data due to API failure'
                    }
                
                return cached_data
        
        return None
    
    async def _graceful_degradation_strategy(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """우아한 성능 저하 전략"""
        self.logger.warning(f"우아한 성능 저하 모드: {endpoint_name}")
        
        # 기본적인 응답 구조 반환 (서비스별로 커스터마이징 필요)
        degraded_response = {
            'status': 'degraded',
            'message': 'Service is temporarily unavailable',
            'timestamp': datetime.now().isoformat(),
            '_fallback_info': {
                'source': 'graceful_degradation',
                'reason': 'All recovery strategies failed',
                'endpoint': endpoint_name
            }
        }
        
        return degraded_response
    
    async def _update_success_metrics(self, endpoint_name: str, result: Dict[str, Any]):
        """성공 메트릭 업데이트"""
        metrics = self.metrics[endpoint_name]
        
        metrics.total_requests += 1
        metrics.successful_requests += 1
        metrics.consecutive_successes += 1
        metrics.consecutive_failures = 0
        metrics.last_success_time = datetime.now()
        
        # 상태 업데이트
        if metrics.consecutive_successes >= 3:
            metrics.current_state = APIConnectionState.HEALTHY
        elif metrics.current_state == APIConnectionState.OFFLINE:
            metrics.current_state = APIConnectionState.RECOVERING
        
        # 마지막 성공 데이터 캐시 (간단한 구현)
        setattr(metrics, 'last_success_data', result.copy())
        
        self.logger.debug(f"성공 메트릭 업데이트: {endpoint_name}")
    
    async def _update_failure_metrics(self, endpoint_name: str):
        """실패 메트릭 업데이트"""
        metrics = self.metrics[endpoint_name]
        
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.consecutive_failures += 1
        metrics.consecutive_successes = 0
        metrics.last_failure_time = datetime.now()
        
        # 상태 업데이트
        if metrics.consecutive_failures >= 5:
            metrics.current_state = APIConnectionState.OFFLINE
        elif metrics.consecutive_failures >= 3:
            metrics.current_state = APIConnectionState.FAILING
        elif metrics.consecutive_failures >= 1:
            metrics.current_state = APIConnectionState.DEGRADED
        
        self.logger.warning(f"실패 메트릭 업데이트: {endpoint_name} (연속 실패: {metrics.consecutive_failures})")
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """엔드포인트 상태 조회"""
        if endpoint_name not in self.endpoints:
            return {"error": "Endpoint not found"}
        
        endpoint = self.endpoints[endpoint_name]
        metrics = self.metrics[endpoint_name]
        
        success_rate = 0.0
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests * 100
        
        return {
            'endpoint_name': endpoint_name,
            'url': endpoint.url,
            'priority': endpoint.priority,
            'is_active': endpoint.is_active,
            'current_state': metrics.current_state.value,
            'total_requests': metrics.total_requests,
            'success_rate': round(success_rate, 2),
            'consecutive_failures': metrics.consecutive_failures,
            'consecutive_successes': metrics.consecutive_successes,
            'last_success_time': metrics.last_success_time.isoformat() if metrics.last_success_time else None,
            'last_failure_time': metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
            'avg_response_time_ms': round(metrics.avg_response_time_ms, 2)
        }
    
    def get_all_endpoints_status(self) -> Dict[str, Any]:
        """모든 엔드포인트 상태 조회"""
        return {
            endpoint_name: self.get_endpoint_status(endpoint_name)
            for endpoint_name in self.endpoints.keys()
        }
    
    async def shutdown(self):
        """리소스 정리"""
        for session in self.session_pool.values():
            if not session.closed:
                await session.close()
        
        self.session_pool.clear()
        self.logger.info("API 연결 복구 시스템 종료")

# 전역 API 연결 복구 시스템
global_api_recovery = APIConnectionRecovery()

# 편의 함수들
def register_api(name: str, url: str, priority: int = 1, **kwargs) -> APIEndpoint:
    """API 엔드포인트 등록 (편의 함수)"""
    endpoint = APIEndpoint(name=name, url=url, priority=priority, **kwargs)
    global_api_recovery.register_api_endpoint(endpoint)
    return endpoint

async def api_request(endpoint_name: str, method: str = "GET", **kwargs) -> Optional[Dict[str, Any]]:
    """API 요청 (편의 함수)"""
    return await global_api_recovery.make_request_with_recovery(
        endpoint_name=endpoint_name, 
        method=method, 
        **kwargs
    )

def get_api_status(endpoint_name: str = None) -> Dict[str, Any]:
    """API 상태 조회 (편의 함수)"""
    if endpoint_name:
        return global_api_recovery.get_endpoint_status(endpoint_name)
    else:
        return global_api_recovery.get_all_endpoints_status()

# 사용 예시 및 테스트
if __name__ == "__main__":
    
    async def test_api_recovery_system():
        print("🧪 API 연결 복구 시스템 테스트")
        
        # 테스트 API 엔드포인트 등록
        primary_api = register_api(
            name="binance_primary",
            url="https://api.binance.com/api/v3",
            priority=1,
            timeout_seconds=10
        )
        
        backup_api = register_api(
            name="coingecko_backup", 
            url="https://api.coingecko.com/api/v3",
            priority=2,
            timeout_seconds=15
        )
        
        # 정상 요청 테스트
        print("\n1️⃣ 정상 요청 테스트")
        result = await api_request(
            endpoint_name="binance_primary",
            method="GET",
            url_suffix="/time"
        )
        
        if result:
            print(f"✅ 정상 요청 성공: {result}")
        else:
            print("❌ 정상 요청 실패")
        
        # Fallback 테스트 (존재하지 않는 엔드포인트)
        print("\n2️⃣ Fallback 테스트")
        fallback_result = await api_request(
            endpoint_name="binance_primary",
            method="GET", 
            url_suffix="/nonexistent"
        )
        
        if fallback_result:
            print(f"✅ Fallback 요청 결과: {fallback_result}")
        else:
            print("❌ 모든 Fallback 실패")
        
        # 상태 확인
        print("\n3️⃣ API 상태 확인")
        status = get_api_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 종료
        await global_api_recovery.shutdown()
    
    # 테스트 실행
    asyncio.run(test_api_recovery_system())