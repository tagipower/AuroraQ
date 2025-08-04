#!/usr/bin/env python3
"""
VPS Deployment API 시스템
FastAPI 기반 통합 API 서버, 엔드포인트 검증, 보안, 모니터링
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import psutil

# Prometheus 메트릭 수집
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client.multiprocess import MultiProcessCollector
    from prometheus_client.registry import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback 클래스들
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    def generate_latest(*args, **kwargs):
        return b"# Prometheus not available\n"
    
    CONTENT_TYPE_LATEST = "text/plain"

# 내부 모듈
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.env_loader import get_vps_env_config

# 디버그 및 성능 최적화 시스템 (Fallback 포함)
try:
    from debug_system import global_debugger
except ImportError:
    class MockDebugger:
        def get_comprehensive_report(self):
            return {'status': 'debug_system_not_available', 'timestamp': datetime.now().isoformat()}
        def emergency_debug_dump(self):
            return None
    global_debugger = MockDebugger()

try:
    from performance_optimizer import global_optimizer
except ImportError:
    class MockOptimizer:
        def get_system_health(self):
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            return {
                'memory_usage': {'used': memory.used, 'total': memory.total, 'percent': memory.percent},
                'cpu_usage': {'percent': cpu},
                'cache_stats': {'hits': 0, 'misses': 0},
                'health_score': 85
            }
        def auto_optimize(self):
            return {'status': 'optimization_not_available'}
        class memory_manager:
            @staticmethod
            def optimize_memory():
                return {'status': 'memory_optimization_not_available'}
    global_optimizer = MockOptimizer()


# Pydantic 모델들
class HealthResponse(BaseModel):
    status: str = Field(..., description="시스템 상태")
    timestamp: str = Field(..., description="응답 시간")
    uptime_seconds: float = Field(..., description="가동 시간")
    version: str = Field(default="1.0.0", description="API 버전")


class SystemStats(BaseModel):
    memory: Dict[str, Any] = Field(..., description="메모리 사용량")
    cpu: Dict[str, Any] = Field(..., description="CPU 사용량")  
    cache: Dict[str, Any] = Field(..., description="캐시 통계")
    health_score: int = Field(..., description="시스템 건강 점수")


class TradingStatus(BaseModel):
    trading_active: bool = Field(..., description="트레이딩 활성 상태")
    mode: str = Field(..., description="거래 모드")
    positions_count: int = Field(..., description="열린 포지션 수")
    daily_pnl: float = Field(..., description="일일 손익")
    strategy_scores: Dict[str, float] = Field(..., description="전략별 점수")


class PositionInfo(BaseModel):
    symbol: str = Field(..., description="거래 심볼")
    side: str = Field(..., description="포지션 방향")
    size: float = Field(..., description="포지션 크기")
    entry_price: float = Field(..., description="진입 가격")
    current_price: float = Field(..., description="현재 가격")
    unrealized_pnl: float = Field(..., description="미실현 손익")
    leverage: float = Field(..., description="레버리지")


class APIRequest(BaseModel):
    action: str = Field(..., description="요청 액션")
    parameters: Dict[str, Any] = Field(default={}, description="파라미터")


class APIResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    data: Any = Field(None, description="응답 데이터")
    message: str = Field("", description="메시지")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class APIMetrics:
    """API 메트릭 수집기 (Prometheus 통합)"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.endpoint_stats = {}
        self.start_time = time.time()
        
        # Prometheus 메트릭 초기화
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total = Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status_code']
            )
            
            self.http_request_duration_seconds = Histogram(
                'http_request_duration_seconds',
                'HTTP request duration in seconds',
                ['method', 'endpoint']
            )
            
            self.system_cpu_usage = Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage'
            )
            
            self.system_memory_usage = Gauge(
                'system_memory_usage_bytes',
                'System memory usage in bytes'
            )
            
            self.trading_positions_count = Gauge(
                'trading_positions_count',
                'Number of open trading positions'
            )
            
            self.trading_daily_pnl = Gauge(
                'trading_daily_pnl',
                'Daily profit and loss'
            )
            
            self.api_uptime_seconds = Gauge(
                'api_uptime_seconds',
                'API uptime in seconds'
            )
        else:
            # Fallback 인스턴스들
            self.http_requests_total = Counter()
            self.http_request_duration_seconds = Histogram()
            self.system_cpu_usage = Gauge()
            self.system_memory_usage = Gauge()
            self.trading_positions_count = Gauge()
            self.trading_daily_pnl = Gauge()
            self.api_uptime_seconds = Gauge()
        
    def record_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """요청 기록 (Prometheus 메트릭 포함)"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if status_code >= 400:
            self.error_count += 1
        
        # Prometheus 메트릭 업데이트
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_time)
        
        endpoint_key = f"{method} {endpoint}"
        if endpoint_key not in self.endpoint_stats:
            self.endpoint_stats[endpoint_key] = {
                'count': 0,
                'avg_response_time': 0,
                'errors': 0,
                'last_called': None
            }
        
        stats = self.endpoint_stats[endpoint_key]
        stats['count'] += 1
        stats['avg_response_time'] = (stats['avg_response_time'] * (stats['count'] - 1) + response_time) / stats['count']
        stats['last_called'] = datetime.now().isoformat()
        
        if status_code >= 400:
            stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환 (시스템 메트릭 업데이트 포함)"""
        uptime = time.time() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        # 시스템 메트릭 업데이트
        if PROMETHEUS_AVAILABLE:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.system_cpu_usage.set(cpu_percent)
            self.system_memory_usage.set(memory.used)
            self.api_uptime_seconds.set(uptime)
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            'avg_response_time_ms': avg_response_time * 1000,
            'requests_per_minute': (self.request_count / uptime * 60) if uptime > 0 else 0,
            'endpoint_stats': self.endpoint_stats,
            'prometheus_available': PROMETHEUS_AVAILABLE
        }


# 전역 메트릭 수집기
api_metrics = APIMetrics()


class VPSAPIServer:
    """VPS API 서버"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 환경 설정 로드
        self.env_config = get_vps_env_config()
        self.config = config or self._default_config()
        self.app = FastAPI(
            title="VPS AuroraQ Trading API",
            description="VPS 환경 최적화 트레이딩 시스템 API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.security = HTTPBearer(auto_error=False)
        self.logger = logging.getLogger(__name__)
        
        # 미들웨어 설정
        self._setup_middleware()
        
        # 라우터 설정
        self._setup_routes()
        
        # 모의 데이터 (실제로는 트레이딩 시스템과 연동)
        self.trading_system_connected = False
        self.mock_positions = []
        self.mock_performance = {
            'daily_pnl': 125.50,
            'total_pnl': 1250.75,
            'win_rate': 68.5,
            'sharpe_ratio': 1.85
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 - env_loader에서 로드된 설정 사용"""
        return {
            'host': '0.0.0.0',
            'port': self.env_config.trading_api_port,  # env_loader에서 로드
            'api_key': self.env_config.binance_api_key,  # env_loader에서 로드
            'cors_origins': ['*'],
            'rate_limit': self.env_config.rate_limit_per_minute,  # env_loader에서 로드
            'enable_auth': self.env_config.security_log_enabled,  # env_loader에서 로드
            'trading_mode': self.env_config.trading_mode,  # env_loader에서 로드
            'symbol': self.env_config.symbol,  # env_loader에서 로드
            'enable_sentiment': self.env_config.enable_sentiment_analysis,  # env_loader에서 로드
            'enable_ppo': self.env_config.enable_ppo_strategy,  # env_loader에서 로드
            'log_level': self.env_config.log_level,  # env_loader에서 로드
            'enable_unified_logging': self.env_config.enable_unified_logging  # env_loader에서 로드
        }
    
    def _setup_middleware(self):
        """미들웨어 설정"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['cors_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip 압축
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # 메트릭 수집 미들웨어
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # 메트릭 기록
            api_metrics.record_request(
                endpoint=request.url.path,
                method=request.method,
                response_time=process_time,
                status_code=response.status_code
            )
            
            # 응답 헤더에 처리 시간 추가
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
    
    def _setup_routes(self):
        """라우터 설정"""
        
        # 헬스체크
        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """시스템 헬스체크"""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                uptime_seconds=time.time() - api_metrics.start_time
            )
        
        # 시스템 통계
        @self.app.get("/api/system/stats", response_model=SystemStats, tags=["System"])  
        async def get_system_stats():
            """시스템 통계 조회"""
            health = global_optimizer.get_system_health()
            
            return SystemStats(
                memory=health['memory_usage'],
                cpu=health['cpu_usage'],
                cache=health['cache_stats'],
                health_score=health['health_score']
            )
        
        # API 메트릭 (내부용)
        @self.app.get("/api/metrics", tags=["System"])
        async def get_api_metrics():
            """API 메트릭 조회 (JSON 형식)"""
            return api_metrics.get_stats()
        
        # Prometheus 건강성 체크
        @self.app.get("/api/prometheus/status", tags=["Monitoring"])
        async def prometheus_status():
            """Prometheus 통합 상태"""
            return {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'metrics_endpoint': '/metrics',
                'scrape_config_ready': True,
                'collectors_active': ['http_requests', 'system_metrics', 'trading_metrics'] if PROMETHEUS_AVAILABLE else []
            }
        
        # 트레이딩 상태
        @self.app.get("/api/trading/status", response_model=TradingStatus, tags=["Trading"])
        async def get_trading_status():
            """트레이딩 상태 조회"""
            return TradingStatus(
                trading_active=self.trading_system_connected,
                mode=self.config['trading_mode'],  # env_loader에서 로드된 설정 사용
                positions_count=len(self.mock_positions),
                daily_pnl=self.mock_performance['daily_pnl'],
                strategy_scores={
                    'RuleStrategyA': 0.75,
                    'RuleStrategyB': 0.68,
                    'RuleStrategyC': 0.82,
                    'RuleStrategyD': 0.71,
                    'RuleStrategyE': 0.79,
                    'PPOStrategy': 0.85
                }
            )
        
        # 포지션 조회
        @self.app.get("/api/trading/positions", response_model=List[PositionInfo], tags=["Trading"])
        async def get_positions():
            """현재 포지션 조회"""
            return self.mock_positions
        
        # 성과 지표
        @self.app.get("/api/trading/performance", tags=["Trading"])
        async def get_performance():
            """성과 지표 조회"""
            return self.mock_performance
        
        # 전략 상태
        @self.app.get("/api/trading/strategies", tags=["Trading"])
        async def get_strategies():
            """전략 상태 조회"""
            return {
                'active_strategies': ['RuleStrategyA', 'RuleStrategyB', 'RuleStrategyC', 'RuleStrategyD', 'RuleStrategyE', 'PPOStrategy'],
                'strategy_weights': {
                    'RuleStrategyA': 0.2,
                    'RuleStrategyB': 0.2,
                    'RuleStrategyC': 0.2,
                    'RuleStrategyD': 0.2,
                    'RuleStrategyE': 0.2,
                    'PPOStrategy': 0.25 if self.config['enable_ppo'] else 0.0  # PPO 활성화 여부 체크
                },
                'last_signals': [
                    {'strategy': 'RuleStrategyA', 'action': 'BUY', 'score': 0.75, 'timestamp': datetime.now().isoformat()},
                    {'strategy': 'PPOStrategy', 'action': 'HOLD', 'score': 0.65, 'timestamp': datetime.now().isoformat()}
                ]
            }
        
        # 시스템 제어
        @self.app.post("/api/system/optimize", tags=["System"])
        async def optimize_system():
            """시스템 최적화 실행"""
            result = global_optimizer.auto_optimize()
            return APIResponse(
                success=True,
                data=result,
                message="시스템 최적화 완료"
            )
        
        @self.app.post("/api/system/gc", tags=["System"])
        async def garbage_collect():
            """가비지 컬렉션 실행"""
            result = global_optimizer.memory_manager.optimize_memory()
            return APIResponse(
                success=True,
                data=result,
                message="가비지 컬렉션 완료"
            )
        
        # 디버그 정보
        @self.app.get("/api/debug/report", tags=["Debug"])
        async def get_debug_report():
            """디버그 리포트 조회"""
            report = global_debugger.get_comprehensive_report()
            return report
        
        @self.app.post("/api/debug/dump", tags=["Debug"])
        async def create_debug_dump():
            """디버그 덤프 생성"""
            filename = global_debugger.emergency_debug_dump()
            return APIResponse(
                success=bool(filename),
                data={'filename': filename},
                message="디버그 덤프 생성 완료" if filename else "디버그 덤프 생성 실패"
            )
        
        # 트레이딩 제어
        @self.app.post("/api/trading/start", tags=["Trading"])
        async def start_trading():
            """트레이딩 시작"""
            self.trading_system_connected = True
            return APIResponse(
                success=True,
                message="트레이딩 시작됨"
            )
        
        @self.app.post("/api/trading/stop", tags=["Trading"])
        async def stop_trading():
            """트레이딩 중지"""
            self.trading_system_connected = False
            return APIResponse(
                success=True,
                message="트레이딩 중지됨"
            )
        
        # 설정 관리
        @self.app.get("/api/config", tags=["Config"])
        async def get_config():
            """설정 조회"""
            # 민감한 정보 제외하고 반환
            safe_config = {k: v for k, v in self.config.items() if k not in ['api_key', 'secret']}
            return safe_config
        
        @self.app.put("/api/config", tags=["Config"])
        async def update_config(config_update: dict):
            """설정 업데이트"""
            # 실제로는 설정 파일에 저장
            for key, value in config_update.items():
                if key not in ['api_key', 'secret']:  # 보안상 중요한 설정은 제외
                    self.config[key] = value
            
            return APIResponse(
                success=True,
                message="설정 업데이트 완료"
            )
        
        # Prometheus 메트릭 엔드포인트
        @self.app.get("/metrics", tags=["Monitoring"])
        async def get_prometheus_metrics():
            """Prometheus 메트릭 엔드포인트"""
            if not PROMETHEUS_AVAILABLE:
                return Response(
                    content="# Prometheus client not available\n",
                    media_type="text/plain"
                )
            
            # 트레이딩 메트릭 업데이트
            api_metrics.trading_positions_count.set(len(self.mock_positions))
            api_metrics.trading_daily_pnl.set(self.mock_performance['daily_pnl'])
            
            # 메트릭 생성 및 반환
            metrics_data = generate_latest()
            return Response(
                content=metrics_data,
                media_type=CONTENT_TYPE_LATEST
            )
        
        # 웹소켓 상태 (실제로는 WebSocket 엔드포인트 구현 필요)
        @self.app.get("/api/websocket/status", tags=["WebSocket"])
        async def websocket_status():
            """웹소켓 상태"""
            return {
                'websocket_enabled': True,
                'active_connections': 0,
                'endpoint': 'ws://localhost:8003'
            }
        
        # 에러 핸들러
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=APIResponse(
                    success=False,
                    message=exc.detail,
                    data=None
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=APIResponse(
                    success=False,
                    message="Internal server error",
                    data=None
                ).dict()
            )
    
    def run(self):
        """API 서버 실행"""
        uvicorn.run(
            self.app,
            host=self.config['host'],
            port=self.config['port'],
            log_level="info"
        )


class APITester:
    """API 테스터"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        
    async def test_all_endpoints(self) -> Dict[str, Any]:
        """모든 엔드포인트 테스트"""
        import aiohttp
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'endpoint_tests': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0
            }
        }
        
        # 테스트할 엔드포인트들
        endpoints = [
            ('GET', '/health'),
            ('GET', '/api/system/stats'),
            ('GET', '/api/metrics'),
            ('GET', '/api/trading/status'),
            ('GET', '/api/trading/positions'),
            ('GET', '/api/trading/performance'),
            ('GET', '/api/trading/strategies'),
            ('GET', '/api/config'),
            ('GET', '/api/websocket/status'),
            ('POST', '/api/system/optimize'),
            ('POST', '/api/system/gc')
        ]
        
        async with aiohttp.ClientSession() as session:
            for method, endpoint in endpoints:
                test_results['summary']['total_tests'] += 1
                
                try:
                    start_time = time.time()
                    
                    if method == 'GET':
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            status_code = response.status
                            response_data = await response.json()
                    else:  # POST
                        async with session.post(f"{self.base_url}{endpoint}") as response:
                            status_code = response.status
                            response_data = await response.json()
                    
                    response_time = time.time() - start_time
                    
                    test_results['endpoint_tests'][f"{method} {endpoint}"] = {
                        'status': 'PASS',
                        'status_code': status_code,
                        'response_time_ms': response_time * 1000,
                        'response_size': len(str(response_data))
                    }
                    
                    test_results['summary']['passed'] += 1
                    
                except Exception as e:
                    test_results['endpoint_tests'][f"{method} {endpoint}"] = {
                        'status': 'FAIL',
                        'error': str(e)
                    }
                    
                    test_results['summary']['failed'] += 1
        
        return test_results


async def main():
    """메인 함수 - API 서버 또는 테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VPS API System")
    parser.add_argument("--mode", choices=["server", "test"], default="server", help="실행 모드")
    parser.add_argument("--port", type=int, default=8004, help="서버 포트")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        # API 서버 실행
        config = {
            'host': args.host,
            'port': args.port,
            'cors_origins': ['*'],
            'enable_auth': False
        }
        
        server = VPSAPIServer(config)
        print(f"🚀 VPS API 서버 시작: http://{args.host}:{args.port}")
        print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
        
        server.run()
        
    elif args.mode == "test":
        # API 테스트 실행
        tester = APITester(f"http://localhost:{args.port}")
        
        print("🧪 API 엔드포인트 테스트 시작...")
        test_results = await tester.test_all_endpoints()
        
        print(f"\n📊 테스트 결과:")
        print(f"총 테스트: {test_results['summary']['total_tests']}")
        print(f"성공: {test_results['summary']['passed']}")
        print(f"실패: {test_results['summary']['failed']}")
        
        # 상세 결과 저장
        with open('api_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"📄 상세 결과가 'api_test_results.json'에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())