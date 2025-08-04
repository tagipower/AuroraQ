#!/usr/bin/env python3
"""
VPS AuroraQ 트레이딩 시스템 모니터링
실시간 상태 모니터링 및 알림
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path
import psutil
import aiohttp
import logging

# Prometheus 메트릭 수집
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
    from prometheus_client import CollectorRegistry, multiprocess, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Install with: pip install prometheus-client")

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 색상 코드
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class VPSMonitor:
    """VPS 트레이딩 시스템 모니터 (Prometheus 통합)"""
    
    def __init__(self):
        self.api_url = "http://localhost:8004"  # VPS API 엔드포인트
        self.refresh_interval = 5  # 5초마다 갱신
        self.start_time = time.time()
        
        # Prometheus 메트릭 초기화
        if PROMETHEUS_AVAILABLE:
            self.setup_prometheus_metrics()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def setup_prometheus_metrics(self):
        """Prometheus 메트릭 설정"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # 시스템 상태 메트릭
        self.system_cpu_usage = Gauge(
            'vps_system_cpu_usage_percent',
            'VPS system CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'vps_system_memory_usage_bytes',
            'VPS system memory usage in bytes'
        )
        
        self.system_disk_usage = Gauge(
            'vps_system_disk_usage_percent',
            'VPS system disk usage percentage'
        )
        
        # 트레이딩 메트릭
        self.trading_positions = Gauge(
            'vps_trading_positions_count',
            'Number of active trading positions'
        )
        
        self.trading_pnl = Gauge(
            'vps_trading_daily_pnl',
            'Daily profit and loss'
        )
        
        self.trading_win_rate = Gauge(
            'vps_trading_win_rate_percent',
            'Trading win rate percentage'
        )
        
        # API 상태 메트릭
        self.api_response_time = Histogram(
            'vps_api_response_time_seconds',
            'API response time in seconds',
            ['endpoint']
        )
        
        self.api_status = Gauge(
            'vps_api_status',
            'API status (1=up, 0=down)',
            ['endpoint']
        )
        
        # 모니터링 메트릭
        self.monitor_uptime = Gauge(
            'vps_monitor_uptime_seconds',
            'Monitor uptime in seconds'
        )
        
        self.monitor_checks_total = Counter(
            'vps_monitor_checks_total',
            'Total number of monitoring checks performed'
        )
        
        self.monitor_errors_total = Counter(
            'vps_monitor_errors_total',
            'Total number of monitoring errors',
            ['error_type']
        )
        
    async def get_system_status(self):
        """시스템 상태 조회 (Prometheus 메트릭 업데이트 포함)"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=10) as resp:
                    response_time = time.time() - start_time
                    
                    if PROMETHEUS_AVAILABLE:
                        self.api_response_time.labels(endpoint='health').observe(response_time)
                        self.api_status.labels(endpoint='health').set(1 if resp.status == 200 else 0)
                        self.monitor_checks_total.inc()
                    
                    if resp.status == 200:
                        data = await resp.json()
                        self.update_system_metrics()
                        return data
                    else:
                        if PROMETHEUS_AVAILABLE:
                            self.monitor_errors_total.labels(error_type='api_error').inc()
                        return None
                        
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                self.api_status.labels(endpoint='health').set(0)
                self.monitor_errors_total.labels(error_type='connection_error').inc()
            self.logger.error(f"Failed to get system status: {e}")
            return None
    
    def update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_disk_usage.set(disk_percent)
            
            # 모니터 가동시간
            uptime = time.time() - self.start_time
            self.monitor_uptime.set(uptime)
            
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
            if PROMETHEUS_AVAILABLE:
                self.monitor_errors_total.labels(error_type='metrics_error').inc()
    
    async def get_trading_status(self):
        """트레이딩 상태 조회 (Prometheus 메트릭 업데이트 포함)"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/trading/status", timeout=10) as resp:
                    response_time = time.time() - start_time
                    
                    if PROMETHEUS_AVAILABLE:
                        self.api_response_time.labels(endpoint='trading_status').observe(response_time)
                        self.api_status.labels(endpoint='trading_status').set(1 if resp.status == 200 else 0)
                    
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # 트레이딩 메트릭 업데이트
                        if PROMETHEUS_AVAILABLE:
                            self.trading_positions.set(data.get('positions_count', 0))
                            self.trading_pnl.set(data.get('daily_pnl', 0.0))
                        
                        return data
                    else:
                        if PROMETHEUS_AVAILABLE:
                            self.monitor_errors_total.labels(error_type='trading_api_error').inc()
                        return None
                        
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                self.api_status.labels(endpoint='trading_status').set(0)
                self.monitor_errors_total.labels(error_type='trading_connection_error').inc()
            self.logger.error(f"Failed to get trading status: {e}")
            return None
    
    async def get_positions(self):
        """현재 포지션 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/positions") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except:
            return []
    
    async def get_performance(self):
        """성과 지표 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/performance") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except:
            return {}
    
    def clear_screen(self):
        """화면 지우기"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_system_resources(self):
        """시스템 리소스 정보"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 프로세스 정보
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'process_memory_mb': process_memory
        }
    
    def format_price(self, price):
        """가격 포맷팅"""
        return f"${price:,.2f}"
    
    def format_percent(self, value):
        """퍼센트 포맷팅"""
        color = Colors.GREEN if value >= 0 else Colors.RED
        return f"{color}{value:+.2f}%{Colors.RESET}"
    
    async def display_dashboard(self):
        """대시보드 표시"""
        while True:
            try:
                # 데이터 수집
                status = await self.get_system_status()
                positions = await self.get_positions()
                performance = await self.get_performance()
                resources = self.get_system_resources()
                
                # 화면 지우고 헤더 표시
                self.clear_screen()
                print(f"{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.CYAN}║             VPS AuroraQ Trading System Monitor                ║{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.CYAN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
                print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # 시스템 상태
                if status:
                    trading_status = status.get('trading_status', 'unknown')
                    status_color = Colors.GREEN if trading_status == 'running' else Colors.YELLOW
                    print(f"{Colors.BOLD}🖥️  시스템 상태{Colors.RESET}")
                    print(f"├─ 상태: {status_color}{trading_status.upper()}{Colors.RESET}")
                    print(f"├─ 모드: {status.get('mode', 'N/A')}")
                    print(f"└─ 업타임: {status.get('uptime', 'N/A')}")
                else:
                    print(f"{Colors.RED}⚠️  시스템에 연결할 수 없습니다{Colors.RESET}")
                print()
                
                # 리소스 사용량
                print(f"{Colors.BOLD}💾 시스템 리소스{Colors.RESET}")
                print(f"├─ CPU: {resources['cpu_percent']:.1f}%")
                print(f"├─ 메모리: {resources['memory_percent']:.1f}% ({resources['memory_used_gb']:.1f}/{resources['memory_total_gb']:.1f} GB)")
                print(f"├─ 디스크: {resources['disk_percent']:.1f}%")
                print(f"└─ 프로세스 메모리: {resources['process_memory_mb']:.1f} MB")
                print()
                
                # 성과 지표
                if performance:
                    print(f"{Colors.BOLD}📊 성과 지표{Colors.RESET}")
                    print(f"├─ 일일 수익률: {self.format_percent(performance.get('daily_return', 0) * 100)}")
                    print(f"├─ 전체 수익률: {self.format_percent(performance.get('total_return', 0) * 100)}")
                    print(f"├─ 승률: {performance.get('win_rate', 0):.1f}%")
                    print(f"├─ 샤프 비율: {performance.get('sharpe_ratio', 0):.2f}")
                    print(f"└─ 최대 낙폭: {self.format_percent(performance.get('max_drawdown', 0) * 100)}")
                print()
                
                # 현재 포지션
                print(f"{Colors.BOLD}📈 현재 포지션{Colors.RESET}")
                if positions:
                    for i, pos in enumerate(positions):
                        symbol = pos.get('symbol', 'N/A')
                        side = pos.get('side', 'N/A')
                        size = pos.get('size', 0)
                        entry_price = pos.get('entry_price', 0)
                        current_price = pos.get('current_price', 0)
                        pnl = pos.get('unrealized_pnl', 0)
                        pnl_percent = pos.get('pnl_percent', 0)
                        
                        side_color = Colors.GREEN if side == 'long' else Colors.RED
                        prefix = "├─" if i < len(positions) - 1 else "└─"
                        
                        print(f"{prefix} {symbol}: {side_color}{side.upper()}{Colors.RESET} "
                              f"{size:.4f} @ {self.format_price(entry_price)} "
                              f"→ {self.format_price(current_price)} "
                              f"({self.format_percent(pnl_percent)})")
                else:
                    print("└─ 열린 포지션 없음")
                print()
                
                # 최근 신호
                if status and 'recent_signals' in status:
                    print(f"{Colors.BOLD}🎯 최근 전략 신호{Colors.RESET}")
                    signals = status['recent_signals'][:5]  # 최근 5개
                    for i, signal in enumerate(signals):
                        strategy = signal.get('strategy', 'N/A')
                        action = signal.get('action', 'N/A')
                        score = signal.get('score', 0)
                        
                        action_color = Colors.GREEN if action == 'BUY' else Colors.RED if action == 'SELL' else Colors.YELLOW
                        prefix = "├─" if i < len(signals) - 1 else "└─"
                        
                        print(f"{prefix} {strategy}: {action_color}{action}{Colors.RESET} (점수: {score:.2f})")
                print()
                
                # 하단 정보
                print(f"{Colors.BOLD}ℹ️  정보{Colors.RESET}")
                print(f"├─ API: {self.api_url}")
                print(f"├─ 갱신 주기: {self.refresh_interval}초")
                print(f"└─ 종료: Ctrl+C")
                
                # 대기
                await asyncio.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Colors.RED}오류: {e}{Colors.RESET}")
                await asyncio.sleep(self.refresh_interval)
        
        print(f"\n{Colors.YELLOW}모니터링 종료{Colors.RESET}")


def start_prometheus_server(port=9090):
    """프로메테우스 메트릭 서버 시작"""
    if not PROMETHEUS_AVAILABLE:
        print(f"{Colors.YELLOW}Warning: Prometheus client not available{Colors.RESET}")
        return False
        
    try:
        start_http_server(port)
        print(f"{Colors.GREEN}✓ Prometheus 메트릭 서버 시작: http://localhost:{port}{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}Prometheus 서버 시작 실패: {e}{Colors.RESET}")
        return False

async def main():
    """메인 함수"""
    monitor = VPSMonitor()
    
    print(f"{Colors.BOLD}{Colors.GREEN}VPS AuroraQ 모니터링 시작...{Colors.RESET}")
    print(f"API 서버 연결 중: {monitor.api_url}")
    
    # Prometheus 메트릭 서버 시작
    prometheus_started = start_prometheus_server(9090)
    if prometheus_started:
        print(f"Prometheus 스크레이핑 엔드포인트: http://localhost:9090")
    print()
    
    await monitor.display_dashboard()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}사용자에 의해 종료됨{Colors.RESET}")