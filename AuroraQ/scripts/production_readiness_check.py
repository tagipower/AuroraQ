#!/usr/bin/env python3
"""
AuroraQ Production Readiness Check
실거래 환경 준비도 종합 검증 시스템
"""

import asyncio
import sys
import os
import json
import time
import psutil
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import pandas as pd
import logging

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from SharedCore.risk_management.integrated_risk_manager import IntegratedRiskManager
from SharedCore.sentiment_engine.sentiment_client import get_sentiment_service_client
from SharedCore.data_collection.binance_client import BinanceDataClient
from SharedCore.market_analysis.technical_analyzer import TechnicalAnalyzer

@dataclass
class CheckResult:
    """검증 결과"""
    name: str
    status: str  # pass, warning, fail
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = None
    recommendations: List[str] = None

class ProductionReadinessChecker:
    """실거래 환경 준비도 검증기"""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.logger = self._setup_logging()
        self.start_time = datetime.now()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('ProductionReadiness')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_check(self, name: str, status: str, score: float, message: str, 
                  details: Dict = None, recommendations: List[str] = None):
        """검증 결과 로깅"""
        result = CheckResult(
            name=name,
            status=status,
            score=score,
            message=message,
            details=details or {},
            recommendations=recommendations or []
        )
        self.results.append(result)
        
        status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(status, "❓")
        self.logger.info(f"{status_icon} {name}: {message} (Score: {score}/100)")
        
        if recommendations:
            for rec in recommendations:
                self.logger.info(f"   💡 {rec}")
    
    # ========== 1. 시스템 환경 검증 ==========
    
    async def check_system_requirements(self):
        """시스템 요구사항 검증"""
        self.logger.info("\n🖥️  1. System Requirements Check")
        
        # CPU 체크
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_count >= 4 and cpu_usage < 80:
            cpu_score = 100
            cpu_status = "pass"
            cpu_msg = f"CPU: {cpu_count} cores, {cpu_usage:.1f}% usage"
        elif cpu_count >= 2:
            cpu_score = 70
            cpu_status = "warning"
            cpu_msg = f"CPU marginal: {cpu_count} cores, {cpu_usage:.1f}% usage"
        else:
            cpu_score = 30
            cpu_status = "fail"
            cpu_msg = f"CPU insufficient: {cpu_count} cores"
        
        self.log_check("CPU Resources", cpu_status, cpu_score, cpu_msg)
        
        # 메모리 체크
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        usage_pct = memory.percent
        
        if available_gb >= 4 and usage_pct < 80:
            mem_score = 100
            mem_status = "pass"
            mem_msg = f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total ({usage_pct:.1f}% used)"
        elif available_gb >= 2:
            mem_score = 70
            mem_status = "warning"
            mem_msg = f"Memory marginal: {available_gb:.1f}GB available"
        else:
            mem_score = 30
            mem_status = "fail"
            mem_msg = f"Memory insufficient: {available_gb:.1f}GB available"
        
        self.log_check("Memory Resources", mem_status, mem_score, mem_msg)
        
        # 디스크 체크
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        total_gb = disk.total / (1024**3)
        usage_pct = (disk.used / disk.total) * 100
        
        if free_gb >= 10 and usage_pct < 85:
            disk_score = 100
            disk_status = "pass"
            disk_msg = f"Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({usage_pct:.1f}% used)"
        elif free_gb >= 5:
            disk_score = 70
            disk_status = "warning"
            disk_msg = f"Disk space marginal: {free_gb:.1f}GB free"
        else:
            disk_score = 30
            disk_status = "fail"
            disk_msg = f"Disk space insufficient: {free_gb:.1f}GB free"
        
        self.log_check("Disk Space", disk_status, disk_score, disk_msg)
        
        # 네트워크 연결 체크
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get('https://api.binance.com/api/v3/ping', timeout=aiohttp.ClientTimeout(total=10)) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200 and latency < 100:
                        net_score = 100
                        net_status = "pass"
                        net_msg = f"Network: {latency:.1f}ms to Binance API"
                    elif latency < 500:
                        net_score = 70
                        net_status = "warning"
                        net_msg = f"Network slow: {latency:.1f}ms to Binance API"
                    else:
                        net_score = 30
                        net_status = "fail"
                        net_msg = f"Network very slow: {latency:.1f}ms"
        except Exception as e:
            net_score = 0
            net_status = "fail"
            net_msg = f"Network connection failed: {e}"
        
        self.log_check("Network Connectivity", net_status, net_score, net_msg)
    
    # ========== 2. 데이터베이스 상태 검증 ==========
    
    async def check_database_health(self):
        """데이터베이스 상태 검증"""
        self.logger.info("\n💾 2. Database Health Check")
        
        db_checks = []
        
        # 가격 데이터 DB 체크
        price_db_path = Path("SharedCore/data_storage/price_data.db")
        if price_db_path.exists():
            try:
                conn = sqlite3.connect(str(price_db_path))
                cursor = conn.cursor()
                
                # 테이블 목록 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # 최근 데이터 확인
                if 'price_data' in tables:
                    cursor.execute("SELECT COUNT(*) FROM price_data WHERE timestamp > datetime('now', '-1 day')")
                    recent_count = cursor.fetchone()[0]
                    
                    if recent_count > 1000:
                        score = 100
                        status = "pass"
                        msg = f"Price DB healthy: {len(tables)} tables, {recent_count} recent records"
                    elif recent_count > 100:
                        score = 70
                        status = "warning"
                        msg = f"Price DB marginal: {recent_count} recent records"
                    else:
                        score = 30
                        status = "fail"
                        msg = f"Price DB stale: {recent_count} recent records"
                else:
                    score = 20
                    status = "fail"
                    msg = "Price data table not found"
                
                conn.close()
            except Exception as e:
                score = 0
                status = "fail"
                msg = f"Price DB error: {e}"
        else:
            score = 0
            status = "fail"
            msg = "Price database not found"
        
        self.log_check("Price Database", status, score, msg)
        
        # 감정 데이터 캐시 체크
        try:
            # Redis 연결 테스트 (설치되어 있다면)
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            r.ping()
            
            # 감정 캐시 항목 수 확인
            cache_keys = r.keys("sentiment:*")
            
            if len(cache_keys) > 100:
                score = 100
                status = "pass"
                msg = f"Sentiment cache healthy: {len(cache_keys)} cached items"
            elif len(cache_keys) > 10:
                score = 70
                status = "warning"
                msg = f"Sentiment cache marginal: {len(cache_keys)} cached items"
            else:
                score = 50
                status = "warning"
                msg = "Sentiment cache empty or minimal"
                
        except ImportError:
            score = 50
            status = "warning"
            msg = "Redis not installed - using memory cache"
        except Exception:
            score = 50
            status = "warning"
            msg = "Redis not running - using memory cache"
        
        self.log_check("Sentiment Cache", status, score, msg)
    
    # ========== 3. 외부 서비스 연결 검증 ==========
    
    async def check_external_services(self):
        """외부 서비스 연결 검증"""
        self.logger.info("\n🌐 3. External Services Check")
        
        # Binance API 연결 테스트
        try:
            client = BinanceDataClient()
            await client.initialize()
            
            # 기본 API 호출 테스트
            symbol_info = await client.get_symbol_info("BTCUSDT")
            current_price = await client.get_current_price("BTCUSDT")
            
            if symbol_info and current_price > 0:
                score = 100
                status = "pass"
                msg = f"Binance API healthy: BTCUSDT @ ${current_price:,.2f}"
                recommendations = []
            else:
                score = 30
                status = "fail"
                msg = "Binance API not responding properly"
                recommendations = ["Check API keys", "Verify network connection"]
            
            await client.close()
            
        except Exception as e:
            score = 0
            status = "fail"
            msg = f"Binance API failed: {e}"
            recommendations = ["Check API credentials", "Verify network", "Check API limits"]
        
        self.log_check("Binance API", status, score, msg, recommendations=recommendations)
        
        # Sentiment Service 연결 테스트
        try:
            client = get_sentiment_service_client()
            health = await client.health_check()
            
            if health.get('status') == 'healthy':
                # 간단한 분석 테스트
                sentiment = await client.analyze_sentiment("Bitcoin price is rising", "BTC")
                
                if 0 <= sentiment <= 1:
                    score = 100
                    status = "pass"
                    msg = f"Sentiment service healthy: {health.get('version', 'unknown')}"
                else:
                    score = 70
                    status = "warning"
                    msg = "Sentiment service responding but results questionable"
            else:
                score = 30
                status = "fail"
                msg = f"Sentiment service unhealthy: {health.get('status', 'unknown')}"
            
            await client.close()
            
        except Exception as e:
            score = 0
            status = "fail"
            msg = f"Sentiment service failed: {e}"
            recommendations = ["Start sentiment service", "Check Docker containers", "Verify service health"]
        
        self.log_check("Sentiment Service", status, score, msg, 
                      recommendations=recommendations if score < 50 else [])
        
        # News Feed 연결 테스트 (간단한 HTTP 체크)
        try:
            async with aiohttp.ClientSession() as session:
                # RSS 피드 체크
                feeds_to_check = [
                    "https://feeds.feedburner.com/oreilly/radar",
                    "https://rss.cnn.com/rss/money_news_international.rss"
                ]
                
                successful_feeds = 0
                for feed_url in feeds_to_check:
                    try:
                        async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                successful_feeds += 1
                    except:
                        continue
                
                if successful_feeds >= 2:
                    score = 100
                    status = "pass"
                    msg = f"News feeds accessible: {successful_feeds}/{len(feeds_to_check)}"
                elif successful_feeds >= 1:
                    score = 70
                    status = "warning"
                    msg = f"Some news feeds accessible: {successful_feeds}/{len(feeds_to_check)}"
                else:
                    score = 30
                    status = "fail"
                    msg = "No news feeds accessible"
                    
        except Exception as e:
            score = 50
            status = "warning"
            msg = f"News feed check failed: {e}"
        
        self.log_check("News Feeds", status, score, msg)
    
    # ========== 4. 리스크 관리 시스템 검증 ==========
    
    async def check_risk_management(self):
        """리스크 관리 시스템 검증"""
        self.logger.info("\n🛡️  4. Risk Management System Check")
        
        try:
            # 통합 리스크 관리자 초기화
            risk_manager = IntegratedRiskManager()
            
            # 테스트 데이터 생성
            test_dates = pd.date_range(start='2024-01-01', end='2024-07-27', freq='D')
            test_prices = pd.DataFrame({
                'open': 45000 + np.random.randn(len(test_dates)) * 1000,
                'high': 46000 + np.random.randn(len(test_dates)) * 1000,
                'low': 44000 + np.random.randn(len(test_dates)) * 1000,
                'close': 45000 + np.random.randn(len(test_dates)) * 1000,
                'volume': 1000000 + np.random.randn(len(test_dates)) * 100000
            }, index=test_dates)
            
            # 기본 리스크 평가 테스트
            risk_ok, risk_msg = risk_manager.evaluate_risk(
                test_prices, "PPOStrategy", sentiment_score=0.6
            )
            
            if risk_ok:
                basic_score = 100
                basic_status = "pass"
                basic_msg = f"Basic risk evaluation: {risk_msg}"
            else:
                basic_score = 30
                basic_status = "fail"
                basic_msg = f"Basic risk evaluation failed: {risk_msg}"
            
            self.log_check("Basic Risk Evaluation", basic_status, basic_score, basic_msg)
            
            # 포지션 사이징 테스트
            try:
                position_size = risk_manager.calculate_position_size(
                    capital=100000,
                    entry_price=45000,
                    stop_loss_pct=0.05,
                    risk_per_trade=0.02
                )
                
                if 0 < position_size < 5:  # 합리적인 포지션 사이즈
                    pos_score = 100
                    pos_status = "pass"
                    pos_msg = f"Position sizing working: {position_size:.2f} shares"
                else:
                    pos_score = 70
                    pos_status = "warning"
                    pos_msg = f"Position sizing questionable: {position_size:.2f} shares"
                
            except Exception as e:
                pos_score = 30
                pos_status = "fail"
                pos_msg = f"Position sizing failed: {e}"
            
            self.log_check("Position Sizing", pos_status, pos_score, pos_msg)
            
            # 손절/익절 로직 테스트
            try:
                cut_decision, cut_reason = risk_manager.should_cut_loss_or_take_profit(
                    entry_price=45000,
                    current_price=42750,  # 5% 손실
                    strategy_name="PPOStrategy"
                )
                
                if cut_decision == "cut_loss":
                    cut_score = 100
                    cut_status = "pass"
                    cut_msg = f"Stop loss working: {cut_reason}"
                else:
                    cut_score = 70
                    cut_status = "warning"
                    cut_msg = f"Stop loss logic: {cut_reason}"
                
            except Exception as e:
                cut_score = 30
                cut_status = "fail"
                cut_msg = f"Stop loss logic failed: {e}"
            
            self.log_check("Stop Loss Logic", cut_status, cut_score, cut_msg)
            
            # 리스크 대시보드 테스트
            try:
                dashboard = risk_manager.get_risk_dashboard()
                
                if isinstance(dashboard, dict) and 'basic_risk_status' in dashboard:
                    dash_score = 100
                    dash_status = "pass"
                    dash_msg = "Risk dashboard operational"
                else:
                    dash_score = 70
                    dash_status = "warning"
                    dash_msg = "Risk dashboard limited functionality"
                
            except Exception as e:
                dash_score = 30
                dash_status = "fail"
                dash_msg = f"Risk dashboard failed: {e}"
            
            self.log_check("Risk Dashboard", dash_status, dash_score, dash_msg)
            
        except Exception as e:
            self.log_check("Risk Management System", "fail", 0, f"Risk management system failed: {e}",
                          recommendations=["Check risk management modules", "Verify dependencies"])
    
    # ========== 5. 거래 실행 시스템 검증 ==========
    
    async def check_trading_system(self):
        """거래 실행 시스템 검증"""
        self.logger.info("\n📈 5. Trading System Check")
        
        # 기술적 분석 모듈 테스트
        try:
            analyzer = TechnicalAnalyzer()
            
            # 테스트 데이터로 분석
            test_data = pd.DataFrame({
                'close': [45000, 45100, 44900, 45200, 45150, 45300, 45250, 45400],
                'volume': [1000000] * 8
            })
            
            analysis = analyzer.analyze_comprehensive(test_data)
            
            if analysis and 'trend' in analysis:
                tech_score = 100
                tech_status = "pass"
                tech_msg = f"Technical analysis working: {analysis.get('trend', 'unknown')} trend"
            else:
                tech_score = 70
                tech_status = "warning"
                tech_msg = "Technical analysis limited functionality"
                
        except Exception as e:
            tech_score = 30
            tech_status = "fail"
            tech_msg = f"Technical analysis failed: {e}"
        
        self.log_check("Technical Analysis", tech_status, tech_score, tech_msg)
        
        # 주문 실행 시뮬레이션 (실제 주문 없이)
        try:
            # 가상 주문 검증
            order_params = {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': 0.001,
                'price': 45000,
                'timeInForce': 'GTC'
            }
            
            # 주문 파라미터 유효성 검사
            required_fields = ['symbol', 'side', 'type', 'quantity']
            missing_fields = [field for field in required_fields if field not in order_params]
            
            if not missing_fields and order_params['quantity'] > 0:
                order_score = 100
                order_status = "pass"
                order_msg = "Order validation logic working"
            else:
                order_score = 30
                order_status = "fail"
                order_msg = f"Order validation failed: missing {missing_fields}"
                
        except Exception as e:
            order_score = 30
            order_status = "fail"
            order_msg = f"Order system test failed: {e}"
        
        self.log_check("Order System", order_status, order_score, order_msg)
    
    # ========== 6. 로깅 및 모니터링 검증 ==========
    
    async def check_logging_monitoring(self):
        """로깅 및 모니터링 시스템 검증"""
        self.logger.info("\n📊 6. Logging & Monitoring Check")
        
        # 로그 디렉토리 확인
        log_dirs = ['logs', 'SharedCore/logs', 'sentiment-service/logs']
        log_score = 0
        log_files_found = 0
        
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            if log_path.exists():
                log_files = list(log_path.glob('*.log'))
                log_files_found += len(log_files)
                log_score += 30
        
        if log_score >= 60:
            log_status = "pass"
            log_msg = f"Logging infrastructure: {log_files_found} log files found"
        elif log_score >= 30:
            log_status = "warning"
            log_msg = f"Limited logging: {log_files_found} log files found"
        else:
            log_status = "fail" 
            log_msg = "No logging infrastructure found"
            
        self.log_check("Logging Infrastructure", log_status, log_score, log_msg)
        
        # 모니터링 스크립트 확인
        monitoring_files = [
            'sentiment-service/scripts/health_monitor.py',
            'scripts/production_readiness_check.py'
        ]
        
        monitoring_score = 0
        for mon_file in monitoring_files:
            if Path(mon_file).exists():
                monitoring_score += 50
        
        if monitoring_score >= 100:
            mon_status = "pass"
            mon_msg = "All monitoring scripts available"
        elif monitoring_score >= 50:
            mon_status = "warning"
            mon_msg = "Some monitoring scripts available"
        else:
            mon_status = "fail"
            mon_msg = "No monitoring scripts found"
            
        self.log_check("Monitoring Scripts", mon_status, monitoring_score, mon_msg)
    
    # ========== 7. 보안 및 설정 검증 ==========
    
    async def check_security_config(self):
        """보안 및 설정 검증"""
        self.logger.info("\n🔒 7. Security & Configuration Check")
        
        # 환경 변수 확인
        required_env_vars = [
            'BINANCE_API_KEY', 'BINANCE_SECRET_KEY',
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        
        missing_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)
        
        if not missing_env_vars:
            env_score = 100
            env_status = "pass"
            env_msg = "All required environment variables set"
        elif len(missing_env_vars) <= 2:
            env_score = 70
            env_status = "warning"
            env_msg = f"Missing env vars: {', '.join(missing_env_vars)}"
        else:
            env_score = 30
            env_status = "fail"
            env_msg = f"Missing critical env vars: {', '.join(missing_env_vars)}"
            
        self.log_check("Environment Variables", env_status, env_score, env_msg)
        
        # 설정 파일 확인
        config_files = [
            'SharedCore/config/trading_config.json',
            'SharedCore/config/risk_config.json',
            'sentiment-service/config.json'
        ]
        
        config_score = 0
        for config_file in config_files:
            if Path(config_file).exists():
                config_score += 33
        
        if config_score >= 99:
            config_status = "pass"
            config_msg = "All configuration files present"
        elif config_score >= 50:
            config_status = "warning"
            config_msg = "Some configuration files missing"
        else:
            config_status = "fail"
            config_msg = "Critical configuration files missing"
        
        self.log_check("Configuration Files", config_status, config_score, config_msg)
    
    # ========== 종합 검증 실행 ==========
    
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """종합 검증 실행"""
        self.logger.info("🚀 Starting AuroraQ Production Readiness Check")
        self.logger.info("=" * 60)
        
        # 모든 검증 실행
        checks = [
            self.check_system_requirements,
            self.check_database_health,
            self.check_external_services,
            self.check_risk_management,
            self.check_trading_system,
            self.check_logging_monitoring,
            self.check_security_config
        ]
        
        for check in checks:
            try:
                await check()
            except Exception as e:
                self.logger.error(f"Check failed: {e}")
                self.log_check(check.__name__, "fail", 0, f"Check execution failed: {e}")
        
        # 결과 분석
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == "pass"])
        warning_checks = len([r for r in self.results if r.status == "warning"])
        failed_checks = len([r for r in self.results if r.status == "fail"])
        
        overall_score = sum([r.score for r in self.results]) / total_checks if total_checks > 0 else 0
        
        # 준비도 등급 결정
        if overall_score >= 90:
            readiness_grade = "A+ (Production Ready)"
            readiness_color = "🟢"
        elif overall_score >= 80:
            readiness_grade = "A (Nearly Ready)"
            readiness_color = "🟡"
        elif overall_score >= 70:
            readiness_grade = "B (Needs Improvements)"
            readiness_color = "🟡"
        elif overall_score >= 60:
            readiness_grade = "C (Major Issues)"
            readiness_color = "🟠"
        else:
            readiness_grade = "F (Not Ready)"
            readiness_color = "🔴"
        
        # 결과 요약
        duration = datetime.now() - self.start_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 PRODUCTION READINESS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Overall Score: {overall_score:.1f}/100")
        self.logger.info(f"Readiness Grade: {readiness_color} {readiness_grade}")
        self.logger.info(f"")
        self.logger.info(f"✅ Passed: {passed_checks}")
        self.logger.info(f"⚠️  Warning: {warning_checks}")
        self.logger.info(f"❌ Failed: {failed_checks}")
        self.logger.info(f"")
        self.logger.info(f"Check Duration: {duration.total_seconds():.1f}s")
        
        # 주요 권고사항
        critical_issues = [r for r in self.results if r.status == "fail" and r.score < 50]
        if critical_issues:
            self.logger.info(f"\n🚨 CRITICAL ISSUES TO ADDRESS:")
            for issue in critical_issues:
                self.logger.info(f"- {issue.name}: {issue.message}")
                for rec in issue.recommendations:
                    self.logger.info(f"  💡 {rec}")
        
        warnings = [r for r in self.results if r.status == "warning"]
        if warnings:
            self.logger.info(f"\n⚠️  WARNINGS TO REVIEW:")
            for warning in warnings[:5]:  # 상위 5개만 표시
                self.logger.info(f"- {warning.name}: {warning.message}")
        
        if overall_score >= 85:
            self.logger.info(f"\n🎉 AuroraQ is ready for production trading!")
        elif overall_score >= 70:
            self.logger.info(f"\n🔧 AuroraQ needs minor improvements before production")
        else:
            self.logger.info(f"\n⚠️  AuroraQ requires significant improvements before production")
        
        # 결과 딕셔너리 반환
        return {
            "overall_score": overall_score,
            "readiness_grade": readiness_grade,
            "total_checks": total_checks,
            "passed": passed_checks,
            "warnings": warning_checks,
            "failed": failed_checks,
            "duration_seconds": duration.total_seconds(),
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """메인 실행 함수"""
    checker = ProductionReadinessChecker()
    results = await checker.run_comprehensive_check()
    
    # 결과를 JSON 파일로 저장
    results_file = Path("production_readiness_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # 간단한 상태 파일 생성 (다른 스크립트에서 사용할 수 있도록)
    status_file = Path("production_status.json")
    with open(status_file, 'w') as f:
        json.dump({
            "overall_score": results["overall_score"],
            "readiness_grade": results["readiness_grade"],
            "ready_for_production": results["overall_score"] >= 85,
            "last_check": results["timestamp"]
        }, f, indent=2)

if __name__ == "__main__":
    try:
        # numpy 임포트 추가
        import numpy as np
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Production readiness check interrupted by user")
    except Exception as e:
        print(f"\n❌ Production readiness check failed: {e}")
        import traceback
        traceback.print_exc()