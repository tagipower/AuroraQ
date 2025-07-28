#!/usr/bin/env python3
"""
Sentiment Service Health Monitor
서비스 상태 모니터링 및 자동 복구 스크립트
"""

import asyncio
import aiohttp
import time
import json
import logging
import subprocess
import psutil
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from email.mime.text import MimeText
from pathlib import Path
import argparse


class HealthMonitor:
    """센티멘트 서비스 헬스 모니터"""
    
    def __init__(self, config_file: str = "monitor_config.json"):
        self.config = self.load_config(config_file)
        self.service_url = self.config.get("service_url", "http://localhost:8000")
        self.check_interval = self.config.get("check_interval", 30)  # 30초
        self.max_failures = self.config.get("max_failures", 3)
        self.auto_restart = self.config.get("auto_restart", True)
        
        # 상태 추적
        self.failure_count = 0
        self.last_healthy_time = datetime.now()
        self.alerts_sent = []
        
        # 로깅 설정
        self.setup_logging()
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        default_config = {
            "service_url": "http://localhost:8000",
            "check_interval": 30,
            "max_failures": 3,
            "auto_restart": True,
            "restart_command": "docker-compose restart sentiment-service",
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "to_address": ""
            },
            "thresholds": {
                "response_time_ms": 5000,
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 80
            }
        }
        
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        else:
            # 기본 설정 파일 생성
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default config file: {config_file}")
            
        return default_config
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sentiment_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SentimentMonitor')
    
    async def check_health(self) -> Dict[str, Any]:
        """서비스 헬스체크"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.service_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    if response.status == 200:
                        health_data = await response.json()
                        health_data['response_time_ms'] = response_time
                        return health_data
                    else:
                        return {
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}',
                            'response_time_ms': response_time
                        }
                        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': None
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """시스템 리소스 체크"""
        try:
            # Docker 컨테이너 통계 가져오기
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', 
                 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
                for line in lines:
                    if 'sentiment' in line.lower():
                        parts = line.split()
                        if len(parts) >= 4:
                            container = parts[0]
                            cpu_percent = float(parts[1].rstrip('%'))
                            memory_usage = parts[2]  # "1.2GiB / 2.0GiB" 형식
                            memory_percent = float(parts[3].rstrip('%'))
                            
                            # 메모리 사용량 파싱 (MB 단위)
                            memory_parts = memory_usage.split(' / ')
                            if len(memory_parts) == 2:
                                used_str = memory_parts[0].replace('GiB', '').replace('MiB', '')
                                used_gb = float(used_str)
                                if 'GiB' in memory_parts[0]:
                                    memory_mb = used_gb * 1024
                                else:
                                    memory_mb = used_gb
                            else:
                                memory_mb = 0
                            
                            return {
                                'container': container,
                                'cpu_percent': cpu_percent,
                                'memory_mb': memory_mb,
                                'memory_percent': memory_percent
                            }
            
            # 전체 시스템 리소스 폴백
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_mb': psutil.virtual_memory().used / (1024**2),
                'memory_percent': psutil.virtual_memory().percent
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return {}
    
    def evaluate_health(self, health_data: Dict[str, Any], resource_data: Dict[str, Any]) -> tuple[bool, list]:
        """헬스 상태 평가"""
        issues = []
        is_healthy = True
        
        # 기본 상태 체크
        if health_data.get('status') != 'healthy':
            is_healthy = False
            issues.append(f"Service status: {health_data.get('status', 'unknown')}")
            if 'error' in health_data:
                issues.append(f"Error: {health_data['error']}")
        
        # 응답 시간 체크
        response_time = health_data.get('response_time_ms')
        if response_time and response_time > self.config['thresholds']['response_time_ms']:
            issues.append(f"Slow response time: {response_time:.1f}ms")
        
        # 리소스 사용량 체크
        if resource_data:
            memory_mb = resource_data.get('memory_mb', 0)
            cpu_percent = resource_data.get('cpu_percent', 0)
            
            if memory_mb > self.config['thresholds']['memory_usage_mb']:
                is_healthy = False
                issues.append(f"High memory usage: {memory_mb:.1f}MB")
            
            if cpu_percent > self.config['thresholds']['cpu_usage_percent']:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        return is_healthy, issues
    
    async def send_alert(self, message: str, is_critical: bool = False):
        """알림 전송"""
        alert_key = f"{message[:50]}_{datetime.now().strftime('%Y%m%d%H')}"
        
        # 중복 알림 방지 (같은 시간대에 같은 메시지)
        if alert_key in self.alerts_sent:
            return
        
        self.alerts_sent.append(alert_key)
        
        # 최근 10개 알림만 유지
        if len(self.alerts_sent) > 10:
            self.alerts_sent = self.alerts_sent[-10:]
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[AuroraQ Sentiment Service Alert]\n\nTime: {timestamp}\nMessage: {message}"
        
        # Telegram 알림
        if self.config['telegram']['enabled']:
            await self.send_telegram_alert(full_message)
        
        # 이메일 알림
        if self.config['email']['enabled']:
            await self.send_email_alert(full_message, is_critical)
    
    async def send_telegram_alert(self, message: str):
        """Telegram 알림 전송"""
        try:
            bot_token = self.config['telegram']['bot_token']
            chat_id = self.config['telegram']['chat_id']
            
            if not bot_token or not chat_id:
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }) as response:
                    if response.status == 200:
                        self.logger.info("Telegram alert sent successfully")
                    else:
                        self.logger.error(f"Failed to send Telegram alert: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Telegram alert failed: {e}")
    
    async def send_email_alert(self, message: str, is_critical: bool = False):
        """이메일 알림 전송"""
        try:
            smtp_config = self.config['email']
            
            if not all([smtp_config['username'], smtp_config['password'], smtp_config['to_address']]):
                return
            
            subject = f"{'🚨 CRITICAL' if is_critical else '⚠️  WARNING'} - AuroraQ Sentiment Service"
            
            msg = MimeText(message)
            msg['Subject'] = subject
            msg['From'] = smtp_config['username']
            msg['To'] = smtp_config['to_address']
            
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    async def restart_service(self):
        """서비스 재시작"""
        try:
            self.logger.warning("Attempting to restart sentiment service...")
            
            restart_cmd = self.config.get("restart_command", "docker-compose restart sentiment-service")
            result = subprocess.run(
                restart_cmd.split(),
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                self.logger.info("Service restart command executed successfully")
                await self.send_alert("🔄 Sentiment service has been restarted", is_critical=True)
                
                # 재시작 후 대기
                await asyncio.sleep(30)
                
                # 재시작 후 헬스체크  
                health_data = await self.check_health()
                if health_data.get('status') == 'healthy':
                    self.logger.info("Service is healthy after restart")
                    self.failure_count = 0
                    self.last_healthy_time = datetime.now()
                else:
                    self.logger.error("Service is still unhealthy after restart")
                    
            else:
                self.logger.error(f"Service restart failed: {result.stderr}")
                await self.send_alert(f"❌ Failed to restart service: {result.stderr}", is_critical=True)
                
        except Exception as e:
            self.logger.error(f"Service restart error: {e}")
            await self.send_alert(f"❌ Service restart error: {e}", is_critical=True)
    
    async def monitor_loop(self):
        """모니터링 메인 루프"""
        self.logger.info(f"Starting health monitor for {self.service_url}")
        self.logger.info(f"Check interval: {self.check_interval}s, Max failures: {self.max_failures}")
        
        while True:
            try:
                # 헬스체크 실행
                health_data = await self.check_health()
                resource_data = self.check_system_resources()
                
                # 상태 평가
                is_healthy, issues = self.evaluate_health(health_data, resource_data)
                
                if is_healthy:
                    if self.failure_count > 0:
                        self.logger.info("Service recovered - back to healthy state")
                        await self.send_alert("✅ Sentiment service has recovered")
                    
                    self.failure_count = 0
                    self.last_healthy_time = datetime.now()
                    
                    # 상태 로깅 (INFO 레벨)
                    response_time = health_data.get('response_time_ms', 0)
                    self.logger.info(f"Service healthy - Response: {response_time:.1f}ms")
                    
                else:
                    self.failure_count += 1
                    self.logger.warning(f"Service unhealthy (failure {self.failure_count}/{self.max_failures})")
                    
                    for issue in issues:
                        self.logger.warning(f"  Issue: {issue}")
                    
                    # 첫 번째 실패 시 경고 알림
                    if self.failure_count == 1:
                        await self.send_alert(f"⚠️ Service issues detected:\n" + "\n".join(issues))
                    
                    # 최대 실패 횟수 도달 시 자동 재시작
                    if self.failure_count >= self.max_failures and self.auto_restart:
                        await self.send_alert(
                            f"🚨 Service failed {self.max_failures} times - initiating restart",
                            is_critical=True
                        )
                        await self.restart_service()
                
                # 다음 체크까지 대기
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Monitor stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def run_single_check(self):
        """단일 헬스체크 실행"""
        print("🔍 Running single health check...")
        
        health_data = await self.check_health()
        resource_data = self.check_system_resources()
        
        print(f"\n📊 Health Status:")
        print(f"   Status: {health_data.get('status', 'unknown')}")
        print(f"   Response Time: {health_data.get('response_time_ms', 0):.1f}ms")
        
        if 'error' in health_data:
            print(f"   Error: {health_data['error']}")
        
        if resource_data:
            print(f"\n💻 Resource Usage:")
            print(f"   CPU: {resource_data.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {resource_data.get('memory_mb', 0):.1f}MB")
            
        is_healthy, issues = self.evaluate_health(health_data, resource_data)
        
        if is_healthy:
            print("\n✅ Service is healthy!")
        else:
            print("\n❌ Service has issues:")
            for issue in issues:
                print(f"   - {issue}")


async def main():
    parser = argparse.ArgumentParser(description='Sentiment Service Health Monitor')
    parser.add_argument('--config', default='monitor_config.json', help='Config file path')
    parser.add_argument('--check', action='store_true', help='Run single health check')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    
    args = parser.parse_args()
    
    monitor = HealthMonitor(args.config)
    
    if args.check:
        await monitor.run_single_check()
    elif args.monitor:
        await monitor.monitor_loop()
    else:
        print("Please specify --check for single check or --monitor for continuous monitoring")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor failed: {e}")
        import traceback
        traceback.print_exc()