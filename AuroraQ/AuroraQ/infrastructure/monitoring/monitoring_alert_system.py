#!/usr/bin/env python3
"""
VPS Deployment 모니터링 및 알림 시스템
실시간 모니터링, 메트릭 수집, 알람, 알림 전송
"""

import asyncio
import json
import logging
import smtplib
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import psutil
import requests
import websockets
from dataclasses import dataclass, asdict


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """메트릭 데이터"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """알림 데이터"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: float
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricCollector:
    """메트릭 수집기"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics = deque(maxlen=10000)  # 최근 10000개 메트릭
        self.metric_aggregates = defaultdict(list)
        self.collecting = False
        self.collection_thread = None
        
        self.logger = logging.getLogger(f"{__name__}.MetricCollector")
    
    def start_collection(self):
        """메트릭 수집 시작"""
        if not self.collecting:
            self.collecting = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            self.logger.info("메트릭 수집 시작")
    
    def stop_collection(self):
        """메트릭 수집 중지"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("메트릭 수집 중지")
    
    def _collection_loop(self):
        """메트릭 수집 루프"""
        while self.collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"메트릭 수집 오류: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        current_time = time.time()
        
        # CPU 메트릭
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric(
            name="system_cpu_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        # 메모리 메트릭
        memory = psutil.virtual_memory()
        self.add_metric(
            name="system_memory_percent",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        self.add_metric(
            name="system_memory_used_mb",
            value=memory.used / 1024 / 1024,
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        # 디스크 메트릭
        disk = psutil.disk_usage('/')
        self.add_metric(
            name="system_disk_percent",
            value=disk.percent,
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        # 프로세스별 메트릭
        process = psutil.Process()
        self.add_metric(
            name="process_memory_mb",
            value=process.memory_info().rss / 1024 / 1024,
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        self.add_metric(
            name="process_cpu_percent",
            value=process.cpu_percent(),
            metric_type=MetricType.GAUGE,
            timestamp=current_time
        )
        
        # 네트워크 메트릭
        try:
            network = psutil.net_io_counters()
            self.add_metric(
                name="network_bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            )
            
            self.add_metric(
                name="network_bytes_recv",
                value=network.bytes_recv,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            )
        except:
            pass  # 네트워크 정보를 가져올 수 없는 경우
    
    def add_metric(self, name: str, value: float, metric_type: MetricType, 
                   timestamp: float = None, labels: Dict[str, str] = None):
        """메트릭 추가"""
        if timestamp is None:
            timestamp = time.time()
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp,
            labels=labels or {}
        )
        
        self.metrics.append(metric)
        self.metric_aggregates[name].append(value)
        
        # 집계 데이터 크기 제한
        if len(self.metric_aggregates[name]) > 1000:
            self.metric_aggregates[name] = self.metric_aggregates[name][-1000:]
    
    def get_metrics(self, name: str = None, since: float = None) -> List[Metric]:
        """메트릭 조회"""
        metrics = list(self.metrics)
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, float]:
        """메트릭 요약 통계"""
        since = time.time() - (duration_minutes * 60)
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0,
            'duration_minutes': duration_minutes
        }


class AlertManager:
    """알림 관리자"""
    
    def __init__(self):
        self.alerts = deque(maxlen=1000)  # 최근 1000개 알림
        self.active_alerts = {}  # 활성 알림
        self.alert_rules = []
        self.notification_channels = {}
        
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                      level: AlertLevel, message_template: str, cooldown_minutes: int = 5):
        """알림 규칙 추가"""
        rule = {
            'name': name,
            'condition': condition,
            'level': level,
            'message_template': message_template,
            'cooldown_minutes': cooldown_minutes,
            'last_triggered': 0
        }
        self.alert_rules.append(rule)
        self.logger.info(f"알림 규칙 추가: {name}")
    
    def check_alert_rules(self, context: Dict[str, Any]):
        """알림 규칙 확인"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                # 쿨다운 체크
                if current_time - rule['last_triggered'] < rule['cooldown_minutes'] * 60:
                    continue
                
                # 조건 확인
                if rule['condition'](context):
                    alert_id = f"{rule['name']}_{int(current_time)}"
                    message = rule['message_template'].format(**context)
                    
                    alert = Alert(
                        id=alert_id,
                        level=rule['level'],
                        title=rule['name'],
                        message=message,
                        timestamp=current_time,
                        source="AlertManager",
                        metadata={'rule': rule['name']}
                    )
                    
                    self.trigger_alert(alert)
                    rule['last_triggered'] = current_time
                    
            except Exception as e:
                self.logger.error(f"알림 규칙 '{rule['name']}' 확인 중 오류: {e}")
    
    def trigger_alert(self, alert: Alert):
        """알림 발생"""
        self.alerts.append(alert)
        self.active_alerts[alert.id] = alert
        
        self.logger.warning(f"알림 발생 [{alert.level.value.upper()}]: {alert.title} - {alert.message}")
        
        # 알림 전송
        self._send_notifications(alert)
    
    def resolve_alert(self, alert_id: str):
        """알림 해결"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            
            self.logger.info(f"알림 해결: {alert.title}")
            return True
        return False
    
    def _send_notifications(self, alert: Alert):
        """알림 전송"""
        for channel_name, channel in self.notification_channels.items():
            try:
                channel.send_notification(alert)
            except Exception as e:
                self.logger.error(f"알림 전송 실패 ({channel_name}): {e}")
    
    def register_notification_channel(self, name: str, channel):
        """알림 채널 등록"""
        self.notification_channels[name] = channel
        self.logger.info(f"알림 채널 등록: {name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, level: AlertLevel = None, limit: int = 100) -> List[Alert]:
        """알림 히스토리 조회"""
        alerts = list(self.alerts)[-limit:]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts


class NotificationChannel:
    """알림 채널 기본 클래스"""
    
    def send_notification(self, alert: Alert):
        """알림 전송 (하위 클래스에서 구현)"""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """이메일 알림 채널"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        
        self.logger = logging.getLogger(f"{__name__}.EmailChannel")
    
    def send_notification(self, alert: Alert):
        """이메일 알림 전송"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
알림 레벨: {alert.level.value.upper()}
제목: {alert.title}
메시지: {alert.message}
발생 시간: {datetime.fromtimestamp(alert.timestamp).isoformat()}
소스: {alert.source}
알림 ID: {alert.id}

메타데이터: {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"이메일 알림 전송 완료: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"이메일 알림 전송 실패: {e}")


class WebhookNotificationChannel(NotificationChannel):
    """웹훅 알림 채널 (Slack, Discord 등)"""
    
    def __init__(self, webhook_url: str, channel_type: str = "webhook"):
        self.webhook_url = webhook_url
        self.channel_type = channel_type
        
        self.logger = logging.getLogger(f"{__name__}.WebhookChannel")
    
    def send_notification(self, alert: Alert):
        """웹훅 알림 전송"""
        try:
            # 알림 레벨에 따른 색상
            color_map = {
                AlertLevel.INFO: "#36a64f",      # 녹색
                AlertLevel.WARNING: "#ff9900",   # 주황색
                AlertLevel.ERROR: "#ff0000",     # 빨간색
                AlertLevel.CRITICAL: "#8b0000"   # 진한 빨간색
            }
            
            payload = {
                "text": f"[{alert.level.value.upper()}] {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "#cccccc"),
                        "fields": [
                            {
                                "title": "메시지",
                                "value": alert.message,
                                "short": False
                            },
                            {
                                "title": "발생 시간",
                                "value": datetime.fromtimestamp(alert.timestamp).isoformat(),
                                "short": True
                            },
                            {
                                "title": "소스",
                                "value": alert.source,
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"웹훅 알림 전송 완료: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"웹훅 알림 전송 실패: {e}")


class ConsoleNotificationChannel(NotificationChannel):
    """콘솔 알림 채널"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConsoleChannel")
    
    def send_notification(self, alert: Alert):
        """콘솔 알림 출력"""
        level_colors = {
            AlertLevel.INFO: "\033[92m",      # 녹색
            AlertLevel.WARNING: "\033[93m",   # 노란색
            AlertLevel.ERROR: "\033[91m",     # 빨간색
            AlertLevel.CRITICAL: "\033[95m"   # 자주색
        }
        
        color = level_colors.get(alert.level, "\033[0m")
        reset_color = "\033[0m"
        
        print(f"\n{color}🚨 [{alert.level.value.upper()}] {alert.title}{reset_color}")
        print(f"📧 {alert.message}")
        print(f"⏰ {datetime.fromtimestamp(alert.timestamp).isoformat()}")
        print(f"🔗 소스: {alert.source}")
        print("-" * 50)


class MonitoringSystem:
    """통합 모니터링 시스템"""
    
    def __init__(self, collection_interval: int = 30):
        self.metric_collector = MetricCollector(collection_interval)
        self.alert_manager = AlertManager()
        self.monitoring = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        # 기본 알림 규칙 등록
        self._register_default_alert_rules()
        
        # 기본 알림 채널 등록 (콘솔)
        self.alert_manager.register_notification_channel(
            "console", 
            ConsoleNotificationChannel()
        )
    
    def _register_default_alert_rules(self):
        """기본 알림 규칙 등록"""
        
        # CPU 사용량 높음
        def high_cpu_condition(context):
            return context.get('system_cpu_percent', 0) > 80
        
        self.alert_manager.add_alert_rule(
            name="High CPU Usage",
            condition=high_cpu_condition,
            level=AlertLevel.WARNING,
            message_template="CPU 사용량이 {system_cpu_percent:.1f}%로 높습니다.",
            cooldown_minutes=5
        )
        
        # 메모리 사용량 높음
        def high_memory_condition(context):
            return context.get('system_memory_percent', 0) > 85
        
        self.alert_manager.add_alert_rule(
            name="High Memory Usage",
            condition=high_memory_condition,
            level=AlertLevel.ERROR,
            message_template="메모리 사용량이 {system_memory_percent:.1f}%로 위험 수준입니다.",
            cooldown_minutes=10
        )
        
        # 디스크 사용량 높음
        def high_disk_condition(context):
            return context.get('system_disk_percent', 0) > 90
        
        self.alert_manager.add_alert_rule(
            name="High Disk Usage",
            condition=high_disk_condition,
            level=AlertLevel.CRITICAL,
            message_template="디스크 사용량이 {system_disk_percent:.1f}%로 임계치를 초과했습니다.",
            cooldown_minutes=15
        )
        
        # 프로세스 메모리 누수
        def memory_leak_condition(context):
            return context.get('process_memory_mb', 0) > 2048  # 2GB 이상
        
        self.alert_manager.add_alert_rule(
            name="Process Memory Leak",
            condition=memory_leak_condition,
            level=AlertLevel.ERROR,
            message_template="프로세스 메모리 사용량이 {process_memory_mb:.0f}MB로 메모리 누수 의심됩니다.",
            cooldown_minutes=10
        )
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring:
            self.monitoring = True
            
            # 메트릭 수집 시작
            self.metric_collector.start_collection()
            
            # 모니터링 루프 시작
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("모니터링 시스템 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        
        # 메트릭 수집 중지
        self.metric_collector.stop_collection()
        
        # 모니터링 스레드 종료
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("모니터링 시스템 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # 최신 메트릭 수집
                context = self._build_alert_context()
                
                # 알림 규칙 확인
                self.alert_manager.check_alert_rules(context)
                
                time.sleep(30)  # 30초마다 확인
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(30)
    
    def _build_alert_context(self) -> Dict[str, Any]:
        """알림 컨텍스트 구성"""
        context = {}
        
        # 최근 메트릭들을 컨텍스트에 추가
        recent_time = time.time() - 60  # 최근 1분
        recent_metrics = self.metric_collector.get_metrics(since=recent_time)
        
        for metric in recent_metrics:
            if metric.name not in context:
                context[metric.name] = metric.value
        
        return context
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE):
        """사용자 정의 메트릭 추가"""
        self.metric_collector.add_metric(name, value, metric_type)
    
    def trigger_custom_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO):
        """사용자 정의 알림 발생"""
        alert = Alert(
            id=f"custom_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            timestamp=time.time(),
            source="CustomAlert"
        )
        
        self.alert_manager.trigger_alert(alert)
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """시스템 대시보드 데이터"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'monitoring_active': self.monitoring,
                'metrics_collected': len(self.metric_collector.metrics),
                'active_alerts': len(self.alert_manager.active_alerts),
                'alert_rules': len(self.alert_manager.alert_rules)
            },
            'recent_metrics': {
                name: self.metric_collector.get_metric_summary(name, 10)
                for name in ['system_cpu_percent', 'system_memory_percent', 'process_memory_mb']
            },
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            'recent_alerts': [asdict(alert) for alert in self.alert_manager.get_alert_history(limit=5)]
        }


# 전역 모니터링 시스템
global_monitoring = MonitoringSystem()


# 편의 함수들
def start_monitoring():
    """모니터링 시작 (편의 함수)"""
    global_monitoring.start_monitoring()


def stop_monitoring():
    """모니터링 중지 (편의 함수)"""
    global_monitoring.stop_monitoring()


def add_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE):
    """메트릭 추가 (편의 함수)"""
    global_monitoring.add_custom_metric(name, value, metric_type)


def trigger_alert(title: str, message: str, level: AlertLevel = AlertLevel.INFO):
    """알림 발생 (편의 함수)"""
    global_monitoring.trigger_custom_alert(title, message, level)


def get_dashboard():
    """대시보드 데이터 (편의 함수)"""
    return global_monitoring.get_system_dashboard()


# 사용 예시
if __name__ == "__main__":
    
    async def test_monitoring_system():
        print("📊 모니터링 및 알림 시스템 테스트")
        
        # 모니터링 시작
        start_monitoring()
        
        # 사용자 정의 메트릭 추가
        add_metric("trading_positions", 3.0)
        add_metric("trading_pnl", 125.50)
        
        # 사용자 정의 알림 발생
        trigger_alert("트레이딩 시작", "VPS 트레이딩 시스템이 시작되었습니다.", AlertLevel.INFO)
        
        # 경고 알림 시뮬레이션
        trigger_alert("높은 CPU 사용량", "CPU 사용량이 85%입니다.", AlertLevel.WARNING)
        
        # 잠시 대기하여 메트릭 수집
        print("메트릭 수집 대기 중...")
        await asyncio.sleep(5)
        
        # 대시보드 데이터 출력
        dashboard = get_dashboard()
        print(f"\n📈 시스템 대시보드:")
        print(json.dumps(dashboard, indent=2, default=str, ensure_ascii=False))
        
        # 모니터링 중지
        print(f"\n모니터링 중지...")
        stop_monitoring()
    
    # 테스트 실행
    asyncio.run(test_monitoring_system())