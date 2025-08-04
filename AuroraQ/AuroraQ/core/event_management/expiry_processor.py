#!/usr/bin/env python3
"""
이벤트 만료 처리 시스템
P8-2: Event Expiry Processing System
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .ttl_event_manager import EventEntry, EventStatus, TTLAction

class ExpiryAction(Enum):
    """만료 액션 타입"""
    DELETE = "delete"              # 즉시 삭제
    SOFT_DELETE = "soft_delete"    # 소프트 삭제 (상태만 변경)
    ARCHIVE = "archive"            # 아카이브로 이동
    BACKUP = "backup"              # 백업 생성
    NOTIFY = "notify"              # 알림 발송
    CALLBACK = "callback"          # 콜백 호출
    EXTEND = "extend"              # TTL 연장
    ESCALATE = "escalate"          # 에스컬레이션
    QUARANTINE = "quarantine"      # 격리
    AUDIT_LOG = "audit_log"        # 감사 로그 기록

class NotificationType(Enum):
    """알림 타입"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    LOG = "log"

@dataclass
class ExpiryRule:
    """만료 규칙"""
    rule_id: str
    event_type_pattern: str        # 이벤트 타입 패턴 (regex)
    priority_filter: Optional[List[str]] = None  # 우선순위 필터
    tag_filter: Optional[List[str]] = None       # 태그 필터
    
    # 액션 설정
    actions: List[ExpiryAction] = field(default_factory=list)
    
    # 알림 설정
    notification_lead_time_minutes: int = 5
    notification_types: List[NotificationType] = field(default_factory=list)
    notification_recipients: List[str] = field(default_factory=list)
    
    # 연장 설정
    auto_extend_conditions: Optional[Dict[str, Any]] = None
    max_extensions: int = 3
    extension_duration_seconds: int = 3600
    
    # 콜백 설정
    callback_url: Optional[str] = None
    callback_headers: Optional[Dict[str, str]] = None
    callback_timeout_seconds: int = 30
    
    # 메타데이터
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class ExpiryContext:
    """만료 처리 컨텍스트"""
    event: EventEntry
    rule: ExpiryRule
    processor_id: str
    started_at: datetime = field(default_factory=datetime.now)
    
    # 처리 상태
    notifications_sent: List[NotificationType] = field(default_factory=list)
    callbacks_executed: List[str] = field(default_factory=list)
    actions_completed: List[ExpiryAction] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # 결과
    success: bool = False
    completion_time: Optional[datetime] = None

class ExpiryHandler(Protocol):
    """만료 핸들러 프로토콜"""
    
    async def handle_expiry(self, context: ExpiryContext) -> bool:
        """만료 처리"""
        ...
    
    async def handle_notification(self, context: ExpiryContext, 
                                notification_type: NotificationType) -> bool:
        """알림 처리"""
        ...

@dataclass
class ProcessingResult:
    """처리 결과"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    # 액션별 통계
    actions_executed: Dict[ExpiryAction, int] = field(default_factory=dict)
    notifications_sent: Dict[NotificationType, int] = field(default_factory=dict)
    
    # 에러 정보
    errors: List[str] = field(default_factory=list)
    
    # 시간 정보
    processing_time_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class ExpiryProcessor:
    """이벤트 만료 처리기"""
    
    def __init__(self, ttl_manager=None):
        # 로거 초기화
        self.logger = logging.getLogger(__name__)
        
        # TTL 매니저 참조
        from .ttl_event_manager import get_ttl_event_manager
        self.ttl_manager = ttl_manager or get_ttl_event_manager()
        
        # 만료 규칙
        self.expiry_rules: Dict[str, ExpiryRule] = {}
        
        # 핸들러
        self.custom_handlers: Dict[str, ExpiryHandler] = {}
        
        # 알림 설정
        self.notification_config = {
            'email': {
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': 'noreply@auroaq.com'
            },
            'webhook': {
                'default_timeout': 30,
                'retry_count': 3
            }
        }
        
        # 처리 통계
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'last_processing_time': None
        }
        
        self.logger.info("Expiry Processor initialized")
    
    def add_expiry_rule(self, rule: ExpiryRule):
        """만료 규칙 추가"""
        self.expiry_rules[rule.rule_id] = rule
        self.logger.info(f"Added expiry rule: {rule.rule_id}")
    
    def remove_expiry_rule(self, rule_id: str) -> bool:
        """만료 규칙 제거"""
        if rule_id in self.expiry_rules:
            del self.expiry_rules[rule_id]
            self.logger.info(f"Removed expiry rule: {rule_id}")
            return True
        return False
    
    def register_handler(self, event_type: str, handler: ExpiryHandler):
        """커스텀 핸들러 등록"""
        self.custom_handlers[event_type] = handler
        self.logger.info(f"Registered custom handler for: {event_type}")
    
    async def process_expired_events(self, batch_size: int = 100) -> ProcessingResult:
        """만료된 이벤트 배치 처리"""
        result = ProcessingResult(started_at=datetime.now())
        
        try:
            # 만료된 이벤트 조회
            expired_events = await self.ttl_manager.list_events(
                status=None,  # 모든 상태 (만료 체크는 별도)
                limit=batch_size
            )
            
            # 실제 만료된 이벤트 필터링
            truly_expired = [event for event in expired_events if event.is_expired()]
            
            result.total_processed = len(truly_expired)
            
            self.logger.info(f"Processing {result.total_processed} expired events")
            
            # 각 이벤트 처리
            for event in truly_expired:
                try:
                    success = await self._process_single_event(event, result)
                    if success:
                        result.successful += 1
                    else:
                        result.failed += 1
                
                except Exception as e:
                    result.failed += 1
                    error_msg = f"Failed to process event {event.event_id}: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            result.completed_at = datetime.now()
            result.processing_time_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            
            # 통계 업데이트
            self._update_processing_stats(result)
            
            self.logger.info(
                f"Batch processing completed: {result.successful}/{result.total_processed} successful"
            )
            
        except Exception as e:
            result.completed_at = datetime.now()
            error_msg = f"Batch processing failed: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    async def _process_single_event(self, event: EventEntry, result: ProcessingResult) -> bool:
        """단일 이벤트 처리"""
        try:
            # 적용 가능한 규칙 찾기
            applicable_rules = self._find_applicable_rules(event)
            
            if not applicable_rules:
                # 기본 처리
                await self._default_expiry_processing(event, result)
                return True
            
            # 규칙별 처리
            success = True
            for rule in applicable_rules:
                try:
                    context = ExpiryContext(
                        event=event,
                        rule=rule,
                        processor_id=str(uuid.uuid4())
                    )
                    
                    rule_success = await self._process_with_rule(context, result)
                    if not rule_success:
                        success = False
                
                except Exception as e:
                    success = False
                    error_msg = f"Rule {rule.rule_id} processing failed for event {event.event_id}: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            return success
            
        except Exception as e:
            error_msg = f"Single event processing failed for {event.event_id}: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _find_applicable_rules(self, event: EventEntry) -> List[ExpiryRule]:
        """이벤트에 적용 가능한 규칙 찾기"""
        applicable_rules = []
        
        for rule in self.expiry_rules.values():
            if not rule.enabled:
                continue
            
            # 이벤트 타입 패턴 매칭
            import re
            if not re.match(rule.event_type_pattern, event.event_type):
                continue
            
            # 우선순위 필터
            if rule.priority_filter and event.priority.value not in rule.priority_filter:
                continue
            
            # 태그 필터
            if rule.tag_filter and not any(tag in event.tags for tag in rule.tag_filter):
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _process_with_rule(self, context: ExpiryContext, result: ProcessingResult) -> bool:
        """규칙에 따른 처리"""
        try:
            # 커스텀 핸들러 확인
            if context.event.event_type in self.custom_handlers:
                custom_success = await self.custom_handlers[context.event.event_type].handle_expiry(context)
                if not custom_success:
                    return False
            
            # 만료 전 알림 (아직 만료되지 않았지만 임박한 경우)
            if (context.event.is_near_expiry(context.rule.notification_lead_time_minutes) and 
                not context.event.is_expired()):
                await self._send_expiry_warnings(context, result)
                return True  # 알림만 보내고 종료
            
            # 자동 연장 체크
            if await self._check_auto_extension(context, result):
                return True  # 연장됨
            
            # 액션 실행
            for action in context.rule.actions:
                try:
                    await self._execute_action(context, action, result)
                    context.actions_completed.append(action)
                    
                    # 통계 업데이트
                    if action not in result.actions_executed:
                        result.actions_executed[action] = 0
                    result.actions_executed[action] += 1
                
                except Exception as e:
                    error_msg = f"Action {action} failed for event {context.event.event_id}: {e}"
                    context.errors.append(error_msg)
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 알림 발송
            for notification_type in context.rule.notification_types:
                try:
                    await self._send_notification(context, notification_type, result)
                    context.notifications_sent.append(notification_type)
                    
                    # 통계 업데이트
                    if notification_type not in result.notifications_sent:
                        result.notifications_sent[notification_type] = 0
                    result.notifications_sent[notification_type] += 1
                
                except Exception as e:
                    error_msg = f"Notification {notification_type} failed for event {context.event.event_id}: {e}"
                    context.errors.append(error_msg)
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 콜백 실행
            if context.rule.callback_url:
                try:
                    await self._execute_callback(context, result)
                    context.callbacks_executed.append(context.rule.callback_url)
                
                except Exception as e:
                    error_msg = f"Callback failed for event {context.event.event_id}: {e}"
                    context.errors.append(error_msg)
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 처리 완료
            context.success = len(context.errors) == 0
            context.completion_time = datetime.now()
            
            return context.success
            
        except Exception as e:
            error_msg = f"Rule processing failed: {e}"
            context.errors.append(error_msg)
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    async def _default_expiry_processing(self, event: EventEntry, result: ProcessingResult):
        """기본 만료 처리"""
        try:
            # 기본 액션들
            default_actions = event.actions if event.actions else [TTLAction.ARCHIVE]
            
            for action in default_actions:
                if action == TTLAction.DELETE:
                    await self.ttl_manager.delete_event(event.event_id)
                elif action == TTLAction.ARCHIVE:
                    event.status = EventStatus.ARCHIVED
                    await self.ttl_manager.update_event(event.event_id, status=EventStatus.ARCHIVED)
                elif action == TTLAction.NOTIFY:
                    self.logger.info(f"Event {event.event_id} expired")
                
                # 통계 업데이트
                if action not in result.actions_executed:
                    result.actions_executed[action] = 0
                result.actions_executed[action] += 1
            
            self.logger.debug(f"Default processing completed for event {event.event_id}")
            
        except Exception as e:
            error_msg = f"Default processing failed for event {event.event_id}: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
    
    async def _check_auto_extension(self, context: ExpiryContext, result: ProcessingResult) -> bool:
        """자동 연장 체크"""
        if not context.rule.auto_extend_conditions:
            return False
        
        try:
            conditions = context.rule.auto_extend_conditions
            
            # 접근 빈도 체크
            if 'min_access_count' in conditions:
                if context.event.access_count < conditions['min_access_count']:
                    return False
            
            # 최근 접근 시간 체크
            if 'recent_access_hours' in conditions:
                if context.event.last_accessed_at:
                    hours_since_access = (
                        datetime.now() - context.event.last_accessed_at
                    ).total_seconds() / 3600
                    
                    if hours_since_access > conditions['recent_access_hours']:
                        return False
                else:
                    return False
            
            # 최대 연장 횟수 체크
            extension_count = context.event.metadata.get('extension_count', 0)
            if extension_count >= context.rule.max_extensions:
                return False
            
            # 연장 실행
            await self.ttl_manager.update_event(
                context.event.event_id,
                extend_ttl_seconds=context.rule.extension_duration_seconds,
                metadata={
                    **context.event.metadata,
                    'extension_count': extension_count + 1,
                    'last_extension_at': datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Auto-extended TTL for event {context.event.event_id}")
            
            # 통계 업데이트
            if ExpiryAction.EXTEND not in result.actions_executed:
                result.actions_executed[ExpiryAction.EXTEND] = 0
            result.actions_executed[ExpiryAction.EXTEND] += 1
            
            return True
            
        except Exception as e:
            error_msg = f"Auto-extension failed for event {context.event.event_id}: {e}"
            context.errors.append(error_msg)
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    async def _execute_action(self, context: ExpiryContext, action: ExpiryAction, result: ProcessingResult):
        """액션 실행"""
        if action == ExpiryAction.DELETE:
            await self.ttl_manager.delete_event(context.event.event_id)
        
        elif action == ExpiryAction.SOFT_DELETE:
            await self.ttl_manager.update_event(
                context.event.event_id,
                status=EventStatus.EXPIRED
            )
        
        elif action == ExpiryAction.ARCHIVE:
            await self.ttl_manager.update_event(
                context.event.event_id,
                status=EventStatus.ARCHIVED
            )
        
        elif action == ExpiryAction.BACKUP:
            # 백업 로직 (간단 구현)
            backup_data = {
                'event': context.event.to_dict(),
                'backed_up_at': datetime.now().isoformat(),
                'backup_reason': 'expiry_processing'
            }
            
            backup_file = Path(self.ttl_manager.temp_dir) / f"backup_{context.event.event_id}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
        
        elif action == ExpiryAction.QUARANTINE:
            await self.ttl_manager.update_event(
                context.event.event_id,
                tags=[*context.event.tags, "quarantined"],
                metadata={
                    **context.event.metadata,
                    'quarantined_at': datetime.now().isoformat(),
                    'quarantine_reason': 'expired'
                }
            )
        
        elif action == ExpiryAction.AUDIT_LOG:
            audit_entry = {
                'event_id': context.event.event_id,
                'action': 'expired',
                'timestamp': datetime.now().isoformat(),
                'processor_id': context.processor_id,
                'rule_id': context.rule.rule_id
            }
            
            # 감사 로그 기록 (간단 구현)
            audit_file = Path(self.ttl_manager.temp_dir) / "audit.log"
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        
        self.logger.debug(f"Executed action {action} for event {context.event.event_id}")
    
    async def _send_expiry_warnings(self, context: ExpiryContext, result: ProcessingResult):
        """만료 경고 알림 발송"""
        for notification_type in context.rule.notification_types:
            try:
                await self._send_notification(context, notification_type, result, is_warning=True)
                
                # 통계 업데이트
                if notification_type not in result.notifications_sent:
                    result.notifications_sent[notification_type] = 0
                result.notifications_sent[notification_type] += 1
            
            except Exception as e:
                error_msg = f"Warning notification {notification_type} failed for event {context.event.event_id}: {e}"
                context.errors.append(error_msg)
                result.errors.append(error_msg)
                self.logger.error(error_msg)
    
    async def _send_notification(self, context: ExpiryContext, 
                               notification_type: NotificationType, 
                               result: ProcessingResult,
                               is_warning: bool = False):
        """알림 발송"""
        
        # 메시지 내용 구성
        subject = f"Event {'Warning' if is_warning else 'Expiry'}: {context.event.event_type}"
        
        time_info = (
            f"Expires in: {context.event.time_to_expiry()}" if is_warning
            else "Event has expired"
        )
        
        message_body = f"""
Event ID: {context.event.event_id}
Event Type: {context.event.event_type}
Priority: {context.event.priority.value}
Status: {time_info}
Created: {context.event.created_at}
Expires: {context.event.expires_at}

Data: {json.dumps(context.event.data, indent=2)}
Tags: {', '.join(context.event.tags)}

Rule: {context.rule.rule_id}
Processor: {context.processor_id}
"""
        
        if notification_type == NotificationType.EMAIL:
            await self._send_email_notification(context, subject, message_body)
        
        elif notification_type == NotificationType.WEBHOOK:
            await self._send_webhook_notification(context, subject, message_body)
        
        elif notification_type == NotificationType.LOG:
            level = logging.WARNING if is_warning else logging.INFO
            self.logger.log(level, f"{subject}: {context.event.event_id}")
        
        # 다른 알림 타입들은 필요시 구현
        else:
            self.logger.info(f"Notification type {notification_type} not implemented")
    
    async def _send_email_notification(self, context: ExpiryContext, subject: str, body: str):
        """이메일 알림 발송"""
        try:
            config = self.notification_config['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(context.rule.notification_recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # SMTP 서버 연결 및 발송 (실제 환경에서는 설정 필요)
            # 여기서는 로그만 기록
            self.logger.info(f"Email notification sent for event {context.event.event_id}")
            
        except Exception as e:
            raise Exception(f"Email notification failed: {e}")
    
    async def _send_webhook_notification(self, context: ExpiryContext, subject: str, body: str):
        """웹훅 알림 발송"""
        try:
            config = self.notification_config['webhook']
            
            payload = {
                'subject': subject,
                'message': body,
                'event': context.event.to_dict(),
                'rule_id': context.rule.rule_id,
                'processor_id': context.processor_id,
                'timestamp': datetime.now().isoformat()
            }
            
            headers = context.rule.callback_headers or {}
            headers['Content-Type'] = 'application/json'
            
            timeout = aiohttp.ClientTimeout(total=config['default_timeout'])
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 알림 수신자들에게 웹훅 발송
                for recipient_url in context.rule.notification_recipients:
                    async with session.post(recipient_url, json=payload, headers=headers) as response:
                        if response.status >= 400:
                            raise Exception(f"Webhook failed with status {response.status}")
            
            self.logger.info(f"Webhook notification sent for event {context.event.event_id}")
            
        except Exception as e:
            raise Exception(f"Webhook notification failed: {e}")
    
    async def _execute_callback(self, context: ExpiryContext, result: ProcessingResult):
        """콜백 실행"""
        try:
            payload = {
                'event_id': context.event.event_id,
                'event_type': context.event.event_type,
                'status': 'expired',
                'expired_at': datetime.now().isoformat(),
                'data': context.event.data,
                'rule_id': context.rule.rule_id,
                'processor_id': context.processor_id,
                'actions_completed': [action.value for action in context.actions_completed]
            }
            
            headers = context.rule.callback_headers or {}
            headers['Content-Type'] = 'application/json'
            
            timeout = aiohttp.ClientTimeout(total=context.rule.callback_timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(context.rule.callback_url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        raise Exception(f"Callback failed with status {response.status}")
            
            self.logger.debug(f"Callback executed for event {context.event.event_id}")
            
        except Exception as e:
            raise Exception(f"Callback execution failed: {e}")
    
    def _update_processing_stats(self, result: ProcessingResult):
        """처리 통계 업데이트"""
        self.processing_stats['total_processed'] += result.total_processed
        self.processing_stats['successful'] += result.successful
        self.processing_stats['failed'] += result.failed
        self.processing_stats['last_processing_time'] = result.completed_at
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 조회"""
        return {
            **self.processing_stats,
            'success_rate': (
                self.processing_stats['successful'] / 
                max(1, self.processing_stats['total_processed'])
            ) * 100,
            'rules_count': len(self.expiry_rules),
            'handlers_count': len(self.custom_handlers)
        }
    
    def get_expiry_rules(self) -> List[ExpiryRule]:
        """만료 규칙 목록 조회"""
        return list(self.expiry_rules.values())
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("Expiry Processor cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Expiry Processor cleanup failed: {e}")

# 전역 프로세서
_global_expiry_processor = None

def get_expiry_processor(ttl_manager=None) -> ExpiryProcessor:
    """전역 만료 처리기 반환"""
    global _global_expiry_processor
    if _global_expiry_processor is None:
        _global_expiry_processor = ExpiryProcessor(ttl_manager)
    return _global_expiry_processor

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_expiry_processor():
        print("🧪 Expiry Processor 테스트")
        
        # 프로세서 초기화
        processor = ExpiryProcessor()
        
        print("\n1️⃣ 만료 규칙 추가")
        rule = ExpiryRule(
            rule_id="test_rule",
            event_type_pattern="test_.*",
            actions=[ExpiryAction.NOTIFY, ExpiryAction.ARCHIVE],
            notification_types=[NotificationType.LOG],
            notification_recipients=["test@example.com"]
        )
        processor.add_expiry_rule(rule)
        print(f"  추가된 규칙: {rule.rule_id}")
        
        print("\n2️⃣ 처리 통계 조회")
        stats = processor.get_processing_stats()
        print(f"  규칙 수: {stats['rules_count']}")
        print(f"  핸들러 수: {stats['handlers_count']}")
        
        print("\n3️⃣ 만료 이벤트 처리 (시뮬레이션)")
        # 실제 이벤트가 없으므로 빈 배치 처리
        result = await processor.process_expired_events(batch_size=10)
        print(f"  처리된 이벤트: {result.total_processed}")
        print(f"  성공: {result.successful}, 실패: {result.failed}")
        
        print("\n🎉 Expiry Processor 테스트 완료!")
        
        # 정리
        await processor.cleanup()
    
    # 테스트 실행
    asyncio.run(test_expiry_processor())