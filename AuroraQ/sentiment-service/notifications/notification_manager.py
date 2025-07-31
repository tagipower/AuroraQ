#!/usr/bin/env python3
"""
Notification Manager for AuroraQ Sentiment Service
알림 관리자 - 모든 알림을 통합 관리
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import structlog

from .telegram_notifier import (
    TelegramNotifier, 
    NotificationLevel, 
    NotificationChannel
)

logger = structlog.get_logger(__name__)

@dataclass
class NotificationConfig:
    """알림 설정"""
    enabled: bool = True
    channels: List[str] = None
    min_level: str = "info"
    rate_limit_minutes: int = 5
    quiet_hours: Dict[str, int] = None  # {"start": 23, "end": 7}
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ["general"]
        if self.quiet_hours is None:
            self.quiet_hours = {"start": 23, "end": 7}

class NotificationManager:
    """통합 알림 관리자"""
    
    def __init__(self,
                 telegram_bot_token: str,
                 telegram_chat_configs: Dict[str, Dict[str, Any]],
                 notification_configs: Dict[str, NotificationConfig] = None):
        """
        초기화
        
        Args:
            telegram_bot_token: 텔레그램 봇 토큰
            telegram_chat_configs: 텔레그램 채팅방 설정
            notification_configs: 알림 유형별 설정
        """
        self.telegram_notifier = TelegramNotifier(
            bot_token=telegram_bot_token,
            chat_configs=telegram_chat_configs
        )
        
        # 기본 알림 설정
        self.configs = notification_configs or {
            "trading": NotificationConfig(
                enabled=True,
                channels=["trading"],
                min_level="info",
                rate_limit_minutes=1
            ),
            "events": NotificationConfig(
                enabled=True,
                channels=["events"],
                min_level="info",
                rate_limit_minutes=5
            ),
            "system": NotificationConfig(
                enabled=True,
                channels=["system"],
                min_level="warning",
                rate_limit_minutes=10,
                quiet_hours={"start": 23, "end": 7}
            ),
            "performance": NotificationConfig(
                enabled=True,
                channels=["system"],
                min_level="info",
                rate_limit_minutes=60
            )
        }
        
        # Rate limiting을 위한 메시지 기록
        self.last_sent = {}
        
        # 통계
        self.stats = {
            "total_notifications": 0,
            "by_type": {},
            "by_level": {},
            "rate_limited": 0,
            "quiet_hour_suppressed": 0,
            "start_time": datetime.now()
        }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.telegram_notifier.start()
        
        # 시작 메시지 발송
        await self.send_startup_notification()
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.telegram_notifier.stop()

    def _is_quiet_hours(self, config: NotificationConfig) -> bool:
        """조용한 시간인지 확인"""
        if not config.quiet_hours:
            return False
            
        now = datetime.now()
        current_hour = now.hour
        start_hour = config.quiet_hours["start"]
        end_hour = config.quiet_hours["end"]
        
        if start_hour <= end_hour:
            # 같은 날 (예: 23시-7시가 아닌 경우)
            return start_hour <= current_hour <= end_hour
        else:
            # 다음 날까지 (예: 23시-7시)
            return current_hour >= start_hour or current_hour <= end_hour

    def _should_send_notification(self,
                                notification_type: str,
                                level: NotificationLevel,
                                message_key: str) -> bool:
        """알림을 보낼지 결정"""
        
        config = self.configs.get(notification_type)
        if not config or not config.enabled:
            return False
        
        # 레벨 체크
        level_priority = {
            "info": 0,
            "warning": 1, 
            "error": 2,
            "critical": 3,
            "trading": 1,
            "event": 1
        }
        
        if level_priority.get(level.value, 0) < level_priority.get(config.min_level, 0):
            return False
        
        # Rate limiting 체크
        now = datetime.now()
        if message_key in self.last_sent:
            last_time = self.last_sent[message_key]
            if now - last_time < timedelta(minutes=config.rate_limit_minutes):
                self.stats["rate_limited"] += 1
                return False
        
        # 조용한 시간 체크 (CRITICAL은 예외)
        if (level != NotificationLevel.CRITICAL and 
            self._is_quiet_hours(config)):
            self.stats["quiet_hour_suppressed"] += 1
            return False
        
        return True

    async def send_trading_signal_notification(self,
                                             signal: Dict[str, Any],
                                             trading_mode: str = "LIVE"):
        """매매 신호 알림"""
        
        message_key = f"trading:{signal.get('symbol')}:{signal.get('direction')}"
        
        if not self._should_send_notification("trading", NotificationLevel.TRADING, message_key):
            return
        
        try:
            await self.telegram_notifier.send_trading_signal(signal, trading_mode)
            
            # 기록 업데이트
            self.last_sent[message_key] = datetime.now()
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["trading"] = self.stats["by_type"].get("trading", 0) + 1
            self.stats["by_level"]["trading"] = self.stats["by_level"].get("trading", 0) + 1
            
            logger.info("Trading signal notification sent",
                       symbol=signal.get("symbol"),
                       direction=signal.get("direction"),
                       mode=trading_mode)
            
        except Exception as e:
            logger.error("Failed to send trading signal notification", error=str(e))

    async def send_big_event_notification(self, event: Dict[str, Any]):
        """빅 이벤트 알림"""
        
        message_key = f"event:{event.get('event_type')}:{event.get('symbol')}"
        
        if not self._should_send_notification("events", NotificationLevel.EVENT, message_key):
            return
        
        try:
            await self.telegram_notifier.send_big_event(event)
            
            # 기록 업데이트
            self.last_sent[message_key] = datetime.now()
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["events"] = self.stats["by_type"].get("events", 0) + 1
            self.stats["by_level"]["event"] = self.stats["by_level"].get("event", 0) + 1
            
            logger.info("Big event notification sent",
                       event_type=event.get("event_type"),
                       symbol=event.get("symbol"))
            
        except Exception as e:
            logger.error("Failed to send big event notification", error=str(e))

    async def send_system_notification(self,
                                     message: str,
                                     level: NotificationLevel = NotificationLevel.INFO,
                                     service: str = "sentiment-service"):
        """시스템 알림"""
        
        message_key = f"system:{level.value}:{hash(message)}"
        
        if not self._should_send_notification("system", level, message_key):
            return
        
        try:
            await self.telegram_notifier.send_system_alert(message, level, service)
            
            # 기록 업데이트
            self.last_sent[message_key] = datetime.now()
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["system"] = self.stats["by_type"].get("system", 0) + 1
            self.stats["by_level"][level.value] = self.stats["by_level"].get(level.value, 0) + 1
            
            logger.info("System notification sent",
                       level=level.value,
                       service=service)
            
        except Exception as e:
            logger.error("Failed to send system notification", error=str(e))

    async def send_performance_notification(self, stats: Dict[str, Any]):
        """성능 보고서 알림"""
        
        message_key = "performance:report"
        
        if not self._should_send_notification("performance", NotificationLevel.INFO, message_key):
            return
        
        try:
            await self.telegram_notifier.send_performance_report(stats)
            
            # 기록 업데이트
            self.last_sent[message_key] = datetime.now()
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["performance"] = self.stats["by_type"].get("performance", 0) + 1
            self.stats["by_level"]["info"] = self.stats["by_level"].get("info", 0) + 1
            
            logger.info("Performance notification sent")
            
        except Exception as e:
            logger.error("Failed to send performance notification", error=str(e))

    async def send_startup_notification(self):
        """서비스 시작 알림"""
        try:
            await self.telegram_notifier.send_startup_message()
            
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["system"] = self.stats["by_type"].get("system", 0) + 1
            self.stats["by_level"]["info"] = self.stats["by_level"].get("info", 0) + 1
            
            logger.info("Startup notification sent")
            
        except Exception as e:
            logger.error("Failed to send startup notification", error=str(e))

    async def send_health_alert(self,
                              component: str,
                              status: str,
                              details: Optional[Dict[str, Any]] = None):
        """헬스체크 알림"""
        
        if status == "healthy":
            return  # 정상 상태는 알림 안함
        
        level = NotificationLevel.ERROR if status == "unhealthy" else NotificationLevel.WARNING
        
        message = f"{component} 상태: {status}"
        if details:
            message += f"\n세부사항: {json.dumps(details, ensure_ascii=False, indent=2)}"
        
        await self.send_system_notification(message, level)

    async def send_error_notification(self,
                                    error_message: str,
                                    error_type: str = "general",
                                    stack_trace: Optional[str] = None):
        """에러 알림"""
        
        level = NotificationLevel.CRITICAL if "critical" in error_type.lower() else NotificationLevel.ERROR
        
        message = f"에러 발생 ({error_type}): {error_message}"
        if stack_trace:
            # 스택 트레이스는 길어질 수 있으므로 요약
            lines = stack_trace.split('\n')
            if len(lines) > 10:
                message += f"\n스택 트레이스 (처음 5줄):\n" + '\n'.join(lines[:5])
                message += f"\n... ({len(lines)-5}줄 더)"
            else:
                message += f"\n스택 트레이스:\n{stack_trace}"
        
        await self.send_system_notification(message, level)

    async def send_custom_notification(self,
                                     text: str,
                                     level: NotificationLevel = NotificationLevel.INFO,
                                     channel: NotificationChannel = NotificationChannel.GENERAL,
                                     disable_notification: bool = False,
                                     bypass_rate_limit: bool = False):
        """커스텀 알림"""
        
        message_key = f"custom:{channel.value}:{hash(text)}"
        
        # Rate limit 체크 (bypass 가능)
        if not bypass_rate_limit:
            config = self.configs.get("system", NotificationConfig())
            if not self._should_send_notification("system", level, message_key):
                return
        
        try:
            await self.telegram_notifier.send_notification(
                text=text,
                level=level,
                channel=channel,
                disable_notification=disable_notification
            )
            
            # 기록 업데이트
            self.last_sent[message_key] = datetime.now()
            self.stats["total_notifications"] += 1
            self.stats["by_type"]["custom"] = self.stats["by_type"].get("custom", 0) + 1
            self.stats["by_level"][level.value] = self.stats["by_level"].get(level.value, 0) + 1
            
            logger.info("Custom notification sent",
                       channel=channel.value,
                       level=level.value)
            
        except Exception as e:
            logger.error("Failed to send custom notification", error=str(e))

    def update_config(self, notification_type: str, config: NotificationConfig):
        """알림 설정 업데이트"""
        self.configs[notification_type] = config
        logger.info("Notification config updated", type=notification_type)

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        uptime = datetime.now() - self.stats["start_time"]
        
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "telegram_stats": self.telegram_notifier.get_stats(),
            "configs": {k: {
                "enabled": v.enabled,
                "channels": v.channels,
                "min_level": v.min_level,
                "rate_limit_minutes": v.rate_limit_minutes
            } for k, v in self.configs.items()}
        }

    async def test_all_notifications(self):
        """모든 알림 타입 테스트"""
        logger.info("Testing all notification types...")
        
        # 매매 신호 테스트
        test_signal = {
            "symbol": "BTC",
            "direction": "buy",
            "strength": "strong",
            "confidence": 0.85,
            "sentiment_score": 0.642
        }
        await self.send_trading_signal_notification(test_signal, "TEST")
        
        # 빅 이벤트 테스트
        test_event = {
            "event_type": "TEST_EVENT",
            "symbol": "BTC",
            "impact_score": 7.5,
            "confidence": 0.90,
            "description": "텔레그램 알림 시스템 테스트"
        }
        await self.send_big_event_notification(test_event)
        
        # 시스템 알림 테스트
        await self.send_system_notification(
            "알림 시스템 테스트 완료", 
            NotificationLevel.INFO
        )
        
        # 성능 보고서 테스트
        test_stats = {
            "avg_response_time": 0.123,
            "total_requests": 1000,
            "success_rate": 0.995,
            "memory_usage": 65.2,
            "cpu_usage": 45.8,
            "disk_usage": 78.1,
            "news_collected": 150,
            "finbert_processed": 75,
            "events_detected": 3
        }
        await self.send_performance_notification(test_stats)
        
        logger.info("All notification tests completed")

    async def cleanup_old_records(self, hours: int = 24):
        """오래된 기록 정리"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Rate limiting 기록 정리
        to_remove = []
        for key, timestamp in self.last_sent.items():
            if timestamp < cutoff:
                to_remove.append(key)
        
        for key in to_remove:
            del self.last_sent[key]
        
        # 텔레그램 알림기 정리
        await self.telegram_notifier.cleanup_old_messages(hours)
        
        logger.info("Cleaned up old notification records", 
                   removed=len(to_remove),
                   remaining=len(self.last_sent))


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_notification_manager():
        """알림 관리자 테스트"""
        
        # 테스트 설정
        bot_token = "YOUR_BOT_TOKEN"
        chat_configs = {
            "general": {"chat_id": "YOUR_CHAT_ID"},
            "trading": {"chat_id": "YOUR_CHAT_ID"},
            "events": {"chat_id": "YOUR_CHAT_ID"},
            "system": {"chat_id": "YOUR_CHAT_ID"}
        }
        
        async with NotificationManager(bot_token, chat_configs) as manager:
            print("🚀 알림 관리자 시작")
            
            # 모든 알림 타입 테스트
            await manager.test_all_notifications()
            
            # 5초 대기
            await asyncio.sleep(5)
            
            # 통계 출력
            stats = manager.get_stats()
            print(f"📊 통계: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    # 테스트 실행
    asyncio.run(test_notification_manager())