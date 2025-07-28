#!/usr/bin/env python3
"""
AuroraQ Telegram Notification System
실거래 환경용 종합 알림 시스템
"""

import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import io
import base64
from pathlib import Path

class NotificationLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class NotificationType(Enum):
    TRADE = "trade"
    SIGNAL = "signal"
    RISK = "risk"
    SYSTEM = "system"
    HEALTH = "health"
    ALERT = "alert"

@dataclass
class TelegramConfig:
    """Telegram 설정"""
    bot_token: str
    chat_id: str
    enabled: bool = True
    max_message_length: int = 4096
    rate_limit_messages: int = 30  # 분당 메시지 수 제한
    rate_limit_window: int = 60    # 제한 시간 (초)
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass 
class NotificationMessage:
    """알림 메시지"""
    level: NotificationLevel
    type: NotificationType
    title: str
    message: str
    timestamp: datetime = None
    data: Dict[str, Any] = None
    parse_mode: str = "HTML"
    disable_preview: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}

class TelegramNotifier:
    """Telegram 알림 시스템"""
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 메시지 레이트 제한
        self.message_history: List[float] = []
        
        # 메시지 캐시 (중복 방지)
        self.message_cache: Dict[str, float] = {}
        self.cache_ttl = 300  # 5분
        
        # 세션 관리
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 상태 추적
        self.last_health_check = None
        self.notification_stats = {
            "sent": 0,
            "failed": 0,
            "rate_limited": 0,
            "duplicates_blocked": 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 생성/반환"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def close(self):
        """리소스 정리"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _format_message(self, notification: NotificationMessage) -> str:
        """메시지 포맷팅"""
        
        # 레벨별 이모지
        level_icons = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        
        # 타입별 이모지
        type_icons = {
            NotificationType.TRADE: "💰",
            NotificationType.SIGNAL: "📊",
            NotificationType.RISK: "🛡️",
            NotificationType.SYSTEM: "⚙️",
            NotificationType.HEALTH: "🏥",
            NotificationType.ALERT: "🔔"
        }
        
        level_icon = level_icons.get(notification.level, "📝")
        type_icon = type_icons.get(notification.type, "📬")
        
        # 기본 메시지 구성
        formatted_message = f"{level_icon} {type_icon} <b>{notification.title}</b>\n\n"
        formatted_message += f"{notification.message}\n\n"
        
        # 시간 정보
        time_str = notification.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message += f"🕐 <i>{time_str}</i>"
        
        # 추가 데이터가 있는 경우
        if notification.data:
            formatted_message += "\n\n<b>Details:</b>\n"
            for key, value in notification.data.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_message += f"• {key}: {value:.4f}\n"
                    else:
                        formatted_message += f"• {key}: {value:,}\n"
                else:
                    formatted_message += f"• {key}: {value}\n"
        
        # 메시지 길이 제한
        if len(formatted_message) > self.config.max_message_length:
            truncated_length = self.config.max_message_length - 100
            formatted_message = formatted_message[:truncated_length] + "\n\n... <i>(message truncated)</i>"
        
        return formatted_message
    
    def _should_send_message(self, notification: NotificationMessage) -> bool:
        """메시지 전송 여부 결정"""
        
        # 비활성화된 경우
        if not self.config.enabled:
            return False
        
        # 중복 메시지 체크
        message_hash = hash(f"{notification.title}:{notification.message}")
        current_time = time.time()
        
        if message_hash in self.message_cache:
            last_sent_time = self.message_cache[message_hash]
            if current_time - last_sent_time < self.cache_ttl:
                self.notification_stats["duplicates_blocked"] += 1
                return False
        
        # 레이트 제한 체크
        current_time = time.time()
        
        # 시간 윈도우 내의 메시지들만 유지
        self.message_history = [
            msg_time for msg_time in self.message_history
            if current_time - msg_time < self.config.rate_limit_window
        ]
        
        if len(self.message_history) >= self.config.rate_limit_messages:
            self.notification_stats["rate_limited"] += 1
            self.logger.warning(f"Rate limit exceeded: {len(self.message_history)} messages in {self.config.rate_limit_window}s")
            return False
        
        return True
    
    async def send_notification(self, notification: NotificationMessage) -> bool:
        """알림 전송"""
        
        if not self._should_send_message(notification):
            return False
        
        try:
            session = await self._get_session()
            formatted_message = self._format_message(notification)
            
            # Telegram Bot API URL
            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            
            # 요청 데이터
            data = {
                "chat_id": self.config.chat_id,
                "text": formatted_message,
                "parse_mode": notification.parse_mode,
                "disable_web_page_preview": notification.disable_preview
            }
            
            # 재시도 로직
            for attempt in range(self.config.retry_attempts):
                try:
                    async with session.post(url, json=data) as response:
                        if response.status == 200:
                            # 성공
                            current_time = time.time()
                            self.message_history.append(current_time)
                            
                            # 메시지 캐시 업데이트
                            message_hash = hash(f"{notification.title}:{notification.message}")
                            self.message_cache[message_hash] = current_time
                            
                            # 캐시 정리
                            self._cleanup_cache()
                            
                            self.notification_stats["sent"] += 1
                            self.logger.debug(f"Notification sent: {notification.title}")
                            return True
                        
                        elif response.status == 429:
                            # 레이트 제한
                            retry_after = int(response.headers.get('Retry-After', 60))
                            self.logger.warning(f"Telegram rate limited, retry after {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        
                        else:
                            # 기타 에러
                            error_text = await response.text()
                            self.logger.error(f"Telegram API error {response.status}: {error_text}")
                            
                except aiohttp.ClientError as e:
                    self.logger.warning(f"Telegram request failed (attempt {attempt + 1}): {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
            
            # 모든 재시도 실패
            self.notification_stats["failed"] += 1
            return False
            
        except Exception as e:
            self.logger.error(f"Notification sending failed: {e}")
            self.notification_stats["failed"] += 1
            return False
    
    def _cleanup_cache(self):
        """캐시 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.message_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.message_cache[key]
        
        # 캐시 크기 제한 (최대 1000개)
        if len(self.message_cache) > 1000:
            # 가장 오래된 항목들 제거
            sorted_items = sorted(self.message_cache.items(), key=lambda x: x[1])
            items_to_remove = sorted_items[:len(self.message_cache) - 800]
            for key, _ in items_to_remove:
                del self.message_cache[key]
    
    async def health_check(self) -> Dict[str, Any]:
        """봇 상태 확인"""
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self.config.bot_token}/getMe"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        self.last_health_check = datetime.now()
                        
                        return {
                            "status": "healthy",
                            "bot_info": bot_info,
                            "last_check": self.last_health_check.isoformat(),
                            "stats": self.notification_stats.copy()
                        }
                    else:
                        return {
                            "status": "error",
                            "error": data.get("description", "Unknown error")
                        }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # ========== 편의 메서드들 ==========
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]):
        """거래 알림"""
        side = trade_data.get("side", "UNKNOWN")
        symbol = trade_data.get("symbol", "UNKNOWN")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        pnl = trade_data.get("pnl", 0)
        
        if pnl > 0:
            level = NotificationLevel.INFO
            title = f"💰 거래 수익: {symbol} {side}"
        elif pnl < 0:
            level = NotificationLevel.WARNING
            title = f"📉 거래 손실: {symbol} {side}"
        else:
            level = NotificationLevel.INFO
            title = f"📊 거래 실행: {symbol} {side}"
        
        message = f"수량: {quantity:,.4f}\n가격: ${price:,.2f}"
        if pnl != 0:
            message += f"\nP&L: ${pnl:,.2f}"
        
        notification = NotificationMessage(
            level=level,
            type=NotificationType.TRADE,
            title=title,
            message=message,
            data=trade_data
        )
        
        return await self.send_notification(notification)
    
    async def send_signal_notification(self, signal_data: Dict[str, Any]):
        """시그널 알림"""
        action = signal_data.get("action", "UNKNOWN")
        symbol = signal_data.get("symbol", "UNKNOWN")
        confidence = signal_data.get("confidence", 0)
        
        title = f"📊 {action} 시그널: {symbol}"
        message = f"신뢰도: {confidence:.1%}"
        
        if confidence > 0.8:
            level = NotificationLevel.INFO
        elif confidence > 0.6:
            level = NotificationLevel.INFO
        else:
            level = NotificationLevel.DEBUG
        
        notification = NotificationMessage(
            level=level,
            type=NotificationType.SIGNAL,
            title=title,
            message=message,
            data=signal_data
        )
        
        return await self.send_notification(notification)
    
    async def send_risk_alert(self, risk_data: Dict[str, Any]):
        """리스크 알림"""
        alert_type = risk_data.get("type", "UNKNOWN")
        severity = risk_data.get("severity", "medium")
        message_text = risk_data.get("message", "Risk alert triggered")
        
        if severity == "critical":
            level = NotificationLevel.CRITICAL
            title = f"🚨 긴급 리스크: {alert_type}"
        elif severity == "high":
            level = NotificationLevel.ERROR
            title = f"❌ 고위험: {alert_type}"
        else:
            level = NotificationLevel.WARNING
            title = f"⚠️ 리스크 경고: {alert_type}"
        
        notification = NotificationMessage(
            level=level,
            type=NotificationType.RISK,
            title=title,
            message=message_text,
            data=risk_data
        )
        
        return await self.send_notification(notification)
    
    async def send_system_alert(self, system_data: Dict[str, Any]):
        """시스템 알림"""
        event_type = system_data.get("type", "UNKNOWN")
        status = system_data.get("status", "unknown")
        message_text = system_data.get("message", "System event occurred")
        
        if status in ["error", "failed", "critical"]:
            level = NotificationLevel.ERROR
            title = f"❌ 시스템 오류: {event_type}"
        elif status in ["warning", "degraded"]:
            level = NotificationLevel.WARNING
            title = f"⚠️ 시스템 경고: {event_type}"
        else:
            level = NotificationLevel.INFO
            title = f"ℹ️ 시스템 정보: {event_type}"
        
        notification = NotificationMessage(
            level=level,
            type=NotificationType.SYSTEM,
            title=title,
            message=message_text,
            data=system_data
        )
        
        return await self.send_notification(notification)
    
    async def send_health_status(self, health_data: Dict[str, Any]):
        """헬스 상태 알림"""
        overall_status = health_data.get("status", "unknown")
        score = health_data.get("score", 0)
        issues = health_data.get("issues", [])
        
        if overall_status == "healthy" and score >= 90:
            level = NotificationLevel.INFO
            title = f"✅ 시스템 정상: {score}/100"
        elif score >= 70:
            level = NotificationLevel.WARNING
            title = f"⚠️ 시스템 주의: {score}/100"
        else:
            level = NotificationLevel.ERROR
            title = f"❌ 시스템 문제: {score}/100"
        
        message = f"상태: {overall_status}"
        if issues:
            message += f"\n문제점: {len(issues)}개"
            for issue in issues[:3]:  # 최대 3개만 표시
                message += f"\n• {issue}"
        
        notification = NotificationMessage(
            level=level,
            type=NotificationType.HEALTH,
            title=title,
            message=message,
            data=health_data
        )
        
        return await self.send_notification(notification)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            "notification_stats": self.notification_stats.copy(),
            "rate_limit": {
                "recent_messages": len(self.message_history),
                "limit": self.config.rate_limit_messages,
                "window_seconds": self.config.rate_limit_window
            },
            "cache": {
                "cached_messages": len(self.message_cache),
                "ttl_seconds": self.cache_ttl
            },
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }

class NotificationManager:
    """알림 관리자 - 여러 채널 통합"""
    
    def __init__(self):
        self.notifiers: Dict[str, TelegramNotifier] = {}
        self.logger = logging.getLogger(__name__)
        
        # 환경변수에서 기본 설정 로드
        self._load_default_config()
    
    def _load_default_config(self):
        """환경변수에서 기본 설정 로드"""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if bot_token and chat_id:
            config = TelegramConfig(
                bot_token=bot_token,
                chat_id=chat_id,
                enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
            )
            
            self.add_notifier("default", config)
            self.logger.info("Default Telegram notifier configured")
        else:
            self.logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in environment")
    
    def add_notifier(self, name: str, config: TelegramConfig):
        """알림기 추가"""
        self.notifiers[name] = TelegramNotifier(config)
        self.logger.info(f"Added notifier: {name}")
    
    def get_notifier(self, name: str = "default") -> Optional[TelegramNotifier]:
        """알림기 조회"""
        return self.notifiers.get(name)
    
    async def broadcast(self, notification: NotificationMessage, notifiers: List[str] = None):
        """여러 알림기로 방송"""
        if notifiers is None:
            notifiers = list(self.notifiers.keys())
        
        results = {}
        for notifier_name in notifiers:
            if notifier_name in self.notifiers:
                try:
                    result = await self.notifiers[notifier_name].send_notification(notification)
                    results[notifier_name] = result
                except Exception as e:
                    self.logger.error(f"Broadcast failed for {notifier_name}: {e}")
                    results[notifier_name] = False
            else:
                self.logger.warning(f"Notifier not found: {notifier_name}")
                results[notifier_name] = False
        
        return results
    
    async def health_check_all(self) -> Dict[str, Any]:
        """모든 알림기 헬스체크"""
        results = {}
        for name, notifier in self.notifiers.items():
            try:
                results[name] = await notifier.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        return results
    
    async def close_all(self):
        """모든 알림기 정리"""
        for notifier in self.notifiers.values():
            await notifier.close()

# 전역 인스턴스
notification_manager = NotificationManager()

def get_notifier(name: str = "default") -> Optional[TelegramNotifier]:
    """전역 알림기 조회"""
    return notification_manager.get_notifier(name)

async def send_notification(notification: NotificationMessage, notifier_name: str = "default") -> bool:
    """편의 함수: 알림 전송"""
    notifier = get_notifier(notifier_name)
    if notifier:
        return await notifier.send_notification(notification)
    return False

# 편의 함수들
async def send_trade_alert(trade_data: Dict[str, Any]) -> bool:
    """거래 알림 전송"""
    notifier = get_notifier()
    if notifier:
        return await notifier.send_trade_notification(trade_data)
    return False

async def send_risk_alert(risk_data: Dict[str, Any]) -> bool:
    """리스크 알림 전송"""
    notifier = get_notifier()
    if notifier:
        return await notifier.send_risk_alert(risk_data)
    return False

async def send_system_alert(system_data: Dict[str, Any]) -> bool:
    """시스템 알림 전송"""
    notifier = get_notifier()
    if notifier:
        return await notifier.send_system_alert(system_data)
    return False

# 테스트 및 예제
async def main():
    """테스트 메인"""
    print("🚀 AuroraQ Telegram Notification System")
    print("=" * 50)
    
    # 설정 확인
    notifier = get_notifier()
    if not notifier:
        print("❌ No notifier configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return
    
    # 헬스체크
    print("\n🏥 Health check...")
    health = await notifier.health_check()
    print(f"Status: {health.get('status')}")
    
    if health.get("status") == "healthy":
        bot_info = health.get("bot_info", {})
        print(f"Bot: @{bot_info.get('username', 'unknown')}")
        
        # 테스트 알림들
        test_notifications = [
            # 시스템 시작 알림
            NotificationMessage(
                level=NotificationLevel.INFO,
                type=NotificationType.SYSTEM,
                title="시스템 시작",
                message="AuroraQ 거래 시스템이 시작되었습니다.",
                data={"version": "1.0.0", "mode": "test"}
            ),
            
            # 거래 알림
            NotificationMessage(
                level=NotificationLevel.INFO,
                type=NotificationType.TRADE,
                title="거래 실행",
                message="BTCUSDT 매수 주문이 체결되었습니다.",
                data={
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": 0.001,
                    "price": 45000,
                    "pnl": 50.0
                }
            ),
            
            # 리스크 경고
            NotificationMessage(
                level=NotificationLevel.WARNING,
                type=NotificationType.RISK,
                title="리스크 경고",
                message="포트폴리오 드로우다운이 10%를 초과했습니다.",
                data={
                    "type": "drawdown",
                    "severity": "medium",
                    "current_drawdown": 0.12,
                    "max_allowed": 0.15
                }
            )
        ]
        
        # 테스트 알림 전송
        print(f"\n📨 Sending {len(test_notifications)} test notifications...")
        for i, notification in enumerate(test_notifications):
            success = await notifier.send_notification(notification)
            status = "✅" if success else "❌"
            print(f"{status} Notification {i+1}: {notification.title}")
            
            # 레이트 제한 방지를 위한 지연
            if i < len(test_notifications) - 1:
                await asyncio.sleep(2)
        
        # 통계 조회
        print(f"\n📊 Notification stats:")
        stats = notifier.get_stats()
        print(json.dumps(stats, indent=2, default=str))
    
    else:
        print(f"❌ Health check failed: {health.get('error')}")
    
    # 정리
    await notifier.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Notification system stopped by user")
    except Exception as e:
        print(f"\n❌ Notification system failed: {e}")
        import traceback
        traceback.print_exc()