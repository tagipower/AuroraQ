#!/usr/bin/env python3
"""
Telegram Notification System for AuroraQ Sentiment Service
텔레그램 알림 시스템
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import structlog

logger = structlog.get_logger(__name__)

class NotificationLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRADING = "trading"
    EVENT = "event"

class NotificationChannel(Enum):
    """알림 채널"""
    GENERAL = "general"        # 일반 알림
    TRADING = "trading"        # 매매 신호
    EVENTS = "events"          # 빅 이벤트
    SYSTEM = "system"          # 시스템 상태
    DEBUG = "debug"            # 디버그 정보

@dataclass
class TelegramMessage:
    """텔레그램 메시지"""
    text: str
    level: NotificationLevel
    channel: NotificationChannel
    timestamp: datetime
    symbol: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    parse_mode: str = "HTML"
    disable_notification: bool = False

class TelegramNotifier:
    """텔레그램 알림 발송기"""
    
    def __init__(self,
                 bot_token: str,
                 chat_configs: Dict[str, Dict[str, Any]],
                 rate_limit_seconds: int = 1):
        """
        초기화
        
        Args:
            bot_token: 텔레그램 봇 토큰
            chat_configs: 채널별 채팅방 설정
            rate_limit_seconds: API 호출 제한 (초)
        """
        self.bot_token = bot_token
        self.chat_configs = chat_configs
        self.rate_limit_seconds = rate_limit_seconds
        
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 메시지 대기열 및 중복 방지
        self.message_queue = asyncio.Queue()
        self.sent_messages = {}  # 중복 방지용
        self.last_api_call = 0
        
        # 통계
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "by_level": {level.value: 0 for level in NotificationLevel},
            "by_channel": {channel.value: 0 for channel in NotificationChannel},
            "last_sent": None,
            "errors": []
        }
        
        self._running = False
        self._worker_task = None

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()

    async def start(self):
        """알림 서비스 시작"""
        if self._running:
            return
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        # 봇 정보 확인
        try:
            bot_info = await self._api_call("getMe")
            logger.info("Telegram bot started", 
                       bot_name=bot_info.get("username"),
                       bot_id=bot_info.get("id"))
        except Exception as e:
            logger.error("Failed to start Telegram bot", error=str(e))
            raise
        
        # 워커 태스크 시작
        self._running = True
        self._worker_task = asyncio.create_task(self._message_worker())
        
        logger.info("Telegram notifier started")

    async def stop(self):
        """알림 서비스 중지"""
        if not self._running:
            return
        
        self._running = False
        
        # 워커 태스크 종료 대기
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # 세션 종료
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Telegram notifier stopped")

    async def _api_call(self, method: str, **kwargs) -> Dict[str, Any]:
        """텔레그램 API 호출"""
        url = f"{self.api_url}/{method}"
        
        # Rate limiting
        now = asyncio.get_event_loop().time()
        if now - self.last_api_call < self.rate_limit_seconds:
            await asyncio.sleep(self.rate_limit_seconds - (now - self.last_api_call))
        
        try:
            async with self.session.post(url, json=kwargs) as response:
                self.last_api_call = asyncio.get_event_loop().time()
                
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        return result.get("result", {})
                    else:
                        raise Exception(f"API error: {result.get('description')}")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error("Telegram API call failed", 
                        method=method, 
                        error=str(e))
            raise

    async def _message_worker(self):
        """메시지 발송 워커"""
        while self._running:
            try:
                # 큐에서 메시지 가져오기 (타임아웃 5초)
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 메시지 발송
                await self._send_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Message worker error", error=str(e))
                await asyncio.sleep(1)

    async def _send_message(self, message: TelegramMessage):
        """실제 메시지 발송"""
        try:
            # 채널 설정 확인
            channel_config = self.chat_configs.get(message.channel.value)
            if not channel_config:
                logger.warning("No config for channel", channel=message.channel.value)
                return
            
            chat_id = channel_config.get("chat_id")
            if not chat_id:
                logger.warning("No chat_id for channel", channel=message.channel.value)
                return
            
            # 중복 메시지 체크 (같은 채널, 같은 텍스트, 5분 이내)
            message_key = f"{chat_id}:{hash(message.text)}"
            now = datetime.now()
            
            if message_key in self.sent_messages:
                last_sent = self.sent_messages[message_key]
                if now - last_sent < timedelta(minutes=5):
                    logger.debug("Duplicate message skipped", 
                               channel=message.channel.value)
                    return
            
            # 메시지 발송
            await self._api_call(
                "sendMessage",
                chat_id=chat_id,
                text=message.text,
                parse_mode=message.parse_mode,
                disable_notification=message.disable_notification
            )
            
            # 통계 업데이트
            self.sent_messages[message_key] = now
            self.stats["total_sent"] += 1
            self.stats["by_level"][message.level.value] += 1
            self.stats["by_channel"][message.channel.value] += 1
            self.stats["last_sent"] = now.isoformat()
            
            logger.info("Telegram message sent", 
                       channel=message.channel.value,
                       level=message.level.value,
                       chat_id=chat_id)
            
        except Exception as e:
            self.stats["total_failed"] += 1
            self.stats["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "channel": message.channel.value
            })
            
            # 에러 로그는 최대 100개만 보관
            if len(self.stats["errors"]) > 100:
                self.stats["errors"] = self.stats["errors"][-50:]
            
            logger.error("Failed to send Telegram message", 
                        channel=message.channel.value,
                        error=str(e))

    async def send_notification(self,
                              text: str,
                              level: NotificationLevel = NotificationLevel.INFO,
                              channel: NotificationChannel = NotificationChannel.GENERAL,
                              symbol: Optional[str] = None,
                              data: Optional[Dict[str, Any]] = None,
                              disable_notification: bool = False):
        """알림 발송"""
        if not self._running:
            logger.warning("Telegram notifier not running")
            return
        
        message = TelegramMessage(
            text=text,
            level=level,
            channel=channel,
            timestamp=datetime.now(),
            symbol=symbol,
            data=data,
            disable_notification=disable_notification
        )
        
        await self.message_queue.put(message)

    async def send_trading_signal(self,
                                signal: Dict[str, Any],
                                trading_mode: str = "LIVE"):
        """매매 신호 알림"""
        
        # 신호 세기에 따른 이모지
        strength_emoji = {
            "very_strong": "🚀",
            "strong": "📈",
            "moderate": "📊",
            "weak": "📉",
            "very_weak": "⚠️"
        }
        
        # 매매 방향에 따른 이모지
        direction_emoji = {
            "buy": "🟢",
            "sell": "🔴", 
            "hold": "🟡"
        }
        
        symbol = signal.get("symbol", "CRYPTO")
        direction = signal.get("direction", "hold")
        strength = signal.get("strength", "moderate")
        confidence = signal.get("confidence", 0)
        score = signal.get("sentiment_score", 0)
        
        text = f"""
{direction_emoji.get(direction, "📊")} <b>매매 신호 [{trading_mode}]</b>

🎯 <b>종목:</b> {symbol}
📊 <b>방향:</b> {direction.upper()}
{strength_emoji.get(strength, "📊")} <b>신호 세기:</b> {strength}
🎲 <b>신뢰도:</b> {confidence:.1%}
📈 <b>감정 점수:</b> {score:+.3f}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        # 강한 신호는 일반 알림 ON
        disable_notification = strength in ["weak", "very_weak"]
        
        await self.send_notification(
            text=text,
            level=NotificationLevel.TRADING,
            channel=NotificationChannel.TRADING,
            symbol=symbol,
            data=signal,
            disable_notification=disable_notification
        )

    async def send_big_event(self, event: Dict[str, Any]):
        """빅 이벤트 알림"""
        
        event_emoji = {
            "FOMC": "🏦",
            "CPI": "📊",
            "ETF_APPROVAL": "🚀",
            "REGULATORY": "⚖️",
            "INSTITUTIONAL": "🏢",
            "TECHNICAL": "📈",
            "MARKET_SENTIMENT": "💭",
            "SOCIAL_TREND": "📱",
            "WHALE_MOVEMENT": "🐋",
            "FORK_UPGRADE": "🔧",
            "PARTNERSHIP": "🤝"
        }
        
        event_type = event.get("event_type", "UNKNOWN")
        symbol = event.get("symbol", "CRYPTO")
        impact_score = event.get("impact_score", 0)
        confidence = event.get("confidence", 0)
        description = event.get("description", "")
        
        text = f"""
{event_emoji.get(event_type, "📢")} <b>빅 이벤트 감지</b>

🏷️ <b>유형:</b> {event_type}
🎯 <b>종목:</b> {symbol}
💥 <b>영향도:</b> {impact_score:.2f}
🎲 <b>신뢰도:</b> {confidence:.1%}

📝 <b>설명:</b>
{description}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        await self.send_notification(
            text=text,
            level=NotificationLevel.EVENT,
            channel=NotificationChannel.EVENTS,
            symbol=symbol,
            data=event
        )

    async def send_system_alert(self,
                              message: str,
                              level: NotificationLevel = NotificationLevel.WARNING,
                              service: str = "sentiment-service"):
        """시스템 알림"""
        
        level_emoji = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨"
        }
        
        text = f"""
{level_emoji.get(level, "📢")} <b>시스템 알림 [{level.value.upper()}]</b>

🖥️ <b>서비스:</b> {service}
📝 <b>메시지:</b> {message}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        # CRITICAL과 ERROR는 알림 ON
        disable_notification = level in [NotificationLevel.INFO, NotificationLevel.WARNING]
        
        await self.send_notification(
            text=text,
            level=level,
            channel=NotificationChannel.SYSTEM,
            disable_notification=disable_notification
        )

    async def send_performance_report(self, stats: Dict[str, Any]):
        """성능 보고서 알림"""
        
        text = f"""
📊 <b>성능 보고서</b>

🎯 <b>API 응답 시간:</b> {stats.get('avg_response_time', 0):.3f}초
📈 <b>처리된 요청:</b> {stats.get('total_requests', 0):,}개
✅ <b>성공률:</b> {stats.get('success_rate', 0):.1%}

💾 <b>메모리 사용량:</b> {stats.get('memory_usage', 0):.1f}%
🖥️ <b>CPU 사용률:</b> {stats.get('cpu_usage', 0):.1f}%
💿 <b>디스크 사용량:</b> {stats.get('disk_usage', 0):.1f}%

🤖 <b>배치 작업:</b>
• 뉴스 수집: {stats.get('news_collected', 0):,}개
• FinBERT 분석: {stats.get('finbert_processed', 0):,}개
• 이벤트 감지: {stats.get('events_detected', 0):,}개

⏰ <b>생성 시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        await self.send_notification(
            text=text,
            level=NotificationLevel.INFO,
            channel=NotificationChannel.SYSTEM,
            disable_notification=True  # 정기 보고서는 무음
        )

    async def send_startup_message(self):
        """서비스 시작 알림"""
        text = f"""
🚀 <b>AuroraQ Sentiment Service 시작</b>

✅ 서비스가 성공적으로 시작되었습니다.
🌐 VPS: 109.123.239.30
⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 활성화된 기능:
• 실시간 감정 분석
• FinBERT 배치 처리 
• 빅 이벤트 감지
• 매매 신호 생성
• 텔레그램 알림

🎯 준비 완료!
        """.strip()
        
        await self.send_notification(
            text=text,
            level=NotificationLevel.INFO,
            channel=NotificationChannel.SYSTEM
        )

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            "running": self._running,
            "queue_size": self.message_queue.qsize(),
            "sent_messages_cache": len(self.sent_messages)
        }

    async def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            await self._api_call("getMe")
            return True
        except Exception as e:
            logger.error("Telegram connection test failed", error=str(e))
            return False

    async def cleanup_old_messages(self, hours: int = 24):
        """오래된 메시지 기록 정리"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # 중복 방지 캐시 정리
        to_remove = []
        for key, timestamp in self.sent_messages.items():
            if timestamp < cutoff:
                to_remove.append(key)
        
        for key in to_remove:
            del self.sent_messages[key]
        
        logger.info("Cleaned up old message cache", 
                   removed=len(to_remove),
                   remaining=len(self.sent_messages))


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    async def test_telegram_notifier():
        """텔레그램 알림 테스트"""
        
        # 테스트 설정
        bot_token = "YOUR_BOT_TOKEN"
        chat_configs = {
            "general": {"chat_id": "YOUR_CHAT_ID"},
            "trading": {"chat_id": "YOUR_CHAT_ID"},
            "events": {"chat_id": "YOUR_CHAT_ID"},
            "system": {"chat_id": "YOUR_CHAT_ID"}
        }
        
        async with TelegramNotifier(bot_token, chat_configs) as notifier:
            # 연결 테스트
            if await notifier.test_connection():
                print("✅ Telegram connection successful")
                
                # 테스트 메시지들
                await notifier.send_startup_message()
                
                # 매매 신호 테스트
                test_signal = {
                    "symbol": "BTC",
                    "direction": "buy",
                    "strength": "strong",
                    "confidence": 0.85,
                    "sentiment_score": 0.642
                }
                await notifier.send_trading_signal(test_signal, "LIVE")
                
                # 빅 이벤트 테스트
                test_event = {
                    "event_type": "ETF_APPROVAL",
                    "symbol": "BTC",
                    "impact_score": 8.5,
                    "confidence": 0.92,
                    "description": "SEC approves first Bitcoin ETF application"
                }
                await notifier.send_big_event(test_event)
                
                # 시스템 알림 테스트
                await notifier.send_system_alert(
                    "서비스 정상 동작 중", 
                    NotificationLevel.INFO
                )
                
                # 통계 출력
                stats = notifier.get_stats()
                print(f"📊 Statistics: {stats}")
                
            else:
                print("❌ Telegram connection failed")
    
    # 테스트 실행
    asyncio.run(test_telegram_notifier())