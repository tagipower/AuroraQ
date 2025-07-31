#!/usr/bin/env python3
"""텔레그램 통합 모듈"""

import asyncio
import aiohttp
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

class TelegramIntegration:
    """텔레그램 봇 통합 서비스"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self.logger = logging.getLogger(__name__)
        
        if not self.bot_token:
            self.logger.warning("Telegram bot token not configured")
        if not self.chat_id:
            self.logger.warning("Telegram chat ID not configured")
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """메시지 전송"""
        if not self.bot_token or not self.chat_id:
            return {
                "success": False,
                "error": "Bot token or chat ID not configured"
            }
        
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            return {
                                "success": True,
                                "message_id": result["result"]["message_id"],
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            return {
                                "success": False,
                                "error": result.get("description", "Unknown error")
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}"
                        }
        
        except Exception as e:
            self.logger.error(f"Telegram send error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_trading_alert(self, symbol: str, action: str, 
                               sentiment_score: float, confidence: float,
                               details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """트레이딩 알림 전송"""
        
        # 감정 이모지 선택
        if sentiment_score > 0.3:
            emoji = "🟢"
            sentiment_text = "긍정적"
        elif sentiment_score < -0.3:
            emoji = "🔴"
            sentiment_text = "부정적"
        else:
            emoji = "🟡"
            sentiment_text = "중립적"
        
        # 메시지 구성
        message = f"""
{emoji} <b>AuroraQ 트레이딩 알림</b>

📊 <b>종목:</b> {symbol}
🎯 <b>권장 액션:</b> {action}
💭 <b>시장 감정:</b> {sentiment_text} ({sentiment_score:.3f})
🎲 <b>신뢰도:</b> {confidence:.1%}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 추가 세부사항
        if details:
            message += "\n📋 <b>추가 정보:</b>\n"
            for key, value in details.items():
                message += f"• {key}: {value}\n"
        
        return await self.send_message(message)
    
    async def send_news_alert(self, title: str, source: str, 
                            sentiment_score: float, url: Optional[str] = None) -> Dict[str, Any]:
        """뉴스 알림 전송"""
        
        # 감정 이모지 선택
        if sentiment_score > 0.3:
            emoji = "📈"
        elif sentiment_score < -0.3:
            emoji = "📉"
        else:
            emoji = "📰"
        
        # 메시지 구성
        message = f"""
{emoji} <b>AuroraQ 뉴스 알림</b>

📰 <b>제목:</b> {title}
🏢 <b>출처:</b> {source}
💭 <b>감정 점수:</b> {sentiment_score:.3f}

⏰ <b>시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if url:
            message += f"\n🔗 <b>링크:</b> {url}"
        
        return await self.send_message(message)
    
    async def send_system_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """시스템 상태 알림 전송"""
        
        status_emoji = "✅" if status.get("healthy", False) else "❌"
        
        message = f"""
{status_emoji} <b>AuroraQ 시스템 상태</b>

🖥️ <b>전체 상태:</b> {status.get('status', 'unknown')}
🔄 <b>활성 수집기:</b> {status.get('active_collectors', 0)}/{status.get('total_collectors', 0)}
📊 <b>수집된 기사:</b> {status.get('total_articles', 0)}개

⏰ <b>확인 시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_message(message)
    
    async def health_check(self) -> Dict[str, Any]:
        """텔레그램 연결 상태 확인"""
        if not self.bot_token:
            return {
                "status": "disabled",
                "reason": "Bot token not configured"
            }
        
        try:
            url = f"{self.base_url}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            bot_info = result["result"]
                            return {
                                "status": "healthy",
                                "bot_name": bot_info.get("first_name"),
                                "bot_username": bot_info.get("username"),
                                "chat_id_configured": bool(self.chat_id)
                            }
                        else:
                            return {
                                "status": "error",
                                "error": result.get("description", "Unknown error")
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

# 전역 인스턴스
telegram_integration = TelegramIntegration()

# 편의 함수
async def send_telegram_message(message: str) -> Dict[str, Any]:
    """텔레그램 메시지 전송 편의 함수"""
    return await telegram_integration.send_message(message)

async def send_telegram_alert(symbol: str, action: str, sentiment_score: float, 
                            confidence: float, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """텔레그램 트레이딩 알림 편의 함수"""
    return await telegram_integration.send_trading_alert(symbol, action, sentiment_score, confidence, details)