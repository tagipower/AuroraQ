#!/usr/bin/env python3
"""
모바일 알림 시스템
Telegram Bot API를 사용한 실시간 알림
"""

import asyncio
import requests
import json
from datetime import datetime

class MobileNotificationSystem:
    def __init__(self, telegram_bot_token: str, chat_id: str):
        self.bot_token = telegram_bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{telegram_bot_token}"
        
    def send_telegram_message(self, message: str):
        """텔레그램 메시지 전송"""
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram 전송 실패: {e}")
            return False
    
    def format_trading_alert(self, trade_data: dict) -> str:
        """거래 알림 포맷"""
        emoji = "🟢" if trade_data['pnl'] > 0 else "🔴"
        
        message = f"""
{emoji} <b>AuroraQ 거래 알림</b>

💰 <b>P&L:</b> ${trade_data['pnl']:.2f}
📊 <b>전략:</b> {trade_data['strategy']}
🎯 <b>심볼:</b> {trade_data['symbol']}
📈 <b>포지션:</b> {trade_data['side']}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return message
    
    def format_system_alert(self, alert_type: str, message: str) -> str:
        """시스템 알림 포맷"""
        emoji_map = {
            'error': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        emoji = emoji_map.get(alert_type, 'ℹ️')
        
        return f"""
{emoji} <b>AuroraQ 시스템</b>

{message}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 사용 예시
if __name__ == "__main__":
    # Telegram Bot 설정
    BOT_TOKEN = "YOUR_BOT_TOKEN"  # @BotFather에서 생성
    CHAT_ID = "YOUR_CHAT_ID"      # 본인의 텔레그램 ID
    
    notifier = MobileNotificationSystem(BOT_TOKEN, CHAT_ID)
    
    # 거래 알림 예시
    trade_alert = {
        'pnl': 15.75,
        'strategy': 'PPOStrategy',
        'symbol': 'BTCUSDT',
        'side': 'LONG'
    }
    
    message = notifier.format_trading_alert(trade_alert)
    notifier.send_telegram_message(message)