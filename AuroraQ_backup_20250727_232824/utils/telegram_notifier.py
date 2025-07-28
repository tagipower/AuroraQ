# utils/telegram_notifier.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not token or not chat_id:
        print("[TELEGRAM] 비활성화됨. 메시지 전송 안 함.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"[텔레그램 오류] 응답 코드: {response.status_code}, 응답 내용: {response.text}")
        else:
            print("[텔레그램] 메시지 전송 완료")
    except Exception as e:
        print(f"[텔레그램 예외] {e}")