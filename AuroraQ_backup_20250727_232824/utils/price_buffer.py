# 📁 utils/price_buffer.py

def get_latest_price():
    return {
        "close": [100 + i for i in range(30)],
        "sentiment": 0.6,
        "high": [105 + i for i in range(30)],
        "low": [95 + i for i in range(30)],
        "open": [98 + i for i in range(30)],
        "volume": [1000 for _ in range(30)],
        "timestamp": "2025-06-11 17:50:00"
    }
