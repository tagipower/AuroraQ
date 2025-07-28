# 📁 utils/indicators.py

import numpy as np

# ✅ 지수 이동 평균 (EMA)
def EMA(data, period):
    ema = [np.mean(data[:period])]
    k = 2 / (period + 1)
    for price in data[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return np.array(ema[-len(data):])


# ✅ 평균 방향성 지수 (ADX)
def ADX(price_data, period=14):
    high = np.array(price_data["high"])
    low = np.array(price_data["low"])
    close = np.array(price_data["close"])

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        plus_dm = high[i] - high[i - 1] if high[i] - high[i - 1] > low[i - 1] - low[i] else 0
        minus_dm = low[i - 1] - low[i] if low[i - 1] - low[i] > high[i] - high[i - 1] else 0
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    tr_smooth = np.convolve(tr_list, np.ones(period)/period, mode='valid')
    plus_di = 100 * np.convolve(plus_dm_list, np.ones(period)/period, mode='valid') / tr_smooth
    minus_di = 100 * np.convolve(minus_dm_list, np.ones(period)/period, mode='valid') / tr_smooth
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.convolve(dx, np.ones(period)/period, mode='valid')

    return np.concatenate([np.full(len(close) - len(adx), np.nan), adx])


# ✅ 볼린저밴드

def BollingerBands(data, window=20):
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    std = np.array([np.std(data[i-window:i]) for i in range(window, len(data)+1)])
    upper = ma + 2 * std
    lower = ma - 2 * std
    padded = len(data) - len(ma)
    return (
        np.concatenate([np.full(padded, np.nan), lower]),
        np.concatenate([np.full(padded, np.nan), ma]),
        np.concatenate([np.full(padded, np.nan), upper]),
    )
