import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기 (AuroraQ 루트에 .env 파일 필요)
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Binance API 클라이언트 생성
client = Client(API_KEY, API_SECRET)

def download_5min_klines(symbol="BTCUSDT", limit=1500, save_path="data/price/"):
    """
    바이낸스에서 지정한 심볼의 5분봉 캔들 데이터를 다운로드하고 CSV로 저장.
    CSV는 timestamp 컬럼을 포함하여 AuroraQ 백테스트 루프와 호환되도록 저장.
    파일명은 backtest_data.csv로 고정.
    """
    print(f"[INFO] {symbol} 5분봉 데이터 {limit}개 다운로드 중...")

    # 5분봉 캔들 데이터 요청
    klines = client.get_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_5MINUTE,
        limit=limit
    )

    # DataFrame 변환
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # 필요한 컬럼만 추출 및 timestamp 컬럼으로 변경
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.drop(columns=['open_time'])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

    # 컬럼 순서 조정
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # 저장 경로 생성 및 CSV 저장
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "backtest_data.csv")
    df.to_csv(file_path, index=False)
    print(f"[INFO] {symbol} 5분봉 데이터 저장 완료: {file_path} ({len(df)} rows)")

if __name__ == "__main__":
    # BTCUSDT 1500개 캔들 다운로드 (기본 설정)
    download_5min_klines(symbol="BTCUSDT", limit=1500)

    # 필요하다면 다른 심볼도 추가 가능 (주석 해제해서 사용)
    # download_5min_klines(symbol="ETHUSDT", limit=1500)
    # download_5min_klines(symbol="SOLUSDT", limit=1500)
