#!/usr/bin/env python3
"""
데이터 관리자 - 가격 데이터 로딩, 검증, 전처리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import requests
import yfinance as yf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DataManager:
    """백테스트 데이터 관리자"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # 지원하는 데이터 소스
        self.supported_sources = ['yahoo', 'binance', 'csv', 'local']
        
        # 캐시된 데이터
        self.cached_data = {}
        
    def load_data(self,
                 symbol: str,
                 source: str = 'yahoo',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 interval: str = '1d',
                 force_reload: bool = False) -> pd.DataFrame:
        """
        데이터 로딩
        
        Args:
            symbol: 심볼 (예: 'BTC-USD', 'AAPL')
            source: 데이터 소스
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            interval: 시간 간격
            force_reload: 강제 재로딩
            
        Returns:
            OHLCV 데이터프레임
        """
        cache_key = f"{symbol}_{source}_{start_date}_{end_date}_{interval}"
        
        # 캐시 확인
        if not force_reload and cache_key in self.cached_data:
            return self.cached_data[cache_key].copy()
        
        # 소스별 데이터 로딩
        if source == 'yahoo':
            data = self._load_yahoo_data(symbol, start_date, end_date, interval)
        elif source == 'binance':
            data = self._load_binance_data(symbol, start_date, end_date, interval)
        elif source == 'csv':
            data = self._load_csv_data(symbol)
        elif source == 'local':
            data = self._load_local_data(symbol)
        else:
            raise ValueError(f"지원하지 않는 데이터 소스: {source}")
        
        # 데이터 검증 및 전처리
        data = self._validate_and_clean_data(data)
        
        # 캐시 저장
        self.cached_data[cache_key] = data.copy()
        
        return data
    
    def _load_yahoo_data(self,
                        symbol: str,
                        start_date: Optional[str],
                        end_date: Optional[str],
                        interval: str) -> pd.DataFrame:
        """Yahoo Finance 데이터 로딩"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 기간 설정
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 데이터 다운로드
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                raise ValueError(f"Yahoo Finance에서 {symbol} 데이터를 찾을 수 없습니다")
            
            # 컬럼명 표준화
            data.columns = data.columns.str.lower()
            data = data.rename(columns={
                'adj close': 'close'  # Adjusted Close 사용
            })
            
            # 필요한 컬럼만 선택
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            data = data[required_columns]
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Yahoo Finance 데이터 로딩 실패: {e}")
    
    def _load_binance_data(self,
                          symbol: str,
                          start_date: Optional[str],
                          end_date: Optional[str],
                          interval: str) -> pd.DataFrame:
        """Binance API 데이터 로딩"""
        try:
            # Binance API 엔드포인트
            base_url = "https://api.binance.com/api/v3/klines"
            
            # 간격 매핑
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            
            if interval not in interval_map:
                raise ValueError(f"지원하지 않는 간격: {interval}")
            
            # 파라미터 설정
            params = {
                'symbol': symbol.upper().replace('-', ''),
                'interval': interval_map[interval],
                'limit': 1000
            }
            
            if start_date:
                start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
                params['startTime'] = start_timestamp
            
            if end_date:
                end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
                params['endTime'] = end_timestamp
            
            # API 호출
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            klines = response.json()
            
            if not klines:
                raise ValueError(f"Binance에서 {symbol} 데이터를 찾을 수 없습니다")
            
            # DataFrame 생성
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ]
            
            data = pd.DataFrame(klines, columns=columns)
            
            # 데이터 타입 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col])
            
            # 타임스탬프 변환
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            # 필요한 컬럼만 선택
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Binance 데이터 로딩 실패: {e}")
    
    def _load_csv_data(self, file_path: str) -> pd.DataFrame:
        """CSV 파일 데이터 로딩"""
        full_path = self.data_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {full_path}")
        
        try:
            # CSV 읽기 (여러 형식 지원)
            data = pd.read_csv(full_path)
            
            # 첫 번째 컬럼이 날짜인 경우 인덱스로 설정
            if data.columns[0].lower() in ['date', 'timestamp', 'time']:
                data.set_index(data.columns[0], inplace=True)
                data.index = pd.to_datetime(data.index)
            
            # 컬럼명 표준화
            column_mapping = {
                'Date': 'date', 'Time': 'timestamp', 'Timestamp': 'timestamp',
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                'Volume': 'volume', 'Adj Close': 'close'
            }
            
            data.rename(columns=column_mapping, inplace=True)
            data.columns = data.columns.str.lower()
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"CSV 데이터 로딩 실패: {e}")
    
    def _load_local_data(self, symbol: str) -> pd.DataFrame:
        """로컬 저장된 데이터 로딩"""
        file_patterns = [
            f"{symbol}.csv",
            f"{symbol}.pkl",
            f"{symbol}_data.csv"
        ]
        
        for pattern in file_patterns:
            file_path = self.data_path / pattern
            if file_path.exists():
                if pattern.endswith('.csv'):
                    return self._load_csv_data(pattern)
                elif pattern.endswith('.pkl'):
                    return pd.read_pickle(file_path)
        
        raise FileNotFoundError(f"로컬에서 {symbol} 데이터를 찾을 수 없습니다")
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 검증 및 정제"""
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        # 데이터 타입 변환
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 결측값 처리
        data = data.dropna()
        
        # 음수 가격 제거
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # OHLC 논리 검증
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            print(f"경고: {invalid_ohlc.sum()}개의 잘못된 OHLC 데이터를 제거했습니다")
            data = data[~invalid_ohlc]
        
        # 중복 인덱스 제거
        data = data[~data.index.duplicated(keep='first')]
        
        # 정렬
        data = data.sort_index()
        
        return data
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """데이터 저장"""
        file_path = self.data_path / filename
        
        if format == 'csv':
            data.to_csv(file_path)
        elif format == 'pkl':
            data.to_pickle(file_path)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
        
        print(f"데이터가 저장되었습니다: {file_path}")
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """데이터 정보 요약"""
        if data.empty:
            return {"error": "데이터가 비어있습니다"}
        
        info = {
            'shape': data.shape,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'duration_days': (data.index[-1] - data.index[0]).days,
            'missing_values': data.isnull().sum().to_dict(),
            'price_range': {
                'min': data['close'].min(),
                'max': data['close'].max(),
                'avg': data['close'].mean()
            },
            'volume_stats': {
                'avg_volume': data['volume'].mean(),
                'max_volume': data['volume'].max(),
                'min_volume': data['volume'].min()
            }
        }
        
        return info
    
    def split_data(self,
                  data: pd.DataFrame,
                  train_ratio: float = 0.8,
                  validation_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 훈련/검증/테스트로 분할"""
        
        total_len = len(data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * validation_ratio)
        
        train_data = data.iloc[:train_len]
        val_data = data.iloc[train_len:train_len + val_len]
        test_data = data.iloc[train_len + val_len:]
        
        return train_data, val_data, test_data
    
    def resample_data(self,
                     data: pd.DataFrame,
                     target_interval: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        
        # 간격 매핑
        freq_map = {
            '1min': '1T', '5min': '5T', '15min': '15T', '30min': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        freq = freq_map.get(target_interval, target_interval)
        
        # OHLCV 리샘플링
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def generate_sample_data(self,
                           symbol: str = "BTC-USD",
                           start_date: str = "2020-01-01",
                           end_date: str = "2023-12-31",
                           initial_price: float = 50000.0) -> pd.DataFrame:
        """샘플 데이터 생성 (테스트용)"""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 랜덤 워크 + 트렌드
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(date_range))  # 일일 수익률
        
        # 가격 시계열 생성
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # OHLC 생성
        daily_volatility = 0.01
        
        data = []
        for i, (date, price) in enumerate(zip(date_range, prices)):
            # 일일 변동 범위
            high_low_range = price * daily_volatility * np.random.uniform(0.5, 1.5)
            
            # OHLC 생성
            if i == 0:
                open_price = price
            else:
                open_price = data[-1]['close'] * (1 + np.random.normal(0, 0.005))
            
            high = max(open_price, price) + np.random.uniform(0, high_low_range)
            low = min(open_price, price) - np.random.uniform(0, high_low_range)
            close = price
            
            # 거래량 생성
            volume = np.random.lognormal(15, 0.5)  # 로그정규분포
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        
        # 파일로 저장
        self.save_data(df, f"{symbol}_sample.csv")
        
        return df
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cached_data.clear()
        print("데이터 캐시가 초기화되었습니다")