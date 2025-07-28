"""
데이터 및 지표 계층 (Data & Indicators Layer)
- 과거 시세 데이터 (OHLCV) 관리
- 감정 점수 통합
- 변동성 지표 계산
- 다중 타임프레임 지원
- 지표 캐싱으로 중복 연산 최소화
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from functools import lru_cache
from collections import OrderedDict
# import talib  # TA-Lib 의존성 제거

logger = logging.getLogger(__name__)


class IndicatorCache:
    """지표 계산 결과 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 최대 캐시 크기
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def _make_key(self, indicator_name: str, params: Dict, data_hash: str) -> str:
        """캐시 키 생성"""
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{indicator_name}_{param_str}_{data_hash}"
    
    def get(self, indicator_name: str, params: Dict, data_hash: str):
        """캐시에서 지표 값 조회 (Series 또는 DataFrame 반환)"""
        key = self._make_key(indicator_name, params, data_hash)
        
        if key in self.cache:
            self.hit_count += 1
            # LRU: 최근 사용한 항목을 맨 뒤로 이동
            self.cache.move_to_end(key)
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, indicator_name: str, params: Dict, data_hash: str, value):
        """캐시에 지표 값 저장 (Series 또는 DataFrame 지원)"""
        key = self._make_key(indicator_name, params, data_hash)
        
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class MultiTimeframeData:
    """다중 타임프레임 데이터 관리"""
    
    def __init__(self, base_timeframe: str = "5min"):
        """
        Args:
            base_timeframe: 기본 타임프레임 (5min, 15min, 1h 등)
        """
        self.base_timeframe = base_timeframe
        self.timeframes = {
            "5min": pd.Timedelta(minutes=5),
            "15min": pd.Timedelta(minutes=15),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1D": pd.Timedelta(days=1)
        }
        self.data = {}
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """데이터를 목표 타임프레임으로 리샘플링"""
        if target_timeframe not in self.timeframes:
            raise ValueError(f"지원하지 않는 타임프레임: {target_timeframe}")
        
        # 타임스탬프 인덱스 설정
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # OHLCV 리샘플링
        resampled = df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
    
    def align_timeframes(self, base_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """모든 타임프레임 데이터 정렬"""
        aligned_data = {"5min": base_data}
        
        for tf in ["15min", "1h"]:
            aligned_data[tf] = self.resample_data(base_data, tf)
        
        return aligned_data


class DataLayer:
    """
    데이터 및 지표 계층
    - OHLCV 데이터 관리
    - 감정 점수 통합
    - 기술적 지표 계산
    - 캐싱 및 최적화
    """
    
    def __init__(self, 
                 cache_size: int = 1000,
                 enable_multiframe: bool = True):
        """
        Args:
            cache_size: 지표 캐시 크기
            enable_multiframe: 다중 타임프레임 활성화
        """
        self.indicator_cache = IndicatorCache(cache_size)
        self.enable_multiframe = enable_multiframe
        
        if enable_multiframe:
            self.mtf_data = MultiTimeframeData()
        
        self.sentiment_data = None
        self.price_data = None
        self.current_index = 0
    
    def load_price_data(self, file_path: str) -> pd.DataFrame:
        """가격 데이터 로드"""
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            
            # 필수 컬럼 확인
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = set(required_columns) - set(df.columns)
            if missing:
                # 대소문자 변환 시도
                df.columns = df.columns.str.lower()
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValueError(f"필수 컬럼 누락: {missing}")
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"가격 데이터 로드 완료: {len(self.price_data)}개 레코드")
            
            return self.price_data
            
        except Exception as e:
            logger.error(f"가격 데이터 로드 실패: {e}")
            raise
    
    def load_sentiment_data(self, file_path: str) -> pd.DataFrame:
        """감정 점수 데이터 로드"""
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.sentiment_data = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"감정 데이터 로드 완료: {len(self.sentiment_data)}개 레코드")
            return self.sentiment_data
            
        except Exception as e:
            logger.warning(f"감정 데이터 로드 실패: {e} - 기본값 사용")
            # 더미 데이터 생성
            return self._create_dummy_sentiment()
    
    def _create_dummy_sentiment(self) -> pd.DataFrame:
        """더미 감정 데이터 생성"""
        if self.price_data is None:
            return pd.DataFrame()
        
        timestamps = self.price_data['timestamp']
        n = len(timestamps)
        
        # 시간에 따른 감정 변화 시뮬레이션
        t = np.linspace(0, 4 * np.pi, n)
        base_sentiment = 0.5
        sentiment_scores = base_sentiment + 0.3 * np.sin(t) + 0.1 * np.random.randn(n)
        sentiment_scores = np.clip(sentiment_scores, 0, 1)
        
        self.sentiment_data = pd.DataFrame({
            'timestamp': timestamps,
            'sentiment_score': sentiment_scores,
            'confidence': 0.8 + 0.2 * np.random.rand(n)
        })
        
        return self.sentiment_data
    
    def get_data_window(self, end_index: int, window_size: int = 100) -> Dict[str, pd.DataFrame]:
        """특정 인덱스까지의 데이터 윈도우 반환"""
        start_index = max(0, end_index - window_size + 1)
        
        window_data = {
            "price": self.price_data.iloc[start_index:end_index + 1].copy()
        }
        
        # 감정 점수 정렬
        if self.sentiment_data is not None:
            current_time = self.price_data.iloc[end_index]['timestamp']
            sentiment_score = self._get_aligned_sentiment(current_time)
            window_data["sentiment_score"] = sentiment_score
        
        # 다중 타임프레임 데이터
        if self.enable_multiframe:
            mtf_windows = self.mtf_data.align_timeframes(window_data["price"])
            window_data["timeframes"] = mtf_windows
        
        return window_data
    
    def _get_aligned_sentiment(self, timestamp: pd.Timestamp) -> float:
        """타임스탬프에 맞는 감정 점수 반환"""
        if self.sentiment_data is None or len(self.sentiment_data) == 0:
            return 0.5
        
        # 가장 가까운 감정 점수 찾기
        time_diff = abs(self.sentiment_data['timestamp'] - timestamp)
        nearest_idx = time_diff.idxmin()
        
        # 1시간 이내의 데이터만 사용
        if time_diff.iloc[nearest_idx] <= pd.Timedelta(hours=1):
            return float(self.sentiment_data.iloc[nearest_idx]['sentiment_score'])
        
        return 0.5  # 기본값
    
    def calculate_indicators(self, price_df: pd.DataFrame, indicators: List[str]) -> Dict[str, pd.Series]:
        """기술적 지표 계산 (캐싱 적용)"""
        # 빈 결과 딕셔너리 초기화
        results = {}
        
        # 입력 검증
        if price_df is None or len(price_df) == 0 or indicators is None:
            logger.warning("입력 데이터가 비어있거나 없음 - 빈 지표 딕셔너리 반환")
            return results
        
        # 데이터 해시 생성 (캐시 키용)
        try:
            data_hash = str(len(price_df)) + str(price_df['close'].iloc[-1])
        except (IndexError, KeyError):
            logger.warning("가격 데이터가 유효하지 않음 - 빈 지표 딕셔너리 반환")
            return results
        
        # 볼린저 밴드와 MACD는 한번만 계산하여 분리해서 저장
        bb_calculated = False
        bb_upper, bb_middle, bb_lower = None, None, None
        macd_calculated = False
        macd_line, macd_signal_line, macd_histogram = None, None, None
        
        for indicator in indicators:
            try:
                # 캐시 확인
                cached = self.indicator_cache.get(indicator, {}, data_hash)
                if cached is not None:
                    results[indicator] = cached
                    continue
                
                # 지표 계산 (TA-Lib 없이 구현)
                value = None
                if indicator == "sma_20":
                    value = price_df['close'].rolling(20).mean()
                elif indicator == "sma_50":
                    value = price_df['close'].rolling(50).mean()
                elif indicator == "ema_12":
                    value = price_df['close'].ewm(span=12).mean()
                elif indicator == "ema_26":
                    value = price_df['close'].ewm(span=26).mean()
                elif indicator == "rsi":
                    value = self._calculate_rsi(price_df['close'], 14)
                elif indicator == "macd":
                    if not macd_calculated:
                        macd_line, macd_signal_line, macd_histogram = self._calculate_macd(price_df['close'])
                        macd_calculated = True
                    value = pd.DataFrame({'macd': macd_line, 'signal': macd_signal_line, 'hist': macd_histogram})
                elif indicator == "macd_line":
                    if not macd_calculated:
                        macd_line, macd_signal_line, macd_histogram = self._calculate_macd(price_df['close'])
                        macd_calculated = True
                    value = macd_line
                elif indicator == "macd_signal":
                    if not macd_calculated:
                        macd_line, macd_signal_line, macd_histogram = self._calculate_macd(price_df['close'])
                        macd_calculated = True
                    value = macd_signal_line
                elif indicator == "macd_hist":
                    if not macd_calculated:
                        macd_line, macd_signal_line, macd_histogram = self._calculate_macd(price_df['close'])
                        macd_calculated = True
                    value = macd_histogram
                elif indicator == "bbands":
                    if not bb_calculated:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_df['close'], 20)
                        bb_calculated = True
                    value = pd.DataFrame({'upper': bb_upper, 'middle': bb_middle, 'lower': bb_lower})
                elif indicator == "bollinger":
                    if not bb_calculated:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_df['close'], 20)
                        bb_calculated = True
                    value = {
                        'upper': bb_upper,
                        'middle': bb_middle, 
                        'lower': bb_lower
                    }
                elif indicator == "bb_upper":
                    if not bb_calculated:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_df['close'], 20)
                        bb_calculated = True
                    value = bb_upper
                elif indicator == "bb_middle":
                    if not bb_calculated:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_df['close'], 20)
                        bb_calculated = True
                    value = bb_middle
                elif indicator == "bb_lower":
                    if not bb_calculated:
                        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_df['close'], 20)
                        bb_calculated = True
                    value = bb_lower
                elif indicator == "atr":
                    value = self._calculate_atr(price_df, 14)
                elif indicator == "adx":
                    value = self._calculate_adx(price_df, 14)
                elif indicator == "volatility":
                    # 단순 변동성 (표준편차)
                    value = price_df['close'].pct_change().rolling(20).std()
                else:
                    logger.warning(f"지원하지 않는 지표: {indicator}")
                    continue
                
                # 계산된 값이 유효한 경우에만 저장
                if value is not None:
                    # 캐시 저장
                    self.indicator_cache.put(indicator, {}, data_hash, value)
                    results[indicator] = value
                    
            except Exception as e:
                logger.error(f"지표 '{indicator}' 계산 중 오류: {e}")
                continue
        
        # 볼린저 밴드 계산된 경우 개별 지표들과 통합 지표 모두 캐시에 저장
        if bb_calculated:
            if bb_upper is not None and "bb_upper" not in results:
                self.indicator_cache.put("bb_upper", {}, data_hash, bb_upper)
                if "bb_upper" in indicators:
                    results["bb_upper"] = bb_upper
            if bb_middle is not None and "bb_middle" not in results:
                self.indicator_cache.put("bb_middle", {}, data_hash, bb_middle)
                if "bb_middle" in indicators:
                    results["bb_middle"] = bb_middle
            if bb_lower is not None and "bb_lower" not in results:
                self.indicator_cache.put("bb_lower", {}, data_hash, bb_lower)
                if "bb_lower" in indicators:
                    results["bb_lower"] = bb_lower
            
            # 통합 bollinger 지표도 저장 (레거시 호환용)
            if bb_upper is not None and bb_middle is not None and bb_lower is not None:
                bollinger_dict = {
                    'upper': bb_upper,
                    'middle': bb_middle, 
                    'lower': bb_lower
                }
                self.indicator_cache.put("bollinger", {}, data_hash, bollinger_dict)
                if "bollinger" in indicators:
                    results["bollinger"] = bollinger_dict
        
        # MACD 계산된 경우 개별 지표들도 캐시에 저장
        if macd_calculated:
            if macd_line is not None and "macd_line" not in results:
                self.indicator_cache.put("macd_line", {}, data_hash, macd_line)
                if "macd_line" in indicators:
                    results["macd_line"] = macd_line
            if macd_signal_line is not None and "macd_signal" not in results:
                self.indicator_cache.put("macd_signal", {}, data_hash, macd_signal_line)
                if "macd_signal" in indicators:
                    results["macd_signal"] = macd_signal_line
            if macd_histogram is not None and "macd_hist" not in results:
                self.indicator_cache.put("macd_hist", {}, data_hash, macd_histogram)
                if "macd_hist" in indicators:
                    results["macd_hist"] = macd_histogram
        
        return results
    
    def get_volatility_metrics(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """변동성 메트릭 계산"""
        returns = price_df['close'].pct_change().dropna()
        
        metrics = {
            "volatility_20d": returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0,
            "volatility_5d": returns.rolling(5).std().iloc[-1] if len(returns) >= 5 else 0,
            "atr": self.calculate_indicators(price_df, ["atr"])["atr"].iloc[-1],
            "high_low_range": (price_df['high'].iloc[-1] - price_df['low'].iloc[-1]) / price_df['close'].iloc[-1]
        }
        
        return metrics
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return self.indicator_cache.get_stats()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> tuple:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX 계산 (간단한 버전)"""
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
        
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return adx
    
    def reset(self):
        """데이터 레이어 초기화"""
        self.indicator_cache.clear()
        self.current_index = 0
        logger.info("데이터 레이어 초기화 완료")