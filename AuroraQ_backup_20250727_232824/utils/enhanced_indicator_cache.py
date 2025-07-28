"""
향상된 지표 캐싱 시스템
- 모든 전략에서 공유하는 통합 캐시
- 타임스탬프 기반 무효화
- 효율적인 메모리 관리
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta
import threading
from utils.logger import get_logger

logger = get_logger("EnhancedIndicatorCache")


@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    value: Any
    timestamp: float
    data_hash: str
    hit_count: int = 0
    
    def is_valid(self, current_hash: str, ttl_seconds: float) -> bool:
        """캐시 유효성 검사"""
        # 해시 변경 확인
        if self.data_hash != current_hash:
            return False
        # TTL 확인
        if time.time() - self.timestamp > ttl_seconds:
            return False
        return True


class EnhancedIndicatorCache:
    """
    향상된 지표 캐싱 시스템
    - 전략 간 공유
    - 스마트 무효화
    - 통계 추적
    """
    
    def __init__(self, max_size: int = 5000, default_ttl: float = 300):
        """
        Args:
            max_size: 최대 캐시 크기
            default_ttl: 기본 TTL (초)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
        # 통계
        self.total_hits = 0
        self.total_misses = 0
        self.total_calculations = 0
        self.calculation_times: Dict[str, float] = {}
        
        # 지표별 계산 함수 등록
        self._register_indicators()
        
    def _register_indicators(self):
        """표준 지표 계산 함수 등록"""
        self.indicator_functions = {
            # 이동평균
            'ema': self._calculate_ema,
            'sma': self._calculate_sma,
            'wma': self._calculate_wma,
            
            # 모멘텀
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'macd_line': self._calculate_macd_line,
            'macd_signal': self._calculate_macd_signal,
            'stochastic': self._calculate_stochastic,
            
            # 변동성
            'bb': self._calculate_bollinger_bands,
            'bollinger': self._calculate_bollinger_bands,
            'bb_upper': self._calculate_bb_upper,
            'bb_middle': self._calculate_bb_middle,
            'bb_lower': self._calculate_bb_lower,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx,
            
            # 볼륨
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap,
            
            # 추세
            'cci': self._calculate_cci,
            'williams_r': self._calculate_williams_r,
        }
    
    def _generate_data_hash(self, data: pd.DataFrame, columns: list) -> str:
        """데이터 해시 생성 (효율적)"""
        try:
            # 관련 컬럼의 마지막 20개 값으로 해시 생성
            hash_data = []
            for col in columns:
                if col in data.columns:
                    values = data[col].tail(20).values
                    hash_data.append(values.tobytes())
            
            combined = b''.join(hash_data)
            return hashlib.md5(combined).hexdigest()[:16]
        except Exception as e:
            logger.error(f"해시 생성 실패: {e}")
            return str(time.time())
    
    def _generate_cache_key(self, indicator: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        param_str = "_".join(f"{k}:{v}" for k, v in sorted(params.items()))
        return f"{indicator}_{param_str}"
    
    def get(self, 
            indicator: str,
            data: pd.DataFrame,
            params: Optional[Dict[str, Any]] = None,
            ttl: Optional[float] = None,
            required_columns: Optional[list] = None) -> Any:
        """
        캐시에서 지표 조회 또는 계산
        
        Args:
            indicator: 지표 이름
            data: 가격 데이터
            params: 지표 파라미터
            ttl: Time To Live (초)
            required_columns: 해시 계산에 사용할 컬럼
        """
        if params is None:
            params = {}
        
        if required_columns is None:
            required_columns = ['close', 'high', 'low', 'volume']
        
        if ttl is None:
            ttl = self.default_ttl
        
        # 캐시 키와 데이터 해시 생성
        cache_key = self._generate_cache_key(indicator, params)
        data_hash = self._generate_data_hash(data, required_columns)
        
        with self.lock:
            # 캐시 확인
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if entry.is_valid(data_hash, ttl):
                    # 캐시 히트
                    self.total_hits += 1
                    entry.hit_count += 1
                    self.cache.move_to_end(cache_key)  # LRU
                    return entry.value
                else:
                    # 무효화된 엔트리 제거
                    del self.cache[cache_key]
            
            # 캐시 미스
            self.total_misses += 1
        
        # 지표 계산 (락 밖에서)
        start_time = time.time()
        
        if indicator in self.indicator_functions:
            # 등록된 함수 사용
            result = self.indicator_functions[indicator](data, **params)
        else:
            # 커스텀 계산 함수가 있다면 사용
            logger.warning(f"미등록 지표: {indicator}")
            return None
        
        calculation_time = time.time() - start_time
        self.total_calculations += 1
        self.calculation_times[indicator] = calculation_time
        
        # 캐시에 저장
        with self.lock:
            entry = CacheEntry(
                value=result,
                timestamp=time.time(),
                data_hash=data_hash
            )
            self.cache[cache_key] = entry
            
            # 크기 제한
            if len(self.cache) > self.max_size:
                # 가장 오래된 항목 제거
                self.cache.popitem(last=False)
        
        return result
    
    # ========== 지표 계산 함수들 ==========
    
    def _calculate_ema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """지수이동평균"""
        return ta.trend.ema_indicator(data['close'], window=period)
    
    def _calculate_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """단순이동평균"""
        return ta.trend.sma_indicator(data['close'], window=period)
    
    def _calculate_wma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """가중이동평균"""
        return ta.trend.wma_indicator(data['close'], window=period)
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI"""
        return ta.momentum.RSIIndicator(data['close'], window=period).rsi()
    
    def _calculate_macd(self, data: pd.DataFrame, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD"""
        macd = ta.trend.MACD(data['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'diff': macd.macd_diff()
        }
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, 
                                  period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
        """볼린저 밴드"""
        bb = ta.volatility.BollingerBands(data['close'], window=period, window_dev=std)
        return {
            'upper': bb.bollinger_hband(),
            'middle': bb.bollinger_mavg(),
            'lower': bb.bollinger_lband(),
            'width': bb.bollinger_wband(),
            'percent': bb.bollinger_pband()
        }
    
    def _calculate_bb_upper(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
        """볼린저 밴드 상단"""
        bb = ta.volatility.BollingerBands(data['close'], window=period, window_dev=std)
        return bb.bollinger_hband()
    
    def _calculate_bb_middle(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """볼린저 밴드 중앙선"""
        bb = ta.volatility.BollingerBands(data['close'], window=period)
        return bb.bollinger_mavg()
    
    def _calculate_bb_lower(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
        """볼린저 밴드 하단"""
        bb = ta.volatility.BollingerBands(data['close'], window=period, window_dev=std)
        return bb.bollinger_lband()
    
    def _calculate_macd_signal(self, data: pd.DataFrame, 
                              fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD 시그널 라인"""
        macd = ta.trend.MACD(data['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        return macd.macd_signal()
    
    def _calculate_macd_line(self, data: pd.DataFrame, 
                            fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD 라인"""
        macd = ta.trend.MACD(data['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        return macd.macd()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        return ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], window=period
        ).average_true_range()
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        return ta.trend.ADXIndicator(
            data['high'], data['low'], data['close'], window=period
        ).adx()
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        return ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        return ta.volume.VolumeWeightedAveragePrice(
            data['high'], data['low'], data['close'], data['volume']
        ).volume_weighted_average_price()
    
    def _calculate_stochastic(self, data: pd.DataFrame, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(
            data['high'], data['low'], data['close'], 
            window=k_period, smooth_window=d_period
        )
        return {
            'k': stoch.stoch(),
            'd': stoch.stoch_signal()
        }
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        return ta.trend.CCIIndicator(
            data['high'], data['low'], data['close'], window=period
        ).cci()
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        return ta.momentum.WilliamsRIndicator(
            data['high'], data['low'], data['close'], lbp=period
        ).williams_r()
    
    # ========== 유틸리티 메서드 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self.lock:
            total_requests = self.total_hits + self.total_misses
            hit_rate = self.total_hits / total_requests if total_requests > 0 else 0
            
            # 지표별 계산 시간 평균
            avg_calc_times = {
                ind: time / self.total_calculations 
                for ind, time in self.calculation_times.items()
            } if self.total_calculations > 0 else {}
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': round(hit_rate, 4),
                'total_hits': self.total_hits,
                'total_misses': self.total_misses,
                'total_calculations': self.total_calculations,
                'avg_calculation_times': avg_calc_times,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """메모리 사용량 추정 (MB)"""
        # 간단한 추정: 각 엔트리당 평균 1KB
        return len(self.cache) * 0.001
    
    def clear(self, indicator: Optional[str] = None):
        """캐시 초기화"""
        with self.lock:
            if indicator:
                # 특정 지표만 제거
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{indicator}_")]
                for key in keys_to_remove:
                    del self.cache[key]
            else:
                # 전체 초기화
                self.cache.clear()
                self.total_hits = 0
                self.total_misses = 0
                self.total_calculations = 0
                self.calculation_times.clear()
    
    def invalidate_old_entries(self, max_age_seconds: float = 3600):
        """오래된 엔트리 무효화"""
        current_time = time.time()
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            if keys_to_remove:
                logger.info(f"무효화된 엔트리: {len(keys_to_remove)}개")


# 전역 캐시 인스턴스
_enhanced_cache = EnhancedIndicatorCache()


def get_enhanced_cache() -> EnhancedIndicatorCache:
    """전역 향상된 캐시 인스턴스 반환"""
    return _enhanced_cache


# 데코레이터
def cached_indicator(indicator_name: str, 
                    ttl: float = 300,
                    required_columns: Optional[list] = None):
    """
    지표 계산 캐싱 데코레이터
    
    사용 예:
    @cached_indicator('custom_ma', ttl=600)
    def calculate_custom_ma(data: pd.DataFrame, period: int = 20):
        return data['close'].rolling(period).mean()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(data: pd.DataFrame, **kwargs):
            # 커스텀 함수를 등록
            if indicator_name not in _enhanced_cache.indicator_functions:
                _enhanced_cache.indicator_functions[indicator_name] = func
            
            return _enhanced_cache.get(
                indicator=indicator_name,
                data=data,
                params=kwargs,
                ttl=ttl,
                required_columns=required_columns
            )
        return wrapper
    return decorator


# 주기적 캐시 정리를 위한 백그라운드 태스크
def start_cache_maintenance(interval_seconds: int = 3600):
    """캐시 유지보수 태스크 시작"""
    import threading
    
    def maintenance_task():
        while True:
            time.sleep(interval_seconds)
            try:
                _enhanced_cache.invalidate_old_entries()
                stats = _enhanced_cache.get_stats()
                logger.info(f"캐시 유지보수 완료: {stats}")
            except Exception as e:
                logger.error(f"캐시 유지보수 오류: {e}")
    
    thread = threading.Thread(target=maintenance_task, daemon=True)
    thread.start()
    logger.info("캐시 유지보수 태스크 시작됨")