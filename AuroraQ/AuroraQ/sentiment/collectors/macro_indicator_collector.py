#!/usr/bin/env python3
"""
Macro Indicator Collector for AuroraQ Sentiment Service
매크로 지표 실시간 수집기 - VIX, DXY, 금, 채권 등
"""

import asyncio
import aiohttp
import time
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np
from collections import deque
import pandas as pd

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """지표 유형"""
    CURRENCY = "currency"     # 통화 (DXY)
    VOLATILITY = "volatility" # 변동성 (VIX)
    COMMODITY = "commodity"   # 원자재 (Gold, Oil)
    BOND = "bond"            # 채권 (10Y Treasury)
    CRYPTO = "crypto"        # 암호화폐 관련
    EQUITY = "equity"        # 주식 지수

@dataclass
class MacroIndicator:
    """매크로 지표 데이터"""
    symbol: str
    name: str
    indicator_type: IndicatorType
    current_value: float
    previous_value: Optional[float] = None
    change_percent: float = 0.0
    change_absolute: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "yahoo_finance"
    
    # 변화율 계산 기간별
    change_1h: Optional[float] = None
    change_4h: Optional[float] = None
    change_24h: Optional[float] = None
    change_7d: Optional[float] = None
    
    # 통계 정보
    volatility_score: float = 0.0
    trend_strength: float = 0.0
    impact_score: float = 0.0  # 시장 영향도
    
    def calculate_changes(self, historical_data: List[Tuple[datetime, float]]):
        """시간대별 변화율 계산"""
        if not historical_data:
            return
        
        current_time = self.timestamp
        
        # 시간대별 변화율 계산
        for dt, value in reversed(historical_data):
            time_diff = (current_time - dt).total_seconds() / 3600  # 시간 단위
            
            if self.change_1h is None and time_diff >= 1:
                self.change_1h = ((self.current_value - value) / value) * 100
            
            if self.change_4h is None and time_diff >= 4:
                self.change_4h = ((self.current_value - value) / value) * 100
            
            if self.change_24h is None and time_diff >= 24:
                self.change_24h = ((self.current_value - value) / value) * 100
            
            if self.change_7d is None and time_diff >= 168:  # 7 * 24
                self.change_7d = ((self.current_value - value) / value) * 100
                break

@dataclass
class MarketImpactScore:
    """시장 영향 점수"""
    symbol: str
    base_score: float        # 기본 영향도 (0-1)
    volatility_multiplier: float  # 변동성 승수
    trend_multiplier: float  # 트렌드 강도 승수
    final_score: float       # 최종 영향도 점수
    confidence: float        # 신뢰도
    timestamp: datetime = field(default_factory=datetime.now)

class MacroIndicatorCollector:
    """매크로 지표 수집기"""
    
    def __init__(self, update_interval: int = 300):  # 5분 간격
        """
        초기화
        
        Args:
            update_interval: 업데이트 간격 (초)
        """
        self.update_interval = update_interval
        
        # 추적할 지표들 정의
        self.indicators_config = {
            # 통화 지표
            "DXY": {"name": "US Dollar Index", "type": IndicatorType.CURRENCY, "impact": 0.8},
            "EURUSD=X": {"name": "EUR/USD", "type": IndicatorType.CURRENCY, "impact": 0.6},
            
            # 변동성 지표
            "^VIX": {"name": "VIX Fear Index", "type": IndicatorType.VOLATILITY, "impact": 0.9},
            "^VXN": {"name": "NASDAQ Volatility", "type": IndicatorType.VOLATILITY, "impact": 0.7},
            
            # 원자재
            "GC=F": {"name": "Gold Futures", "type": IndicatorType.COMMODITY, "impact": 0.7},
            "CL=F": {"name": "Crude Oil", "type": IndicatorType.COMMODITY, "impact": 0.6},
            
            # 채권
            "^TNX": {"name": "10-Year Treasury", "type": IndicatorType.BOND, "impact": 0.8},
            "^FVX": {"name": "5-Year Treasury", "type": IndicatorType.BOND, "impact": 0.6},
            
            # 주식 지수
            "^GSPC": {"name": "S&P 500", "type": IndicatorType.EQUITY, "impact": 0.8},
            "^IXIC": {"name": "NASDAQ", "type": IndicatorType.EQUITY, "impact": 0.7},
            
            # 암호화폐 관련 (Grayscale Trust 등)
            "GBTC": {"name": "Grayscale Bitcoin Trust", "type": IndicatorType.CRYPTO, "impact": 0.5}
        }
        
        # 데이터 저장소
        self.current_indicators: Dict[str, MacroIndicator] = {}
        self.historical_data: Dict[str, deque] = {}  # 최근 7일 데이터
        self.impact_scores: Dict[str, MarketImpactScore] = {}
        
        # 실행 제어
        self.is_running = False
        self.collector_task: Optional[asyncio.Task] = None
        
        # 통계
        self.stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "last_update": None,
            "indicators_count": len(self.indicators_config),
            "avg_update_time": 0.0
        }
        
        # 각 지표별 히스토리 초기화
        for symbol in self.indicators_config.keys():
            self.historical_data[symbol] = deque(maxlen=2016)  # 7일 * 24시간 * 12 (5분 간격)
    
    async def start(self):
        """수집기 시작"""
        if self.is_running:
            logger.warning("Macro indicator collector already running")
            return
        
        self.is_running = True
        self.collector_task = asyncio.create_task(self._collection_loop())
        logger.info(f"Macro indicator collector started (interval: {self.update_interval}s)")
        
        # 초기 데이터 수집
        await self._collect_all_indicators()
    
    async def stop(self):
        """수집기 중지"""
        self.is_running = False
        
        if self.collector_task and not self.collector_task.done():
            self.collector_task.cancel()
            try:
                await self.collector_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Macro indicator collector stopped")
    
    async def _collection_loop(self):
        """수집 루프"""
        while self.is_running:
            try:
                await self._collect_all_indicators()
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    async def _collect_all_indicators(self):
        """모든 지표 수집"""
        start_time = time.time()
        
        try:
            logger.debug("Collecting macro indicators...")
            
            # 병렬로 데이터 수집
            tasks = []
            for symbol in self.indicators_config.keys():
                task = asyncio.create_task(self._collect_single_indicator(symbol))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            successful = 0
            failed = 0
            
            for symbol, result in zip(self.indicators_config.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect {symbol}: {result}")
                    failed += 1
                elif result:
                    successful += 1
            
            # 영향도 점수 계산
            self._calculate_impact_scores()
            
            # 통계 업데이트
            collection_time = time.time() - start_time
            self._update_stats(successful, failed, collection_time)
            
            logger.debug(f"Macro indicators collected: {successful} successful, {failed} failed in {collection_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to collect macro indicators: {e}")
    
    async def _collect_single_indicator(self, symbol: str) -> bool:
        """단일 지표 수집"""
        try:
            config = self.indicators_config[symbol]
            
            # yfinance를 사용하여 데이터 수집
            ticker = yf.Ticker(symbol)
            
            # 최근 데이터 가져오기 (1분 간격, 1일)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return False
            
            # 최신 가격
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            
            # 변화율 계산
            change_absolute = current_price - previous_price
            change_percent = (change_absolute / previous_price) * 100 if previous_price != 0 else 0
            
            # MacroIndicator 객체 생성
            indicator = MacroIndicator(
                symbol=symbol,
                name=config["name"],
                indicator_type=config["type"],
                current_value=current_price,
                previous_value=previous_price,
                change_percent=change_percent,
                change_absolute=change_absolute,
                timestamp=datetime.now(),
                source="yahoo_finance"
            )
            
            # 히스토리컬 데이터로 시간대별 변화율 계산
            historical_points = []
            for i in range(len(hist)):
                dt = hist.index[i].to_pydatetime()
                price = float(hist['Close'].iloc[i])
                historical_points.append((dt, price))
            
            indicator.calculate_changes(historical_points)
            
            # 변동성 및 트렌드 계산
            self._calculate_indicator_metrics(indicator, hist)
            
            # 저장
            self.current_indicators[symbol] = indicator
            
            # 히스토리 업데이트
            self.historical_data[symbol].append((datetime.now(), current_price))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect indicator {symbol}: {e}")
            return False
    
    def _calculate_indicator_metrics(self, indicator: MacroIndicator, hist_data: pd.DataFrame):
        """지표별 메트릭 계산"""
        try:
            if len(hist_data) < 10:
                return
            
            prices = hist_data['Close'].values
            
            # 변동성 점수 (최근 20분간 표준편차)
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                volatility = np.std(recent_prices) / np.mean(recent_prices)
                indicator.volatility_score = min(1.0, volatility * 10)  # 0-1 범위로 정규화
            
            # 트렌드 강도 (선형 회귀 기울기)
            if len(prices) >= 30:
                x = np.arange(len(prices[-30:]))
                y = prices[-30:]
                slope = np.polyfit(x, y, 1)[0]
                trend_strength = abs(slope) / np.mean(y)
                indicator.trend_strength = min(1.0, trend_strength * 100)
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {indicator.symbol}: {e}")
    
    def _calculate_impact_scores(self):
        """시장 영향도 점수 계산"""
        try:
            for symbol, indicator in self.current_indicators.items():
                config = self.indicators_config[symbol]
                base_impact = config["impact"]
                
                # 변동성 승수 (높은 변동성 = 높은 영향)
                volatility_multiplier = 1.0 + (indicator.volatility_score * 0.5)
                
                # 트렌드 승수 (강한 트렌드 = 높은 영향)
                trend_multiplier = 1.0 + (indicator.trend_strength * 0.3)
                
                # 변화율 기반 추가 승수
                change_multiplier = 1.0
                if abs(indicator.change_percent) > 1.0:  # 1% 이상 변화
                    change_multiplier = 1.0 + (abs(indicator.change_percent) / 10.0)
                
                # 최종 점수 계산
                final_score = base_impact * volatility_multiplier * trend_multiplier * change_multiplier
                final_score = min(2.0, final_score)  # 최대 2.0으로 제한
                
                # 신뢰도 계산 (최근 업데이트 시간 기반)
                time_diff = (datetime.now() - indicator.timestamp).total_seconds()
                confidence = max(0.5, 1.0 - (time_diff / 3600))  # 1시간 기준
                
                impact_score = MarketImpactScore(
                    symbol=symbol,
                    base_score=base_impact,
                    volatility_multiplier=volatility_multiplier,
                    trend_multiplier=trend_multiplier,
                    final_score=final_score,
                    confidence=confidence
                )
                
                self.impact_scores[symbol] = impact_score
                indicator.impact_score = final_score
                
        except Exception as e:
            logger.error(f"Failed to calculate impact scores: {e}")
    
    def _update_stats(self, successful: int, failed: int, collection_time: float):
        """통계 업데이트"""
        self.stats["total_updates"] += 1
        self.stats["successful_updates"] += successful
        self.stats["failed_updates"] += failed
        self.stats["last_update"] = datetime.now().isoformat()
        
        # 평균 업데이트 시간
        current_avg = self.stats["avg_update_time"]
        total_updates = self.stats["total_updates"]
        self.stats["avg_update_time"] = (
            (current_avg * (total_updates - 1) + collection_time) / total_updates
        )
    
    def get_indicator(self, symbol: str) -> Optional[MacroIndicator]:
        """특정 지표 조회"""
        return self.current_indicators.get(symbol)
    
    def get_indicators_by_type(self, indicator_type: IndicatorType) -> List[MacroIndicator]:
        """유형별 지표 조회"""
        return [
            indicator for indicator in self.current_indicators.values()
            if indicator.indicator_type == indicator_type
        ]
    
    def get_top_impact_indicators(self, limit: int = 5) -> List[Tuple[str, MarketImpactScore]]:
        """영향도 순 상위 지표들"""
        sorted_impacts = sorted(
            self.impact_scores.items(),
            key=lambda x: x[1].final_score,
            reverse=True
        )
        return sorted_impacts[:limit]
    
    def get_macro_summary(self) -> Dict[str, Any]:
        """매크로 지표 요약"""
        if not self.current_indicators:
            return {"status": "no_data"}
        
        # 유형별 평균 변화율
        type_changes = {}
        for indicator_type in IndicatorType:
            indicators = self.get_indicators_by_type(indicator_type)
            if indicators:
                avg_change = sum(ind.change_percent for ind in indicators) / len(indicators)
                type_changes[indicator_type.value] = {
                    "avg_change_percent": round(avg_change, 2),
                    "count": len(indicators)
                }
        
        # 상위 영향도 지표
        top_impacts = self.get_top_impact_indicators(3)
        
        # 시장 스트레스 지표
        stress_indicators = []
        vix = self.get_indicator("^VIX")
        if vix and vix.current_value > 20:  # VIX 20 이상은 스트레스
            stress_indicators.append(f"VIX High: {vix.current_value:.1f}")
        
        dxy = self.get_indicator("DXY")
        if dxy and abs(dxy.change_percent) > 0.5:  # DXY 0.5% 이상 변화
            direction = "강세" if dxy.change_percent > 0 else "약세"
            stress_indicators.append(f"USD {direction}: {dxy.change_percent:+.2f}%")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "indicators_count": len(self.current_indicators),
            "type_summary": type_changes,
            "top_impact_indicators": [
                {
                    "symbol": symbol,
                    "name": self.current_indicators[symbol].name,
                    "impact_score": round(impact.final_score, 2),
                    "change_percent": round(self.current_indicators[symbol].change_percent, 2)
                }
                for symbol, impact in top_impacts
            ],
            "market_stress_indicators": stress_indicators,
            "overall_sentiment": self._calculate_overall_macro_sentiment()
        }
    
    def _calculate_overall_macro_sentiment(self) -> Dict[str, Any]:
        """전체 매크로 감정 계산"""
        if not self.current_indicators:
            return {"score": 0.0, "direction": "neutral"}
        
        # 가중 평균 계산
        weighted_sum = 0.0
        total_weight = 0.0
        
        for symbol, indicator in self.current_indicators.items():
            config = self.indicators_config[symbol]
            weight = config["impact"]
            
            # 변화율을 감정 점수로 변환 (-1 ~ 1)
            change_score = np.tanh(indicator.change_percent / 5.0)  # 5% 변화를 기준으로 정규화
            
            # VIX는 역방향 (높을수록 부정적)
            if symbol == "^VIX":
                change_score = -change_score
            
            weighted_sum += change_score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # 방향 결정
        if overall_score > 0.1:
            direction = "bullish"
        elif overall_score < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return {
            "score": round(overall_score, 3),
            "direction": direction,
            "confidence": min(1.0, abs(overall_score) * 2)  # 절댓값이 클수록 높은 신뢰도
        }
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """수집기 통계"""
        return {
            "is_running": self.is_running,
            "update_interval": self.update_interval,
            **self.stats,
            "data_freshness": {
                symbol: (datetime.now() - indicator.timestamp).total_seconds()
                for symbol, indicator in self.current_indicators.items()
            }
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_macro_collector():
        """매크로 지표 수집기 테스트"""
        
        print("=== Macro Indicator Collector 테스트 ===")
        
        collector = MacroIndicatorCollector(update_interval=60)  # 1분 간격 테스트
        
        try:
            await collector.start()
            
            # 30초 대기 후 결과 확인
            await asyncio.sleep(30)
            
            print(f"\n1. 수집된 지표 개수: {len(collector.current_indicators)}")
            
            # 주요 지표들 출력
            key_indicators = ["^VIX", "DXY", "^GSPC", "GC=F"]
            for symbol in key_indicators:
                indicator = collector.get_indicator(symbol)
                if indicator:
                    print(f"\n{indicator.name} ({symbol}):")
                    print(f"  현재 가격: {indicator.current_value:.2f}")
                    print(f"  변화율: {indicator.change_percent:+.2f}%")
                    print(f"  변동성 점수: {indicator.volatility_score:.3f}")
                    print(f"  트렌드 강도: {indicator.trend_strength:.3f}")
                    print(f"  영향도 점수: {indicator.impact_score:.3f}")
            
            # 영향도 순위
            print(f"\n2. 상위 영향도 지표:")
            top_impacts = collector.get_top_impact_indicators(5)
            for i, (symbol, impact) in enumerate(top_impacts, 1):
                indicator = collector.current_indicators[symbol]
                print(f"  {i}. {indicator.name}: {impact.final_score:.2f} "
                      f"({indicator.change_percent:+.2f}%)")
            
            # 매크로 요약
            print(f"\n3. 매크로 지표 요약:")
            summary = collector.get_macro_summary()
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
            # 통계
            print(f"\n4. 수집기 통계:")
            stats = collector.get_collector_stats()
            for key, value in stats.items():
                if key != "data_freshness":
                    print(f"  {key}: {value}")
            
        finally:
            await collector.stop()
        
        print(f"\n=== 테스트 완료 ===")
    
    # 테스트 실행
    asyncio.run(test_macro_collector())