#!/usr/bin/env python3
"""
AuroraQ Integration Adapter
감정 서비스와 AuroraQ 메인 시스템 간의 통합 어댑터
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiohttp
from pathlib import Path

# 로컬 임포트
from ..processors.sentiment_fusion_manager import SentimentFusionManager, FusedSentiment
from ..processors.big_event_detector import BigEventDetector, BigEvent
from ..models.keyword_scorer import KeywordScorer
from ..utils.content_cache_manager import ContentCacheManager

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """매매 모드"""
    LIVE = "live"           # 실전 매매
    PAPER = "paper"         # 가상 매매
    BACKTEST = "backtest"   # 백테스팅

class SignalStrength(Enum):
    """신호 강도"""
    VERY_STRONG = "very_strong"     # 0.8 ~ 1.0
    STRONG = "strong"               # 0.6 ~ 0.8
    MODERATE = "moderate"           # 0.4 ~ 0.6
    WEAK = "weak"                   # 0.2 ~ 0.4
    VERY_WEAK = "very_weak"         # 0.0 ~ 0.2

@dataclass
class TradingSignal:
    """AuroraQ 매매 신호"""
    symbol: str
    signal_type: str  # "buy", "sell", "hold"
    strength: SignalStrength
    confidence: float
    sentiment_score: float
    sentiment_direction: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_level: str = "medium"  # "low", "medium", "high"
    
    # 감정 분석 메타데이터
    news_count: int = 0
    big_events: List[str] = None
    fusion_method: str = "adaptive"
    processing_time: float = 0.0
    
    # 타임스탬프
    created_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if self.big_events is None:
            self.big_events = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=4)  # 4시간 유효
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "strength": self.strength.value,
            "confidence": round(self.confidence, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_direction": self.sentiment_direction,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "risk_level": self.risk_level,
            "news_count": self.news_count,
            "big_events": self.big_events,
            "fusion_method": self.fusion_method,
            "processing_time": round(self.processing_time, 3),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }

@dataclass
class MarketContext:
    """시장 컨텍스트"""
    symbol: str
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    
    # 기술적 지표
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    # 시장 상태
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    market_sentiment: Optional[str] = None

class AuroraQAdapter:
    """AuroraQ 통합 어댑터"""
    
    def __init__(self,
                 fusion_manager: SentimentFusionManager,
                 event_detector: BigEventDetector,
                 keyword_scorer: KeywordScorer,
                 cache_manager: Optional[ContentCacheManager] = None,
                 aurora_api_url: str = "http://localhost:8080",
                 aurora_api_key: str = ""):
        """
        초기화
        
        Args:
            fusion_manager: 감정 융합 매니저
            event_detector: 빅 이벤트 감지기
            keyword_scorer: 키워드 스코어러
            cache_manager: 캐시 매니저
            aurora_api_url: AuroraQ API URL
            aurora_api_key: AuroraQ API 키
        """
        self.fusion_manager = fusion_manager
        self.event_detector = event_detector
        self.keyword_scorer = keyword_scorer
        self.cache_manager = cache_manager
        self.aurora_api_url = aurora_api_url.rstrip('/')
        self.aurora_api_key = aurora_api_key
        
        # HTTP 세션
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 신호 생성 설정
        self.signal_thresholds = {
            "buy_strong": 0.6,      # 강한 매수 신호
            "buy_moderate": 0.3,    # 보통 매수 신호
            "sell_strong": -0.6,    # 강한 매도 신호
            "sell_moderate": -0.3,  # 보통 매도 신호
            "min_confidence": 0.5,  # 최소 신뢰도
            "min_news_count": 2     # 최소 뉴스 개수
        }
        
        # 리스크 매핑
        self.risk_mapping = {
            "very_strong": "low",
            "strong": "low", 
            "moderate": "medium",
            "weak": "medium",
            "very_weak": "high"
        }
        
        # 신호 히스토리
        self.signal_history: List[TradingSignal] = []
        self.max_history_size = 1000
        
        # 통계
        self.stats = {
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_confidence": 0.0,
            "last_signal_time": None,
            "aurora_api_calls": 0,
            "aurora_api_errors": 0
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'AuroraQ-SentimentAdapter/1.0'
            }
        )
        
        if self.aurora_api_key:
            self.session.headers['Authorization'] = f'Bearer {self.aurora_api_key}'
        
        logger.info("AuroraQ adapter initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
        logger.info("AuroraQ adapter closed")
    
    def _determine_signal_strength(self, sentiment_score: float, confidence: float) -> SignalStrength:
        """신호 강도 결정"""
        
        # 절대값과 신뢰도를 종합하여 강도 계산
        abs_score = abs(sentiment_score)
        combined_strength = (abs_score * 0.7) + (confidence * 0.3)
        
        if combined_strength >= 0.8:
            return SignalStrength.VERY_STRONG
        elif combined_strength >= 0.6:
            return SignalStrength.STRONG
        elif combined_strength >= 0.4:
            return SignalStrength.MODERATE
        elif combined_strength >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _determine_signal_type(self, sentiment_score: float, confidence: float) -> str:
        """매매 신호 유형 결정"""
        
        if confidence < self.signal_thresholds["min_confidence"]:
            return "hold"
        
        if sentiment_score >= self.signal_thresholds["buy_strong"]:
            return "buy"
        elif sentiment_score >= self.signal_thresholds["buy_moderate"]:
            return "buy"
        elif sentiment_score <= self.signal_thresholds["sell_strong"]:
            return "sell"
        elif sentiment_score <= self.signal_thresholds["sell_moderate"]:
            return "sell"
        else:
            return "hold"
    
    def _calculate_price_targets(self,
                               current_price: float,
                               sentiment_score: float,
                               volatility: float = 0.05) -> tuple:
        """가격 목표 계산"""
        
        # 감정 점수에 따른 가격 변동 예상
        expected_change = sentiment_score * volatility
        
        if sentiment_score > 0:  # 매수 신호
            price_target = current_price * (1 + abs(expected_change))
            stop_loss = current_price * (1 - volatility * 0.5)
        else:  # 매도 신호
            price_target = current_price * (1 - abs(expected_change))
            stop_loss = current_price * (1 + volatility * 0.5)
        
        return price_target, stop_loss
    
    async def get_market_context(self, symbol: str) -> Optional[MarketContext]:
        """AuroraQ에서 시장 컨텍스트 조회"""
        
        if not self.session:
            logger.error("Session not initialized")
            return None
        
        try:
            self.stats["aurora_api_calls"] += 1
            
            async with self.session.get(
                f"{self.aurora_api_url}/api/v1/market/context/{symbol}"
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    return MarketContext(
                        symbol=symbol,
                        current_price=data.get("current_price"),
                        price_change_24h=data.get("price_change_24h"),
                        volume_24h=data.get("volume_24h"),
                        market_cap=data.get("market_cap"),
                        rsi=data.get("technical_indicators", {}).get("rsi"),
                        macd=data.get("technical_indicators", {}).get("macd"),
                        bollinger_position=data.get("technical_indicators", {}).get("bollinger_position"),
                        volatility=data.get("volatility"),
                        liquidity_score=data.get("liquidity_score"),
                        market_sentiment=data.get("market_sentiment")
                    )
                else:
                    logger.warning(f"Failed to get market context: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            self.stats["aurora_api_errors"] += 1
            return None
    
    async def generate_trading_signal(self,
                                    symbol: str,
                                    text_content: Optional[str] = None,
                                    market_context: Optional[MarketContext] = None) -> Optional[TradingSignal]:
        """매매 신호 생성"""
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating trading signal for {symbol}")
            
            # 1. 시장 컨텍스트 조회 (제공되지 않은 경우)
            if market_context is None:
                market_context = await self.get_market_context(symbol)
            
            # 2. 최신 감정 분석 결과 조회
            realtime_sentiment = await self.fusion_manager.get_realtime_sentiment(
                symbol=symbol, hours_back=24
            )
            
            # 3. 빅 이벤트 조회
            big_events = self.event_detector.get_events_by_symbol(symbol)
            recent_events = [
                event for event in big_events
                if (datetime.now() - event.detected_at).total_seconds() < 86400  # 24시간 내
            ]
            
            # 4. 텍스트 콘텐츠가 있는 경우 즉시 분석
            if text_content:
                content_hash = f"signal_{symbol}_{int(time.time())}"
                fusion_result = await self.fusion_manager.fuse_sentiment(
                    content_hash=content_hash,
                    text=text_content
                )
                sentiment_score = fusion_result.final_score
                confidence = fusion_result.final_confidence
                direction = fusion_result.direction
                fusion_method = fusion_result.fusion_method.value
            else:
                # 실시간 감정 데이터 사용
                sentiment_score = realtime_sentiment.get("score", 0.0)
                confidence = realtime_sentiment.get("confidence", 0.0)
                direction = realtime_sentiment.get("trend", "neutral")
                fusion_method = "realtime_aggregate"
            
            # 5. 뉴스 개수 확인
            news_count = realtime_sentiment.get("news_count", 0)
            if news_count < self.signal_thresholds["min_news_count"]:
                logger.info(f"Insufficient news count for {symbol}: {news_count}")
                # return None  # 뉴스가 부족하더라도 신호 생성 (hold 신호)
            
            # 6. 빅 이벤트 영향도 조정
            event_impact_adjustment = 0.0
            event_descriptions = []
            
            for event in recent_events:
                # 이벤트 영향도를 감정 점수에 반영
                if event.final_impact_score > 1.0:  # 고영향 이벤트
                    adjustment = event.sentiment_bias * 0.2  # 최대 20% 조정
                    event_impact_adjustment += adjustment
                    event_descriptions.append(f"{event.event_type.value}:{event.final_impact_score:.2f}")
            
            # 최종 감정 점수 조정
            adjusted_sentiment_score = max(-1.0, min(1.0, sentiment_score + event_impact_adjustment))
            
            # 7. 매매 신호 결정
            signal_type = self._determine_signal_type(adjusted_sentiment_score, confidence)
            signal_strength = self._determine_signal_strength(adjusted_sentiment_score, confidence)
            
            # 8. 가격 목표 계산 (시장 컨텍스트가 있는 경우)
            price_target = None
            stop_loss = None
            
            if market_context and market_context.current_price:
                volatility = market_context.volatility or 0.05
                price_target, stop_loss = self._calculate_price_targets(
                    market_context.current_price,
                    adjusted_sentiment_score,
                    volatility
                )
            
            # 9. 리스크 레벨 결정
            risk_level = self.risk_mapping.get(signal_strength.value, "medium")
            
            # 10. 매매 신호 생성
            trading_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=confidence,
                sentiment_score=adjusted_sentiment_score,
                sentiment_direction=direction,
                price_target=price_target,
                stop_loss=stop_loss,
                risk_level=risk_level,
                news_count=news_count,
                big_events=event_descriptions,
                fusion_method=fusion_method,
                processing_time=time.time() - start_time
            )
            
            # 11. 히스토리에 추가
            self.signal_history.append(trading_signal)
            if len(self.signal_history) > self.max_history_size:
                self.signal_history.pop(0)
            
            # 12. 통계 업데이트
            self._update_stats(trading_signal)
            
            logger.info(f"Trading signal generated: {symbol} -> {signal_type.upper()} "
                       f"({signal_strength.value}, confidence: {confidence:.3f})")
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Failed to generate trading signal for {symbol}: {e}")
            return None
    
    def _update_stats(self, signal: TradingSignal):
        """통계 업데이트"""
        
        self.stats["signals_generated"] += 1
        
        if signal.signal_type == "buy":
            self.stats["buy_signals"] += 1
        elif signal.signal_type == "sell":
            self.stats["sell_signals"] += 1
        else:
            self.stats["hold_signals"] += 1
        
        # 평균 신뢰도 업데이트
        total_signals = self.stats["signals_generated"]
        current_avg = self.stats["avg_confidence"]
        
        self.stats["avg_confidence"] = (
            (current_avg * (total_signals - 1) + signal.confidence) / total_signals
        )
        
        self.stats["last_signal_time"] = signal.created_at.isoformat()
    
    async def send_signal_to_aurora(self,
                                  signal: TradingSignal,
                                  trading_mode: TradingMode = TradingMode.PAPER) -> bool:
        """AuroraQ에 매매 신호 전송"""
        
        if not self.session:
            logger.error("Session not initialized")
            return False
        
        try:
            self.stats["aurora_api_calls"] += 1
            
            payload = {
                "trading_mode": trading_mode.value,
                "signal": signal.to_dict(),
                "source": "sentiment_service",
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(
                f"{self.aurora_api_url}/api/v1/signals/sentiment",
                json=payload
            ) as response:
                
                if response.status == 200:
                    logger.info(f"Signal sent to AuroraQ: {signal.symbol} {signal.signal_type}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send signal to AuroraQ: {response.status} - {error_text}")
                    self.stats["aurora_api_errors"] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending signal to AuroraQ: {e}")
            self.stats["aurora_api_errors"] += 1
            return False
    
    async def process_symbol_batch(self,
                                 symbols: List[str],
                                 trading_mode: TradingMode = TradingMode.PAPER,
                                 send_to_aurora: bool = True) -> List[TradingSignal]:
        """심볼 배치 처리"""
        
        logger.info(f"Processing {len(symbols)} symbols in batch")
        
        signals = []
        
        for symbol in symbols:
            try:
                signal = await self.generate_trading_signal(symbol)
                
                if signal:
                    signals.append(signal)
                    
                    if send_to_aurora:
                        await self.send_signal_to_aurora(signal, trading_mode)
                
                # API 호출 간격 조절
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue
        
        logger.info(f"Batch processing completed: {len(signals)} signals generated")
        return signals
    
    def get_signal_history(self,
                          symbol: Optional[str] = None,
                          hours_back: int = 24,
                          signal_type: Optional[str] = None) -> List[TradingSignal]:
        """신호 히스토리 조회"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_signals = []
        for signal in self.signal_history:
            if signal.created_at < cutoff_time:
                continue
            
            if symbol and signal.symbol != symbol.upper():
                continue
            
            if signal_type and signal.signal_type != signal_type:
                continue
            
            filtered_signals.append(signal)
        
        return sorted(filtered_signals, key=lambda x: x.created_at, reverse=True)
    
    def get_adapter_stats(self) -> Dict[str, Any]:
        """어댑터 통계 반환"""
        
        return {
            **self.stats,
            "signal_history_size": len(self.signal_history),
            "signal_thresholds": self.signal_thresholds,
            "aurora_api_url": self.aurora_api_url,
            "aurora_api_configured": bool(self.aurora_api_key)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        
        health_status = {
            "adapter_status": "healthy",
            "fusion_manager": "unknown",
            "event_detector": "unknown", 
            "aurora_connectivity": "unknown",
            "last_signal": self.stats.get("last_signal_time"),
            "total_signals": self.stats["signals_generated"]
        }
        
        try:
            # 융합 매니저 상태 확인
            fusion_stats = self.fusion_manager.get_fusion_stats()
            health_status["fusion_manager"] = "healthy" if fusion_stats else "unhealthy"
        except Exception:
            health_status["fusion_manager"] = "error"
        
        try:
            # 이벤트 감지기 상태 확인
            detector_stats = self.event_detector.get_detector_stats()
            health_status["event_detector"] = "healthy" if detector_stats else "unhealthy"
        except Exception:
            health_status["event_detector"] = "error"
        
        try:
            # AuroraQ 연결 확인
            if self.session:
                async with self.session.get(f"{self.aurora_api_url}/health") as response:
                    health_status["aurora_connectivity"] = "healthy" if response.status == 200 else "degraded"
            else:
                health_status["aurora_connectivity"] = "disconnected"
        except Exception:
            health_status["aurora_connectivity"] = "error"
        
        # 전체 상태 결정
        if all(status in ["healthy", "unknown"] for status in health_status.values() if status != health_status["adapter_status"]):
            health_status["adapter_status"] = "healthy"
        else:
            health_status["adapter_status"] = "degraded"
        
        return health_status


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_aurora_adapter():
        """AuroraQ 어댑터 테스트"""
        
        print("=== AuroraQ 어댑터 테스트 ===")
        
        # Mock 컴포넌트들 (실제 구현에서는 실제 객체 사용)
        keyword_scorer = KeywordScorer()
        
        # 가상의 융합 매니저와 이벤트 감지기 (테스트용)
        fusion_manager = None  # 실제로는 SentimentFusionManager 인스턴스
        event_detector = None  # 실제로는 BigEventDetector 인스턴스
        
        # 어댑터 초기화 (테스트용 - 실제 AuroraQ API 없이)
        adapter = AuroraQAdapter(
            fusion_manager=fusion_manager,
            event_detector=event_detector,
            keyword_scorer=keyword_scorer,
            aurora_api_url="http://localhost:8080",  # 테스트용
            aurora_api_key="test_key"
        )
        
        # 비동기 컨텍스트 매니저 테스트
        async with adapter:
            print("어댑터 초기화 완료")
            
            # 헬스체크 테스트
            health = await adapter.health_check()
            print(f"헬스체크 결과: {health}")
            
            # 통계 조회 테스트
            stats = adapter.get_adapter_stats()
            print(f"어댑터 통계: {stats}")
            
            print("AuroraQ 어댑터 테스트 완료")
    
    # 테스트 실행
    asyncio.run(test_aurora_adapter())