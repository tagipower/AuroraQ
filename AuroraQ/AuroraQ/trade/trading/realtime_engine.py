"""
AuroraQ 실시간 매매 엔진
ONNX 센티먼트 분석 + PPO 에이전트 + 룰 기반 매매

48GB VPS 최적화:
- 메모리 효율적 처리 (2-3GB 사용)
- 실시간 WebSocket 연동
- 고품질 로깅 시스템
- 11패널 대시보드 연동
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import aiohttp
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import websockets
import ccxt.async_support as ccxt
from collections import deque, defaultdict
import psutil
import threading
# from concurrent.futures import ThreadPoolExecutor  # 사용하지 않음

# Sentiment Integration
try:
    from trading.sentiment_integration import (
        get_sentiment_client, SentimentScore, MarketSentiment,
        get_current_sentiment, get_market_sentiment_state
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment integration not available")

# VPS 최적화 설정
VPS_TRADING_CONFIG = {
    "memory_limit_mb": 3072,  # 3GB 메모리 제한
    "max_positions": 5,       # 최대 동시 포지션
    "update_interval": 30,    # 30초 간격 업데이트
    "websocket_timeout": 60,  # WebSocket 타임아웃
    "order_timeout": 10,      # 주문 타임아웃
    "risk_check_interval": 10, # 10초 간격 리스크 체크
    "sentiment_update_interval": 300,  # 5분 간격 감정 업데이트
    "sentiment_weight": 0.15   # 감정 가중치 (15%)
}

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MarketData:
    """시장 데이터 구조"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    spread: float
    news_sentiment: float = 0.0
    technical_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradingSignal:
    """매매 신호 구조"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    source: str  # PPO, RULE, SENTIMENT
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    last_update: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingStats:
    """매매 통계"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    last_trade_time: Optional[datetime] = None
    
    # 감정 분석 관련 통계
    sentiment_signals: int = 0
    sentiment_accuracy: float = 0.0
    avg_sentiment_score: float = 0.0

@dataclass
class SentimentSignal:
    """감정 분석 신호"""
    symbol: str
    score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    recommendation: str  # BUY, SELL, HOLD
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SentimentIntegrationManager:
    """감정 분석 통합 관리자"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.sentiment_client = get_sentiment_client() if SENTIMENT_AVAILABLE else None
        self.sentiment_cache = {}
        self.sentiment_history = deque(maxlen=100)
        self.last_update = {}
        self.logger = logging.getLogger("SentimentManager")
        
        # 감정 신호 설정
        self.sentiment_thresholds = {
            'strong_buy': 0.6,
            'buy': 0.3,
            'neutral': 0.1,
            'sell': -0.3,
            'strong_sell': -0.6
        }
        
        # 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'avg_sentiment': 0.0,
            'last_error': None
        }
    
    async def get_sentiment_signal(self, symbol: str, force_update: bool = False) -> Optional[SentimentSignal]:
        """감정 분석 신호 조회"""
        if not SENTIMENT_AVAILABLE or not self.sentiment_client:
            return None
        
        current_time = datetime.now()
        cache_key = symbol
        
        # 캐시 확인 (5분 TTL)
        if (not force_update and 
            cache_key in self.sentiment_cache and
            cache_key in self.last_update and
            (current_time - self.last_update[cache_key]).seconds < 300):
            
            self.stats['cache_hits'] += 1
            return self.sentiment_cache[cache_key]
        
        try:
            self.stats['total_requests'] += 1
            
            # 감정 점수 조회
            sentiment_score = await get_current_sentiment(symbol)
            
            # 감정 신호 생성
            recommendation = self._score_to_recommendation(sentiment_score.weighted_score)
            
            signal = SentimentSignal(
                symbol=symbol,
                score=sentiment_score.weighted_score,
                confidence=sentiment_score.confidence,
                recommendation=recommendation,
                timestamp=current_time,
                metadata={
                    'raw_score': sentiment_score.value,
                    'source': sentiment_score.source,
                    'normalized': sentiment_score.normalized_score
                }
            )
            
            # 캐시 업데이트
            self.sentiment_cache[cache_key] = signal
            self.last_update[cache_key] = current_time
            self.sentiment_history.append(signal)
            
            self.stats['successful_requests'] += 1
            self._update_avg_sentiment(signal.score)
            
            self.logger.debug(f"감정 신호 업데이트: {symbol} -> {recommendation} ({signal.score:.3f})")
            return signal
            
        except Exception as e:
            self.stats['last_error'] = str(e)
            self.logger.warning(f"감정 신호 조회 실패: {e}")
            
            # 캐시된 신호 반환 (있는 경우)
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            return None
    
    def _score_to_recommendation(self, score: float) -> str:
        """감정 점수를 추천 액션으로 변환"""
        if score >= self.sentiment_thresholds['strong_buy']:
            return "STRONG_BUY"
        elif score >= self.sentiment_thresholds['buy']:
            return "BUY"
        elif score <= self.sentiment_thresholds['strong_sell']:
            return "STRONG_SELL"
        elif score <= self.sentiment_thresholds['sell']:
            return "SELL"
        else:
            return "HOLD"
    
    def _update_avg_sentiment(self, new_score: float):
        """평균 감정 점수 업데이트"""
        alpha = 0.1  # 지수 이동 평균 계수
        self.stats['avg_sentiment'] = (
            alpha * new_score + (1 - alpha) * self.stats['avg_sentiment']
        )
    
    async def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """시장 전체 감정 요약"""
        if not SENTIMENT_AVAILABLE or not self.sentiment_client:
            return {'available': False}
        
        try:
            # 모든 심볼에 대한 감정 수집
            sentiment_signals = []
            
            for symbol in self.symbols:
                signal = await self.get_sentiment_signal(symbol)
                if signal:
                    sentiment_signals.append(signal)
            
            if not sentiment_signals:
                return {'available': True, 'signals': 0}
            
            # 요약 통계 계산
            scores = [s.score for s in sentiment_signals]
            recommendations = [s.recommendation for s in sentiment_signals]
            
            summary = {
                'available': True,
                'signals_count': len(sentiment_signals),
                'avg_score': np.mean(scores),
                'median_score': np.median(scores),
                'score_std': np.std(scores),
                'recommendations_summary': {
                    rec: recommendations.count(rec) for rec in set(recommendations)
                },
                'last_update': max(s.timestamp for s in sentiment_signals).isoformat(),
                'stats': self.get_stats()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"시장 감정 요약 실패: {e}")
            return {'available': True, 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """감정 분석 통계 조회"""
        success_rate = (
            self.stats['successful_requests'] / max(1, self.stats['total_requests']) * 100
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / max(1, self.stats['total_requests']) * 100
        )
        
        return {
            **self.stats,
            'success_rate_percent': success_rate,
            'cache_hit_rate_percent': cache_hit_rate,
            'cache_size': len(self.sentiment_cache),
            'history_size': len(self.sentiment_history)
        }

class MarketDataHandler:
    """실시간 시장 데이터 핸들러"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_buffer = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.latest_data = {}
        self.subscribers = []
        self.is_running = False
        self.exchange = None
        self.logger = logging.getLogger("MarketDataHandler")
        
        # 감정 분석 통합
        self.sentiment_manager = SentimentIntegrationManager(symbols) if SENTIMENT_AVAILABLE else None
        
    async def start(self):
        """시장 데이터 스트리밍 시작"""
        self.is_running = True
        
        # Binance 연결 설정
        self.exchange = ccxt.binance({
            'apiKey': '',  # 환경변수에서 읽기
            'secret': '',
            'sandbox': True,  # 테스트넷 사용
            'enableRateLimit': True,
        })
        
        # WebSocket 연결
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self._stream_symbol_data(symbol))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
    
    async def _stream_symbol_data(self, symbol: str):
        """개별 심볼 데이터 스트리밍"""
        try:
            while self.is_running:
                # REST API로 데이터 조회 (WebSocket 대신 간단히)
                ticker = await self.exchange.fetch_ticker(symbol)
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=1)
                
                if ohlcv:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ohlcv[0][0] / 1000),
                        open=ohlcv[0][1],
                        high=ohlcv[0][2],
                        low=ohlcv[0][3],
                        close=ohlcv[0][4],
                        volume=ohlcv[0][5],
                        bid=ticker['bid'],
                        ask=ticker['ask'],
                        spread=(ticker['ask'] - ticker['bid']) / ticker['bid']
                    )
                    
                    # 데이터 저장
                    self.data_buffer[symbol].append(market_data)
                    self.latest_data[symbol] = market_data
                    
                    # 구독자들에게 브로드캐스트
                    await self._notify_subscribers(market_data)
                
                await asyncio.sleep(1)  # 1초 간격
                
        except Exception as e:
            self.logger.error(f"Market data streaming error for {symbol}: {e}")
    
    async def _notify_subscribers(self, data: MarketData):
        """구독자들에게 데이터 전송"""
        for callback in self.subscribers:
            try:
                await callback(data)
            except Exception as e:
                self.logger.error(f"Subscriber notification error: {e}")
    
    def subscribe(self, callback):
        """데이터 구독"""
        self.subscribers.append(callback)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """최신 데이터 조회"""
        return self.latest_data.get(symbol)
    
    def get_ohlcv_history(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """OHLCV 히스토리 조회"""
        buffer = self.data_buffer.get(symbol, deque())
        return list(buffer)[-limit:]

class SentimentAnalyzer:
    """ONNX 센티먼트 분석기 클라이언트"""
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5분 캐시
        self.logger = logging.getLogger("SentimentAnalyzer")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_market_sentiment(self, symbol: str) -> float:
        """시장 센티먼트 분석"""
        cache_key = f"sentiment_{symbol}"
        
        # 캐시 확인
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # VPS 메트릭 API에서 센티먼트 조회
            async with self.session.get(f"{self.api_url}/metrics/dashboard") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # 센티먼트 융합 패널에서 데이터 추출
                    sentiment_panel = data.get("panels", {}).get("1_sentiment_fusion", {})
                    fusion_score = sentiment_panel.get("fusion_score", 0.0)
                    
                    # 캐시 저장
                    self.cache[cache_key] = (fusion_score, time.time())
                    
                    return fusion_score
                    
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
        
        return 0.0  # 중립

class PPOAgent:
    """PPO 기반 강화학습 에이전트 (경량화 버전)"""
    
    def __init__(self):
        self.model_loaded = False
        self.feature_scaler = None
        self.action_space = [-1, 0, 1]  # SELL, HOLD, BUY
        self.logger = logging.getLogger("PPOAgent")
        
        # 간단한 규칙 기반 모델 (PPO 모델 대신)
        self.technical_weights = {
            'rsi': -0.02,      # RSI 높으면 매도 신호
            'macd': 0.15,      # MACD 양수면 매수 신호
            'bb_position': 0.1, # 볼린저 밴드 위치
            'volume_ratio': 0.05,
            'price_momentum': 0.2
        }
    
    def predict(self, market_data: MarketData, sentiment: float) -> TradingSignal:
        """매매 신호 예측"""
        try:
            # 기술적 지표 계산
            features = self._extract_features(market_data)
            
            # 간단한 스코어링 모델
            score = 0.0
            
            # 센티먼트 가중치
            score += sentiment * 0.3
            
            # 기술적 지표 가중치
            for indicator, weight in self.technical_weights.items():
                if indicator in features:
                    score += features[indicator] * weight
            
            # 신호 결정
            if score > 0.15:
                action = "BUY"
                confidence = min(0.9, abs(score))
            elif score < -0.15:
                action = "SELL"
                confidence = min(0.9, abs(score))
            else:
                action = "HOLD"
                confidence = 0.5
            
            return TradingSignal(
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                source="PPO",
                reasoning=f"Score: {score:.3f}, Sentiment: {sentiment:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"PPO prediction error: {e}")
            return TradingSignal(
                symbol=market_data.symbol,
                action="HOLD",
                confidence=0.0,
                source="PPO_ERROR",
                reasoning=str(e)
            )
    
    def _extract_features(self, market_data: MarketData) -> Dict[str, float]:
        """기술적 지표 추출"""
        features = {}
        
        # 간단한 모멘텀
        if market_data.close > 0 and market_data.open > 0:
            features['price_momentum'] = (market_data.close - market_data.open) / market_data.open
        
        # 스프레드 비율
        features['spread_ratio'] = market_data.spread
        
        # 더미 RSI (실제로는 히스토리컬 데이터 필요)
        features['rsi'] = 50.0 + np.random.normal(0, 10)
        features['macd'] = np.random.normal(0, 0.1)
        features['bb_position'] = np.random.uniform(0, 1)
        features['volume_ratio'] = 1.0
        
        return features

class RuleBasedEngine:
    """룰 기반 매매 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger("RuleBasedEngine")
        
        # 매매 규칙 설정
        self.rules = {
            'max_position_size': 0.1,     # 포트폴리오의 10%
            'stop_loss_pct': 0.02,        # 2% 손절
            'take_profit_pct': 0.06,      # 6% 익절
            'min_confidence': 0.6,         # 최소 신뢰도
            'max_spread': 0.001,          # 최대 스프레드 0.1%
            'min_volume': 1000000,        # 최소 거래량
        }
    
    def evaluate(self, market_data: MarketData, sentiment: float, 
                ppo_signal: TradingSignal, portfolio_state: Dict) -> TradingSignal:
        """PPO 신호를 룰 기반으로 검증"""
        
        # 기본적으로 PPO 신호 사용
        final_signal = ppo_signal
        
        try:
            # 1. 신뢰도 체크
            if ppo_signal.confidence < self.rules['min_confidence']:
                final_signal.action = "HOLD"
                final_signal.reasoning += " | Low confidence"
            
            # 2. 스프레드 체크
            if market_data.spread > self.rules['max_spread']:
                final_signal.action = "HOLD"
                final_signal.reasoning += " | High spread"
            
            # 3. 거래량 체크
            if market_data.volume < self.rules['min_volume']:
                final_signal.action = "HOLD"
                final_signal.reasoning += " | Low volume"
            
            # 4. 포지션 크기 제한
            current_exposure = portfolio_state.get('exposure', {}).get(market_data.symbol, 0.0)
            max_size = self.rules['max_position_size']
            
            if abs(current_exposure) >= max_size and final_signal.action in ['BUY', 'SELL']:
                final_signal.action = "HOLD"
                final_signal.reasoning += " | Max position reached"
            
            # 5. 손절/익절 가격 설정
            if final_signal.action == "BUY":
                final_signal.stop_loss = market_data.close * (1 - self.rules['stop_loss_pct'])
                final_signal.take_profit = market_data.close * (1 + self.rules['take_profit_pct'])
                final_signal.position_size = min(max_size - current_exposure, 0.05)
                
            elif final_signal.action == "SELL":
                final_signal.stop_loss = market_data.close * (1 + self.rules['stop_loss_pct'])
                final_signal.take_profit = market_data.close * (1 - self.rules['take_profit_pct'])
                final_signal.position_size = min(max_size + current_exposure, 0.05)
            
            final_signal.source = "RULE_VERIFIED"
            
        except Exception as e:
            self.logger.error(f"Rule evaluation error: {e}")
            final_signal.action = "HOLD"
            final_signal.reasoning += f" | Rule error: {str(e)}"
        
        return final_signal

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, max_daily_loss: float = 0.02):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.logger = logging.getLogger("RiskManager")
    
    def validate_signal(self, signal: TradingSignal, portfolio_state: Dict) -> TradingSignal:
        """신호 검증 및 리스크 체크"""
        
        # 일일 손실 리셋
        now = datetime.now()
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_pnl = 0.0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
        
        # 현재 일일 손실 업데이트
        self.daily_pnl = portfolio_state.get('daily_pnl', 0.0)
        
        # 일일 손실 한도 체크
        if self.daily_pnl <= -self.max_daily_loss:
            signal.action = "HOLD"
            signal.reasoning += " | Daily loss limit reached"
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2%}")
        
        # 포트폴리오 전체 리스크 체크
        total_exposure = sum(abs(pos) for pos in portfolio_state.get('exposure', {}).values())
        if total_exposure > 0.5:  # 50% 이상 노출 금지
            signal.action = "HOLD"
            signal.reasoning += " | Portfolio overexposed"
        
        return signal

class OrderExecutor:
    """주문 실행기"""
    
    def __init__(self):
        self.exchange = None
        self.logger = logging.getLogger("OrderExecutor")
        self.is_paper_trading = True  # 페이퍼 트레이딩 모드
        
    async def initialize(self):
        """거래소 초기화"""
        if not self.is_paper_trading:
            self.exchange = ccxt.binance({
                'apiKey': '',  # 환경변수에서 읽기
                'secret': '',
                'sandbox': True,
                'enableRateLimit': True,
            })
    
    async def execute_order(self, signal: TradingSignal) -> Dict[str, Any]:
        """주문 실행"""
        order_result = {
            'success': False,
            'order_id': None,
            'executed_price': 0.0,
            'executed_quantity': 0.0,
            'timestamp': datetime.now(),
            'error': None
        }
        
        try:
            if self.is_paper_trading:
                # 페이퍼 트레이딩 시뮬레이션
                order_result.update({
                    'success': True,
                    'order_id': f"paper_{int(time.time())}",
                    'executed_price': signal.price_target or 45000,  # 더미 가격
                    'executed_quantity': signal.position_size,
                    'fees': signal.position_size * 0.001,  # 0.1% 수수료
                })
                
                self.logger.info(f"Paper order executed: {signal.action} {signal.symbol}")
                
            else:
                # 실제 주문 실행 (구현 필요)
                pass
                
        except Exception as e:
            order_result['error'] = str(e)
            self.logger.error(f"Order execution error: {e}")
        
        return order_result

class PositionManager:
    """포지션 관리자"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.trades_history = []
        self.logger = logging.getLogger("PositionManager")
    
    def update_position(self, symbol: str, order_result: Dict[str, Any], 
                       signal: TradingSignal, current_price: float):
        """포지션 업데이트"""
        
        if not order_result['success']:
            return
        
        now = datetime.now()
        
        if symbol not in self.positions:
            # 새 포지션 생성
            side = PositionSide.LONG if signal.action == "BUY" else PositionSide.SHORT
            
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                size=order_result['executed_quantity'],
                entry_price=order_result['executed_price'],
                current_price=current_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=now,
                last_update=now,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
        else:
            # 기존 포지션 업데이트
            position = self.positions[symbol]
            
            if (position.side == PositionSide.LONG and signal.action == "SELL") or \
               (position.side == PositionSide.SHORT and signal.action == "BUY"):
                # 포지션 청산
                pnl = self._calculate_pnl(position, order_result['executed_price'])
                position.realized_pnl += pnl
                
                self.trades_history.append({
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'exit_price': order_result['executed_price'],
                    'pnl': pnl,
                    'duration': now - position.entry_time,
                    'timestamp': now
                })
                
                # 포지션 제거 또는 크기 조정
                if order_result['executed_quantity'] >= position.size:
                    del self.positions[symbol]
                else:
                    position.size -= order_result['executed_quantity']
                    position.last_update = now
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """미실현 손익 업데이트"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.unrealized_pnl = self._calculate_pnl(position, current_price)
            position.last_update = datetime.now()
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """손익 계산"""
        if position.side == PositionSide.LONG:
            return (current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - current_price) * position.size
    
    def get_all_positions(self) -> Dict[str, Position]:
        """모든 포지션 조회"""
        return self.positions.copy()
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """포트폴리오 통계"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        
        exposure = {}
        for symbol, position in self.positions.items():
            value = position.size * position.current_price
            exposure[symbol] = value
        
        return {
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'total_pnl': total_unrealized + total_realized,
            'active_positions': len(self.positions),
            'exposure': exposure,
            'daily_pnl': total_realized  # 단순화
        }

class AuroraQTradingEngine:
    """AuroraQ 메인 매매 엔진"""
    
    def __init__(self, symbols: List[str] = ["BTC/USDT"]):
        self.symbols = symbols
        self.is_running = False
        
        # 컴포넌트 초기화
        self.market_data_handler = MarketDataHandler(symbols)
        self.sentiment_analyzer = None
        self.ppo_agent = PPOAgent()
        self.rule_engine = RuleBasedEngine()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.position_manager = PositionManager()
        
        # 성능 모니터링
        self.stats = TradingStats()
        self.logger = logging.getLogger("TradingEngine")
        
        # WebSocket 서버 (대시보드 연동용)
        self.websocket_clients = set()
        
    async def start(self):
        """매매 엔진 시작"""
        self.is_running = True
        self.logger.info("AuroraQ Trading Engine starting...")
        
        # 컴포넌트 초기화
        await self.order_executor.initialize()
        
        # 센티먼트 분석기 초기화
        self.sentiment_analyzer = SentimentAnalyzer()
        await self.sentiment_analyzer.__aenter__()
        
        # 시장 데이터 핸들러 구독
        self.market_data_handler.subscribe(self._on_market_data)
        
        # 백그라운드 태스크들 시작
        tasks = [
            asyncio.create_task(self.market_data_handler.start()),
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._websocket_server())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Trading engine stopped by user")
        finally:
            await self.stop()
    
    async def stop(self):
        """매매 엔진 중지"""
        self.is_running = False
        if self.sentiment_analyzer:
            await self.sentiment_analyzer.__aexit__(None, None, None)
        self.logger.info("AuroraQ Trading Engine stopped")
    
    async def _on_market_data(self, data: MarketData):
        """시장 데이터 수신 시 호출"""
        # 미실현 손익 업데이트
        self.position_manager.update_unrealized_pnl(data.symbol, data.close)
        
        # WebSocket 클라이언트들에게 브로드캐스트
        await self._broadcast_market_data(data)
    
    async def _trading_loop(self):
        """메인 매매 루프"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # 최신 시장 데이터 조회
                    market_data = self.market_data_handler.get_latest_data(symbol)
                    if not market_data:
                        continue
                    
                    # 센티먼트 분석
                    sentiment = await self.sentiment_analyzer.analyze_market_sentiment(symbol)
                    market_data.news_sentiment = sentiment
                    
                    # PPO 에이전트 예측
                    ppo_signal = self.ppo_agent.predict(market_data, sentiment)
                    
                    # 룰 기반 검증
                    portfolio_state = self.position_manager.get_portfolio_stats()
                    rule_signal = self.rule_engine.evaluate(
                        market_data, sentiment, ppo_signal, portfolio_state
                    )
                    
                    # 리스크 관리 검증
                    final_signal = self.risk_manager.validate_signal(rule_signal, portfolio_state)
                    
                    # 주문 실행
                    if final_signal.action != "HOLD":
                        order_result = await self.order_executor.execute_order(final_signal)
                        
                        # 포지션 업데이트
                        if order_result['success']:
                            self.position_manager.update_position(
                                symbol, order_result, final_signal, market_data.close
                            )
                            
                            # 통계 업데이트
                            self._update_stats(order_result, final_signal)
                    
                    # 트레이딩 신호 로깅
                    await self._log_trading_decision({
                        'symbol': symbol,
                        'market_data': market_data,
                        'sentiment': sentiment,
                        'ppo_signal': ppo_signal,
                        'rule_signal': rule_signal,
                        'final_signal': final_signal,
                        'portfolio_stats': portfolio_state
                    })
                
                await asyncio.sleep(VPS_TRADING_CONFIG["update_interval"])
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 포지션 리스크 체크
                positions = self.position_manager.get_all_positions()
                
                for symbol, position in positions.items():
                    # 손절/익절 체크
                    current_price = position.current_price
                    
                    should_close = False
                    close_reason = ""
                    
                    if position.side == PositionSide.LONG:
                        if position.stop_loss and current_price <= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif position.take_profit and current_price >= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    
                    elif position.side == PositionSide.SHORT:
                        if position.stop_loss and current_price >= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif position.take_profit and current_price <= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    
                    if should_close:
                        # 강제 청산 신호 생성
                        close_signal = TradingSignal(
                            symbol=symbol,
                            action="SELL" if position.side == PositionSide.LONG else "BUY",
                            confidence=1.0,
                            source="RISK_MANAGEMENT",
                            position_size=position.size,
                            reasoning=f"Auto close: {close_reason}"
                        )
                        
                        # 주문 실행
                        order_result = await self.order_executor.execute_order(close_signal)
                        if order_result['success']:
                            self.position_manager.update_position(
                                symbol, order_result, close_signal, current_price
                            )
                            
                            self.logger.info(f"Position auto-closed: {symbol} - {close_reason}")
                
                # 시스템 리소스 모니터링
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if memory_usage > VPS_TRADING_CONFIG["memory_limit_mb"]:
                    self.logger.warning(f"High memory usage: {memory_usage:.1f}MB")
                
                await asyncio.sleep(VPS_TRADING_CONFIG["risk_check_interval"])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _websocket_server(self):
        """WebSocket 서버 (대시보드 연동)"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                async for message in websocket:
                    # 클라이언트로부터 메시지 처리
                    pass
            except Exception as e:
                self.logger.error(f"WebSocket client error: {e}")
            finally:
                self.websocket_clients.remove(websocket)
        
        try:
            server = await websockets.serve(handle_client, "localhost", 8002)
            self.logger.info("WebSocket server started on ws://localhost:8002")
            await server.wait_closed()
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
    
    async def _broadcast_market_data(self, data: MarketData):
        """시장 데이터 브로드캐스트"""
        if self.websocket_clients:
            message = {
                'type': 'market_data',
                'symbol': data.symbol,
                'price': data.close,
                'volume': data.volume,
                'sentiment': data.news_sentiment,
                'timestamp': data.timestamp.isoformat()
            }
            
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except:
                    disconnected.add(client)
            
            # 연결이 끊어진 클라이언트 제거
            self.websocket_clients -= disconnected
    
    def _update_stats(self, order_result: Dict[str, Any], signal: TradingSignal):
        """통계 업데이트"""
        self.stats.total_trades += 1
        self.stats.last_trade_time = datetime.now()
        
        # 추가 통계 계산은 포지션 청산 시 수행
    
    async def _log_trading_decision(self, decision_data: Dict[str, Any]):
        """트레이딩 결정 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': decision_data['symbol'],
            'market_price': decision_data['market_data'].close,
            'sentiment': decision_data['sentiment'],
            'ppo_action': decision_data['ppo_signal'].action,
            'ppo_confidence': decision_data['ppo_signal'].confidence,
            'rule_action': decision_data['rule_signal'].action,
            'final_action': decision_data['final_signal'].action,
            'reasoning': decision_data['final_signal'].reasoning,
            'portfolio_pnl': decision_data['portfolio_stats'].get('total_pnl', 0.0)
        }
        
        # Raw 로그에 기록
        self.logger.info(f"TRADING_DECISION: {json.dumps(log_entry)}")
    
    def get_realtime_status(self) -> Dict[str, Any]:
        """실시간 상태 조회 (11패널 대시보드용)"""
        positions = self.position_manager.get_all_positions()
        portfolio_stats = self.position_manager.get_portfolio_stats()
        
        return {
            'is_running': self.is_running,
            'active_positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'pnl_pct': (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
                }
                for pos in positions.values()
            ],
            'portfolio_stats': portfolio_stats,
            'trading_stats': {
                'total_trades': self.stats.total_trades,
                'win_rate': self.stats.win_rate,
                'total_pnl': portfolio_stats['total_pnl'],
                'max_drawdown': self.stats.max_drawdown
            },
            'system_status': {
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'uptime': datetime.now().isoformat(),
                'websocket_clients': len(self.websocket_clients)
            },
            'latest_signals': []  # 최근 신호들
        }

# 메인 실행 함수
async def main():
    """메인 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/opt/auroraq/logs/raw/trading/trading_engine.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("Main")
    logger.info("Starting AuroraQ Trading Engine v2.0")
    
    # 매매 엔진 시작
    symbols = ["BTC/USDT", "ETH/USDT"]
    engine = AuroraQTradingEngine(symbols)
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Shutting down trading engine...")
    except Exception as e:
        logger.error(f"Engine error: {e}")
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())