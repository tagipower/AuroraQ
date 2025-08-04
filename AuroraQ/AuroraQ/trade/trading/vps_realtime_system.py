#!/usr/bin/env python3
"""
VPS 실전매매 시스템 (통합 로깅 연동)
AuroraQ Production에서 검증된 실전매매 시스템을 VPS 환경에 최적화
"""

import os
import sys
import json
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from threading import Thread, Event
import websockets
import ccxt.async_support as ccxt
from collections import deque
import psutil
from pathlib import Path

# VPS 환경변수 로더
try:
    # VPS 환경에서 절대 import 시도
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.env_loader import get_vps_env_config, EnvConfig
    ENV_LOADER_AVAILABLE = True
except ImportError:
    try:
        from vps_deployment.config.env_loader import get_vps_env_config, EnvConfig
        ENV_LOADER_AVAILABLE = True
    except ImportError:
        ENV_LOADER_AVAILABLE = False

# VPS 통합 로깅 시스템
try:
    from vps_logging import get_vps_log_integrator, LogCategory, LogLevel
except ImportError:
    try:
        from vps_deployment.vps_logging import get_vps_log_integrator, LogCategory, LogLevel
    except ImportError:
        # Fallback: 더미 로깅 클래스
        class DummyLogIntegrator:
            def get_logger(self, name): return None
            async def log_security_event(self, **kwargs): pass
            async def log_trading_decision(self, **kwargs): pass
            async def log_onnx_inference(self, **kwargs): pass
            async def log_system_metrics(self, **kwargs): pass
        def get_vps_log_integrator(): return DummyLogIntegrator()
        LogCategory, LogLevel = None, None

# VPS 최적화 모듈들
from trading.vps_position_manager import VPSPositionManager, VPSTradingLimits
from trading.vps_order_manager import VPSOrderManager, OrderType
from trading.vps_market_data import VPSMarketDataProvider
from trading.vps_strategy_adapter import VPSStrategyAdapter, EnhancedVPSStrategyAdapter, create_enhanced_vps_strategy_adapter

# ONNX 센티먼트 연동
try:
    from sentiment_service.models.keyword_scorer import analyze_sentiment_unified
    SENTIMENT_AVAILABLE = True
except ImportError:
    try:
        from vps_deployment.sentiment_service.models.keyword_scorer import analyze_sentiment_unified
        SENTIMENT_AVAILABLE = True
    except ImportError:
        try:
            # Sentiment integration 모듈 사용
            from trading.sentiment_integration import get_current_sentiment, get_market_sentiment_state
            async def analyze_sentiment_unified(text, metadata=None):
                sentiment = await get_current_sentiment(metadata.get('symbol', 'BTCUSDT') if metadata else 'BTCUSDT')
                return {
                    'sentiment': sentiment.value,
                    'confidence': sentiment.confidence,
                    'processing_time': 0.1
                }
            SENTIMENT_AVAILABLE = True
        except ImportError:
            SENTIMENT_AVAILABLE = False

@dataclass
class VPSTradingConfig:
    """VPS 실전매매 설정"""
    # 기본 설정
    mode: str = "paper"  # paper, live
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    
    # VPS 최적화 설정 (48GB RAM 기준)
    memory_limit_mb: int = 3072  # 3GB 메모리 제한
    max_positions: int = 5       # 최대 동시 포지션
    update_interval: int = 30    # 30초 간격 업데이트
    
    @classmethod
    def from_env_config(cls, env_config: 'EnvConfig') -> 'VPSTradingConfig':
        """환경변수 설정에서 VPS 거래 설정 생성"""
        # 메모리 제한 파싱 (예: "3G" -> 3072MB)
        memory_str = env_config.vps_memory_limit.upper()
        if memory_str.endswith('G'):
            memory_mb = int(float(memory_str[:-1]) * 1024)
        elif memory_str.endswith('M'):
            memory_mb = int(memory_str[:-1])
        else:
            memory_mb = 3072  # 기본값
        
        return cls(
            mode=env_config.trading_mode,
            symbol=env_config.symbol,
            exchange=env_config.exchange,
            memory_limit_mb=memory_mb,
            max_positions=5,  # 고정값
            update_interval=30  # 고정값
        )
    
    # 실전매매 전략 설정
    rule_strategies: List[str] = field(default_factory=lambda: ["RuleStrategyA"])
    enable_ppo: bool = True
    enable_sentiment: bool = True
    hybrid_mode: str = "ensemble" # ensemble, weighted, sequential
    
    # 리스크 관리
    max_position_size: float = 0.1     # 최대 포지션 크기 (10%)
    emergency_stop_loss: float = 0.05   # 긴급 손절 (5%)
    max_daily_trades: int = 10          # 일일 최대 거래
    risk_tolerance: str = "moderate"    # conservative, moderate, aggressive
    
    # 데이터 설정
    lookback_periods: int = 100
    min_data_points: int = 50
    websocket_timeout: int = 60
    order_timeout: int = 10
    
    # 알림 및 로깅
    enable_notifications: bool = True
    enable_unified_logging: bool = True
    log_trading_decisions: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["console", "file", "telegram"])

class VPSRealtimeSystem:
    """VPS 최적화 실전매매 시스템"""
    
    def __init__(self, config: VPSTradingConfig):
        """
        VPS 실전매매 시스템 초기화
        
        Args:
            config: VPS 거래 설정
        """
        self.config = config
        self.is_running = False
        self.stop_event = Event()
        
        # 환경변수 설정 로딩
        self.env_config = None
        if ENV_LOADER_AVAILABLE:
            try:
                self.env_config = get_vps_env_config()
            except Exception as e:
                print(f"Failed to load environment config: {e}")
                self.env_config = None
        
        # 통합 로깅 시스템
        if config.enable_unified_logging:
            self.log_integrator = get_vps_log_integrator()
            self.logger = self.log_integrator.get_logger("vps_realtime_system")
        else:
            self.log_integrator = None
            self.logger = None
        
        # 핵심 컴포넌트 초기화
        self._initialize_components()
        
        # 데이터 버퍼 (메모리 효율적)
        self.price_buffer = deque(maxlen=config.lookback_periods)
        self.sentiment_buffer = deque(maxlen=100)  # 센티먼트 이력
        
        # 성능 통계
        self.stats = {
            "total_signals": 0,
            "executed_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0,
            "last_update_time": None,
            "memory_usage_mb": 0.0,
            "cpu_usage": 0.0
        }
        
        # 실시간 상태
        self.current_price = 0.0
        self.current_sentiment = 0.0
        self.market_volatility = 0.0
        
        # WebSocket 연결 상태
        self.websocket_connected = False
        self.last_heartbeat = None
    
    def _initialize_components(self):
        """핵심 컴포넌트 초기화"""
        # 거래 한도 설정
        trading_limits = VPSTradingLimits(
            max_position_size=self.config.max_position_size,
            max_daily_trades=self.config.max_daily_trades,
            emergency_stop_loss=self.config.emergency_stop_loss
        )
        
        # 포지션 관리자
        self.position_manager = VPSPositionManager(trading_limits)
        
        # 주문 관리자
        self.order_manager = VPSOrderManager()
        
        # 시장 데이터 제공자
        self.market_data_provider = VPSMarketDataProvider(
            exchange=self.config.exchange,
            mode="simulation" if self.config.mode == "paper" else "live"
        )
        
        # 전략 어댑터 (향상된 Rule 전략 어댑터 사용)
        self.strategy_adapter = None
        self._initialize_strategy_adapter()
        
        # CCXT 거래소 연결 (지연 로딩)
        self.exchange_client = None
    
    def _initialize_strategy_adapter(self):
        """향상된 Rule 전략 어댑터 초기화"""
        try:
            # 알림 설정 구성
            adapter_config = {
                'telegram_enabled': 'telegram' in self.config.notification_channels,
                'websocket_enabled': True,  # WebSocket은 항상 활성화
                'telegram_token': self.env_config.telegram_bot_token if self.env_config else os.getenv('TELEGRAM_BOT_TOKEN'),
                'telegram_chat_id': self.env_config.telegram_chat_id if self.env_config else os.getenv('TELEGRAM_CHAT_ID')
            }
            
            # Enhanced Rule 전략 어댑터 생성
            self.strategy_adapter = create_enhanced_vps_strategy_adapter(adapter_config)
            
            if self.logger:
                self.logger.info(f"Enhanced Rule 전략 어댑터 초기화 완료")
                stats = self.strategy_adapter.get_strategy_statistics()
                self.logger.info(f"Rule 전략 {stats['total_strategies']}개 로드됨")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"전략 어댑터 초기화 실패: {e}")
            
            # 폴백: 기본 Enhanced 어댑터 생성
            self.strategy_adapter = create_enhanced_vps_strategy_adapter()
            
        if self.logger:
            self.logger.info("전략 어댑터 초기화 완료")
    
    def _import_strategy(self, strategy_name: str):
        """전략 클래스 임포트"""
        try:
            # AuroraQ Production 전략들 임포트 (절대 경로 사용)
            auroaq_path = str(Path(__file__).parent.parent.parent / "AuroraQ")
            if auroaq_path not in sys.path:
                sys.path.insert(0, auroaq_path)
            
            if strategy_name == "RuleStrategyA":
                from production.strategies.rule_strategies import RuleStrategyA
                return RuleStrategyA()
            elif strategy_name == "OptimizedRuleStrategyE":
                from production.strategies.optimized_rule_strategy_e import OptimizedRuleStrategyE
                return OptimizedRuleStrategyE()
            else:
                # 동적 임포트 시도
                try:
                    module = __import__(f"production.strategies.{strategy_name.lower()}", fromlist=[strategy_name])
                    strategy_class = getattr(module, strategy_name)
                    return strategy_class()
                except (ImportError, AttributeError):
                    # Rule strategies에서 찾기
                    from production.strategies.rule_strategies import RuleStrategyA
                    return RuleStrategyA()  # Fallback
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Strategy import failed for {strategy_name}: {e}")
            return None
    
    def _load_ppo_agent(self):
        """PPO 에이전트 로드"""
        try:
            # AuroraQ path 확인 및 추가
            auroaq_path = str(Path(__file__).parent.parent.parent / "AuroraQ")
            if auroaq_path not in sys.path:
                sys.path.insert(0, auroaq_path)
            
            from production.strategies.ppo_strategy import PPOStrategy
            return PPOStrategy()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"PPO agent load failed: {e}")
            return None
    
    async def get_strategy_metrics(self) -> Dict[str, Any]:
        """전략 성과 메트릭 반환"""
        if self.strategy_adapter:
            return self.strategy_adapter.get_enhanced_metrics()
        return {}
    
    async def update_strategy_performance(self, strategy_id: str, trade_result: Dict[str, Any]):
        """거래 결과를 바탕으로 전략 성과 업데이트"""
        if self.strategy_adapter:
            await self.strategy_adapter.update_strategy_performance(strategy_id, trade_result)
    
    def add_websocket_client(self, client):
        """전략 알림용 WebSocket 클라이언트 추가"""
        if self.strategy_adapter:
            self.strategy_adapter.add_websocket_client(client)
    
    def remove_websocket_client(self, client):
        """전략 알림용 WebSocket 클라이언트 제거"""
        if self.strategy_adapter:
            self.strategy_adapter.remove_websocket_client(client)
    
    def get_ppo_feedback_data(self) -> List[Dict[str, Any]]:
        """전략 어댑터로부터 PPO 피드백 데이터 반환"""
        if self.strategy_adapter:
            return self.strategy_adapter.get_ppo_feedback_data()
        return []
    
    async def start_trading(self):
        """실전매매 시작"""
        try:
            self.is_running = True
            
            # 시작 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="trading_started",
                    severity="medium",
                    description=f"VPS trading system started: {self.config.mode} mode",
                    symbol=self.config.symbol,
                    strategies=list(self.strategy_adapter.rule_strategies.keys()) if self.strategy_adapter else []
                )
            
            # 거래소 연결
            await self._connect_exchange()
            
            # 백그라운드 작업들 시작
            tasks = [
                asyncio.create_task(self._market_data_loop()),
                asyncio.create_task(self._trading_logic_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._performance_monitoring_loop())
            ]
            
            if self.config.enable_sentiment:
                tasks.append(asyncio.create_task(self._sentiment_analysis_loop()))
            
            # 모든 작업 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Trading system startup failed: {e}")
            
            # 중요 에러는 보안 이벤트로 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="trading_system_failure",
                    severity="critical",
                    description=f"Trading system startup failed: {str(e)}",
                    error_details=str(e)
                )
            raise
    
    async def _connect_exchange(self):
        """거래소 연결"""
        try:
            if self.config.mode == "live" and self.config.exchange == "binance":
                # 실제 거래용 API 키 설정 (환경변수 설정에서)
                api_key = None
                api_secret = None
                
                if self.env_config:
                    api_key = self.env_config.binance_api_key
                    api_secret = self.env_config.binance_api_secret
                else:
                    # Fallback: 직접 환경변수에서
                    api_key = os.environ.get("BINANCE_API_KEY")
                    api_secret = os.environ.get("BINANCE_API_SECRET")
                
                if not api_key or not api_secret:
                    raise ValueError("API credentials not found in environment")
                
                self.exchange_client = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': False,  # 실제 거래
                    'enableRateLimit': True,
                })
            else:
                # 페이퍼 트레이딩 또는 시뮬레이션
                testnet = True
                if self.env_config:
                    testnet = self.env_config.binance_testnet
                    
                self.exchange_client = ccxt.binance({
                    'sandbox': testnet,
                    'enableRateLimit': True,
                })
            
            # 연결 테스트
            await self.exchange_client.load_markets()
            
            if self.logger:
                self.logger.info(f"Exchange connected: {self.config.exchange} ({self.config.mode})")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Exchange connection failed: {e}")
            raise
    
    async def _market_data_loop(self):
        """시장 데이터 수집 루프"""
        while self.is_running:
            try:
                # 현재 가격 데이터 수집
                ticker_data = await self.market_data_provider.get_current_price(self.config.symbol)
                
                if ticker_data and 'price' in ticker_data:
                    self.current_price = float(ticker_data['price'])
                    
                    # 가격 버퍼 업데이트
                    self.price_buffer.append({
                        'timestamp': datetime.now(),
                        'price': self.current_price,
                        'volume': ticker_data.get('volume', 0)
                    })
                    
                    # 변동성 계산
                    if len(self.price_buffer) >= 20:
                        recent_prices = [p['price'] for p in list(self.price_buffer)[-20:]]
                        self.market_volatility = np.std(recent_prices) / np.mean(recent_prices)
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Market data collection error: {e}")
                await asyncio.sleep(5)  # 에러 시 짧은 대기
    
    async def _sentiment_analysis_loop(self):
        """센티먼트 분석 루프"""
        if not SENTIMENT_AVAILABLE:
            if self.logger:
                self.logger.warning("Sentiment analysis not available")
            return
        
        while self.is_running:
            try:
                # 최신 뉴스/소셜 데이터 수집 (간단한 예시)
                news_text = await self._collect_latest_news()
                
                if news_text:
                    # ONNX 센티먼트 분석
                    sentiment_result = await analyze_sentiment_unified(
                        news_text,
                        metadata={"symbol": self.config.symbol, "source": "vps_trading"}
                    )
                    
                    if sentiment_result:
                        self.current_sentiment = sentiment_result.get('sentiment', 0.0)
                        confidence = sentiment_result.get('confidence', 0.0)
                        
                        # 센티먼트 버퍼 업데이트
                        self.sentiment_buffer.append({
                            'timestamp': datetime.now(),
                            'sentiment': self.current_sentiment,
                            'confidence': confidence,
                            'text_length': len(news_text)
                        })
                        
                        # 통합 로깅
                        if self.log_integrator:
                            await self.log_integrator.log_onnx_inference(
                                text=news_text[:500],  # 처음 500자만
                                inference_time=sentiment_result.get('processing_time', 0),
                                confidence=confidence,
                                sentiment_score=self.current_sentiment,
                                component="vps_trading"
                            )
                
                await asyncio.sleep(120)  # 2분마다 센티먼트 분석
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Sentiment analysis error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_latest_news(self) -> str:
        """최신 뉴스 수집 (간단한 구현)"""
        try:
            # 기존 뉴스 수집기 사용 시도
            # from ..sentiment_service.collectors.enhanced_news_collector import collect_bitcoin_news
            # news_items = await collect_bitcoin_news(limit=5)
            news_items = []  # 임시 빈 리스트
            
            if news_items:
                return " ".join([item.get('title', '') + " " + item.get('description', '') 
                                for item in news_items])
            return ""
            
        except Exception:
            # 폴백: 더미 뉴스 (실제로는 외부 API 사용)
            return f"Bitcoin market analysis for {self.config.symbol} at {datetime.now()}"
    
    async def _trading_logic_loop(self):
        """거래 로직 실행 루프"""
        while self.is_running:
            try:
                # 충분한 데이터가 있는지 확인
                if len(self.price_buffer) < self.config.min_data_points:
                    await asyncio.sleep(10)
                    continue
                
                # Rule 전략 어댑터를 사용하여 최적 신호 수집
                final_signal = await self._get_enhanced_trading_signal()
                
                # 거래 실행 결정
                if final_signal and final_signal.get('action') != 'HOLD' and final_signal.get('strength', 0) > 0.5:
                    await self._execute_enhanced_trading_decision(final_signal)
                
                self.stats["total_signals"] += 1
                self.stats["last_update_time"] = datetime.now()
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Trading logic error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_trading_signals(self) -> List[Dict[str, Any]]:
        """전략별 거래 신호 수집"""
        signals = []
        
        # 현재 시장 데이터 준비
        market_data = self._prepare_market_data()
        
        for strategy_name, adapter in self.strategy_adapters.items():
            try:
                # 전략 실행
                signal = await self._get_strategy_signal(adapter, market_data)
                
                if signal:
                    signal['strategy'] = strategy_name
                    signals.append(signal)
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Signal collection failed for {strategy_name}: {e}")
        
        return signals
    
    def _prepare_market_data(self) -> Dict[str, Any]:
        """전략 실행용 시장 데이터 준비"""
        if not self.price_buffer:
            return {}
        
        # 가격 데이터를 DataFrame으로 변환
        price_data = list(self.price_buffer)
        df = pd.DataFrame(price_data)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return {
            'price_data': df,
            'current_price': self.current_price,
            'sentiment': self.current_sentiment,
            'volatility': self.market_volatility,
            'symbol': self.config.symbol
        }
    
    def _prepare_enhanced_market_data(self) -> Dict[str, Any]:
        """향상된 전략 어댑터용 시장 데이터 준비"""
        if not self.price_buffer:
            return {
                'current_price': 0.0,
                'sentiment': 0.0, 
                'volatility': 0.0,
                'symbol': self.config.symbol
            }
        
        # 가격 데이터를 DataFrame으로 변환
        price_data = list(self.price_buffer)
        df = pd.DataFrame(price_data)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Rule 전략 어댑터가 기대하는 OHLCV 형식으로 변환
            if 'close' not in df.columns and 'price' in df.columns:
                df['close'] = df['price']
                df['open'] = df['price'].shift(1).fillna(df['price'])
                df['high'] = df[['open', 'close']].max(axis=1)
                df['low'] = df[['open', 'close']].min(axis=1)
            
            df.set_index('timestamp', inplace=True)
        
        return {
            'price_data': df,
            'current_price': self.current_price,
            'price': self.current_price,  # Rule 전략 어댑터 호환성
            'sentiment': self.current_sentiment,
            'volatility': self.market_volatility,
            'symbol': self.config.symbol,
            'volume': df['volume'].iloc[-1] if len(df) > 0 and 'volume' in df.columns else 0
        }
    
    async def _get_enhanced_trading_signal(self) -> Optional[Dict[str, Any]]:
        """향상된 Rule 전략 어댑터로부터 최적 거래 신호 수집"""
        try:
            if not self.strategy_adapter:
                return None
            
            # 현재 시장 데이터 준비
            market_data = self._prepare_enhanced_market_data()
            
            # Enhanced Rule 전략 어댑터로부터 최적 신호 수집
            signal = await self.strategy_adapter.get_best_trading_signal(market_data)
            
            return signal
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Enhanced trading signal collection failed: {e}")
            return None
    
    async def _execute_enhanced_trading_decision(self, signal: Dict[str, Any]):
        """향상된 거래 결정 실행"""
        try:
            action = signal.get('action', 'HOLD')
            strength = signal.get('strength', 0.0)
            strategy_id = signal.get('strategy_id', 'unknown')
            
            if self.logger:
                self.logger.info(f"거래 신호: {strategy_id} -> {action} (강도: {strength:.3f})")
            
            # 상량한 신호만 실행
            if action == 'BUY' and strength > 0.5:
                await self._execute_buy_order(signal)
            elif action == 'SELL' and strength > 0.5:
                await self._execute_sell_order(signal)
            
            # 거래 결과 로깅
            if self.log_integrator:
                await self.log_integrator.log_trading_decision(
                    action=action,
                    strategy=strategy_id,
                    confidence=signal.get('confidence', 0.0),
                    price=self.current_price,
                    reasoning=signal.get('reasoning', '자동 선택')
                )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Enhanced trading decision execution failed: {e}")
    
    async def _execute_buy_order(self, signal: Dict[str, Any]):
        """매수 주문 실행"""
        try:
            # 포지션 관리자를 통한 매수 로직
            position = await self.position_manager.open_position(
                symbol=self.config.symbol,
                side='long',
                size=signal.get('strength', 0.5) * self.config.max_position_size,
                price=self.current_price
            )
            
            if position:
                self.stats["executed_trades"] += 1
                
                # 전략 성과 업데이트 (진입 시점)
                await self.update_strategy_performance(
                    signal.get('strategy_id', 'unknown'),
                    {'action': 'buy', 'price': self.current_price, 'timestamp': datetime.now()}
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Buy order execution failed: {e}")
    
    async def _execute_sell_order(self, signal: Dict[str, Any]):
        """매도 주문 실행"""
        try:
            # 포지션 관리자를 통한 매도 로직
            positions = await self.position_manager.get_open_positions()
            
            for position in positions:
                if position.symbol == self.config.symbol and position.side == 'long':
                    # 포지션 청산
                    closed_position = await self.position_manager.close_position(
                        position.id, self.current_price
                    )
                    
                    if closed_position:
                        self.stats["executed_trades"] += 1
                        
                        # PnL 계산
                        pnl = (self.current_price - position.entry_price) / position.entry_price
                        
                        if pnl > 0:
                            self.stats["successful_trades"] += 1
                        else:
                            self.stats["failed_trades"] += 1
                        
                        self.stats["total_pnl"] += pnl
                        
                        # 전략 성과 업데이트 (종료 시점)
                        await self.update_strategy_performance(
                            signal.get('strategy_id', 'unknown'),
                            {
                                'action': 'sell',
                                'pnl': pnl,
                                'entry_price': position.entry_price,
                                'exit_price': self.current_price,
                                'duration': datetime.now() - position.open_time,
                                'timestamp': datetime.now()
                            }
                        )
                        
                        break
                        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Sell order execution failed: {e}")
    
    async def _get_strategy_signal(self, adapter: EnhancedVPSStrategyAdapter, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 전략 신호 가져오기"""
        try:
            # Enhanced 전략 어댑터에서 최적 신호 가져오기
            signal = await adapter.get_best_trading_signal(market_data)
            return signal
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Strategy signal error: {e}")
            return None
    
    async def _combine_signals(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """하이브리드 신호 결합"""
        if not signals:
            return None
        
        try:
            if self.config.hybrid_mode == "ensemble":
                # 앙상블 방식: 다수결 + 강도 평균
                buy_votes = sum(1 for s in signals if s.get('action') == 'BUY')
                sell_votes = sum(1 for s in signals if s.get('action') == 'SELL')
                
                if buy_votes > sell_votes:
                    action = "BUY"
                elif sell_votes > buy_votes:
                    action = "SELL"
                else:
                    action = "HOLD"
                
                # 신호 강도 평균
                avg_strength = np.mean([s.get('strength', 0) for s in signals])
                
                return {
                    'action': action,
                    'strength': avg_strength,
                    'confidence': len(signals) / len(self.strategy_adapters),
                    'contributing_strategies': [s['strategy'] for s in signals],
                    'price_target': self.current_price * (1 + avg_strength * 0.02)  # 2% 기준
                }
                
            elif self.config.hybrid_mode == "weighted":
                # 가중 평균 방식 (전략별 성과 기반)
                weights = self._get_strategy_weights()
                weighted_signal = self._calculate_weighted_signal(signals, weights)
                return weighted_signal
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Signal combination error: {e}")
        
        return None
    
    def _get_strategy_weights(self) -> Dict[str, float]:
        """전략별 가중치 (성과 기반)"""
        # 간단한 균등 가중치 (실제로는 성과 히스토리 기반)
        weights = {}
        num_strategies = len(self.strategy_adapters)
        
        for strategy_name in self.strategy_adapters.keys():
            weights[strategy_name] = 1.0 / num_strategies
        
        return weights
    
    def _calculate_weighted_signal(self, signals: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
        """가중 평균 신호 계산"""
        total_weight = 0
        weighted_strength = 0
        
        for signal in signals:
            strategy = signal['strategy']
            weight = weights.get(strategy, 0)
            strength = signal.get('strength', 0)
            
            weighted_strength += strength * weight
            total_weight += weight
        
        if total_weight > 0:
            final_strength = weighted_strength / total_weight
            action = "BUY" if final_strength > 0.1 else "SELL" if final_strength < -0.1 else "HOLD"
            
            return {
                'action': action,
                'strength': final_strength,
                'confidence': total_weight,
                'method': 'weighted_average'
            }
        
        return {'action': 'HOLD', 'strength': 0, 'confidence': 0}
    
    async def _execute_trading_decision(self, signal: Dict[str, Any]):
        """거래 결정 실행"""
        try:
            action = signal['action']
            strength = signal['strength']
            
            # 리스크 체크
            risk_approved = await self._check_trading_risk(signal)
            if not risk_approved:
                return
            
            # 포지션 크기 계산
            position_size = self._calculate_position_size(signal)
            
            # 주문 생성 및 실행
            if action in ["BUY", "SELL"]:
                order_result = await self._place_order(action, position_size, signal)
                
                if order_result['success']:
                    self.stats["executed_trades"] += 1
                    self.stats["successful_trades"] += 1
                    
                    # 거래 결정 로깅 (Tagged 범주 - 중요 이벤트)
                    if self.log_integrator:
                        await self.log_integrator.log_security_event(
                            event_type="trading_decision_executed",
                            severity="medium",
                            description=f"Trading order executed: {action} {position_size} {self.config.symbol}",
                            symbol=self.config.symbol,
                            action=action,
                            size=position_size,
                            price=self.current_price,
                            signal_strength=strength,
                            strategies=signal.get('contributing_strategies', [])
                        )
                else:
                    self.stats["failed_trades"] += 1
                    
                    if self.logger:
                        self.logger.error(f"Order execution failed: {order_result.get('error')}")
                        
        except Exception as e:
            self.stats["failed_trades"] += 1
            
            if self.logger:
                self.logger.error(f"Trading execution error: {e}")
            
            # 실행 실패는 보안 이벤트로 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="trading_execution_failure",
                    severity="high",
                    description=f"Trading execution failed: {str(e)}",
                    error_details=str(e),
                    signal=signal
                )
    
    async def _check_trading_risk(self, signal: Dict[str, Any]) -> bool:
        """거래 리스크 체크"""
        try:
            # 포지션 한도 체크
            if not self.position_manager.can_open_position(
                size=signal.get('strength', 0) * self.config.max_position_size,
                current_price=self.current_price
            )[0]:
                return False
            
            # 변동성 체크
            if self.market_volatility > 0.05:  # 5% 이상 변동성
                if self.logger:
                    self.logger.warning(f"High volatility detected: {self.market_volatility:.4f}")
                return False
            
            # 센티먼트와 신호 일치성 체크
            if self.config.enable_sentiment and SENTIMENT_AVAILABLE:
                sentiment_direction = 1 if self.current_sentiment > 0.1 else -1 if self.current_sentiment < -0.1 else 0
                signal_direction = 1 if signal['action'] == 'BUY' else -1 if signal['action'] == 'SELL' else 0
                
                # 센티먼트와 신호가 반대 방향인 경우 주의
                if sentiment_direction != 0 and signal_direction != 0 and sentiment_direction != signal_direction:
                    if self.logger:
                        self.logger.warning("Sentiment-signal mismatch detected")
                    return signal.get('confidence', 0) > 0.8  # 높은 신뢰도에서만 허용
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Risk check error: {e}")
            return False
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """포지션 크기 계산"""
        base_size = self.config.max_position_size
        strength = abs(signal.get('strength', 0))
        confidence = signal.get('confidence', 0.5)
        
        # 신호 강도와 신뢰도에 따른 크기 조정
        adjusted_size = base_size * strength * confidence
        
        # 최소/최대 제한
        min_size = 0.001  # 최소 0.1%
        max_size = self.config.max_position_size
        
        return max(min_size, min(adjusted_size, max_size))
    
    async def _place_order(self, action: str, size: float, signal: Dict[str, Any]) -> Dict[str, Any]:
        """주문 실행"""
        try:
            if self.config.mode == "paper":
                # 페이퍼 트레이딩
                order_result = await self._simulate_order(action, size, signal)
            else:
                # 실제 주문
                order_result = await self._place_real_order(action, size, signal)
            
            return order_result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _simulate_order(self, action: str, size: float, signal: Dict[str, Any]) -> Dict[str, Any]:
        """시뮬레이션 주문"""
        try:
            # 간단한 페이퍼 트레이딩 시뮬레이션
            simulated_slippage = 0.001  # 0.1% 슬리피지
            execution_price = self.current_price * (1 + simulated_slippage if action == "BUY" else 1 - simulated_slippage)
            
            # 포지션 업데이트
            position_change = size if action == "BUY" else -size
            self.position_manager.update_position(position_change, execution_price)
            
            return {
                "success": True,
                "order_id": f"sim_{int(time.time())}",
                "action": action,
                "size": size,
                "price": execution_price,
                "mode": "simulation"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _place_real_order(self, action: str, size: float, signal: Dict[str, Any]) -> Dict[str, Any]:
        """실제 주문 실행"""
        try:
            if not self.exchange_client:
                raise ValueError("Exchange client not initialized")
            
            # 주문 타입 및 파라미터
            side = 'buy' if action == 'BUY' else 'sell'
            
            # 시장가 주문
            order = await self.exchange_client.create_market_order(
                symbol=self.config.symbol,
                side=side,
                amount=size,
                params={'newClientOrderId': f'aurora_{int(time.time())}'}
            )
            
            return {
                "success": True,
                "order_id": order['id'],
                "action": action,
                "size": size,
                "price": order.get('price', self.current_price),
                "mode": "live"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _risk_monitoring_loop(self):
        """리스크 모니터링 루프"""
        while self.is_running:
            try:
                # 현재 포지션 리스크 체크
                current_risk = self.position_manager.get_portfolio_risk(self.current_price)
                
                # 긴급 손절 체크
                if current_risk > self.config.emergency_stop_loss:
                    await self._emergency_stop_loss()
                
                # 일일 거래 한도 체크
                if self.position_manager.daily_trade_count >= self.config.max_daily_trades:
                    if self.logger:
                        self.logger.warning("Daily trade limit reached")
                
                await asyncio.sleep(10)  # 10초마다 리스크 체크
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _emergency_stop_loss(self):
        """긴급 손절 실행"""
        try:
            if self.logger:
                self.logger.critical("Emergency stop loss triggered!")
            
            # 모든 포지션 청산
            await self.position_manager.close_all_positions(self.current_price)
            
            # 보안 이벤트로 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="emergency_stop_loss",
                    severity="critical",
                    description="Emergency stop loss executed - all positions closed",
                    current_price=self.current_price,
                    portfolio_risk=self.position_manager.get_portfolio_risk(self.current_price)
                )
            
            # 거래 중단
            self.is_running = False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Emergency stop loss error: {e}")
    
    async def _performance_monitoring_loop(self):
        """성능 모니터링 루프"""
        while self.is_running:
            try:
                # 시스템 리소스 모니터링
                process = psutil.Process()
                memory_info = process.memory_info()
                
                self.stats["memory_usage_mb"] = memory_info.rss / 1024 / 1024
                self.stats["cpu_usage"] = process.cpu_percent()
                
                # P&L 계산
                self.stats["total_pnl"] = self.position_manager.get_total_pnl(self.current_price)
                
                # 메모리 제한 체크
                if self.stats["memory_usage_mb"] > self.config.memory_limit_mb:
                    if self.logger:
                        self.logger.warning(f"Memory usage high: {self.stats['memory_usage_mb']:.2f}MB")
                
                # 시스템 메트릭 로깅
                if self.log_integrator:
                    await self.log_integrator.log_system_metrics(
                        component="vps_realtime_system",
                        metrics={
                            "memory_usage_mb": self.stats["memory_usage_mb"],
                            "cpu_usage_percent": self.stats["cpu_usage"],
                            "total_pnl": self.stats["total_pnl"],
                            "active_positions": len(self.position_manager.positions),
                            "total_trades": self.stats["executed_trades"]
                        }
                    )
                
                await asyncio.sleep(60)  # 1분마다 성능 모니터링
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def stop_trading(self):
        """거래 중단"""
        try:
            self.is_running = False
            self.stop_event.set()
            
            # 종료 로깅
            if self.log_integrator:
                await self.log_integrator.log_security_event(
                    event_type="trading_stopped",
                    severity="medium",
                    description="VPS trading system stopped",
                    final_stats=self.stats
                )
            
            # 거래소 연결 종료
            if self.exchange_client:
                await self.exchange_client.close()
            
            if self.logger:
                self.logger.info("VPS trading system stopped")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Trading stop error: {e}")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """거래 통계 반환"""
        stats = self.stats.copy()
        stats.update({
            "current_price": self.current_price,
            "current_sentiment": self.current_sentiment,
            "market_volatility": self.market_volatility,
            "active_positions": len(self.position_manager.positions) if self.position_manager else 0,
            "total_portfolio_value": self.position_manager.get_portfolio_value(self.current_price) if self.position_manager else 0,
            "strategy_count": len(self.strategy_adapters),
            "websocket_connected": self.websocket_connected,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        })
        return stats

# VPS deployment와의 통합을 위한 팩토리 함수
def create_vps_trading_system(config: Optional[VPSTradingConfig] = None) -> VPSRealtimeSystem:
    """VPS 최적화된 실전매매 시스템 생성"""
    if config is None:
        config = VPSTradingConfig()
    
    return VPSRealtimeSystem(config)

if __name__ == "__main__":
    # 테스트 실행
    async def test_vps_trading():
        config = VPSTradingConfig(
            mode="paper",
            symbol="BTCUSDT",
            enable_sentiment=True,
            enable_unified_logging=True
        )
        
        trading_system = create_vps_trading_system(config)
        
        try:
            # 짧은 테스트 (30초)
            task = asyncio.create_task(trading_system.start_trading())
            await asyncio.sleep(30)
            
            await trading_system.stop_trading()
            
            stats = trading_system.get_trading_stats()
            print("Trading stats:", stats)
            
        except KeyboardInterrupt:
            await trading_system.stop_trading()
    
    asyncio.run(test_vps_trading())