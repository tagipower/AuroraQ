#!/usr/bin/env python3
"""
VPS PPO Strategy - 독립적인 PPO 강화학습 전략
선택적 의존성으로 VPS deployment에서 안전하게 동작
"""


# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import sys
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

# Sentiment Integration
try:
    from trading.sentiment_integration import get_sentiment_client, SentimentScore, MarketSentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    # Fallback classes
    class SentimentScore:
        def __init__(self):
            self.value = 0.0
            self.confidence = 0.0
            self.weighted_score = 0.0
        
    class MarketSentiment:
        def to_feature_vector(self):
            return [0.0] * 6
    
    def get_sentiment_client():
        return None

# BaseRuleStrategy import (같은 디렉토리)
try:
    from trading.rule_strategies import BaseRuleStrategy
except ImportError:
    # 독립 실행 시
    sys.path.append(os.path.dirname(__file__)) 
    from rule_strategies import BaseRuleStrategy

# PPO Agent and Trainer modules
PPO_MODULES_AVAILABLE = False
try:
    from ppo_agent import PPOAgent, PPOAgentConfig, ActionResult
    from ppo_trainer import PPOTrainer, TrainingConfig
    PPO_MODULES_AVAILABLE = True
except ImportError:
    try:
        from trading.ppo_agent import PPOAgent, PPOAgentConfig, ActionResult
        from trading.ppo_trainer import PPOTrainer, TrainingConfig
        PPO_MODULES_AVAILABLE = True
    except ImportError:
        PPO_MODULES_AVAILABLE = False

# 선택적 PPO 의존성
PPO_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    PPO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("PPO dependencies available")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"PPO dependencies not available: {e}")

@dataclass
class PPOConfig:
    """PPO 설정"""
    model_path: str = os.getenv('PPO_MODEL_PATH', '/app/models/ppo_model.zip')
    confidence_threshold: float = 0.7
    max_positions: int = 2
    state_features: int = 26  # 기존 20 + 감정 특성 6개
    action_space_size: int = 3  # BUY, SELL, HOLD
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    
    # P1-3: 동적 배치 크기 조정 설정
    enable_dynamic_batching: bool = True
    min_batch_size: int = 16
    max_batch_size: int = 256
    target_processing_time_s: float = 2.0
    target_memory_mb: float = 1000.0

class PPOStrategy(BaseRuleStrategy):
    """VPS용 PPO 강화학습 전략"""
    
    def __init__(self, config: PPOConfig = None):
        super().__init__(name="PPOStrategy")
        
        self.config = config or PPOConfig()
        self.model = None
        self.model_loaded = False
        self.last_prediction = None
        self.prediction_cache = {}
        
        # PPO 전용 성과 추적
        self.ppo_predictions = 0
        self.ppo_successes = 0
        self.model_confidence_history = []
        
        # BaseRuleStrategy 호환성을 위한 속성들
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.profit_factor = 1.0
        
        # 상태 전처리 설정 (감정 특성 포함)
        self.feature_columns = [
            'close', 'volume', 'rsi_14', 'macd', 'bb_upper', 'bb_lower',
            'ema_12', 'ema_26', 'adx', 'stoch_k', 'stoch_d',
            # 감정 분석 특성
            'sentiment_overall', 'sentiment_fear_greed', 'sentiment_volatility',
            'sentiment_trend', 'sentiment_social', 'sentiment_news'
        ]
        
        # PPO Agent 초기화
        if PPO_MODULES_AVAILABLE:
            agent_config = PPOAgentConfig(
                model_path=self.config.model_path,
                confidence_threshold=self.config.confidence_threshold,
                state_features=self.config.state_features
            )
            self.agent = PPOAgent(agent_config)
            self.model_loaded = self.agent.is_ready()
        else:
            self.agent = None
            self.model_loaded = False

        # 기존 호환성을 위한 속성들
        self.model = self.agent.model if self.agent else None

        # Trainer 초기화 (필요 시)
        self.trainer = None
        if PPO_MODULES_AVAILABLE:
            training_config = TrainingConfig(
                state_features=self.config.state_features,
                model_save_dir=str(Path(self.config.model_path).parent)
            )
            self.trainer = PPOTrainer(training_config)

        # 감정 분석 클라이언트
        self.sentiment_client = get_sentiment_client() if SENTIMENT_AVAILABLE else None
        self.sentiment_cache = {}
        self.last_sentiment_update = 0
        
        # P1-3: 동적 배치 크기 관리자 초기화
        self.batch_manager = None
        if self.config.enable_dynamic_batching:
            try:
                from core.performance.dynamic_batch_manager import DynamicBatchManager, BatchConfig
                batch_config = BatchConfig(
                    initial_batch_size=self.config.batch_size,
                    min_batch_size=self.config.min_batch_size,
                    max_batch_size=self.config.max_batch_size,
                    target_processing_time_s=self.config.target_processing_time_s,
                    target_memory_mb=self.config.target_memory_mb
                )
                self.batch_manager = DynamicBatchManager(batch_config)
                logger.info(f"Dynamic batch manager initialized: {self.config.batch_size} -> {self.batch_manager.get_current_batch_size()}")
            except ImportError as e:
                logger.warning(f"Dynamic batch manager not available: {e}")
                self.batch_manager = None
        
        # Agent 기반 모델 상태 확인 (Agent 자체적으로 로드됨)
        if self.agent:
            self.model_loaded = self.agent.is_ready()
            if self.model_loaded:
                self.model = self.agent.model
        
        logger.info(f"PPOStrategy 초기화 완료 - PPO 사용가능: {PPO_AVAILABLE}, Agent 준비: {self.model_loaded}, 감정분석: {SENTIMENT_AVAILABLE}, Modules: {PPO_MODULES_AVAILABLE}")
    
    def _load_model(self) -> bool:
        """PPO 모델 로드 (Agent 기반)"""
        if not PPO_MODULES_AVAILABLE:
            logger.warning("PPO modules not available - using fallback mode")
            return False
        
        try:
            if self.agent and self.agent.is_ready():
                self.model_loaded = True
                # 호환성을 위해 model 참조도 설정
                self.model = self.agent.model
                logger.info(f"PPO Agent loaded successfully from {self.config.model_path}")
                return True
            else:
                logger.warning(f"PPO Agent failed to initialize - using fallback mode")
                return False
                
        except Exception as e:
            logger.error(f"PPO Agent loading failed: {e} - using fallback mode")
            return False
    
    async def _get_sentiment_features(self, symbol: str = "BTCUSDT") -> List[float]:
        """감정 분석 특성 수집"""
        try:
            # 캐시 확인 (2분 캐시)
            current_time = time.time()
            cache_key = f"{symbol}_sentiment"
            
            if (cache_key in self.sentiment_cache and 
                current_time - self.last_sentiment_update < 120):
                return self.sentiment_cache[cache_key]
            
            if self.sentiment_client:
                # 시장 감정 상태 수집
                market_sentiment = await self.sentiment_client.get_market_sentiment(symbol)
                sentiment_features = market_sentiment.to_feature_vector()
                
                # 캐시 업데이트
                self.sentiment_cache[cache_key] = sentiment_features
                self.last_sentiment_update = current_time
                
                return sentiment_features
            else:
                # Fallback: 중립 감정
                return [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
                
        except Exception as e:
            logger.warning(f"Sentiment feature collection failed: {e}")
            return [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]  # Fallback
    
    def _extract_features(self, price_data: pd.DataFrame, sentiment_features: List[float] = None) -> Optional[np.ndarray]:
        """가격 데이터에서 특성 추출 (감정 특성 포함)"""
        try:
            logger.debug(f"특성 추출 시작 - 데이터 길이: {len(price_data)}, 감정 특성 수: {len(sentiment_features) if sentiment_features else 0}")
            
            if len(price_data) < 50:
                logger.debug("데이터 길이 부족 (< 50)")
                return None
            
            # 기본 지표 계산 (BaseRuleStrategy의 메서드 활용)
            indicators = self.calculate_indicators(price_data)
            
            if indicators is None or len(indicators) == 0:
                return None
            
            # 특성 벡터 구성
            features = []
            
            # 가격 정보
            current_close = self.safe_last(price_data, "close")
            features.append(current_close / 50000.0)  # 정규화 (BTC 기준)
            
            # 거래량 정보
            current_volume = self.safe_last(price_data, "volume", 0)
            if "volume" in price_data.columns:
                avg_volume = price_data["volume"].tail(20).mean()
            else:
                avg_volume = 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            features.append(min(volume_ratio, 5.0) / 5.0)  # 정규화
            
            # 기술적 지표
            features.append(indicators.get('rsi_14', 50) / 100.0)  # RSI
            features.append(indicators.get('macd', 0) / 1000.0)    # MACD
            features.append(indicators.get('adx', 25) / 100.0)     # ADX
            
            # 이동평균
            ema_12 = indicators.get('ema_12', current_close)
            ema_26 = indicators.get('ema_26', current_close)
            features.append((current_close - ema_12) / current_close)  # EMA12 비율
            features.append((current_close - ema_26) / current_close)  # EMA26 비율
            features.append((ema_12 - ema_26) / current_close)         # EMA 차이
            
            # 볼린저 밴드
            bb_upper = indicators.get('bb_upper', current_close * 1.02)
            bb_lower = indicators.get('bb_lower', current_close * 0.98)
            
            # Series를 float로 변환
            if hasattr(bb_upper, 'iloc'):  # Series인 경우
                bb_upper = float(bb_upper.iloc[-1])
            if hasattr(bb_lower, 'iloc'):  # Series인 경우
                bb_lower = float(bb_lower.iloc[-1])
            
            bb_diff = bb_upper - bb_lower
            bb_position = (current_close - bb_lower) / bb_diff if abs(bb_diff) > 1e-8 else 0.5
            features.append(bb_position)
            
            # 스토캐스틱
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            features.append(stoch_k / 100.0)
            features.append(stoch_d / 100.0)
            
            # 가격 변화율 (여러 기간)
            if len(price_data) >= 5:
                price_5_ago = price_data['close'].iloc[-5]
                features.append((current_close - price_5_ago) / price_5_ago)
            else:
                features.append(0.0)
            
            if len(price_data) >= 20:
                price_20_ago = price_data['close'].iloc[-20]
                features.append((current_close - price_20_ago) / price_20_ago)
            else:
                features.append(0.0)
            
            # 변동성
            if len(price_data) >= 14:
                returns = price_data['close'].pct_change().tail(14).dropna()
                volatility = returns.std()
                features.append(min(volatility * 100, 10.0) / 10.0)  # 정규화
            else:
                features.append(0.02)  # 기본값
            
            # 트렌드 강도
            if len(price_data) >= 10:
                close_values = price_data['close'].tail(10).values  # Series → array 변환
                trend_slope = np.polyfit(range(10), close_values, 1)[0]
                features.append(trend_slope / current_close * 1000)  # 정규화
            else:
                features.append(0.0)
            
            # 감정 분석 특성 추가 (6개)
            if sentiment_features and len(sentiment_features) >= 6:
                # sentiment_features의 각 요소를 float로 변환
                for i in range(6):
                    sentiment_val = sentiment_features[i] if i < len(sentiment_features) else 0.0
                    features.append(float(sentiment_val))
            else:
                # Fallback: 중립 감정 특성
                features.extend([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
            
            # 모든 특성을 float로 변환하여 26개로 맞추기
            final_features = []
            for i, feat in enumerate(features):
                try:
                    final_features.append(float(feat))
                except (ValueError, TypeError) as e:
                    logger.debug(f"특성 {i} 변환 실패: {feat} -> 0.0 ({e})")
                    final_features.append(0.0)
            
            logger.debug(f"변환된 특성 수: {len(final_features)}")
            
            # 26개로 맞추기
            while len(final_features) < self.config.state_features:
                final_features.append(0.0)
            
            # 26개로 제한
            final_features = final_features[:self.config.state_features]
            
            logger.debug(f"최종 특성 배열 크기: {len(final_features)}")
            return np.array(final_features, dtype=np.float32)
            
        except Exception as e:
            import traceback
            logger.error(f"Feature extraction error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _predict_action(self, features: np.ndarray) -> Tuple[int, float]:
        """PPO 모델로 액션 예측 (Agent 사용)"""
        try:
            if not PPO_MODULES_AVAILABLE or not self.model_loaded or not self.agent:
                return 2, 0.5  # HOLD, 중립 신뢰도
            
            # 새로운 Agent를 사용한 예측
            action_result = self.agent.predict(features)
            
            if action_result:
                self.ppo_predictions += 1
                return action_result.action, action_result.confidence
            else:
                logger.warning("Agent prediction returned None")
                return 2, 0.3  # HOLD, 낮은 신뢰도
            
        except Exception as e:
            logger.error(f"PPO Agent prediction error: {e}")
            return 2, 0.3  # HOLD, 낮은 신뢰도
    
    async def should_enter(self, price_data: pd.DataFrame, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """진입 조건 확인 (PPO 기반, 감정 분석 포함)"""
        try:
            # 감정 분석 특성 수집
            sentiment_features = await self._get_sentiment_features(symbol)
            
            # 특성 추출 (감정 포함)
            features = self._extract_features(price_data, sentiment_features)
            if features is None:
                return None
            
            # PPO 예측
            action, confidence = self._predict_action(features)
            
            # 신뢰도 임계값 확인
            if confidence < self.config.confidence_threshold:
                return None
            
            current_price = self.safe_last(price_data, "close")
            
            # 액션에 따른 신호 생성
            if action == 0:  # BUY
                return {
                    "side": "LONG",
                    "confidence": confidence,
                    "reason": f"PPO BUY signal (confidence: {confidence:.3f}, sentiment: {sentiment_features[0]:.2f})",
                    "stop_loss": current_price * 0.98,
                    "take_profit": current_price * 1.04,
                    "ppo_action": action,
                    "features_count": len(features),
                    "sentiment_boost": sentiment_features[0] > 0.1,  # 긍정적 감정 시 신호 강화
                    "sentiment_features": sentiment_features
                }
            elif action == 1:  # SELL
                return {
                    "side": "SHORT", 
                    "confidence": confidence,
                    "reason": f"PPO SELL signal (confidence: {confidence:.3f}, sentiment: {sentiment_features[0]:.2f})",
                    "stop_loss": current_price * 1.02,
                    "take_profit": current_price * 0.96,
                    "ppo_action": action,
                    "features_count": len(features),
                    "sentiment_boost": sentiment_features[0] < -0.1,  # 부정적 감정 시 신호 강화
                    "sentiment_features": sentiment_features
                }
            else:  # HOLD
                return None
                
        except Exception as e:
            logger.error(f"PPO should_enter error: {e}")
            return None
    
    def should_exit(self, position, price_data: pd.DataFrame) -> Optional[str]:
        """청산 조건 확인"""
        try:
            # 기본 리스크 관리
            current_price = self.safe_last(price_data, "close")
            
            if hasattr(position, 'entry_price') and hasattr(position, 'side'):
                entry_price = position.entry_price
                side = position.side
                
                # 손절/익절 확인
                if side == "LONG":
                    if current_price <= entry_price * 0.97:  # 3% 손절
                        return "stop_loss"
                    elif current_price >= entry_price * 1.05:  # 5% 익절
                        return "take_profit"
                elif side == "SHORT":
                    if current_price >= entry_price * 1.03:  # 3% 손절
                        return "stop_loss"
                    elif current_price <= entry_price * 0.95:  # 5% 익절
                        return "take_profit"
            
            # PPO 재평가
            features = self._extract_features(price_data)
            if features is not None:
                action, confidence = self._predict_action(features)
                
                # 반대 신호가 강하게 나올 때 청산
                if confidence > 0.8:
                    if hasattr(position, 'side'):
                        if position.side == "LONG" and action == 1:  # SELL signal
                            return "ppo_reversal"
                        elif position.side == "SHORT" and action == 0:  # BUY signal
                            return "ppo_reversal"
            
            return None
            
        except Exception as e:
            logger.error(f"PPO should_exit error: {e}")
            return None
    
    def score(self, price_data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """PPO 전략 점수 계산 (Profit Factor 포함)"""
        try:
            scores = {}
            
            if len(price_data) < 50:
                return 0.0, scores
            
            # 1. PPO 모델 사용 가능성 점수
            if self.model_loaded and PPO_AVAILABLE:
                scores['model_availability'] = 1.0
            else:
                scores['model_availability'] = 0.0
                # 모델이 없으면 낮은 점수 반환
                return 0.2, scores
            
            # 2. 특성 추출 품질 점수
            features = self._extract_features(price_data)
            if features is not None:
                # 특성 값들의 분산으로 품질 측정
                feature_variance = np.var(features)
                scores['feature_quality'] = min(1.0, feature_variance * 10)
            else:
                scores['feature_quality'] = 0.0
            
            # 3. PPO 예측 신뢰도 점수
            if features is not None:
                action, confidence = self._predict_action(features)
                scores['prediction_confidence'] = confidence
                scores['action_decisiveness'] = 1.0 if action != 2 else 0.5  # HOLD이 아니면 결정적
            else:
                scores['prediction_confidence'] = 0.0
                scores['action_decisiveness'] = 0.0
            
            # 4. 모델 성과 기록 점수
            if self.ppo_predictions > 0:
                success_rate = self.ppo_successes / self.ppo_predictions
                scores['historical_performance'] = success_rate
            else:
                scores['historical_performance'] = 0.5  # 중립
            
            # 5. Profit Factor 보너스 점수 (BaseRuleStrategy 상속)
            if hasattr(self, 'profit_factor') and self.total_trades > 5:
                if self.profit_factor == float('inf'):
                    scores['profit_factor'] = 1.0
                elif self.profit_factor > 2.0:
                    scores['profit_factor'] = 1.0
                elif self.profit_factor > 1.5:
                    scores['profit_factor'] = 0.8 + (self.profit_factor - 1.5) * 0.4
                elif self.profit_factor > 1.0:
                    scores['profit_factor'] = 0.5 + (self.profit_factor - 1.0) * 0.6
                else:
                    scores['profit_factor'] = max(0.0, self.profit_factor * 0.5)
            else:
                scores['profit_factor'] = 0.5
            
            # 6. 시장 조건 적합성 점수
            current_price = self.safe_last(price_data, "close")
            if len(price_data) >= 20:
                price_volatility = price_data['close'].pct_change().tail(20).std()
                # PPO는 변동성이 있는 시장에서 더 효과적
                scores['market_suitability'] = min(1.0, price_volatility * 50)
            else:
                scores['market_suitability'] = 0.5
            
            # 종합 점수 계산 (가중 평균)
            weights = {
                'model_availability': 0.25,    # 모델 사용 가능성
                'feature_quality': 0.15,       # 특성 품질
                'prediction_confidence': 0.20, # 예측 신뢰도
                'action_decisiveness': 0.10,   # 액션 결정성
                'historical_performance': 0.15, # 과거 성과
                'profit_factor': 0.10,         # Profit Factor 보너스
                'market_suitability': 0.05     # 시장 적합성
            }
            
            composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
            
            return composite_score, scores
            
        except Exception as e:
            logger.error(f"PPO score calculation error: {e}")
            return 0.0, {'error': 1.0}
    
    async def generate_signal(self, price_data: pd.DataFrame, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """VPS용 신호 생성 메서드 (감정 분석 통합, P1-3: 동적 배치 성능 추적)"""
        start_time = time.time()
        
        try:
            if len(price_data) < 50:
                return {
                    "action": "HOLD",
                    "strength": 0.0,
                    "price": self.safe_last(price_data, "close"),
                    "metadata": {
                        "strategy": "PPOStrategy",
                        "reason": "Insufficient data",
                        "confidence": 0.0,
                        "composite_score": 0.0,
                        "detail_scores": {},
                        "sentiment_integrated": SENTIMENT_AVAILABLE,
                        "dynamic_batching": self.batch_manager is not None
                    }
                }
            
            # P1-3: 현재 배치 크기 가져오기
            current_batch_size = self.batch_manager.get_current_batch_size() if self.batch_manager else self.config.batch_size
            
            # 진입 조건 확인 (감정 분석 포함)
            entry_signal = await self.should_enter(price_data, symbol) 
            current_price = self.safe_last(price_data, "close")
            
            if entry_signal:
                # 점수 계산
                composite_score, detail_scores = self.score(price_data)
                
                # 감정 부스트 적용
                strength = entry_signal.get("confidence", 0.5) * composite_score
                if entry_signal.get("sentiment_boost", False):
                    strength = min(strength * 1.1, 1.0)  # 10% 부스트
                
                # P1-3: 성공적인 신호 생성 시 배치 성능 업데이트
                processing_time = time.time() - start_time
                if self.batch_manager:
                    self.batch_manager.update_batch_performance(
                        processing_time=processing_time,
                        items_processed=current_batch_size,
                        success_rate=1.0,  # 신호 생성 성공
                        custom_metrics={"confidence": entry_signal.get("confidence", 0.5)}
                    )
                
                return {
                    "action": "BUY" if entry_signal["side"] == "LONG" else "SELL",
                    "strength": strength,
                    "price": current_price,
                    "metadata": {
                        "strategy": "PPOStrategy",
                        "reason": entry_signal.get("reason", ""),
                        "confidence": entry_signal.get("confidence", 0.5),
                        "composite_score": composite_score,
                        "detail_scores": detail_scores,
                        "stop_loss": entry_signal.get("stop_loss"),
                        "take_profit": entry_signal.get("take_profit"),
                        "ppo_action": entry_signal.get("ppo_action"),
                        "features_count": entry_signal.get("features_count", 0),
                        "model_loaded": self.model_loaded,
                        "ppo_available": PPO_AVAILABLE,
                        "sentiment_integrated": SENTIMENT_AVAILABLE,
                        "sentiment_boost": entry_signal.get("sentiment_boost", False),
                        "sentiment_features": entry_signal.get("sentiment_features", []),
                        "dynamic_batching": self.batch_manager is not None,
                        "current_batch_size": current_batch_size,
                        "processing_time_ms": round(processing_time * 1000, 2)
                    }
                }
            
            # HOLD 신호
            composite_score, detail_scores = self.score(price_data)
            
            # P1-3: HOLD 신호 시에도 배치 성능 업데이트
            processing_time = time.time() - start_time
            if self.batch_manager:
                self.batch_manager.update_batch_performance(
                    processing_time=processing_time,
                    items_processed=current_batch_size,
                    success_rate=0.5,  # HOLD는 중간 성공률
                    custom_metrics={"signal_type": "hold"}
                )
            
            # PPO 점수 로깅 (HOLD 신호)
            try:
                from trading.ppo_score_logger import get_ppo_score_logger
                ppo_logger = get_ppo_score_logger()
                ppo_logger.log_score(
                    strategy_score=composite_score,
                    confidence=0.0,
                    action="HOLD",
                    selected=False,  # HOLD는 선택되지 않음
                    total_predictions=self.ppo_predictions,
                    success_rate=self.ppo_successes / self.ppo_predictions if self.ppo_predictions > 0 else 0.0
                )
            except ImportError:
                logger.warning("PPO 점수 로거를 불러올 수 없습니다")
            
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {
                    "strategy": "PPOStrategy",
                    "reason": "No strong signal or low confidence",
                    "confidence": 0.0,
                    "composite_score": composite_score,
                    "detail_scores": detail_scores,
                    "model_loaded": self.model_loaded,
                    "ppo_available": PPO_AVAILABLE,
                    "sentiment_integrated": SENTIMENT_AVAILABLE,
                    "dynamic_batching": self.batch_manager is not None,
                    "current_batch_size": current_batch_size,
                    "processing_time_ms": round(processing_time * 1000, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"PPO generate_signal error: {e}")
            current_price = self.safe_last(price_data, "close")
            
            # P1-3: 에러 발생 시에도 배치 성능 업데이트 (낮은 성공률)
            processing_time = time.time() - start_time
            if self.batch_manager:
                self.batch_manager.update_batch_performance(
                    processing_time=processing_time,
                    items_processed=current_batch_size if 'current_batch_size' in locals() else self.config.batch_size,
                    success_rate=0.0,  # 에러 발생 시 0% 성공률
                    custom_metrics={"error": str(e)[:50]}  # 에러 메시지 일부
                )
            
            return {
                "action": "HOLD",
                "strength": 0.0,
                "price": current_price,
                "metadata": {
                    "strategy": "PPOStrategy",
                    "reason": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "composite_score": 0.0,
                    "detail_scores": {"error": 1.0},
                    "model_loaded": self.model_loaded,
                    "ppo_available": PPO_AVAILABLE,
                    "sentiment_integrated": SENTIMENT_AVAILABLE
                }
            }
    
    def update_prediction_result(self, success: bool):
        """PPO 예측 결과 업데이트"""
        if success:
            self.ppo_successes += 1
            self.model_confidence_history.append(1.0)
        else:
            self.model_confidence_history.append(0.0)
        
        # 최근 20개만 유지
        if len(self.model_confidence_history) > 20:
            self.model_confidence_history.pop(0)
    
    def get_ppo_statistics(self) -> Dict[str, Any]:
        """PPO 전용 통계 (P1-3: 동적 배치 성능 포함)"""
        stats = {
            'ppo_available': PPO_AVAILABLE,
            'ppo_modules_available': PPO_MODULES_AVAILABLE,
            'model_loaded': self.model_loaded,
            'agent_ready': self.agent.is_ready() if self.agent else False,
            'model_path': self.config.model_path,
            'total_predictions': self.ppo_predictions,
            'successful_predictions': self.ppo_successes,
            'prediction_success_rate': self.ppo_successes / self.ppo_predictions if self.ppo_predictions > 0 else 0,
            'confidence_threshold': self.config.confidence_threshold,
            'recent_confidence_avg': np.mean(self.model_confidence_history) if self.model_confidence_history else 0.0,
            'deployment_mode': 'inference_only',  # VPS는 추론만 수행
            'training_available': self.trainer is not None  # Trainer 가용성
        }
        
        # P1-3: 동적 배치 성능 통계 추가
        if self.batch_manager:
            batch_summary = self.batch_manager.get_performance_summary()
            stats.update({
                'dynamic_batching_enabled': True,
                'batch_performance': batch_summary,
                'batch_recommendations': self.batch_manager.get_optimization_recommendations()
            })
        else:
            stats.update({
                'dynamic_batching_enabled': False,
                'static_batch_size': self.config.batch_size
            })
        
        return stats
    
    def add_training_experience(self, state: np.ndarray, action: int, reward: float, 
                               next_state: np.ndarray, done: bool):
        """트레이닝 경험 추가 (Trainer 사용)"""
        if self.trainer and PPO_MODULES_AVAILABLE:
            try:
                self.trainer.add_experience(state, action, reward, next_state, done)
                logger.debug(f"Training experience added: action={action}, reward={reward}")
            except Exception as e:
                logger.error(f"Failed to add training experience: {e}")
        else:
            logger.warning("Trainer not available for adding experience")
    
    def add_score_based_reward(self, state: np.ndarray, action: int, strategy_score: float, 
                              market_outcome: float, next_state: np.ndarray, done: bool = False,
                              confidence: float = 0.5, action_str: str = 'UNKNOWN') -> float:
        """
        🎯 정밀화된 전략 점수 → 보상 변환 시스템
        Rule 전략과 유사한 성과 기반 보상 강화, PPOTrainer에 효과적 shaping 피드백 제공
        
        Args:
            state: 현재 상태 벡터
            action: 선택된 액션 (0=BUY, 1=SELL, 2=HOLD)
            strategy_score: PPO 전략 점수 (0.0~1.0)
            market_outcome: 시장 결과 (손익률, -1.0~1.0)
            next_state: 다음 상태 벡터
            done: 에피소드 완료 여부
            confidence: 예측 신뢰도 (0.0~1.0)
            action_str: 액션 문자열 (로깅용)
            
        Returns:
            최종 계산된 보상값 (-1.0~1.0)
        """
        try:
            # 1. 기본 보상 컴포넌트 계산
            base_reward = self._calculate_base_reward(strategy_score, confidence)
            market_reward = self._calculate_market_reward(market_outcome)
            consistency_reward = self._calculate_consistency_reward(action, market_outcome)
            risk_penalty = self._calculate_risk_penalty(state, action)
            exploration_bonus = self._calculate_exploration_bonus(action)
            
            # 2. 동적 가중치 계산 (Rule 전략 대비 성과 기반)
            performance_weights = self._calculate_dynamic_weights()
            
            # 3. 멀티컴포넌트 보상 조합
            final_reward = (
                base_reward * performance_weights['base'] +              # 전략 점수 & 신뢰도
                market_reward * performance_weights['market'] +          # 시장 결과
                consistency_reward * performance_weights['consistency'] + # 일관성 보너스
                risk_penalty * performance_weights['risk'] +             # 리스크 페널티
                exploration_bonus * performance_weights['exploration']   # 탐험 보너스
            )
            
            # 4. 적응형 보상 클리핑 및 스케일링
            final_reward = self._adaptive_reward_scaling(final_reward, strategy_score, confidence)
            
            # 5. 고급 경험 메타데이터 생성
            experience_metadata = {
                'strategy_score': strategy_score,
                'market_outcome': market_outcome,
                'final_reward': final_reward,
                'confidence': confidence,
                'action_str': action_str,
                'reward_components': {
                    'base_reward': base_reward,
                    'market_reward': market_reward,
                    'consistency_reward': consistency_reward,
                    'risk_penalty': risk_penalty,
                    'exploration_bonus': exploration_bonus
                },
                'weights': performance_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            # 6. 향상된 경험 추가 (메타데이터 포함)
            self.add_enhanced_training_experience(
                state, action, final_reward, next_state, done, experience_metadata
            )
            
            # 7. 성과 추적 업데이트
            self._update_performance_tracking(final_reward, strategy_score, market_outcome, confidence)
            
            # 8. 향상된 로깅
            self._log_reward_shaping_details(experience_metadata)
            
            return final_reward
            
        except Exception as e:
            logger.error(f"고급 점수 기반 보상 계산 실패: {e}")
            # 폴백: 기본 보상 계산
            fallback_reward = (strategy_score * 0.3) + (market_outcome * 0.7)
            return max(-1.0, min(1.0, fallback_reward))
    
    def _calculate_base_reward(self, strategy_score: float, confidence: float) -> float:
        """기본 보상 계산 (전략 점수 + 신뢰도)"""
        # 전략 점수와 신뢰도의 조화 평균 사용 (극값 방지)
        if strategy_score + confidence > 0:
            harmonic_mean = 2 * strategy_score * confidence / (strategy_score + confidence)
        else:
            harmonic_mean = 0.0
        
        # 신뢰도 임계값 보너스
        confidence_bonus = 0.1 if confidence > 0.8 else 0.0
        
        return min(1.0, harmonic_mean + confidence_bonus)
    
    def _calculate_market_reward(self, market_outcome: float) -> float:
        """시장 보상 계산 (비선형 스케일링)"""
        # 작은 수익도 포지티브하게 보상, 큰 손실은 강하게 페널티
        if market_outcome > 0:
            # 수익: 제곱근 스케일링으로 작은 수익도 인센티브
            return min(1.0, np.sqrt(abs(market_outcome)) * np.sign(market_outcome))
        else:
            # 손실: 제곱 스케일링으로 큰 손실 강력 페널티
            return max(-1.0, -(abs(market_outcome) ** 1.5))
    
    def _calculate_consistency_reward(self, action: int, market_outcome: float) -> float:
        """일관성 보상 계산 (액션과 결과의 일치도)"""
        # BUY 액션에서 수익, SELL 액션에서 수익 시 보너스
        if action == 0 and market_outcome > 0:  # BUY에서 수익
            return 0.15
        elif action == 1 and market_outcome > 0:  # SELL에서 수익 (숏 포지션)
            return 0.15
        elif action == 2:  # HOLD 액션
            return 0.05 if abs(market_outcome) < 0.01 else -0.05  # 안정성 보상/페널티
        else:
            return -0.1  # 액션과 결과 불일치 페널티
    
    def _calculate_risk_penalty(self, state: np.ndarray, action: int) -> float:
        """리스크 페널티 계산 (시장 조건 기반)"""
        try:
            # 상태 벡터에서 변동성 추출 (인덱스 12는 변동성 특성)
            volatility = state[12] if len(state) > 12 else 0.02
            
            # 고변동성에서 공격적 액션 페널티
            if volatility > 0.8 and action != 2:  # 고변동성에서 HOLD가 아닌 액션
                return -0.1
            
            # RSI 과매수/과매도에서 역방향 액션 페널티
            rsi = state[2] if len(state) > 2 else 0.5  # 인덱스 2는 RSI
            if rsi > 0.8 and action == 0:  # 과매수에서 BUY
                return -0.15
            elif rsi < 0.2 and action == 1:  # 과매도에서 SELL
                return -0.15
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"리스크 페널티 계산 오류: {e}")
            return 0.0
    
    def _calculate_exploration_bonus(self, action: int) -> float:
        """탐험 보너스 계산 (액션 다양성 장려)"""
        if not hasattr(self, 'recent_actions'):
            self.recent_actions = []
        
        self.recent_actions.append(action)
        
        # 최근 10개 액션만 유지
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # 액션 다양성 계산
        if len(self.recent_actions) >= 5:
            unique_actions = len(set(self.recent_actions[-5:]))
            if unique_actions >= 3:  # 3가지 이상 액션 사용
                return 0.05
            elif unique_actions == 1:  # 같은 액션만 반복
                return -0.03
        
        return 0.0
    
    def _calculate_dynamic_weights(self) -> Dict[str, float]:
        """Rule 전략 대비 성과 기반 동적 가중치 계산 (실제 데이터 활용)"""
        try:
            # 1. Rule 전략 성과 데이터 수집 시도
            rule_avg_performance = self._get_rule_performance_benchmark()
            
            # 2. PPO 자체 성과 계산
            ppo_performance = self._calculate_ppo_performance_score()
            
            # 3. 성과 비교 비율 계산
            performance_ratio = ppo_performance / rule_avg_performance if rule_avg_performance > 0 else 1.0
            
            # 4. 최근 성과 트렌드 분석
            recent_trend = self._analyze_recent_performance_trend()
            
            # 5. 동적 가중치 조정 (성과 기반 + 트렌드 기반)
            base_weights = self._get_base_weights(performance_ratio)
            trend_adjusted_weights = self._apply_trend_adjustment(base_weights, recent_trend, performance_ratio)
            
            # 6. 학습 단계별 가중치 조정
            learning_adjusted_weights = self._apply_learning_stage_adjustment(trend_adjusted_weights)
            
            return learning_adjusted_weights
            
        except Exception as e:
            logger.debug(f"동적 가중치 계산 오류: {e}")
            return self._get_fallback_weights()
    
    def _get_rule_performance_benchmark(self) -> float:
        """Rule 전략 성과 벤치마크 수집"""
        try:
            # 외부 컨텍스트에서 Rule 성과 가져오기 시도
            if hasattr(self, 'external_rule_performance'):
                return self.external_rule_performance
            
            # 글로벌 성과 메트릭 접근 시도 (vps_strategy_adapter와의 연동)
            import sys
            current_module = sys.modules.get(__name__)
            if hasattr(current_module, 'global_rule_performance'):
                return getattr(current_module, 'global_rule_performance')
            
            # Profit Factor 기반 추정
            if hasattr(self, 'profit_factor') and self.total_trades > 5:
                # Rule 전략들의 일반적인 Profit Factor 범위: 1.0~2.5
                estimated_rule_pf = 1.5  # 평균적인 Rule 전략 PF
                ppo_pf = min(self.profit_factor, 5.0) if self.profit_factor != float('inf') else 2.0
                
                # PF를 0~1 성과 점수로 변환
                rule_performance = min(1.0, (estimated_rule_pf - 1.0) / 1.5)  # 1.0~2.5 PF를 0~1로
                return max(0.3, min(0.9, rule_performance))
            
            # 기본값: 중간 수준 성과
            return 0.6
            
        except Exception as e:
            logger.debug(f"Rule 성과 벤치마크 수집 실패: {e}")
            return 0.6
    
    def _calculate_ppo_performance_score(self) -> float:
        """PPO 성과 점수 계산 (다차원 평가)"""
        try:
            if self.ppo_predictions == 0:
                return 0.5  # 중립
            
            # 1. 기본 성공률
            success_rate = self.ppo_successes / self.ppo_predictions
            
            # 2. 최근 신뢰도 평균
            recent_confidence = np.mean(self.model_confidence_history) if self.model_confidence_history else 0.5
            
            # 3. 보상 히스토리 기반 성과
            reward_performance = 0.5
            if hasattr(self, 'reward_history') and self.reward_history:
                recent_rewards = [r['final_reward'] for r in self.reward_history[-10:]]
                positive_ratio = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                avg_reward = np.mean(recent_rewards)
                reward_performance = (positive_ratio * 0.6) + (min(1.0, max(0.0, avg_reward + 0.5)) * 0.4)
            
            # 4. Profit Factor 기반 성과 (Rule 전략과 동일한 기준)
            pf_performance = 0.5
            if hasattr(self, 'profit_factor') and self.total_trades > 0:
                if self.profit_factor == float('inf'):
                    pf_performance = 1.0
                elif self.profit_factor > 2.0:
                    pf_performance = 1.0
                elif self.profit_factor > 1.5:
                    pf_performance = 0.8 + (self.profit_factor - 1.5) * 0.4
                elif self.profit_factor > 1.0:
                    pf_performance = 0.5 + (self.profit_factor - 1.0) * 0.6
                else:
                    pf_performance = max(0.0, self.profit_factor * 0.5)
            
            # 5. 종합 성과 점수 (가중 평균)
            composite_score = (
                success_rate * 0.25 +           # 예측 성공률 25%
                recent_confidence * 0.20 +      # 신뢰도 20%
                reward_performance * 0.30 +     # 보상 성과 30%
                pf_performance * 0.25           # Profit Factor 성과 25%
            )
            
            return min(1.0, max(0.0, composite_score))
            
        except Exception as e:
            logger.debug(f"PPO 성과 점수 계산 실패: {e}")
            return 0.5
    
    def _analyze_recent_performance_trend(self) -> Dict[str, float]:
        """최근 성과 트렌드 분석"""
        try:
            trend_data = {
                'direction': 0.0,  # -1(하락) ~ 1(상승)
                'strength': 0.0,   # 0(약함) ~ 1(강함)
                'stability': 0.0   # 0(불안정) ~ 1(안정)
            }
            
            if not hasattr(self, 'reward_history') or len(self.reward_history) < 5:
                return trend_data
            
            # 최근 보상 데이터
            recent_rewards = [r['final_reward'] for r in self.reward_history[-10:]]
            
            if len(recent_rewards) >= 5:
                # 트렌드 방향 계산 (선형 회귀 기울기)
                x = np.arange(len(recent_rewards))
                slope = np.polyfit(x, recent_rewards, 1)[0]
                trend_data['direction'] = max(-1.0, min(1.0, slope * 10))  # 스케일링
                
                # 트렌드 강도 계산 (R² 값)
                correlation = np.corrcoef(x, recent_rewards)[0, 1] if len(recent_rewards) > 1 else 0
                trend_data['strength'] = abs(correlation)
                
                # 안정성 계산 (변동성의 역수)
                volatility = np.std(recent_rewards)
                trend_data['stability'] = 1.0 / (1.0 + volatility * 5)  # 정규화
            
            return trend_data
            
        except Exception as e:
            logger.debug(f"성과 트렌드 분석 실패: {e}")
            return {'direction': 0.0, 'strength': 0.0, 'stability': 0.0}
    
    def _get_base_weights(self, performance_ratio: float) -> Dict[str, float]:
        """성과 비율 기반 기본 가중치 계산"""
        if performance_ratio > 1.3:  # PPO가 Rule보다 30% 이상 좋음
            return {
                'base': 0.50,      # 전략 점수 신뢰도 최대
                'market': 0.30,    # 시장 결과 의존도 최소
                'consistency': 0.12,
                'risk': 0.05,
                'exploration': 0.03
            }
        elif performance_ratio > 1.1:  # PPO가 Rule보다 10% 이상 좋음
            return {
                'base': 0.45,
                'market': 0.35,
                'consistency': 0.12,
                'risk': 0.05,
                'exploration': 0.03
            }
        elif performance_ratio > 0.9:  # 비슷한 성과
            return {
                'base': 0.35,
                'market': 0.45,
                'consistency': 0.13,
                'risk': 0.04,
                'exploration': 0.03
            }
        elif performance_ratio > 0.7:  # PPO가 Rule보다 약간 나쁨
            return {
                'base': 0.28,
                'market': 0.52,
                'consistency': 0.15,
                'risk': 0.03,
                'exploration': 0.02
            }
        else:  # PPO가 Rule보다 30% 이상 나쁨
            return {
                'base': 0.20,      # 전략 점수 신뢰도 최소
                'market': 0.60,    # 시장 결과 의존도 최대
                'consistency': 0.15,
                'risk': 0.03,
                'exploration': 0.02
            }
    
    def _apply_trend_adjustment(self, base_weights: Dict[str, float], 
                              trend: Dict[str, float], performance_ratio: float) -> Dict[str, float]:
        """트렌드 기반 가중치 조정"""
        try:
            adjusted_weights = base_weights.copy()
            
            # 상승 트렌드일 때
            if trend['direction'] > 0.3 and trend['strength'] > 0.5:
                # 전략 점수 가중치 증가 (모델이 학습하고 있음)
                boost = min(0.1, trend['direction'] * trend['strength'] * 0.2)
                adjusted_weights['base'] = min(0.6, adjusted_weights['base'] + boost)
                adjusted_weights['market'] = max(0.2, adjusted_weights['market'] - boost)
            
            # 하락 트렌드일 때
            elif trend['direction'] < -0.3 and trend['strength'] > 0.5:
                # 시장 결과 가중치 증가 (모델 성과 하락)
                penalty = min(0.1, abs(trend['direction']) * trend['strength'] * 0.2)
                adjusted_weights['market'] = min(0.7, adjusted_weights['market'] + penalty)
                adjusted_weights['base'] = max(0.15, adjusted_weights['base'] - penalty)
            
            # 불안정한 성과일 때
            if trend['stability'] < 0.3:
                # 탐험 보너스 증가 (더 많은 탐험 필요)
                adjusted_weights['exploration'] = min(0.1, adjusted_weights['exploration'] + 0.02)
                adjusted_weights['base'] = max(0.15, adjusted_weights['base'] - 0.02)
            
            return adjusted_weights
            
        except Exception as e:
            logger.debug(f"트렌드 조정 실패: {e}")
            return base_weights
    
    def _apply_learning_stage_adjustment(self, weights: Dict[str, float]) -> Dict[str, float]:
        """학습 단계별 가중치 조정"""
        try:
            adjusted_weights = weights.copy()
            
            # 초기 학습 단계 (예측 수 < 50)
            if self.ppo_predictions < 50:
                # 탐험과 일관성을 더 중시
                adjusted_weights['exploration'] = min(0.15, adjusted_weights['exploration'] + 0.05)
                adjusted_weights['consistency'] = min(0.2, adjusted_weights['consistency'] + 0.03)
                adjusted_weights['base'] = max(0.15, adjusted_weights['base'] - 0.04)
                adjusted_weights['market'] = max(0.2, adjusted_weights['market'] - 0.04)
            
            # 중간 학습 단계 (50 <= 예측 수 < 200)
            elif self.ppo_predictions < 200:
                # 균형잡힌 학습
                pass  # 기본 가중치 유지
            
            # 성숙 학습 단계 (예측 수 >= 200)
            else:
                # 전략 점수 신뢰도 증가
                if self.ppo_successes / self.ppo_predictions > 0.6:  # 성과가 좋다면
                    adjusted_weights['base'] = min(0.55, adjusted_weights['base'] + 0.05)
                    adjusted_weights['market'] = max(0.25, adjusted_weights['market'] - 0.05)
            
            # 가중치 정규화 (합계가 1.0이 되도록)
            total = sum(adjusted_weights.values())
            if total > 0:
                for key in adjusted_weights:
                    adjusted_weights[key] /= total
            
            return adjusted_weights
            
        except Exception as e:
            logger.debug(f"학습 단계 조정 실패: {e}")
            return weights
    
    def _get_fallback_weights(self) -> Dict[str, float]:
        """폴백 가중치 반환"""
        return {
            'base': 0.35,
            'market': 0.45,
            'consistency': 0.13,
            'risk': 0.04,
            'exploration': 0.03
        }
    
    def set_external_rule_performance(self, rule_performance: float):
        """외부에서 Rule 전략 성과 설정 (vps_strategy_adapter에서 호출)"""
        self.external_rule_performance = max(0.1, min(1.0, rule_performance))
    
    def _adaptive_reward_scaling(self, reward: float, strategy_score: float, confidence: float) -> float:
        """적응형 보상 스케일링 및 클리핑"""
        try:
            # 1. 신뢰도 기반 스케일링
            confidence_scale = 0.5 + (confidence * 0.5)  # 0.5~1.0 범위
            scaled_reward = reward * confidence_scale
            
            # 2. 전략 점수 기반 증폭/감쇠
            if strategy_score > 0.8:
                scaled_reward *= 1.1  # 고점수 시 증폭
            elif strategy_score < 0.3:
                scaled_reward *= 0.8  # 저점수 시 감쇠
            
            # 3. 적응형 클리핑 (절대값 기준)
            max_reward = 0.9 if confidence > 0.7 else 0.7
            min_reward = -max_reward
            
            # 4. 부드러운 클리핑 (tanh 함수 사용)
            final_reward = max_reward * np.tanh(scaled_reward / max_reward)
            
            return final_reward
            
        except Exception as e:
            logger.debug(f"적응형 보상 스케일링 오류: {e}")
            return max(-1.0, min(1.0, reward))
    
    def add_enhanced_training_experience(self, state: np.ndarray, action: int, reward: float, 
                                       next_state: np.ndarray, done: bool, metadata: Dict[str, Any]):
        """향상된 트레이닝 경험 추가 (메타데이터 포함)"""
        try:
            # 기본 경험 추가
            self.add_training_experience(state, action, reward, next_state, done)
            
            # 메타데이터 저장 (향후 분석용)
            if not hasattr(self, 'experience_metadata'):
                self.experience_metadata = []
            
            self.experience_metadata.append(metadata)
            
            # 최근 100개만 유지 (메모리 관리)
            if len(self.experience_metadata) > 100:
                self.experience_metadata.pop(0)
            
        except Exception as e:
            logger.error(f"향상된 경험 추가 실패: {e}")
    
    def _update_performance_tracking(self, final_reward: float, strategy_score: float, 
                                   market_outcome: float, confidence: float):
        """성과 추적 업데이트"""
        try:
            # 통계 업데이트
            self.ppo_predictions += 1
            if final_reward > 0:
                self.ppo_successes += 1
            
            # 신뢰도 히스토리 업데이트
            self.model_confidence_history.append(confidence)
            if len(self.model_confidence_history) > 50:
                self.model_confidence_history.pop(0)
            
            # 보상 히스토리 추가
            if not hasattr(self, 'reward_history'):
                self.reward_history = []
            
            self.reward_history.append({
                'final_reward': final_reward,
                'strategy_score': strategy_score,
                'market_outcome': market_outcome,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # 최근 50개만 유지
            if len(self.reward_history) > 50:
                self.reward_history.pop(0)
            
        except Exception as e:
            logger.error(f"성과 추적 업데이트 실패: {e}")
    
    def _log_reward_shaping_details(self, metadata: Dict[str, Any]):
        """보상 shaping 상세 로깅"""
        try:
            # PPO 점수 로거 사용
            from trading.ppo_score_logger import get_ppo_score_logger
            ppo_logger = get_ppo_score_logger()
            
            # 상세 로깅
            ppo_logger.log_score(
                strategy_score=metadata['strategy_score'],
                confidence=metadata['confidence'],
                action=metadata['action_str'],
                market_outcome=metadata['market_outcome'],
                final_reward=metadata['final_reward'],
                selected=True,
                total_predictions=self.ppo_predictions,
                success_rate=self.ppo_successes / self.ppo_predictions if self.ppo_predictions > 0 else 0.0
            )
            
            # 상세 컴포넌트 로깅 (디버그)
            components = metadata['reward_components']
            logger.debug(
                f"🎯 PPO 보상 Shaping: "
                f"최종={metadata['final_reward']:.4f} "
                f"[기본={components['base_reward']:.3f}, "
                f"시장={components['market_reward']:.3f}, "
                f"일관성={components['consistency_reward']:.3f}, "
                f"리스크={components['risk_penalty']:.3f}, "
                f"탐험={components['exploration_bonus']:.3f}] "
                f"전략점수={metadata['strategy_score']:.3f} "
                f"신뢰도={metadata['confidence']:.3f}"
            )
            
        except ImportError:
            logger.warning("PPO 점수 로거를 불러올 수 없습니다")
        except Exception as e:
            logger.error(f"보상 shaping 로깅 실패: {e}")
    
    def get_reward_shaping_statistics(self) -> Dict[str, Any]:
        """보상 shaping 통계 반환"""
        try:
            stats = {
                'total_rewards_shaped': len(getattr(self, 'reward_history', [])),
                'average_reward': 0.0,
                'reward_variance': 0.0,
                'positive_reward_ratio': 0.0,
                'recent_confidence_avg': 0.0,
                'component_averages': {}
            }
            
            if hasattr(self, 'reward_history') and self.reward_history:
                rewards = [r['final_reward'] for r in self.reward_history]
                confidences = [r['confidence'] for r in self.reward_history]
                
                stats['average_reward'] = np.mean(rewards)
                stats['reward_variance'] = np.var(rewards)
                stats['positive_reward_ratio'] = sum(1 for r in rewards if r > 0) / len(rewards)
                stats['recent_confidence_avg'] = np.mean(confidences)
            
            if hasattr(self, 'experience_metadata') and self.experience_metadata:
                # 컴포넌트별 평균 계산
                components = ['base_reward', 'market_reward', 'consistency_reward', 'risk_penalty', 'exploration_bonus']
                for comp in components:
                    values = [m['reward_components'][comp] for m in self.experience_metadata if comp in m.get('reward_components', {})]
                    if values:
                        stats['component_averages'][comp] = np.mean(values)
            
            return stats
            
        except Exception as e:
            logger.error(f"보상 shaping 통계 계산 실패: {e}")
            return {'error': str(e)}
    
    def train_model(self, episodes: int = 1000) -> bool:
        """모델 트레이닝 실행 (Trainer 사용)"""
        if not self.trainer or not PPO_MODULES_AVAILABLE:
            logger.warning("Trainer not available for training")
            return False
            
        try:
            logger.info(f"Starting PPO training for {episodes} episodes")
            self.trainer.train(episodes)
            
            # 새로운 모델로 Agent 업데이트
            if self.agent:
                self.agent.load_model(self.config.model_path)
                self.model = self.agent.model
                logger.info("Agent updated with newly trained model")
            
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def save_model(self, path: str = None) -> bool:
        """모델 저장"""
        if not self.agent or not PPO_MODULES_AVAILABLE:
            logger.warning("Agent not available for saving")
            return False
            
        try:
            save_path = path or self.config.model_path
            success = self.agent.save_model(save_path)
            if success:
                logger.info(f"Model saved to {save_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def evaluate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """PPO 전략 결과 평가 (Rule 전략과 호환성)"""
        try:
            # 기본 평가 정보
            evaluation = {
                'strategy_name': 'PPOStrategy',
                'timestamp': datetime.now().isoformat(),
                'ppo_specific': True
            }
            
            # 거래 결과 정보 추출
            pnl = result.get('pnl', 0.0)
            entry_price = result.get('entry_price', 0.0)
            exit_price = result.get('exit_price', 0.0)
            side = result.get('side', 'LONG')
            
            # 성과 메트릭 업데이트
            self.add_market_outcome_ppo(pnl)
            
            # 평가 결과
            evaluation.update({
                'pnl': pnl,
                'profitable': pnl > 0,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': side,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0,
                'profit_factor': self.profit_factor,
                'total_pnl': self.total_pnl,
                'ppo_predictions': self.ppo_predictions,
                'ppo_successes': self.ppo_successes,
                'ppo_success_rate': self.ppo_successes / self.ppo_predictions if self.ppo_predictions > 0 else 0.0,
                'model_loaded': self.model_loaded,
                'agent_ready': self.agent.is_ready() if self.agent else False
            })
            
            logger.info(f"PPO 전략 결과 평가 완료: PnL={pnl:.3f}, "
                       f"Win Rate={evaluation['win_rate']:.3f}, "
                       f"PPO Success Rate={evaluation['ppo_success_rate']:.3f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"PPO evaluate_result 실패: {e}")
            return {
                'strategy_name': 'PPOStrategy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'ppo_specific': True
            }
    
    def add_market_outcome_ppo(self, outcome: float) -> None:
        """시장 결과 피드백 추가 (BaseRuleStrategy 호환)"""
        try:
            self.total_trades += 1
            self.total_pnl += outcome
            
            if outcome > 0:
                self.winning_trades += 1
                self.total_profit += outcome
            else:
                self.losing_trades += 1
                self.total_loss += abs(outcome)
            
            # Profit factor 업데이트
            if self.total_loss > 0:
                self.profit_factor = self.total_profit / self.total_loss
            else:
                self.profit_factor = float('inf') if self.total_profit > 0 else 1.0
            
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
            
            logger.debug(f"PPO 성과 업데이트: outcome={outcome:.3f}, "
                        f"win_rate={win_rate:.3f}, profit_factor={self.profit_factor:.3f}")
            
        except Exception as e:
            logger.error(f"PPO 시장 결과 처리 실패: {e}")

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 PPOStrategy 테스트")
    
    # 더미 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50200,
        'low': np.random.randn(100).cumsum() + 49800,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.randint(100, 1000, 100)
    })
    
    # PPO 전략 테스트
    config = PPOConfig(
        model_path="/tmp/test_ppo.zip",  # 존재하지 않는 경로
        confidence_threshold=0.6
    )
    
    strategy = PPOStrategy(config)
    
    # 신호 생성 테스트
    signal = strategy.generate_signal(price_data)
    print(f"생성된 신호: {signal}")
    
    # 점수 계산 테스트
    score, details = strategy.score(price_data)
    print(f"전략 점수: {score:.3f}")
    print(f"상세 점수: {details}")
    
    # 통계 확인
    stats = strategy.get_ppo_statistics()
    print(f"PPO 통계: {stats}")
    
    # Agent 테스트
    if strategy.agent:
        print(f"Agent 준비상태: {strategy.agent.is_ready()}")
        if strategy.agent.is_ready():
            # 더미 특성으로 예측 테스트
            dummy_features = np.random.randn(26).astype(np.float32)
            result = strategy.agent.predict(dummy_features)
            if result:
                print(f"Agent 예측 결과: action={result.action}, confidence={result.confidence}")
    
    # Trainer 테스트
    if strategy.trainer:
        print(f"Trainer 사용가능: True")
        # 더미 경험 추가
        dummy_state = np.random.randn(26).astype(np.float32)
        dummy_next_state = np.random.randn(26).astype(np.float32)
        strategy.add_training_experience(dummy_state, 1, 0.1, dummy_next_state, False)
        print("더미 트레이닝 경험 추가 완료")
    else:
        print(f"Trainer 사용가능: False")