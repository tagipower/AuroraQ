#!/usr/bin/env python3
"""
통합 PPO 시스템 - AuroraQ
최적화되고 통합된 PPO 강화학습 시스템
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 프로젝트 모듈
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# 프로젝트 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.path_config import get_model_path
except ImportError:
    # 백업 경로 설정
    def get_model_path(name: str) -> Path:
        return Path("models") / f"{name}.zip"

try:
    from core.state_preprocessor import StatePreprocessor
except ImportError:
    StatePreprocessor = None

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO 설정 클래스"""
    
    # 모델 구조
    input_dim: int = 3  # SB3 모델 호환성 (가격수익률, 거래량비율, 추세지표)
    output_dim: int = 3  # HOLD, BUY, SELL
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    # 학습 파라미터
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # 모델 관리
    model_save_path: str = "models/ppo_optimized"
    device: str = "auto"  # auto, cpu, cuda
    normalize_observations: bool = True
    
    # 신호 생성
    confidence_threshold: float = 0.6
    position_size: float = 0.1
    use_deterministic: bool = True


class PPONetworkOptimized(nn.Module):
    """최적화된 PPO 네트워크 (Policy + Value 통합)"""
    
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        # 공유 특성 추출기
        self.shared_layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
            nn.Softmax(dim=-1)
        )
        
        # 가치 헤드 (상태 가치)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 정규화 통계
        self.register_buffer('obs_mean', torch.zeros(config.input_dim))
        self.register_buffer('obs_std', torch.ones(config.input_dim))
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파: (action_probs, value)"""
        if self.config.normalize_observations:
            x = self._normalize_obs(x)
        
        shared_features = self.shared_layers(x)
        action_probs = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return action_probs, value
    
    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """관측값 정규화"""
        return (x - self.obs_mean) / (self.obs_std + 1e-8)
    
    def update_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """정규화 통계 업데이트"""
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std)
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """행동 선택"""
        with torch.no_grad():
            action_probs, _ = self.forward(obs)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
                confidence = action_probs.max(dim=-1)[0]
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                confidence = action_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            
            return action.item(), confidence.item()


class PPOSignalResult:
    """PPO 신호 결과 클래스"""
    
    def __init__(self, action: str, confidence: float, 
                 price: float, position_size: float = 0.1,
                 metadata: Optional[Dict[str, Any]] = None):
        self.action = action
        self.confidence = confidence
        self.price = price
        self.position_size = position_size
        self.side = action if action in ["BUY", "SELL"] else None
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.strategy_name = "PPOStrategy"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "action": self.action,
            "side": self.side,
            "confidence": self.confidence,
            "price": self.price,
            "position_size": self.position_size,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy_name,
            "metadata": self.metadata
        }


class PPOModelManager:
    """PPO 모델 관리자"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = self._get_device()
        self.model: Optional[PPONetworkOptimized] = None
        self.sb3_model: Optional[PPO] = None
        self.model_type: str = "none"  # "pytorch", "sb3", "none"
        
        logger.info(f"PPOModelManager 초기화됨 (device: {self.device})")
    
    def _get_device(self) -> str:
        """디바이스 자동 선택"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """모델 로드 (PyTorch 우선, SB3 백업)"""
        if model_path is None:
            model_path = self.config.model_save_path
        
        # PyTorch 모델 시도
        pytorch_path = f"{model_path}.pt"
        if os.path.exists(pytorch_path):
            if self._load_pytorch_model(pytorch_path):
                self.model_type = "pytorch"
                logger.info(f"PyTorch PPO 모델 로드됨: {pytorch_path}")
                return True
        
        # SB3 모델 시도
        if SB3_AVAILABLE:
            sb3_paths = [f"{model_path}.zip", "models/ppo_latest.zip", 
                        "models/best_model.zip", "models/final_ppo_model_sentiment.zip"]
            
            for sb3_path in sb3_paths:
                if os.path.exists(sb3_path):
                    if self._load_sb3_model(sb3_path):
                        self.model_type = "sb3"
                        logger.info(f"SB3 PPO 모델 로드됨: {sb3_path}")
                        return True
        
        logger.warning("PPO 모델을 찾을 수 없습니다")
        return False
    
    def _load_pytorch_model(self, path: str) -> bool:
        """PyTorch 모델 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # 설정 업데이트
            if "config" in checkpoint:
                self.config = checkpoint["config"]
            
            # 모델 생성 및 로드
            self.model = PPONetworkOptimized(self.config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # 정규화 통계 로드
            if "obs_mean" in checkpoint and "obs_std" in checkpoint:
                self.model.update_normalization_stats(
                    checkpoint["obs_mean"], checkpoint["obs_std"]
                )
            
            return True
        except Exception as e:
            logger.error(f"PyTorch 모델 로드 실패: {e}")
            return False
    
    def _load_sb3_model(self, path: str) -> bool:
        """SB3 모델 로드"""
        try:
            self.sb3_model = PPO.load(path, device=self.device)
            return True
        except Exception as e:
            logger.error(f"SB3 모델 로드 실패: {e}")
            return False
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """모델 저장 (PyTorch만)"""
        if self.model is None:
            logger.error("저장할 PyTorch 모델이 없습니다")
            return False
        
        if path is None:
            path = f"{self.config.model_save_path}.pt"
        
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "obs_mean": self.model.obs_mean,
                "obs_std": self.model.obs_std,
                "timestamp": datetime.now().isoformat()
            }, path)
            
            logger.info(f"PyTorch PPO 모델 저장됨: {path}")
            return True
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, float]:
        """예측 수행"""
        if self.model_type == "pytorch" and self.model is not None:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            return self.model.get_action(obs_tensor, deterministic)
        
        elif self.model_type == "sb3" and self.sb3_model is not None:
            action, _ = self.sb3_model.predict(observation, deterministic=deterministic)
            # SB3는 신뢰도를 제공하지 않으므로 기본값 사용
            confidence = 0.7 if deterministic else 0.5
            return int(action), confidence
        
        else:
            logger.warning("사용 가능한 PPO 모델이 없습니다")
            return 0, 0.0  # HOLD with low confidence


class PPOStrategyOptimized:
    """최적화된 PPO 전략 클래스"""
    
    def __init__(self, config: Optional[PPOConfig] = None):
        """
        Args:
            config: PPO 설정 (None이면 기본값 사용)
        """
        self.config = config or PPOConfig()
        self.name = "PPOStrategy"
        self.model_manager = PPOModelManager(self.config)
        self.preprocessor = self._create_preprocessor()
        self.is_loaded = False
        
        # 모델 자동 로드
        self._initialize()
    
    def _initialize(self):
        """초기화"""
        try:
            self.is_loaded = self.model_manager.load_model()
            if self.is_loaded:
                logger.info("PPO 전략 초기화 완료")
            else:
                logger.warning("PPO 모델 로드 실패 - 더미 모드로 동작")
        except Exception as e:
            logger.error(f"PPO 전략 초기화 실패: {e}")
            self.is_loaded = False
    
    def _create_preprocessor(self):
        """상태 전처리기 생성"""
        if StatePreprocessor is not None:
            try:
                return StatePreprocessor(normalize=self.config.normalize_observations)
            except Exception:
                pass
        
        # StatePreprocessor가 없는 경우 간단한 처리기 사용
        class SimplePreprocessor:
            def preprocess(self, data):
                if isinstance(data, dict):
                    return self._extract_features(data)
                return np.array(data, dtype=np.float32)
            
            def _extract_features(self, price_data: Dict) -> np.ndarray:
                """가격 데이터에서 특성 추출 (SB3 모델 호환성을 위해 3차원)"""
                features = []
                
                # 가격 수익률
                if "close" in price_data:
                    closes = self._safe_array_conversion(price_data["close"])
                    if len(closes) >= 2:
                        returns = (closes[-1] - closes[-2]) / closes[-2]
                        features.append(float(returns))
                    else:
                        features.append(0.0)
                
                # 거래량 비율
                if "volume" in price_data:
                    volumes = self._safe_array_conversion(price_data["volume"])
                    if len(volumes) >= 2:
                        vol_ratio = volumes[-1] / volumes[-2] if volumes[-2] > 0 else 1.0
                        features.append(float(vol_ratio))
                    else:
                        features.append(1.0)
                
                # 추세 지표 (간단한 RSI 근사)
                if "close" in price_data:
                    closes = self._safe_array_conversion(price_data["close"])
                    if len(closes) >= 5:
                        # 최근 5일 평균 대비 현재 가격 비율
                        avg_price = np.mean(closes[-5:])
                        trend = (closes[-1] - avg_price) / avg_price
                        features.append(float(trend))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
                
                # SB3 모델은 3차원 입력을 기대하므로 정확히 3개 특성만 반환
                while len(features) < 3:
                    features.append(0.0)
                
                return np.array(features[:3], dtype=np.float32)
            
            def _safe_array_conversion(self, data) -> np.ndarray:
                """안전한 배열 변환 (Timestamp 처리 포함)"""
                try:
                    # pandas Series나 DataFrame인 경우
                    if hasattr(data, 'values'):
                        values = data.values
                    else:
                        values = data
                    
                    # Timestamp 객체 처리
                    converted_values = []
                    for val in values:
                        if hasattr(val, 'timestamp'):  # pandas.Timestamp
                            converted_values.append(float(val.timestamp()))
                        elif hasattr(val, 'value'):  # numpy.datetime64
                            converted_values.append(float(val.value))
                        else:
                            converted_values.append(float(val))
                    
                    return np.array(converted_values, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"배열 변환 실패, 기본값 사용: {e}")
                    return np.array([0.0], dtype=np.float32)
        
        return SimplePreprocessor()
    
    def generate_signal(self, price_data: Union[Dict, pd.DataFrame]) -> Optional[PPOSignalResult]:
        """트레이딩 신호 생성"""
        if not self.is_loaded:
            return self._generate_dummy_signal(price_data)
        
        try:
            # DataFrame을 Dict로 변환 (필요한 경우)
            processed_data = self._prepare_data_for_processing(price_data)
            
            # 데이터 전처리
            observation = self.preprocessor.preprocess(processed_data)
            
            # 예측 수행
            action_idx, confidence = self.model_manager.predict(
                observation, deterministic=self.config.use_deterministic
            )
            
            # 신호 변환
            return self._convert_to_signal(action_idx, confidence, price_data)
            
        except Exception as e:
            logger.error(f"PPO 신호 생성 실패: {e}")
            return self._generate_dummy_signal(price_data)
    
    def _prepare_data_for_processing(self, price_data: Union[Dict, pd.DataFrame]) -> Dict:
        """데이터 전처리를 위해 DataFrame을 Dict로 변환"""
        try:
            if isinstance(price_data, pd.DataFrame):
                # DataFrame을 Dict로 변환하면서 Timestamp 안전 처리
                processed_dict = {}
                for col in ['close', 'open', 'high', 'low', 'volume']:
                    if col in price_data.columns:
                        # 수치형 데이터만 추출
                        col_data = price_data[col]
                        if col_data.dtype == 'object' or 'datetime' in str(col_data.dtype):
                            # Timestamp나 object 타입인 경우 스킵
                            continue
                        processed_dict[col] = col_data.values.tolist()
                
                # 최소한 close 데이터가 있어야 함
                if 'close' not in processed_dict and len(price_data) > 0:
                    # 마지막 행의 숫자 컬럼들 중 첫 번째를 close로 사용
                    numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        processed_dict['close'] = price_data[numeric_cols[0]].values.tolist()
                
                return processed_dict
            
            elif isinstance(price_data, dict):
                return price_data
            
            else:
                # 다른 타입인 경우 기본 구조 반환
                return {"close": [100], "volume": [1000]}
                
        except Exception as e:
            logger.warning(f"데이터 전처리 중 오류, 기본값 사용: {e}")
            return {"close": [100], "volume": [1000]}
    
    def _convert_to_signal(self, action_idx: int, confidence: float, 
                          price_data: Union[Dict, pd.DataFrame]) -> PPOSignalResult:
        """액션을 트레이딩 신호로 변환"""
        # 현재 가격 추출
        if isinstance(price_data, dict):
            current_price = float(price_data.get("close", [0])[-1]) if "close" in price_data else 0.0
        elif isinstance(price_data, pd.DataFrame):
            current_price = float(price_data["close"].iloc[-1]) if "close" in price_data.columns else 0.0
        else:
            current_price = 0.0
        
        # 액션 매핑
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action = action_map.get(action_idx, "HOLD")
        
        # 신뢰도 필터링
        if confidence < self.config.confidence_threshold:
            action = "HOLD"
            confidence = 0.5
        
        return PPOSignalResult(
            action=action,
            confidence=confidence,
            price=current_price,
            position_size=self.config.position_size,
            metadata={
                "model_type": self.model_manager.model_type,
                "raw_action": action_idx,
                "threshold": self.config.confidence_threshold
            }
        )
    
    def _generate_dummy_signal(self, price_data: Union[Dict, pd.DataFrame]) -> PPOSignalResult:
        """더미 신호 생성 (모델 없을 때)"""
        if isinstance(price_data, dict):
            current_price = float(price_data.get("close", [0])[-1]) if "close" in price_data else 0.0
        elif isinstance(price_data, pd.DataFrame):
            current_price = float(price_data["close"].iloc[-1]) if "close" in price_data.columns else 0.0
        else:
            current_price = 0.0
        
        return PPOSignalResult(
            action="HOLD",
            confidence=0.1,
            price=current_price,
            position_size=0.0,
            metadata={"dummy": True, "reason": "No model loaded"}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "strategy_name": self.name,
            "is_loaded": self.is_loaded,
            "model_type": self.model_manager.model_type,
            "device": self.model_manager.device,
            "config": self.config.__dict__,
            "model_path": self.config.model_save_path
        }
    
    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """모델 리로드"""
        logger.info("PPO 모델 리로드 중...")
        self.is_loaded = self.model_manager.load_model(model_path)
        return self.is_loaded


# 하위 호환성을 위한 별칭
PPOStrategy = PPOStrategyOptimized


def create_optimized_ppo_strategy(config_dict: Optional[Dict[str, Any]] = None) -> PPOStrategyOptimized:
    """최적화된 PPO 전략 팩토리 함수"""
    if config_dict:
        config = PPOConfig(**config_dict)
    else:
        config = PPOConfig()
    
    return PPOStrategyOptimized(config)


# 테스트 및 예제
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    print("=== PPO 최적화 시스템 테스트 ===")
    
    # 전략 생성
    strategy = PPOStrategyOptimized()
    
    # 모델 정보 출력
    info = strategy.get_model_info()
    print(f"모델 정보: {info}")
    
    # 샘플 데이터로 신호 생성 테스트
    sample_data = {
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1100, 900, 1200, 1050],
        "timestamp": datetime.now()
    }
    
    signal = strategy.generate_signal(sample_data)
    if signal:
        print(f"생성된 신호: {signal.to_dict()}")
    else:
        print("신호 생성 실패")
    
    print("테스트 완료!")