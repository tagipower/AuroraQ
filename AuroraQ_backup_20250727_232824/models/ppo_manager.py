# models/ppo_manager.py

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from stable_baselines3 import PPO
import torch

from core.path_config import get_model_path

logger = logging.getLogger(__name__)

class PPOStrategy:
    """
    PPO (Proximal Policy Optimization) 전략 클래스
    stable-baselines3를 사용한 강화학습 기반 트레이딩 전략
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: PPO 모델 파일 경로 (None이면 기본 경로 사용)
        """
        self.name = "PPOStrategy"
        self.model_path = Path(model_path) if model_path else get_model_path("ppo_model")
        self.model: Optional[PPO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드
        self._load_model()
        logger.info(f"PPOStrategy initialized with device: {self.device}")

    def _load_model(self) -> None:
        """모델 파일 로드"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = PPO.load(
                str(self.model_path),
                device=self.device
            )
            logger.info(f"PPO model loaded successfully from: {self.model_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")

    def generate_signal(self, price_data: Union[Dict, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        가격 데이터를 기반으로 트레이딩 신호 생성
        
        Args:
            price_data: 가격 데이터 (dict 또는 numpy array)
            
        Returns:
            트레이딩 신호 dict 또는 None
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # 데이터 전처리
            obs = self._preprocess(price_data)
            
            # 예측 수행
            action, _ = self.model.predict(obs, deterministic=True)
            
            # 후처리 및 신호 변환
            signal = self._postprocess(action, price_data)
            
            if signal:
                logger.debug(f"PPO signal generated: {signal}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate PPO signal: {e}", exc_info=True)
            return None

    def _preprocess(self, price_data: Union[Dict, np.ndarray]) -> np.ndarray:
        """
        가격 데이터를 PPO 모델 입력 형식으로 전처리
        
        Args:
            price_data: 원본 가격 데이터
            
        Returns:
            전처리된 numpy 배열
        """
        try:
            if isinstance(price_data, dict):
                # 필요한 feature 추출
                features = []
                
                # 가격 데이터
                if "close" in price_data:
                    close_prices = np.array(price_data["close"])
                    if len(close_prices) >= 20:
                        # 최근 20개 데이터의 수익률
                        returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                        features.extend([
                            returns.mean(),  # 평균 수익률
                            returns.std(),   # 변동성
                            returns[-1],     # 최근 수익률
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                
                # 거래량 데이터
                if "volume" in price_data:
                    volumes = np.array(price_data["volume"])
                    if len(volumes) >= 10:
                        vol_ratio = volumes[-1] / volumes[-10:].mean()
                        features.append(vol_ratio)
                    else:
                        features.append(1.0)
                
                # RSI 또는 기타 기술적 지표 추가 가능
                
                return np.array(features, dtype=np.float32).reshape(1, -1)
            
            elif isinstance(price_data, np.ndarray):
                # 이미 전처리된 데이터
                return price_data.reshape(1, -1)
            
            else:
                raise ValueError(f"Unsupported data type: {type(price_data)}")
                
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # 기본값 반환
            return np.zeros((1, 4), dtype=np.float32)

    def _postprocess(self, action: Union[int, np.ndarray], price_data: Dict) -> Optional[Dict[str, Any]]:
        """
        PPO 액션을 트레이딩 신호로 변환
        
        Args:
            action: PPO 모델의 출력 액션
            price_data: 현재 가격 데이터 (신호 생성에 필요)
            
        Returns:
            트레이딩 신호 dict
        """
        try:
            # numpy array인 경우 int로 변환
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            current_price = float(price_data.get("close", [0])[-1]) if "close" in price_data else 0
            
            # 액션 매핑 (0: HOLD, 1: BUY, 2: SELL)
            if action == 0:
                return {
                    "action": "HOLD",
                    "price": current_price,
                    "confidence": 0.5,
                    "reason": "PPO model suggests holding"
                }
            elif action == 1:
                return {
                    "action": "BUY",
                    "side": "BUY",
                    "price": current_price,
                    "size": 0.1,  # 기본 포지션 크기
                    "confidence": 0.7,
                    "reason": "PPO model suggests buying"
                }
            elif action == 2:
                return {
                    "action": "SELL",
                    "side": "SELL",
                    "price": current_price,
                    "size": 0.1,  # 기본 포지션 크기
                    "confidence": 0.7,
                    "reason": "PPO model suggests selling"
                }
            else:
                logger.warning(f"Unknown action: {action}")
                return None
                
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return None


    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 조회"""
        if self.model is None:
            return {"status": "not_loaded", "path": str(self.model_path)}
        
        return {
            "status": "loaded",
            "path": str(self.model_path),
            "device": self.device,
            "policy_class": type(self.model.policy).__name__,
            "observation_space": str(self.model.observation_space),
            "action_space": str(self.model.action_space)
        }
    
    def reload_model(self, new_path: Optional[str] = None) -> None:
        """모델 리로드"""
        if new_path:
            self.model_path = Path(new_path)
        
        logger.info(f"Reloading model from: {self.model_path}")
        self._load_model()


def load_ppo_model(path: Optional[str] = None, device: Optional[str] = None) -> PPO:
    """
    PPO 모델 로드 헬퍼 함수
    
    Args:
        path: 모델 파일 경로 (None이면 기본 경로)
        device: 사용할 디바이스 (None이면 자동 선택)
        
    Returns:
        로드된 PPO 모델
    """
    model_path = Path(path) if path else get_model_path("ppo_model")
    
    if not model_path.exists():
        # 대체 경로 시도
        alt_paths = [
            get_model_path("ppo_latest"),
            get_model_path("best_model"),
            get_model_path("final_ppo_model_sentiment")
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                logger.info(f"Using alternative model path: {alt_path}")
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"No PPO model found in {model_path.parent}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = PPO.load(str(model_path), device=device)
        logger.info(f"PPO model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load PPO model: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}")
