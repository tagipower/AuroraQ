# models/ppo_model.py

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from core.path_config import get_model_path

logger = logging.getLogger(__name__)

class PPOPolicyNetwork(nn.Module):
    """
    PPO 정책 네트워크 (Policy Network)
    트레이딩 행동의 확률 분포를 출력하는 신경망
    """
    
    def __init__(
        self, 
        input_dim: int = 5, 
        output_dim: int = 3,  # HOLD, BUY, SELL
        hidden_dim: int = 128,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 feature 차원
            output_dim: 출력 행동 개수 (3: HOLD, BUY, SELL)
            hidden_dim: 은닉층 크기
            dropout_rate: 드롭아웃 비율
        """
        super(PPOPolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 숫값 정규화를 위한 통계
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
        # 주 네트워크
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate)
        )
        
        # 중간층
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 출력층
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1)  # 확률 분포
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        logger.info(
            f"PPOPolicyNetwork initialized: "
            f"input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward pass)
        
        Args:
            x: 입력 텐서 [batch_size, input_dim]
            
        Returns:
            행동 확률 분포 [batch_size, output_dim]
        """
        # 입력 정규화
        x = self._normalize_input(x)
        
        # 특징 추출
        features = self.feature_extractor(x)
        
        # 중간층 통과
        hidden = self.hidden_layers(features)
        
        # 출력 확률
        output = self.output_layer(hidden)
        
        return output
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """입력 정규화"""
        # 안전한 정규화 (표준편차가 0인 경우 처리)
        std = self.input_std.clamp(min=1e-8)
        return (x - self.input_mean) / std
    
    def update_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """정규화 통계 업데이트"""
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
        logger.debug("Normalization statistics updated")
    
    def _initialize_weights(self):
        """가중치 초기화 (Xavier/He initialization)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        상태에서 행동 선택
        
        Args:
            state: 현재 상태
            deterministic: True면 가장 높은 확률의 행동 선택
            
        Returns:
            (action, action_probs)
        """
        with torch.no_grad():
            action_probs = self.forward(state)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                # 확률적 샘플링
                action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            
            return action.item(), action_probs


class PPOValueNetwork(nn.Module):
    """
    PPO 가치 네트워크 (Value Network)
    현재 상태의 가치를 추정하는 신경망
    """
    
    def __init__(
        self, 
        input_dim: int = 5, 
        hidden_dim: int = 128,
        dropout_rate: float = 0.1
    ):
        super(PPOValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)  # 단일 가치 출력
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        logger.info(f"PPOValueNetwork initialized with hidden_dim={hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


def save_ppo_networks(
    policy_net: PPOPolicyNetwork,
    value_net: Optional[PPOValueNetwork] = None,
    save_dir: Optional[str] = None,
    prefix: str = "ppo"
) -> Dict[str, Path]:
    """
    PPO 네트워크 저장
    
    Args:
        policy_net: 정책 네트워크
        value_net: 가치 네트워크 (optional)
        save_dir: 저장 디렉토리
        prefix: 파일 이름 접두사
        
    Returns:
        저장된 파일 경로들
    """
    if save_dir is None:
        save_dir = get_model_path("").parent
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    
    try:
        # 정책 네트워크 저장
        policy_path = save_dir / f"{prefix}_policy.pt"
        torch.save({
            'state_dict': policy_net.state_dict(),
            'input_dim': policy_net.input_dim,
            'output_dim': policy_net.output_dim,
            'hidden_dim': policy_net.hidden_dim,
            'input_mean': policy_net.input_mean,
            'input_std': policy_net.input_std
        }, policy_path)
        saved_paths['policy'] = policy_path
        logger.info(f"Policy network saved to: {policy_path}")
        
        # 가치 네트워크 저장 (있는 경우)
        if value_net is not None:
            value_path = save_dir / f"{prefix}_value.pt"
            torch.save({
                'state_dict': value_net.state_dict()
            }, value_path)
            saved_paths['value'] = value_path
            logger.info(f"Value network saved to: {value_path}")
        
        return saved_paths
        
    except Exception as e:
        logger.error(f"Failed to save networks: {e}", exc_info=True)
        raise


def load_ppo_policy_network(
    path: Optional[str] = None,
    device: Optional[str] = None
) -> PPOPolicyNetwork:
    """
    저장된 PPO 정책 네트워크 로드
    
    Args:
        path: 모델 파일 경로
        device: 사용할 디바이스
        
    Returns:
        로드된 정책 네트워크
    """
    if path is None:
        path = get_model_path("ppo_policy")
    else:
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # 네트워크 생성
        model = PPOPolicyNetwork(
            input_dim=checkpoint.get('input_dim', 5),
            output_dim=checkpoint.get('output_dim', 3),
            hidden_dim=checkpoint.get('hidden_dim', 128)
        )
        
        # 가중치 로드
        model.load_state_dict(checkpoint['state_dict'])
        
        # 정규화 통계 로드
        if 'input_mean' in checkpoint and 'input_std' in checkpoint:
            model.update_normalization_stats(
                checkpoint['input_mean'],
                checkpoint['input_std']
            )
        
        model.to(device)
        model.eval()
        
        logger.info(f"PPO policy network loaded from: {path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load policy network: {e}", exc_info=True)
        raise


# 테스트 코드
if __name__ == "__main__":
    # 모델 생성
    policy_net = PPOPolicyNetwork(input_dim=5, output_dim=3)
    value_net = PPOValueNetwork(input_dim=5)
    
    # 샘플 입력 테스트
    sample_input = torch.randn(1, 5)  # 배치 크기 1, 5차원 입력
    
    # 정책 네트워크 테스트
    action_probs = policy_net(sample_input)
    action, probs = policy_net.get_action(sample_input, deterministic=True)
    print(f"Policy output shape: {action_probs.shape}")
    print(f"Action probabilities: {action_probs}")
    print(f"Selected action: {action}")
    
    # 가치 네트워크 테스트
    value = value_net(sample_input)
    print(f"\nValue output shape: {value.shape}")
    print(f"State value: {value.item():.4f}")
    
    logger.info("PPO networks test completed successfully")
