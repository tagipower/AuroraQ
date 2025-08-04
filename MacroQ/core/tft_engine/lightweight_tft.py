"""
경량화된 Temporal Fusion Transformer 구현
CPU 친화적이며 VPS에서 실행 가능하도록 최적화
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    """TFT 모델 설정"""
    # 입력 차원
    num_static_features: int = 5      # 정적 특성 (자산 타입 등)
    num_time_features: int = 10       # 시계열 특성 (가격, 거래량 등)
    num_known_features: int = 3       # 미래 알려진 특성 (이벤트 등)
    
    # 모델 차원 (경량화)
    hidden_size: int = 128            # 작은 히든 크기
    num_attention_heads: int = 4      # 적은 어텐션 헤드
    num_lstm_layers: int = 2          
    dropout_rate: float = 0.1
    
    # 예측 설정
    prediction_horizons: List[int] = None  # [7, 30, 90] 일
    num_assets: int = 5               # 초기에는 5개 자산만
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [7, 30, 90]


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network - 변수 중요도 학습"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)
        
        # Gating mechanism
        self.gate_dense1 = nn.Linear(input_size, hidden_size)
        self.gate_dense2 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection
        self.skip_layer = nn.Linear(input_size, output_size) if input_size != output_size else None
        
    def forward(self, x):
        # Regular path
        hidden = self.elu(self.dense1(x))
        hidden = self.dropout(hidden)
        output = self.dense2(hidden)
        
        # Gating path
        gate_hidden = self.elu(self.gate_dense1(x))
        gate_hidden = self.dropout(gate_hidden)
        gate = self.sigmoid(self.gate_dense2(gate_hidden))
        
        # Apply gating
        output = gate * output
        
        # Skip connection
        if self.skip_layer:
            output = output + self.skip_layer(x)
        else:
            output = output + x
            
        return output


class LightweightTFT(nn.Module):
    """
    경량화된 TFT 모델
    - CPU에서 실행 가능하도록 최적화
    - 메모리 효율적인 구조
    - 빠른 추론 속도
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Variable Selection Networks
        self.static_vsn = GatedResidualNetwork(
            config.num_static_features,
            config.hidden_size // 2,
            config.hidden_size
        )
        
        self.temporal_vsn = GatedResidualNetwork(
            config.num_time_features,
            config.hidden_size // 2,
            config.hidden_size
        )
        
        # LSTM Encoder (경량화)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,  # 작은 LSTM
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout_rate
        )
        
        # Attention (간단한 self-attention)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size // 2,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Output layers for each horizon
        self.output_layers = nn.ModuleDict({
            str(horizon): nn.Linear(config.hidden_size // 2, config.num_assets)
            for horizon in config.prediction_horizons
        })
        
        # Quantile layers (예측 불확실성)
        self.quantile_layers = nn.ModuleDict({
            str(horizon): nn.Linear(config.hidden_size // 2, config.num_assets * 3)  # 0.1, 0.5, 0.9 quantiles
            for horizon in config.prediction_horizons
        })
        
        logger.info(f"Initialized LightweightTFT with config: {config}")
        
    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: torch.Tensor,
        known_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            static_features: (batch_size, num_static_features)
            temporal_features: (batch_size, seq_len, num_time_features)
            known_features: (batch_size, num_horizons, num_known_features)
            
        Returns:
            Dict with predictions for each horizon
        """
        batch_size = temporal_features.size(0)
        seq_len = temporal_features.size(1)
        
        # Variable selection
        static_encoded = self.static_vsn(static_features)  # (batch, hidden)
        
        # Temporal encoding
        temporal_encoded = []
        for t in range(seq_len):
            t_encoded = self.temporal_vsn(temporal_features[:, t, :])
            # Add static context
            t_encoded = t_encoded + static_encoded
            temporal_encoded.append(t_encoded)
            
        temporal_encoded = torch.stack(temporal_encoded, dim=1)  # (batch, seq_len, hidden)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(temporal_encoded)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state for prediction
        final_hidden = attn_out[:, -1, :]  # (batch, hidden//2)
        
        # Generate predictions for each horizon
        predictions = {}
        quantiles = {}
        
        for horizon in self.config.prediction_horizons:
            horizon_key = str(horizon)
            
            # Point predictions
            predictions[f'horizon_{horizon}'] = self.output_layers[horizon_key](final_hidden)
            
            # Quantile predictions
            q_out = self.quantile_layers[horizon_key](final_hidden)
            q_out = q_out.view(batch_size, self.config.num_assets, 3)
            quantiles[f'horizon_{horizon}_quantiles'] = q_out
            
        # Attention weights for interpretability
        predictions['attention_weights'] = attn_weights
        predictions.update(quantiles)
        
        return predictions
    
    def predict(
        self,
        market_data: pd.DataFrame,
        static_info: Dict[str, Any],
        future_events: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        고수준 예측 인터페이스
        
        Args:
            market_data: 시장 데이터 (OHLCV)
            static_info: 자산 정보
            future_events: 미래 이벤트
            
        Returns:
            예측 결과
        """
        self.eval()
        
        # 데이터 전처리
        static_features, temporal_features = self._preprocess_data(
            market_data, static_info
        )
        
        with torch.no_grad():
            predictions = self.forward(static_features, temporal_features)
            
        # 후처리
        results = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu().numpy()
                
        return results
    
    def _preprocess_data(
        self,
        market_data: pd.DataFrame,
        static_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터 전처리"""
        # 간단한 전처리 (실제로는 더 복잡)
        
        # Static features
        static_features = torch.zeros(1, self.config.num_static_features)
        
        # Temporal features (가격 변화율, 거래량 등)
        returns = market_data['close'].pct_change().fillna(0)
        volume_norm = market_data['volume'] / market_data['volume'].mean()
        
        temporal_data = pd.DataFrame({
            'returns': returns,
            'volume_norm': volume_norm,
            'high_low_ratio': market_data['high'] / market_data['low'],
            'close_open_ratio': market_data['close'] / market_data['open']
        })
        
        # 추가 특성들
        for i in range(self.config.num_time_features - 4):
            temporal_data[f'feature_{i}'] = 0
            
        temporal_features = torch.tensor(
            temporal_data.values[-100:],  # 최근 100개 시점
            dtype=torch.float32
        ).unsqueeze(0)
        
        return static_features, temporal_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """변수 중요도 추출 (GRN 가중치 기반)"""
        importance = {}
        
        # Static VSN weights
        static_weights = self.static_vsn.gate_dense2.weight.abs().mean(dim=0)
        for i, w in enumerate(static_weights):
            importance[f'static_feature_{i}'] = float(w)
            
        # Temporal VSN weights  
        temporal_weights = self.temporal_vsn.gate_dense2.weight.abs().mean(dim=0)
        for i, w in enumerate(temporal_weights):
            importance[f'temporal_feature_{i}'] = float(w)
            
        return importance