# 📁 backtest/model_loader_backtest.py

import os
from stable_baselines3 import PPO

def load_ppo_model(model_path: str):
    """
    PPO 모델을 지정 경로에서 로드합니다.
    
    Args:
        model_path (str): PPO 모델이 저장된 파일 경로 (.zip)

    Returns:
        PPO: 로드된 PPO 모델 객체
        None: 로드 실패 시
    """
    if not os.path.exists(model_path):
        print(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
        return None

    try:
        model = PPO.load(model_path)
        print(f"✅ PPO 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"❌ PPO 모델 로드 실패: {model_path}\n에러: {e}")
        return None
