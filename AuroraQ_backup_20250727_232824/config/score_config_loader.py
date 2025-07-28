import yaml
import os

def load_score_weights(config_path="config/score_config.yaml") -> dict:
    """
    전략 점수 평가에 사용될 메트릭 가중치를 YAML에서 로드합니다.
    
    Args:
        config_path (str): 가중치 설정 파일 경로

    Returns:
        dict: 메트릭별 가중치 (예: {'sharpe': 0.4, 'win_rate': 0.2, ...})
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"score_config.yaml not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config.get("weights", {})
