import yaml
import os

def get_rule_params(rule_name: str):
    """최적화된 파라미터를 우선적으로 로드"""
    
    # 최적화된 파라미터 파일 경로
    optimized_path = os.path.join("config", "optimized_rule_params.yaml")
    
    # 1. 최적화된 파라미터 파일 시도
    if os.path.exists(optimized_path):
        try:
            with open(optimized_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if rule_name in config:
                print(f"[INFO] {rule_name} - 최적화된 파라미터 사용")
                return config.get(rule_name, {})
        except Exception as e:
            print(f"[WARNING] 최적화된 파라미터 로드 실패: {e}")
    
    # 2. 기본 파라미터 파일로 fallback
    default_path = os.path.join("config", "rule_params.yaml")
    
    # ✅ 인코딩 명시적으로 지정
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"[INFO] {rule_name} - 기본 파라미터 사용")
    return config.get(rule_name, {})
