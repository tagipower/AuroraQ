import yaml

def load_trade_config(path="config/trade_config.yaml"):
    """
    trading 설정 전용 함수
    예: 주문 방식, 슬리피지, 거래소 API 설정 등
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_yaml_config(path: str):
    """
    범용 YAML 설정 로딩 함수
    예: strategy_weight.yaml, reward_config.yaml 등
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def save_yaml_config(data: dict, path: str):
    """
    범용 YAML 설정 저장 함수
    """
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True, sort_keys=False)
