# utils/reward_shaping_loader.py

import yaml
import os


def load_reward_shaping_config(path="config/reward_schema.yaml") -> dict:
    """
    reward_schema.yaml 파일을 로딩하고, reward_shaping / weights / thresholds 항목을 검증하여 반환합니다.

    :param path: yaml 파일 경로
    :return: dict 구조의 설정값
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[reward_shaping_loader] 설정 파일이 존재하지 않습니다: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 필수 섹션 검증
    required_sections = ["reward_shaping", "weights", "thresholds"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"[reward_shaping_loader] reward_schema.yaml에 '{section}' 섹션이 누락되었습니다.")

    return config
