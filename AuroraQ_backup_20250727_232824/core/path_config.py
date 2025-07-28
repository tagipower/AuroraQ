# core/path_config.py

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

# 기본 경로 설정
BASE_DIR = Path(__file__).parent.parent  # AuroraQ 루트 디렉토리
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"

# 로그 파일 경로
LOG_PATHS = {
    "position_state": LOGS_DIR / "position_state.json",
    "regime_status": LOGS_DIR / "regime_status.json",
    "mab_score": LOGS_DIR / "mab_score_log.csv",
    "strategy_selection": LOGS_DIR / "strategy_selection_log.csv",
    "trade_log": LOGS_DIR / "trade_log.csv",
}

# 데이터 파일 경로
DATA_PATHS = {
    "sentiment": DATA_DIR / "sentiment" / "news_sentiment_log.csv",
    "price_data": DATA_DIR / "price" / "btc_price_data.csv",
}

# 설정 파일 경로
CONFIG_PATHS = {
    "strategy_weight": CONFIG_DIR / "strategy_weight.yaml",
    "mab_config": CONFIG_DIR / "mab_config.yaml",
    "trade_config": CONFIG_DIR / "trade_config.yaml",
}

# 모델 파일 경로
MODEL_PATHS = {
    "ppo_model": MODELS_DIR / "ppo_model.zip",
    "ppo_weights": MODELS_DIR / "ppo_weights.pt",
}


class PathConfig:
    """경로 설정 관리 클래스"""
    
    def __init__(self, custom_config_path: Optional[str] = None):
        """
        Args:
            custom_config_path: 사용자 정의 경로 설정 파일
        """
        self.base_dir = BASE_DIR
        self.custom_config_path = custom_config_path
        
        # 기본 경로 설정
        self.paths = {
            "logs": LOG_PATHS.copy(),
            "data": DATA_PATHS.copy(),
            "config": CONFIG_PATHS.copy(),
            "models": MODEL_PATHS.copy(),
        }
        
        # 사용자 설정 로드
        if custom_config_path and Path(custom_config_path).exists():
            self._load_custom_config(custom_config_path)
        
        # 필수 디렉토리 생성
        self._ensure_directories()
    
    def _load_custom_config(self, config_path: str):
        """사용자 정의 경로 설정 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                custom_paths = json.load(f)
            
            # 사용자 설정으로 기본 경로 업데이트
            for category, paths in custom_paths.items():
                if category in self.paths:
                    for key, path in paths.items():
                        self.paths[category][key] = Path(path)
                        
            logger.info(f"Custom path configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load custom path config: {e}")
    
    def _ensure_directories(self):
        """필수 디렉토리 생성"""
        directories = {
            LOGS_DIR,
            DATA_DIR,
            DATA_DIR / "sentiment",
            DATA_DIR / "price",
            CONFIG_DIR,
            MODELS_DIR,
        }
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug("All required directories ensured")
    
    def get_log_path(self, name: str) -> Path:
        """로그 파일 경로 조회"""
        return self.paths["logs"].get(name, LOGS_DIR / f"{name}.log")
    
    def get_data_path(self, name: str) -> Path:
        """데이터 파일 경로 조회"""
        return self.paths["data"].get(name, DATA_DIR / f"{name}.csv")
    
    def get_config_path(self, name: str) -> Path:
        """설정 파일 경로 조회"""
        return self.paths["config"].get(name, CONFIG_DIR / f"{name}.yaml")
    
    def get_model_path(self, name: str) -> Path:
        """모델 파일 경로 조회"""
        return self.paths["models"].get(name, MODELS_DIR / f"{name}.pt")
    
    def get_all_paths(self) -> Dict[str, Dict[str, Path]]:
        """모든 경로 설정 반환"""
        return self.paths.copy()
    
    def save_custom_config(self, output_path: str):
        """현재 경로 설정을 파일로 저장"""
        try:
            config_data = {}
            for category, paths in self.paths.items():
                config_data[category] = {
                    key: str(path) for key, path in paths.items()
                }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Path configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save path config: {e}")


# 전역 경로 설정 인스턴스
_path_config = PathConfig()


# 편의 함수들
def get_log_path(name: str) -> Path:
    """로그 파일 경로 조회"""
    return _path_config.get_log_path(name)


def get_data_path(name: str) -> Path:
    """데이터 파일 경로 조회"""
    return _path_config.get_data_path(name)


def get_config_path(name: str) -> Path:
    """설정 파일 경로 조회"""
    return _path_config.get_config_path(name)


def get_model_path(name: str) -> Path:
    """모델 파일 경로 조회"""
    return _path_config.get_model_path(name)


def reload_with_custom_config(config_path: str):
    """사용자 정의 설정으로 재로드"""
    global _path_config
    _path_config = PathConfig(config_path)
    logger.info("Path configuration reloaded")