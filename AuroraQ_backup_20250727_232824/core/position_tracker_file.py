# core/position_tracker_file.py

import os
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from core.path_config import get_log_path

logger = logging.getLogger(__name__)

# 전역 설정
_lock = threading.Lock()


def get_position_file_path() -> Path:
    """포지션 파일 경로 조회"""
    return get_log_path("position_state")


def set_current_position(position: Optional[str]) -> bool:
    """
    현재 포지션 상태를 파일에 저장
    
    Args:
        position: 'long', 'short', 또는 None (포지션 없음)
        
    Returns:
        bool: 저장 성공 여부
    """
    if position not in ("long", "short", None):
        logger.error(f"Invalid position value: {position}")
        return False
    
    with _lock:
        try:
            position_file = get_position_file_path()
            position_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "position": position,
                "updated_by": "position_tracker_file"
            }
            
            with open(position_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Position saved: {position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            return False


def get_current_position() -> Optional[str]:
    """
    파일에서 현재 포지션 상태 읽기
    
    Returns:
        Optional[str]: 'long', 'short', 또는 None
    """
    with _lock:
        try:
            position_file = get_position_file_path()
            if not position_file.exists():
                return None
            
            with open(position_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("position")
                
        except Exception as e:
            logger.error(f"Failed to read position: {e}")
            return None


def get_position_info() -> Dict[str, Any]:
    """
    포지션 상태와 메타데이터 조회
    
    Returns:
        Dict: 포지션 정보 (timestamp, position, updated_by)
    """
    with _lock:
        try:
            position_file = get_position_file_path()
            if not position_file.exists():
                return {
                    "timestamp": None,
                    "position": None,
                    "updated_by": None
                }
            
            with open(position_file, "r", encoding="utf-8") as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to read position info: {e}")
            return {
                "timestamp": None,
                "position": None,
                "updated_by": None,
                "error": str(e)
            }


def clear_position_file() -> bool:
    """
    포지션 파일 초기화 (테스트/리셋용)
    
    Returns:
        bool: 삭제 성공 여부
    """
    with _lock:
        try:
            position_file = get_position_file_path()
            if position_file.exists():
                position_file.unlink()
                logger.info("Position file cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear position file: {e}")
            return False