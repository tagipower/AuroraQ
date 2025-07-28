# core/position_tracker.py

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from core.path_config import get_log_path

logger = logging.getLogger(__name__)

class PositionTracker:
    """포지션 상태 추적 및 관리 클래스"""
    
    def __init__(self, persist_file: Optional[str] = None):
        """
        Args:
            persist_file: 포지션 상태를 저장할 파일 경로 (None이면 기본 경로 사용)
        """
        if persist_file:
            self.persist_file = Path(persist_file)
        else:
            self.persist_file = get_log_path("position_state")
        self.position: Optional[str] = None
        self.last_update: datetime = datetime.utcnow()
        
        # 파일에서 초기 상태 로드
        if self.persist_file:
            self.position = self._load_from_file()
            logger.info(f"PositionTracker initialized with position: {self.position}")

    def set_position(self, position: Optional[str]) -> bool:
        """
        포지션 설정
        
        Args:
            position: 'long', 'short', 또는 None
            
        Returns:
            bool: 설정 성공 여부
        """
        if position not in ("long", "short", None):
            logger.error(f"Invalid position value: {position}")
            raise ValueError("position must be 'long', 'short', or None")
        
        old_position = self.position
        self.position = position
        self.last_update = datetime.utcnow()
        
        # 상태 변경 로깅
        if old_position != position:
            logger.info(f"Position changed: {old_position} -> {position}")
        
        # 파일에 저장
        if self.persist_file:
            return self._save_to_file()
        return True

    def get_position(self) -> Optional[str]:
        """현재 포지션 조회"""
        return self.position
    
    def get_position_info(self) -> Dict[str, Any]:
        """포지션 상태와 메타데이터 조회"""
        return {
            "position": self.position,
            "last_update": self.last_update.isoformat(),
            "persist_file": str(self.persist_file) if self.persist_file else None
        }

    def _save_to_file(self) -> bool:
        """포지션 상태를 파일에 저장"""
        try:
            # 디렉토리 생성
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "timestamp": self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                "position": self.position,
                "updated_by": "PositionTracker"
            }
            
            # 임시 파일에 먼저 쓰고 원자적으로 교체
            temp_file = self.persist_file.with_suffix('.tmp')
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            # 원자적 파일 교체 (Windows에서도 안전)
            temp_file.replace(self.persist_file)
            
            logger.debug(f"Position saved to {self.persist_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save position to file: {e}", exc_info=True)
            return False

    def _load_from_file(self) -> Optional[str]:
        """파일에서 포지션 상태 로드"""
        if not self.persist_file.exists():
            logger.debug(f"Position file not found: {self.persist_file}")
            return None
        
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 타임스탬프 파싱 시도
            if "timestamp" in data:
                try:
                    timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
                    self.last_update = timestamp
                except ValueError:
                    logger.warning(f"Invalid timestamp format in position file")
            
            position = data.get("position")
            logger.debug(f"Position loaded from file: {position}")
            return position
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in position file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load position from file: {e}", exc_info=True)
            return None
    
    def clear_position(self) -> bool:
        """포지션 초기화"""
        return self.set_position(None)
