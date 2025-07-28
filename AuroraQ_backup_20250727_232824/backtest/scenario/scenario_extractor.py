import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScenarioConfig:
    """시나리오 설정 클래스"""
    name: str
    min_duration: int = 10
    max_duration: int = 30
    conditions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []

# ------------------------
# 유틸리티 함수
# ------------------------
def parse_duration(duration_str: str) -> Tuple[int, int]:
    """'10~30' 같은 문자열을 (10,30) 튜플로 변환."""
    try:
        if not isinstance(duration_str, str):
            raise ValueError(f"duration_str must be string, got {type(duration_str)}")
            
        parts = duration_str.replace(" ", "").split("~")
        if len(parts) == 2:
            min_val, max_val = int(parts[0]), int(parts[1])
            if min_val <= 0 or max_val <= 0 or min_val > max_val:
                raise ValueError(f"Invalid duration range: {duration_str}")
            return min_val, max_val
        elif len(parts) == 1:
            val = int(parts[0])
            if val <= 0:
                raise ValueError(f"Invalid duration value: {duration_str}")
            return val, val
        else:
            raise ValueError(f"Invalid duration format: {duration_str}")
    except (ValueError, AttributeError) as e:
        logger.warning(f"Duration parsing failed for '{duration_str}': {e}, using default (10, 30)")
        return 10, 30

def validate_dataframe(df: pd.DataFrame) -> bool:
    """데이터프레임 유효성 검증"""
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    required_columns = ['close', 'high', 'low', 'open', 'volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    return True

def parse_condition_value(val: str) -> Tuple[str, List[float]]:
    """조건 값을 파싱하여 연산자와 값 반환"""
    val = val.strip()
    
    if val.startswith(">="):
        return ">=", [float(val[2:])]
    elif val.startswith("<="):
        return "<=", [float(val[2:])]
    elif val.startswith(">"):
        return ">", [float(val[1:])]
    elif val.startswith("<"):
        return "<", [float(val[1:])]
    elif "between" in val.lower():
        vals = val.lower().replace("between", "").replace(" ", "").split(",")
        if len(vals) == 2:
            return "between", [float(vals[0]), float(vals[1])]
    
    raise ValueError(f"Invalid condition format: {val}")

def match_condition(row: pd.Series, condition: Dict[str, Any]) -> bool:
    """단일 행(row)이 조건을 만족하는지 평가."""
    try:
        for key, val in condition.items():
            if key == "duration":
                continue  # duration은 윈도우 길이에만 사용
            
            if not isinstance(val, str):
                logger.warning(f"Condition value for '{key}' is not string: {val}")
                continue
                
            try:
                operator, values = parse_condition_value(val)
                row_value = row.get(key, 0)
                
                if operator == ">" and row_value <= values[0]:
                    return False
                elif operator == "<" and row_value >= values[0]:
                    return False
                elif operator == ">=" and row_value < values[0]:
                    return False
                elif operator == "<=" and row_value > values[0]:
                    return False
                elif operator == "between" and not (values[0] <= row_value <= values[1]):
                    return False
                    
            except ValueError as e:
                logger.warning(f"Failed to parse condition '{key}': {val}, error: {e}")
                continue
                
        return True
    except Exception as e:
        logger.error(f"Error in match_condition: {e}")
        return False


# ------------------------
# 메인 함수
# ------------------------
def extract_scenario_windows_optimized(df: pd.DataFrame, scenario_config: ScenarioConfig) -> List[pd.DataFrame]:
    """최적화된 시나리오 윈도우 추출 (벡터화 연산 사용)"""
    if not validate_dataframe(df):
        return []
    
    # 조건별 마스크 생성
    condition_masks = []
    for condition in scenario_config.conditions:
        if 'duration' in condition:
            continue
            
        mask = pd.Series(True, index=df.index)
        for key, val in condition.items():
            if key == 'duration':
                continue
                
            try:
                operator, values = parse_condition_value(val)
                col_data = df[key] if key in df.columns else pd.Series(0, index=df.index)
                
                if operator == ">":
                    mask &= (col_data > values[0])
                elif operator == "<":
                    mask &= (col_data < values[0])
                elif operator == ">=":
                    mask &= (col_data >= values[0])
                elif operator == "<=":
                    mask &= (col_data <= values[0])
                elif operator == "between":
                    mask &= (col_data >= values[0]) & (col_data <= values[1])
                    
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping condition {key}={val}: {e}")
                continue
        
        condition_masks.append(mask)
    
    # 모든 조건을 만족하는 인덱스 찾기
    if condition_masks:
        combined_mask = condition_masks[0]
        for mask in condition_masks[1:]:
            combined_mask &= mask
        valid_indices = df.index[combined_mask].tolist()
    else:
        valid_indices = df.index.tolist()
    
    # 연속된 구간 찾기
    segments = []
    if valid_indices:
        segments = _extract_continuous_segments(
            df, valid_indices, scenario_config.min_duration, scenario_config.max_duration
        )
    
    logger.info(f"Extracted {len(segments)} segments for scenario '{scenario_config.name}'")
    return segments

def _extract_continuous_segments(df: pd.DataFrame, valid_indices: List, min_dur: int, max_dur: int) -> List[pd.DataFrame]:
    """연속된 유효 인덱스에서 세그먼트 추출"""
    segments = []
    valid_positions = [df.index.get_loc(idx) for idx in valid_indices]
    
    i = 0
    while i < len(valid_positions) - min_dur + 1:
        start_pos = valid_positions[i]
        
        # 연속된 구간의 최대 길이 찾기
        max_continuous = min_dur
        for j in range(i + min_dur - 1, min(i + max_dur, len(valid_positions))):
            if valid_positions[j] - valid_positions[i] + 1 <= max_dur:
                max_continuous = valid_positions[j] - valid_positions[i] + 1
            else:
                break
        
        # 가능한 윈도우 크기들에 대해 세그먼트 생성
        for window_size in range(min_dur, min(max_continuous + 1, max_dur + 1)):
            end_pos = start_pos + window_size
            if end_pos <= len(df):
                segment = df.iloc[start_pos:end_pos].copy()
                if not segment.empty:
                    segments.append(segment)
        
        i += 1
    
    return segments

def extract_scenario_windows(df: pd.DataFrame, scenario: Dict[str, Any]) -> List[pd.DataFrame]:
    """기존 API 호환성을 위한 래퍼 함수"""
    try:
        scenario_name = scenario.get("name", "unknown_scenario")
        conds = scenario.get("conditions", [])
        
        # conditions가 dict이면 리스트로 변환
        if isinstance(conds, dict):
            conds = [{k: v} for k, v in conds.items()]
        
        # duration 추출
        duration_str = "10~30"
        for cond in conds:
            if isinstance(cond, dict) and "duration" in cond:
                duration_str = cond["duration"]
                break
        
        min_dur, max_dur = parse_duration(duration_str)
        
        # ScenarioConfig 객체 생성
        config = ScenarioConfig(
            name=scenario_name,
            min_duration=min_dur,
            max_duration=max_dur,
            conditions=conds
        )
        
        return extract_scenario_windows_optimized(df, config)
        
    except Exception as e:
        logger.error(f"Error extracting scenario windows: {e}")
        return []
