# utils/helpers.py

import os
import pandas as pd
from datetime import datetime

def ensure_dir(path: str):
    """
    지정한 경로에 디렉토리가 존재하지 않으면 생성합니다.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(df: pd.DataFrame, path: str):
    """
    DataFrame을 CSV로 저장. 상위 디렉토리가 없으면 자동 생성.
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료: {path}")

def get_timestamp(fmt: str = "%Y%m%d_%H%M") -> str:
    """
    현재 시각을 지정한 형식의 문자열로 반환합니다.
    기본 형식은 YYYYMMDD_HHMM.
    """
    return datetime.now().strftime(fmt)

def load_latest_csv(folder: str) -> str:
    """
    지정된 폴더에서 가장 최근 CSV 파일의 경로를 반환합니다.
    """
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        return None
    latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    return os.path.join(folder, latest_file)
