# sentiment/sentiment_score.py

import os
import logging
import pandas as pd
from datetime import datetime
from sentiment.sentiment_score_refiner import get_sentiment_score as analyze_sentiment
from core.path_config import get_data_path

logger = logging.getLogger(__name__)

# CSV 파일 경로 설정
SENTIMENT_SCORE_PATH = str(get_data_path("sentiment")) if get_data_path else "data/sentiment/sentiment_scores.csv"

def load_cached_sentiment_scores() -> pd.DataFrame:
    """
    감정 점수 CSV를 로딩하여 DataFrame 반환
    """
    if not os.path.exists(SENTIMENT_SCORE_PATH):
        logger.warning(f"Sentiment score file not found: {SENTIMENT_SCORE_PATH}")
        return pd.DataFrame(columns=["date", "sentiment_score"])
    
    df = pd.read_csv(SENTIMENT_SCORE_PATH)
    if "date" not in df.columns or "sentiment_score" not in df.columns:
        logger.error(f"Required columns missing: 'date', 'sentiment_score'")
        return pd.DataFrame(columns=["date", "sentiment_score"])
    
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_sentiment_score_by_date(date: str) -> float:
    """
    지정된 날짜의 감정 점수(float)를 반환
    :param date: 'YYYY-MM-DD' 문자열
    :return: 감정 점수(float), 없으면 0.0
    """
    try:
        df = load_cached_sentiment_scores()
        target_date = pd.to_datetime(date).date()
        row = df[df["date"].dt.date == target_date]

        if row.empty:
            logger.debug(f"No sentiment score found for date: {target_date}")
            return 0.0

        return float(row.iloc[0]["sentiment_score"])

    except Exception as e:
        logger.error(f"Error retrieving sentiment score: {e}")
        return 0.0


def get_sentiment_score(text: str) -> float:
    """
    뉴스 텍스트의 감정 점수를 실시간으로 분석하여 반환
    sentiment_router.py의 live 모드에서 호출되는 함수
    
    :param text: 분석할 뉴스 텍스트
    :return: 0~1 사이의 감정 점수
    """
    return analyze_sentiment(text)


def get_sentiment_score_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    지정된 날짜 범위의 감정 점수 시계열 반환
    :param start_date: 시작일 (YYYY-MM-DD)
    :param end_date: 종료일 (YYYY-MM-DD)
    :return: 날짜별 점수가 포함된 DataFrame
    """
    try:
        df = load_cached_sentiment_scores()
        df_filtered = df[
            (df["date"] >= pd.to_datetime(start_date)) &
            (df["date"] <= pd.to_datetime(end_date))
        ].copy()
        return df_filtered.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error retrieving sentiment score range: {e}")
        return pd.DataFrame(columns=["date", "sentiment_score"])


def get_latest_sentiment_score() -> float:
    """
    가장 최근 날짜의 감정 점수를 반환
    :return: 최근 감정 점수 (float), 없으면 0.0
    """
    try:
        df = load_cached_sentiment_scores()
        if df.empty:
            return 0.0
        latest_row = df.sort_values("date", ascending=False).iloc[0]
        return float(latest_row["sentiment_score"])
    except Exception as e:
        logger.error(f"Error retrieving latest sentiment score: {e}")
        return 0.0


def save_sentiment_score(date: str, score: float, text: str = None) -> bool:
    """
    감정 점수를 CSV에 저장
    
    :param date: 날짜 (YYYY-MM-DD)
    :param score: 감정 점수 (0~1)
    :param text: 원본 텍스트 (선택사항)
    :return: 저장 성공 여부
    """
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(SENTIMENT_SCORE_PATH), exist_ok=True)
        
        # 새 데이터 생성
        new_row = pd.DataFrame({
            "date": [pd.to_datetime(date)],
            "sentiment_score": [score],
            "text": [text if text else ""]
        })
        
        # 기존 파일이 있는지 확인
        if os.path.exists(SENTIMENT_SCORE_PATH):
            try:
                # 기존 데이터 로드
                df = pd.read_csv(SENTIMENT_SCORE_PATH)
                
                # 컬럼 검증 및 추가
                if "date" not in df.columns:
                    df["date"] = pd.NaT
                if "sentiment_score" not in df.columns:
                    df["sentiment_score"] = 0.0
                if "text" not in df.columns:
                    df["text"] = ""
                
                # 날짜 컬럼을 datetime으로 변환
                df["date"] = pd.to_datetime(df["date"])
                
                # 중복 날짜 체크 및 업데이트
                target_date = pd.to_datetime(date).date()
                mask = df["date"].dt.date == target_date
                
                if mask.any():
                    # 기존 데이터 업데이트
                    df.loc[mask, "sentiment_score"] = score
                    if text:
                        df.loc[mask, "text"] = text
                else:
                    # 새 데이터 추가
                    df = pd.concat([df, new_row], ignore_index=True)
                    
            except Exception as e:
                logger.warning(f"Error reading existing file, creating new: {e}")
                df = new_row
        else:
            # 새 파일 생성
            df = new_row
        
        # 날짜순 정렬 후 저장
        df = df.sort_values("date", na_position='last')
        df.to_csv(SENTIMENT_SCORE_PATH, index=False)
        
        logger.info(f"Sentiment score saved: {date} = {score:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save sentiment score: {e}", exc_info=True)
        return False