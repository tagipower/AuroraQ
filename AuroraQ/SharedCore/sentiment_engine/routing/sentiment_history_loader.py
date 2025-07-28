# SharedCore/sentiment_engine/routing/sentiment_history_loader.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path
import csv

logger = logging.getLogger(__name__)

class SentimentHistoryLoader:
    """
    CSV 파일에서 백테스트용 감정 점수 히스토리를 로드하는 클래스
    """
    
    def __init__(self, csv_path: str, time_column: str = "datetime", score_column: str = "sentiment_score"):
        """
        Args:
            csv_path: CSV 파일 경로
            time_column: 시간 컬럼명
            score_column: 감정점수 컬럼명
        """
        self.csv_path = Path(csv_path)
        self.time_column = time_column
        self.score_column = score_column
        
        self.data = None
        self.loaded = False
        
        # 기본 감정 점수 통계
        self.default_score = 0.5
        self.score_std = 0.1
        
        try:
            self._load_data()
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] CSV 로딩 실패: {e}")
            logger.warning("[SentimentHistoryLoader] 시뮬레이션 모드로 동작합니다")
    
    def _load_data(self):
        """CSV 데이터 로드"""
        if not self.csv_path.exists():
            logger.warning(f"[SentimentHistoryLoader] CSV 파일이 없습니다: {self.csv_path}")
            self._create_sample_data()
            return
        
        try:
            # CSV 읽기
            self.data = pd.read_csv(self.csv_path)
            
            # 필수 컬럼 확인
            if self.time_column not in self.data.columns:
                logger.error(f"[SentimentHistoryLoader] 시간 컬럼이 없습니다: {self.time_column}")
                self._create_sample_data()
                return
            
            if self.score_column not in self.data.columns:
                logger.error(f"[SentimentHistoryLoader] 점수 컬럼이 없습니다: {self.score_column}")
                self._create_sample_data()
                return
            
            # 시간 컬럼을 datetime으로 변환
            self.data[self.time_column] = pd.to_datetime(self.data[self.time_column])
            
            # 시간순 정렬
            self.data = self.data.sort_values(self.time_column).reset_index(drop=True)
            
            # 통계 계산
            scores = self.data[self.score_column].astype(float)
            self.default_score = scores.mean()
            self.score_std = scores.std()
            
            self.loaded = True
            logger.info(f"[SentimentHistoryLoader] CSV 로드 완료: {len(self.data)} 레코드")
            logger.info(f"[SentimentHistoryLoader] 시간 범위: {self.data[self.time_column].min()} ~ {self.data[self.time_column].max()}")
            logger.info(f"[SentimentHistoryLoader] 평균 점수: {self.default_score:.4f} (±{self.score_std:.4f})")
            
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] CSV 처리 실패: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """샘플 데이터 생성 (CSV가 없거나 로딩 실패시)"""
        logger.info("[SentimentHistoryLoader] 샘플 데이터를 생성합니다")
        
        # 지난 30일간의 시간별 데이터 생성
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        time_range = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # 랜덤 워크 기반 감정 점수 생성
        np.random.seed(42)  # 재현 가능한 결과
        
        scores = []
        current_score = 0.5
        
        for _ in time_range:
            # 랜덤 워크 + 평균 회귀
            change = np.random.normal(0, 0.02)  # 작은 변화
            current_score += change
            
            # 평균 회귀 (0.5로 수렴)
            current_score = current_score * 0.98 + 0.5 * 0.02
            
            # 범위 제한
            current_score = np.clip(current_score, 0.0, 1.0)
            scores.append(current_score)
        
        # DataFrame 생성
        self.data = pd.DataFrame({
            self.time_column: time_range,
            self.score_column: scores,
            'source': 'simulated',
            'confidence': np.random.uniform(0.6, 0.9, len(time_range))
        })
        
        # 통계 설정
        self.default_score = np.mean(scores)
        self.score_std = np.std(scores)
        
        self.loaded = True
        logger.info(f"[SentimentHistoryLoader] 샘플 데이터 생성 완료: {len(self.data)} 레코드")
    
    def get_score_at(self, timestamp: datetime, interpolation: str = "nearest") -> float:
        """
        특정 시간의 감정 점수 조회
        
        Args:
            timestamp: 조회할 시간
            interpolation: 보간 방법 ("nearest", "linear", "forward", "backward")
            
        Returns:
            감정 점수 (0.0 ~ 1.0)
        """
        if not self.loaded or self.data is None or len(self.data) == 0:
            logger.warning("[SentimentHistoryLoader] 데이터가 로드되지 않았습니다")
            return self.default_score
        
        try:
            # 시간 범위 확인
            min_time = self.data[self.time_column].min()
            max_time = self.data[self.time_column].max()
            
            if timestamp < min_time or timestamp > max_time:
                logger.debug(f"[SentimentHistoryLoader] 시간 범위 초과: {timestamp}")
                return self._extrapolate_score(timestamp, min_time, max_time)
            
            # 정확히 일치하는 시간 찾기
            exact_match = self.data[self.data[self.time_column] == timestamp]
            if not exact_match.empty:
                return float(exact_match[self.score_column].iloc[0])
            
            # 보간 처리
            if interpolation == "nearest":
                return self._get_nearest_score(timestamp)
            elif interpolation == "linear":
                return self._get_linear_score(timestamp)
            elif interpolation == "forward":
                return self._get_forward_score(timestamp)
            elif interpolation == "backward":
                return self._get_backward_score(timestamp)
            else:
                logger.warning(f"[SentimentHistoryLoader] 알 수 없는 보간 방법: {interpolation}")
                return self._get_nearest_score(timestamp)
        
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] 점수 조회 실패: {e}")
            return self.default_score
    
    def _get_nearest_score(self, timestamp: datetime) -> float:
        """가장 가까운 시간의 점수 반환"""
        time_diffs = (self.data[self.time_column] - timestamp).abs()
        nearest_idx = time_diffs.argmin()
        return float(self.data[self.score_column].iloc[nearest_idx])
    
    def _get_linear_score(self, timestamp: datetime) -> float:
        """선형 보간으로 점수 계산"""
        # 앞뒤 데이터 포인트 찾기
        before_data = self.data[self.data[self.time_column] <= timestamp]
        after_data = self.data[self.data[self.time_column] > timestamp]
        
        if before_data.empty:
            return float(after_data[self.score_column].iloc[0])
        if after_data.empty:
            return float(before_data[self.score_column].iloc[-1])
        
        # 선형 보간
        t1, s1 = before_data[self.time_column].iloc[-1], before_data[self.score_column].iloc[-1]
        t2, s2 = after_data[self.time_column].iloc[0], after_data[self.score_column].iloc[0]
        
        # 시간 비율 계산
        time_ratio = (timestamp - t1).total_seconds() / (t2 - t1).total_seconds()
        interpolated_score = s1 + (s2 - s1) * time_ratio
        
        return float(np.clip(interpolated_score, 0.0, 1.0))
    
    def _get_forward_score(self, timestamp: datetime) -> float:
        """이후 첫 번째 값 사용"""
        after_data = self.data[self.data[self.time_column] > timestamp]
        if after_data.empty:
            return float(self.data[self.score_column].iloc[-1])
        return float(after_data[self.score_column].iloc[0])
    
    def _get_backward_score(self, timestamp: datetime) -> float:
        """이전 마지막 값 사용"""
        before_data = self.data[self.data[self.time_column] <= timestamp]
        if before_data.empty:
            return float(self.data[self.score_column].iloc[0])
        return float(before_data[self.score_column].iloc[-1])
    
    def _extrapolate_score(self, timestamp: datetime, min_time: datetime, max_time: datetime) -> float:
        """범위 외 시간에 대한 외삽"""
        if timestamp < min_time:
            # 과거: 첫 번째 값 + 약간의 노이즈
            first_score = float(self.data[self.score_column].iloc[0])
            noise = np.random.normal(0, self.score_std * 0.1)
            return np.clip(first_score + noise, 0.0, 1.0)
        else:
            # 미래: 마지막 값 + 약간의 노이즈
            last_score = float(self.data[self.score_column].iloc[-1])
            noise = np.random.normal(0, self.score_std * 0.1)
            return np.clip(last_score + noise, 0.0, 1.0)
    
    def get_score_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        시간 범위의 감정 점수 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            해당 기간의 데이터프레임
        """
        if not self.loaded or self.data is None:
            logger.warning("[SentimentHistoryLoader] 데이터가 로드되지 않았습니다")
            return pd.DataFrame()
        
        try:
            mask = (self.data[self.time_column] >= start_time) & (self.data[self.time_column] <= end_time)
            result = self.data[mask].copy()
            
            logger.debug(f"[SentimentHistoryLoader] 범위 조회: {len(result)} 레코드")
            return result
            
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] 범위 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_statistics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        감정 점수 통계 정보
        
        Args:
            start_time: 시작 시간 (None이면 전체)
            end_time: 종료 시간 (None이면 전체)
            
        Returns:
            통계 정보 딕셔너리
        """
        if not self.loaded or self.data is None:
            return {
                "count": 0,
                "mean": self.default_score,
                "std": self.score_std,
                "min": 0.0,
                "max": 1.0
            }
        
        try:
            # 시간 범위 필터링
            if start_time is not None or end_time is not None:
                data_subset = self.get_score_range(
                    start_time or self.data[self.time_column].min(),
                    end_time or self.data[self.time_column].max()
                )
            else:
                data_subset = self.data
            
            if data_subset.empty:
                return {
                    "count": 0,
                    "mean": self.default_score,
                    "std": self.score_std,
                    "min": 0.0,
                    "max": 1.0
                }
            
            scores = data_subset[self.score_column].astype(float)
            
            return {
                "count": len(scores),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "median": float(scores.median()),
                "q25": float(scores.quantile(0.25)),
                "q75": float(scores.quantile(0.75)),
                "time_range": {
                    "start": data_subset[self.time_column].min(),
                    "end": data_subset[self.time_column].max()
                }
            }
            
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] 통계 계산 실패: {e}")
            return {
                "count": 0,
                "mean": self.default_score,
                "std": self.score_std,
                "min": 0.0,
                "max": 1.0,
                "error": str(e)
            }
    
    def save_sample_data(self, output_path: Optional[str] = None):
        """
        현재 데이터를 CSV로 저장
        
        Args:
            output_path: 출력 파일 경로 (None이면 기본 경로)
        """
        if not self.loaded or self.data is None:
            logger.warning("[SentimentHistoryLoader] 저장할 데이터가 없습니다")
            return
        
        try:
            save_path = Path(output_path) if output_path else self.csv_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.data.to_csv(save_path, index=False)
            logger.info(f"[SentimentHistoryLoader] 데이터 저장 완료: {save_path}")
            
        except Exception as e:
            logger.error(f"[SentimentHistoryLoader] 데이터 저장 실패: {e}")
    
    def reload(self):
        """데이터 다시 로드"""
        self.loaded = False
        self.data = None
        self._load_data()


if __name__ == "__main__":
    # 테스트 코드
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 경로
    test_csv_path = "test_sentiment_history.csv"
    
    print("=== SentimentHistoryLoader 테스트 ===\n")
    
    # 로더 생성 (샘플 데이터 자동 생성)
    loader = SentimentHistoryLoader(test_csv_path)
    
    # 통계 정보
    print("1. 전체 통계:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 특정 시간 조회
    print("\n2. 특정 시간 점수 조회:")
    test_times = [
        datetime.now() - timedelta(days=15),
        datetime.now() - timedelta(days=10),
        datetime.now() - timedelta(days=5),
        datetime.now()
    ]
    
    for test_time in test_times:
        score = loader.get_score_at(test_time)
        print(f"   {test_time.strftime('%Y-%m-%d %H:%M')}: {score:.4f}")
    
    # 범위 조회
    print("\n3. 범위 조회:")
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now() - timedelta(days=1)
    
    range_data = loader.get_score_range(start_time, end_time)
    print(f"   기간: {start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}")
    print(f"   레코드 수: {len(range_data)}")
    
    if not range_data.empty:
        print(f"   평균 점수: {range_data['sentiment_score'].mean():.4f}")
        print(f"   최고/최저: {range_data['sentiment_score'].max():.4f} / {range_data['sentiment_score'].min():.4f}")
    
    # 보간 방법 비교
    print("\n4. 보간 방법 비교:")
    test_time = datetime.now() - timedelta(days=7, hours=6, minutes=30)  # 정확한 시간이 아닌 중간값
    
    methods = ["nearest", "linear", "forward", "backward"]
    for method in methods:
        score = loader.get_score_at(test_time, interpolation=method)
        print(f"   {method}: {score:.4f}")
    
    # 샘플 데이터 저장
    print("\n5. 샘플 데이터 저장:")
    loader.save_sample_data()
    print(f"   저장 완료: {test_csv_path}")
    
    print("\n=== 테스트 완료 ===")
    
    # 테스트 파일 정리
    import os
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"테스트 파일 삭제: {test_csv_path}")