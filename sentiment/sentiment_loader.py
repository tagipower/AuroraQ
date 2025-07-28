# sentiment/sentiment_loader.py

import csv
import os
from datetime import datetime, timedelta
from bisect import bisect_right
from typing import List, Tuple, Optional
import logging

# 로거 설정
logger = logging.getLogger("SentimentScoreLoader")

class SentimentScoreLoader:
    """
    CSV 파일에서 시계열 감정 점수 데이터를 효율적으로 로드하고 조회하는 클래스
    백테스트를 위한 과거 감정 점수 데이터 관리
    """
    
    def __init__(self, filepath: str, default_score: float = 0.5):
        """
        :param filepath: CSV 파일 경로
        :param default_score: 데이터가 없을 때 반환할 기본 점수
        """
        self.filepath = filepath
        self.default_score = default_score
        self.timestamps: List[datetime] = []
        self.scores: List[float] = []
        self.is_loaded = False
        self._load()

    def _load(self) -> None:
        """CSV 파일을 메모리로 로드"""
        if not os.path.exists(self.filepath):
            logger.error(f"CSV 파일이 존재하지 않습니다: {self.filepath}")
            return
            
        try:
            with open(self.filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # 헤더 확인
                header = next(reader, None)
                if not header:
                    logger.warning("CSV 파일이 비어있습니다")
                    return
                
                # 헤더 인덱스 찾기 (유연한 처리)
                timestamp_idx = None
                score_idx = None
                
                for idx, col in enumerate(header):
                    col_lower = col.lower().strip()
                    if 'timestamp' in col_lower or 'time' in col_lower or 'date' in col_lower:
                        timestamp_idx = idx
                    elif 'score' in col_lower or 'sentiment' in col_lower:
                        score_idx = idx
                
                if timestamp_idx is None or score_idx is None:
                    logger.error(f"필수 컬럼을 찾을 수 없습니다. 헤더: {header}")
                    return
                
                # 데이터 로드
                row_count = 0
                error_count = 0
                
                for row in reader:
                    row_count += 1
                    
                    if len(row) <= max(timestamp_idx, score_idx):
                        error_count += 1
                        continue
                    
                    try:
                        timestamp_str = row[timestamp_idx].strip()
                        score_str = row[score_idx].strip()
                        
                        # 다양한 날짜 형식 처리
                        timestamp = self._parse_timestamp(timestamp_str)
                        score = float(score_str)
                        
                        # 유효성 검증
                        if not (0 <= score <= 1):
                            logger.warning(f"비정상 점수 값 (행 {row_count}): {score}")
                            score = max(0, min(1, score))  # 클리핑
                        
                        self.timestamps.append(timestamp)
                        self.scores.append(score)
                        
                    except (ValueError, TypeError) as e:
                        error_count += 1
                        if error_count <= 5:  # 처음 5개 에러만 로깅
                            logger.debug(f"행 {row_count} 파싱 실패: {e}")
                        continue
                
                # 시간순 정렬 (이진 탐색을 위해 필수)
                if self.timestamps:
                    sorted_pairs = sorted(zip(self.timestamps, self.scores))
                    self.timestamps, self.scores = zip(*sorted_pairs)
                    self.timestamps = list(self.timestamps)
                    self.scores = list(self.scores)
                
                self.is_loaded = True
                logger.info(f"✅ CSV 로드 완료: {len(self.timestamps)}개 레코드 "
                          f"({error_count}개 오류, {row_count}개 중)")
                
                if self.timestamps:
                    logger.info(f"📅 데이터 범위: {self.timestamps[0]} ~ {self.timestamps[-1]}")
                
        except Exception as e:
            logger.error(f"CSV 파일 로딩 중 오류 발생: {e}")
            self.is_loaded = False

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """다양한 날짜 형식을 파싱"""
        # ISO 형식 우선 시도
        try:
            return datetime.fromisoformat(timestamp_str)
        except:
            pass
        
        # 다른 일반적인 형식들 시도
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except:
                continue
        
        # 모든 형식 실패 시 예외 발생
        raise ValueError(f"파싱할 수 없는 날짜 형식: {timestamp_str}")

    def get_score_at(self, target_time: datetime) -> float:
        """
        주어진 시간보다 작거나 같은 가장 가까운 시간의 감정 점수를 반환
        
        :param target_time: 조회할 시간
        :return: 감정 점수 (0~1)
        """
        if not self.is_loaded or not self.timestamps:
            logger.warning(f"데이터가 로드되지 않았습니다. 기본값 {self.default_score} 반환")
            return self.default_score
        
        # 이진 탐색으로 효율적인 조회
        idx = bisect_right(self.timestamps, target_time) - 1
        
        if idx >= 0:
            time_diff = (target_time - self.timestamps[idx]).total_seconds()
            
            # 너무 오래된 데이터인 경우 경고
            if time_diff > 86400:  # 24시간 이상 차이
                logger.debug(f"⚠️ 조회 시간과 가장 가까운 데이터가 "
                           f"{time_diff/3600:.1f}시간 차이납니다")
            
            return self.scores[idx]
        
        # target_time이 첫 번째 타임스탬프보다 이전인 경우
        logger.debug(f"조회 시간({target_time})이 데이터 시작 시간"
                    f"({self.timestamps[0]})보다 이전입니다")
        return self.default_score

    def get_score_range(self, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """
        시간 범위 내의 모든 감정 점수 반환
        
        :param start_time: 시작 시간
        :param end_time: 종료 시간
        :return: [(timestamp, score), ...] 리스트
        """
        if not self.is_loaded or not self.timestamps:
            return []
        
        start_idx = bisect_right(self.timestamps, start_time) - 1
        end_idx = bisect_right(self.timestamps, end_time)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(self.timestamps), end_idx)
        
        return list(zip(
            self.timestamps[start_idx:end_idx],
            self.scores[start_idx:end_idx]
        ))

    def get_average_score(self, start_time: datetime, end_time: datetime) -> Optional[float]:
        """
        시간 범위 내의 평균 감정 점수 계산
        
        :param start_time: 시작 시간
        :param end_time: 종료 시간
        :return: 평균 점수 또는 None
        """
        range_data = self.get_score_range(start_time, end_time)
        
        if not range_data:
            return None
        
        scores = [score for _, score in range_data]
        return sum(scores) / len(scores)

    def get_statistics(self) -> dict:
        """로드된 데이터의 통계 정보 반환"""
        if not self.is_loaded or not self.scores:
            return {
                "loaded": False,
                "count": 0
            }
        
        return {
            "loaded": True,
            "count": len(self.scores),
            "start_time": self.timestamps[0],
            "end_time": self.timestamps[-1],
            "min_score": min(self.scores),
            "max_score": max(self.scores),
            "avg_score": sum(self.scores) / len(self.scores),
            "file_path": self.filepath
        }

    def reload(self) -> None:
        """CSV 파일을 다시 로드"""
        logger.info("CSV 파일 재로드 중...")
        self.timestamps.clear()
        self.scores.clear()
        self.is_loaded = False
        self._load()


# 테스트용 코드
if __name__ == '__main__':
    # 테스트 CSV 파일 생성
    test_file = "test_sentiment_log.csv"
    
    # 테스트 데이터 생성
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'sentiment_score', 'text'])
        
        base_time = datetime(2025, 6, 19, 12, 0)
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 30)
            score = 0.5 + 0.3 * (i % 3 - 1)  # 0.2, 0.5, 0.8 반복
            writer.writerow([timestamp.isoformat(), score, f"Test news {i}"])
    
    # 로더 테스트
    print("=== SentimentScoreLoader 테스트 ===")
    loader = SentimentScoreLoader(test_file)
    
    # 통계 정보
    stats = loader.get_statistics()
    print(f"\n📊 통계 정보:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 특정 시간 조회
    print(f"\n🔍 특정 시간 조회:")
    test_times = [
        datetime(2025, 6, 19, 11, 30),  # 데이터 이전
        datetime(2025, 6, 19, 12, 15),  # 데이터 중간
        datetime(2025, 6, 19, 17, 0),   # 데이터 이후
    ]
    
    for test_time in test_times:
        score = loader.get_score_at(test_time)
        print(f"  [{test_time}] → {score:.4f}")
    
    # 범위 조회
    print(f"\n📈 범위 조회 (12:30 ~ 14:00):")
    range_data = loader.get_score_range(
        datetime(2025, 6, 19, 12, 30),
        datetime(2025, 6, 19, 14, 0)
    )
    for timestamp, score in range_data:
        print(f"  [{timestamp}] = {score:.4f}")
    
    # 평균 계산
    avg_score = loader.get_average_score(
        datetime(2025, 6, 19, 12, 0),
        datetime(2025, 6, 19, 14, 0)
    )
    print(f"\n📊 평균 점수 (12:00 ~ 14:00): {avg_score:.4f}")
    
    # 테스트 파일 삭제
    os.remove(test_file)
    print(f"\n🗑️ 테스트 파일 삭제 완료")