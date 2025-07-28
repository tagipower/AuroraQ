# sentiment/sentiment_router.py

from datetime import datetime
from typing import Optional
import os
import sys

# 프로젝트 루트 경로 추가 (import 문제 해결)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.sentiment_score_refiner import get_sentiment_score
from sentiment.sentiment_loader import SentimentScoreLoader
from utils.logger import get_logger

logger = get_logger("SentimentScoreRouter")

class SentimentScoreRouter:
    """
    감정 점수 라우터: 실시간/백테스트 모드에 따라 적절한 소스에서 감정 점수를 가져옴
    """
    
    def __init__(self, mode: str = "live", csv_path: str = None):
        """
        mode: "live" | "backtest"
        csv_path: 백테스트에서 참조할 CSV 경로 (기본값: 환경변수 또는 고정경로)
        """
        if mode not in ["live", "backtest"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'live' or 'backtest'")
            
        self.mode = mode
        self.loader = None
        self._score_cache = {}  # 간단한 캐싱 추가

        if self.mode == "backtest":
            path = csv_path or os.getenv("SENTIMENT_CSV_PATH", "data/sentiment/news_sentiment_log.csv")
            try:
                self.loader = SentimentScoreLoader(path)
                logger.info(f"[Router] 📄 백테스트용 감정 점수 로더 로드 완료: {path}")
            except Exception as e:
                logger.error(f"[Router] 감정 점수 로더 로딩 실패: {e}")
                logger.warning("[Router] 백테스트 모드이지만 CSV 로드 실패. 기본값 반환 예정")

    def get_score(self, news_text: str = None, timestamp: datetime = None) -> float:
        """
        감정 점수를 반환합니다.
        - live 모드: 뉴스 텍스트를 받아 실시간 분석
        - backtest 모드: 주어진 timestamp에 해당하는 저장 점수 반환
        
        :param news_text: 분석할 뉴스 텍스트 (live 모드에서 필수)
        :param timestamp: 조회할 시간 (backtest 모드에서 필수)
        :return: 0~1 사이의 감정 점수
        """
        try:
            if self.mode == "live":
                if not news_text:
                    logger.warning("[Router] Live 모드에서 뉴스 텍스트가 없습니다. 중립값 반환")
                    return 0.5
                
                # 캐시 확인 (동일 텍스트 반복 분석 방지)
                cache_key = hash(news_text[:100])  # 텍스트 앞부분만 해시
                if cache_key in self._score_cache:
                    logger.debug(f"[Router] 캐시에서 점수 반환: {self._score_cache[cache_key]}")
                    return self._score_cache[cache_key]
                
                # 실시간 분석
                score = get_sentiment_score(news_text)
                
                # 캐시 저장 (최대 100개 유지)
                if len(self._score_cache) > 100:
                    self._score_cache.pop(next(iter(self._score_cache)))
                self._score_cache[cache_key] = score
                
                logger.info(f"[Router] Live 분석 완료 - 점수: {score:.4f}")
                return score

            elif self.mode == "backtest":
                if not timestamp:
                    logger.warning("[Router] Backtest 모드에서 타임스탬프가 없습니다. 중립값 반환")
                    return 0.5
                
                if not self.loader:
                    logger.error("[Router] 백테스트 로더가 초기화되지 않았습니다. 중립값 반환")
                    return 0.5
                
                score = self.loader.get_score_at(timestamp)
                logger.debug(f"[Router] Backtest 점수 조회 [{timestamp}]: {score:.4f}")
                return score

        except Exception as e:
            logger.error(f"[Router] 감정 점수 추출 실패: {e}")
            return 0.5

    def get_score_with_metadata(self, news_text: str = None, timestamp: datetime = None) -> dict:
        """
        감정 점수와 함께 메타데이터를 반환
        
        :return: {"score": float, "mode": str, "timestamp": datetime, "cached": bool}
        """
        score = self.get_score(news_text, timestamp)
        
        metadata = {
            "score": score,
            "mode": self.mode,
            "timestamp": timestamp or datetime.now(),
            "cached": False
        }
        
        if self.mode == "live" and news_text:
            cache_key = hash(news_text[:100])
            metadata["cached"] = cache_key in self._score_cache
            
        return metadata

    def clear_cache(self):
        """캐시 초기화"""
        self._score_cache.clear()
        logger.info("[Router] 캐시가 초기화되었습니다")

    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.mode

    def switch_mode(self, new_mode: str, csv_path: str = None):
        """
        모드 전환 (실행 중 모드 변경)
        
        :param new_mode: "live" | "backtest"
        :param csv_path: backtest 모드일 때 CSV 경로
        """
        if new_mode not in ["live", "backtest"]:
            raise ValueError(f"Invalid mode: {new_mode}")
        
        if new_mode == self.mode:
            logger.info(f"[Router] 이미 {new_mode} 모드입니다")
            return
        
        self.mode = new_mode
        self.clear_cache()
        
        if new_mode == "backtest":
            path = csv_path or os.getenv("SENTIMENT_CSV_PATH", "data/sentiment/news_sentiment_log.csv")
            try:
                self.loader = SentimentScoreLoader(path)
                logger.info(f"[Router] 모드 전환 완료: {new_mode}, CSV: {path}")
            except Exception as e:
                logger.error(f"[Router] 백테스트 모드 전환 실패: {e}")
        else:
            self.loader = None
            logger.info(f"[Router] 모드 전환 완료: {new_mode}")


# 싱글톤 패턴으로 전역 라우터 제공
_router_instance: Optional[SentimentScoreRouter] = None

def get_router(mode: str = "live", csv_path: str = None) -> SentimentScoreRouter:
    """
    전역 라우터 인스턴스 반환 (싱글톤)
    """
    global _router_instance
    
    if _router_instance is None:
        _router_instance = SentimentScoreRouter(mode, csv_path)
    elif _router_instance.get_mode() != mode:
        _router_instance.switch_mode(mode, csv_path)
    
    return _router_instance


# ✅ 사용 예시
if __name__ == "__main__":
    import time
    
    # 1. 실시간 모드 테스트
    print("=== 실시간 모드 테스트 ===")
    router_live = SentimentScoreRouter(mode="live")
    
    test_texts = [
        "Bitcoin ETF approval likely by SEC next week",
        "Major cryptocurrency exchange hacked, $100M stolen",
        "Market remains stable despite global uncertainty"
    ]
    
    for text in test_texts:
        score = router_live.get_score(news_text=text)
        print(f"텍스트: {text[:50]}...")
        print(f"점수: {score:.4f}\n")
    
    # 2. 백테스트 모드 테스트
    print("\n=== 백테스트 모드 테스트 ===")
    router_back = SentimentScoreRouter(mode="backtest")
    
    test_times = [
        datetime(2025, 6, 19, 14, 30),
        datetime(2025, 6, 19, 15, 00),
        datetime(2025, 6, 19, 15, 30)
    ]
    
    for test_time in test_times:
        score = router_back.get_score(timestamp=test_time)
        print(f"시간: {test_time}")
        print(f"점수: {score:.4f}\n")
    
    # 3. 메타데이터 포함 조회
    print("\n=== 메타데이터 테스트 ===")
    metadata = router_live.get_score_with_metadata(
        news_text="Positive market sentiment continues"
    )
    print(f"메타데이터: {metadata}")
    
    # 4. 모드 전환 테스트
    print("\n=== 모드 전환 테스트 ===")
    router = get_router("live")
    print(f"현재 모드: {router.get_mode()}")
    
    router = get_router("backtest")
    print(f"전환 후 모드: {router.get_mode()}")