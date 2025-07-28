# SharedCore/sentiment_engine/routing/sentiment_router.py

from datetime import datetime
from typing import Optional, Dict, Any
import os
import sys
import asyncio
import logging

# Import components
from ..analyzers.finbert_analyzer import get_finbert_analyzer
from ..fusion.sentiment_fusion_manager import get_fusion_manager
from ...utils.logger import get_logger

logger = get_logger("SentimentRouter")

class SentimentRouter:
    """
    감정 점수 라우터: 실시간/백테스트 모드에 따라 적절한 소스에서 감정 점수를 가져옴
    """
    
    def __init__(self, mode: str = "live", csv_path: str = None):
        """
        Args:
            mode: "live" | "backtest"
            csv_path: 백테스트에서 참조할 CSV 경로 (기본값: 환경변수 또는 고정경로)
        """
        if mode not in ["live", "backtest"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'live' or 'backtest'")
        
        self.mode = mode
        self.csv_loader = None
        self._score_cache = {}  # 간단한 캐싱 추가
        self._initialized = False
        
        # Fusion manager와 analyzer 인스턴스
        self.fusion_manager = None
        self.analyzer = None
        
        if self.mode == "backtest":
            path = csv_path or os.getenv("SENTIMENT_CSV_PATH", "data/sentiment/news_sentiment_log.csv")
            try:
                # CSV 로더 초기화는 필요시에만
                self.csv_path = path
                logger.info(f"[Router] 백테스트 모드 설정: {path}")
            except Exception as e:
                logger.error(f"[Router] 백테스트 모드 설정 실패: {e}")
                logger.warning("[Router] 백테스트 모드이지만 CSV 설정 실패. 기본값 반환 예정")
    
    async def initialize(self):
        """비동기 초기화"""
        if self._initialized:
            return
        
        try:
            # Fusion manager와 analyzer 초기화
            self.fusion_manager = await get_fusion_manager()
            self.analyzer = await get_finbert_analyzer()
            
            # 백테스트 모드인 경우 CSV 로더 초기화
            if self.mode == "backtest" and hasattr(self, 'csv_path'):
                try:
                    from .sentiment_history_loader import SentimentHistoryLoader
                    self.csv_loader = SentimentHistoryLoader(self.csv_path)
                    logger.info(f"[Router] 백테스트용 감정 점수 로더 로드 완료: {self.csv_path}")
                except Exception as e:
                    logger.error(f"[Router] 감정 점수 로더 로딩 실패: {e}")
                    logger.warning("[Router] 백테스트 모드이지만 CSV 로드 실패. 실시간 분석 사용")
            
            self._initialized = True
            logger.info(f"[Router] 초기화 완료 - 모드: {self.mode}")
            
        except Exception as e:
            logger.error(f"[Router] 초기화 실패: {e}")
            raise
    
    async def get_score(self, news_text: str = None, timestamp: datetime = None) -> float:
        """
        감정 점수를 반환합니다.
        - live 모드: 뉴스 텍스트를 받아 실시간 분석
        - backtest 모드: 주어진 timestamp에 해당하는 저장 점수 반환
        
        Args:
            news_text: 분석할 뉴스 텍스트 (live 모드에서 필수)
            timestamp: 조회할 시간 (backtest 모드에서 필수)
            
        Returns:
            0~1 사이의 감정 점수
        """
        try:
            # 초기화 확인
            if not self._initialized:
                await self.initialize()
            
            if self.mode == "live":
                if not news_text:
                    logger.warning("[Router] Live 모드에서 뉴스 텍스트가 없습니다. 중립값 반환")
                    return 0.5
                
                # 캐시 확인 (동일 텍스트 반복 분석 방지)
                cache_key = hash(news_text[:100])  # 텍스트 앞부분만 해시
                if cache_key in self._score_cache:
                    logger.debug(f"[Router] 캐시에서 점수 반환: {self._score_cache[cache_key]}")
                    return self._score_cache[cache_key]
                
                # 실시간 분석 - FinBERT + Fusion 사용
                score = await self._analyze_live(news_text)
                
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
                
                if not self.csv_loader:
                    logger.error("[Router] 백테스트 로더가 초기화되지 않았습니다. 중립값 반환")
                    return 0.5
                
                score = await self._get_backtest_score(timestamp)
                logger.debug(f"[Router] Backtest 점수 조회 [{timestamp}]: {score:.4f}")
                return score
        
        except Exception as e:
            logger.error(f"[Router] 감정 점수 추출 실패: {e}")
            return 0.5
    
    async def _analyze_live(self, news_text: str) -> float:
        """실시간 감정 분석"""
        try:
            # 직접 FinBERT 분석 사용
            if self.analyzer:
                raw_score = await self.analyzer.analyze(news_text)
                
                # Fusion manager를 통한 점수 융합 (단일 소스이지만 보정 효과)
                if self.fusion_manager:
                    fused_score = await self.fusion_manager.fuse({
                        "news": raw_score
                    })
                    return fused_score
                else:
                    return raw_score
            else:
                logger.warning("[Router] Analyzer not available, using fallback")
                return 0.5
                
        except Exception as e:
            logger.error(f"[Router] 실시간 분석 실패: {e}")
            return 0.5
    
    async def _get_backtest_score(self, timestamp: datetime) -> float:
        """백테스트 점수 조회"""
        try:
            if self.csv_loader:
                return self.csv_loader.get_score_at(timestamp)
            else:
                logger.warning("[Router] CSV 로더가 없어 중립값 반환")
                return 0.5
        except Exception as e:
            logger.error(f"[Router] 백테스트 점수 조회 실패: {e}")
            return 0.5
    
    async def get_score_with_metadata(self, news_text: str = None, timestamp: datetime = None) -> Dict[str, Any]:
        """
        감정 점수와 함께 메타데이터를 반환
        
        Returns:
            {"score": float, "mode": str, "timestamp": datetime, "cached": bool}
        """
        score = await self.get_score(news_text, timestamp)
        
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
    
    async def switch_mode(self, new_mode: str, csv_path: str = None):
        """
        모드 전환 (실행 중 모드 변경)
        
        Args:
            new_mode: "live" | "backtest"
            csv_path: backtest 모드일 때 CSV 경로
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
                from .sentiment_history_loader import SentimentHistoryLoader
                self.csv_loader = SentimentHistoryLoader(path)
                self.csv_path = path
                logger.info(f"[Router] 모드 전환 완료: {new_mode}, CSV: {path}")
            except Exception as e:
                logger.error(f"[Router] 백테스트 모드 전환 실패: {e}")
        else:
            self.csv_loader = None
            if hasattr(self, 'csv_path'):
                delattr(self, 'csv_path')
            logger.info(f"[Router] 모드 전환 완료: {new_mode}")
    
    async def analyze_articles_batch(self, articles: list, symbol: str = "BTCUSDT") -> list:
        """
        기사 리스트에 대한 배치 감정 분석
        
        Args:
            articles: 기사 리스트
            symbol: 심볼
            
        Returns:
            감정 분석 결과 리스트
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.mode == "live" and self.fusion_manager:
                # Fusion manager를 통한 배치 분석
                return await self.fusion_manager.get_fused_scores(articles, symbol)
            else:
                # 기본 배치 분석
                results = []
                for article in articles:
                    try:
                        if isinstance(article, dict):
                            text = f"{article.get('title', '')} {article.get('snippet', '')}".strip()
                        else:
                            text = str(article)
                        
                        score = await self.get_score(news_text=text)
                        
                        result = {
                            "datetime": article.get("datetime", datetime.utcnow()) if isinstance(article, dict) else datetime.utcnow(),
                            "symbol": symbol,
                            "sentiment_score": score,
                            "label": "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral",
                            "confidence": 0.8,  # 기본값
                            "keywords": [],
                            "scenario_tag": "일반 분석",
                            "volatility": 0.0,
                            "trend": "neutral"
                        }
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"[Router] 기사 분석 실패: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"[Router] 배치 분석 실패: {e}")
            return []
    
    async def close(self):
        """리소스 정리"""
        if self.fusion_manager:
            await self.fusion_manager.close()
        
        if self.analyzer:
            await self.analyzer.close()
        
        logger.info("[Router] 리소스 정리 완료")


# 싱글톤 패턴으로 전역 라우터 제공
_router_instance: Optional[SentimentRouter] = None

async def get_router(mode: str = "live", csv_path: str = None) -> SentimentRouter:
    """
    전역 라우터 인스턴스 반환 (싱글톤)
    """
    global _router_instance
    
    if _router_instance is None:
        _router_instance = SentimentRouter(mode, csv_path)
        await _router_instance.initialize()
    elif _router_instance.get_mode() != mode:
        await _router_instance.switch_mode(mode, csv_path)
    
    return _router_instance


# 호환성을 위한 래퍼 함수들
async def get_sentiment_score(news_text: str) -> float:
    """기존 API와의 호환성을 위한 래퍼"""
    router = await get_router("live")
    return await router.get_score(news_text=news_text)


if __name__ == "__main__":
    import time
    
    async def test_router():
        """라우터 테스트"""
        print("=== 실시간 모드 테스트 ===")
        router_live = SentimentRouter(mode="live")
        await router_live.initialize()
        
        test_texts = [
            "Bitcoin ETF approval likely by SEC next week",
            "Major cryptocurrency exchange hacked, $100M stolen", 
            "Market remains stable despite global uncertainty"
        ]
        
        for text in test_texts:
            score = await router_live.get_score(news_text=text)
            print(f"텍스트: {text[:50]}...")
            print(f"점수: {score:.4f}\n")
        
        # 메타데이터 포함 조회
        print("\n=== 메타데이터 테스트 ===")
        metadata = await router_live.get_score_with_metadata(
            news_text="Positive market sentiment continues"
        )
        print(f"메타데이터: {metadata}")
        
        # 배치 분석 테스트
        print("\n=== 배치 분석 테스트 ===")
        articles = [
            {"title": "Bitcoin Surges", "snippet": "Strong bullish momentum"},
            {"title": "Market Crash", "snippet": "Investors flee to safety"}
        ]
        
        batch_results = await router_live.analyze_articles_batch(articles)
        for i, result in enumerate(batch_results):
            print(f"Article {i+1}: {result['sentiment_score']:.4f} - {result['trend']}")
        
        await router_live.close()
        print("\n=== 테스트 완료 ===")
    
    # 테스트 실행
    asyncio.run(test_router())