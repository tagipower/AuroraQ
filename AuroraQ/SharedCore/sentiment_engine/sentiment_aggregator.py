"""
감정 점수 통합 관리자 v3.0 - FinBERT + Fusion + Router 연동
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os

from .batch_processor import BatchSentimentProcessor
from .news_collectors.news_collector import NewsCollector, create_news_collector
from .routing.sentiment_router import get_router, SentimentRouter  
from .fusion.sentiment_fusion_manager import get_fusion_manager
from .analyzers.finbert_analyzer import get_finbert_analyzer
from ..utils.logger import get_logger

logger = get_logger("SentimentAggregator")


class SentimentAggregator:
    """
    다중 소스 감정 점수 통합 및 관리 v3.0
    - FinBERT 기반 고급 감정분석
    - Feedly 실시간 뉴스 수집
    - 센티멘트 융합 관리자 연동
    - Live/Backtest 모드 라우팅
    - 시간별 가중치 적용
    - 자산별 감정 추적
    """
    
    def __init__(self, cache_manager=None, mode: str = "live", csv_path: str = None):
        self.cache_manager = cache_manager
        self.batch_processor = BatchSentimentProcessor()
        
        # 모드 설정
        self.mode = mode
        self.csv_path = csv_path
        
        # Feedly 뉴스 수집기 초기화
        feedly_token = os.getenv('FEEDLY_ACCESS_TOKEN')
        self.feedly_collector = create_feedly_collector(feedly_token)
        self.feedly_connected = False
        
        # 고급 감정 분석 컴포넌트 (비동기 초기화)
        self.sentiment_router = None
        self.fusion_manager = None
        self.finbert_analyzer = None
        self._initialized = False
        
        # 소스별 가중치 (뉴스 비중 더욱 증가)
        self.source_weights = {
            'news': 0.6,      # 뉴스 (FinBERT + Feedly) - 더욱 증가
            'social': 0.2,    # 소셜 미디어 
            'forum': 0.1,     # 전문 포럼
            'analyst': 0.1    # 애널리스트 리포트
        }
        
        # 시간 감쇠 파라미터 (오래된 감정일수록 영향력 감소)
        self.time_decay_hours = 12  # 12시간 후 50% 감쇠
        
        # 감정 히스토리 (메모리 캐시)
        self.sentiment_history = defaultdict(list)
        
        # 뉴스 캐시 (중복 방지)
        self.news_cache = {}
        self.last_news_update = None
    
    async def initialize(self):
        """비동기 초기화 - 고급 감정 분석 컴포넌트 로드"""
        if self._initialized:
            return
        
        try:
            # 고급 감정 분석 컴포넌트 초기화
            self.sentiment_router = await get_router(self.mode, self.csv_path)
            self.fusion_manager = await get_fusion_manager()
            self.finbert_analyzer = await get_finbert_analyzer()
            
            self._initialized = True
            logger.info(f"SentimentAggregator v3.0 초기화 완료 - 모드: {self.mode}")
            
        except Exception as e:
            logger.error(f"SentimentAggregator 초기화 실패: {e}")
            # 폴백: 기본 모드로 동작
            self._initialized = False
            raise
        
    async def aggregate_sentiment(
        self,
        asset: str,
        timestamp: Optional[datetime] = None,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """
        특정 자산의 통합 감정 점수 계산 (FinBERT + Fusion 기반)
        
        Args:
            asset: 자산 심볼 (BTC, SPY 등)
            timestamp: 기준 시간 (None이면 현재)
            lookback_hours: 과거 몇 시간의 데이터를 볼 것인가
            
        Returns:
            {
                'overall': 0.65,      # 전체 감정 점수 (0 ~ 1, FinBERT 정규화)
                'news': 0.7,          # 뉴스 감정 (FinBERT 기반)
                'social': 0.6,        # 소셜 감정
                'trend': 'improving', # 추세 (improving/declining/stable)
                'confidence': 0.8,    # 신뢰도 (0 ~ 1)
                'fusion_metadata': {} # 융합 메타데이터
            }
        """
        # 초기화 확인
        if not self._initialized:
            await self.initialize()
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # 캐시 확인
        cache_key = f"sentiment_agg_v3:{asset}:{timestamp.strftime('%Y%m%d%H')}"
        if self.cache_manager:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return cached
        
        try:
            # 각 소스별 감정 수집 (FinBERT 기반)
            source_sentiments = await self._collect_advanced_sentiments(
                asset, timestamp, lookback_hours
            )
            
            # Fusion Manager를 통한 고급 융합
            if self.fusion_manager and source_sentiments:
                # 소스별 점수를 Fusion Manager 형식으로 변환
                fusion_scores = {}
                for source, data in source_sentiments.items():
                    if data and 'score' in data:
                        fusion_scores[source] = data['score']
                
                # 융합된 점수 계산
                overall_score = await self.fusion_manager.fuse(
                    fusion_scores, 
                    symbol=asset, 
                    timestamp=timestamp
                )
                
                # 융합 통계 가져오기
                fusion_stats = self.fusion_manager.get_statistics(asset)
                
            else:
                # 폴백: 기본 가중 평균
                overall_score = self._calculate_overall_score(source_sentiments)
                fusion_stats = {}
            
            # 추세 분석
            trend = await self._analyze_trend(asset, timestamp)
            
            # 신뢰도 계산 (융합 기반)
            if fusion_stats and 'average_confidence' in fusion_stats:
                confidence = fusion_stats['average_confidence']
            else:
                confidence = self._calculate_confidence(source_sentiments)
            
            result = {
                'overall': float(overall_score),
                'news': source_sentiments.get('news', {}).get('score', 0.5),
                'social': source_sentiments.get('social', {}).get('score', 0.5),
                'trend': trend,
                'confidence': float(confidence),
                'timestamp': timestamp.isoformat(),
                'sources': len(source_sentiments),
                'fusion_metadata': fusion_stats,
                'mode': self.mode
            }
            
            # 캐시 저장
            if self.cache_manager:
                await self._save_to_cache(cache_key, result, ttl=3600)  # 1시간
                
            # 히스토리 업데이트
            self._update_history(asset, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced sentiment aggregation failed: {e}")
            # 폴백: 기본 방식
            return await self._fallback_aggregate_sentiment(asset, timestamp, lookback_hours)
    
    async def _collect_advanced_sentiments(
        self,
        asset: str,
        timestamp: datetime,
        lookback_hours: int
    ) -> Dict[str, Dict]:
        """FinBERT 기반 고급 감정 수집"""
        source_sentiments = {}
        
        start_time = timestamp - timedelta(hours=lookback_hours)
        
        try:
            # 뉴스 감정 수집 (FinBERT + Feedly)
            news_items = await self._fetch_news_items(asset, start_time, timestamp)
            if news_items and self.sentiment_router:
                # Router를 통한 배치 분석 (FinBERT 기반)
                analyzed_articles = await self.sentiment_router.analyze_articles_batch(
                    news_items, symbol=asset
                )
                
                if analyzed_articles:
                    # 감정 점수 평균 계산
                    scores = [article['sentiment_score'] for article in analyzed_articles]
                    confidence_scores = [article.get('confidence', 0.8) for article in analyzed_articles]
                    
                    if scores:
                        weighted_avg = sum(s * c for s, c in zip(scores, confidence_scores)) / sum(confidence_scores)
                        
                        source_sentiments['news'] = {
                            'score': float(weighted_avg),
                            'count': len(scores),
                            'confidence': float(sum(confidence_scores) / len(confidence_scores)),
                            'metadata': {
                                'articles': len(analyzed_articles),
                                'sources': list(set(a.get('source', 'unknown') for a in news_items)),
                                'keywords': [kw for article in analyzed_articles for kw in article.get('keywords', [])],
                                'scenarios': [article.get('scenario_tag', '') for article in analyzed_articles]
                            }
                        }
            
            # 소셜 미디어 감정 (추후 구현 - 현재는 더미)
            source_sentiments['social'] = self._generate_dummy_sentiment_data('social', 10)
            source_sentiments['forum'] = self._generate_dummy_sentiment_data('forum', 5)
            
            logger.info(f"Advanced sentiment collection complete: {len(source_sentiments)} sources for {asset}")
            return source_sentiments
            
        except Exception as e:
            logger.error(f"Advanced sentiment collection failed: {e}")
            # 폴백: 기본 수집 방식
            return await self._collect_source_sentiments(asset, timestamp, lookback_hours)
    
    def _generate_dummy_sentiment_data(self, source: str, count: int) -> Dict:
        """더미 감정 데이터 생성 (고급 형식)"""
        base_score = 0.5 + np.random.randn() * 0.1  # 0.3 ~ 0.7 범위
        return {
            'score': float(np.clip(base_score, 0.0, 1.0)),
            'count': count,
            'confidence': float(np.random.uniform(0.6, 0.9)),
            'metadata': {
                'source': source,
                'generated': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def _fallback_aggregate_sentiment(self, asset: str, timestamp: datetime, lookback_hours: int) -> Dict:
        """폴백: 기본 감정 분석 방식"""
        try:
            # 기본 방식으로 감정 수집
            source_sentiments = await self._collect_source_sentiments(asset, timestamp, lookback_hours)
            
            # 기본 가중 평균
            overall_score = self._calculate_overall_score(source_sentiments)
            
            # 기본 추세 및 신뢰도
            trend = await self._analyze_trend(asset, timestamp)
            confidence = self._calculate_confidence(source_sentiments)
            
            return {
                'overall': overall_score,
                'news': source_sentiments.get('news', {}).get('score', 0.5),
                'social': source_sentiments.get('social', {}).get('score', 0.5),
                'trend': trend,
                'confidence': confidence,
                'timestamp': timestamp.isoformat(),
                'sources': len(source_sentiments),
                'mode': 'fallback',
                'fusion_metadata': {}
            }
        except Exception as e:
            logger.error(f"Fallback sentiment aggregation failed: {e}")
            return {
                'overall': 0.5,
                'news': 0.5,
                'social': 0.5,
                'trend': 'stable',
                'confidence': 0.0,
                'timestamp': timestamp.isoformat(),
                'sources': 0,
                'mode': 'error',
                'error': str(e)
            }
    
    async def _collect_source_sentiments(
        self,
        asset: str,
        timestamp: datetime,
        lookback_hours: int
    ) -> Dict[str, List[Dict]]:
        """각 소스에서 감정 데이터 수집"""
        source_sentiments = defaultdict(list)
        
        start_time = timestamp - timedelta(hours=lookback_hours)
        
        # 뉴스 감정 수집 (실제로는 뉴스 수집기에서)
        news_items = await self._fetch_news_items(asset, start_time, timestamp)
        if news_items:
            news_sentiments = await self.batch_processor.process_news_items(news_items)
            source_sentiments['news'] = news_sentiments
        
        # 소셜 미디어 감정 (추후 구현)
        # social_items = await self._fetch_social_items(asset, start_time, timestamp)
        # source_sentiments['social'] = social_items
        
        # 임시 더미 데이터
        source_sentiments['social'] = self._generate_dummy_sentiments('social', 10)
        source_sentiments['forum'] = self._generate_dummy_sentiments('forum', 5)
        
        return source_sentiments
    
    def _apply_time_weights(
        self,
        source_sentiments: Dict[str, List[Dict]],
        current_time: datetime
    ) -> Dict[str, Dict]:
        """시간 기반 가중치 적용"""
        weighted_results = {}
        
        for source, items in source_sentiments.items():
            if not items:
                continue
                
            scores = []
            weights = []
            
            for item in items:
                # 감정 점수 추출
                if 'sentiment_score' in item:
                    score = item['sentiment_score']
                elif 'sentiment' in item:
                    sent = item['sentiment']
                    score = sent.get('positive', 0) - sent.get('negative', 0)
                else:
                    continue
                
                # 시간 가중치 계산
                item_time = item.get('timestamp', current_time)
                if isinstance(item_time, str):
                    item_time = datetime.fromisoformat(item_time)
                    
                hours_ago = (current_time - item_time).total_seconds() / 3600
                time_weight = np.exp(-hours_ago / self.time_decay_hours)
                
                scores.append(score)
                weights.append(time_weight)
            
            if scores:
                # 가중 평균 계산
                weighted_score = np.average(scores, weights=weights)
                weighted_results[source] = {
                    'score': float(weighted_score),
                    'count': len(scores),
                    'avg_weight': float(np.mean(weights))
                }
        
        return weighted_results
    
    def _calculate_overall_score(self, weighted_sentiments: Dict[str, Dict]) -> float:
        """전체 감정 점수 계산"""
        if not weighted_sentiments:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for source, data in weighted_sentiments.items():
            source_weight = self.source_weights.get(source, 0.1)
            score = data.get('score', 0.0)
            count_factor = min(1.0, data.get('count', 1) / 10)  # 데이터 수 고려
            
            total_score += score * source_weight * count_factor
            total_weight += source_weight * count_factor
        
        if total_weight > 0:
            return float(np.clip(total_score / total_weight, -1, 1))
        
        return 0.0
    
    async def _analyze_trend(self, asset: str, current_time: datetime) -> str:
        """감정 추세 분석"""
        history = self.sentiment_history.get(asset, [])
        
        if len(history) < 3:
            return 'stable'
        
        # 최근 3개 시점의 감정 점수
        recent_scores = [h['overall'] for h in history[-3:]]
        
        # 추세 계산
        if len(recent_scores) >= 2:
            diff = recent_scores[-1] - recent_scores[0]
            
            if diff > 0.1:
                return 'improving'
            elif diff < -0.1:
                return 'declining'
        
        return 'stable'
    
    def _calculate_confidence(self, weighted_sentiments: Dict[str, Dict]) -> float:
        """신뢰도 계산 (데이터 양과 일관성 기반)"""
        if not weighted_sentiments:
            return 0.0
        
        # 데이터 소스 수
        source_count = len(weighted_sentiments)
        source_factor = min(1.0, source_count / 3)  # 3개 이상 소스면 최대
        
        # 데이터 양
        total_items = sum(data.get('count', 0) for data in weighted_sentiments.values())
        volume_factor = min(1.0, total_items / 20)  # 20개 이상이면 최대
        
        # 감정 일관성 (표준편차가 낮을수록 신뢰도 높음)
        scores = [data.get('score', 0) for data in weighted_sentiments.values()]
        if len(scores) > 1:
            consistency_factor = 1.0 - min(1.0, np.std(scores))
        else:
            consistency_factor = 0.5
        
        # 종합 신뢰도
        confidence = (source_factor * 0.3 + volume_factor * 0.4 + consistency_factor * 0.3)
        
        return float(np.clip(confidence, 0, 1))
    
    def _update_history(self, asset: str, sentiment_data: Dict):
        """감정 히스토리 업데이트"""
        history = self.sentiment_history[asset]
        history.append(sentiment_data)
        
        # 최대 24시간 데이터만 유지
        cutoff_time = datetime.now() - timedelta(hours=24)
        history[:] = [
            h for h in history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
    
    async def _fetch_news_items(
        self,
        asset: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Feedly에서 실제 뉴스 아이템 가져오기"""
        try:
            # Feedly 연결 확인
            if not self.feedly_connected:
                await self.feedly_collector.connect()
                self.feedly_connected = True
            
            # 캐시 확인 (30분마다 업데이트)
            cache_key = f"news_{asset}_{start_time.strftime('%Y%m%d%H')}"
            current_time = datetime.now()
            
            if (self.last_news_update and 
                current_time - self.last_news_update < timedelta(minutes=30) and
                cache_key in self.news_cache):
                logger.debug(f"Using cached news for {asset}")
                return self.news_cache[cache_key]
            
            # Feedly에서 최신 암호화폐 뉴스 수집
            hours_back = int((end_time - start_time).total_seconds() / 3600)
            articles = await self.feedly_collector.get_latest_crypto_news(
                hours_back=max(hours_back, 6),  # 최소 6시간
                max_articles=50
            )
            
            # 자산별 필터링 (BTC -> bitcoin, ETH -> ethereum 등)
            asset_keywords = self._get_asset_keywords(asset)
            filtered_articles = []
            
            for article in articles:
                article_text = (article.title + ' ' + article.summary).lower()
                
                # 자산 관련성 체크
                relevance_score = 0
                for keyword in asset_keywords:
                    if keyword in article_text:
                        relevance_score += 1
                
                if relevance_score > 0 or asset.lower() == 'crypto':  # crypto 전반
                    # 뉴스 아이템 변환
                    news_item = {
                        'title': article.title,
                        'content': article.summary or article.content[:500],
                        'source': article.source,
                        'url': article.url,
                        'timestamp': article.published,
                        'keywords': article.keywords,
                        'engagement': article.engagement,
                        'relevance_score': relevance_score
                    }
                    filtered_articles.append(news_item)
            
            # 관련성과 최신성 기준 정렬
            filtered_articles.sort(
                key=lambda x: (x['relevance_score'], x['timestamp']),
                reverse=True
            )
            
            # 상위 20개만 선택
            result = filtered_articles[:20]
            
            # 캐시 업데이트
            self.news_cache[cache_key] = result
            self.last_news_update = current_time
            
            logger.info(f"Fetched {len(result)} relevant news articles for {asset}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch news from Feedly: {e}")
            # 실패시 더미 데이터 반환
            return self._generate_dummy_news(asset, 5)
    
    def _get_asset_keywords(self, asset: str) -> List[str]:
        """자산별 관련 키워드 매핑"""
        keyword_map = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'BNB': ['binance', 'bnb'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'DOT': ['polkadot', 'dot'],
            'AVAX': ['avalanche', 'avax'],
            'MATIC': ['polygon', 'matic'],
            'LINK': ['chainlink', 'link'],
            'UNI': ['uniswap', 'uni'],
            'DOGE': ['dogecoin', 'doge'],
            'SHIB': ['shiba', 'shib'],
            'CRYPTO': ['cryptocurrency', 'crypto', 'bitcoin', 'ethereum', 'blockchain']
        }
        
        return keyword_map.get(asset.upper(), [asset.lower()])
    
    def _generate_dummy_news(self, asset: str, count: int) -> List[Dict]:
        """Feedly 실패시 더미 뉴스 생성"""
        dummy_news = []
        base_time = datetime.now()
        
        for i in range(count):
            dummy_news.append({
                'title': f'{asset} Market Update #{i+1}',
                'content': f'Latest developments in {asset} market...',
                'source': 'Market News',
                'timestamp': base_time - timedelta(hours=i * 2),
                'keywords': [asset.lower()],
                'engagement': {'shares': 0, 'comments': 0, 'likes': 0},
                'relevance_score': 1
            })
        
        return dummy_news
    
    def _generate_dummy_sentiments(self, source: str, count: int) -> List[Dict]:
        """더미 감정 데이터 생성"""
        sentiments = []
        base_time = datetime.now()
        
        for i in range(count):
            sentiment_score = np.random.randn() * 0.3  # -0.9 ~ 0.9 범위
            sentiments.append({
                'source': source,
                'sentiment_score': float(np.clip(sentiment_score, -1, 1)),
                'timestamp': base_time - timedelta(hours=i * 2),
                'confidence': np.random.uniform(0.5, 1.0)
            })
        
        return sentiments
    
    async def _get_from_cache(self, key: str) -> Optional[Dict]:
        """캐시에서 데이터 조회"""
        # Redis 캐시 매니저 연동 (추후 구현)
        return None
    
    async def _save_to_cache(self, key: str, data: Dict, ttl: int):
        """캐시에 데이터 저장"""
        # Redis 캐시 매니저 연동 (추후 구현)
        pass
    
    def get_sentiment_summary(self, asset: str) -> Dict:
        """특정 자산의 감정 요약"""
        history = self.sentiment_history.get(asset, [])
        
        if not history:
            return {'status': 'no_data'}
        
        recent_scores = [h['overall'] for h in history[-10:]]
        
        return {
            'current': history[-1]['overall'] if history else 0,
            'average_24h': float(np.mean(recent_scores)),
            'std_24h': float(np.std(recent_scores)),
            'min_24h': float(np.min(recent_scores)),
            'max_24h': float(np.max(recent_scores)),
            'trend': history[-1].get('trend', 'stable') if history else 'stable',
            'last_update': history[-1]['timestamp'] if history else None
        }
    
    async def get_real_time_sentiment(self, asset: str = "CRYPTO") -> Dict[str, Any]:
        """실시간 센티멘트 분석 (FinBERT + Feedly + Fusion 기반)"""
        try:
            # 초기화 확인
            if not self._initialized:
                await self.initialize()
            
            current_time = datetime.now()
            start_time = current_time - timedelta(hours=6)  # 최근 6시간
            
            # 뉴스 수집
            news_items = await self._fetch_news_items(asset, start_time, current_time)
            
            if not news_items:
                return {
                    'sentiment_score': 0.5,
                    'confidence': 0.0,
                    'article_count': 0,
                    'last_update': current_time.isoformat(),
                    'source': 'finbert_feedly',
                    'status': 'no_data'
                }
            
            # FinBERT 기반 고급 감정 분석
            if self.sentiment_router:
                analyzed_articles = await self.sentiment_router.analyze_articles_batch(
                    news_items, symbol=asset
                )
                
                if analyzed_articles:
                    # 고급 통계 계산
                    sentiment_scores = [article['sentiment_score'] for article in analyzed_articles]
                    confidence_scores = [article.get('confidence', 0.8) for article in analyzed_articles]
                    keywords = [kw for article in analyzed_articles for kw in article.get('keywords', [])]
                    scenarios = [article.get('scenario_tag', '') for article in analyzed_articles]
                    
                    # Fusion Manager를 통한 융합 (단일 뉴스 소스지만 보정 효과)
                    if self.fusion_manager and sentiment_scores:
                        avg_score = sum(sentiment_scores) / len(sentiment_scores)
                        fused_score = await self.fusion_manager.fuse(
                            {"news": avg_score}, 
                            symbol=asset, 
                            timestamp=current_time
                        )
                    else:
                        # 가중 평균 계산
                        if confidence_scores and sum(confidence_scores) > 0:
                            fused_score = sum(
                                s * c for s, c in zip(sentiment_scores, confidence_scores)
                            ) / sum(confidence_scores)
                        else:
                            fused_score = sum(sentiment_scores) / len(sentiment_scores)
                    
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    
                    # 트렌드 분석
                    positive_ratio = sum(1 for s in sentiment_scores if s > 0.6) / len(sentiment_scores)
                    negative_ratio = sum(1 for s in sentiment_scores if s < 0.4) / len(sentiment_scores)
                    
                    if positive_ratio > 0.6:
                        trend = "bullish"
                    elif negative_ratio > 0.6:
                        trend = "bearish"
                    else:
                        trend = "neutral"
                    
                    # 결과 반환
                    return {
                        'sentiment_score': float(fused_score),
                        'confidence': float(avg_confidence),
                        'article_count': len(analyzed_articles),
                        'positive_ratio': positive_ratio,
                        'negative_ratio': negative_ratio,
                        'neutral_ratio': 1.0 - positive_ratio - negative_ratio,
                        'trend': trend,
                        'last_update': current_time.isoformat(),
                        'source': 'finbert_feedly_fusion',
                        'status': 'success',
                        'top_sources': list(set(item.get('source', 'unknown') for item in news_items[:5])),
                        'top_keywords': list(set(keywords[:10])),  # 상위 10개 키워드
                        'scenario_distribution': {
                            scenario: scenarios.count(scenario) / len(scenarios) 
                            for scenario in set(scenarios) if scenario
                        },
                        'analysis_metadata': {
                            'finbert_enabled': True,
                            'fusion_enabled': self.fusion_manager is not None,
                            'mode': self.mode
                        }
                    }
            
            # 폴백: Feedly 기본 감정 분석
            sentiment_scores = []
            confidence_scores = []
            
            for item in news_items:
                # Feedly 자체 감정 분석 사용
                sentiment_data = self.feedly_collector.calculate_basic_sentiment(
                    # NewsArticle 객체로 변환
                    type('Article', (), {
                        'title': item['title'],
                        'summary': item['content'],
                        'engagement': item['engagement']
                    })()
                )
                
                sentiment_scores.append(sentiment_data['sentiment'])
                confidence_scores.append(sentiment_data['confidence'])
            
            # 가중 평균 계산
            if confidence_scores and sum(confidence_scores) > 0:
                weighted_sentiment = sum(
                    s * c for s, c in zip(sentiment_scores, confidence_scores)
                ) / sum(confidence_scores)
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                weighted_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                avg_confidence = 0.5
            
            # 결과 반환 (폴백)
            return {
                'sentiment_score': float(weighted_sentiment),
                'confidence': float(avg_confidence),
                'article_count': len(news_items),
                'positive_ratio': sum(1 for s in sentiment_scores if s > 0.6) / len(sentiment_scores),
                'negative_ratio': sum(1 for s in sentiment_scores if s < 0.4) / len(sentiment_scores),
                'last_update': current_time.isoformat(),
                'source': 'feedly_fallback',
                'status': 'success',
                'top_sources': list(set(item['source'] for item in news_items[:5])),
                'analysis_metadata': {
                    'finbert_enabled': False,
                    'fusion_enabled': False,
                    'mode': 'fallback'
                }
            }
            
        except Exception as e:
            logger.error(f"Real-time sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'article_count': 0,
                'last_update': datetime.now().isoformat(),
                'source': 'error',
                'status': 'error',
                'error': str(e)
            }
    
    async def close(self):
        """리소스 정리"""
        try:
            # Feedly 수집기 정리
            if self.feedly_collector:
                await self.feedly_collector.close()
            
            # 고급 감정 분석 컴포넌트 정리
            if self.sentiment_router:
                await self.sentiment_router.close()
            
            if self.fusion_manager:
                await self.fusion_manager.close()
                
            if self.finbert_analyzer:
                await self.finbert_analyzer.close()
            
            # 캐시 정리
            self.news_cache.clear()
            self.sentiment_history.clear()
            
            logger.info("SentimentAggregator v3.0 closed - all resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during SentimentAggregator cleanup: {e}")
    
    async def switch_mode(self, new_mode: str, csv_path: str = None):
        """모드 전환 (live <-> backtest)"""
        if self.sentiment_router:
            await self.sentiment_router.switch_mode(new_mode, csv_path)
            self.mode = new_mode
            self.csv_path = csv_path
            logger.info(f"SentimentAggregator 모드 전환: {new_mode}")
        else:
            logger.warning("SentimentRouter가 초기화되지 않아 모드 전환 불가")
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.mode