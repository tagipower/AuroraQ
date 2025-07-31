#!/usr/bin/env python3
"""
News Aggregation System
모든 무료 뉴스 수집기를 통합하는 시스템
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base_collector import (
    NewsAggregator, NewsArticle, NewsCategory, 
    CollectorConfig, SentimentScore
)
from .collectors.google_news_collector import GoogleNewsCollector
from .collectors.yahoo_finance_collector import YahooFinanceCollector
from .collectors.reddit_collector import RedditCollector
from .collectors.newsapi_collector import NewsAPICollector
from .collectors.finnhub_collector import FinnhubCollector
from .bitcoin_futures_filter import BitcoinFuturesNewsFilter, MarketRelevanceScore
from .advanced_news_query_engine import AdvancedNewsQueryEngine, SearchQuery, MarketImpactCategory


class AuroraQNewsAggregator(NewsAggregator):
    """AuroraQ 전용 뉴스 수집 시스템"""
    
    def __init__(self):
        super().__init__()
        self.setup_collectors()
        
        # 비트코인 선물시장 특화 필터 초기화
        self.btc_futures_filter = BitcoinFuturesNewsFilter()
        
        # 고급 뉴스 쿼리 엔진 초기화
        self.query_engine = AdvancedNewsQueryEngine()
        
        # 수집 통계
        self.collection_stats = {
            "total_articles": 0,
            "by_category": {},
            "by_source": {},
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "last_collection": None,
            "high_impact_articles": 0,
            "market_moving_events": []
        }
    
    def setup_collectors(self):
        """모든 수집기 초기화 및 등록"""
        try:
            # 1. Google News (항상 사용 가능)
            google_collector = GoogleNewsCollector()
            self.register_collector("google_news", google_collector)
            self.logger.info("✅ Google News collector registered")
            
            # 2. Yahoo Finance (항상 사용 가능)
            yahoo_collector = YahooFinanceCollector()
            self.register_collector("yahoo_finance", yahoo_collector)
            self.logger.info("✅ Yahoo Finance collector registered")
            
            # 3. Reddit (API 차단으로 비활성화됨)
            reddit_collector = RedditCollector()
            self.register_collector("reddit", reddit_collector)
            self.logger.warning("⚠️ Reddit collector registered but API is blocked - will return empty results")
            
            # 4. NewsAPI (API 키 필요)
            if os.getenv("NEWSAPI_KEY"):
                newsapi_collector = NewsAPICollector()
                self.register_collector("newsapi", newsapi_collector)
                self.logger.info("✅ NewsAPI collector registered")
            else:
                self.logger.warning("⚠️ NewsAPI key not found - skipping")
            
            # 5. Finnhub (API 키 필요)
            if os.getenv("FINNHUB_API_KEY"):
                finnhub_collector = FinnhubCollector()
                self.register_collector("finnhub", finnhub_collector)
                self.logger.info("✅ Finnhub collector registered")
            else:
                self.logger.warning("⚠️ Finnhub key not found - skipping")
                
        except Exception as e:
            self.logger.error(f"Error setting up collectors: {e}")
    
    async def collect_comprehensive_news(self, 
                                       categories: List[NewsCategory] = None,
                                       hours_back: int = 6,
                                       articles_per_category: int = 20) -> Dict[str, List[NewsArticle]]:
        """포괄적 뉴스 수집"""
        if categories is None:
            categories = [
                NewsCategory.HEADLINE,
                NewsCategory.CRYPTO,
                NewsCategory.FINANCE,
                NewsCategory.MACRO,
                NewsCategory.BREAKING
            ]
        
        results = {}
        since = datetime.now() - timedelta(hours=hours_back)
        
        self.logger.info(f"🔄 Starting comprehensive news collection for {len(categories)} categories")
        
        # 병렬로 카테고리별 수집
        tasks = []
        for category in categories:
            task = self._collect_category_news(category, since, articles_per_category)
            tasks.append(task)
        
        category_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        for i, category in enumerate(categories):
            if isinstance(category_results[i], Exception):
                self.logger.error(f"Error collecting {category.value}: {category_results[i]}")
                results[category.value] = []
            else:
                results[category.value] = category_results[i]
                self.logger.info(f"✅ {category.value}: {len(category_results[i])} articles")
        
        # 통계 업데이트
        self._update_collection_stats(results)
        
        return results
    
    async def _collect_category_news(self, category: NewsCategory, 
                                   since: datetime, 
                                   count: int) -> List[NewsArticle]:
        """특정 카테고리 뉴스 수집"""
        articles = []
        
        # 카테고리별 키워드 매핑
        category_keywords = {
            NewsCategory.CRYPTO: ["bitcoin", "ethereum", "crypto", "blockchain"],
            NewsCategory.FINANCE: ["stock", "market", "trading", "finance"],
            NewsCategory.MACRO: ["federal reserve", "inflation", "gdp", "cpi"],
            NewsCategory.PERSON: ["ceo", "elon musk", "jerome powell"],
            NewsCategory.BREAKING: ["breaking", "urgent", "alert"]
        }
        
        keywords = category_keywords.get(category, ["news"])
        
        # 각 수집기에서 병렬 수집
        tasks = []
        for name, collector in self.collectors.items():
            if category == NewsCategory.HEADLINE:
                task = collector.collect_headlines(count=count//len(self.collectors))
            elif category == NewsCategory.BREAKING:
                task = collector.get_breaking_news(minutes=60)
            else:
                task = collector.search_news(keywords, since=since, count=count//len(self.collectors))
            
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in collector {i}: {result}")
                    continue
                
                for article in result:
                    article.category = category  # 카테고리 강제 설정
                    articles.append(article)
        
        except Exception as e:
            self.logger.error(f"Error in category collection: {e}")
        
        # 중복 제거 및 정렬
        return self._deduplicate_and_sort(articles)[:count]
    
    async def get_market_moving_news(self, minutes: int = 30) -> Dict[str, Any]:
        """시장 영향 뉴스 분석"""
        self.logger.info(f"📈 Analyzing market-moving news (last {minutes} minutes)")
        
        # 최근 속보 수집
        breaking_news = await self.get_breaking_news_all(minutes=minutes)
        
        # 고중요도 뉴스 필터링
        high_impact_news = []
        market_keywords = [
            "federal reserve", "inflation", "recession", "rate hike",
            "earnings", "bitcoin", "crypto", "stock market",
            "gdp", "unemployment", "cpi", "fomc"
        ]
        
        for article in breaking_news:
            text = (article.title + " " + article.summary).lower()
            if any(keyword in text for keyword in market_keywords):
                high_impact_news.append(article)
        
        # 감정 분석
        sentiment_scores = []
        for article in high_impact_news:
            if article.sentiment_score:
                sentiment_scores.append(article.sentiment_score)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # 주요 이벤트 추출
        major_events = self._extract_major_events(high_impact_news)
        
        return {
            "analysis_time": datetime.now().isoformat(),
            "time_window_minutes": minutes,
            "total_breaking_news": len(breaking_news),
            "high_impact_count": len(high_impact_news),
            "market_sentiment": {
                "score": avg_sentiment,
                "label": "bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral"
            },
            "major_events": major_events,
            "top_articles": [
                {
                    "title": article.title,
                    "source": article.source,
                    "sentiment": article.sentiment_score,
                    "url": article.url,
                    "published": article.published_date.isoformat()
                }
                for article in high_impact_news[:5]
            ]
        }
    
    def _extract_major_events(self, articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """주요 이벤트 추출"""
        events = []
        
        # 이벤트 패턴 매칭
        event_patterns = {
            "fed_announcement": ["federal reserve", "fed", "fomc", "jerome powell"],
            "earnings_release": ["earnings", "quarterly results", "revenue"],
            "crypto_movement": ["bitcoin", "ethereum", "crypto"],
            "economic_data": ["inflation", "cpi", "gdp", "unemployment"],
            "market_volatility": ["crash", "surge", "plunge", "rally"]
        }
        
        for event_type, keywords in event_patterns.items():
            matching_articles = []
            
            for article in articles:
                text = (article.title + " " + article.summary).lower()
                if any(keyword in text for keyword in keywords):
                    matching_articles.append(article)
            
            if matching_articles:
                # 가장 높은 관련성을 가진 기사
                best_article = max(matching_articles, 
                                 key=lambda x: x.relevance_score or 0)
                
                events.append({
                    "event_type": event_type,
                    "article_count": len(matching_articles),
                    "primary_article": {
                        "title": best_article.title,
                        "source": best_article.source,
                        "sentiment": best_article.sentiment_score
                    },
                    "overall_sentiment": sum(a.sentiment_score or 0 for a in matching_articles) / len(matching_articles)
                })
        
        return events
    
    async def analyze_sentiment_trends(self, hours: int = 24) -> Dict[str, Any]:
        """감정 트렌드 분석"""
        self.logger.info(f"📊 Analyzing sentiment trends (last {hours} hours)")
        
        # 시간별 데이터 수집
        time_windows = []
        window_size = hours // 6  # 6개 구간으로 나누기
        
        for i in range(6):
            window_start = datetime.now() - timedelta(hours=hours - (i * window_size))
            window_end = datetime.now() - timedelta(hours=hours - ((i + 1) * window_size))
            
            # 해당 시간대 뉴스 수집
            window_articles = []
            for name, collector in self.collectors.items():
                try:
                    articles = await collector.search_news(
                        keywords=["bitcoin", "stock", "market"],
                        since=window_end,
                        until=window_start,
                        count=20
                    )
                    window_articles.extend(articles)
                except:
                    continue
            
            # 감정 집계
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            total_score = 0
            
            for article in window_articles:
                if article.sentiment_label:
                    if article.sentiment_label.value > 0:
                        sentiments["positive"] += 1
                    elif article.sentiment_label.value < 0:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
                
                if article.sentiment_score:
                    total_score += article.sentiment_score
            
            avg_sentiment = total_score / len(window_articles) if window_articles else 0
            
            time_windows.append({
                "time_start": window_start.isoformat(),
                "time_end": window_end.isoformat(),
                "article_count": len(window_articles),
                "sentiment_distribution": sentiments,
                "average_sentiment": avg_sentiment
            })
        
        return {
            "analysis_period_hours": hours,
            "time_windows": time_windows,
            "overall_trend": self._calculate_trend(time_windows),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, windows: List[Dict[str, Any]]) -> str:
        """트렌드 계산"""
        if len(windows) < 2:
            return "insufficient_data"
        
        early_sentiment = sum(w["average_sentiment"] for w in windows[:3]) / 3
        late_sentiment = sum(w["average_sentiment"] for w in windows[-3:]) / 3
        
        change = late_sentiment - early_sentiment
        
        if change > 0.1:
            return "improving"
        elif change < -0.1:
            return "deteriorating"
        else:
            return "stable"
    
    async def get_personalized_feed(self, interests: List[str], count: int = 50) -> List[NewsArticle]:
        """개인화된 뉴스 피드"""
        self.logger.info(f"🎯 Creating personalized feed for interests: {interests}")
        
        # 관심사별 가중치
        interest_weights = {interest: 1.0 for interest in interests}
        
        # 각 관심사별로 뉴스 수집
        all_articles = []
        articles_per_interest = count // len(interests) if interests else count
        
        for interest in interests:
            articles = await self.search_all_sources(
                keywords=[interest],
                since=datetime.now() - timedelta(hours=24),
                count_per_source=articles_per_interest
            )
            
            # 관심사 가중치 적용
            for article in articles:
                if article.relevance_score:
                    article.relevance_score *= interest_weights[interest]
            
            all_articles.extend(articles)
        
        # 중복 제거 및 관련성 점수로 정렬
        unique_articles = self._deduplicate_and_sort(all_articles)
        
        return unique_articles[:count]
    
    def _update_collection_stats(self, results: Dict[str, List[NewsArticle]]):
        """수집 통계 업데이트"""
        total_articles = sum(len(articles) for articles in results.values())
        self.collection_stats["total_articles"] = total_articles
        self.collection_stats["last_collection"] = datetime.now().isoformat()
        
        # 카테고리별 통계
        for category, articles in results.items():
            self.collection_stats["by_category"][category] = len(articles)
        
        # 소스별 통계
        source_counts = {}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for articles in results.values():
            for article in articles:
                source_counts[article.source] = source_counts.get(article.source, 0) + 1
                
                if article.sentiment_label:
                    if article.sentiment_label.value > 0:
                        sentiment_counts["positive"] += 1
                    elif article.sentiment_label.value < 0:
                        sentiment_counts["negative"] += 1
                    else:
                        sentiment_counts["neutral"] += 1
        
        self.collection_stats["by_source"] = source_counts
        self.collection_stats["sentiment_distribution"] = sentiment_counts
    
    async def get_system_health(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        health_data = {
            "status": "healthy",
            "collectors": {},
            "total_collectors": len(self.collectors),
            "active_collectors": 0,
            "last_collection": self.collection_stats.get("last_collection"),
            "collection_stats": self.collection_stats.copy()
        }
        
        # 각 수집기 상태 확인
        for name, collector in self.collectors.items():
            try:
                collector_health = await collector.health_check()
                health_data["collectors"][name] = collector_health
                
                if collector_health.get("status") == "healthy":
                    health_data["active_collectors"] += 1
                    
            except Exception as e:
                health_data["collectors"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # 전체 상태 결정
        if health_data["active_collectors"] == 0:
            health_data["status"] = "critical"
        elif health_data["active_collectors"] < health_data["total_collectors"] / 2:
            health_data["status"] = "degraded"
        
        return health_data
    
    async def export_news_data(self, format: str = "json", 
                              since: Optional[datetime] = None) -> str:
        """뉴스 데이터 내보내기"""
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        # 최근 뉴스 수집
        all_news = await self.collect_comprehensive_news(hours_back=24)
        
        if format.lower() == "json":
            import json
            return json.dumps({
                "export_time": datetime.now().isoformat(),
                "since": since.isoformat(),
                "total_articles": sum(len(articles) for articles in all_news.values()),
                "categories": {
                    category: [article.to_dict() for article in articles]
                    for category, articles in all_news.items()
                }
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def get_bitcoin_futures_impact_news(self, hours_back: int = 6, 
                                            min_impact_score: float = 0.6) -> Dict[str, Any]:
        """비트코인 선물시장 영향 뉴스 분석"""
        self.logger.info(f"🎯 Bitcoin futures impact analysis (last {hours_back} hours, min score: {min_impact_score})")
        
        # 1. 포괄적 뉴스 수집
        all_categories = [
            NewsCategory.CRYPTO, NewsCategory.FINANCE, 
            NewsCategory.MACRO, NewsCategory.HEADLINE, 
            NewsCategory.BREAKING
        ]
        
        comprehensive_news = await self.collect_comprehensive_news(
            categories=all_categories,
            hours_back=hours_back,
            articles_per_category=20
        )
        
        # 2. 모든 기사를 하나의 리스트로 통합
        all_articles = []
        for articles_list in comprehensive_news.values():
            all_articles.extend(articles_list)
        
        # 3. 비트코인 선물시장 특화 필터링
        market_analysis = self.btc_futures_filter.get_market_moving_news(all_articles)
        
        # 4. 고영향 기사만 추출
        high_impact_articles = [
            article_data for article_data in market_analysis['top_articles']
            if article_data['composite_score'] >= min_impact_score
        ]
        
        # 5. 통계 업데이트
        self.collection_stats["high_impact_articles"] = len(high_impact_articles)
        self.collection_stats["market_moving_events"] = market_analysis['event_type_distribution']
        
        return {
            "analysis_time": datetime.now().isoformat(),
            "period_hours": hours_back,
            "min_impact_score": min_impact_score,
            "total_articles_analyzed": len(all_articles),
            "high_impact_count": len(high_impact_articles),
            "market_sentiment": market_analysis['market_sentiment'],
            "impact_distribution": market_analysis['impact_distribution'],
            "event_type_distribution": market_analysis['event_type_distribution'],
            "high_impact_articles": high_impact_articles,
            "trading_signals": self._generate_trading_signals(market_analysis),
            "summary": self._generate_market_summary(market_analysis, high_impact_articles)
        }
    
    def _generate_trading_signals(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """시장 분석 기반 거래 신호 생성"""
        market_sentiment = market_analysis['market_sentiment']
        impact_dist = market_analysis['impact_distribution']
        
        # 기본 신호 계산
        sentiment_score = market_sentiment['score']
        
        # 긴급성 가중치 계산
        urgency_weight = 0.0
        if 'critical' in impact_dist and impact_dist['critical']:
            urgency_weight += len(impact_dist['critical']) * 0.4
        if 'high' in impact_dist and impact_dist['high']:
            urgency_weight += len(impact_dist['high']) * 0.3
        
        # 최종 신호 점수 (-1.0 to +1.0)
        signal_strength = max(-1.0, min(1.0, sentiment_score + urgency_weight * 0.2))
        
        # 신호 분류
        if signal_strength > 0.6:
            signal = "STRONG_BUY"
        elif signal_strength > 0.3:
            signal = "BUY"
        elif signal_strength > -0.3:
            signal = "NEUTRAL"
        elif signal_strength > -0.6:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"
        
        # 시간 민감성 (분 단위)
        time_sensitivity = 240  # 기본 4시간
        if 'critical' in impact_dist and impact_dist['critical']:
            time_sensitivity = min([article['time_sensitivity'] for article in impact_dist['critical']])
        elif 'high' in impact_dist and impact_dist['high']:
            time_sensitivity = min([article['time_sensitivity'] for article in impact_dist['high']])
        
        return {
            "signal": signal,
            "strength": abs(signal_strength),
            "direction": "LONG" if signal_strength > 0 else "SHORT" if signal_strength < 0 else "FLAT",
            "confidence": min(1.0, abs(signal_strength) * 1.2),
            "time_sensitivity_minutes": time_sensitivity,
            "recommended_position_size": self._calculate_position_size(signal_strength),
            "risk_level": self._assess_risk_level(market_analysis),
            "validity_minutes": time_sensitivity * 2  # 신호 유효성
        }
    
    def _calculate_position_size(self, signal_strength: float) -> float:
        """신호 강도 기반 포지션 크기 계산 (0.0-1.0)"""
        base_size = abs(signal_strength) * 0.5  # 최대 50% 포지션
        
        # 보수적 접근
        if abs(signal_strength) > 0.8:
            return min(0.3, base_size)  # 강한 신호도 30% 제한
        elif abs(signal_strength) > 0.5:
            return min(0.2, base_size)  # 중간 신호는 20% 제한
        else:
            return min(0.1, base_size)  # 약한 신호는 10% 제한
    
    def _assess_risk_level(self, market_analysis: Dict[str, Any]) -> str:
        """위험 수준 평가"""
        impact_dist = market_analysis['impact_distribution']
        event_dist = market_analysis['event_type_distribution']
        
        # 위험 점수 계산
        risk_score = 0.0
        
        # 임팩트 기반 위험도
        if 'critical' in impact_dist:
            risk_score += len(impact_dist['critical']) * 0.4
        if 'high' in impact_dist:
            risk_score += len(impact_dist['high']) * 0.3
        
        # 이벤트 유형 기반 위험도
        high_risk_events = ['regulatory', 'monetary_policy', 'geopolitical']
        for event_type in high_risk_events:
            if event_type in event_dist:
                risk_score += event_dist[event_type] * 0.2
        
        # 위험 수준 분류
        if risk_score > 2.0:
            return "VERY_HIGH"
        elif risk_score > 1.5:
            return "HIGH"
        elif risk_score > 1.0:
            return "MEDIUM"
        elif risk_score > 0.5:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_market_summary(self, market_analysis: Dict[str, Any], 
                               high_impact_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시장 요약 생성"""
        
        # 주요 테마 추출
        themes = {}
        for article in high_impact_articles:
            event_type = article['event_type']
            if event_type not in themes:
                themes[event_type] = []
            themes[event_type].append(article['title'])
        
        # 가장 영향력 있는 테마
        dominant_theme = max(themes.keys(), key=lambda k: len(themes[k])) if themes else "none"
        
        # 시간 민감도 분석
        urgent_articles = [
            article for article in high_impact_articles 
            if article['time_sensitivity_minutes'] <= 60
        ]
        
        return {
            "dominant_theme": dominant_theme,
            "theme_distribution": {theme: len(articles) for theme, articles in themes.items()},
            "urgent_news_count": len(urgent_articles),
            "market_sentiment_label": market_analysis['market_sentiment']['label'],
            "key_concerns": [
                article['title'] for article in high_impact_articles[:3]
            ],
            "recommendation": self._generate_recommendation(market_analysis, high_impact_articles)
        }
    
    def _generate_recommendation(self, market_analysis: Dict[str, Any], 
                               high_impact_articles: List[Dict[str, Any]]) -> str:
        """종합 추천사항 생성"""
        sentiment = market_analysis['market_sentiment']['label']
        urgent_count = len([a for a in high_impact_articles if a['time_sensitivity_minutes'] <= 60])
        total_impact = len(high_impact_articles)
        
        if urgent_count > 3:
            return f"⚠️ 긴급 뉴스 {urgent_count}개 발생. 즉시 포지션 검토 필요"
        elif total_impact > 5 and sentiment == "bearish":
            return f"📉 {total_impact}개 부정적 고영향 뉴스. 보수적 접근 권장"
        elif total_impact > 5 and sentiment == "bullish":
            return f"📈 {total_impact}개 긍정적 고영향 뉴스. 적극적 포지션 고려"
        elif sentiment == "neutral":
            return f"⚖️ 중립적 시장 감정. 관망 또는 소량 포지션 권장"
        else:
            return f"📊 일반적인 시장 상황. 정상적인 거래 전략 유지"
    
    async def get_real_time_market_alerts(self, alert_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """실시간 시장 알림 생성"""
        
        # 최근 30분 뉴스 수집
        recent_news = await self.get_breaking_news_all(minutes=30)
        
        if not recent_news:
            return []
        
        # 고영향 기사 필터링
        high_impact_articles = self.btc_futures_filter.filter_high_impact_articles(
            recent_news, min_composite_score=alert_threshold
        )
        
        alerts = []
        for article, relevance in high_impact_articles:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "alert_level": relevance.impact_level.value.upper(),
                "event_type": relevance.event_type.value,
                "title": article.title,
                "source": article.source,
                "composite_score": relevance.composite_score,
                "time_sensitive_minutes": relevance.time_sensitivity,
                "sentiment_impact": relevance.sentiment_multiplier,
                "recommended_action": self._get_alert_action(relevance),
                "article_url": article.url
            }
            alerts.append(alert)
        
        return alerts
    
    def _get_alert_action(self, relevance: MarketRelevanceScore) -> str:
        """알림별 권장 액션"""
        if relevance.impact_level.value == "critical":
            return "IMMEDIATE_REVIEW"
        elif relevance.impact_level.value == "high":
            return "MONITOR_CLOSELY"
        elif relevance.time_sensitivity <= 60:
            return "CHECK_POSITION"
        else:
            return "STAY_INFORMED"
    
    async def search_with_advanced_query(self, query: str, 
                                       hours_back: int = 24,
                                       min_score: float = 2.0,
                                       max_results: int = 50) -> Dict[str, Any]:
        """Feedly Power Search 스타일 고급 검색"""
        self.logger.info(f"🔍 Advanced query search: '{query}' (last {hours_back}h)")
        
        # 1. 전체 뉴스 수집
        all_categories = [
            NewsCategory.CRYPTO, NewsCategory.FINANCE, 
            NewsCategory.MACRO, NewsCategory.HEADLINE, 
            NewsCategory.BREAKING
        ]
        
        comprehensive_news = await self.collect_comprehensive_news(
            categories=all_categories,
            hours_back=hours_back,
            articles_per_category=30
        )
        
        # 2. 모든 기사를 하나의 리스트로 통합
        all_articles = []
        for articles_list in comprehensive_news.values():
            all_articles.extend(articles_list)
        
        # 3. 고급 쿼리로 검색
        matched_articles = self.query_engine.search_articles_with_advanced_query(
            articles=all_articles,
            query=query,
            min_score=min_score
        )
        
        # 4. 결과 제한
        limited_results = matched_articles[:max_results]
        
        # 5. 결과 분석
        if limited_results:
            avg_score = sum(score for _, score in limited_results) / len(limited_results)
            top_score = limited_results[0][1]
            
            # 카테고리별 분포
            category_dist = {}
            for article, score in limited_results:
                cat = article.category.value
                if cat not in category_dist:
                    category_dist[cat] = []
                category_dist[cat].append({"title": article.title, "score": score})
            
            # 소스별 분포
            source_dist = {}
            for article, score in limited_results:
                source = article.source
                if source not in source_dist:
                    source_dist[source] = 0
                source_dist[source] += 1
        else:
            avg_score = 0.0
            top_score = 0.0
            category_dist = {}
            source_dist = {}
        
        return {
            "query": query,
            "search_time": datetime.now().isoformat(),
            "period_hours": hours_back,
            "min_score": min_score,
            "total_articles_searched": len(all_articles),
            "matched_articles": len(matched_articles),
            "returned_results": len(limited_results),
            "avg_match_score": avg_score,
            "top_match_score": top_score,
            "category_distribution": category_dist,
            "source_distribution": source_dist,
            "results": [
                {
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "published": article.published_date.isoformat(),
                    "category": article.category.value,
                    "match_score": score,
                    "sentiment_score": article.sentiment_score,
                    "summary": article.summary[:200] + "..." if len(article.summary) > 200 else article.summary
                }
                for article, score in limited_results
            ]
        }
    
    async def get_smart_market_queries(self, market_condition: str = "normal") -> List[Dict[str, Any]]:
        """시장 상황별 스마트 쿼리 추천"""
        self.logger.info(f"📊 Generating smart queries for market condition: {market_condition}")
        
        # 시장 상황별 쿼리 생성
        if market_condition.lower() == "volatile":
            categories = [
                MarketImpactCategory.PRICE_MOVEMENT,
                MarketImpactCategory.MACRO_ECONOMIC,
                MarketImpactCategory.REGULATORY_POLICY
            ]
        elif market_condition.lower() == "bullish":
            categories = [
                MarketImpactCategory.INSTITUTIONAL_FLOW,
                MarketImpactCategory.REGULATORY_POLICY,
                MarketImpactCategory.CELEBRITY_INFLUENCE
            ]
        elif market_condition.lower() == "bearish":
            categories = [
                MarketImpactCategory.REGULATORY_POLICY,
                MarketImpactCategory.TECHNOLOGY_SECURITY,
                MarketImpactCategory.GEOPOLITICAL
            ]
        else:  # normal
            categories = [
                MarketImpactCategory.INSTITUTIONAL_FLOW,
                MarketImpactCategory.MACRO_ECONOMIC,
                MarketImpactCategory.PRICE_MOVEMENT
            ]
        
        smart_queries = []
        
        for category in categories:
            category_queries = self.query_engine.build_smart_queries_for_category(
                category=category,
                impact_level="HIGH"
            )
            
            for query in category_queries:
                smart_queries.append({
                    "query": query.raw_query,
                    "category": category.value,
                    "estimated_impact": query.estimated_impact,
                    "priority_score": query.priority_score,
                    "description": self._get_query_description(query, category)
                })
        
        # 우선순위별 정렬
        smart_queries.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return smart_queries[:10]  # 상위 10개만 반환
    
    def _get_query_description(self, query: SearchQuery, category: MarketImpactCategory) -> str:
        """쿼리 설명 생성"""
        descriptions = {
            MarketImpactCategory.REGULATORY_POLICY: "규제 기관 및 정책 변화 모니터링",
            MarketImpactCategory.INSTITUTIONAL_FLOW: "기관 투자자 동향 및 자금 흐름 추적",
            MarketImpactCategory.MACRO_ECONOMIC: "거시경제 지표 및 통화정책 영향 분석",
            MarketImpactCategory.PRICE_MOVEMENT: "가격 변동 및 시장 감정 추적",
            MarketImpactCategory.TECHNOLOGY_SECURITY: "기술적 발전 및 보안 이슈 모니터링",
            MarketImpactCategory.CELEBRITY_INFLUENCE: "주요 인물 발언 및 영향력 추적",
            MarketImpactCategory.GEOPOLITICAL: "지정학적 리스크 및 국제 정세 모니터링"
        }
        return descriptions.get(category, "일반적인 시장 뉴스 모니터링")
    
    async def execute_trending_queries(self, max_results_per_query: int = 10) -> Dict[str, Any]:
        """트렌딩 쿼리 실행"""
        self.logger.info("📈 Executing trending queries")
        
        # 현재 시간대별 트렌딩 쿼리 가져오기
        trending_queries = self.query_engine.get_trending_queries(timeframe_hours=24)
        
        # 전체 뉴스 수집 (한 번만)
        all_categories = [
            NewsCategory.CRYPTO, NewsCategory.FINANCE, 
            NewsCategory.MACRO, NewsCategory.HEADLINE, 
            NewsCategory.BREAKING
        ]
        
        comprehensive_news = await self.collect_comprehensive_news(
            categories=all_categories,
            hours_back=12,
            articles_per_category=25
        )
        
        all_articles = []
        for articles_list in comprehensive_news.values():
            all_articles.extend(articles_list)
        
        # 각 트렌딩 쿼리 실행
        trending_results = {}
        total_high_impact = 0
        
        for i, query in enumerate(trending_queries):
            matched_articles = self.query_engine.search_articles_with_advanced_query(
                articles=all_articles,
                query=query,
                min_score=2.5  # 높은 임계값
            )
            
            limited_results = matched_articles[:max_results_per_query]
            
            if limited_results:
                avg_score = sum(score for _, score in limited_results) / len(limited_results)
                high_impact_count = len([s for _, s in limited_results if s >= 4.0])
                total_high_impact += high_impact_count
                
                trending_results[f"query_{i}"] = {
                    "query": query.raw_query,
                    "category": query.categories[0].value if query.categories else "unknown",
                    "estimated_impact": query.estimated_impact,
                    "priority_score": query.priority_score,
                    "matched_count": len(matched_articles),
                    "returned_count": len(limited_results),
                    "avg_score": avg_score,
                    "high_impact_count": high_impact_count,
                    "top_articles": [
                        {
                            "title": article.title,
                            "source": article.source,
                            "match_score": score,
                            "published": article.published_date.isoformat()
                        }
                        for article, score in limited_results[:3]
                    ]
                }
        
        return {
            "execution_time": datetime.now().isoformat(),
            "total_articles_analyzed": len(all_articles),
            "trending_queries_count": len(trending_queries),
            "total_high_impact_articles": total_high_impact,
            "queries_with_results": len([r for r in trending_results.values() if r["matched_count"] > 0]),
            "market_alert_level": self._assess_market_alert_level(total_high_impact, len(trending_queries)),
            "trending_results": trending_results
        }
    
    def _assess_market_alert_level(self, high_impact_count: int, total_queries: int) -> str:
        """시장 경보 수준 평가"""
        if total_queries == 0:
            return "UNKNOWN"
        
        impact_ratio = high_impact_count / total_queries
        
        if impact_ratio >= 0.8:
            return "CRITICAL"
        elif impact_ratio >= 0.6:
            return "HIGH"
        elif impact_ratio >= 0.4:
            return "MEDIUM"
        elif impact_ratio >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"


# 사용 예제
async def main():
    """뉴스 수집 시스템 테스트"""
    aggregator = AuroraQNewsAggregator()
    
    try:
        # 시스템 상태 확인
        print("🏥 Checking system health...")
        health = await aggregator.get_system_health()
        print(f"Status: {health['status']}")
        print(f"Active collectors: {health['active_collectors']}/{health['total_collectors']}")
        
        # 포괄적 뉴스 수집
        print("\n📰 Collecting comprehensive news...")
        news_data = await aggregator.collect_comprehensive_news(
            categories=[NewsCategory.CRYPTO, NewsCategory.FINANCE],
            hours_back=6,
            articles_per_category=10
        )
        
        for category, articles in news_data.items():
            print(f"\n{category.upper()}: {len(articles)} articles")
            for article in articles[:3]:
                print(f"  - {article.title[:60]}...")
                print(f"    Source: {article.source}, Sentiment: {article.sentiment_label.name if article.sentiment_label else 'N/A'}")
        
        # 시장 영향 뉴스
        print("\n📈 Market-moving news analysis...")
        market_news = await aggregator.get_market_moving_news(minutes=60)
        print(f"Market sentiment: {market_news['market_sentiment']['label']}")
        print(f"High impact articles: {market_news['high_impact_count']}")
        
        # 개인화 피드
        print("\n🎯 Personalized feed...")
        feed = await aggregator.get_personalized_feed(
            interests=["bitcoin", "federal reserve"],
            count=10
        )
        print(f"Personalized feed: {len(feed)} articles")
        
    finally:
        await aggregator.close_all()


if __name__ == "__main__":
    asyncio.run(main())