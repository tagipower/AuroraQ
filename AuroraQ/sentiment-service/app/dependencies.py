#!/usr/bin/env python3
"""
Dependencies for AuroraQ Sentiment Service
의존성 주입 및 싱글톤 관리
"""

import asyncio
from typing import Optional
from functools import lru_cache

# 조건부 import 로 경로 문제 해결
try:
    from ..models.keyword_scorer import KeywordScorer
    from ..models.advanced_keyword_scorer import AdvancedKeywordScorer
except ImportError:
    try:
        from models.keyword_scorer import KeywordScorer
        from models.advanced_keyword_scorer import AdvancedKeywordScorer
    except ImportError:
        # Mock classes for testing
        KeywordScorer = None
        AdvancedKeywordScorer = None

try:
    from ..processors.sentiment_fusion_manager import SentimentFusionManager
    from ..processors.advanced_fusion_manager import AdvancedFusionManager
except ImportError:
    try:
        from processors.sentiment_fusion_manager import SentimentFusionManager
        from processors.advanced_fusion_manager import AdvancedFusionManager
    except ImportError:
        SentimentFusionManager = None
        AdvancedFusionManager = None

try:
    from ..config.settings import settings
except ImportError:
    try:
        from config.settings import settings
    except ImportError:
        # Mock settings
        class MockSettings:
            redis_url = "redis://localhost:6379"
            cache_ttl = 300
        settings = MockSettings()

# Mock화 나머지 의존성들
FinBERTBatchProcessor = None
BigEventDetector = None
EnhancedNewsCollector = None
RedditCollector = None
PowerSearchEngine = None
ContentCacheManager = None
DataQualityValidator = None
AuroraQAdapter = None
BatchScheduler = None
NotificationManager = None
NotificationConfig = None

# Global instances (싱글톤)
_keyword_scorer: Optional[KeywordScorer] = None
_advanced_keyword_scorer: Optional[AdvancedKeywordScorer] = None
_finbert_processor: Optional[FinBERTBatchProcessor] = None
_fusion_manager: Optional[SentimentFusionManager] = None
_advanced_fusion_manager: Optional[AdvancedFusionManager] = None
_big_event_detector: Optional[BigEventDetector] = None
_cache_manager: Optional[ContentCacheManager] = None
_data_validator: Optional[DataQualityValidator] = None
_news_collector: Optional[EnhancedNewsCollector] = None
_reddit_collector: Optional[RedditCollector] = None
_power_search: Optional[PowerSearchEngine] = None
_aurora_adapter: Optional[AuroraQAdapter] = None
_batch_scheduler: Optional[BatchScheduler] = None
_notification_manager: Optional[NotificationManager] = None

@lru_cache()
def get_keyword_scorer():
    """키워드 스코어러 싱글톤"""
    global _keyword_scorer
    if _keyword_scorer is None and KeywordScorer:
        _keyword_scorer = KeywordScorer()
    return _keyword_scorer

@lru_cache()
def get_advanced_keyword_scorer():
    """고급 키워드 스코어러 싱글톤"""
    global _advanced_keyword_scorer
    if _advanced_keyword_scorer is None and AdvancedKeywordScorer:
        _advanced_keyword_scorer = AdvancedKeywordScorer()
    return _advanced_keyword_scorer

async def get_cache_manager():
    """캐시 매니저 싱글톤"""
    global _cache_manager
    if _cache_manager is None and ContentCacheManager:
        _cache_manager = ContentCacheManager(redis_url=settings.redis_url)
        # 비동기 컨텍스트 매니저 진입
        try:
            await _cache_manager.__aenter__()
        except Exception:
            # 캐시 연결 실패 시 None 반환
            _cache_manager = None
    return _cache_manager

async def get_data_validator() -> DataQualityValidator:
    """데이터 품질 검증기 싱글톤"""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataQualityValidator()
    return _data_validator

async def get_finbert_processor() -> FinBERTBatchProcessor:
    """FinBERT 배치 프로세서 싱글톤"""
    global _finbert_processor
    if _finbert_processor is None:
        cache_manager = await get_cache_manager()
        _finbert_processor = FinBERTBatchProcessor(
            model_name="ProsusAI/finbert",
            cache_manager=cache_manager,
            max_batch_size=16,
            max_sequence_length=512
        )
        # 비동기 컨텍스트 매니저 진입
        await _finbert_processor.__aenter__()
    return _finbert_processor

async def get_fusion_manager() -> SentimentFusionManager:
    """감정 융합 매니저 싱글톤"""
    global _fusion_manager
    if _fusion_manager is None:
        keyword_scorer = get_keyword_scorer()
        finbert_processor = await get_finbert_processor()
        cache_manager = await get_cache_manager()
        
        _fusion_manager = SentimentFusionManager(
            keyword_scorer=keyword_scorer,
            finbert_processor=finbert_processor,
            cache_manager=cache_manager
        )
    return _fusion_manager

async def get_advanced_fusion_manager():
    """고급 융합 매니저 싱글톤"""
    global _advanced_fusion_manager
    if _advanced_fusion_manager is None and AdvancedFusionManager:
        advanced_keyword_scorer = get_advanced_keyword_scorer()
        cache_manager = await get_cache_manager()
        
        if advanced_keyword_scorer:
            _advanced_fusion_manager = AdvancedFusionManager(
                advanced_scorer=advanced_keyword_scorer,
                cache_manager=cache_manager
            )
    return _advanced_fusion_manager

async def get_big_event_detector() -> BigEventDetector:
    """빅 이벤트 감지기 싱글톤"""
    global _big_event_detector
    if _big_event_detector is None:
        data_validator = await get_data_validator()
        cache_manager = await get_cache_manager()
        
        _big_event_detector = BigEventDetector(
            data_validator=data_validator,
            cache_manager=cache_manager
        )
    return _big_event_detector

async def get_enhanced_news_collector() -> EnhancedNewsCollector:
    """향상된 뉴스 수집기 싱글톤"""
    global _news_collector
    if _news_collector is None:
        # API 키들 설정
        api_keys = {
            "google_news_key": settings.google_news_api_key,
            "yahoo_finance_key": settings.yahoo_finance_api_key,
            "newsapi_key": settings.newsapi_key,
            "finnhub_key": settings.finnhub_api_key
        }
        
        _news_collector = EnhancedNewsCollector(api_keys=api_keys)
        # 비동기 컨텍스트 매니저 진입
        await _news_collector.__aenter__()
    return _news_collector

async def get_reddit_collector() -> RedditCollector:
    """Reddit 수집기 싱글톤"""
    global _reddit_collector
    if _reddit_collector is None:
        reddit_config = {
            "client_id": settings.reddit_client_id,
            "client_secret": settings.reddit_client_secret,
            "user_agent": "AuroraQ-RedditCollector/1.0"
        }
        
        _reddit_collector = RedditCollector(reddit_config=reddit_config)
        # 비동기 컨텍스트 매니저 진입
        await _reddit_collector.__aenter__()
    return _reddit_collector

async def get_power_search_engine() -> PowerSearchEngine:
    """파워 서치 엔진 싱글톤"""
    global _power_search
    if _power_search is None:
        api_keys = {
            "google_search_key": settings.google_search_api_key,
            "google_cx": settings.google_custom_search_id,
            "bing_search_key": settings.bing_search_api_key
        }
        
        _power_search = PowerSearchEngine(api_keys=api_keys)
        # 비동기 컨텍스트 매니저 진입
        await _power_search.__aenter__()
    return _power_search

async def get_aurora_adapter() -> AuroraQAdapter:
    """AuroraQ 어댑터 싱글톤"""
    global _aurora_adapter
    if _aurora_adapter is None:
        # 필요한 컴포넌트들 가져오기
        fusion_manager = await get_fusion_manager()
        event_detector = await get_big_event_detector()
        keyword_scorer = get_keyword_scorer()
        cache_manager = await get_cache_manager()
        
        _aurora_adapter = AuroraQAdapter(
            fusion_manager=fusion_manager,
            event_detector=event_detector,
            keyword_scorer=keyword_scorer,
            cache_manager=cache_manager,
            aurora_api_url=getattr(settings, 'aurora_api_url', 'http://localhost:8080'),
            aurora_api_key=getattr(settings, 'aurora_api_key', '')
        )
        # 비동기 컨텍스트 매니저 진입
        await _aurora_adapter.__aenter__()
    return _aurora_adapter

async def get_batch_scheduler() -> BatchScheduler:
    """배치 스케줄러 싱글톤"""
    global _batch_scheduler
    if _batch_scheduler is None:
        # 필요한 컴포넌트들 가져오기
        finbert_processor = await get_finbert_processor()
        fusion_manager = await get_fusion_manager()
        event_detector = await get_big_event_detector()
        news_collector = await get_enhanced_news_collector()
        aurora_adapter = await get_aurora_adapter()
        cache_manager = await get_cache_manager()
        
        # 알림 통합 가져오기
        notification_manager = await get_notification_manager()
        notification_integration = None
        if notification_manager:
            from ..integrations.notification_integration import get_notification_integration
            notification_integration = get_notification_integration(notification_manager)
        
        _batch_scheduler = BatchScheduler(
            finbert_processor=finbert_processor,
            fusion_manager=fusion_manager,
            event_detector=event_detector,
            news_collector=news_collector,
            aurora_adapter=aurora_adapter,
            cache_manager=cache_manager,
            notification_integration=notification_integration
        )
        # 스케줄러 시작은 별도로 관리 (startup에서)
    return _batch_scheduler

async def get_notification_manager() -> NotificationManager:
    """알림 관리자 싱글톤"""
    global _notification_manager
    if _notification_manager is None:
        # 텔레그램 설정이 있는 경우에만 활성화
        if settings.telegram_bot_token and settings.telegram_enabled:
            # 채팅방 설정
            chat_configs = {}
            if settings.telegram_chat_id_general:
                chat_configs["general"] = {"chat_id": settings.telegram_chat_id_general}
            if settings.telegram_chat_id_trading:
                chat_configs["trading"] = {"chat_id": settings.telegram_chat_id_trading}
            if settings.telegram_chat_id_events:
                chat_configs["events"] = {"chat_id": settings.telegram_chat_id_events}
            if settings.telegram_chat_id_system:
                chat_configs["system"] = {"chat_id": settings.telegram_chat_id_system}
            
            # 알림 설정
            notification_configs = {
                "trading": NotificationConfig(
                    enabled=True,
                    channels=["trading"],
                    min_level="info",
                    rate_limit_minutes=1
                ),
                "events": NotificationConfig(
                    enabled=True,
                    channels=["events"],
                    min_level="info",
                    rate_limit_minutes=5
                ),
                "system": NotificationConfig(
                    enabled=True,
                    channels=["system"],
                    min_level="warning",
                    rate_limit_minutes=10,
                    quiet_hours={
                        "start": settings.telegram_quiet_hours_start,
                        "end": settings.telegram_quiet_hours_end
                    }
                ),
                "performance": NotificationConfig(
                    enabled=True,
                    channels=["system"],
                    min_level="info",
                    rate_limit_minutes=60
                )
            }
            
            _notification_manager = NotificationManager(
                telegram_bot_token=settings.telegram_bot_token,
                telegram_chat_configs=chat_configs,
                notification_configs=notification_configs
            )
            
            # 비동기 컨텍스트 매니저 진입
            await _notification_manager.__aenter__()
        else:
            # 더미 알림 관리자 (알림 비활성화)
            _notification_manager = None
    
    return _notification_manager

# 레거시 함수들 (기존 코드 호환성)
async def get_sentiment_analyzer():
    """레거시: 감정 분석기 (키워드 스코어러 반환)"""
    return get_keyword_scorer()

# 정리 함수들
async def cleanup_dependencies():
    """모든 의존성 정리"""
    global _finbert_processor, _fusion_manager, _advanced_fusion_manager, _cache_manager
    global _news_collector, _reddit_collector, _power_search, _aurora_adapter, _batch_scheduler, _notification_manager
    
    # 비동기 컨텍스트 매니저들 정리
    if _finbert_processor:
        await _finbert_processor.__aexit__(None, None, None)
        _finbert_processor = None
    
    if _cache_manager:
        await _cache_manager.__aexit__(None, None, None)
        _cache_manager = None
    
    if _news_collector:
        await _news_collector.__aexit__(None, None, None)
        _news_collector = None
    
    if _reddit_collector:
        await _reddit_collector.__aexit__(None, None, None)
        _reddit_collector = None
    
    if _power_search:
        await _power_search.__aexit__(None, None, None)
        _power_search = None
    
    if _aurora_adapter:
        await _aurora_adapter.__aexit__(None, None, None)
        _aurora_adapter = None
    
    if _batch_scheduler:
        await _batch_scheduler.stop()
        _batch_scheduler = None
    
    if _notification_manager:
        await _notification_manager.__aexit__(None, None, None)
        _notification_manager = None

async def health_check_dependencies() -> dict:
    """의존성 상태 확인"""
    status = {}
    
    try:
        # 키워드 스코어러 확인
        keyword_scorer = get_keyword_scorer()
        test_result = keyword_scorer.analyze("test")
        status["keyword_scorer"] = "healthy" if test_result else "unhealthy"
    except Exception as e:
        status["keyword_scorer"] = f"error: {str(e)}"
    
    try:
        # 캐시 매니저 확인
        cache_manager = await get_cache_manager()
        status["cache_manager"] = "healthy" if cache_manager.redis else "unhealthy"
    except Exception as e:
        status["cache_manager"] = f"error: {str(e)}"
    
    try:
        # FinBERT 프로세서 확인
        finbert_processor = await get_finbert_processor()
        processor_stats = finbert_processor.get_processor_stats()
        status["finbert_processor"] = "healthy" if processor_stats["model_loaded"] else "loading"
    except Exception as e:
        status["finbert_processor"] = f"error: {str(e)}"
    
    try:
        # 융합 매니저 확인
        fusion_manager = await get_fusion_manager()
        fusion_stats = fusion_manager.get_fusion_stats()
        status["fusion_manager"] = "healthy" if fusion_stats else "unhealthy"
    except Exception as e:
        status["fusion_manager"] = f"error: {str(e)}"
    
    try:
        # 이벤트 감지기 확인
        event_detector = await get_big_event_detector()
        detector_stats = event_detector.get_detector_stats()
        status["big_event_detector"] = "healthy" if detector_stats else "unhealthy"
    except Exception as e:
        status["big_event_detector"] = f"error: {str(e)}"
    
    try:
        # AuroraQ 어댑터 확인
        aurora_adapter = await get_aurora_adapter()
        adapter_health = await aurora_adapter.health_check()
        status["aurora_adapter"] = adapter_health.get("adapter_status", "unknown")
    except Exception as e:
        status["aurora_adapter"] = f"error: {str(e)}"
    
    try:
        # 배치 스케줄러 확인
        batch_scheduler = await get_batch_scheduler()
        scheduler_stats = batch_scheduler.get_scheduler_stats()
        status["batch_scheduler"] = "healthy" if scheduler_stats.get("scheduled_tasks_count", 0) > 0 else "no_tasks"
    except Exception as e:
        status["batch_scheduler"] = f"error: {str(e)}"
    
    try:
        # 알림 관리자 확인
        notification_manager = await get_notification_manager()
        if notification_manager:
            notif_stats = notification_manager.get_stats()
            status["notification_manager"] = "healthy" if notif_stats.get("telegram_stats", {}).get("running", False) else "disabled"
        else:
            status["notification_manager"] = "disabled"
    except Exception as e:
        status["notification_manager"] = f"error: {str(e)}"
    
    return status