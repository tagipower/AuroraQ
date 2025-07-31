# config/settings.py
"""Sentiment Service Configuration Settings"""

import os
from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Service Configuration
    app_name: str = "AuroraQ Sentiment Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    grpc_port: int = Field(default=50051, env="GRPC_PORT")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    
    # FinBERT Model Configuration
    finbert_model_name: str = Field(default="ProsusAI/finbert", env="FINBERT_MODEL_NAME")
    finbert_model_path: str = Field(default="/app/models/finbert", env="FINBERT_MODEL_PATH") 
    finbert_cache_dir: str = Field(default="/app/cache/transformers", env="FINBERT_CACHE_DIR")
    finbert_max_length: int = Field(default=512, env="FINBERT_MAX_LENGTH")
    finbert_batch_size: int = Field(default=16, env="FINBERT_BATCH_SIZE")
    
    # Performance Configuration
    enable_model_caching: bool = Field(default=True, env="ENABLE_MODEL_CACHING")
    model_warmup: bool = Field(default=True, env="MODEL_WARMUP")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: float = Field(default=30.0, env="REQUEST_TIMEOUT")
    
    # Fusion Configuration
    fusion_source_weights: Dict[str, float] = {
        "news": 0.4,
        "social": 0.3,
        "technical": 0.2,
        "historical": 0.1
    }
    fusion_outlier_threshold: float = Field(default=3.0, env="FUSION_OUTLIER_THRESHOLD")
    fusion_confidence_threshold: float = Field(default=0.6, env="FUSION_CONFIDENCE_THRESHOLD")
    
    # Security Configuration
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_hosts: str = Field(default="*", env="ALLOWED_HOSTS")
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    prometheus_port: int = Field(default=8080, env="PROMETHEUS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Logging Configuration  
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json, text
    log_file_path: str = Field(default="/app/logs/sentiment-service.log", env="LOG_FILE_PATH")
    log_rotation: str = Field(default="1 day", env="LOG_ROTATION")
    log_retention: str = Field(default="30 days", env="LOG_RETENTION")
    
    # External API Keys
    google_news_api_key: str = Field(default="", env="GOOGLE_NEWS_API_KEY")
    yahoo_finance_api_key: str = Field(default="", env="YAHOO_FINANCE_API_KEY")
    newsapi_key: str = Field(default="", env="NEWSAPI_KEY")
    finnhub_api_key: str = Field(default="", env="FINNHUB_API_KEY")
    reddit_client_id: str = Field(default="", env="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    google_search_api_key: str = Field(default="", env="GOOGLE_SEARCH_API_KEY")
    google_custom_search_id: str = Field(default="", env="GOOGLE_CUSTOM_SEARCH_ID")
    bing_search_api_key: str = Field(default="", env="BING_SEARCH_API_KEY")
    
    # AuroraQ Integration
    aurora_api_url: str = Field(default="http://localhost:8080", env="AURORA_API_URL")
    aurora_api_key: str = Field(default="", env="AURORA_API_KEY")
    aurora_timeout: int = Field(default=30, env="AURORA_TIMEOUT")
    aurora_retry_attempts: int = Field(default=3, env="AURORA_RETRY_ATTEMPTS")
    
    # Telegram Notifications
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id_general: str = Field(default="", env="TELEGRAM_CHAT_ID_GENERAL")
    telegram_chat_id_trading: str = Field(default="", env="TELEGRAM_CHAT_ID_TRADING")
    telegram_chat_id_events: str = Field(default="", env="TELEGRAM_CHAT_ID_EVENTS")
    telegram_chat_id_system: str = Field(default="", env="TELEGRAM_CHAT_ID_SYSTEM")
    telegram_enabled: bool = Field(default=True, env="TELEGRAM_ENABLED")
    telegram_quiet_hours_start: int = Field(default=23, env="TELEGRAM_QUIET_HOURS_START")
    telegram_quiet_hours_end: int = Field(default=7, env="TELEGRAM_QUIET_HOURS_END")
    
    # Feedly Integration (레거시, 백업용)
    feedly_access_token: str = Field(default="", env="FEEDLY_ACCESS_TOKEN")
    feedly_user_id: str = Field(default="", env="FEEDLY_USER_ID")
    feedly_rate_limit: int = Field(default=100, env="FEEDLY_RATE_LIMIT")  # requests per hour
    
    # News Processing
    news_max_articles: int = Field(default=50, env="NEWS_MAX_ARTICLES")
    news_hours_back: int = Field(default=24, env="NEWS_HOURS_BACK")
    news_relevance_threshold: float = Field(default=0.3, env="NEWS_RELEVANCE_THRESHOLD")
    
    # Sentiment History
    sentiment_history_path: str = Field(default="/app/data/sentiment_history.csv", env="SENTIMENT_HISTORY_PATH")
    sentiment_history_size: int = Field(default=10000, env="SENTIMENT_HISTORY_SIZE")
    
    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ('settings_',)
    }


class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    enable_model_caching: bool = False  # Disable for development
    model_warmup: bool = False


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    enable_model_caching: bool = True
    model_warmup: bool = True
    
    # Production Redis with authentication
    redis_url: str = Field(env="REDIS_URL")
    
    # Stricter security
    allowed_hosts: str = Field(env="ALLOWED_HOSTS")
    cors_origins: str = Field(env="CORS_ORIGINS")


class TestSettings(Settings):
    """Test environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    redis_url: str = "redis://localhost:6379/1"  # Use different DB for tests
    enable_model_caching: bool = False
    model_warmup: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()


# Asset keyword mappings for news filtering
ASSET_KEYWORDS = {
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

# Sentiment label mappings
SENTIMENT_LABELS = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# Model configuration
MODEL_CONFIG = {
    "finbert": {
        "model_name": settings.finbert_model_name,
        "cache_dir": settings.finbert_cache_dir,
        "max_length": settings.finbert_max_length,
        "batch_size": settings.finbert_batch_size,
        "device": "cpu"  # Default to CPU, can be overridden
    }
}