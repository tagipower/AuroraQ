#!/usr/bin/env python3
"""
VPS Deployment Environment Variable Loader
.env 파일 및 시스템 환경변수 통합 로딩 시스템
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

# python-dotenv가 있으면 사용, 없으면 기본 로딩
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

@dataclass
class EnvConfig:
    """환경변수 설정 클래스"""
    # 거래 설정
    trading_mode: str = "paper"
    symbol: str = "BTCUSDT" 
    exchange: str = "binance"
    
    # API 설정
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: bool = True
    
    # VPS 최적화 설정
    vps_memory_limit: str = "3G"
    max_daily_trades: int = 10
    max_position_size: float = 0.05
    emergency_stop_loss: float = 0.05
    
    # 포트 설정
    trading_api_port: int = 8004
    trading_websocket_port: int = 8003
    
    # 센티먼트 분석
    enable_sentiment_analysis: bool = False
    sentiment_service_url: str = "http://localhost:8000"
    sentiment_confidence_threshold: float = 0.6
    
    # 로깅 설정
    enable_unified_logging: bool = True
    log_level: str = "INFO"
    log_directory: str = "./logs"
    log_retention_days: int = 30
    
    # 알림 설정
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # 리스크 관리
    default_leverage: float = 3.0
    max_leverage: float = 10.0
    min_leverage: float = 1.0
    healthy_margin_ratio: float = 0.3
    warning_margin_ratio: float = 0.2
    danger_margin_ratio: float = 0.1
    critical_margin_ratio: float = 0.05
    liquidation_buffer: float = 0.02
    auto_add_margin: bool = True
    auto_reduce_position: bool = True
    
    # 데이터베이스
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl: int = 1800
    database_url: str = "sqlite:///vps_trading.db"
    
    # 모니터링
    health_check_interval: int = 60
    system_alert_cpu_threshold: int = 85
    system_alert_memory_threshold: int = 85
    service_timeout: int = 30
    api_timeout: int = 15
    
    # 보안
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20
    mask_sensitive_data: bool = True
    security_log_enabled: bool = True
    
    # PPO 전략
    enable_ppo_strategy: bool = False
    ppo_model_path: str = "./models/ppo_model.zip"
    ppo_confidence_threshold: float = 0.7
    ppo_update_interval: int = 300
    
    # 성능 최적화
    max_memory_usage_mb: int = 2048
    garbage_collection_interval: int = 3600
    max_cpu_usage_percent: int = 80
    thread_pool_size: int = 4
    connection_pool_size: int = 10
    request_timeout: int = 30
    max_retries: int = 3

class VPSEnvironmentLoader:
    """VPS 환경변수 로더"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        환경변수 로더 초기화
        
        Args:
            base_dir: 기본 디렉터리 (.env 파일 위치)
        """
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.logger = self._setup_logger()
        self.env_files = self._find_env_files()
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("vps_env_loader")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _find_env_files(self) -> list[Path]:
        """환경변수 파일 찾기"""
        env_files = []
        
        # 우선순위 순서로 .env 파일 찾기
        candidates = [
            self.base_dir / ".env.local",
            self.base_dir / ".env.production", 
            self.base_dir / ".env.development",
            self.base_dir / ".env",
            Path(__file__).parent.parent.parent / ".env"  # 메인 .env
        ]
        
        for candidate in candidates:
            if candidate.exists():
                env_files.append(candidate)
                self.logger.info(f"Found .env file: {candidate}")
        
        return env_files
    
    def _load_dotenv_file(self, env_file: Path) -> Dict[str, str]:
        """수동 .env 파일 파싱 (python-dotenv 없는 경우)"""
        env_vars = {}
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 빈 줄이나 주석 스킵
                    if not line or line.startswith('#'):
                        continue
                    
                    # KEY=VALUE 형식 파싱
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 따옴표 제거
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        env_vars[key] = value
                    else:
                        self.logger.warning(f"Invalid line in {env_file}:{line_num}: {line}")
        
        except Exception as e:
            self.logger.error(f"Error reading {env_file}: {e}")
        
        return env_vars
    
    def load_environment(self) -> EnvConfig:
        """환경변수 로딩"""
        # .env 파일들 로딩
        for env_file in self.env_files:
            if DOTENV_AVAILABLE:
                load_dotenv(env_file, override=False)  # 기존 값을 덮어쓰지 않음
                self.logger.info(f"Loaded {env_file} using python-dotenv")
            else:
                # 수동 로딩
                env_vars = self._load_dotenv_file(env_file)
                for key, value in env_vars.items():
                    if key not in os.environ:  # 기존 값 보존
                        os.environ[key] = value
                self.logger.info(f"Manually loaded {env_file}")
        
        # EnvConfig 생성
        config = EnvConfig()
        
        # 환경변수에서 값 읽어오기
        self._populate_config(config)
        
        # 보안 검증
        self._validate_config(config)
        
        return config
    
    def _populate_config(self, config: EnvConfig):
        """환경변수에서 설정 읽어오기"""
        
        # 거래 설정
        config.trading_mode = os.getenv('TRADING_MODE', config.trading_mode)
        config.symbol = os.getenv('SYMBOL', config.symbol)
        config.exchange = os.getenv('EXCHANGE', config.exchange)
        
        # API 설정
        config.binance_api_key = os.getenv('BINANCE_API_KEY')
        config.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        config.binance_testnet = self._get_bool('BINANCE_TESTNET', config.binance_testnet)
        
        # VPS 설정
        config.vps_memory_limit = os.getenv('VPS_MEMORY_LIMIT', config.vps_memory_limit)
        config.max_daily_trades = self._get_int('VPS_MAX_DAILY_TRADES', config.max_daily_trades)
        config.max_position_size = self._get_float('MAX_POSITION_SIZE', config.max_position_size)
        config.emergency_stop_loss = self._get_float('EMERGENCY_STOP_LOSS', config.emergency_stop_loss)
        
        # 포트 설정
        config.trading_api_port = self._get_int('TRADING_API_PORT', config.trading_api_port)
        config.trading_websocket_port = self._get_int('TRADING_WEBSOCKET_PORT', config.trading_websocket_port)
        
        # 센티먼트 분석
        config.enable_sentiment_analysis = self._get_bool('ENABLE_SENTIMENT_ANALYSIS', config.enable_sentiment_analysis)
        config.sentiment_service_url = os.getenv('SENTIMENT_SERVICE_URL', config.sentiment_service_url)
        config.sentiment_confidence_threshold = self._get_float('SENTIMENT_CONFIDENCE_THRESHOLD', config.sentiment_confidence_threshold)
        
        # 로깅
        config.enable_unified_logging = self._get_bool('ENABLE_UNIFIED_LOGGING', config.enable_unified_logging)
        config.log_level = os.getenv('LOG_LEVEL', config.log_level)
        config.log_directory = os.getenv('LOG_DIRECTORY', config.log_directory)
        config.log_retention_days = self._get_int('LOG_RETENTION_DAYS', config.log_retention_days)
        
        # 알림
        config.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # 리스크 관리
        config.default_leverage = self._get_float('DEFAULT_LEVERAGE', config.default_leverage)
        config.max_leverage = self._get_float('MAX_LEVERAGE', config.max_leverage)
        config.min_leverage = self._get_float('MIN_LEVERAGE', config.min_leverage)
        config.healthy_margin_ratio = self._get_float('HEALTHY_MARGIN_RATIO', config.healthy_margin_ratio)
        config.warning_margin_ratio = self._get_float('WARNING_MARGIN_RATIO', config.warning_margin_ratio)
        config.danger_margin_ratio = self._get_float('DANGER_MARGIN_RATIO', config.danger_margin_ratio)
        config.critical_margin_ratio = self._get_float('CRITICAL_MARGIN_RATIO', config.critical_margin_ratio)
        config.liquidation_buffer = self._get_float('LIQUIDATION_BUFFER', config.liquidation_buffer)
        config.auto_add_margin = self._get_bool('AUTO_ADD_MARGIN', config.auto_add_margin)
        config.auto_reduce_position = self._get_bool('AUTO_REDUCE_POSITION', config.auto_reduce_position)
        
        # 데이터베이스
        config.redis_url = os.getenv('REDIS_URL', config.redis_url)
        config.redis_ttl = self._get_int('REDIS_TTL', config.redis_ttl)
        config.database_url = os.getenv('DATABASE_URL', config.database_url)
        
        # 모니터링
        config.health_check_interval = self._get_int('HEALTH_CHECK_INTERVAL', config.health_check_interval)
        config.system_alert_cpu_threshold = self._get_int('SYSTEM_ALERT_CPU_THRESHOLD', config.system_alert_cpu_threshold)
        config.system_alert_memory_threshold = self._get_int('SYSTEM_ALERT_MEMORY_THRESHOLD', config.system_alert_memory_threshold)
        config.service_timeout = self._get_int('SERVICE_TIMEOUT', config.service_timeout)
        config.api_timeout = self._get_int('API_TIMEOUT', config.api_timeout)
        
        # 보안
        config.rate_limit_per_minute = self._get_int('RATE_LIMIT_PER_MINUTE', config.rate_limit_per_minute)
        config.rate_limit_burst = self._get_int('RATE_LIMIT_BURST', config.rate_limit_burst)
        config.mask_sensitive_data = self._get_bool('MASK_SENSITIVE_DATA', config.mask_sensitive_data)
        config.security_log_enabled = self._get_bool('SECURITY_LOG_ENABLED', config.security_log_enabled)
        
        # PPO 전략
        config.enable_ppo_strategy = self._get_bool('ENABLE_PPO_STRATEGY', config.enable_ppo_strategy)
        config.ppo_model_path = os.getenv('PPO_MODEL_PATH', config.ppo_model_path)
        config.ppo_confidence_threshold = self._get_float('PPO_CONFIDENCE_THRESHOLD', config.ppo_confidence_threshold)
        config.ppo_update_interval = self._get_int('PPO_UPDATE_INTERVAL', config.ppo_update_interval)
        
        # 성능 최적화
        config.max_memory_usage_mb = self._get_int('MAX_MEMORY_USAGE_MB', config.max_memory_usage_mb)
        config.garbage_collection_interval = self._get_int('GARBAGE_COLLECTION_INTERVAL', config.garbage_collection_interval)
        config.max_cpu_usage_percent = self._get_int('MAX_CPU_USAGE_PERCENT', config.max_cpu_usage_percent)
        config.thread_pool_size = self._get_int('THREAD_POOL_SIZE', config.thread_pool_size)
        config.connection_pool_size = self._get_int('CONNECTION_POOL_SIZE', config.connection_pool_size)
        config.request_timeout = self._get_int('REQUEST_TIMEOUT', config.request_timeout)
        config.max_retries = self._get_int('MAX_RETRIES', config.max_retries)
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """환경변수에서 boolean 값 읽기"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def _get_int(self, key: str, default: int) -> int:
        """환경변수에서 정수 값 읽기"""
        try:
            value = os.getenv(key)
            return int(value) if value is not None else default
        except ValueError:
            self.logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float(self, key: str, default: float) -> float:
        """환경변수에서 실수 값 읽기"""
        try:
            value = os.getenv(key)
            return float(value) if value is not None else default
        except ValueError:
            self.logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def _validate_config(self, config: EnvConfig):
        """설정 검증"""
        warnings = []
        errors = []
        
        # 중요한 설정들 검증
        if config.trading_mode == "live":
            if not config.binance_api_key or not config.binance_api_secret:
                errors.append("Live trading mode requires BINANCE_API_KEY and BINANCE_API_SECRET")
            
            if config.binance_testnet:
                warnings.append("Live trading mode with testnet=true - this might be unintended")
        
        # 리스크 관리 파라미터 검증
        if config.max_leverage > 20:
            warnings.append(f"High leverage detected: {config.max_leverage}x - ensure this is intended")
        
        if config.max_position_size > 1.0:
            warnings.append(f"Large position size: {config.max_position_size} - ensure this is intended")
        
        if config.emergency_stop_loss > 0.2:
            warnings.append(f"Large stop loss: {config.emergency_stop_loss*100}% - ensure this is intended")
        
        # 마진 비율 검증
        margin_ratios = [
            config.healthy_margin_ratio,
            config.warning_margin_ratio, 
            config.danger_margin_ratio,
            config.critical_margin_ratio
        ]
        
        if not all(margin_ratios[i] > margin_ratios[i+1] for i in range(len(margin_ratios)-1)):
            errors.append("Margin ratios must be in descending order (healthy > warning > danger > critical)")
        
        # 로그 출력
        for warning in warnings:
            self.logger.warning(f"Configuration warning: {warning}")
        
        for error in errors:
            self.logger.error(f"Configuration error: {error}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

# 전역 환경변수 로더 인스턴스
_global_loader = None
_global_config = None

def get_vps_env_config() -> EnvConfig:
    """VPS 환경변수 설정 가져오기 (싱글톤)"""
    global _global_loader, _global_config
    
    if _global_config is None:
        _global_loader = VPSEnvironmentLoader()
        _global_config = _global_loader.load_environment()
    
    return _global_config

def reload_vps_env_config() -> EnvConfig:
    """VPS 환경변수 설정 재로딩"""
    global _global_loader, _global_config
    
    _global_loader = VPSEnvironmentLoader()
    _global_config = _global_loader.load_environment()
    
    return _global_config

if __name__ == "__main__":
    # 테스트 실행
    loader = VPSEnvironmentLoader()
    config = loader.load_environment()
    
    print("=== VPS Environment Configuration ===")
    print(f"Trading Mode: {config.trading_mode}")
    print(f"Symbol: {config.symbol}")
    print(f"Exchange: {config.exchange}")
    print(f"Testnet: {config.binance_testnet}")
    print(f"API Key Set: {'Yes' if config.binance_api_key else 'No'}")
    print(f"Memory Limit: {config.vps_memory_limit}")
    print(f"Max Daily Trades: {config.max_daily_trades}")
    print(f"Max Position Size: {config.max_position_size}")
    print(f"Default Leverage: {config.default_leverage}x")
    print(f"Sentiment Analysis: {config.enable_sentiment_analysis}")
    print(f"Unified Logging: {config.enable_unified_logging}")
    print(f"PPO Strategy: {config.enable_ppo_strategy}")
    print("=" * 40)