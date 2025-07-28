#!/usr/bin/env python3
"""
설정 관리자
YAML 설정 파일 로드 및 관리
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .logger import get_logger

logger = get_logger("ConfigManager")

@dataclass
class TradingConfig:
    """거래 설정"""
    max_position_size: float = 0.1
    emergency_stop_loss: float = 0.05
    max_daily_trades: int = 10
    update_interval_seconds: int = 60
    lookback_periods: int = 100
    min_data_points: int = 50

@dataclass
class StrategyConfig:
    """전략 설정"""
    rule_strategies: list = field(default_factory=lambda: ["RuleStrategyA"])
    enable_ppo: bool = True
    hybrid_mode: str = "ensemble"
    execution_strategy: str = "market"
    risk_tolerance: str = "moderate"
    ppo_weight: float = 0.3
    min_confidence: float = 0.6

@dataclass
class RiskConfig:
    """리스크 설정"""
    max_drawdown: float = 0.15
    max_portfolio_risk: float = 0.02
    position_concentration_limit: float = 0.3
    correlation_threshold: float = 0.7
    var_confidence_level: float = 0.95

@dataclass
class NotificationConfig:
    """알림 설정"""
    enable_notifications: bool = True
    channels: list = field(default_factory=lambda: ["console", "file"])
    email_recipients: list = field(default_factory=list)
    slack_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # 추가 설정
    log_level: str = "INFO"
    data_path: str = "data"
    model_path: str = "models"
    results_path: str = "results"

class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.load_config()
    
    def load_config(self) -> AppConfig:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                
                self.config = self._dict_to_config(config_dict)
                logger.info(f"설정 파일 로드 완료: {self.config_path}")
            else:
                self.config = AppConfig()  # 기본 설정 사용
                self.save_config()  # 기본 설정 파일 생성
                logger.info("기본 설정으로 초기화됨")
        
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            self.config = AppConfig()  # 기본 설정으로 폴백
        
        return self.config
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            config_dict = self._config_to_dict(self.config)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else ".", exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"설정 파일 저장 완료: {self.config_path}")
        
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {e}")
    
    def get_config(self) -> AppConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.save_config()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """딕셔너리를 Config 객체로 변환"""
        trading_config = TradingConfig(**config_dict.get('trading', {}))
        strategy_config = StrategyConfig(**config_dict.get('strategy', {}))
        risk_config = RiskConfig(**config_dict.get('risk', {}))
        notification_config = NotificationConfig(**config_dict.get('notifications', {}))
        
        # 나머지 설정
        other_config = {k: v for k, v in config_dict.items() 
                       if k not in ['trading', 'strategy', 'risk', 'notifications']}
        
        return AppConfig(
            trading=trading_config,
            strategy=strategy_config,
            risk=risk_config,
            notifications=notification_config,
            **other_config
        )
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Config 객체를 딕셔너리로 변환"""
        return {
            'trading': {
                'max_position_size': config.trading.max_position_size,
                'emergency_stop_loss': config.trading.emergency_stop_loss,
                'max_daily_trades': config.trading.max_daily_trades,
                'update_interval_seconds': config.trading.update_interval_seconds,
                'lookback_periods': config.trading.lookback_periods,
                'min_data_points': config.trading.min_data_points
            },
            'strategy': {
                'rule_strategies': config.strategy.rule_strategies,
                'enable_ppo': config.strategy.enable_ppo,
                'hybrid_mode': config.strategy.hybrid_mode,
                'execution_strategy': config.strategy.execution_strategy,
                'risk_tolerance': config.strategy.risk_tolerance,
                'ppo_weight': config.strategy.ppo_weight,
                'min_confidence': config.strategy.min_confidence
            },
            'risk': {
                'max_drawdown': config.risk.max_drawdown,
                'max_portfolio_risk': config.risk.max_portfolio_risk,
                'position_concentration_limit': config.risk.position_concentration_limit,
                'correlation_threshold': config.risk.correlation_threshold,
                'var_confidence_level': config.risk.var_confidence_level
            },
            'notifications': {
                'enable_notifications': config.notifications.enable_notifications,
                'channels': config.notifications.channels,
                'email_recipients': config.notifications.email_recipients,
                'slack_webhook': config.notifications.slack_webhook,
                'telegram_bot_token': config.notifications.telegram_bot_token,
                'telegram_chat_id': config.notifications.telegram_chat_id
            },
            'log_level': config.log_level,
            'data_path': config.data_path,
            'model_path': config.model_path,
            'results_path': config.results_path
        }

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """설정 로드 헬퍼 함수"""
    manager = ConfigManager(config_path)
    return manager.get_config()