#!/usr/bin/env python3
"""
바이낸스 API 설정 관리 모듈
테스트넷/실전 모드 동적 전환 지원
"""

import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class TradingMode(Enum):
    """거래 모드 열거형"""
    TESTNET = "testnet"
    MAINNET = "mainnet"
    PAPER = "paper"  # 모의 거래

@dataclass
class BinanceCredentials:
    """바이낸스 API 인증 정보"""
    api_key: str
    api_secret: str
    sandbox: bool
    base_url: Optional[str] = None

class BinanceConfigManager:
    """바이낸스 설정 관리자"""
    
    def __init__(self):
        self.current_mode = self._get_default_mode()
        self._credentials_cache = {}
    
    def _get_default_mode(self) -> TradingMode:
        """기본 거래 모드 결정"""
        testnet_enabled = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        if testnet_enabled:
            return TradingMode.TESTNET
        else:
            return TradingMode.MAINNET
    
    def get_credentials(self, mode: Optional[TradingMode] = None) -> BinanceCredentials:
        """거래 모드에 따른 API 인증 정보 반환"""
        target_mode = mode or self.current_mode
        
        # 캐시된 인증 정보 확인
        if target_mode in self._credentials_cache:
            return self._credentials_cache[target_mode]
        
        if target_mode == TradingMode.TESTNET:
            credentials = self._get_testnet_credentials()
        elif target_mode == TradingMode.MAINNET:
            credentials = self._get_mainnet_credentials()
        elif target_mode == TradingMode.PAPER:
            credentials = self._get_paper_credentials()
        else:
            raise ValueError(f"지원되지 않는 거래 모드: {target_mode}")
        
        # 캐시에 저장
        self._credentials_cache[target_mode] = credentials
        return credentials
    
    def _get_testnet_credentials(self) -> BinanceCredentials:
        """테스트넷 인증 정보"""
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("테스트넷 API 키가 설정되지 않았습니다")
        
        return BinanceCredentials(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=True,
            base_url="https://testnet.binance.vision"
        )
    
    def _get_mainnet_credentials(self) -> BinanceCredentials:
        """메인넷 인증 정보"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("메인넷 API 키가 설정되지 않았습니다")
        
        return BinanceCredentials(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=False,
            base_url="https://api.binance.com"
        )
    
    def _get_paper_credentials(self) -> BinanceCredentials:
        """모의 거래 인증 정보 (테스트넷 기반)"""
        return self._get_testnet_credentials()
    
    def switch_mode(self, mode: TradingMode) -> None:
        """거래 모드 전환"""
        old_mode = self.current_mode
        self.current_mode = mode
        
        print(f"🔄 거래 모드 변경: {old_mode.value} → {mode.value}")
        
        # 새 모드 인증 정보 검증
        try:
            credentials = self.get_credentials(mode)
            print(f"✅ {mode.value} 모드 인증 정보 확인됨")
        except Exception as e:
            print(f"❌ {mode.value} 모드 인증 실패: {e}")
            # 이전 모드로 롤백
            self.current_mode = old_mode
            raise
    
    def get_ccxt_config(self, mode: Optional[TradingMode] = None) -> Dict[str, Any]:
        """CCXT 라이브러리용 설정 반환"""
        credentials = self.get_credentials(mode)
        
        config = {
            'apiKey': credentials.api_key,
            'secret': credentials.api_secret,
            'sandbox': credentials.sandbox,
            'enableRateLimit': True,
            'timeout': 30000,  # 30초
        }
        
        if credentials.base_url:
            config['urls'] = {'api': credentials.base_url}
        
        return config
    
    def get_trading_limits(self, mode: Optional[TradingMode] = None) -> Dict[str, Any]:
        """거래 모드별 제한사항 반환"""
        target_mode = mode or self.current_mode
        
        if target_mode == TradingMode.TESTNET:
            return {
                'max_order_value_usdt': 1000,  # 최대 주문 금액
                'max_daily_trades': 100,        # 일일 최대 거래 횟수
                'allowed_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                'risk_level': 'low'
            }
        elif target_mode == TradingMode.MAINNET:
            return {
                'max_order_value_usdt': 10000,
                'max_daily_trades': 50,
                'allowed_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'],
                'risk_level': 'medium'
            }
        elif target_mode == TradingMode.PAPER:
            return {
                'max_order_value_usdt': 10000,  # 모의 거래는 제한 없음
                'max_daily_trades': 1000,
                'allowed_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                'risk_level': 'test'
            }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """현재 설정 유효성 검사"""
        results = {
            'current_mode': self.current_mode.value,
            'testnet_available': False,
            'mainnet_available': False,
            'paper_available': False,
            'errors': []
        }
        
        # 각 모드별 인증 정보 확인
        for mode in TradingMode:
            try:
                credentials = self.get_credentials(mode)
                if mode == TradingMode.TESTNET:
                    results['testnet_available'] = True
                elif mode == TradingMode.MAINNET:
                    results['mainnet_available'] = True
                elif mode == TradingMode.PAPER:
                    results['paper_available'] = True
            except Exception as e:
                results['errors'].append(f"{mode.value}: {str(e)}")
        
        return results
    
    def get_mode_info(self) -> Dict[str, Any]:
        """현재 모드 정보 반환"""
        credentials = self.get_credentials()
        limits = self.get_trading_limits()
        
        return {
            'mode': self.current_mode.value,
            'sandbox': credentials.sandbox,
            'base_url': credentials.base_url,
            'api_key_length': len(credentials.api_key),
            'limits': limits,
            'description': self._get_mode_description()
        }
    
    def _get_mode_description(self) -> str:
        """모드별 설명 반환"""
        descriptions = {
            TradingMode.TESTNET: "테스트넷 - 가상 자금으로 실제 API 테스트",
            TradingMode.MAINNET: "메인넷 - 실제 자금을 사용한 실전 거래",
            TradingMode.PAPER: "모의 거래 - 실제 주문 없이 거래 시뮬레이션"
        }
        return descriptions.get(self.current_mode, "알 수 없는 모드")

# 전역 설정 관리자 인스턴스
binance_config = BinanceConfigManager()

def get_binance_config() -> BinanceConfigManager:
    """전역 바이낸스 설정 관리자 반환"""
    return binance_config

def set_trading_mode(mode: TradingMode) -> None:
    """전역 거래 모드 설정"""
    binance_config.switch_mode(mode)

def get_current_mode() -> TradingMode:
    """현재 거래 모드 반환"""
    return binance_config.current_mode

# 편의 함수들
def is_testnet() -> bool:
    """테스트넷 모드 여부 확인"""
    return binance_config.current_mode == TradingMode.TESTNET

def is_mainnet() -> bool:
    """메인넷 모드 여부 확인"""
    return binance_config.current_mode == TradingMode.MAINNET

def is_paper_trading() -> bool:
    """모의 거래 모드 여부 확인"""
    return binance_config.current_mode == TradingMode.PAPER

if __name__ == "__main__":
    # 설정 테스트
    print("🧪 바이낸스 설정 관리자 테스트")
    print("=" * 50)
    
    config_manager = BinanceConfigManager()
    
    # 현재 모드 정보
    mode_info = config_manager.get_mode_info()
    print(f"현재 모드: {mode_info['mode']}")
    print(f"설명: {mode_info['description']}")
    print(f"샌드박스: {mode_info['sandbox']}")
    print(f"API 키 길이: {mode_info['api_key_length']}")
    
    # 설정 유효성 검사
    validation = config_manager.validate_configuration()
    print(f"\n설정 유효성 검사:")
    print(f"테스트넷 사용 가능: {validation['testnet_available']}")
    print(f"메인넷 사용 가능: {validation['mainnet_available']}")
    print(f"모의 거래 사용 가능: {validation['paper_available']}")
    
    if validation['errors']:
        print("오류:")
        for error in validation['errors']:
            print(f"  - {error}")