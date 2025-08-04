#!/usr/bin/env python3
"""
VPS 배포 환경변수 검증 시스템
실전 거래 시 필수 환경변수 체크 및 보안 검증
"""

import os
import re
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """거래 모드"""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"

class ValidationLevel(Enum):
    """검증 레벨"""
    REQUIRED = "required"        # 필수 (없으면 실행 불가)
    RECOMMENDED = "recommended"  # 권장 (경고만)
    OPTIONAL = "optional"        # 선택 (정보만)

@dataclass
class EnvVarSpec:
    """환경변수 명세"""
    name: str
    description: str
    required_for_modes: List[TradingMode]
    validation_level: ValidationLevel
    pattern: Optional[str] = None  # 정규식 패턴
    min_length: Optional[int] = None
    is_sensitive: bool = False  # 민감한 정보 (로그에 마스킹)
    default_value: Optional[str] = None
    example: Optional[str] = None

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    missing_required: List[str]
    missing_recommended: List[str]
    invalid_format: List[str]
    security_issues: List[str]
    warnings: List[str]
    config_summary: Dict[str, Any]

class VPSEnvironmentValidator:
    """VPS 환경변수 검증기"""
    
    def __init__(self):
        self.env_specs = self._init_env_specs()
        
    def _init_env_specs(self) -> Dict[str, EnvVarSpec]:
        """환경변수 명세 초기화"""
        specs = {}
        
        # Binance API (실전 거래 필수)
        specs['BINANCE_API_KEY'] = EnvVarSpec(
            name='BINANCE_API_KEY',
            description='바이낸스 API 키',
            required_for_modes=[TradingMode.LIVE],
            validation_level=ValidationLevel.REQUIRED,
            pattern=r'^[A-Za-z0-9]{64}$',
            min_length=64,
            is_sensitive=True,
            example='abcd1234....(64자)'
        )
        
        specs['BINANCE_API_SECRET'] = EnvVarSpec(
            name='BINANCE_API_SECRET',
            description='바이낸스 API 시크릿',
            required_for_modes=[TradingMode.LIVE],
            validation_level=ValidationLevel.REQUIRED,
            pattern=r'^[A-Za-z0-9]{64}$',
            min_length=64,
            is_sensitive=True,
            example='xyz9876....(64자)'
        )
        
        # 바이낸스 테스트넷 (개발용)
        specs['BINANCE_TESTNET_API_KEY'] = EnvVarSpec(
            name='BINANCE_TESTNET_API_KEY',
            description='바이낸스 테스트넷 API 키',
            required_for_modes=[TradingMode.PAPER],
            validation_level=ValidationLevel.RECOMMENDED,
            is_sensitive=True
        )
        
        specs['BINANCE_TESTNET_API_SECRET'] = EnvVarSpec(
            name='BINANCE_TESTNET_API_SECRET',
            description='바이낸스 테스트넷 API 시크릿',
            required_for_modes=[TradingMode.PAPER],
            validation_level=ValidationLevel.RECOMMENDED,
            is_sensitive=True
        )
        
        # PPO 모델 설정
        specs['PPO_MODEL_PATH'] = EnvVarSpec(
            name='PPO_MODEL_PATH',
            description='PPO 모델 파일 경로',
            required_for_modes=[TradingMode.LIVE, TradingMode.PAPER],
            validation_level=ValidationLevel.RECOMMENDED,
            default_value='/app/models/ppo_model.zip',
            example='/app/models/ppo_model.zip'
        )
        
        # 감정 분석 서비스
        specs['SENTIMENT_SERVICE_URL'] = EnvVarSpec(
            name='SENTIMENT_SERVICE_URL',
            description='감정 분석 서비스 URL',
            required_for_modes=[TradingMode.LIVE, TradingMode.PAPER],
            validation_level=ValidationLevel.RECOMMENDED,
            pattern=r'^https?://[a-zA-Z0-9.-]+:[0-9]+/?$',
            default_value='http://sentiment-service:8001',
            example='http://sentiment-service:8001'
        )
        
        # Redis 설정
        specs['REDIS_URL'] = EnvVarSpec(
            name='REDIS_URL',
            description='Redis 연결 URL',
            required_for_modes=[TradingMode.LIVE, TradingMode.PAPER],
            validation_level=ValidationLevel.RECOMMENDED,
            pattern=r'^redis://[a-zA-Z0-9.-]+:[0-9]+/?[0-9]*$',
            default_value='redis://redis:6379/0',
            example='redis://redis:6379/0'
        )
        
        # 로깅 설정
        specs['LOG_LEVEL'] = EnvVarSpec(
            name='LOG_LEVEL',
            description='로그 레벨',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$',
            default_value='INFO',
            example='INFO'
        )
        
        # VPS 최적화 설정
        specs['VPS_MEMORY_LIMIT_GB'] = EnvVarSpec(
            name='VPS_MEMORY_LIMIT_GB',
            description='VPS 메모리 제한 (GB)',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^[1-9][0-9]*$',
            default_value='3',
            example='3'
        )
        
        specs['VPS_CPU_CORES'] = EnvVarSpec(
            name='VPS_CPU_CORES',
            description='VPS CPU 코어 수',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^[1-9][0-9]*$',
            default_value='2',
            example='2'
        )
        
        # 보안 설정
        specs['TRADING_MODE'] = EnvVarSpec(
            name='TRADING_MODE',
            description='거래 모드 (live/paper/backtest)',
            required_for_modes=[TradingMode.LIVE, TradingMode.PAPER],
            validation_level=ValidationLevel.REQUIRED,
            pattern=r'^(live|paper|backtest|dry_run)$',
            default_value='paper',
            example='paper'
        )
        
        # 알림 설정 (선택)
        specs['SLACK_WEBHOOK_URL'] = EnvVarSpec(
            name='SLACK_WEBHOOK_URL',
            description='Slack 웹훅 URL',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^https://hooks\.slack\.com/services/.*$',
            is_sensitive=True,
            example='https://hooks.slack.com/services/...'
        )
        
        specs['TELEGRAM_BOT_TOKEN'] = EnvVarSpec(
            name='TELEGRAM_BOT_TOKEN',
            description='텔레그램 봇 토큰',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^[0-9]+:[A-Za-z0-9_-]+$',
            is_sensitive=True,
            example='123456789:ABCdefGHIjklMNOpqrsTUVwxyz'
        )
        
        specs['TELEGRAM_CHAT_ID'] = EnvVarSpec(
            name='TELEGRAM_CHAT_ID',
            description='텔레그램 채팅 ID',
            required_for_modes=[],
            validation_level=ValidationLevel.OPTIONAL,
            pattern=r'^-?[0-9]+$',
            example='-123456789'
        )
        
        return specs
    
    def validate_environment(self, trading_mode: Union[str, TradingMode]) -> ValidationResult:
        """환경변수 검증"""
        if isinstance(trading_mode, str):
            trading_mode = TradingMode(trading_mode.lower())
        
        result = ValidationResult(
            is_valid=True,
            missing_required=[],
            missing_recommended=[],
            invalid_format=[],
            security_issues=[],
            warnings=[],
            config_summary={}
        )
        
        logger.info(f"🔍 환경변수 검증 시작 (모드: {trading_mode.value})")
        
        for spec in self.env_specs.values():
            env_value = os.getenv(spec.name)
            
            # 필수 변수 체크
            if trading_mode in spec.required_for_modes:
                if not env_value:
                    if spec.validation_level == ValidationLevel.REQUIRED:
                        result.missing_required.append(spec.name)
                        result.is_valid = False
                    elif spec.validation_level == ValidationLevel.RECOMMENDED:
                        result.missing_recommended.append(spec.name)
                        result.warnings.append(f"{spec.name} 권장 설정이 누락됨")
            
            if env_value:
                # 형식 검증
                if spec.pattern and not re.match(spec.pattern, env_value):
                    result.invalid_format.append(f"{spec.name}: 형식이 올바르지 않음")
                    result.is_valid = False
                
                # 길이 검증
                if spec.min_length and len(env_value) < spec.min_length:
                    result.invalid_format.append(f"{spec.name}: 최소 {spec.min_length}자 필요")
                    result.is_valid = False
                
                # 보안 이슈 체크
                security_issues = self._check_security_issues(spec.name, env_value)
                result.security_issues.extend(security_issues)
                
                # 설정 요약 (민감한 정보는 마스킹)
                if spec.is_sensitive:
                    result.config_summary[spec.name] = self._mask_sensitive_value(env_value)
                else:
                    result.config_summary[spec.name] = env_value
            else:
                # 기본값 사용
                if spec.default_value:
                    result.config_summary[spec.name] = f"{spec.default_value} (기본값)"
                else:
                    result.config_summary[spec.name] = "미설정"
        
        # 추가 검증
        self._validate_combinations(result, trading_mode)
        
        return result
    
    def _check_security_issues(self, name: str, value: str) -> List[str]:
        """보안 이슈 체크"""
        issues = []
        
        # API 키 보안 체크
        if 'API_KEY' in name or 'API_SECRET' in name:
            # 약한 키 패턴 체크
            if len(set(value)) < 10:  # 문자 다양성 부족
                issues.append(f"{name}: 키의 문자 다양성이 부족함")
            
            # 테스트 키 패턴 체크
            test_patterns = ['test', 'demo', '1234', 'abcd', 'sample']
            if any(pattern in value.lower() for pattern in test_patterns):
                issues.append(f"{name}: 테스트용 키로 의심됨")
        
        # URL 보안 체크
        if 'URL' in name and value.startswith('http://'):
            issues.append(f"{name}: HTTP 사용 권장하지 않음 (HTTPS 사용 권장)")
        
        return issues
    
    def _validate_combinations(self, result: ValidationResult, trading_mode: TradingMode):
        """조합 검증"""
        # 실전 거래 시 추가 검증
        if trading_mode == TradingMode.LIVE:
            # API 키와 시크릿이 모두 있는지 확인
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if api_key and api_secret:
                if len(api_key) == len(api_secret):
                    result.warnings.append("API 키와 시크릿 길이가 동일함 (정상이 아닐 수 있음)")
            
            # 실전 모드인데 테스트넷 설정이 있으면 경고
            if os.getenv('BINANCE_TESTNET_API_KEY'):
                result.warnings.append("실전 모드인데 테스트넷 설정이 발견됨")
        
        # 모의 거래 시 검증
        elif trading_mode == TradingMode.PAPER:
            # 실전 API 키가 설정된 경우 경고
            if os.getenv('BINANCE_API_KEY') and not os.getenv('BINANCE_TESTNET_API_KEY'):
                result.warnings.append("모의 거래 모드이나 실전 API 키만 설정됨")
    
    def _mask_sensitive_value(self, value: str) -> str:
        """민감한 값 마스킹"""
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "*" * (len(value) - 8) + value[-4:]
    
    def generate_env_template(self, trading_mode: TradingMode) -> str:
        """환경변수 템플릿 생성"""
        template_lines = [
            "# AuroraQ VPS 배포 환경변수 템플릿",
            f"# 거래 모드: {trading_mode.value}",
            "",
            "# 거래 모드 설정",
            f"TRADING_MODE={trading_mode.value}",
            ""
        ]
        
        # 모드별 필수/권장 변수 그룹화
        required_specs = []
        recommended_specs = []
        optional_specs = []
        
        for spec in self.env_specs.values():
            if trading_mode in spec.required_for_modes:
                if spec.validation_level == ValidationLevel.REQUIRED:
                    required_specs.append(spec)
                elif spec.validation_level == ValidationLevel.RECOMMENDED:
                    recommended_specs.append(spec)
            elif spec.validation_level == ValidationLevel.OPTIONAL:
                optional_specs.append(spec)
        
        # 필수 변수
        if required_specs:
            template_lines.extend([
                "# ==========================================",
                "# 필수 환경변수 (반드시 설정 필요)",
                "# =========================================="
            ])
            
            for spec in required_specs:
                template_lines.append(f"# {spec.description}")
                if spec.example:
                    template_lines.append(f"# 예시: {spec.example}")
                template_lines.append(f"{spec.name}=")
                template_lines.append("")
        
        # 권장 변수
        if recommended_specs:
            template_lines.extend([
                "# ==========================================", 
                "# 권장 환경변수 (성능/안정성 향상)",
                "# =========================================="
            ])
            
            for spec in recommended_specs:
                template_lines.append(f"# {spec.description}")
                if spec.default_value:
                    template_lines.append(f"# 기본값: {spec.default_value}")
                if spec.example:
                    template_lines.append(f"# 예시: {spec.example}")
                template_lines.append(f"{spec.name}={spec.default_value or ''}")
                template_lines.append("")
        
        # 선택 변수
        if optional_specs:
            template_lines.extend([
                "# ==========================================",
                "# 선택적 환경변수 (기능 확장)",
                "# =========================================="
            ])
            
            for spec in optional_specs:
                template_lines.append(f"# {spec.description}")
                if spec.default_value:
                    template_lines.append(f"# 기본값: {spec.default_value}")
                if spec.example:
                    template_lines.append(f"# 예시: {spec.example}")
                template_lines.append(f"# {spec.name}={spec.example or spec.default_value or ''}")
                template_lines.append("")
        
        return "\n".join(template_lines)
    
    def save_validation_report(self, result: ValidationResult, output_path: str):
        """검증 결과 리포트 저장"""
        report = {
            "validation_summary": {
                "is_valid": result.is_valid,
                "total_issues": len(result.missing_required) + len(result.missing_recommended) + 
                               len(result.invalid_format) + len(result.security_issues),
                "critical_issues": len(result.missing_required) + len(result.invalid_format),
                "warnings": len(result.missing_recommended) + len(result.warnings)
            },
            "issues": {
                "missing_required": result.missing_required,
                "missing_recommended": result.missing_recommended,
                "invalid_format": result.invalid_format,
                "security_issues": result.security_issues,
                "warnings": result.warnings
            },
            "configuration": result.config_summary
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 검증 리포트 저장: {output_path}")

def validate_vps_environment(trading_mode: str = "paper") -> ValidationResult:
    """VPS 환경 검증 실행"""
    validator = VPSEnvironmentValidator()
    return validator.validate_environment(trading_mode)

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VPS 환경변수 검증")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "dry_run"],
        default="paper",
        help="거래 모드"
    )
    parser.add_argument("--generate-template", action="store_true", help="환경변수 템플릿 생성")
    parser.add_argument("--output", help="출력 파일 경로")
    
    args = parser.parse_args()
    
    validator = VPSEnvironmentValidator()
    
    if args.generate_template:
        # 템플릿 생성
        trading_mode = TradingMode(args.mode)
        template = validator.generate_env_template(trading_mode)
        
        output_path = args.output or f".env.{args.mode}.template"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"✅ 환경변수 템플릿 생성: {output_path}")
    else:
        # 검증 실행
        result = validator.validate_environment(args.mode)
        
        # 결과 출력
        print(f"🔍 환경변수 검증 결과 (모드: {args.mode})")
        print(f"전체 상태: {'✅ 통과' if result.is_valid else '❌ 실패'}")
        
        if result.missing_required:
            print(f"\n❌ 필수 변수 누락 ({len(result.missing_required)}개):")
            for var in result.missing_required:
                print(f"  - {var}")
        
        if result.missing_recommended:
            print(f"\n⚠️ 권장 변수 누락 ({len(result.missing_recommended)}개):")
            for var in result.missing_recommended:
                print(f"  - {var}")
        
        if result.invalid_format:
            print(f"\n❌ 형식 오류 ({len(result.invalid_format)}개):")
            for issue in result.invalid_format:
                print(f"  - {issue}")
        
        if result.security_issues:
            print(f"\n🔒 보안 이슈 ({len(result.security_issues)}개):")
            for issue in result.security_issues:
                print(f"  - {issue}")
        
        if result.warnings:
            print(f"\n⚠️ 경고 ({len(result.warnings)}개):")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        # 리포트 저장
        if args.output:
            validator.save_validation_report(result, args.output)

if __name__ == "__main__":
    main()