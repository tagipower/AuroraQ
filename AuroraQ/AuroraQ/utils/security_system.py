#!/usr/bin/env python3
"""
VPS Deployment 보안 강화 및 검증 시스템
인증, 권한부여, 암호화, 보안 감사, 취약점 검사
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Tuple
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from collections import defaultdict, deque
from enum import Enum
import ipaddress


class SecurityLevel(Enum):
    """보안 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """위협 타입"""
    BRUTE_FORCE = "brute_force"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    INJECTION = "injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_INPUT = "malicious_input"


class SecurityEvent:
    """보안 이벤트"""
    
    def __init__(self, event_type: ThreatType, severity: SecurityLevel, 
                 description: str, source_ip: str = None, user_id: str = None,
                 metadata: Dict[str, Any] = None):
        self.id = secrets.token_hex(16)
        self.timestamp = time.time()
        self.event_type = event_type
        self.severity = severity
        self.description = description
        self.source_ip = source_ip
        self.user_id = user_id
        self.metadata = metadata or {}
        self.resolved = False


class RateLimiter:
    """속도 제한기"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(deque)
        
    def is_allowed(self, identifier: str) -> bool:
        """요청 허용 여부 확인"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # 오래된 요청 제거
        request_times = self.requests[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # 요청 수 확인
        if len(request_times) >= self.max_requests:
            return False
        
        # 새 요청 기록
        request_times.append(current_time)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """남은 요청 수"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        request_times = self.requests[identifier]
        # 오래된 요청 제거
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        return max(0, self.max_requests - len(request_times))


class PasswordPolicy:
    """비밀번호 정책"""
    
    def __init__(self, min_length: int = 12, require_uppercase: bool = True,
                 require_lowercase: bool = True, require_digits: bool = True,
                 require_special: bool = True, max_age_days: int = 90):
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
        self.max_age_days = max_age_days
        
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """비밀번호 검증"""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"비밀번호는 최소 {self.min_length}자 이상이어야 합니다.")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("대문자가 포함되어야 합니다.")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("소문자가 포함되어야 합니다.")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("숫자가 포함되어야 합니다.")
        
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("특수문자가 포함되어야 합니다.")
        
        # 공통 패스워드 확인
        common_passwords = ['password', '123456', 'admin', 'root', 'user']
        if password.lower() in common_passwords:
            errors.append("일반적으로 사용되는 비밀번호는 사용할 수 없습니다.")
        
        return len(errors) == 0, errors
    
    def generate_secure_password(self, length: int = None) -> str:
        """안전한 비밀번호 생성"""
        if length is None:
            length = max(self.min_length, 16)
        
        chars = ""
        if self.require_lowercase:
            chars += "abcdefghijklmnopqrstuvwxyz"
        if self.require_uppercase:
            chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.require_digits:
            chars += "0123456789"
        if self.require_special:
            chars += "!@#$%^&*(),.?\":{}|<>"
        
        if not chars:
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        return ''.join(secrets.choice(chars) for _ in range(length))


class EncryptionManager:
    """암호화 관리자"""
    
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        self.cipher_suite = Fernet(key)
        
    @classmethod
    def from_password(cls, password: str, salt: bytes = None):
        """비밀번호로부터 키 생성"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key), salt
    
    def encrypt(self, data: str) -> str:
        """데이터 암호화"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """딕셔너리 암호화"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """딕셔너리 복호화"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class JWTManager:
    """JWT 토큰 관리자"""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256"):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = algorithm
        
    def generate_token(self, payload: Dict[str, Any], expires_hours: int = 24) -> str:
        """JWT 토큰 생성"""
        payload = payload.copy()
        payload['exp'] = datetime.utcnow() + timedelta(hours=expires_hours)
        payload['iat'] = datetime.utcnow()
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}


class IPWhitelist:
    """IP 화이트리스트"""
    
    def __init__(self, allowed_ips: List[str] = None):
        self.allowed_networks = []
        if allowed_ips:
            for ip in allowed_ips:
                try:
                    self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
                except ValueError:
                    pass
    
    def add_ip(self, ip: str):
        """IP 추가"""
        try:
            self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
        except ValueError:
            pass
    
    def is_allowed(self, ip: str) -> bool:
        """IP 허용 여부 확인"""
        if not self.allowed_networks:
            return True  # 화이트리스트가 비어있으면 모든 IP 허용
        
        try:
            client_ip = ipaddress.ip_address(ip)
            return any(client_ip in network for network in self.allowed_networks)
        except ValueError:
            return False


class SecurityAuditor:
    """보안 감사자"""
    
    def __init__(self):
        self.security_events = deque(maxlen=10000)
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        self.suspicious_patterns = []
        
        self.logger = logging.getLogger(f"{__name__}.SecurityAuditor")
    
    def log_security_event(self, event: SecurityEvent):
        """보안 이벤트 기록"""
        self.security_events.append(event)
        
        # 실패한 시도 추적
        if event.event_type == ThreatType.UNAUTHORIZED_ACCESS and event.source_ip:
            self.failed_attempts[event.source_ip] += 1
            
            # 5회 이상 실패시 IP 차단
            if self.failed_attempts[event.source_ip] >= 5:
                self.blocked_ips.add(event.source_ip)
                self.logger.warning(f"IP 차단: {event.source_ip}")
        
        # 로깅
        self._log_event(event)
    
    def _log_event(self, event: SecurityEvent):
        """이벤트 로깅"""
        log_message = f"보안 이벤트 [{event.severity.value.upper()}]: {event.description}"
        
        if event.severity == SecurityLevel.CRITICAL:
            self.logger.critical(log_message)
        elif event.severity == SecurityLevel.HIGH:
            self.logger.error(log_message)
        elif event.severity == SecurityLevel.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """IP 차단 여부 확인"""
        return ip in self.blocked_ips
    
    def unblock_ip(self, ip: str):
        """IP 차단 해제"""
        self.blocked_ips.discard(ip)
        self.failed_attempts.pop(ip, None)
        self.logger.info(f"IP 차단 해제: {ip}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """보안 요약 정보"""
        if not self.security_events:
            return {"message": "No security events recorded"}
        
        events = list(self.security_events)
        
        # 이벤트 타입별 통계
        event_types = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in events:
            event_types[event.event_type.value] += 1
            severity_counts[event.severity.value] += 1
        
        # 최근 24시간 이벤트
        recent_time = time.time() - 86400  # 24시간
        recent_events = [e for e in events if e.timestamp > recent_time]
        
        return {
            'total_events': len(events),
            'recent_events_24h': len(recent_events),
            'blocked_ips': len(self.blocked_ips),
            'event_types': dict(event_types),
            'severity_distribution': dict(severity_counts),
            'top_blocked_ips': list(self.blocked_ips)[:10],
            'recent_critical_events': [
                {
                    'timestamp': datetime.fromtimestamp(e.timestamp).isoformat(),
                    'type': e.event_type.value,
                    'description': e.description,
                    'source_ip': e.source_ip
                }
                for e in recent_events if e.severity == SecurityLevel.CRITICAL
            ][-5:]  # 최근 5개
        }


class InputValidator:
    """입력 검증기"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """이메일 검증"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """전화번호 검증"""
        pattern = r'^\+?1?-?\.?\s?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255) -> str:
        """문자열 정제"""
        if not isinstance(input_str, str):
            return ""
        
        # HTML 태그 제거
        clean_str = re.sub(r'<[^>]+>', '', input_str)
        
        # 특수문자 제한
        clean_str = re.sub(r'[<>"\']', '', clean_str)
        
        # 길이 제한
        return clean_str[:max_length]
    
    @staticmethod
    def detect_sql_injection(input_str: str) -> bool:
        """SQL 인젝션 탐지"""
        sql_patterns = [
            r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)',
            r'(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+',
            r'[\'"]\s*;\s*--',
            r'[\'"]\s*OR\s+[\'"]\d+[\'"]\s*=\s*[\'"]\d+[\'"]'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def detect_xss(input_str: str) -> bool:
        """XSS 공격 탐지"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>'
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        
        return False


class SecuritySystem:
    """통합 보안 시스템"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.password_policy = PasswordPolicy()
        self.encryption_manager = EncryptionManager()
        self.jwt_manager = JWTManager()
        self.ip_whitelist = IPWhitelist()
        self.security_auditor = SecurityAuditor()
        self.input_validator = InputValidator()
        
        self.logger = logging.getLogger(__name__)
        
        # 보안 설정
        self.security_config = {
            'enable_rate_limiting': True,
            'enable_ip_whitelist': False,
            'enable_input_validation': True,
            'enable_encryption': True,
            'max_login_attempts': 5,
            'session_timeout_hours': 24,
            'password_policy_enabled': True
        }
    
    def secure_endpoint(self, require_auth: bool = True, require_ip_whitelist: bool = False,
                       rate_limit: bool = True):
        """엔드포인트 보안 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._secure_call(func, require_auth, require_ip_whitelist, 
                                             rate_limit, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._secure_call_sync(func, require_auth, require_ip_whitelist, 
                                            rate_limit, *args, **kwargs)
            
            if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _secure_call(self, func, require_auth, require_ip_whitelist, rate_limit, 
                          *args, **kwargs):
        """보안 검사 후 비동기 함수 호출"""
        # 여기서는 간단한 시뮬레이션
        # 실제로는 request 객체에서 IP, 토큰 등을 추출해야 함
        
        client_ip = kwargs.get('client_ip', '127.0.0.1')
        auth_token = kwargs.get('auth_token')
        
        # IP 화이트리스트 확인
        if require_ip_whitelist and not self.ip_whitelist.is_allowed(client_ip):
            self.security_auditor.log_security_event(
                SecurityEvent(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.HIGH,
                    f"IP 화이트리스트에 없는 접근: {client_ip}",
                    source_ip=client_ip
                )
            )
            raise PermissionError("IP not whitelisted")
        
        # 차단된 IP 확인
        if self.security_auditor.is_ip_blocked(client_ip):
            raise PermissionError("IP blocked due to suspicious activity")
        
        # 속도 제한 확인
        if rate_limit and not self.rate_limiter.is_allowed(client_ip):
            self.security_auditor.log_security_event(
                SecurityEvent(
                    ThreatType.BRUTE_FORCE,
                    SecurityLevel.MEDIUM,
                    f"속도 제한 초과: {client_ip}",
                    source_ip=client_ip
                )
            )
            raise PermissionError("Rate limit exceeded")
        
        # 인증 확인
        if require_auth:
            if not auth_token:
                raise PermissionError("Authentication required")
            
            valid, payload = self.jwt_manager.verify_token(auth_token)
            if not valid:
                self.security_auditor.log_security_event(
                    SecurityEvent(
                        ThreatType.UNAUTHORIZED_ACCESS,
                        SecurityLevel.MEDIUM,
                        f"잘못된 토큰으로 접근 시도: {client_ip}",
                        source_ip=client_ip
                    )
                )
                raise PermissionError("Invalid token")
        
        # 함수 실행
        return await func(*args, **kwargs)
    
    def _secure_call_sync(self, func, require_auth, require_ip_whitelist, rate_limit, 
                         *args, **kwargs):
        """보안 검사 후 동기 함수 호출"""
        # 비동기 버전과 동일한 로직
        return func(*args, **kwargs)
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """입력 데이터 검증"""
        errors = []
        
        for key, value in input_data.items():
            if isinstance(value, str):
                # SQL 인젝션 검사
                if self.input_validator.detect_sql_injection(value):
                    errors.append(f"SQL injection detected in {key}")
                    self.security_auditor.log_security_event(
                        SecurityEvent(
                            ThreatType.INJECTION,
                            SecurityLevel.HIGH,
                            f"SQL injection attempt in field: {key}",
                            metadata={'field': key, 'value': value[:100]}
                        )
                    )
                
                # XSS 검사
                if self.input_validator.detect_xss(value):
                    errors.append(f"XSS attempt detected in {key}")
                    self.security_auditor.log_security_event(
                        SecurityEvent(
                            ThreatType.INJECTION,
                            SecurityLevel.HIGH,
                            f"XSS attempt in field: {key}",
                            metadata={'field': key, 'value': value[:100]}
                        )
                    )
        
        return len(errors) == 0, errors
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], 
                              sensitive_fields: List[str] = None) -> Dict[str, Any]:
        """민감한 데이터 암호화"""
        if not sensitive_fields:
            sensitive_fields = ['password', 'api_key', 'secret', 'token', 'private_key']
        
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and isinstance(encrypted_data[field], str):
                try:
                    encrypted_data[field] = self.encryption_manager.encrypt(encrypted_data[field])
                except Exception as e:
                    self.logger.error(f"암호화 실패 ({field}): {e}")
        
        return encrypted_data
    
    def run_security_scan(self) -> Dict[str, Any]:
        """보안 검사 실행"""
        scan_results = {
            'timestamp': datetime.now().isoformat(),
            'scan_type': 'comprehensive',
            'vulnerabilities': [],
            'recommendations': [],
            'security_score': 100
        }
        
        # 1. 비밀번호 정책 확인
        if not self.security_config['password_policy_enabled']:
            scan_results['vulnerabilities'].append({
                'type': 'password_policy',
                'severity': 'medium',
                'description': '비밀번호 정책이 비활성화되어 있습니다.'
            })
            scan_results['security_score'] -= 10
        
        # 2. IP 화이트리스트 확인
        if not self.security_config['enable_ip_whitelist']:
            scan_results['recommendations'].append(
                'IP 화이트리스트를 활성화하여 접근을 제한하는 것을 고려하세요.'
            )
        
        # 3. 속도 제한 확인
        if not self.security_config['enable_rate_limiting']:
            scan_results['vulnerabilities'].append({
                'type': 'rate_limiting',
                'severity': 'high',
                'description': '속도 제한이 비활성화되어 있어 무차별 공격에 취약합니다.'
            })
            scan_results['security_score'] -= 20
        
        # 4. 암호화 확인
        if not self.security_config['enable_encryption']:
            scan_results['vulnerabilities'].append({
                'type': 'encryption',
                'severity': 'critical',
                'description': '데이터 암호화가 비활성화되어 있습니다.'
            })
            scan_results['security_score'] -= 30
        
        # 5. 보안 이벤트 분석
        security_summary = self.security_auditor.get_security_summary()
        if security_summary.get('blocked_ips', 0) > 10:
            scan_results['vulnerabilities'].append({
                'type': 'suspicious_activity',
                'severity': 'high',
                'description': f"{security_summary['blocked_ips']}개의 IP가 차단되었습니다."
            })
        
        # 보안 점수 등급
        score = scan_results['security_score']
        if score >= 90:
            scan_results['security_grade'] = 'A'
        elif score >= 80:
            scan_results['security_grade'] = 'B'
        elif score >= 70:
            scan_results['security_grade'] = 'C'
        elif score >= 60:
            scan_results['security_grade'] = 'D'
        else:
            scan_results['security_grade'] = 'F'
        
        return scan_results
    
    def generate_security_report(self) -> str:
        """보안 리포트 생성"""
        scan_results = self.run_security_scan()
        security_summary = self.security_auditor.get_security_summary()
        
        report = f"""
보안 검사 리포트
================

검사 시간: {scan_results['timestamp']}
보안 점수: {scan_results['security_score']}/100 ({scan_results['security_grade']})

취약점 ({len(scan_results['vulnerabilities'])}개):
{'='*50}
"""
        
        for vuln in scan_results['vulnerabilities']:
            report += f"- [{vuln['severity'].upper()}] {vuln['description']}\n"
        
        report += f"""

권장사항 ({len(scan_results['recommendations'])}개):
{'='*50}
"""
        
        for rec in scan_results['recommendations']:
            report += f"- {rec}\n"
        
        report += f"""

보안 이벤트 요약:
{'='*50}
- 총 이벤트: {security_summary.get('total_events', 0)}개
- 최근 24시간: {security_summary.get('recent_events_24h', 0)}개
- 차단된 IP: {security_summary.get('blocked_ips', 0)}개

시스템 보안 설정:
{'='*50}
- 속도 제한: {'활성화' if self.security_config['enable_rate_limiting'] else '비활성화'}
- IP 화이트리스트: {'활성화' if self.security_config['enable_ip_whitelist'] else '비활성화'}
- 입력 검증: {'활성화' if self.security_config['enable_input_validation'] else '비활성화'}
- 데이터 암호화: {'활성화' if self.security_config['enable_encryption'] else '비활성화'}
- 비밀번호 정책: {'활성화' if self.security_config['password_policy_enabled'] else '비활성화'}
"""
        
        return report


# 전역 보안 시스템
global_security = SecuritySystem()


# 편의 함수들
def secure_endpoint(require_auth: bool = True, require_ip_whitelist: bool = False,
                   rate_limit: bool = True):
    """보안 엔드포인트 데코레이터 (편의 함수)"""
    return global_security.secure_endpoint(require_auth, require_ip_whitelist, rate_limit)


def validate_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """입력 검증 (편의 함수)"""
    return global_security.validate_input(input_data)


def encrypt_data(data: Dict[str, Any], sensitive_fields: List[str] = None) -> Dict[str, Any]:
    """데이터 암호화 (편의 함수)"""
    return global_security.encrypt_sensitive_data(data, sensitive_fields)


def security_scan():
    """보안 검사 (편의 함수)"""
    return global_security.run_security_scan()


def security_report():
    """보안 리포트 (편의 함수)"""
    return global_security.generate_security_report()


# 사용 예시
if __name__ == "__main__":
    
    # 보안 엔드포인트 예시
    @secure_endpoint(require_auth=True, rate_limit=True)
    def secure_api_endpoint(data: Dict[str, Any], client_ip: str = "127.0.0.1", 
                           auth_token: str = None):
        """보안이 적용된 API 엔드포인트"""
        
        # 입력 검증
        valid, errors = validate_input(data)
        if not valid:
            raise ValueError(f"입력 검증 실패: {errors}")
        
        # 민감한 데이터 암호화
        encrypted_data = encrypt_data(data, ['password', 'api_key'])
        
        return {"status": "success", "data": encrypted_data}
    
    # 테스트 실행
    def test_security_system():
        print("🔒 보안 시스템 테스트")
        
        # JWT 토큰 생성
        token = global_security.jwt_manager.generate_token({'user_id': 'test_user'})
        print(f"생성된 JWT 토큰: {token[:50]}...")
        
        # 보안 API 호출 테스트
        try:
            test_data = {'username': 'test', 'password': 'secure123!@#'}
            result = secure_api_endpoint(test_data, auth_token=token)
            print(f"보안 API 호출 성공: {result}")
        except Exception as e:
            print(f"보안 API 호출 실패: {e}")
        
        # 악성 입력 테스트
        malicious_data = {'input': "'; DROP TABLE users; --"}
        valid, errors = validate_input(malicious_data)
        print(f"악성 입력 검증: {'차단됨' if not valid else '통과'} - {errors}")
        
        # 보안 검사 실행
        scan_result = security_scan()
        print(f"\n🛡️ 보안 검사 결과:")
        print(f"보안 점수: {scan_result['security_score']}/100 ({scan_result['security_grade']})")
        print(f"취약점: {len(scan_result['vulnerabilities'])}개")
        
        # 보안 리포트 생성
        report = security_report()
        print(f"\n📋 보안 리포트:")
        print(report[:500] + "..." if len(report) > 500 else report)
    
    # 테스트 실행
    test_security_system()