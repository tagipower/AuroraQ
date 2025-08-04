#!/usr/bin/env python3
"""
VPS Deployment 통합 검증 스크립트
주요 시스템 간 연동 상태를 체계적으로 검증합니다.
"""

import os
import sys
import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json

# 프로젝트 루트 경로 설정
VPS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VPS_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_validator")


class IntegrationValidator:
    """시스템 통합 검증기"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        
    def check_module_import(self, module_path: str, module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """모듈 임포트 테스트"""
        try:
            # 모듈 경로 구성
            full_path = VPS_ROOT / module_path
            
            if not full_path.exists():
                return False, None, f"Module file not found: {full_path}"
            
            # 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            if spec is None or spec.loader is None:
                return False, None, f"Failed to create module spec for {module_name}"
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return True, module, None
            
        except Exception as e:
            return False, None, f"Import error: {str(e)}"
    
    def validate_env_loader_integration(self) -> Dict[str, Any]:
        """1. deployment/api_system.py ↔ config/env_loader.py 연동 검증"""
        logger.info("Checking env_loader integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # env_loader 임포트
            success, env_module, error = self.check_module_import(
                "config/env_loader.py", 
                "env_loader"
            )
            
            if not success:
                result["status"] = "FAIL"
                result["error"] = error
                return result
            
            # 환경 설정 로드 테스트
            config = env_module.get_vps_env_config()
            
            # 주요 설정 확인
            critical_fields = [
                "trading_api_port",
                "trading_mode", 
                "symbol",
                "vps_memory_limit",
                "enable_unified_logging"
            ]
            
            missing_fields = []
            for field in critical_fields:
                if not hasattr(config, field):
                    missing_fields.append(field)
                else:
                    result["details"][field] = getattr(config, field)
            
            if missing_fields:
                result["status"] = "WARN"
                result["warning"] = f"Missing config fields: {missing_fields}"
            else:
                result["status"] = "PASS"
                result["message"] = "Environment configuration loaded successfully"
                
            # API 시스템이 env_loader를 사용하는지 확인
            api_path = VPS_ROOT / "deployment/api_system.py"
            with open(api_path, 'r', encoding='utf-8') as f:
                api_content = f.read()
                
            if "env_loader" not in api_content and "get_vps_env_config" not in api_content:
                result["recommendation"] = "api_system.py should import and use env_loader for configuration"
                result["status"] = "WARN"
            
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    def validate_trading_strategy_integration(self) -> Dict[str, Any]:
        """2. trading 시스템 내부 전략 연결 검증"""
        logger.info("Checking trading strategy integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # 주요 모듈 임포트
            modules_to_check = [
                ("trading/vps_realtime_system.py", "vps_realtime_system"),
                ("trading/vps_strategy_adapter.py", "vps_strategy_adapter"),
                ("trading/rule_strategies.py", "rule_strategies"),
                ("trading/ppo_strategy.py", "ppo_strategy"),
                ("trading/unified_signal_interface.py", "unified_signal_interface")
            ]
            
            loaded_modules = {}
            for path, name in modules_to_check:
                success, module, error = self.check_module_import(path, name)
                if success:
                    loaded_modules[name] = module
                    result["details"][name] = "Loaded successfully"
                else:
                    result["details"][name] = f"Failed: {error}"
            
            # 전략 연결 확인
            if all(name in loaded_modules for name in ["rule_strategies", "unified_signal_interface"]):
                # Rule 전략 클래스 확인
                rule_strategies = []
                for attr_name in dir(loaded_modules["rule_strategies"]):
                    attr = getattr(loaded_modules["rule_strategies"], attr_name)
                    if isinstance(attr, type) and attr_name.startswith("RuleStrategy"):
                        rule_strategies.append(attr_name)
                
                result["details"]["rule_strategies_found"] = rule_strategies
                
                # 통합 신호 인터페이스 확인
                if hasattr(loaded_modules["unified_signal_interface"], "UnifiedSignalInterface"):
                    result["details"]["unified_interface"] = "Available"
                else:
                    result["details"]["unified_interface"] = "Not found"
            
            # 연동 상태 판단
            if len(loaded_modules) == len(modules_to_check):
                result["status"] = "PASS"
                result["message"] = "All trading modules loaded successfully"
            elif len(loaded_modules) > len(modules_to_check) // 2:
                result["status"] = "WARN"
                result["warning"] = "Some trading modules failed to load"
            else:
                result["status"] = "FAIL"
                result["error"] = "Critical trading modules missing"
                
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    def validate_monitoring_prometheus_integration(self) -> Dict[str, Any]:
        """3. monitoring 시스템과 Prometheus 연동 확인"""
        logger.info("Checking monitoring/Prometheus integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # Prometheus 설정 파일 확인
            prometheus_configs = [
                VPS_ROOT / "monitoring/prometheus.yml",
                VPS_ROOT / "monitoring/prometheus-simple.yml"
            ]
            
            config_found = False
            for config_path in prometheus_configs:
                if config_path.exists():
                    config_found = True
                    result["details"][config_path.name] = "Found"
                    
                    # YAML 파일 기본 검증
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "scrape_configs" in content:
                            result["details"][f"{config_path.name}_valid"] = True
            
            if not config_found:
                result["status"] = "FAIL"
                result["error"] = "No Prometheus configuration found"
                return result
            
            # 모니터링 스크립트 확인
            monitor_path = VPS_ROOT / "monitoring/monitor_vps_trading.py"
            alert_path = VPS_ROOT / "monitoring/monitoring_alert_system.py"
            
            if monitor_path.exists():
                with open(monitor_path, 'r', encoding='utf-8') as f:
                    monitor_content = f.read()
                    
                # Prometheus 연동 패턴 확인
                prometheus_patterns = [
                    "prometheus",
                    "metrics",
                    "gauge",
                    "counter",
                    "histogram"
                ]
                
                patterns_found = [p for p in prometheus_patterns if p.lower() in monitor_content.lower()]
                result["details"]["prometheus_integration"] = len(patterns_found) > 0
                result["details"]["metrics_patterns_found"] = patterns_found
            
            if alert_path.exists():
                result["details"]["alert_system"] = "Available"
            
            # 최종 상태 판단
            if config_found and result["details"].get("prometheus_integration"):
                result["status"] = "PASS"
                result["message"] = "Prometheus monitoring properly configured"
            else:
                result["status"] = "WARN"
                result["warning"] = "Prometheus configuration found but integration unclear"
                
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    def validate_sentiment_trading_integration(self) -> Dict[str, Any]:
        """4. sentiment-service와 trading 시스템 연동 검증"""
        logger.info("Checking sentiment/trading integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # 감정 분석 API 라우터 확인
            sentiment_api_path = VPS_ROOT / "sentiment-service/api/metrics_router.py"
            if sentiment_api_path.exists():
                result["details"]["sentiment_api"] = "Found"
                
                with open(sentiment_api_path, 'r', encoding='utf-8') as f:
                    api_content = f.read()
                    
                # API 엔드포인트 확인
                if "sentiment" in api_content.lower():
                    result["details"]["sentiment_endpoints"] = True
            
            # PPO 전략에서 감정 분석 사용 확인
            ppo_path = VPS_ROOT / "trading/ppo_strategy.py"
            if ppo_path.exists():
                with open(ppo_path, 'r', encoding='utf-8') as f:
                    ppo_content = f.read()
                    
                # 감정 분석 통합 패턴 확인
                sentiment_patterns = [
                    "sentiment",
                    "emotion",
                    "mood",
                    "fear",
                    "greed"
                ]
                
                patterns_in_ppo = [p for p in sentiment_patterns if p.lower() in ppo_content.lower()]
                
                if patterns_in_ppo:
                    result["details"]["ppo_sentiment_integration"] = True
                    result["details"]["sentiment_patterns_in_ppo"] = patterns_in_ppo
                else:
                    result["details"]["ppo_sentiment_integration"] = False
                    result["recommendation"] = "Consider integrating sentiment analysis into PPO strategy"
            
            # 키워드 스코어러 확인
            keyword_scorer_path = VPS_ROOT / "sentiment-service/models/keyword_scorer.py"
            if keyword_scorer_path.exists():
                result["details"]["keyword_scorer"] = "Available"
            
            # 최종 판단
            if result["details"].get("sentiment_api") and result["details"].get("keyword_scorer"):
                if result["details"].get("ppo_sentiment_integration"):
                    result["status"] = "PASS"
                    result["message"] = "Sentiment service integrated with trading"
                else:
                    result["status"] = "WARN"
                    result["warning"] = "Sentiment service available but not fully integrated"
            else:
                result["status"] = "FAIL"
                result["error"] = "Sentiment service components missing"
                
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    def validate_ppo_integration(self) -> Dict[str, Any]:
        """5. PPO 통합 검증"""
        logger.info("Checking PPO integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # PPO 검증 스크립트 확인
            verify_ppo_path = VPS_ROOT / "scripts/verify_ppo_integration.py"
            
            if verify_ppo_path.exists():
                result["details"]["verify_script"] = "Found"
                
                # 검증 스크립트 실행 가능 여부 확인
                success, verify_module, error = self.check_module_import(
                    "scripts/verify_ppo_integration.py",
                    "verify_ppo_integration"
                )
                
                if success:
                    result["details"]["verify_script_valid"] = True
                    
                    # PPO 검증 함수 확인
                    if hasattr(verify_module, "verify_ppo_integration"):
                        result["details"]["verification_function"] = "Available"
                else:
                    result["details"]["verify_script_error"] = error
            
            # PPO 전략 모듈 확인
            ppo_strategy_path = VPS_ROOT / "trading/ppo_strategy.py"
            if ppo_strategy_path.exists():
                with open(ppo_strategy_path, 'r', encoding='utf-8') as f:
                    ppo_content = f.read()
                    
                # PPO 핵심 컴포넌트 확인
                ppo_components = [
                    "PPOStrategy",
                    "policy",
                    "value_function",
                    "advantage",
                    "train",
                    "predict"
                ]
                
                found_components = [c for c in ppo_components if c in ppo_content]
                result["details"]["ppo_components"] = found_components
                
                if len(found_components) >= 3:
                    result["details"]["ppo_implementation"] = "Complete"
                else:
                    result["details"]["ppo_implementation"] = "Partial"
            
            # 최종 판단
            if result["details"].get("verify_script_valid") and \
               result["details"].get("ppo_implementation") == "Complete":
                result["status"] = "PASS"
                result["message"] = "PPO fully integrated and verifiable"
            elif result["details"].get("ppo_implementation"):
                result["status"] = "WARN"
                result["warning"] = "PPO implemented but verification incomplete"
            else:
                result["status"] = "FAIL"
                result["error"] = "PPO integration missing or incomplete"
                
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    def validate_logging_integration(self) -> Dict[str, Any]:
        """6. 로깅 시스템 통합 검증"""
        logger.info("Checking logging system integration...")
        result = {"status": "UNKNOWN", "details": {}}
        
        try:
            # 통합 로그 관리자 확인
            log_manager_path = VPS_ROOT / "vps_logging/unified_log_manager.py"
            
            if log_manager_path.exists():
                result["details"]["unified_log_manager"] = "Found"
                
                # 로그 관리자 임포트
                success, log_module, error = self.check_module_import(
                    "vps_logging/unified_log_manager.py",
                    "unified_log_manager"
                )
                
                if success:
                    result["details"]["log_manager_valid"] = True
                    
                    # 로그 관리자 클래스 확인
                    if hasattr(log_module, "UnifiedLogManager"):
                        result["details"]["UnifiedLogManager"] = "Available"
                else:
                    result["details"]["log_manager_error"] = error
            
            # 로그 보존 정책 확인
            retention_path = VPS_ROOT / "vps_logging/log_retention_policy.py"
            if retention_path.exists():
                result["details"]["retention_policy"] = "Available"
            
            # VPS 통합 확인
            vps_integration_path = VPS_ROOT / "vps_logging/vps_integration.py"
            if vps_integration_path.exists():
                result["details"]["vps_integration"] = "Available"
            
            # 로그 디렉토리 구조 확인
            log_dir = VPS_ROOT / "logs"
            if log_dir.exists():
                subdirs = [d.name for d in log_dir.iterdir() if d.is_dir()]
                result["details"]["log_directories"] = subdirs if subdirs else "No subdirectories"
            
            # 로깅 사용 패턴 확인 (샘플링)
            sample_files = [
                "deployment/api_system.py",
                "trading/vps_realtime_system.py",
                "monitoring/monitor_vps_trading.py"
            ]
            
            logging_usage = {}
            for file_path in sample_files:
                full_path = VPS_ROOT / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 로깅 패턴 확인
                    has_logging = any(pattern in content for pattern in [
                        "import logging",
                        "from vps_logging",
                        "unified_log_manager",
                        "UnifiedLogManager"
                    ])
                    
                    logging_usage[file_path] = has_logging
            
            result["details"]["logging_usage"] = logging_usage
            
            # 최종 판단
            components_found = sum([
                result["details"].get("unified_log_manager") == "Found",
                result["details"].get("UnifiedLogManager") == "Available",
                result["details"].get("retention_policy") == "Available",
                result["details"].get("vps_integration") == "Available"
            ])
            
            if components_found >= 3:
                result["status"] = "PASS"
                result["message"] = "Logging system properly integrated"
            elif components_found >= 2:
                result["status"] = "WARN"
                result["warning"] = "Logging system partially integrated"
            else:
                result["status"] = "FAIL"
                result["error"] = "Logging system integration incomplete"
                
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            
        return result
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """모든 검증 실행"""
        logger.info("Starting comprehensive integration validation...")
        
        # 검증 함수들
        validations = [
            ("env_loader_integration", self.validate_env_loader_integration),
            ("trading_strategy_integration", self.validate_trading_strategy_integration),
            ("monitoring_prometheus_integration", self.validate_monitoring_prometheus_integration),
            ("sentiment_trading_integration", self.validate_sentiment_trading_integration),
            ("ppo_integration", self.validate_ppo_integration),
            ("logging_integration", self.validate_logging_integration)
        ]
        
        # 각 검증 실행
        for name, validation_func in validations:
            logger.info(f"\nRunning {name}...")
            result = validation_func()
            self.results["checks"][name] = result
            
            # 결과 집계
            self.results["summary"]["total"] += 1
            if result["status"] == "PASS":
                self.results["summary"]["passed"] += 1
            elif result["status"] == "WARN":
                self.results["summary"]["warnings"] += 1
            else:
                self.results["summary"]["failed"] += 1
        
        return self.results
    
    def generate_report(self) -> str:
        """검증 보고서 생성"""
        lines = []
        lines.append("=" * 80)
        lines.append("VPS DEPLOYMENT INTEGRATION VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.results['timestamp']}")
        lines.append(f"Total Checks: {self.results['summary']['total']}")
        lines.append(f"Passed: {self.results['summary']['passed']}")
        lines.append(f"Warnings: {self.results['summary']['warnings']}")
        lines.append(f"Failed: {self.results['summary']['failed']}")
        lines.append("")
        
        # 각 검증 결과
        for check_name, result in self.results["checks"].items():
            lines.append("-" * 60)
            lines.append(f"CHECK: {check_name}")
            lines.append(f"STATUS: {result['status']}")
            
            if "message" in result:
                lines.append(f"MESSAGE: {result['message']}")
            if "warning" in result:
                lines.append(f"WARNING: {result['warning']}")
            if "error" in result:
                lines.append(f"ERROR: {result['error']}")
            if "recommendation" in result:
                lines.append(f"RECOMMENDATION: {result['recommendation']}")
            
            if "details" in result and result["details"]:
                lines.append("DETAILS:")
                for key, value in result["details"].items():
                    lines.append(f"  - {key}: {value}")
            
            lines.append("")
        
        # 권장사항
        lines.append("=" * 80)
        lines.append("RECOMMENDATIONS:")
        
        if self.results["summary"]["failed"] > 0:
            lines.append("1. Fix critical integration failures first")
            lines.append("2. Review error messages for specific issues")
        
        if self.results["summary"]["warnings"] > 0:
            lines.append("3. Address warnings to improve system reliability")
            lines.append("4. Consider implementing missing integrations")
        
        if self.results["checks"].get("env_loader_integration", {}).get("status") != "PASS":
            lines.append("5. Update api_system.py to use env_loader for configuration")
        
        if self.results["checks"].get("sentiment_trading_integration", {}).get("status") == "WARN":
            lines.append("6. Integrate sentiment analysis into trading strategies")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_results(self, output_path: Optional[str] = None):
        """결과 저장"""
        if output_path is None:
            output_path = VPS_ROOT / "docs" / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        
        # 텍스트 보고서도 저장
        report_path = output_path.with_suffix('.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        logger.info(f"Report saved to: {report_path}")


async def main():
    """메인 실행 함수"""
    validator = IntegrationValidator()
    
    try:
        # 검증 실행
        results = await validator.run_all_validations()
        
        # 보고서 출력
        print("\n" + validator.generate_report())
        
        # 결과 저장
        validator.save_results()
        
        # 종료 코드 결정
        if results["summary"]["failed"] > 0:
            sys.exit(1)
        elif results["summary"]["warnings"] > 0:
            sys.exit(0)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())