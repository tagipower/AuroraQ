#!/usr/bin/env python3
"""
VPS Deployment 시스템 검증기
임포트, 의존성, 최적화, 엔드포인트, 보안 등 전체 검증
"""

import ast
import asyncio
import json
import os
import sys
import time
import traceback
import importlib
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from datetime import datetime
from collections import defaultdict
import subprocess
import requests
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemValidator:
    """시스템 전체 검증기"""
    
    def __init__(self, root_path: str = None):
        self.root_path = Path(root_path) if root_path else Path(__file__).parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': {}
        }
        
    def analyze_imports(self) -> Dict[str, Any]:
        """임포트 의존성 분석"""
        logger.info("🔍 임포트 의존성 분석 시작...")
        
        import_analysis = {
            'python_files': [],
            'import_graph': defaultdict(set),
            'external_dependencies': set(),
            'internal_dependencies': set(),
            'circular_imports': [],
            'missing_imports': [],
            'unused_imports': [],
            'optimization_suggestions': []
        }
        
        # Python 파일 찾기
        python_files = list(self.root_path.rglob("*.py"))
        import_analysis['python_files'] = [str(f.relative_to(self.root_path)) for f in python_files]
        
        # 각 파일의 임포트 분석
        for py_file in python_files:
            if py_file.name.startswith('.') or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_imports = self._extract_imports(tree)
                
                rel_path = str(py_file.relative_to(self.root_path))
                import_analysis['import_graph'][rel_path] = file_imports
                
                # 외부/내부 의존성 분류
                for imp in file_imports:
                    if imp.startswith('.') or any(part in imp for part in ['trading', 'sentiment-service', 'vps_logging']):
                        import_analysis['internal_dependencies'].add(imp)
                    else:
                        import_analysis['external_dependencies'].add(imp)
                        
            except Exception as e:
                logger.error(f"파일 분석 실패 {py_file}: {e}")
        
        # 순환 임포트 검사
        import_analysis['circular_imports'] = self._detect_circular_imports(import_analysis['import_graph'])
        
        # 누락된 임포트 검사
        import_analysis['missing_imports'] = self._check_missing_imports(import_analysis['import_graph'])
        
        # 최적화 제안
        import_analysis['optimization_suggestions'] = self._generate_import_optimizations(import_analysis)
        
        return import_analysis
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """AST에서 임포트 추출"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        return imports
    
    def _detect_circular_imports(self, import_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """순환 임포트 탐지"""
        circular = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 순환 발견
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in import_graph.get(node, []):
                if neighbor in import_graph:  # 내부 파일만
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for file in import_graph:
            if file not in visited:
                dfs(file, [])
        
        return circular
    
    def _check_missing_imports(self, import_graph: Dict[str, Set[str]]) -> List[Dict[str, str]]:
        """누락된 임포트 검사"""
        missing = []
        
        for file_path, imports in import_graph.items():
            for imp in imports:
                try:
                    if not imp.startswith('.'):
                        importlib.import_module(imp)
                except ImportError as e:
                    missing.append({
                        'file': file_path,
                        'import': imp,
                        'error': str(e)
                    })
        
        return missing
    
    def _generate_import_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """임포트 최적화 제안"""
        suggestions = []
        
        # 중복된 외부 의존성
        ext_deps = analysis['external_dependencies']
        if len(ext_deps) > 50:
            suggestions.append(f"외부 의존성이 {len(ext_deps)}개로 많습니다. 필수적인 것만 남기고 정리를 권장합니다.")
        
        # 순환 임포트
        if analysis['circular_imports']:
            suggestions.append(f"순환 임포트 {len(analysis['circular_imports'])}개 발견. 구조 재설계가 필요합니다.")
        
        # 누락된 임포트
        if analysis['missing_imports']:
            suggestions.append(f"누락된 임포트 {len(analysis['missing_imports'])}개 발견. requirements.txt 업데이트가 필요합니다.")
        
        return suggestions
    
    def validate_debugging_system(self) -> Dict[str, Any]:
        """디버깅 시스템 검증"""
        logger.info("🐛 디버깅 시스템 검증 시작...")
        
        debug_validation = {
            'logging_setup': False,
            'error_handling': [],
            'trace_capabilities': False,
            'debug_endpoints': [],
            'log_levels': [],
            'debug_modes': [],
            'suggestions': []
        }
        
        # 로깅 설정 확인
        logging_files = list(self.root_path.rglob("*log*.py"))
        if logging_files:
            debug_validation['logging_setup'] = True
            debug_validation['suggestions'].append("✅ 로깅 시스템 파일 발견")
        
        # 에러 핸들링 패턴 검사
        python_files = list(self.root_path.rglob("*.py"))
        error_patterns = ['try:', 'except:', 'raise', 'logger.error', 'logger.warning']
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                error_handling_count = sum(1 for pattern in error_patterns if pattern in content)
                if error_handling_count > 0:
                    debug_validation['error_handling'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'error_handling_patterns': error_handling_count
                    })
            except:
                continue
        
        # 디버깅 제안
        if len(debug_validation['error_handling']) < 5:
            debug_validation['suggestions'].append("⚠️ 에러 핸들링이 부족합니다. try-except 블록을 더 추가하세요.")
        
        return debug_validation
    
    def analyze_performance(self) -> Dict[str, Any]:
        """성능 분석"""
        logger.info("⚡ 성능 분석 시작...")
        
        performance_analysis = {
            'memory_usage': {},
            'cpu_usage': {},
            'async_patterns': [],
            'bottlenecks': [],
            'optimization_opportunities': [],
            'suggestions': []
        }
        
        # 현재 시스템 리소스
        process = psutil.Process()
        performance_analysis['memory_usage'] = {
            'rss_mb': process.memory_info().rss / 1024 / 1024,
            'vms_mb': process.memory_info().vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
        
        performance_analysis['cpu_usage'] = {
            'percent': process.cpu_percent(interval=1),
            'num_threads': process.num_threads()
        }
        
        # 비동기 패턴 검사
        python_files = list(self.root_path.rglob("*.py"))
        async_patterns = ['async def', 'await ', 'asyncio.', 'aiohttp']
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                async_count = sum(1 for pattern in async_patterns if pattern in content)
                if async_count > 0:
                    performance_analysis['async_patterns'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'async_patterns': async_count
                    })
            except:
                continue
        
        # 최적화 제안
        if performance_analysis['memory_usage']['rss_mb'] > 500:
            performance_analysis['suggestions'].append("⚠️ 메모리 사용량이 높습니다. 메모리 최적화를 고려하세요.")
        
        if len(performance_analysis['async_patterns']) < 3:
            performance_analysis['suggestions'].append("💡 비동기 패턴 사용을 늘려 성능을 향상시킬 수 있습니다.")
        
        return performance_analysis
    
    async def validate_endpoints(self) -> Dict[str, Any]:
        """API 엔드포인트 검증"""
        logger.info("🌐 API 엔드포인트 검증 시작...")
        
        endpoint_validation = {
            'api_files': [],
            'endpoints_found': [],
            'health_checks': [],
            'security_headers': [],
            'response_times': {},
            'status_codes': {},
            'suggestions': []
        }
        
        # API 관련 파일 찾기
        api_patterns = ['router', 'api', 'endpoint', 'server']
        api_files = []
        
        for pattern in api_patterns:
            api_files.extend(list(self.root_path.rglob(f"*{pattern}*.py")))
        
        endpoint_validation['api_files'] = [str(f.relative_to(self.root_path)) for f in api_files]
        
        # 엔드포인트 패턴 검사
        endpoint_patterns = ['@app.', '@router.', 'app.get', 'app.post', 'FastAPI']
        
        for api_file in api_files:
            try:
                with open(api_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                endpoints = []
                for pattern in endpoint_patterns:
                    if pattern in content:
                        endpoints.append(pattern)
                
                if endpoints:
                    endpoint_validation['endpoints_found'].append({
                        'file': str(api_file.relative_to(self.root_path)),
                        'patterns': endpoints
                    })
            except:
                continue
        
        # 로컬 API 테스트 (포트 8004)
        test_urls = [
            'http://localhost:8004/health',
            'http://localhost:8004/api/status',
            'http://localhost:8004/api/positions'
        ]
        
        for url in test_urls:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                response_time = time.time() - start_time
                
                endpoint_validation['response_times'][url] = response_time
                endpoint_validation['status_codes'][url] = response.status_code
                
                if response.status_code == 200:
                    endpoint_validation['health_checks'].append(f"✅ {url}")
                else:
                    endpoint_validation['health_checks'].append(f"⚠️ {url} - Status: {response.status_code}")
                    
            except Exception as e:
                endpoint_validation['health_checks'].append(f"❌ {url} - Error: {str(e)}")
        
        # 제안사항
        if not endpoint_validation['endpoints_found']:
            endpoint_validation['suggestions'].append("⚠️ API 엔드포인트를 찾을 수 없습니다.")
        
        if not endpoint_validation['health_checks']:
            endpoint_validation['suggestions'].append("💡 헬스체크 엔드포인트 추가를 권장합니다.")
        
        return endpoint_validation
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """에러 핸들링 검증"""
        logger.info("🚨 에러 핸들링 검증 시작...")
        
        error_validation = {
            'try_catch_blocks': 0,
            'error_types_handled': set(),
            'logging_in_errors': 0,
            'recovery_mechanisms': [],
            'error_propagation': [],
            'suggestions': []
        }
        
        python_files = list(self.root_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # try-except 블록 개수
                error_validation['try_catch_blocks'] += content.count('try:')
                
                # 로깅이 있는 에러 처리
                if 'except' in content and 'logger' in content:
                    error_validation['logging_in_errors'] += content.count('logger.error')
                
                # 에러 타입 추출
                lines = content.split('\n')
                for line in lines:
                    if 'except ' in line and ':' in line:
                        # except Exception as e: 같은 패턴 추출
                        try:
                            error_type = line.split('except ')[1].split(' as')[0].split(':')[0].strip()
                            if error_type and error_type != 'Exception':
                                error_validation['error_types_handled'].add(error_type)
                        except:
                            continue
                            
            except Exception as e:
                logger.error(f"에러 핸들링 분석 실패 {py_file}: {e}")
        
        # 제안사항
        if error_validation['try_catch_blocks'] < 10:
            error_validation['suggestions'].append("⚠️ try-except 블록이 부족합니다. 더 많은 에러 처리가 필요합니다.")
        
        if error_validation['logging_in_errors'] < 5:
            error_validation['suggestions'].append("💡 에러 발생 시 로깅을 추가하여 디버깅을 개선하세요.")
        
        if len(error_validation['error_types_handled']) < 5:
            error_validation['suggestions'].append("💡 구체적인 예외 타입을 더 많이 처리하세요.")
        
        # set을 list로 변환
        error_validation['error_types_handled'] = list(error_validation['error_types_handled'])
        
        return error_validation
    
    def validate_security(self) -> Dict[str, Any]:
        """보안 검증"""
        logger.info("🔒 보안 검증 시작...")
        
        security_validation = {
            'api_key_exposure': [],
            'hardcoded_secrets': [],
            'input_validation': [],
            'authentication_methods': [],
            'encryption_usage': [],
            'security_headers': [],
            'suggestions': []
        }
        
        python_files = list(self.root_path.rglob("*.py"))
        sensitive_patterns = [
            'api_key', 'secret', 'password', 'token', 'private_key'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 민감 정보 노출 검사
                for pattern in sensitive_patterns:
                    if f'{pattern} = ' in content.lower() and 'os.getenv' not in content.lower():
                        security_validation['hardcoded_secrets'].append({
                            'file': str(py_file.relative_to(self.root_path)),
                            'pattern': pattern
                        })
                
                # 입력 검증 패턴
                validation_patterns = ['validate', 'sanitize', 'escape', 'filter']
                for pattern in validation_patterns:
                    if pattern in content.lower():
                        security_validation['input_validation'].append({
                            'file': str(py_file.relative_to(self.root_path)),
                            'pattern': pattern
                        })
                
                # 인증 메서드
                auth_patterns = ['authenticate', 'authorize', 'jwt', 'oauth', 'login']
                for pattern in auth_patterns:
                    if pattern in content.lower():
                        security_validation['authentication_methods'].append({
                            'file': str(py_file.relative_to(self.root_path)),
                            'pattern': pattern
                        })
                        
            except Exception as e:
                logger.error(f"보안 분석 실패 {py_file}: {e}")
        
        # 제안사항
        if security_validation['hardcoded_secrets']:
            security_validation['suggestions'].append("🚨 하드코딩된 비밀 정보가 발견되었습니다. 환경 변수를 사용하세요.")
        
        if not security_validation['input_validation']:
            security_validation['suggestions'].append("⚠️ 입력 검증 로직을 추가하세요.")
        
        if not security_validation['authentication_methods']:
            security_validation['suggestions'].append("💡 인증 메커니즘을 구현하세요.")
        
        return security_validation
    
    def generate_requirements_txt(self) -> Dict[str, Any]:
        """requirements.txt 생성 및 검증"""
        logger.info("📦 Requirements 분석 시작...")
        
        requirements_analysis = {
            'current_requirements': [],
            'detected_imports': set(),
            'missing_packages': [],
            'unused_packages': [],
            'version_conflicts': [],
            'suggestions': []
        }
        
        # 현재 requirements.txt 읽기
        req_file = self.root_path / 'requirements.txt'
        if req_file.exists():
            with open(req_file, 'r') as f:
                requirements_analysis['current_requirements'] = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # 코드에서 임포트 추출
        python_files = list(self.root_path.rglob("*.py"))
        common_packages = {
            'ccxt': 'ccxt',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'aiohttp': 'aiohttp',
            'asyncio': '',  # 내장 모듈
            'requests': 'requests',
            'psutil': 'psutil',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'redis': 'redis',
            'sqlalchemy': 'sqlalchemy',
            'python-dotenv': 'python-dotenv'
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                
                for imp in imports:
                    if imp in common_packages and common_packages[imp]:
                        requirements_analysis['detected_imports'].add(common_packages[imp])
                        
            except:
                continue
        
        # 누락된 패키지 확인
        current_packages = {req.split('==')[0].split('>=')[0].split('<=')[0] for req in requirements_analysis['current_requirements']}
        for detected in requirements_analysis['detected_imports']:
            if detected not in current_packages:
                requirements_analysis['missing_packages'].append(detected)
        
        # 제안사항
        if requirements_analysis['missing_packages']:
            requirements_analysis['suggestions'].append(f"📦 누락된 패키지 {len(requirements_analysis['missing_packages'])}개를 requirements.txt에 추가하세요.")
        
        requirements_analysis['detected_imports'] = list(requirements_analysis['detected_imports'])
        
        return requirements_analysis
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """종합 검증 실행"""
        logger.info("🚀 VPS Deployment 종합 검증 시작...")
        
        start_time = time.time()
        
        # 1. 임포트 분석
        self.results['validation_results']['import_analysis'] = self.analyze_imports()
        
        # 2. 디버깅 시스템
        self.results['validation_results']['debugging_system'] = self.validate_debugging_system()
        
        # 3. 성능 분석
        self.results['validation_results']['performance_analysis'] = self.analyze_performance()
        
        # 4. 엔드포인트 검증
        self.results['validation_results']['endpoint_validation'] = await self.validate_endpoints()
        
        # 5. 에러 핸들링
        self.results['validation_results']['error_handling'] = self.validate_error_handling()
        
        # 6. 보안 검증
        self.results['validation_results']['security_validation'] = self.validate_security()
        
        # 7. Requirements 분석
        self.results['validation_results']['requirements_analysis'] = self.generate_requirements_txt()
        
        # 실행 시간
        self.results['execution_time_seconds'] = time.time() - start_time
        
        # 종합 점수 계산
        self.results['overall_score'] = self._calculate_overall_score()
        
        return self.results
    
    def _calculate_overall_score(self) -> Dict[str, Any]:
        """종합 점수 계산"""
        scores = {
            'import_health': 0,
            'debugging_readiness': 0,
            'performance_efficiency': 0,
            'api_reliability': 0,
            'error_resilience': 0,
            'security_strength': 0,
            'overall_rating': 'F'
        }
        
        # 임포트 건전성 (20점)
        import_result = self.results['validation_results']['import_analysis']
        if not import_result['circular_imports']:
            scores['import_health'] += 10
        if len(import_result['missing_imports']) < 3:
            scores['import_health'] += 10
        
        # 디버깅 준비도 (15점)
        debug_result = self.results['validation_results']['debugging_system']
        if debug_result['logging_setup']:
            scores['debugging_readiness'] += 8
        if len(debug_result['error_handling']) > 5:
            scores['debugging_readiness'] += 7
        
        # 성능 효율성 (15점)
        perf_result = self.results['validation_results']['performance_analysis']
        if perf_result['memory_usage']['rss_mb'] < 300:
            scores['performance_efficiency'] += 8
        if len(perf_result['async_patterns']) > 3:
            scores['performance_efficiency'] += 7
        
        # API 신뢰성 (15점)
        api_result = self.results['validation_results']['endpoint_validation']
        if len(api_result['endpoints_found']) > 0:
            scores['api_reliability'] += 8
        if len(api_result['health_checks']) > 0:
            scores['api_reliability'] += 7
        
        # 에러 복원력 (20점)
        error_result = self.results['validation_results']['error_handling']
        if error_result['try_catch_blocks'] > 10:
            scores['error_resilience'] += 10
        if error_result['logging_in_errors'] > 5:
            scores['error_resilience'] += 10
        
        # 보안 강도 (15점)
        security_result = self.results['validation_results']['security_validation']
        if not security_result['hardcoded_secrets']:
            scores['security_strength'] += 8
        if len(security_result['input_validation']) > 0:
            scores['security_strength'] += 7
        
        # 전체 점수 (100점 만점)
        total_score = sum(scores.values())
        
        if total_score >= 90:
            scores['overall_rating'] = 'A+'
        elif total_score >= 80:
            scores['overall_rating'] = 'A'
        elif total_score >= 70:
            scores['overall_rating'] = 'B'
        elif total_score >= 60:
            scores['overall_rating'] = 'C'
        elif total_score >= 50:
            scores['overall_rating'] = 'D'
        else:
            scores['overall_rating'] = 'F'
        
        scores['total_score'] = total_score
        
        return scores
    
    def generate_report(self) -> str:
        """검증 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("🔍 VPS DEPLOYMENT 시스템 검증 리포트")
        report_lines.append("=" * 60)
        report_lines.append(f"검증 시간: {self.results['timestamp']}")
        report_lines.append(f"실행 시간: {self.results['execution_time_seconds']:.2f}초")
        report_lines.append("")
        
        # 종합 점수
        score = self.results['overall_score']
        report_lines.append(f"📊 종합 점수: {score['total_score']}/100 ({score['overall_rating']})")
        report_lines.append("")
        
        # 각 영역별 결과
        results = self.results['validation_results']
        
        # 1. 임포트 분석
        import_result = results['import_analysis']
        report_lines.append("1. 📦 임포트 분석")
        report_lines.append(f"   - Python 파일: {len(import_result['python_files'])}개")
        report_lines.append(f"   - 외부 의존성: {len(import_result['external_dependencies'])}개")
        report_lines.append(f"   - 내부 의존성: {len(import_result['internal_dependencies'])}개")
        report_lines.append(f"   - 순환 임포트: {len(import_result['circular_imports'])}개")
        report_lines.append(f"   - 누락된 임포트: {len(import_result['missing_imports'])}개")
        report_lines.append("")
        
        # 2. 디버깅 시스템
        debug_result = results['debugging_system']
        report_lines.append("2. 🐛 디버깅 시스템")
        report_lines.append(f"   - 로깅 설정: {'✅' if debug_result['logging_setup'] else '❌'}")
        report_lines.append(f"   - 에러 핸들링 파일: {len(debug_result['error_handling'])}개")
        report_lines.append("")
        
        # 3. 성능 분석
        perf_result = results['performance_analysis']
        report_lines.append("3. ⚡ 성능 분석")
        report_lines.append(f"   - 메모리 사용량: {perf_result['memory_usage']['rss_mb']:.1f}MB")
        report_lines.append(f"   - CPU 사용률: {perf_result['cpu_usage']['percent']:.1f}%")
        report_lines.append(f"   - 비동기 패턴: {len(perf_result['async_patterns'])}개 파일")
        report_lines.append("")
        
        # 4. API 검증
        api_result = results['endpoint_validation']
        report_lines.append("4. 🌐 API 엔드포인트")
        report_lines.append(f"   - API 파일: {len(api_result['api_files'])}개")
        report_lines.append(f"   - 엔드포인트 발견: {len(api_result['endpoints_found'])}개")
        report_lines.append(f"   - 헬스체크: {len(api_result['health_checks'])}개")
        report_lines.append("")
        
        # 5. 에러 핸들링
        error_result = results['error_handling']
        report_lines.append("5. 🚨 에러 핸들링")
        report_lines.append(f"   - Try-Catch 블록: {error_result['try_catch_blocks']}개")
        report_lines.append(f"   - 에러 로깅: {error_result['logging_in_errors']}개")
        report_lines.append(f"   - 처리하는 에러 타입: {len(error_result['error_types_handled'])}개")
        report_lines.append("")
        
        # 6. 보안 검증
        security_result = results['security_validation']
        report_lines.append("6. 🔒 보안 검증")
        report_lines.append(f"   - 하드코딩된 비밀정보: {len(security_result['hardcoded_secrets'])}개")
        report_lines.append(f"   - 입력 검증: {len(security_result['input_validation'])}개")
        report_lines.append(f"   - 인증 메서드: {len(security_result['authentication_methods'])}개")
        report_lines.append("")
        
        # 제안사항
        report_lines.append("💡 주요 개선사항:")
        all_suggestions = []
        for category, result in results.items():
            if 'suggestions' in result:
                all_suggestions.extend(result['suggestions'])
        
        for i, suggestion in enumerate(all_suggestions[:10], 1):  # 상위 10개만
            report_lines.append(f"   {i}. {suggestion}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


async def main():
    """메인 실행 함수"""
    validator = SystemValidator()
    
    print("🚀 VPS Deployment 시스템 종합 검증 시작...")
    print("이 작업은 몇 분이 소요될 수 있습니다...\n")
    
    try:
        # 종합 검증 실행
        results = await validator.run_comprehensive_validation()
        
        # 리포트 생성
        report = validator.generate_report()
        print(report)
        
        # 결과 파일 저장
        with open('system_validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open('system_validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 상세 결과가 다음 파일에 저장되었습니다:")
        print(f"   - system_validation_results.json")
        print(f"   - system_validation_report.txt")
        
    except Exception as e:
        logger.error(f"검증 중 오류 발생: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())