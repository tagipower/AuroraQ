#!/usr/bin/env python3
"""
VPS 배포 시스템 임포트 구조 수정 도구
상대 임포트를 절대 임포트로 변환
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImportFixer:
    """임포트 구조 수정 도구"""
    
    def __init__(self, vps_deployment_root: str):
        self.root = Path(vps_deployment_root)
        self.trading_dir = self.root / "trading"
        self.vps_logging_dir = self.root / "vps_logging"
        
        # 상대 임포트 패턴
        self.relative_import_patterns = [
            # ..vps_logging import
            (r'from \.\.vps_logging import (.+)', r'from vps_logging import \1'),
            
            # .module import (같은 디렉토리)
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import (.+)', r'from trading.\1 import \2'),
            
            # ..sentiment_service import
            (r'from \.\.sentiment-service\.(.+) import (.+)', r'from sentiment_service.\1 import \2'),
            
            # ..logging import
            (r'from \.\.logging\.(.+) import (.+)', r'from vps_logging.\1 import \2'),
        ]
        
        # 절대 경로 추가가 필요한 파일들
        self.path_insert_template = '''
# VPS 배포 시스템 경로 설정
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
'''
    
    def fix_file_imports(self, file_path: Path) -> bool:
        """단일 파일의 임포트 구조 수정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 경로 설정 코드가 없으면 추가
            if 'sys.path.insert' not in content and any(pattern[0] for pattern, _ in self.relative_import_patterns if re.search(pattern, content)):
                # 첫 번째 import 앞에 경로 설정 추가
                import_match = re.search(r'^(import |from )', content, re.MULTILINE)
                if import_match:
                    insert_pos = import_match.start()
                    content = content[:insert_pos] + self.path_insert_template + '\n' + content[insert_pos:]
                    modified = True
            
            # 상대 임포트를 절대 임포트로 변환
            for pattern, replacement in self.relative_import_patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
            
            # 파일이 수정되었으면 저장
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"✅ 수정 완료: {file_path.relative_to(self.root)}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 파일 수정 실패 {file_path}: {e}")
            return False
    
    def fix_trading_module(self) -> Dict[str, int]:
        """trading 모듈 전체 임포트 수정"""
        results = {
            'total_files': 0,
            'modified_files': 0,
            'failed_files': 0
        }
        
        # Python 파일 찾기
        python_files = list(self.trading_dir.glob('**/*.py'))
        results['total_files'] = len(python_files)
        
        logger.info(f"🔍 Trading 모듈 {len(python_files)}개 파일 검사 시작...")
        
        for py_file in python_files:
            try:
                if self.fix_file_imports(py_file):
                    results['modified_files'] += 1
            except Exception as e:
                logger.error(f"❌ {py_file}: {e}")
                results['failed_files'] += 1
        
        return results
    
    def create_init_files(self):
        """필요한 __init__.py 파일 생성"""
        init_content = '''"""
VPS 배포 시스템 모듈
"""

# 버전 정보
__version__ = "3.0.0"
__author__ = "AuroraQ Team"
'''
        
        # vps_logging/__init__.py
        vps_logging_init = self.vps_logging_dir / "__init__.py"
        if not vps_logging_init.exists():
            with open(vps_logging_init, 'w', encoding='utf-8') as f:
                f.write(init_content)
            logger.info(f"✅ 생성: {vps_logging_init.relative_to(self.root)}")
    
    def validate_imports(self) -> List[str]:
        """임포트 검증"""
        issues = []
        
        for py_file in self.trading_dir.glob('**/*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 여전히 상대 임포트가 있는지 확인
                for pattern, _ in self.relative_import_patterns:
                    if re.search(pattern, content):
                        issues.append(f"상대 임포트 발견: {py_file.relative_to(self.root)} - {pattern}")
                
                # 순환 임포트 가능성 체크
                if 'from trading.' in content and 'from vps_' in content:
                    issues.append(f"순환 임포트 가능성: {py_file.relative_to(self.root)}")
                        
            except Exception as e:
                issues.append(f"검증 오류: {py_file} - {e}")
        
        return issues
    
    def run_full_fix(self) -> Dict[str, any]:
        """전체 임포트 구조 수정 실행"""
        logger.info("🚀 VPS 배포 시스템 임포트 구조 수정 시작")
        
        # 1. __init__.py 파일 생성
        self.create_init_files()
        
        # 2. trading 모듈 수정
        trading_results = self.fix_trading_module()
        
        # 3. 검증
        issues = self.validate_imports()
        
        # 결과 요약
        results = {
            'trading_module': trading_results,
            'remaining_issues': len(issues),
            'issues_detail': issues[:10],  # 상위 10개만
            'success': trading_results['failed_files'] == 0 and len(issues) == 0
        }
        
        # 결과 출력
        logger.info("📊 임포트 구조 수정 결과:")
        logger.info(f"  📁 전체 파일: {trading_results['total_files']}개")
        logger.info(f"  ✅ 수정 완료: {trading_results['modified_files']}개")
        logger.info(f"  ❌ 실패: {trading_results['failed_files']}개")
        logger.info(f"  ⚠️ 남은 이슈: {len(issues)}개")
        
        if issues:
            logger.warning("남은 이슈들:")
            for issue in issues[:5]:
                logger.warning(f"  - {issue}")
        
        return results

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VPS 배포 시스템 임포트 구조 수정")
    parser.add_argument(
        "--vps-root", 
        default="C:/Users/경남교육청/Desktop/AuroraQ/vps-deployment",
        help="VPS 배포 루트 디렉토리"
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 수정 없이 검사만")
    
    args = parser.parse_args()
    
    fixer = ImportFixer(args.vps_root)
    
    if args.dry_run:
        # 검증만 실행
        issues = fixer.validate_imports()
        print(f"발견된 이슈: {len(issues)}개")
        for issue in issues:
            print(f"  - {issue}")
    else:
        # 전체 수정 실행
        results = fixer.run_full_fix()
        
        if results['success']:
            print("🎉 임포트 구조 수정 성공!")
        else:
            print("⚠️ 일부 이슈가 남아있습니다. 수동 확인이 필요합니다.")

if __name__ == "__main__":
    main()