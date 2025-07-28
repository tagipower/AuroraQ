#!/usr/bin/env python3
"""
AuroraQ Project Cleanup Script
프로젝트 구조 정리 및 최적화
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import json

class ProjectCleaner:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root.parent / f"AuroraQ_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 삭제할 파일 패턴
        self.files_to_delete = [
            "test_optimized_rulestrategy.py",
            "run_backtest_direct.py", 
            "run_optimized_backtest.py",
            "simple_test.py",
            "full_test.py",
            "run_backtest.py",  # 루트의 임시 파일
        ]
        
        # 이동할 파일들
        self.files_to_move = {
            # 테스트 파일들을 tests 폴더로
            "backtest/v2/test_optimized_strategies.py": "tests/test_optimized_strategies.py",
            "backtest/v2/test_optimized_integration.py": "tests/test_optimized_integration.py",
            "backtest/v2/debug_signals.py": "tests/debug_signals.py",
            "backtest/v2/example_usage.py": "tests/example_usage.py",
            
            # 실행 스크립트들을 scripts 폴더로
            "backtest/v2/run_backtest.py": "scripts/run_backtest_v2.py",
            "backtest/v2/patch_rule_strategy_e.py": "scripts/patch_rule_strategy_e.py",
            "loops/run_loop.py": "scripts/run_main_loop.py",
            
            # v2 utils를 메인 backtest/utils로
            "backtest/v2/utils/html_report_generator.py": "backtest/utils/html_report_generator.py",
            "backtest/v2/utils/strategy_scores_generator.py": "backtest/utils/strategy_scores_generator.py",
        }
        
        # v2 시스템을 메인으로 통합
        self.v2_integration = {
            "backtest/v2/layers": "backtest/layers",
            "backtest/v2/integration": "backtest/integration", 
            "backtest/v2/strategies": "backtest/strategies",
            "backtest/v2/config": "backtest/config",
        }
    
    def create_backup(self):
        """전체 프로젝트 백업"""
        print(f"Creating backup at: {self.backup_dir}")
        shutil.copytree(self.project_root, self.backup_dir, 
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        print("Backup completed!")
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        dirs_to_create = [
            "tests",
            "scripts", 
            "backtest/utils",
            "backtest/strategies",
            "backtest/config",
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def delete_files(self):
        """불필요한 파일 삭제"""
        deleted_count = 0
        for file_name in self.files_to_delete:
            file_path = self.project_root / file_name
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted: {file_name}")
                deleted_count += 1
        
        # .backup 파일들 삭제
        for backup_file in self.project_root.rglob("*.backup"):
            backup_file.unlink()
            print(f"Deleted backup: {backup_file.name}")
            deleted_count += 1
            
        print(f"Total files deleted: {deleted_count}")
    
    def move_files(self):
        """파일 이동 및 재구성"""
        moved_count = 0
        for src, dst in self.files_to_move.items():
            src_path = self.project_root / src
            dst_path = self.project_root / dst
            
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                print(f"Moved: {src} -> {dst}")
                moved_count += 1
                
        print(f"Total files moved: {moved_count}")
    
    def integrate_v2(self):
        """v2 시스템을 메인으로 통합"""
        for src, dst in self.v2_integration.items():
            src_path = self.project_root / src
            dst_path = self.project_root / dst
            
            if src_path.exists():
                if dst_path.exists():
                    # 기존 폴더가 있으면 백업
                    backup_path = Path(str(dst_path) + "_old")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(str(dst_path), str(backup_path))
                    print(f"Backed up existing: {dst} -> {dst}_old")
                
                shutil.move(str(src_path), str(dst_path))
                print(f"Integrated: {src} -> {dst}")
    
    def copy_optimized_params(self):
        """최적화된 파라미터를 메인 config로 복사"""
        src = self.project_root / "config" / "optimized_rule_params.yaml"
        
        # 이미 있으면 스킵
        if src.exists():
            print("Optimized parameters already in main config")
            return
            
        # v2에서 복사
        v2_src = self.project_root / "backtest" / "v2" / "config" / "optimized_rule_params.yaml"
        if v2_src.exists():
            shutil.copy2(v2_src, src)
            print("Copied optimized parameters to main config")
    
    def consolidate_reports(self):
        """리포트 폴더 통합"""
        # reports와 report 폴더 통합
        report_dir = self.project_root / "report"
        reports_dir = self.project_root / "reports"
        
        if report_dir.exists() and reports_dir.exists():
            # report의 내용을 reports로 이동
            for item in report_dir.iterdir():
                if item.is_dir():
                    dst = reports_dir / item.name
                    if not dst.exists():
                        shutil.move(str(item), str(dst))
                        print(f"Moved report folder: {item.name}")
            
            # report 폴더 삭제
            shutil.rmtree(report_dir)
            print("Removed old report directory")
    
    def update_imports(self):
        """import 경로 업데이트를 위한 가이드 생성"""
        guide = {
            "import_updates": {
                "from backtest.v2.layers": "from backtest.layers",
                "from backtest.v2.integration": "from backtest.integration",
                "from backtest.v2.utils": "from backtest.utils",
                "from backtest.v2.strategies": "from backtest.strategies",
            },
            "file_relocations": self.files_to_move,
            "v2_integration": self.v2_integration
        }
        
        guide_path = self.project_root / "import_update_guide.json"
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        
        print(f"Import update guide saved to: {guide_path}")
    
    def cleanup_v2_folder(self):
        """v2 폴더 정리"""
        v2_path = self.project_root / "backtest" / "v2"
        if v2_path.exists():
            # reports 폴더만 남기고 나머지 삭제
            items_to_keep = ["reports", "README.md"]
            
            for item in v2_path.iterdir():
                if item.name not in items_to_keep:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"Removed from v2: {item.name}")
    
    def generate_summary(self):
        """정리 요약 생성"""
        summary = f"""
# AuroraQ Project Cleanup Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Backup Location
{self.backup_dir}

## Structure Changes
- Integrated v2 backtest system into main backtest folder
- Created tests/ folder for all test files  
- Created scripts/ folder for execution scripts
- Consolidated reports into single reports/ folder
- Moved optimized parameters to main config/

## Next Steps
1. Update import paths in Python files (see import_update_guide.json)
2. Test the main backtest system with: python scripts/run_backtest_v2.py
3. Verify optimized parameters are being used
4. Remove backup after confirming everything works

## Key Locations
- Main backtest system: backtest/
- Optimized parameters: config/optimized_rule_params.yaml
- Test files: tests/
- Execution scripts: scripts/
- Reports: reports/
"""
        
        summary_path = self.project_root / "CLEANUP_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\nCleanup summary saved to: {summary_path}")
        print(summary)
    
    def run(self):
        """전체 정리 프로세스 실행"""
        print("=== Starting AuroraQ Project Cleanup ===\n")
        
        # 1. 백업 생성
        self.create_backup()
        
        # 2. 디렉토리 생성
        print("\n--- Creating directories ---")
        self.create_directories()
        
        # 3. 파일 삭제
        print("\n--- Deleting unnecessary files ---")
        self.delete_files()
        
        # 4. 파일 이동
        print("\n--- Moving files ---")
        self.move_files()
        
        # 5. v2 통합
        print("\n--- Integrating v2 system ---")
        self.integrate_v2()
        
        # 6. 파라미터 복사
        print("\n--- Copying optimized parameters ---")
        self.copy_optimized_params()
        
        # 7. 리포트 통합
        print("\n--- Consolidating reports ---") 
        self.consolidate_reports()
        
        # 8. v2 폴더 정리
        print("\n--- Cleaning up v2 folder ---")
        self.cleanup_v2_folder()
        
        # 9. import 가이드 생성
        print("\n--- Generating import guide ---")
        self.update_imports()
        
        # 10. 요약 생성
        print("\n--- Generating summary ---")
        self.generate_summary()
        
        print("\n=== Cleanup Completed Successfully! ===")
        print(f"Backup saved at: {self.backup_dir}")
        print("Please check CLEANUP_SUMMARY.md for details")


if __name__ == "__main__":
    # 자동 실행 모드
    print("Starting AuroraQ project cleanup...")
    
    # 프로젝트 루트 경로
    project_root = r"C:\Users\경남교육청\Desktop\AuroraQ"
    
    # 정리 실행
    cleaner = ProjectCleaner(project_root)
    cleaner.run()