#!/usr/bin/env python3
"""
Patch RuleStrategyE to use optimized parameters
최적화된 파라미터를 사용하도록 RuleStrategyE 패치
"""

import os
import sys
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def patch_rule_param_loader():
    """
    rule_param_loader.py를 패치하여 최적화된 파라미터를 사용하도록 수정
    """
    # 원본 rule_param_loader.py 경로
    original_path = os.path.join(project_root, 'config', 'rule_param_loader.py')
    
    # 백업 생성
    backup_path = original_path + '.backup'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(original_path, backup_path)
        print(f"백업 생성: {backup_path}")
    
    # 새로운 내용 작성
    new_content = '''import yaml
import os

def get_rule_params(rule_name: str):
    """최적화된 파라미터를 우선적으로 로드"""
    
    # 최적화된 파라미터 파일 경로
    optimized_path = os.path.join("config", "optimized_rule_params.yaml")
    
    # 1. 최적화된 파라미터 파일 시도
    if os.path.exists(optimized_path):
        try:
            with open(optimized_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if rule_name in config:
                print(f"[INFO] {rule_name} - 최적화된 파라미터 사용")
                return config.get(rule_name, {})
        except Exception as e:
            print(f"[WARNING] 최적화된 파라미터 로드 실패: {e}")
    
    # 2. 기본 파라미터 파일로 fallback
    default_path = os.path.join("config", "rule_params.yaml")
    
    # ✅ 인코딩 명시적으로 지정
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"[INFO] {rule_name} - 기본 파라미터 사용")
    return config.get(rule_name, {})
'''
    
    # 파일 덮어쓰기
    with open(original_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"패치 완료: {original_path}")
    print("이제 RuleStrategyE가 최적화된 파라미터를 자동으로 사용합니다.")
    
    return True

def restore_original():
    """원본 파일 복원"""
    original_path = os.path.join(project_root, 'config', 'rule_param_loader.py')
    backup_path = original_path + '.backup'
    
    if os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, original_path)
        print(f"원본 파일 복원됨: {original_path}")
        return True
    else:
        print("백업 파일이 없습니다.")
        return False

def verify_optimized_params():
    """최적화된 파라미터 파일 확인"""
    optimized_path = os.path.join(project_root, 'config', 'optimized_rule_params.yaml')
    
    if not os.path.exists(optimized_path):
        print(f"❌ 최적화된 파라미터 파일이 없습니다: {optimized_path}")
        
        # backtest/v2/config에서 복사 시도
        v2_path = os.path.join(project_root, 'backtest', 'v2', 'config', 'optimized_rule_params.yaml')
        if os.path.exists(v2_path):
            import shutil
            shutil.copy2(v2_path, optimized_path)
            print(f"✅ 최적화된 파라미터 파일 복사 완료: {optimized_path}")
            return True
        else:
            print(f"❌ 소스 파일도 없습니다: {v2_path}")
            return False
    else:
        print(f"✅ 최적화된 파라미터 파일 존재: {optimized_path}")
        
        # 파일 내용 확인
        with open(optimized_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'RuleE' in config:
            print(f"✅ RuleE 파라미터 확인됨")
            print(f"   - take_profit_pct: {config['RuleE'].get('take_profit_pct', 'N/A')}")
            print(f"   - stop_loss_pct: {config['RuleE'].get('stop_loss_pct', 'N/A')}")
            print(f"   - rsi_breakout_threshold: {config['RuleE'].get('rsi_breakout_threshold', 'N/A')}")
            return True
        else:
            print("❌ RuleE 파라미터가 없습니다.")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RuleStrategyE 최적화 파라미터 패치')
    parser.add_argument('--restore', action='store_true', help='원본 파일로 복원')
    parser.add_argument('--verify', action='store_true', help='파라미터 파일 확인만')
    
    args = parser.parse_args()
    
    if args.restore:
        restore_original()
    elif args.verify:
        verify_optimized_params()
    else:
        # 파라미터 파일 확인
        if verify_optimized_params():
            # 패치 적용
            if patch_rule_param_loader():
                print("\n✅ 패치가 성공적으로 적용되었습니다!")
                print("이제 기존 전략 시스템에서도 최적화된 파라미터를 사용합니다.")
        else:
            print("\n❌ 최적화된 파라미터 파일이 필요합니다.")