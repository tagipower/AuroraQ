# report/strategy_param_logger.py

import os
import yaml
import datetime

def log_strategy_params(strategy_name, param_dict, output_dir="logs/strategy_params"):
    """
    전략 실행에 사용된 파라미터를 yaml 형식으로 저장.
    파일명은 전략명_날짜시간.yaml 형식으로 생성됨
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy_name}_{timestamp}.yaml"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w') as f:
            yaml.dump({strategy_name: param_dict}, f, allow_unicode=True)
        print(f"✅ 전략 파라미터 저장 완료: {filepath}")
    except Exception as e:
        print(f"⚠️ 전략 파라미터 저장 실패: {e}")


def load_logged_params(filepath):
    """
    저장된 전략 파라미터 파일을 불러오는 함수 (선택적)
    """
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ 파라미터 로딩 실패: {e}")
        return None
