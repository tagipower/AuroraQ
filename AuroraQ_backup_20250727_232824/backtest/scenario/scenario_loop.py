# backtest/scenario_loop.py

import os
import yaml
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from backtest.scenario.scenario_extractor import extract_scenario_windows
from backtest.scenario.scenario_runner import run_all_segments
from backtest.scenario.scenario_report_generator import generate_scenario_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScenarioLoopConfig:
    """시나리오 루프 설정 클래스"""
    scenario_def_path: str = "backtest/scenario/scenario_definitions.yaml"
    data_path: str = "data/merged_data.csv"
    model_path: Optional[str] = "models/ppo_latest.zip"
    log_dir: str = "logs"
    
    def __post_init__(self):
        # 경로 검증
        if not Path(self.scenario_def_path).exists():
            raise FileNotFoundError(f"Scenario definition file not found: {self.scenario_def_path}")
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if self.model_path and not Path(self.model_path).exists():
            logger.warning(f"Model file not found: {self.model_path}")
        
        # 로그 디렉토리 생성
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


def load_scenario_definitions(file_path: str) -> List[Dict]:
    """시나리오 정의 YAML 파일을 안전하게 불러옵니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scenarios = yaml.safe_load(f)
        
        if not isinstance(scenarios, list):
            raise ValueError("Scenario definitions must be a list")
        
        # 각 시나리오 검증
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                raise ValueError(f"Scenario {i} must be a dictionary")
            if 'name' not in scenario:
                raise ValueError(f"Scenario {i} missing required 'name' field")
            if 'conditions' not in scenario:
                logger.warning(f"Scenario {scenario.get('name', i)} missing 'conditions' field")
        
        logger.info(f"Loaded {len(scenarios)} scenario definitions")
        return scenarios
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load scenario definitions: {e}")
        raise

def load_data_safely(data_path: str) -> pd.DataFrame:
    """데이터 파일을 안전하게 로드하고 검증합니다."""
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded data: {len(df)} rows, columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        raise


def run_single_scenario(scenario: Dict, merged_df: pd.DataFrame, config: ScenarioLoopConfig) -> bool:
    """단일 시나리오를 실행하고 결과를 저장합니다."""
    name = scenario["name"]
    logger.info(f"🧪 시나리오 실행: {name}")
    
    try:
        # 시나리오 구간 추출
        segments = extract_scenario_windows(merged_df, scenario)
        
        if not segments:
            logger.warning(f"시나리오 '{name}'에 해당하는 구간이 없습니다.")
            return False
        
        logger.info(f"추출된 세그먼트 수: {len(segments)}")
        
        # 전략 실행
        result_df = run_all_segments(segments, model_path=config.model_path)
        
        if result_df.empty:
            logger.warning(f"시나리오 '{name}'의 실행 결과가 비어있습니다.")
            return False
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 메인 결과 저장
        filename = f"scenario_result_{name}_{timestamp}.csv"
        result_path = Path(config.log_dir) / filename
        result_df.to_csv(result_path, index=False)
        logger.info(f"📄 결과 저장됨: {result_path}")
        
        # 요약 리포트 생성 및 저장
        try:
            summary_df = generate_scenario_summary(result_df)
            if not summary_df.empty:
                summary_filename = f"scenario_summary_{name}_{timestamp}.csv"
                summary_path = Path(config.log_dir) / summary_filename
                summary_df.to_csv(summary_path, index=False)
                logger.info(f"📊 요약 저장됨: {summary_path}")
        except Exception as e:
            logger.error(f"요약 리포트 생성 실패: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"시나리오 '{name}' 실행 중 오류: {e}")
        return False

def run_scenario_loop(config: Optional[ScenarioLoopConfig] = None):
    """개선된 시나리오 루프 실행 함수"""
    if config is None:
        config = ScenarioLoopConfig()
    
    try:
        # 시나리오 정의 및 데이터 로드
        scenarios = load_scenario_definitions(config.scenario_def_path)
        merged_df = load_data_safely(config.data_path)
        
        success_count = 0
        total_count = len(scenarios)
        
        logger.info(f"총 {total_count}개 시나리오 실행 시작")
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n진행률: {i}/{total_count}")
            
            if run_single_scenario(scenario, merged_df, config):
                success_count += 1
        
        logger.info(f"\n✅ 시나리오 루프 완료: {success_count}/{total_count} 성공")
        
    except Exception as e:
        logger.error(f"시나리오 루프 실행 실패: {e}")
        raise


if __name__ == "__main__":
    # 기본 설정으로 실행
    run_scenario_loop()
    
    # 커스텀 설정으로 실행하는 예시:
    # config = ScenarioLoopConfig(
    #     scenario_def_path="custom_scenarios.yaml",
    #     data_path="data/custom_data.csv",
    #     model_path="models/custom_model.zip",
    #     log_dir="custom_logs"
    # )
    # run_scenario_loop(config)
