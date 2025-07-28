# scenario_report_generator.py

import pandas as pd
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REPORT_DIR = "logs"

@dataclass
class ReportConfig:
    """리포트 설정 클래스"""
    report_dir: str = REPORT_DIR
    required_columns: List[str] = None
    aggregation_methods: Dict[str, str] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ["strategy", "total_return", "sharpe", "win_rate", "mdd", "trades"]
        if self.aggregation_methods is None:
            self.aggregation_methods = {
                "total_return": "mean",
                "sharpe": "mean", 
                "win_rate": "mean",
                "mdd": "mean",
                "trades": "sum"  # trades는 합계가 더 의미있음
            }

class ScenarioReportGenerator:
    def __init__(self, scenario_name: str, config: Optional[ReportConfig] = None):
        self.scenario_name = scenario_name
        self.config = config or ReportConfig()
        self.report_files = self._find_reports()

    def _find_reports(self) -> List[str]:
        """리포트 파일을 안전하게 검색합니다."""
        try:
            report_dir = Path(self.config.report_dir)
            if not report_dir.exists():
                logger.warning(f"Report directory does not exist: {report_dir}")
                return []
            
            pattern = f"scenario_result_{self.scenario_name}_*.csv"
            matched = list(report_dir.glob(pattern))
            matched_paths = [str(path) for path in matched]
            
            if not matched_paths:
                logger.warning(f"No report files found for scenario '{self.scenario_name}'")
            else:
                logger.info(f"Found {len(matched_paths)} report files for scenario '{self.scenario_name}'")
            
            return matched_paths
            
        except Exception as e:
            logger.error(f"Error finding reports: {e}")
            return []

    def _load_report_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """리포트 파일을 안전하게 로드합니다."""
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Empty report file: {file_path}")
                return None
            
            # 필수 컨럼 검증
            missing_cols = [col for col in self.config.required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                # 빠진 컨럼에 대한 기본값 설정
                for col in missing_cols:
                    if col in ["total_return", "sharpe", "win_rate", "mdd"]:
                        df[col] = 0.0
                    elif col == "trades":
                        df[col] = 0
                    elif col == "strategy":
                        df[col] = "unknown"
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading report file {file_path}: {e}")
            return None

    def generate_summary(self) -> pd.DataFrame:
        """개선된 요약 리포트 생성"""
        if not self.report_files:
            logger.warning("No report files to process")
            return pd.DataFrame()

        all_results = []
        failed_files = []
        
        for file_path in self.report_files:
            df = self._load_report_file(file_path)
            if df is not None:
                all_results.append(df)
            else:
                failed_files.append(file_path)

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        if not all_results:
            logger.error("No valid data to analyze")
            return pd.DataFrame()

        df_total = pd.concat(all_results, ignore_index=True)
        logger.info(f"Loaded {len(df_total)} total records from {len(all_results)} files")

        # 전략별 평균 성과 요약
        try:
            # 사용 가능한 컨럼만 집계
            available_agg_cols = {
                col: method for col, method in self.config.aggregation_methods.items() 
                if col in df_total.columns
            }
            
            if not available_agg_cols:
                logger.error("No aggregatable columns found")
                return pd.DataFrame()
            
            summary = df_total.groupby("strategy").agg(available_agg_cols).reset_index()
            
            # 성과순으로 정렬 (가능한 경우)
            sort_col = "total_return" if "total_return" in summary.columns else summary.columns[1]
            summary = summary.sort_values(by=sort_col, ascending=False)

            # 결과 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config.report_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = output_dir / f"scenario_summary_{self.scenario_name}_{timestamp}.csv"
            summary.to_csv(out_path, index=False)

            logger.info(f"Summary report saved: {out_path}")
            logger.info(f"Summary:\n{summary}")

            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return pd.DataFrame()

def validate_result_dataframe(result_df: pd.DataFrame) -> bool:
    """결과 데이터프레임 유효성 검증"""
    if result_df.empty:
        logger.error("Result DataFrame is empty")
        return False
    
    if "strategy" not in result_df.columns:
        logger.error("'strategy' column not found in result DataFrame")
        return False
    
    return True

def create_flexible_aggregation(result_df: pd.DataFrame) -> Dict[str, str]:
    """사용 가능한 컨럼에 따라 유연한 집계 설정 생성"""
    default_agg = {
        "total_return": "mean",
        "sharpe": "mean",
        "win_rate": "mean",
        "mdd": "mean",
        "trades": "sum",
        "score": "mean",  # 추가 가능한 컨럼
        "signal": "count"  # signal 횟수
    }
    
    # 사용 가능한 컨럼만 필터링
    available_agg = {
        col: method for col, method in default_agg.items() 
        if col in result_df.columns and col != "strategy"
    }
    
    if not available_agg:
        # 기본 대비책: 숫자 컨럼들에 대해 mean 적용
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        available_agg = {col: "mean" for col in numeric_cols if col != "strategy"}
    
    return available_agg

# ✅ 외부 모듈 사용을 위한 개선된 독립 함수
def generate_scenario_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    개선된 외부 모듈용 전략별 성과 요약 함수.
    - 데이터 유효성 검증 강화
    - 유연한 컨럼 처리
    - 상세한 로깅
    
    :param result_df: concat된 scenario result DataFrame
    :return: 요약된 DataFrame
    """
    try:
        if not validate_result_dataframe(result_df):
            return pd.DataFrame()
        
        logger.info(f"Generating summary for {len(result_df)} records")
        
        # 사용 가능한 컨럼에 따라 집계 설정 생성
        agg_config = create_flexible_aggregation(result_df)
        
        if not agg_config:
            logger.error("No numeric columns available for aggregation")
            return pd.DataFrame()
        
        logger.info(f"Using aggregation config: {agg_config}")
        
        # 전략별 집계
        summary = result_df.groupby("strategy").agg(agg_config).reset_index()
        
        # 성과순 정렬 (가능한 경우)
        sort_columns = ["total_return", "score", "sharpe"]
        sort_col = None
        for col in sort_columns:
            if col in summary.columns:
                sort_col = col
                break
        
        if sort_col:
            summary = summary.sort_values(by=sort_col, ascending=False)
            logger.info(f"Summary sorted by '{sort_col}'")
        
        logger.info(f"Generated summary with {len(summary)} strategies")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating scenario summary: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # 단독 실행 테스트용
    logging.basicConfig(level=logging.INFO)
    
    # 기본 설정으로 테스트
    generator = ScenarioReportGenerator(scenario_name="trend_breakout")
    summary = generator.generate_summary()
    
    if not summary.empty:
        print("\n성공적으로 요약 리포트를 생성했습니다.")
    else:
        print("\n요약 리포트 생성에 실패했습니다.")
    
    # 커스텀 설정으로 테스트
    # custom_config = ReportConfig(
    #     report_dir="custom_logs",
    #     required_columns=["strategy", "score"],
    #     aggregation_methods={"score": "mean"}
    # )
    # custom_generator = ScenarioReportGenerator("custom_scenario", custom_config)
    # custom_generator.generate_summary()
