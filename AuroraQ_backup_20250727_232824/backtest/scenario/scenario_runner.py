# scenario_runner.py (리팩토링)
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from core.strategy_selector import StrategySelector
from core.strategy_score_manager import get_all_current_scores
from utils.logger import get_logger

logger = get_logger("ScenarioRunner")

@dataclass
class SegmentResult:
    """세그먼트 실행 결과"""
    timestamp: Any
    strategy: str
    score: float
    signal: str
    segment_index: int
    execution_time: float = 0.0
    error: Optional[str] = None

@dataclass 
class RunnerConfig:
    """러너 설정"""
    window_start: int = 50
    max_workers: int = 4
    sentiment_file: str = "data/sentiment/news_sentiment_log.csv"
    timeout_seconds: int = 30

def validate_segment_data(price_segment: pd.DataFrame, sentiment_segment: pd.DataFrame) -> bool:
    """세그먼트 데이터 유효성 검증"""
    if price_segment.empty:
        logger.error("Price segment is empty")
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_columns if col not in price_segment.columns]
    if missing_cols:
        logger.error(f"Missing required columns in price segment: {missing_cols}")
        return False
    
    return True

def create_window_data(price_segment: pd.DataFrame, sentiment_segment: pd.DataFrame, i: int) -> Dict[str, Any]:
    """윈도우 데이터 생성"""
    try:
        regime_data = sentiment_segment.get("regime_score", ["neutral"] * i)
        regime = regime_data[-1] if isinstance(regime_data, list) and regime_data else "neutral"
        
        window = {
            "close": price_segment["close"].iloc[:i].tolist(),
            "high": price_segment["high"].iloc[:i].tolist(),
            "low": price_segment["low"].iloc[:i].tolist(),
            "open": price_segment["open"].iloc[:i].tolist(),
            "volume": price_segment["volume"].iloc[:i].tolist(),
            "timestamp": price_segment.index[:i].tolist(),
            "regime": regime,
            "news_text": None  # 필요 시 감정 모듈 연동
        }
        return window
    except Exception as e:
        logger.error(f"Error creating window data for index {i}: {e}")
        return None

def run_backtest_for_segment(price_segment: pd.DataFrame, sentiment_segment: pd.DataFrame, 
                           model_path: Optional[str] = None, config: Optional[RunnerConfig] = None) -> pd.DataFrame:
    """
    개선된 세그먼트 백테스트 실행
    - 데이터 검증 추가
    - 오류 처리 개선
    - 성능 모니터링
    """
    if config is None:
        config = RunnerConfig()
    
    if not validate_segment_data(price_segment, sentiment_segment):
        return pd.DataFrame()
    
    start_time = time.time()
    selector = StrategySelector(sentiment_file=config.sentiment_file)
    results = []
    
    total_iterations = len(price_segment) - config.window_start
    if total_iterations <= 0:
        logger.warning(f"Segment too small for analysis: {len(price_segment)} rows")
        return pd.DataFrame()
    
    logger.info(f"Starting segment analysis: {total_iterations} iterations")
    
    success_count = 0
    error_count = 0
    
    for i in range(config.window_start, len(price_segment)):
        try:
            window = create_window_data(price_segment, sentiment_segment, i)
            if window is None:
                error_count += 1
                continue
            
            outcome = selector.select(window)
            results.append({
                "timestamp": window["timestamp"][-1],
                "strategy": outcome["strategy"],
                "score": outcome["score"], 
                "signal": outcome["signal"]
            })
            success_count += 1
            
        except Exception as e:
            error_count += 1
            logger.debug(f"[ScenarioRunner] 루프 {i} 스킵: {e}")
            continue
    
    execution_time = time.time() - start_time
    logger.info(f"Segment analysis complete: {success_count} success, {error_count} errors, {execution_time:.2f}s")
    
    return pd.DataFrame(results)

def run_segment_parallel(segment_data: tuple) -> pd.DataFrame:
    """병렬 처리를 위한 세그먼트 실행 함수"""
    i, seg_df, model_path, config = segment_data
    
    try:
        logger.info(f"📦 Segment {i+1} 실행: {seg_df.index[0]} → {seg_df.index[-1]}")
        
        # 데이터 추출
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        sentiment_cols = ['sentiment_score', 'regime_score']
        
        available_price_cols = [col for col in price_cols if col in seg_df.columns]
        available_sentiment_cols = [col for col in sentiment_cols if col in seg_df.columns]
        
        if not available_price_cols:
            logger.error(f"Segment {i+1}: No price columns available")
            return pd.DataFrame()
        
        price = seg_df[available_price_cols].copy()
        sentiment = seg_df[available_sentiment_cols].copy() if available_sentiment_cols else pd.DataFrame()
        
        # 세그먼트 실행
        seg_result = run_backtest_for_segment(price, sentiment, model_path, config)
        seg_result['segment_index'] = i
        
        return seg_result
        
    except Exception as e:
        logger.error(f"Segment {i+1} 실행 실패: {e}")
        return pd.DataFrame()

def run_all_segments(segments: List[pd.DataFrame], model_path: Optional[str] = None, 
                    config: Optional[RunnerConfig] = None, parallel: bool = False) -> pd.DataFrame:
    """
    개선된 여러 세그먼트 실행
    - 병렬 처리 옵션 추가
    - 개선된 오류 처리
    - 성능 모니터링
    """
    if config is None:
        config = RunnerConfig()
    
    if not segments:
        logger.warning("No segments to process")
        return pd.DataFrame()
    
    start_time = time.time()
    logger.info(f"Starting {len(segments)} segments ({'parallel' if parallel else 'sequential'})")
    
    all_results = []
    
    if parallel and len(segments) > 1:
        # 병렬 처리
        segment_data = [(i, seg_df, model_path, config) for i, seg_df in enumerate(segments)]
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_segment = {
                executor.submit(run_segment_parallel, data): i 
                for i, data in enumerate(segment_data)
            }
            
            for future in as_completed(future_to_segment, timeout=config.timeout_seconds):
                segment_idx = future_to_segment[future]
                try:
                    result = future.result()
                    if not result.empty:
                        all_results.append(result)
                    logger.info(f"Segment {segment_idx+1} completed")
                except Exception as e:
                    logger.error(f"Segment {segment_idx+1} failed: {e}")
    else:
        # 순차 처리
        for i, seg_df in enumerate(segments):
            try:
                logger.info(f"\n📦 Segment {i+1}/{len(segments)} 실행: {seg_df.index[0]} → {seg_df.index[-1]}")
                
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                sentiment_cols = ['sentiment_score', 'regime_score']
                
                available_price_cols = [col for col in price_cols if col in seg_df.columns]
                available_sentiment_cols = [col for col in sentiment_cols if col in seg_df.columns]
                
                if not available_price_cols:
                    logger.warning(f"Segment {i+1}: No price columns, skipping")
                    continue
                
                price = seg_df[available_price_cols].copy()
                sentiment = seg_df[available_sentiment_cols].copy() if available_sentiment_cols else pd.DataFrame()
                
                seg_result = run_backtest_for_segment(price, sentiment, model_path, config)
                if not seg_result.empty:
                    seg_result['segment_index'] = i
                    all_results.append(seg_result)
                    
            except Exception as e:
                logger.error(f"Segment {i+1} 실행 중 오류: {e}")
                continue
    
    # 결과 결합
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"Successfully processed {len(all_results)}/{len(segments)} segments")
    else:
        logger.warning("No successful segment results")
        combined = pd.DataFrame()
    
    # 최종 점수 요약
    try:
        final_scores = get_all_current_scores()
        logger.info(f"[ScenarioRunner] 모든 세그먼트 최종 점수: {final_scores}")
    except Exception as e:
        logger.warning(f"Failed to get final scores: {e}")
    
    execution_time = time.time() - start_time
    logger.info(f"All segments execution completed in {execution_time:.2f}s")
    
    return combined
