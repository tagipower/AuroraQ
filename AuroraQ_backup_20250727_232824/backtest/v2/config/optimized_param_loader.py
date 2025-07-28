"""
Optimized Parameter Loader
백테스트 v2에서 최적화된 전략 파라미터를 로드하는 유틸리티
"""

import yaml
import os
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("OptimizedParamLoader")


def get_optimized_rule_params(rule_name: str) -> Dict[str, Any]:
    """
    최적화된 전략 파라미터 로드
    
    Args:
        rule_name: 전략 이름 (예: "RuleE")
    
    Returns:
        최적화된 파라미터 딕셔너리
    """
    # 최적화된 파라미터 파일 경로
    optimized_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "..", 
        "config", 
        "optimized_rule_params.yaml"
    )
    
    # 백테스트 v2 설정 파일 경로
    v2_optimized_path = os.path.join(
        os.path.dirname(__file__), 
        "optimized_rule_params.yaml"
    )
    
    # 기본 파라미터 파일 경로 (fallback)
    default_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "..", 
        "config", 
        "rule_params.yaml"
    )
    
    # 1. 최적화된 파라미터 파일 우선 시도
    for config_path in [v2_optimized_path, optimized_path]:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                
                if rule_name in config:
                    logger.info(f"최적화된 파라미터 로드 성공: {config_path}")
                    params = config[rule_name]
                    
                    # 성능 타겟 정보도 포함
                    if "performance_targets" in config:
                        params["performance_targets"] = config["performance_targets"]
                    
                    return params
                    
            except Exception as e:
                logger.warning(f"최적화된 파라미터 파일 로드 실패 ({config_path}): {e}")
                continue
    
    # 2. 기본 파라미터 파일로 fallback
    if os.path.exists(default_path):
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            logger.warning(f"기본 파라미터 파일 사용: {default_path}")
            return config.get(rule_name, {})
            
        except Exception as e:
            logger.error(f"기본 파라미터 파일 로드 실패: {e}")
    
    # 3. 모든 파일 로드 실패 시 기본값 반환
    logger.error(f"모든 파라미터 파일 로드 실패 - 기본값 사용")
    return get_default_params(rule_name)


def get_default_params(rule_name: str) -> Dict[str, Any]:
    """
    기본 파라미터 반환 (파일 로드 실패 시 사용)
    """
    if rule_name == "RuleE":
        return {
            # 기본 브레이크아웃 설정
            "breakout_window_short": 12,
            "breakout_window_medium": 20,
            "breakout_window_long": 35,
            "breakout_buffer": 0.002,
            
            # 기본 레인지/변동성 설정
            "range_window": 10,
            "range_std_threshold": 0.0015,
            "range_atr_threshold": 0.003,
            "squeeze_window": 30,
            "expansion_ratio": 2.2,
            
            # 기본 거래량 설정
            "volume_window": 10,
            "volume_spike_ratio": 2.0,
            "volume_ma_window": 20,
            "obv_window": 15,
            "min_volume_ratio": 1.5,
            
            # 기본 RSI 설정
            "rsi_window": 14,
            "rsi_breakout_threshold": 60,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            
            # 기본 MACD 설정
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "macd_hist_threshold": 0.0001,
            
            # 기본 모멘텀 설정
            "momentum_window": 8,
            "momentum_threshold": 0.008,
            "min_confidence": 0.8,
            
            # 기본 리스크 관리
            "take_profit_pct": 0.025,
            "vol_multiplier_tp": 2.0,
            "stop_loss_pct": 0.012,
            "vol_multiplier_sl": 1.5,
            
            # 기본 포지션 관리
            "max_hold_bars": 8,
            "trailing_stop_activation": 0.012,
            "trailing_stop_distance": 0.006,
            
            # 기본 필터 설정
            "support_resistance_window": 120,
            "sr_touch_threshold": 0.0005,
            "trend_filter_window": 60,
            
            # 기본 재진입 설정
            "enable_reentry": False,
            "reentry_cooldown": 900,
            "max_reentries": 1,
            
            # 기본 추가 필터
            "require_higher_tf_trend": True,
            "higher_tf_period": 100,
            "enable_regime_filter": True,
            "regime_window": 50,
            "min_trend_strength": 0.3,
            "max_volatility_threshold": 0.08,
            "min_volatility_threshold": 0.005,
            "avoid_market_open_minutes": 30,
            "avoid_market_close_minutes": 30
        }
    
    # 다른 전략들의 기본값도 필요시 추가
    return {}


def copy_optimized_params_to_v2():
    """
    최적화된 파라미터를 backtest/v2/config/로 복사
    """
    import shutil
    
    source_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "..", 
        "config", 
        "optimized_rule_params.yaml"
    )
    
    target_path = os.path.join(
        os.path.dirname(__file__), 
        "optimized_rule_params.yaml"
    )
    
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"최적화된 파라미터 복사 완료: {target_path}")
            return True
        except Exception as e:
            logger.error(f"파라미터 복사 실패: {e}")
            return False
    else:
        logger.warning(f"소스 파일 없음: {source_path}")
        return False


if __name__ == "__main__":
    # 테스트
    copy_optimized_params_to_v2()
    params = get_optimized_rule_params("RuleE")
    print(f"로드된 파라미터: {len(params)}개")
    print(f"주요 파라미터:")
    print(f"  - breakout_window_short: {params.get('breakout_window_short')}")
    print(f"  - rsi_breakout_threshold: {params.get('rsi_breakout_threshold')}")
    print(f"  - take_profit_pct: {params.get('take_profit_pct')}")
    print(f"  - stop_loss_pct: {params.get('stop_loss_pct')}")