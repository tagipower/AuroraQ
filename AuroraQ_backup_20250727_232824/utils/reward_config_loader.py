# utils/reward_config_loader.py

import yaml

def load_reward_config(path="config/reward_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 기본값 지정 포함 전체 반환
    return {
        "fee_market_order": config.get("fee_rate", {}).get("market", 0.0004),
        "fee_limit_order": config.get("fee_rate", {}).get("limit", 0.0002),
        "fee_type": config.get("fee_type", "market"),
        "max_holding": config.get("reward", {}).get("max_holding", 30),
        "over_hold_penalty": config.get("reward", {}).get("over_hold_penalty", 2.0),
        "use_sentiment_reward": config.get("reward", {}).get("use_sentiment_reward", True),
        "sentiment_risk_penalty": config.get("reward", {}).get("sentiment_risk_penalty", True),
        "sentiment_weight": config.get("reward", {}).get("sentiment_weight", 3.0),
        "event_penalty": config.get("reward", {}).get("event_penalty", 1.5),
        "regime_penalty": config.get("reward", {}).get("regime_penalty", 1.0),
        "trend_mismatch_penalty": config.get("reward", {}).get("trend_mismatch_penalty", 1.0),
        "hold_reward_ratio": config.get("reward", {}).get("hold_reward_ratio", 0.01),
        "use_limit_order": config.get("reward", {}).get("use_limit_order", False),
        
        # ✅ 로그 관련 통합
        "enable_reward_logging": config.get("logging", {}).get("enable_reward_logging", False),
        "reward_log_path": config.get("logging", {}).get("reward_log_path", "logs/reward_log.csv")
    }
