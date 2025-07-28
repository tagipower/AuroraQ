# env_router.py

from envs.sentiment_trading_env import SentimentTradingEnv
from envs.rule_strategy_a_env import RuleStrategyAEnv
from envs.rule_strategy_b_env import RuleStrategyBEnv
from envs.rule_strategy_c_env import RuleStrategyCEnv
from envs.rule_strategy_d_env import RuleStrategyDEnv
from envs.rule_strategy_e_env import RuleStrategyEEnv

ENV_CLASSES = {
    "sentiment": SentimentTradingEnv,
    "rule_a": RuleStrategyAEnv,
    "rule_b": RuleStrategyBEnv,
    "rule_c": RuleStrategyCEnv,
    "rule_d": RuleStrategyDEnv,
    "rule_e": RuleStrategyEEnv,
}


def get_env(env_name, data, reward_calculator, config, window_size=10):
    """
    환경 이름에 따라 해당 TradingEnv 인스턴스를 반환합니다.
    
    Args:
        env_name (str): 환경 이름 (예: 'sentiment', 'rule_a' 등)
        data (pd.DataFrame): 가격 및 감정점수 등 입력 데이터
        reward_calculator (RewardCalculator): 보상 계산기 인스턴스
        config (dict): 설정 값
        window_size (int): 관측 윈도우 크기

    Returns:
        gym.Env: PPO 학습 또는 시뮬레이션용 환경 인스턴스
    """
    if env_name not in ENV_CLASSES:
        raise ValueError(f"❌ 알 수 없는 환경 이름입니다: {env_name} (가능한 값: {list(ENV_CLASSES.keys())})")

    EnvClass = ENV_CLASSES[env_name]

    return EnvClass(
        price_data=data,
        window_size=window_size,
        reward_calculator=reward_calculator,
        config=config
    )
