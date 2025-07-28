# backtest/reward_backtest.py

class RewardCalculator:
    def __init__(self, sentiment_weight: float = 1.0, time_penalty_factor: float = 0.05):
        """
        보상 계산기 (RewardCalculator)
        - 감정 점수와 포지션 유지 기간을 고려하여 보상을 계산
        - PPO 학습 및 백테스트 모두에서 사용 가능

        Args:
            sentiment_weight (float): 감정 점수의 가중치 (기본값 1.0)
            time_penalty_factor (float): 포지션 유지 기간에 따른 패널티 계수 (기본값 0.05)
        """
        self.sentiment_weight = sentiment_weight
        self.time_penalty_factor = time_penalty_factor

    def calculate(self, current_price: float, entry_price: float,
                  holding_period: int, sentiment_score: float = 1.0, **kwargs) -> float:
        """
        보상 계산 함수 (추가 인자 허용)
        - 가격 수익률, 감정 점수, 포지션 유지 기간을 종합해 보상 산출
        - 불필요한 인자는 무시 (예: price, close 등)

        Args:
            current_price (float): 현재 가격
            entry_price (float): 진입 가격
            holding_period (int): 포지션 보유 기간 (시간 단위)
            sentiment_score (float): 감정 점수 (기본값 1.0)
            **kwargs: 불필요한 추가 인자들 (무시됨)

        Returns:
            float: 계산된 보상 값
        """
        if entry_price == 0:
            return 0.0

        # 기본 수익률 (진입 대비 변화율)
        base_return = (current_price - entry_price) / entry_price

        # 포지션 유지 기간에 따른 패널티
        time_penalty = 1 + self.time_penalty_factor * holding_period

        # 감정 점수 보정 및 패널티 적용
        adjusted_reward = base_return * (sentiment_score ** self.sentiment_weight) / time_penalty

        return adjusted_reward

    def __call__(self, current_price: float = None, entry_price: float = None,
                 holding_period: int = 0, sentiment_score: float = 1.0, **kwargs) -> float:
        """
        RewardCalculator를 함수처럼 호출 가능하게 하는 래퍼
        env나 backtester에서 reward_fn 형태로 사용할 때 활용
        - **kwargs로 들어오는 'price', 'close' 등의 불필요한 인자를 안전하게 무시
        """
        # 기본값 처리 (혹은 fallback)
        current_price = kwargs.get('price', current_price)
        if current_price is None and 'close' in kwargs:
            current_price = kwargs['close']

        return self.calculate(
            current_price=current_price or 0.0,
            entry_price=entry_price or 0.0,
            holding_period=holding_period,
            sentiment_score=sentiment_score,
            **kwargs
        )
