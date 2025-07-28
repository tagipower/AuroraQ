"""
컨트롤러 계층 (Controller Layer)
- BacktestLoop 오케스트레이터: 데이터 피드, 전략 실행, 로그 기록
- PPO 학습과 MABSelector로 결과 피드백 전달
- Exploration 모드 지원 (백테스트 시 데이터 확보용)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 로거 먼저 생성
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기존 시스템 임포트 (선택적)
try:
    from core.strategy_selector import StrategySelector
    from strategy.mab_selector import MABSelector
    from core.ppo_agent_proxy import PPOAgentProxy
    from core.strategy_score_manager import update_strategy_metrics
    LEGACY_SYSTEM_AVAILABLE = True
    logger.info("기존 전략 시스템 연동 성공")
except ImportError as e:
    logger.info(f"기존 시스템 연동 실패 - 독립 모드로 실행: {e}")
    StrategySelector = None
    MABSelector = None
    PPOAgentProxy = None
    update_strategy_metrics = None
    LEGACY_SYSTEM_AVAILABLE = False

# 새로운 계층 임포트
from .data_layer import DataLayer
from .signal_layer import SignalProcessor, SignalResult
from .execution_layer import ExecutionSimulator
from .evaluation_layer import MetricsEvaluator

# 피드백 시스템 임포트
try:
    from ..integration.ppo_mab_bridge import BacktestFeedbackBridge
except ImportError:
    logger.warning("피드백 브리지 임포트 실패 - 상대 경로 시도")
    try:
        from backtest.v2.integration.ppo_mab_bridge import BacktestFeedbackBridge
    except ImportError:
        logger.error("피드백 브리지 임포트 실패 - 피드백 기능 비활성화")
        BacktestFeedbackBridge = None


class BacktestMode:
    """백테스트 모드"""
    NORMAL = "normal"           # 일반 백테스트
    EXPLORATION = "exploration" # 탐색 모드 (다양한 전략 시도)
    VALIDATION = "validation"   # 검증 모드 (엄격한 조건)
    WALK_FORWARD = "walk_forward"  # 워크포워드 분석


class BacktestController:
    """
    백테스트 컨트롤러
    각 계층을 조정하여 백테스트 수행
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 mode: str = BacktestMode.NORMAL,
                 enable_multiframe: bool = True,
                 enable_exploration: bool = False,
                 cache_size: int = 1000):
        """
        Args:
            initial_capital: 초기 자본
            mode: 백테스트 모드
            enable_multiframe: 다중 타임프레임 활성화
            enable_exploration: 탐색 모드 활성화
            cache_size: 캐시 크기
        """
        self.initial_capital = initial_capital
        self.mode = mode
        self.enable_exploration = enable_exploration
        
        # 계층 초기화
        self.data_layer = DataLayer(
            cache_size=cache_size,
            enable_multiframe=enable_multiframe
        )
        self.signal_processor = SignalProcessor()
        self.execution_simulator = ExecutionSimulator(
            initial_capital=initial_capital,
            enable_risk_adjustment=True,
            enable_market_impact=True
        )
        self.metrics_evaluator = MetricsEvaluator(
            initial_capital=initial_capital,
            min_sample_size=30
        )
        
        # 기존 시스템 통합
        self.strategy_selector = None
        self.ppo_agent = None
        self.mab_selector = None
        
        # 피드백 시스템
        self.feedback_bridge = None
        if BacktestFeedbackBridge:
            self.feedback_bridge = BacktestFeedbackBridge(
                enable_ppo_feedback=True,
                enable_mab_feedback=True
            )
        
        # 백테스트 상태
        self.current_positions = {}
        self.closed_trades = []
        self.exploration_history = []
        
        # 성능 추적
        self.performance_stats = {
            "total_signals": 0,
            "executed_trades": 0,
            "exploration_trades": 0,
            "processing_time": []
        }
    
    def initialize_strategies(self, 
                            sentiment_file: Optional[str] = None,
                            enable_ppo: bool = True):
        """전략 시스템 초기화"""
        if not LEGACY_SYSTEM_AVAILABLE:
            logger.info("기존 전략 시스템 없음 - 독립 모드로 실행")
            self._initialize_dummy_strategies()
            return
            
        try:
            # 전략 선택기 초기화
            self.strategy_selector = StrategySelector(
                sentiment_file=sentiment_file,
                enable_ppo=enable_ppo
            )
            
            # MAB 선택기 초기화
            strategy_names = list(self.strategy_selector.strategies.keys())
            self.mab_selector = MABSelector(strategy_names)
            
            # PPO 에이전트 초기화 (있는 경우)
            if enable_ppo:
                try:
                    self.ppo_agent = PPOAgentProxy()
                except Exception as e:
                    logger.warning(f"PPO 에이전트 초기화 실패: {e}")
            
            logger.info(f"전략 시스템 초기화 완료: {len(strategy_names)}개 전략")
            
        except Exception as e:
            logger.error(f"전략 시스템 초기화 실패: {e}")
            logger.warning("더미 전략으로 대체")
            self._initialize_dummy_strategies()
    
    def _initialize_dummy_strategies(self):
        """실제 전략들로 초기화"""
        try:
            # 전략 어댑터 시스템 임포트
            from ..integration.strategy_adapter import get_strategy_registry, register_builtin_strategies
            
            # 내장 전략들 등록
            register_builtin_strategies()
            
            # 전략 레지스트리 가져오기
            self.strategy_registry = get_strategy_registry()
            strategy_names = self.strategy_registry.get_all_strategy_names()
            
            if not strategy_names:
                logger.warning("등록된 전략이 없습니다. 기본 더미 전략을 사용합니다.")
                self._create_fallback_dummy_strategies()
                return
            
            # 실제 전략 선택기
            class RealStrategySelector:
                def __init__(self, strategy_registry):
                    self.strategy_registry = strategy_registry
                    self.strategies = {}
                    
                    # 전략 어댑터들을 등록
                    for name in strategy_registry.get_all_strategy_names():
                        adapter = strategy_registry.get_strategy_adapter(name)
                        if adapter:
                            self.strategies[name] = adapter
                
                def select(self, price_data):
                    """전략 선택 및 신호 생성"""
                    import random
                    
                    # 최적화된 전략 우선 선택
                    if "OptimizedRuleStrategyE" in self.strategies:
                        strategy_name = "OptimizedRuleStrategyE"
                        logger.info("최적화된 RuleStrategyE 사용")
                    else:
                        # 백업: 랜덤 선택
                        strategy_name = random.choice(list(self.strategies.keys()))
                        logger.info(f"랜덤 전략 선택: {strategy_name}")
                    
                    strategy_adapter = self.strategies[strategy_name]
                    
                    # 신호 생성
                    signal = strategy_adapter.generate_signal(price_data)
                    
                    return {
                        "strategy": strategy_name,
                        "strategy_object": strategy_adapter,
                        "signal": signal,
                        "score": signal.get("strength", 0.5),
                        "base_score": signal.get("strength", 0.5),
                        "sentiment_score": random.uniform(0.4, 0.7),
                        "regime": "neutral",
                        "volatility": random.uniform(0.01, 0.03),
                        "trend": "sideways"
                    }
            
            # 실제 MAB 선택기 (단순 버전)
            class RealMABSelector:
                def __init__(self, strategies):
                    self.strategies = strategies
                    self.scores = {s: 0.5 for s in strategies}
                    self.counts = {s: 0 for s in strategies}
                
                def select(self):
                    import random
                    # 간단한 epsilon-greedy
                    if random.random() < 0.1:  # 10% 탐색
                        return random.choice(self.strategies)
                    else:  # 90% 활용
                        return max(self.strategies, key=lambda s: self.scores.get(s, 0.5))
                
                def update(self, strategy, reward):
                    if strategy in self.scores:
                        # 이동평균으로 점수 업데이트
                        alpha = 0.1
                        self.scores[strategy] = (1-alpha) * self.scores[strategy] + alpha * reward
                        self.counts[strategy] += 1
                
                def get_scores(self):
                    return self.scores
            
            self.strategy_selector = RealStrategySelector(self.strategy_registry)
            self.mab_selector = RealMABSelector(strategy_names)
            
            logger.info(f"실제 전략 시스템 초기화 완료: {len(strategy_names)}개 전략")
            logger.info(f"등록된 전략들: {strategy_names}")
            
        except Exception as e:
            logger.error(f"실제 전략 초기화 실패: {e}")
            logger.warning("기본 더미 전략으로 대체합니다.")
            self._create_fallback_dummy_strategies()
    
    def _create_fallback_dummy_strategies(self):
        """대체용 더미 전략 생성"""
        class DummyStrategy:
            def __init__(self, name):
                self.name = name
            
            def generate_signal(self, price_data):
                """간단한 더미 신호 생성"""
                import random
                actions = ["HOLD", "BUY", "SELL"]
                return {
                    "action": random.choice(actions),
                    "strength": random.uniform(0.3, 0.9),
                    "price": price_data['close'].iloc[-1] if len(price_data) > 0 else 50000
                }
        
        # 더미 전략 선택기
        class DummyStrategySelector:
            def __init__(self):
                self.strategies = {
                    "SimpleMA": DummyStrategy("SimpleMA"),
                    "RSIStrategy": DummyStrategy("RSIStrategy"),
                    "DummyStrategy": DummyStrategy("DummyStrategy")
                }
            
            def select(self, price_data):
                """랜덤 전략 선택"""
                import random
                strategy_name = random.choice(list(self.strategies.keys()))
                strategy = self.strategies[strategy_name]
                signal = strategy.generate_signal(price_data)
                
                return {
                    "strategy": strategy_name,
                    "strategy_object": strategy,
                    "signal": signal,
                    "score": random.uniform(0.4, 0.8),
                    "base_score": random.uniform(0.3, 0.7),
                    "sentiment_score": random.uniform(0.2, 0.8),
                    "regime": random.choice(["bull", "bear", "neutral"]),
                    "volatility": random.uniform(0.01, 0.05),
                    "trend": random.choice(["uptrend", "downtrend", "sideways"])
                }
        
        # 더미 MAB 선택기
        class DummyMABSelector:
            def __init__(self, strategies):
                self.strategies = strategies
                self.scores = {s: 0.5 for s in strategies}
            
            def select(self):
                import random
                return random.choice(self.strategies)
            
            def update(self, strategy, reward):
                self.scores[strategy] = reward
            
            def get_scores(self):
                return self.scores
        
        self.strategy_selector = DummyStrategySelector()
        self.mab_selector = DummyMABSelector(list(self.strategy_selector.strategies.keys()))
        logger.info(f"대체 더미 전략 시스템 초기화 완료: {len(self.strategy_selector.strategies)}개 전략")
    
    def run_backtest(self,
                    price_data_path: str,
                    sentiment_data_path: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    window_size: int = 100,
                    indicators: List[str] = None) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            price_data_path: 가격 데이터 경로
            sentiment_data_path: 감정 데이터 경로
            start_date: 시작 날짜
            end_date: 종료 날짜
            window_size: 데이터 윈도우 크기
            indicators: 사용할 지표 목록
            
        Returns:
            백테스트 결과
        """
        start_time = time.time()
        
        # 기본 지표 설정
        if indicators is None:
            indicators = [
                "sma_20", "sma_50", "ema_12", "ema_26",
                "rsi", "macd", "macd_line", "macd_signal", "macd_hist",
                "bbands", "bollinger", "bb_upper", "bb_middle", "bb_lower", 
                "atr", "adx", "volatility"
            ]
        
        try:
            # 1. 데이터 로드
            logger.info("데이터 로드 중...")
            price_data = self.data_layer.load_price_data(price_data_path)
            
            if sentiment_data_path:
                self.data_layer.load_sentiment_data(sentiment_data_path)
            
            # 날짜 필터링
            if start_date:
                price_data = price_data[price_data['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                price_data = price_data[price_data['timestamp'] <= pd.to_datetime(end_date)]
            
            logger.info(f"데이터 로드 완료: {len(price_data)}개 레코드")
            
            # 2. 백테스트 루프
            results = self._run_backtest_loop(
                price_data, window_size, indicators
            )
            
            # 3. 최종 평가
            final_metrics = self._evaluate_results()
            
            # 4. 보고서 생성
            reports = self.metrics_evaluator.generate_reports(format="both")
            
            # 5. 피드백 데이터 저장
            feedback_stats = {}
            if self.feedback_bridge:
                feedback_stats = {
                    "mab_statistics": self.feedback_bridge.get_mab_statistics()
                }
                
                # 피드백 데이터 저장
                feedback_file = "reports/backtest/feedback_data.json"
                os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
                self.feedback_bridge.save_feedback_data(feedback_file)
            
            # 실행 시간
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "metrics": final_metrics,
                "reports": reports,
                "stats": {
                    **self.performance_stats,
                    "execution_time": execution_time,
                    "cache_stats": self.data_layer.get_cache_stats(),
                    "feedback_stats": feedback_stats
                }
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_backtest_loop(self,
                         price_data: pd.DataFrame,
                         window_size: int,
                         indicators: List[str]) -> List[Dict[str, Any]]:
        """백테스트 메인 루프"""
        results = []
        
        # 진행 표시
        with tqdm(total=len(price_data), desc="백테스트 진행") as pbar:
            for i in range(window_size, len(price_data)):
                step_start = time.time()
                
                # 데이터 윈도우 준비
                data_window = self.data_layer.get_data_window(i, window_size)
                
                # 지표 계산
                price_window = data_window["price"]
                calculated_indicators = self.data_layer.calculate_indicators(
                    price_window, indicators
                )
                
                # 현재 타임스탬프
                current_timestamp = price_window['timestamp'].iloc[-1]
                
                # 전략 선택 및 신호 생성
                signal_result = self._process_signal(
                    data_window, calculated_indicators, current_timestamp
                )
                
                # 신호 처리 결과 저장
                if signal_result:
                    results.append(signal_result)
                
                # 성능 추적
                step_time = time.time() - step_start
                self.performance_stats["processing_time"].append(step_time)
                
                # 진행 상황 업데이트
                pbar.update(1)
                
                # 주기적 로깅
                if i % 1000 == 0:
                    self._log_progress(i, len(price_data))
        
        return results
    
    def _process_signal(self,
                       data_window: Dict[str, Any],
                       indicators: Dict[str, pd.Series],
                       timestamp: datetime) -> Optional[Dict[str, Any]]:
        """신호 처리"""
        try:
            # 1. 전략 선택
            if self.enable_exploration and np.random.random() < 0.1:
                # 탐색 모드: 10% 확률로 랜덤 전략 선택
                strategy_name = self._explore_strategy()
                is_exploration = True
            else:
                # 일반 모드: StrategySelector 사용
                selection = self.strategy_selector.select(data_window["price"])
                strategy_name = selection.get("strategy", "UNKNOWN")
                is_exploration = False
            
            self.performance_stats["total_signals"] += 1
            
            # 2. 전략 실행
            strategy = self.strategy_selector.strategies.get(strategy_name)
            if not strategy:
                return None
            
            # 지표 설정 (전략 어댑터에 백테스트 지표 전달)
            if hasattr(strategy, 'set_indicators'):
                strategy.set_indicators(indicators)
            
            raw_signal = strategy.generate_signal(data_window["price"])
            
            # 3. 신호 처리
            sentiment_score = data_window.get("sentiment_score", 0.5)
            processed_signal = self.signal_processor.process_signal(
                raw_signal,
                data_window,
                indicators,
                sentiment_score
            )
            
            # 4. 실행 시뮬레이션
            if processed_signal.action.upper() != "HOLD":
                execution_result = self.execution_simulator.execute_signal(
                    processed_signal,
                    data_window,
                    timestamp
                )
                
                if execution_result.get("executed"):
                    self.performance_stats["executed_trades"] += 1
                    if is_exploration:
                        self.performance_stats["exploration_trades"] += 1
                    
                    # 거래 기록
                    self._record_trade(
                        strategy_name,
                        execution_result,
                        processed_signal,
                        timestamp
                    )
                    
                    # PPO/MAB 피드백
                    self._update_learning_systems(
                        strategy_name,
                        execution_result,
                        is_exploration
                    )
                    
                    # 새로운 피드백 시스템
                    if self.feedback_bridge:
                        feedback_result = self.feedback_bridge.process_backtest_step(
                            strategy=strategy_name,
                            market_data={
                                **data_window,
                                "indicators": indicators
                            },
                            signal_result=processed_signal,
                            execution_result=execution_result,
                            is_exploration=is_exploration
                        )
            
            return {
                "timestamp": timestamp,
                "strategy": strategy_name,
                "signal": processed_signal,
                "is_exploration": is_exploration
            }
            
        except Exception as e:
            logger.error(f"신호 처리 오류: {e}")
            return None
    
    def _explore_strategy(self) -> str:
        """탐색 모드에서 전략 선택"""
        # Epsilon-greedy 탐색
        if np.random.random() < 0.3:
            # 30% 확률로 완전 랜덤
            strategies = list(self.strategy_selector.strategies.keys())
            selected = np.random.choice(strategies)
        else:
            # 70% 확률로 성과가 낮은 전략 우선
            scores = self.mab_selector.get_scores()
            # 점수를 반전시켜 낮은 점수가 높은 확률
            inv_scores = 1 / (np.array(list(scores.values())) + 0.1)
            probs = inv_scores / inv_scores.sum()
            
            strategies = list(scores.keys())
            selected = np.random.choice(strategies, p=probs)
        
        self.exploration_history.append({
            "timestamp": datetime.now(),
            "strategy": selected
        })
        
        return selected
    
    def _record_trade(self,
                     strategy: str,
                     execution_result: Dict[str, Any],
                     signal: SignalResult,
                     timestamp: datetime):
        """거래 기록"""
        trade = execution_result.get("trade", {})
        
        # PnL 계산 (포지션 청산 시)
        pnl = 0
        pnl_pct = 0
        holding_time = 0
        
        # 반대 포지션이 있는 경우 청산으로 간주
        if strategy in self.current_positions:
            position = self.current_positions[strategy]
            if (position["side"] == "BUY" and signal.action == "SELL") or \
               (position["side"] == "SELL" and signal.action == "BUY"):
                # PnL 계산
                if position["side"] == "BUY":
                    pnl = (trade["execution_price"] - position["entry_price"]) * position["quantity"]
                else:
                    pnl = (position["entry_price"] - trade["execution_price"]) * position["quantity"]
                
                pnl_pct = pnl / (position["entry_price"] * position["quantity"])
                holding_time = (timestamp - position["timestamp"]).total_seconds()
                
                # 포지션 제거
                del self.current_positions[strategy]
                
                # 청산 거래 기록
                self.closed_trades.append({
                    "strategy": strategy,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_time": holding_time
                })
        
        # 새 포지션 기록
        if signal.action in ["BUY", "SELL"]:
            self.current_positions[strategy] = {
                "side": signal.action,
                "entry_price": trade["execution_price"],
                "quantity": trade["quantity"],
                "timestamp": timestamp
            }
        
        # 평가 계층에 거래 추가
        self.metrics_evaluator.add_trade(strategy, {
            "timestamp": timestamp,
            "signal_action": signal.action,
            "entry_price": trade.get("price", 0),
            "exit_price": trade.get("execution_price", 0) if pnl != 0 else None,
            "quantity": trade.get("quantity", 0),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_time": holding_time,
            "commission": trade.get("commission", 0),
            "slippage": trade.get("slippage", 0),
            "market_impact": trade.get("market_impact", 0),
            "signal_confidence": signal.confidence,
            "position_size": signal.position_size,
            "regime": signal.metadata.get("regime", "") if signal.metadata else "",
            "volatility": signal.metadata.get("volatility", 0) if signal.metadata else 0,
            "sentiment_score": signal.metadata.get("sentiment_score", 0) if signal.metadata else 0
        })
    
    def _update_learning_systems(self,
                               strategy: str,
                               execution_result: Dict[str, Any],
                               is_exploration: bool):
        """학습 시스템 업데이트 (PPO/MAB)"""
        # MAB 업데이트
        reward = self._calculate_reward(execution_result)
        
        # 탐색 모드에서는 보상을 조정
        if is_exploration:
            reward *= 1.2  # 탐색 보너스
        
        self.mab_selector.update(strategy, reward)
        
        # 전략 메트릭 업데이트
        if update_strategy_metrics:
            update_strategy_metrics(strategy, {
                "last_reward": reward,
                "is_exploration": is_exploration,
                "execution_details": execution_result.get("execution_details", {})
            })
        
        # PPO 업데이트 (있는 경우)
        if self.ppo_agent and strategy == "PPOStrategy":
            # PPO 버퍼에 경험 추가
            self._add_ppo_experience(execution_result)
    
    def _calculate_reward(self, execution_result: Dict[str, Any]) -> float:
        """보상 계산"""
        trade = execution_result.get("trade", {})
        details = execution_result.get("execution_details", {})
        
        # 기본 보상 (신뢰도 기반)
        base_reward = trade.get("signal_confidence", 0.5)
        
        # 실행 비용 페널티
        total_cost = details.get("total_cost", 0)
        cost_penalty = total_cost / trade.get("price", 1) if trade.get("price") else 0
        
        # 최종 보상
        reward = base_reward - cost_penalty
        
        return max(0, min(1, reward))
    
    def _add_ppo_experience(self, execution_result: Dict[str, Any]):
        """PPO 경험 추가"""
        # TODO: PPO 에이전트 통합
        pass
    
    def _evaluate_results(self) -> Dict[str, Any]:
        """최종 결과 평가"""
        all_metrics = {}
        
        # 각 전략 평가
        for strategy in self.strategy_selector.strategies.keys():
            metrics = self.metrics_evaluator.evaluate_strategy(strategy)
            all_metrics[strategy] = metrics
        
        # 최고 전략 선택
        best_strategy, best_metrics = self.metrics_evaluator.get_best_strategy()
        
        # 전략 비교
        comparison_df = self.metrics_evaluator.compare_strategies()
        
        return {
            "all_metrics": all_metrics,
            "best_strategy": best_strategy,
            "best_metrics": best_metrics,
            "comparison": comparison_df.to_dict() if not comparison_df.empty else {}
        }
    
    def _log_progress(self, current_step: int, total_steps: int):
        """진행 상황 로깅"""
        progress = current_step / total_steps * 100
        avg_time = np.mean(self.performance_stats["processing_time"][-1000:])
        
        logger.info(f"진행률: {progress:.1f}% | "
                   f"실행 거래: {self.performance_stats['executed_trades']} | "
                   f"평균 처리 시간: {avg_time*1000:.2f}ms")


class BacktestOrchestrator:
    """
    백테스트 오케스트레이터
    여러 백테스트를 관리하고 조정
    """
    
    def __init__(self, n_workers: int = 4):
        """
        Args:
            n_workers: 병렬 작업자 수
        """
        self.n_workers = n_workers
        self.controllers = []
        self.results = []
    
    def run_multiple_backtests(self,
                             configurations: List[Dict[str, Any]],
                             parallel: bool = True) -> List[Dict[str, Any]]:
        """
        여러 백테스트 실행
        
        Args:
            configurations: 백테스트 설정 목록
            parallel: 병렬 실행 여부
            
        Returns:
            백테스트 결과 목록
        """
        if parallel and len(configurations) > 1:
            return self._run_parallel(configurations)
        else:
            return self._run_sequential(configurations)
    
    def _run_parallel(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """병렬 백테스트 실행"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # 백테스트 제출
            future_to_config = {
                executor.submit(self._run_single_backtest, config): config
                for config in configurations
            }
            
            # 결과 수집
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"백테스트 완료: {config.get('name', 'unnamed')}")
                except Exception as e:
                    logger.error(f"백테스트 실패: {config.get('name', 'unnamed')} - {e}")
                    results.append({
                        "success": False,
                        "config": config,
                        "error": str(e)
                    })
        
        return results
    
    def _run_sequential(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """순차 백테스트 실행"""
        results = []
        
        for config in configurations:
            try:
                result = self._run_single_backtest(config)
                results.append(result)
                logger.info(f"백테스트 완료: {config.get('name', 'unnamed')}")
            except Exception as e:
                logger.error(f"백테스트 실패: {config.get('name', 'unnamed')} - {e}")
                results.append({
                    "success": False,
                    "config": config,
                    "error": str(e)
                })
        
        return results
    
    def _run_single_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """단일 백테스트 실행"""
        # 컨트롤러 생성
        controller = BacktestController(
            initial_capital=config.get("initial_capital", 1000000),
            mode=config.get("mode", BacktestMode.NORMAL),
            enable_multiframe=config.get("enable_multiframe", True),
            enable_exploration=config.get("enable_exploration", False),
            cache_size=config.get("cache_size", 1000)
        )
        
        # 전략 초기화
        controller.initialize_strategies(
            sentiment_file=config.get("sentiment_file"),
            enable_ppo=config.get("enable_ppo", True)
        )
        
        # 백테스트 실행
        result = controller.run_backtest(
            price_data_path=config["price_data_path"],
            sentiment_data_path=config.get("sentiment_data_path"),
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            window_size=config.get("window_size", 100),
            indicators=config.get("indicators")
        )
        
        # 설정 정보 추가
        result["config"] = config
        
        return result
    
    def walk_forward_analysis(self,
                            base_config: Dict[str, Any],
                            n_windows: int = 10,
                            train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        워크포워드 분석
        
        Args:
            base_config: 기본 백테스트 설정
            n_windows: 분석 윈도우 수
            train_ratio: 훈련 데이터 비율
            
        Returns:
            워크포워드 분석 결과
        """
        # 데이터 기간 분할
        start_date = pd.to_datetime(base_config.get("start_date", "2023-01-01"))
        end_date = pd.to_datetime(base_config.get("end_date", datetime.now()))
        total_days = (end_date - start_date).days
        
        window_size = total_days // n_windows
        train_size = int(window_size * train_ratio)
        test_size = window_size - train_size
        
        results = []
        
        for i in range(n_windows):
            # 훈련 기간
            train_start = start_date + timedelta(days=i * window_size)
            train_end = train_start + timedelta(days=train_size)
            
            # 테스트 기간
            test_start = train_end
            test_end = test_start + timedelta(days=test_size)
            
            # 훈련 백테스트
            train_config = base_config.copy()
            train_config.update({
                "name": f"train_window_{i}",
                "start_date": train_start.strftime("%Y-%m-%d"),
                "end_date": train_end.strftime("%Y-%m-%d"),
                "mode": BacktestMode.WALK_FORWARD
            })
            
            # 테스트 백테스트
            test_config = base_config.copy()
            test_config.update({
                "name": f"test_window_{i}",
                "start_date": test_start.strftime("%Y-%m-%d"),
                "end_date": test_end.strftime("%Y-%m-%d"),
                "mode": BacktestMode.VALIDATION
            })
            
            # 실행
            train_result = self._run_single_backtest(train_config)
            test_result = self._run_single_backtest(test_config)
            
            results.append({
                "window": i,
                "train": train_result,
                "test": test_result
            })
        
        # 워크포워드 통계
        wf_stats = self._calculate_walk_forward_stats(results)
        
        return {
            "windows": results,
            "statistics": wf_stats
        }
    
    def _calculate_walk_forward_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """워크포워드 통계 계산"""
        train_metrics = []
        test_metrics = []
        
        for window in results:
            if window["train"].get("success") and window["test"].get("success"):
                train_best = window["train"]["metrics"]["best_metrics"]
                test_best = window["test"]["metrics"]["best_metrics"]
                
                train_metrics.append(train_best.roi)
                test_metrics.append(test_best.roi)
        
        if not train_metrics:
            return {}
        
        # 통계 계산
        train_avg = np.mean(train_metrics)
        test_avg = np.mean(test_metrics)
        
        # 효율성 비율
        efficiency_ratio = test_avg / train_avg if train_avg != 0 else 0
        
        return {
            "train_avg_roi": train_avg,
            "test_avg_roi": test_avg,
            "efficiency_ratio": efficiency_ratio,
            "consistency": np.corrcoef(train_metrics, test_metrics)[0, 1] if len(train_metrics) > 1 else 0
        }