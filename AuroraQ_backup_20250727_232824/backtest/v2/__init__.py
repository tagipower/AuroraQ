"""
AuroraQ 백테스트 시스템 v2
적응형·확률적·다중프레임 백테스트 환경
"""

from .layers.data_layer import DataLayer, IndicatorCache
from .layers.signal_layer import SignalProcessor, AdaptiveEntrySystem
from .layers.execution_layer import ExecutionSimulator, SlippageModel
from .layers.evaluation_layer import MetricsEvaluator
from .layers.controller_layer import BacktestController, BacktestOrchestrator

__all__ = [
    'DataLayer',
    'IndicatorCache',
    'SignalProcessor', 
    'AdaptiveEntrySystem',
    'ExecutionSimulator',
    'SlippageModel',
    'MetricsEvaluator',
    'BacktestController',
    'BacktestOrchestrator'
]

__version__ = '2.0.0'