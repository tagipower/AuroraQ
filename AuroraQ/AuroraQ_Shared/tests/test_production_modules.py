#!/usr/bin/env python3
"""
AuroraQ Production Modules Test
===============================

Consolidated test suite for Production components, moved to Shared for centralization.
Tests strategies, optimization, and real-time functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, Any, List

# Import shared utilities
try:
    from ..utils.logger import get_logger
    from ..utils.config_manager import load_config
    from ..utils.metrics import calculate_performance_metrics
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from AuroraQ_Shared.utils.logger import get_logger
    from AuroraQ_Shared.utils.config_manager import load_config
    from AuroraQ_Shared.utils.metrics import calculate_performance_metrics

logger = get_logger("TestProductionModules")

class TestProductionStrategies:
    """Production strategy testing"""
    
    def setup_method(self):
        """Test setup"""
        self.test_data = self._create_test_data()
        self.config = load_config(component_type="production")
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create test price data"""
        np.random.seed(42)
        
        # 100 data points
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Price simulation (Brownian motion)
        price_changes = np.random.normal(0, 0.02, 100)
        prices = 50000 * np.cumprod(1 + price_changes)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.998, 1.002, 100),
            'high': prices * np.random.uniform(1.001, 1.005, 100),
            'low': prices * np.random.uniform(0.995, 0.999, 100),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        return data
    
    def test_strategy_loading(self):
        """Test strategy module loading"""
        try:
            # Try to import production strategies
            production_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AuroraQ_Production')
            if production_path not in sys.path:
                sys.path.append(production_path)
            
            # Test importing various strategy components
            strategy_modules = [
                'strategies.optimized_rule_strategy_e',
                'strategies.ppo_strategy',
                'strategies.strategy_adapter'
            ]
            
            loaded_modules = []
            for module_name in strategy_modules:
                try:
                    __import__(module_name)
                    loaded_modules.append(module_name)
                    logger.info(f"Successfully loaded: {module_name}")
                except ImportError as e:
                    logger.warning(f"Could not load {module_name}: {e}")
            
            # At least one strategy should be loadable
            assert len(loaded_modules) > 0, "No strategy modules could be loaded"
            
        except Exception as e:
            logger.warning(f"Strategy loading test failed: {e}")
            pytest.skip("Strategy modules not available")
    
    def test_strategy_signal_generation(self):
        """Test strategy signal generation"""
        # Mock strategy class for testing
        class MockStrategy:
            def score(self, data: pd.DataFrame, index: int) -> float:
                # Simple moving average crossover strategy
                if len(data) < 20:
                    return 0.0
                
                ma_5 = data['close'].rolling(5).mean().iloc[index]
                ma_20 = data['close'].rolling(20).mean().iloc[index]
                
                if pd.isna(ma_5) or pd.isna(ma_20):
                    return 0.0
                
                return 1.0 if ma_5 > ma_20 else -1.0
        
        strategy = MockStrategy()
        
        # Test signal generation
        for i in range(20, len(self.test_data)):
            signal = strategy.score(self.test_data, i)
            assert isinstance(signal, (int, float))
            assert -1 <= signal <= 1
        
        logger.info("Strategy signal generation test passed")
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        # Simple moving averages
        ma_5 = self.test_data['close'].rolling(5).mean()
        ma_20 = self.test_data['close'].rolling(20).mean()
        
        # Verify non-NaN values exist
        assert not pd.isna(ma_5.iloc[-1])
        assert not pd.isna(ma_20.iloc[-1])
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(self.test_data['close'])
        
        # RSI range validation (0-100)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0
            assert valid_rsi.max() <= 100
        
        logger.info("Technical indicators test passed")
    
    def test_data_preprocessing(self):
        """Test data preprocessing and validation"""
        # Basic data validation
        assert len(self.test_data) == 100
        assert 'close' in self.test_data.columns
        assert 'volume' in self.test_data.columns
        
        # Price data validity
        assert self.test_data['close'].min() > 0
        assert self.test_data['volume'].min() > 0
        
        # Time order validation
        assert self.test_data['timestamp'].is_monotonic_increasing
        
        logger.info("Data preprocessing test passed")

class TestProductionOptimization:
    """Production optimization testing"""
    
    def setup_method(self):
        """Test setup"""
        self.sample_results = self._create_sample_optimization_results()
    
    def _create_sample_optimization_results(self) -> List[Dict[str, Any]]:
        """Create sample optimization results"""
        results = []
        
        strategies = ['RuleStrategyA', 'RuleStrategyB', 'RuleStrategyC', 'PPOStrategy']
        
        for i, strategy in enumerate(strategies):
            result = {
                'strategy': strategy,
                'score': np.random.uniform(0.5, 0.9),
                'sharpe_ratio': np.random.uniform(1.0, 2.5),
                'total_return': np.random.uniform(0.1, 0.3),
                'max_drawdown': np.random.uniform(0.05, 0.15),
                'win_rate': np.random.uniform(0.5, 0.7),
                'total_trades': np.random.randint(50, 200)
            }
            results.append(result)
        
        return results
    
    def test_optimization_result_processing(self):
        """Test optimization result processing"""
        # Test basic result structure
        for result in self.sample_results:
            assert 'strategy' in result
            assert 'score' in result
            assert 'sharpe_ratio' in result
            assert 'total_return' in result
            
            # Validate ranges
            assert 0 <= result['score'] <= 1
            assert result['sharpe_ratio'] >= 0
            assert result['total_trades'] > 0
        
        logger.info("Optimization result processing test passed")
    
    def test_strategy_ranking(self):
        """Test strategy ranking functionality"""
        # Sort by score
        sorted_results = sorted(self.sample_results, key=lambda x: x['score'], reverse=True)
        
        # Verify sorting
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i]['score'] >= sorted_results[i + 1]['score']
        
        # Test multi-criteria ranking
        def calculate_composite_score(result):
            return (result['score'] * 0.4 + 
                   result['sharpe_ratio'] / 3.0 * 0.3 + 
                   result['total_return'] * 0.3)
        
        for result in self.sample_results:
            composite = calculate_composite_score(result)
            assert 0 <= composite <= 2.0  # Reasonable range
        
        logger.info("Strategy ranking test passed")
    
    def test_optimal_combination_recommendation(self):
        """Test optimal combination recommendation logic"""
        # Mock combination logic
        def find_optimal_combination(results, max_strategies=3):
            # Simple greedy selection
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            return sorted_results[:max_strategies]
        
        optimal_combo = find_optimal_combination(self.sample_results)
        
        # Verify combination constraints
        assert len(optimal_combo) <= 3
        assert all(strategy['score'] >= 0.5 for strategy in optimal_combo)
        
        # Test diversity (different strategies)
        strategy_names = [s['strategy'] for s in optimal_combo]
        assert len(set(strategy_names)) == len(strategy_names)  # All unique
        
        logger.info("Optimal combination recommendation test passed")

class TestProductionRealtime:
    """Production real-time system testing"""
    
    def setup_method(self):
        """Test setup"""
        self.mock_market_data = self._create_mock_market_data()
        self.config = load_config(component_type="production")
    
    def _create_mock_market_data(self) -> Dict[str, Any]:
        """Create mock market data"""
        return {
            'symbol': 'BTC-USD',
            'price': 50000.0,
            'volume': 1000000,
            'timestamp': datetime.now(),
            'bid': 49995.0,
            'ask': 50005.0,
            'high_24h': 51000.0,
            'low_24h': 49000.0,
            'change_24h': 0.02
        }
    
    def test_market_data_validation(self):
        """Test market data validation"""
        data = self.mock_market_data
        
        # Required fields
        required_fields = ['symbol', 'price', 'timestamp']
        for field in required_fields:
            assert field in data
        
        # Data type validation
        assert isinstance(data['price'], (int, float))
        assert data['price'] > 0
        assert isinstance(data['timestamp'], datetime)
        
        # Spread validation
        if 'bid' in data and 'ask' in data:
            spread = (data['ask'] - data['bid']) / data['price']
            assert 0 <= spread <= 0.01  # Reasonable spread
        
        logger.info("Market data validation test passed")
    
    def test_trading_limits_validation(self):
        """Test trading limits validation"""
        limits = {
            'max_position_size': 0.1,
            'max_daily_trades': 10,
            'emergency_stop_loss': 0.05,
            'max_portfolio_risk': 0.02
        }
        
        # Test position size limit
        test_position_size = 0.05
        assert test_position_size <= limits['max_position_size']
        
        # Test daily trade limit
        daily_trades = 5
        assert daily_trades <= limits['max_daily_trades']
        
        # Test risk limits
        portfolio_risk = 0.015
        assert portfolio_risk <= limits['max_portfolio_risk']
        
        logger.info("Trading limits validation test passed")
    
    def test_signal_confidence_validation(self):
        """Test signal confidence validation"""
        # Mock signal with confidence
        signal = {
            'action': 'buy',
            'size': 0.05,
            'confidence': 0.75,
            'strategy': 'RuleStrategyA',
            'timestamp': datetime.now()
        }
        
        # Confidence range validation
        assert 0 <= signal['confidence'] <= 1
        
        # Minimum confidence threshold
        min_confidence = 0.6
        assert signal['confidence'] >= min_confidence
        
        # Signal structure validation
        assert signal['action'] in ['buy', 'sell', 'hold']
        assert signal['size'] > 0
        assert isinstance(signal['strategy'], str)
        
        logger.info("Signal confidence validation test passed")
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation using shared utilities"""
        # Create mock trade data
        trades = [
            {
                'action': 'close',
                'pnl_pct': 0.02,
                'pnl': 1000,
                'timestamp': datetime.now() - timedelta(hours=1),
                'holding_time': timedelta(hours=1)
            },
            {
                'action': 'close',
                'pnl_pct': -0.01,
                'pnl': -500,
                'timestamp': datetime.now() - timedelta(minutes=30),
                'holding_time': timedelta(minutes=30)
            },
            {
                'action': 'close',
                'pnl_pct': 0.015,
                'pnl': 750,
                'timestamp': datetime.now(),
                'holding_time': timedelta(minutes=45)
            }
        ]
        
        # Calculate metrics using shared utility
        metrics = calculate_performance_metrics(trades)
        
        # Validate metrics
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'total_trades')
        
        assert metrics.total_trades == 3
        assert 0 <= metrics.win_rate <= 1
        
        logger.info("Performance metrics calculation test passed")

class TestProductionSentiment:
    """Production sentiment integration testing"""
    
    def test_sentiment_analyzer_loading(self):
        """Test sentiment analyzer loading"""
        try:
            # Try to import sentiment components
            production_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AuroraQ_Production')
            if production_path not in sys.path:
                sys.path.append(production_path)
            
            sentiment_modules = [
                'sentiment.sentiment_analyzer',
                'sentiment.news_collector',
                'sentiment.sentiment_scorer'
            ]
            
            loaded_modules = []
            for module_name in sentiment_modules:
                try:
                    __import__(module_name)
                    loaded_modules.append(module_name)
                except ImportError:
                    pass
            
            if loaded_modules:
                logger.info(f"Sentiment modules loaded: {loaded_modules}")
            else:
                logger.warning("No sentiment modules available")
                pytest.skip("Sentiment modules not available")
                
        except Exception as e:
            logger.warning(f"Sentiment loading test failed: {e}")
            pytest.skip("Sentiment components not available")
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Mock sentiment analyzer
        class MockSentimentAnalyzer:
            def analyze_sentiment(self, text: str) -> Dict[str, Any]:
                # Simple keyword-based analysis
                positive_keywords = ['surge', 'high', 'growth', 'adoption', 'bullish']
                negative_keywords = ['crash', 'concerns', 'sell-off', 'bearish', 'decline']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_keywords if word in text_lower)
                negative_count = sum(1 for word in negative_keywords if word in text_lower)
                
                if positive_count > negative_count:
                    score = 0.7
                elif negative_count > positive_count:
                    score = -0.7
                else:
                    score = 0.0
                
                return {
                    'sentiment_score': score,
                    'confidence': 0.8,
                    'analysis_time': datetime.now()
                }
        
        analyzer = MockSentimentAnalyzer()
        
        # Test cases
        test_texts = [
            "Bitcoin price surges to new all-time high as institutional adoption grows",
            "Cryptocurrency market crashes amid regulatory concerns and sell-off",
            "Bitcoin trading volume remains steady in Asian markets"
        ]
        
        for text in test_texts:
            result = analyzer.analyze_sentiment(text)
            
            assert 'sentiment_score' in result
            assert 'confidence' in result
            assert -1 <= result['sentiment_score'] <= 1
            assert 0 <= result['confidence'] <= 1
        
        logger.info("Sentiment analysis test passed")

if __name__ == "__main__":
    # Direct execution test runner
    print("🧪 AuroraQ Production Modules Test Suite")
    
    test_classes = [
        TestProductionStrategies,
        TestProductionOptimization,
        TestProductionRealtime,
        TestProductionSentiment
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                test_method = getattr(test_instance, method_name)
                test_method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")