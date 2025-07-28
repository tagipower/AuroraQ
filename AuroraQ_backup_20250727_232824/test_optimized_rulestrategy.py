#!/usr/bin/env python3
"""
Test script to verify RuleStrategyE is using optimized parameters
RuleStrategyE가 최적화된 파라미터를 사용하는지 확인
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import strategy
from strategy.rule_strategy_e import RuleStrategyE
from config.rule_param_loader import get_rule_params

print("=== Testing RuleStrategyE with Optimized Parameters ===\n")

# 1. Check parameter loading
print("1. Parameter Loading Test:")
params = get_rule_params("RuleE")
print(f"   - take_profit_pct: {params.get('take_profit_pct', 'N/A')}")
print(f"   - stop_loss_pct: {params.get('stop_loss_pct', 'N/A')}")
print(f"   - rsi_breakout_threshold: {params.get('rsi_breakout_threshold', 'N/A')}")
print(f"   - breakout_window_short: {params.get('breakout_window_short', 'N/A')}")

# 2. Initialize strategy
print("\n2. Strategy Initialization:")
strategy = RuleStrategyE()
print(f"   - Strategy name: {strategy.name}")
# Check if attributes exist with different names
if hasattr(strategy, 'tp_pct'):
    print(f"   - TP/SL ratio: {strategy.tp_pct}/{strategy.sl_pct} = {strategy.tp_pct/strategy.sl_pct:.2f}:1")
elif hasattr(strategy, 'take_profit_pct'):
    print(f"   - TP/SL ratio: {strategy.take_profit_pct}/{strategy.stop_loss_pct} = {strategy.take_profit_pct/strategy.stop_loss_pct:.2f}:1")
else:
    print("   - TP/SL parameters not found as attributes")

# 3. Check if optimized parameters are loaded
print("\n3. Parameter Verification:")
expected_tp = 0.025  # From optimized_rule_params.yaml
expected_sl = 0.012
expected_rsi = 60

is_optimized = (
    abs(strategy.take_profit_pct - expected_tp) < 0.0001 and
    abs(strategy.stop_loss_pct - expected_sl) < 0.0001 and
    strategy.rsi_breakout_threshold == expected_rsi
)

if is_optimized:
    print("   ✅ Strategy is using OPTIMIZED parameters!")
else:
    print("   ❌ Strategy is using DEFAULT parameters!")
    
# 4. Create sample data for signal test
print("\n4. Signal Generation Test:")
# Generate sample price data
dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
prices = 50000 + np.random.randn(100).cumsum() * 100
volume = np.random.uniform(100, 1000, 100)

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices + np.random.randn(100) * 50,
    'high': prices + abs(np.random.randn(100) * 100),
    'low': prices - abs(np.random.randn(100) * 100),
    'close': prices,
    'volume': volume
})

# Get signal
signal = strategy.get_signal(df)
print(f"   - Signal generated: {signal.get('action', 'N/A')}")
print(f"   - Signal strength: {signal.get('strength', 0):.3f}")

# 5. Performance expectations
print("\n5. Performance Expectations with Optimized Parameters:")
print("   - Target Win Rate: 35%+ (vs previous 13.51%)")
print("   - Target Profit Factor: 0.6+ (vs previous 0.15)")
print("   - Target Sharpe Ratio: 0.5+")
print("   - Risk/Reward Ratio: 2.08:1 (improved from 1.67:1)")

print("\n=== Test Complete ===")

# Run actual backtest if requested
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run-backtest', action='store_true', help='Run full backtest')
args = parser.parse_args()

if args.run_backtest:
    print("\n=== Running Full Backtest ===")
    from loops.run_loop import main
    main()