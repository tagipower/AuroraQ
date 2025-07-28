#!/usr/bin/env python3
"""
Run backtest with optimized RuleStrategyE
최적화된 RuleStrategyE로 백테스트 실행
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import necessary modules
from core.strategy_selector import StrategySelector
from strategy.mab_selector import MABSelector
from loops.run_loop import main
import pandas as pd

print("=== Running Optimized Backtest ===")
print(f"Start time: {datetime.now()}")

# Verify optimized parameters are loaded
from config.rule_param_loader import get_rule_params
params = get_rule_params("RuleE")
print(f"\nOptimized Parameters Loaded:")
print(f"  - Take Profit: {params.get('take_profit_pct', 'N/A')}")
print(f"  - Stop Loss: {params.get('stop_loss_pct', 'N/A')}")
print(f"  - RSI Threshold: {params.get('rsi_breakout_threshold', 'N/A')}")

# Run the main backtest loop
print("\n=== Starting Backtest Loop ===")
try:
    # The main function from run_loop.py handles everything
    main()
    
except Exception as e:
    print(f"\nError during backtest: {e}")
    import traceback
    traceback.print_exc()

print(f"\nEnd time: {datetime.now()}")
print("\n=== Backtest Complete ===")

# Post-process results
print("\n=== Checking Results ===")
import glob
import json

# Find latest strategy scores
score_files = glob.glob('reports/strategy_scores/strategy_scores_*.json')
if score_files:
    latest_score = max(score_files, key=os.path.getctime)
    print(f"\nLatest strategy scores: {latest_score}")
    
    with open(latest_score, 'r') as f:
        scores = json.load(f)
    
    # Check RuleE performance
    if 'scores' in scores and 'RuleStrategyE' in scores['scores']:
        rule_e_score = scores['scores']['RuleStrategyE']
        print(f"\nRuleStrategyE Performance:")
        print(f"  - ROI: {rule_e_score['roi']*100:.2f}%")
        print(f"  - Win Rate: {rule_e_score['win_rate']*100:.1f}%")
        print(f"  - Profit Factor: {rule_e_score['profit_factor']:.2f}")
        print(f"  - Sharpe Ratio: {rule_e_score['sharpe_ratio']:.2f}")
        print(f"  - Overall Grade: {rule_e_score['overall_grade']}")
        
        # Check if targets met
        print(f"\nTarget Achievement:")
        win_rate_target = 35.0
        pf_target = 0.6
        
        win_rate_achieved = rule_e_score['win_rate'] * 100 >= win_rate_target
        pf_achieved = rule_e_score['profit_factor'] >= pf_target
        
        print(f"  - Win Rate: {'✅' if win_rate_achieved else '❌'} ({rule_e_score['win_rate']*100:.1f}% vs {win_rate_target}% target)")
        print(f"  - Profit Factor: {'✅' if pf_achieved else '❌'} ({rule_e_score['profit_factor']:.2f} vs {pf_target} target)")
        
        if win_rate_achieved and pf_achieved:
            print("\n🎉 OPTIMIZATION SUCCESS! Both targets achieved!")
        else:
            print("\n⚠️ Further optimization may be needed.")

# Find latest HTML report
html_files = glob.glob('reports/html/backtest_report_*.html')
if html_files:
    latest_html = max(html_files, key=os.path.getctime)
    print(f"\nHTML Report: {latest_html}")