#!/usr/bin/env python3
"""
Test script for optimized strategy parameters
최적화된 전략 파라미터 테스트
"""

import sys
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

from layers.controller_layer import BacktestController
from utils.strategy_scores_generator import generate_strategy_scores_from_backtest
from utils.html_report_generator import generate_html_report

def test_optimized_strategies():
    """최적화된 전략 테스트"""
    print("=== Testing Optimized Strategy Parameters ===")
    
    # Initialize controller
    controller = BacktestController()
    controller.initialize_strategies()
    
    # Load data
    df = pd.read_csv('../../data/btc_5m_sample.csv')
    print(f'Loaded data shape: {df.shape}')
    
    print(f'\n=== Running backtest with optimized parameters ===')
    
    try:
        # Run backtest with optimized parameters
        results = controller.run_backtest(
            price_data_path='../../data/btc_5m_sample.csv',
            window_size=100
        )
        
        print(f"Backtest completed successfully!")
        
        # Extract key metrics
        all_metrics = results.get('metrics', {}).get('all_metrics', {})
        stats = results.get('stats', {})
        
        print(f"\n=== Performance Summary ===")
        
        total_trades = 0
        profitable_strategies = 0
        
        for strategy_name, metrics in all_metrics.items():
            trades = metrics.total_trades
            roi = float(metrics.roi) * 100
            win_rate = float(metrics.win_rate) * 100
            profit_factor = float(metrics.profit_factor)
            sharpe = float(metrics.sharpe_ratio)
            max_dd = float(metrics.max_drawdown) * 100
            
            total_trades += trades
            if roi > 0:
                profitable_strategies += 1
            
            print(f"\n{strategy_name}:")
            print(f"  Trades: {trades}")
            print(f"  ROI: {roi:.2f}%")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            print(f"  Max Drawdown: {max_dd:.2f}%")
        
        print(f"\n=== Overall Statistics ===")
        print(f"Total Signals: {stats.get('total_signals', 0):,}")
        print(f"Executed Trades: {stats.get('executed_trades', 0):,}")
        print(f"Total Strategy Trades: {total_trades}")
        print(f"Profitable Strategies: {profitable_strategies}/{len(all_metrics)}")
        print(f"Execution Time: {stats.get('execution_time', 0):.1f}s")
        
        # Performance analysis
        execution_rate = (stats.get('executed_trades', 0) / stats.get('total_signals', 1)) * 100
        print(f"Execution Rate: {execution_rate:.1f}%")
        
        # Generate strategy scores
        print(f"\n=== Generating Strategy Scores ===")
        try:
            scores_file = generate_strategy_scores_from_backtest(results)
            print(f"Strategy scores saved to: {scores_file}")
        except Exception as e:
            print(f"Failed to generate strategy scores: {e}")
        
        # Generate HTML report
        print(f"\n=== Generating HTML Report ===")
        try:
            html_file = generate_html_report(results, title="Optimized Strategy Performance Report")
            print(f"HTML report saved to: {html_file}")
        except Exception as e:
            print(f"Failed to generate HTML report: {e}")
        
        # Performance evaluation
        print(f"\n=== Performance Evaluation ===")
        if total_trades > 0:
            avg_roi = sum(float(m.roi) for m in all_metrics.values()) / len(all_metrics) * 100
            avg_win_rate = sum(float(m.win_rate) for m in all_metrics.values()) / len(all_metrics) * 100
            avg_profit_factor = sum(float(m.profit_factor) for m in all_metrics.values()) / len(all_metrics)
            
            print(f"Average ROI: {avg_roi:.2f}%")
            print(f"Average Win Rate: {avg_win_rate:.1f}%") 
            print(f"Average Profit Factor: {avg_profit_factor:.2f}")
            
            # Compare with targets
            print(f"\n=== Target Comparison ===")
            target_win_rate = 35  # From optimized_rule_params.yaml
            target_profit_factor = 0.6
            
            print(f"Win Rate Target: {target_win_rate}% | Achieved: {avg_win_rate:.1f}% | {'✅' if avg_win_rate >= target_win_rate else '❌'}")
            print(f"Profit Factor Target: {target_profit_factor} | Achieved: {avg_profit_factor:.2f} | {'✅' if avg_profit_factor >= target_profit_factor else '❌'}")
            
            if avg_win_rate >= target_win_rate and avg_profit_factor >= target_profit_factor:
                print(f"\n🎉 OPTIMIZATION SUCCESS! Both targets achieved.")
            else:
                print(f"\n⚠️ Optimization needs further tuning.")
                print(f"Recommendations:")
                if avg_win_rate < target_win_rate:
                    print(f"  - Tighten entry conditions to improve win rate")
                if avg_profit_factor < target_profit_factor:
                    print(f"  - Adjust stop loss/take profit ratios")
        else:
            print(f"❌ No trades executed - strategy parameters may be too restrictive")
            
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
                
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_optimized_strategies()