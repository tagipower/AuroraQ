#!/usr/bin/env python3
"""
Optimized Strategy Integration Test
최적화된 전략이 실제로 백테스트 시스템에서 작동하는지 테스트
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

def test_optimized_integration():
    """최적화된 전략 통합 테스트"""
    print("=== Testing Optimized Strategy Integration ===")
    
    # Initialize controller
    controller = BacktestController()
    controller.initialize_strategies()
    
    # Check if optimized strategy is loaded
    if hasattr(controller, 'strategy_registry'):
        registry = controller.strategy_registry
        available_strategies = registry.get_all_strategy_names()
        print(f'Available strategies: {available_strategies}')
        
        if "OptimizedRuleStrategyE" in available_strategies:
            print("✅ OptimizedRuleStrategyE successfully registered!")
        else:
            print("❌ OptimizedRuleStrategyE not found in registry")
            print("Available strategies:", available_strategies)
    
    # Load data
    df = pd.read_csv('../../data/btc_5m_sample.csv')
    print(f'Loaded data shape: {df.shape}')
    
    print(f'\n=== Running backtest with optimized strategy ===')
    
    try:
        # Run backtest
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
        optimized_results = None
        
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
            
            # Track optimized strategy results
            if "OptimizedRuleStrategyE" in strategy_name or "Optimized" in strategy_name:
                optimized_results = {
                    "strategy_name": strategy_name,
                    "trades": trades,
                    "roi": roi,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd
                }
        
        print(f"\n=== Overall Statistics ===")
        print(f"Total Signals: {stats.get('total_signals', 0):,}")
        print(f"Executed Trades: {stats.get('executed_trades', 0):,}")
        print(f"Total Strategy Trades: {total_trades}")
        print(f"Profitable Strategies: {profitable_strategies}/{len(all_metrics)}")
        print(f"Execution Time: {stats.get('execution_time', 0):.1f}s")
        
        # Optimized Strategy Performance Check
        if optimized_results:
            print(f"\n=== Optimized Strategy Performance ===")
            print(f"Strategy: {optimized_results['strategy_name']}")
            print(f"Trades Executed: {optimized_results['trades']}")
            print(f"ROI: {optimized_results['roi']:.2f}%")
            print(f"Win Rate: {optimized_results['win_rate']:.1f}%")
            print(f"Profit Factor: {optimized_results['profit_factor']:.2f}")
            
            # Performance Target Comparison
            print(f"\n=== Target vs Actual Comparison ===")
            target_win_rate = 35.0  # From optimized_rule_params.yaml
            target_profit_factor = 0.6
            target_sharpe = 0.5
            
            win_rate_check = "✅" if optimized_results['win_rate'] >= target_win_rate else "❌"
            pf_check = "✅" if optimized_results['profit_factor'] >= target_profit_factor else "❌"
            sharpe_check = "✅" if optimized_results['sharpe'] >= target_sharpe else "❌"
            
            print(f"Win Rate: {optimized_results['win_rate']:.1f}% | Target: {target_win_rate}% | {win_rate_check}")
            print(f"Profit Factor: {optimized_results['profit_factor']:.2f} | Target: {target_profit_factor} | {pf_check}")
            print(f"Sharpe Ratio: {optimized_results['sharpe']:.2f} | Target: {target_sharpe} | {sharpe_check}")
            
            # Overall Assessment
            targets_met = sum([
                optimized_results['win_rate'] >= target_win_rate,
                optimized_results['profit_factor'] >= target_profit_factor,
                optimized_results['sharpe'] >= target_sharpe
            ])
            
            if targets_met >= 2:
                print(f"\n🎉 OPTIMIZATION SUCCESS! {targets_met}/3 targets achieved.")
                if optimized_results['trades'] > 0:
                    print(f"✅ Strategy is generating actual trades (not just HOLD signals)")
                else:
                    print(f"⚠️ Strategy not generating trades - may need parameter tuning")
            else:
                print(f"\n⚠️ Optimization needs improvement. Only {targets_met}/3 targets met.")
                
                # Provide specific recommendations
                if optimized_results['win_rate'] < target_win_rate:
                    print(f"  📈 Improve Win Rate: Tighten entry conditions")
                if optimized_results['profit_factor'] < target_profit_factor:
                    print(f"  💰 Improve Profit Factor: Adjust risk/reward ratio")
                if optimized_results['sharpe'] < target_sharpe:
                    print(f"  📊 Improve Sharpe: Reduce volatility or increase returns")
        else:
            print(f"\n❌ No optimized strategy found in results!")
            print(f"Available strategy results: {list(all_metrics.keys())}")
        
        # Generate reports
        print(f"\n=== Generating Reports ===")
        try:
            scores_file = generate_strategy_scores_from_backtest(results)
            print(f"Strategy scores saved to: {scores_file}")
        except Exception as e:
            print(f"Failed to generate strategy scores: {e}")
        
        try:
            html_file = generate_html_report(results, title="Optimized Strategy Performance Report")
            print(f"HTML report saved to: {html_file}")
        except Exception as e:
            print(f"Failed to generate HTML report: {e}")
        
        # Strategy Comparison
        if len(all_metrics) > 1:
            print(f"\n=== Strategy Performance Ranking ===")
            strategy_performance = []
            for name, metrics in all_metrics.items():
                composite_score = (
                    float(metrics.roi) * 0.4 + 
                    float(metrics.win_rate) * 0.3 + 
                    float(metrics.profit_factor) * 0.2 + 
                    float(metrics.sharpe_ratio) * 0.1
                )
                strategy_performance.append({
                    "name": name,
                    "score": composite_score,
                    "roi": float(metrics.roi) * 100,
                    "trades": metrics.total_trades
                })
            
            # Sort by composite score
            strategy_performance.sort(key=lambda x: x["score"], reverse=True)
            
            for i, strategy in enumerate(strategy_performance, 1):
                print(f"{i}. {strategy['name']}: Score {strategy['score']:.3f} "
                      f"(ROI: {strategy['roi']:.2f}%, Trades: {strategy['trades']})")
            
            # Check if optimized strategy is #1
            if strategy_performance[0]["name"] == "OptimizedRuleStrategyE":
                print(f"\n🏆 Optimized strategy is the top performer!")
            else:
                print(f"\n📊 Optimized strategy ranking: "
                      f"#{[s['name'] for s in strategy_performance].index('OptimizedRuleStrategyE')+1 if 'OptimizedRuleStrategyE' in [s['name'] for s in strategy_performance] else 'Not found'}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return False
                
    print(f"\n=== Integration Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_optimized_integration()
    if success:
        print("✅ Integration test completed successfully!")
    else:
        print("❌ Integration test failed!")