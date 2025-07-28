#!/usr/bin/env python3
"""
Direct backtest execution with optimized parameters
최적화된 파라미터로 직접 백테스트 실행
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

print("=== Direct Backtest Execution ===")
print(f"Start time: {datetime.now()}")

# Import and run the actual run_loop code
from loops.run_loop import *

# Run the backtest
if __name__ == "__main__":
    print("\n[INFO] RuleE 초기화 시 최적화된 파라미터 사용:")
    print("  - Take Profit: 2.5% (기존 3.0%)")
    print("  - Stop Loss: 1.2% (기존 1.8%)")
    print("  - RSI Threshold: 60 (기존 55)")
    print("  - Risk/Reward Ratio: 2.08:1 (기존 1.67:1)")
    
    print("\n백테스트 시작...\n")
    
    # Execute the main loop
    try:
        # Initialize components
        selector = StrategySelector(sentiment_file='data/labeled_sentiment_data_combined.csv')
        state_preprocessor = StatePreprocessor(sentiment_file='data/labeled_sentiment_data_combined.csv')
        risk_controller = EnhancedRiskController()
        event_manager = EventManager()
        order_client = MockBinanceOrderClient()
        
        # Setup strategy names
        rule_strategy_names = [f"RuleStrategy{letter}" for letter in ["A", "B", "C", "D", "E"]]
        
        # Initialize MAB selector
        mab_selector = MABSelector(
            strategy_names=rule_strategy_names,
            algorithm='epsilon_greedy',
            epsilon=0.15
        )
        
        # Main loop
        data_generator = DataGeneratorProxy(window_size=20)
        trade_history = []
        accumulated_rewards = {}
        
        print(f"Starting backtest loop with {len(rule_strategy_names)} strategies...")
        
        for i, (timestamp, price_df) in enumerate(data_generator.generate_data_stream()):
            if i >= 1000:  # Limit iterations for testing
                break
                
            # Select strategy
            selected_strategy_name = mab_selector.select()
            
            # Get signal
            selection_result = selector.select(price_df)
            
            if selection_result['strategy'] == 'RuleStrategyE':
                print(f"[{i}] RuleStrategyE selected - using optimized parameters")
            
            # Process signal and execute
            if selection_result['signal'].get('action') != 'HOLD':
                trade_history.append({
                    'timestamp': timestamp,
                    'strategy': selection_result['strategy'],
                    'action': selection_result['signal']['action'],
                    'price': price_df['close'].iloc[-1]
                })
            
            # Update MAB
            reward = calculate_combined_reward(selection_result)
            mab_selector.update(selected_strategy_name, reward)
        
        print(f"\nBacktest completed!")
        print(f"Total trades executed: {len(trade_history)}")
        
        # Show strategy performance
        strategy_trades = {}
        for trade in trade_history:
            strategy = trade['strategy']
            if strategy not in strategy_trades:
                strategy_trades[strategy] = 0
            strategy_trades[strategy] += 1
        
        print(f"\nTrades by strategy:")
        for strategy, count in sorted(strategy_trades.items()):
            print(f"  {strategy}: {count} trades")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\nEnd time: {datetime.now()}")