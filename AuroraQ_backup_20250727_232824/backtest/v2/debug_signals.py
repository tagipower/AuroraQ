#!/usr/bin/env python3
"""
Debug script to test signal generation
"""

import sys
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

from layers.controller_layer import BacktestController

def test_signal_generation():
    """Test signal generation with debug output"""
    print("=== Debug Signal Generation Test ===")
    
    # Initialize controller
    controller = BacktestController()
    controller.initialize_strategies()
    
    # Load data
    df = pd.read_csv('../../data/btc_5m_sample.csv')
    print(f'Loaded data shape: {df.shape}')
    
    # Run a short backtest to see signal generation
    print(f'\n=== Running short backtest to see signal generation ===')
    
    try:
        results = controller.run_backtest(
            price_data_path='../../data/btc_5m_sample.csv',
            window_size=50  # Small window for debugging
        )
        
        print(f"Backtest results: {results}")
        
        # Check if any actual trades were made
        metrics = results.get('metrics', {})
        trades = metrics.get('total_trades', 0)
        print(f"Total trades executed: {trades}")
        
        if trades == 0:
            print("⚠️ No trades were executed - strategies are still only returning 'hold' signals")
        else:
            print(f"✅ Found {trades} trades - signal generation is working!")
            
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
                
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_signal_generation()