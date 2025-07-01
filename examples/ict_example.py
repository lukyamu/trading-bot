#!/usr/bin/env python3
"""
ICT Strategy Example Usage

Demonstrates how to use the ICT Strategy Module for trading analysis.
This example shows how to:
- Load and prepare price data
- Run ICT analysis (FVG, Order Blocks, Liquidity Sweeps)
- Generate trading signals
- Export results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# Import ICT modules
from ict_strategy import ICTStrategy, analyze_ict_data, get_ict_signals
from config import ICT_CONFIG, TRADING_PAIRS
from utils.ict_utils import (
    validate_ohlcv_data, clean_price_data, setup_ict_logger,
    log_analysis_summary, export_analysis_results
)

# Setup logging
logger = setup_ict_logger('ict_example', 'INFO')

def generate_sample_data(symbol: str = 'EURUSD', periods: int = 200) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    logger.info(f"Generating sample data for {symbol} with {periods} periods")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 1.1000  # Starting price for EURUSD
    
    # Generate price movements
    returns = np.random.normal(0, 0.0005, len(date_range))  # Small random movements
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(date_range, prices)):
        # Add some randomness to create realistic OHLC
        volatility = abs(np.random.normal(0, 0.0002))
        
        high = close_price + volatility
        low = close_price - volatility
        
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = np.random.randint(1000, 10000)  # Random volume
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    return df

def run_basic_ict_analysis(df: pd.DataFrame) -> dict:
    """Run basic ICT analysis on price data"""
    logger.info("Starting basic ICT analysis...")
    
    # Validate data first
    if not validate_ohlcv_data(df):
        raise ValueError("Invalid OHLCV data provided")
    
    # Clean data
    df_clean = clean_price_data(df)
    logger.info(f"Data cleaned: {len(df_clean)} valid records")
    
    # Run comprehensive ICT analysis
    results = analyze_ict_data(df_clean)
    
    # Log summary
    log_analysis_summary(results, logger)
    
    return results

def generate_trading_signals(df: pd.DataFrame, current_price: float = None) -> dict:
    """Generate trading signals based on ICT analysis"""
    if current_price is None:
        current_price = df['close'].iloc[-1]
    
    logger.info(f"Generating trading signals for current price: {current_price:.5f}")
    
    # Get ICT-based signals
    signals = get_ict_signals(df, current_price)
    
    logger.info(f"Signal generated: {signals['direction']} with {signals['confidence']}% confidence")
    
    if signals['reasons']:
        logger.info("Signal reasons:")
        for reason in signals['reasons']:
            logger.info(f"  - {reason}")
    
    return signals

def demonstrate_live_analysis():
    """Demonstrate live analysis workflow"""
    logger.info("=== ICT Strategy Live Analysis Demo ===")
    
    # Initialize ICT strategy
    ict = ICTStrategy()
    
    # Generate sample data
    df = generate_sample_data('EURUSD', 100)
    
    # Run analysis step by step
    logger.info("Step 1: Identifying Fair Value Gaps...")
    fvgs = ict.identify_fair_value_gaps(df)
    logger.info(f"Found {len(fvgs)} Fair Value Gaps")
    
    logger.info("Step 2: Identifying Order Blocks...")
    obs = ict.identify_order_blocks(df)
    logger.info(f"Found {len(obs)} Order Blocks")
    
    logger.info("Step 3: Detecting Liquidity Sweeps...")
    sweeps = ict.detect_liquidity_sweeps(df)
    logger.info(f"Found {len(sweeps)} Liquidity Sweeps")
    
    logger.info("Step 4: Analyzing Market Structure...")
    structure = ict.analyze_market_structure(df)
    logger.info(f"Market Structure: {structure.trend} trend, {structure.structure}")
    
    logger.info("Step 5: Checking for Imbalances...")
    imbalances = ict.check_imbalances(df)
    logger.info(f"Found {len(imbalances)} Price Imbalances")
    
    # Get current active levels
    active_levels = ict.get_active_levels()
    logger.info(f"Active FVGs: {len(active_levels['fair_value_gaps'])}")
    logger.info(f"Active Order Blocks: {len(active_levels['order_blocks'])}")
    
    # Generate signals
    current_price = df['close'].iloc[-1]
    signals = ict.get_trade_signals(df, current_price)
    
    logger.info("=== Trading Signal Summary ===")
    logger.info(f"Direction: {signals['direction']}")
    logger.info(f"Entry Price: {signals['entry_price']:.5f}")
    logger.info(f"Stop Loss: {signals['stop_loss']}")
    logger.info(f"Take Profit: {signals['take_profit']}")
    logger.info(f"Confidence: {signals['confidence']}%")
    
    return {
        'fvgs': fvgs,
        'order_blocks': obs,
        'liquidity_sweeps': sweeps,
        'market_structure': structure,
        'imbalances': imbalances,
        'signals': signals,
        'active_levels': active_levels
    }

def main():
    """Main example function"""
    logger.info("Starting ICT Strategy Examples...")
    
    try:
        # Example 1: Basic Analysis
        logger.info("\n=== Example 1: Basic ICT Analysis ===")
        df = generate_sample_data('EURUSD', 200)
        results = run_basic_ict_analysis(df)
        
        # Example 2: Signal Generation
        logger.info("\n=== Example 2: Trading Signal Generation ===")
        signals = generate_trading_signals(df)
        
        # Example 3: Live Analysis Demo
        logger.info("\n=== Example 3: Live Analysis Workflow ===")
        live_results = demonstrate_live_analysis()
        
        logger.info("\n=== All Examples Completed Successfully! ===")
        
        return {
            'basic_analysis': results,
            'signals': signals,
            'live_analysis': live_results
        }
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    results = main()
    
    print("\n=== ICT Strategy Examples Complete ===")
    print("Check the logs above for detailed analysis results.")
    print("\nTo use this in your own code:")
    print("1. Import the ICT modules")
    print("2. Load your OHLCV data into a pandas DataFrame")
    print("3. Call analyze_ict_data(df) for comprehensive analysis")
    print("4. Call get_ict_signals(df, current_price) for trading signals")
