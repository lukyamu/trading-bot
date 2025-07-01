"""
ICT Utility Functions

Helper functions for ICT strategy calculations, data processing,
and technical analysis operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import asdict

# Import ICT data structures
try:
    from ict_strategy import FairValueGap, OrderBlock, LiquiditySweep, MarketStructure
except ImportError:
    # Fallback if running independently
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Optional
    
    @dataclass
    class FairValueGap:
        start_time: datetime
        end_time: datetime
        top: float
        bottom: float
        direction: str
        filled: bool = False
        fill_time: Optional[datetime] = None
        strength: float = 0.0

logger = logging.getLogger(__name__)

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate OHLCV data format and completeness"""
    required_columns = ['open', 'high', 'low', 'close']
    
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns. Expected: {required_columns}")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    # Check for NaN values
    if df[required_columns].isnull().any().any():
        logger.warning("DataFrame contains NaN values")
        return False
    
    # Validate OHLC relationships
    invalid_rows = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid_rows.any():
        logger.error(f"Invalid OHLC relationships found in {invalid_rows.sum()} rows")
        return False
    
    return True

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare price data for ICT analysis"""
    df_clean = df.copy()
    
    # Remove any duplicate timestamps
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Forward fill any remaining NaN values
    df_clean = df_clean.fillna(method='ffill')
    
    # Ensure proper data types
    numeric_columns = ['open', 'high', 'low', 'close']
    if 'volume' in df_clean.columns:
        numeric_columns.append('volume')
    
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove any rows with invalid data after conversion
    df_clean = df_clean.dropna()
    
    return df_clean

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate price volatility using standard deviation of returns"""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=period).std()
    return volatility

# =============================================================================
# SWING POINT DETECTION
# =============================================================================

def find_swing_highs(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Find swing high points in price data"""
    highs = df['high'].rolling(window=window*2+1, center=True).max()
    swing_highs = df['high'] == highs
    return swing_highs

def find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Find swing low points in price data"""
    lows = df['low'].rolling(window=window*2+1, center=True).min()
    swing_lows = df['low'] == lows
    return swing_lows

def get_recent_swing_points(df: pd.DataFrame, lookback: int = 50, window: int = 5) -> Dict[str, List[Tuple]]:
    """Get recent swing highs and lows"""
    recent_data = df.tail(lookback)
    
    swing_highs = find_swing_highs(recent_data, window)
    swing_lows = find_swing_lows(recent_data, window)
    
    high_points = []
    low_points = []
    
    for idx, is_high in swing_highs.items():
        if is_high:
            high_points.append((idx, recent_data.loc[idx, 'high']))
    
    for idx, is_low in swing_lows.items():
        if is_low:
            low_points.append((idx, recent_data.loc[idx, 'low']))
    
    return {
        'swing_highs': high_points,
        'swing_lows': low_points
    }

# =============================================================================
# LEVEL MANAGEMENT UTILITIES
# =============================================================================

def is_level_touched(price: float, level: float, buffer: float = 0.0001) -> bool:
    """Check if price has touched a specific level within buffer"""
    return abs(price - level) <= buffer

def is_price_in_range(price: float, top: float, bottom: float) -> bool:
    """Check if price is within a range"""
    return bottom <= price <= top

def calculate_level_strength(touches: int, age_hours: float, volume_factor: float = 1.0) -> float:
    """Calculate the strength of a price level"""
    # More touches = stronger level, but diminishing returns
    touch_strength = min(touches * 0.2, 1.0)
    
    # Newer levels are generally stronger
    age_factor = max(0.1, 1.0 - (age_hours / 168))  # Decay over a week
    
    # Volume factor
    volume_strength = min(volume_factor, 2.0)
    
    return touch_strength * age_factor * volume_strength

def merge_overlapping_levels(levels: List[Dict], threshold: float = 0.0005) -> List[Dict]:
    """Merge overlapping price levels"""
    if not levels:
        return []
    
    # Sort levels by price
    sorted_levels = sorted(levels, key=lambda x: x.get('price', 0))
    merged = [sorted_levels[0]]
    
    for current in sorted_levels[1:]:
        last_merged = merged[-1]
        
        # Check if levels are close enough to merge
        if abs(current['price'] - last_merged['price']) <= threshold:
            # Merge levels - keep the stronger one or average
            if current.get('strength', 0) > last_merged.get('strength', 0):
                merged[-1] = current
            else:
                # Average the prices, keep higher strength
                merged[-1]['price'] = (current['price'] + last_merged['price']) / 2
                merged[-1]['strength'] = max(current.get('strength', 0), last_merged.get('strength', 0))
        else:
            merged.append(current)
    
    return merged

# =============================================================================
# TIME AND SESSION UTILITIES
# =============================================================================

def get_market_session(timestamp: datetime, timezone_str: str = 'UTC') -> str:
    """Determine which market session a timestamp belongs to"""
    hour = timestamp.hour
    
    # Simplified session detection (UTC times)
    if 0 <= hour < 9:
        return 'tokyo'
    elif 8 <= hour < 17:
        return 'london'
    elif 13 <= hour < 22:
        return 'new_york'
    else:
        return 'sydney'

def is_high_impact_time(timestamp: datetime) -> bool:
    """Check if timestamp is during high-impact trading hours"""
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    # Weekend
    if weekday >= 5:
        return False
    
    # London-NY overlap (13:00-17:00 UTC)
    if 13 <= hour < 17:
        return True
    
    # London open (08:00-10:00 UTC)
    if 8 <= hour < 10:
        return True
    
    # NY open (13:00-15:00 UTC)
    if 13 <= hour < 15:
        return True
    
    return False

def calculate_time_since(start_time: datetime, current_time: datetime) -> Dict[str, float]:
    """Calculate time difference in various units"""
    delta = current_time - start_time
    
    return {
        'seconds': delta.total_seconds(),
        'minutes': delta.total_seconds() / 60,
        'hours': delta.total_seconds() / 3600,
        'days': delta.days
    }

# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def calculate_percentile_levels(df: pd.DataFrame, periods: int = 100) -> Dict[str, float]:
    """Calculate key percentile levels for recent price action"""
    recent_data = df.tail(periods)
    all_prices = pd.concat([recent_data['high'], recent_data['low']])
    
    return {
        'p95': all_prices.quantile(0.95),
        'p90': all_prices.quantile(0.90),
        'p75': all_prices.quantile(0.75),
        'p50': all_prices.quantile(0.50),
        'p25': all_prices.quantile(0.25),
        'p10': all_prices.quantile(0.10),
        'p5': all_prices.quantile(0.05)
    }

def calculate_support_resistance_levels(df: pd.DataFrame, periods: int = 50, num_levels: int = 5) -> Dict[str, List[float]]:
    """Calculate dynamic support and resistance levels"""
    recent_data = df.tail(periods)
    
    # Get swing points
    swing_points = get_recent_swing_points(recent_data, periods)
    
    # Extract prices
    resistance_prices = [point[1] for point in swing_points['swing_highs']]
    support_prices = [point[1] for point in swing_points['swing_lows']]
    
    # Sort and get top levels
    resistance_levels = sorted(resistance_prices, reverse=True)[:num_levels]
    support_levels = sorted(support_prices)[:num_levels]
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }

# =============================================================================
# PATTERN RECOGNITION UTILITIES
# =============================================================================

def detect_consolidation(df: pd.DataFrame, periods: int = 20, threshold: float = 0.002) -> bool:
    """Detect if price is in consolidation"""
    recent_data = df.tail(periods)
    
    high_range = recent_data['high'].max() - recent_data['high'].min()
    low_range = recent_data['low'].max() - recent_data['low'].min()
    avg_price = recent_data['close'].mean()
    
    # Check if range is small relative to average price
    consolidation_range = max(high_range, low_range) / avg_price
    
    return consolidation_range < threshold

def detect_breakout(df: pd.DataFrame, periods: int = 20, breakout_threshold: float = 0.001) -> Dict[str, bool]:
    """Detect breakout from recent range"""
    if len(df) < periods + 1:
        return {'bullish_breakout': False, 'bearish_breakout': False}
    
    # Get recent range
    recent_data = df.iloc[-(periods+1):-1]  # Exclude current candle
    current_candle = df.iloc[-1]
    
    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()
    
    # Check for breakouts
    bullish_breakout = current_candle['close'] > recent_high * (1 + breakout_threshold)
    bearish_breakout = current_candle['close'] < recent_low * (1 - breakout_threshold)
    
    return {
        'bullish_breakout': bullish_breakout,
        'bearish_breakout': bearish_breakout,
        'recent_high': recent_high,
        'recent_low': recent_low
    }

# =============================================================================
# EXPORT AND SERIALIZATION UTILITIES
# =============================================================================

def serialize_ict_levels(levels: List[Union[FairValueGap, OrderBlock, LiquiditySweep]]) -> List[Dict]:
    """Serialize ICT levels to dictionary format"""
    serialized = []
    
    for level in levels:
        if hasattr(level, '__dict__'):
            level_dict = asdict(level)
            # Convert datetime objects to ISO format
            for key, value in level_dict.items():
                if isinstance(value, datetime):
                    level_dict[key] = value.isoformat()
            serialized.append(level_dict)
    
    return serialized

def export_analysis_results(results: Dict, filename: str = None) -> str:
    """Export ICT analysis results to JSON format"""
    import json
    
    # Serialize datetime objects
    def datetime_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    json_data = json.dumps(results, default=datetime_serializer, indent=2)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(json_data)
        logger.info(f"Analysis results exported to {filename}")
    
    return json_data

# =============================================================================
# LOGGING AND DEBUGGING UTILITIES
# =============================================================================

def setup_ict_logger(name: str = 'ict_utils', level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """Setup logger for ICT utilities"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_analysis_summary(results: Dict, logger: logging.Logger = None):
    """Log a summary of ICT analysis results"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=== ICT Analysis Summary ===")
    
    if 'fair_value_gaps' in results:
        fvg_count = len(results['fair_value_gaps'])
        logger.info(f"Fair Value Gaps found: {fvg_count}")
    
    if 'order_blocks' in results:
        ob_count = len(results['order_blocks'])
        logger.info(f"Order Blocks found: {ob_count}")
    
    if 'liquidity_sweeps' in results:
        sweep_count = len(results['liquidity_sweeps'])
        logger.info(f"Liquidity Sweeps found: {sweep_count}")
    
    if 'market_structure' in results:
        structure = results['market_structure']
        if hasattr(structure, 'trend'):
            logger.info(f"Market Structure: {structure.trend} trend, {structure.structure}")
    
    logger.info("=== End Analysis Summary ===")
