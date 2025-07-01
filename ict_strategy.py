"""
ICT Strategy Module

Implements Inner Circle Trader (ICT) concepts including:
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Liquidity sweeps
- Market structure analysis
- Imbalances
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Configuration for ICT parameters
ICT_CONFIG = {
    'fair_value_gaps': {
        'min_gap_size': 0.0001,  # Minimum gap size in price units
        'max_age_hours': 24,     # Maximum age of FVG in hours
    },
    'order_blocks': {
        'block_validation_candles': 5,  # Number of candles to validate OB
        'min_block_size': 0.0005,       # Minimum block size
    },
    'liquidity_sweeps': {
        'lookback_periods': 20,         # Periods to look back for highs/lows
        'sweep_threshold': 0.0002,      # Threshold for sweep detection
    },
    'market_structure': {
        'trend_periods': 50,            # Periods for trend analysis
        'structure_break_threshold': 0.001,  # Threshold for structure break
    }
}

@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    start_time: datetime
    end_time: datetime
    top: float
    bottom: float
    direction: str  # 'bullish' or 'bearish'
    filled: bool = False
    fill_time: Optional[datetime] = None
    strength: float = 0.0

@dataclass
class OrderBlock:
    """Order Block structure"""
    time: datetime
    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    validated: bool = False
    touched: bool = False
    strength: float = 0.0

@dataclass
class LiquiditySweep:
    """Liquidity Sweep structure"""
    time: datetime
    price: float
    direction: str  # 'buy_side' or 'sell_side'
    previous_level: float
    sweep_distance: float

@dataclass
class MarketStructure:
    """Market Structure analysis"""
    trend: str  # 'bullish', 'bearish', 'ranging'
    structure: str  # 'intact', 'broken'
    last_higher_high: Optional[float] = None
    last_higher_low: Optional[float] = None
    last_lower_high: Optional[float] = None
    last_lower_low: Optional[float] = None

class ICTStrategy:
    """ICT Strategy implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fair_value_gaps: List[FairValueGap] = []
        self.order_blocks: List[OrderBlock] = []
        self.liquidity_sweeps: List[LiquiditySweep] = []
        self.market_structure: Optional[MarketStructure] = None
    
    def identify_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Identify Fair Value Gaps in price data"""
        gaps = []
        
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2].low and candle[i].high
            if (df.iloc[i-2]['low'] > df.iloc[i]['high'] and 
                df.iloc[i-2]['low'] - df.iloc[i]['high'] >= ICT_CONFIG['fair_value_gaps']['min_gap_size']):
                
                gap = FairValueGap(
                    start_time=df.index[i-2],
                    end_time=df.index[i],
                    top=df.iloc[i-2]['low'],
                    bottom=df.iloc[i]['high'],
                    direction='bullish',
                    strength=self._calculate_fvg_strength(df, i-2, i)
                )
                gaps.append(gap)
            
            # Bearish FVG: gap between candle[i-2].high and candle[i].low
            elif (df.iloc[i-2]['high'] < df.iloc[i]['low'] and 
                  df.iloc[i]['low'] - df.iloc[i-2]['high'] >= ICT_CONFIG['fair_value_gaps']['min_gap_size']):
                
                gap = FairValueGap(
                    start_time=df.index[i-2],
                    end_time=df.index[i],
                    top=df.iloc[i]['low'],
                    bottom=df.iloc[i-2]['high'],
                    direction='bearish',
                    strength=self._calculate_fvg_strength(df, i-2, i)
                )
                gaps.append(gap)
        
        self.fair_value_gaps.extend(gaps)
        return gaps
    
    def _calculate_fvg_strength(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate the strength of a Fair Value Gap"""
        gap_size = abs(df.iloc[start_idx]['low'] - df.iloc[end_idx]['high'])
        volume_factor = df.iloc[start_idx:end_idx+1]['volume'].mean() if 'volume' in df.columns else 1.0
        return gap_size * volume_factor
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Identify Order Blocks"""
        order_blocks = []
        lookback = ICT_CONFIG['order_blocks']['block_validation_candles']
        
        for i in range(lookback, len(df) - lookback):
            current_candle = df.iloc[i]
            
            # Bullish Order Block: last bearish candle before bullish move
            if (current_candle['close'] < current_candle['open'] and
                self._is_bullish_move_after(df, i, lookback)):
                
                ob = OrderBlock(
                    time=df.index[i],
                    high=current_candle['high'],
                    low=current_candle['low'],
                    direction='bullish',
                    strength=self._calculate_ob_strength(df, i)
                )
                order_blocks.append(ob)
            
            # Bearish Order Block: last bullish candle before bearish move
            elif (current_candle['close'] > current_candle['open'] and
                  self._is_bearish_move_after(df, i, lookback)):
                
                ob = OrderBlock(
                    time=df.index[i],
                    high=current_candle['high'],
                    low=current_candle['low'],
                    direction='bearish',
                    strength=self._calculate_ob_strength(df, i)
                )
                order_blocks.append(ob)
        
        self.order_blocks.extend(order_blocks)
        return order_blocks
    
    def _is_bullish_move_after(self, df: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a bullish move after the given index"""
        for i in range(index + 1, min(index + lookback + 1, len(df))):
            if df.iloc[i]['close'] > df.iloc[index]['high']:
                return True
        return False
    
    def _is_bearish_move_after(self, df: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a bearish move after the given index"""
        for i in range(index + 1, min(index + lookback + 1, len(df))):
            if df.iloc[i]['close'] < df.iloc[index]['low']:
                return True
        return False
    
    def _calculate_ob_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of an Order Block"""
        candle = df.iloc[index]
        body_size = abs(candle['close'] - candle['open'])
        wick_size = (candle['high'] - candle['low']) - body_size
        volume_factor = candle['volume'] if 'volume' in df.columns else 1.0
        return (body_size / (body_size + wick_size)) * volume_factor
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[LiquiditySweep]:
        """Detect liquidity sweeps"""
        sweeps = []
        lookback = ICT_CONFIG['liquidity_sweeps']['lookback_periods']
        threshold = ICT_CONFIG['liquidity_sweeps']['sweep_threshold']
        
        for i in range(lookback, len(df)):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Look for recent highs and lows
            recent_data = df.iloc[i-lookback:i]
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            # Buy-side liquidity sweep (sweep above recent high)
            if current_high > recent_high + threshold:
                sweep = LiquiditySweep(
                    time=df.index[i],
                    price=current_high,
                    direction='buy_side',
                    previous_level=recent_high,
                    sweep_distance=current_high - recent_high
                )
                sweeps.append(sweep)
            
            # Sell-side liquidity sweep (sweep below recent low)
            if current_low < recent_low - threshold:
                sweep = LiquiditySweep(
                    time=df.index[i],
                    price=current_low,
                    direction='sell_side',
                    previous_level=recent_low,
                    sweep_distance=recent_low - current_low
                )
                sweeps.append(sweep)
        
        self.liquidity_sweeps.extend(sweeps)
        return sweeps
    
    def analyze_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Analyze overall market structure"""
        periods = ICT_CONFIG['market_structure']['trend_periods']
        recent_data = df.tail(periods)
        
        # Calculate swing highs and lows
        highs = recent_data['high'].rolling(window=5, center=True).max()
        lows = recent_data['low'].rolling(window=5, center=True).min()
        
        # Identify trend based on higher highs/higher lows or lower highs/lower lows
        trend = self._determine_trend(recent_data)
        structure = self._determine_structure_integrity(recent_data)
        
        market_structure = MarketStructure(
            trend=trend,
            structure=structure
        )
        
        self.market_structure = market_structure
        return market_structure
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine the current trend"""
        if len(df) < 10:
            return 'ranging'
        
        # Simple trend determination based on moving averages
        short_ma = df['close'].rolling(window=10).mean().iloc[-1]
        long_ma = df['close'].rolling(window=20).mean().iloc[-1]
        
        if short_ma > long_ma * 1.001:
            return 'bullish'
        elif short_ma < long_ma * 0.999:
            return 'bearish'
        else:
            return 'ranging'
    
    def _determine_structure_integrity(self, df: pd.DataFrame) -> str:
        """Determine if market structure is intact or broken"""
        # Simplified structure analysis
        recent_volatility = df['close'].pct_change().std()
        if recent_volatility > 0.02:  # High volatility indicates potential structure break
            return 'broken'
        return 'intact'
    
    def check_imbalances(self, df: pd.DataFrame) -> List[Dict]:
        """Check for price imbalances"""
        imbalances = []
        
        for i in range(1, len(df)):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            
            # Gap up (bullish imbalance)
            if current_candle['low'] > prev_candle['high']:
                imbalance = {
                    'time': df.index[i],
                    'type': 'bullish_imbalance',
                    'top': current_candle['low'],
                    'bottom': prev_candle['high'],
                    'size': current_candle['low'] - prev_candle['high']
                }
                imbalances.append(imbalance)
            
            # Gap down (bearish imbalance)
            elif current_candle['high'] < prev_candle['low']:
                imbalance = {
                    'time': df.index[i],
                    'type': 'bearish_imbalance',
                    'top': prev_candle['low'],
                    'bottom': current_candle['high'],
                    'size': prev_candle['low'] - current_candle['high']
                }
                imbalances.append(imbalance)
        
        return imbalances
    
    def get_trade_signals(self, df: pd.DataFrame, current_price: float) -> Dict[str, any]:
        """Generate trade signals based on ICT concepts"""
        signals = {
            'direction': None,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0,
            'reasons': []
        }
        
        # Analyze market structure
        structure = self.analyze_market_structure(df)
        
        # Check for order block interactions
        for ob in self.order_blocks[-10:]:  # Check last 10 order blocks
            if not ob.touched and ob.low <= current_price <= ob.high:
                if ob.direction == 'bullish' and structure.trend in ['bullish', 'ranging']:
                    signals['direction'] = 'long'
                    signals['confidence'] += 40
                    signals['reasons'].append(f'Bullish order block interaction at {ob.time}')
                    signals['stop_loss'] = ob.low * 0.999
                    signals['take_profit'] = current_price * 1.02
                
                elif ob.direction == 'bearish' and structure.trend in ['bearish', 'ranging']:
                    signals['direction'] = 'short'
                    signals['confidence'] += 40
                    signals['reasons'].append(f'Bearish order block interaction at {ob.time}')
                    signals['stop_loss'] = ob.high * 1.001
                    signals['take_profit'] = current_price * 0.98
        
        # Check for Fair Value Gap interactions
        for fvg in self.fair_value_gaps[-5:]:  # Check last 5 FVGs
            if not fvg.filled and fvg.bottom <= current_price <= fvg.top:
                if fvg.direction == 'bullish':
                    signals['confidence'] += 30
                    signals['reasons'].append(f'Bullish FVG interaction at {fvg.start_time}')
                else:
                    signals['confidence'] += 30
                    signals['reasons'].append(f'Bearish FVG interaction at {fvg.start_time}')
        
        # Check for recent liquidity sweeps
        recent_sweeps = [s for s in self.liquidity_sweeps[-3:]]
        for sweep in recent_sweeps:
            if sweep.direction == 'buy_side' and structure.trend == 'bearish':
                signals['confidence'] += 20
                signals['reasons'].append(f'Buy-side liquidity swept at {sweep.time}')
            elif sweep.direction == 'sell_side' and structure.trend == 'bullish':
                signals['confidence'] += 20
                signals['reasons'].append(f'Sell-side liquidity swept at {sweep.time}')
        
        # Adjust confidence based on market structure
        if structure.structure == 'intact':
            signals['confidence'] += 10
        else:
            signals['confidence'] -= 10
        
        # Cap confidence at 100
        signals['confidence'] = min(signals['confidence'], 100)
        
        return signals
    
    def update_fvg_status(self, df: pd.DataFrame, current_price: float, current_time: datetime):
        """Update Fair Value Gap fill status"""
        for fvg in self.fair_value_gaps:
            if not fvg.filled:
                if fvg.direction == 'bullish' and current_price <= fvg.bottom:
                    fvg.filled = True
                    fvg.fill_time = current_time
                elif fvg.direction == 'bearish' and current_price >= fvg.top:
                    fvg.filled = True
                    fvg.fill_time = current_time
    
    def update_ob_status(self, current_price: float):
        """Update Order Block touch status"""
        for ob in self.order_blocks:
            if not ob.touched and ob.low <= current_price <= ob.high:
                ob.touched = True
    
    def get_active_levels(self) -> Dict[str, List]:
        """Get currently active ICT levels"""
        active_fvgs = [fvg for fvg in self.fair_value_gaps if not fvg.filled]
        active_obs = [ob for ob in self.order_blocks if not ob.touched]
        recent_sweeps = self.liquidity_sweeps[-5:]
        
        return {
            'fair_value_gaps': active_fvgs,
            'order_blocks': active_obs,
            'liquidity_sweeps': recent_sweeps,
            'market_structure': self.market_structure
        }
    
    def reset_analysis(self):
        """Reset all analysis data"""
        self.fair_value_gaps.clear()
        self.order_blocks.clear()
        self.liquidity_sweeps.clear()
        self.market_structure = None

# Global ICT strategy instance
ict_strategy = ICTStrategy()

# Utility functions for easy access
def analyze_ict_data(df: pd.DataFrame) -> Dict[str, any]:
    """Comprehensive ICT analysis of price data"""
    ict_strategy.reset_analysis()
    
    # Run all ICT analyses
    fvgs = ict_strategy.identify_fair_value_gaps(df)
    obs = ict_strategy.identify_order_blocks(df)
    sweeps = ict_strategy.detect_liquidity_sweeps(df)
    structure = ict_strategy.analyze_market_structure(df)
    imbalances = ict_strategy.check_imbalances(df)
    
    return {
        'fair_value_gaps': fvgs,
        'order_blocks': obs,
        'liquidity_sweeps': sweeps,
        'market_structure': structure,
        'imbalances': imbalances,
        'active_levels': ict_strategy.get_active_levels()
    }

def get_ict_signals(df: pd.DataFrame, current_price: float) -> Dict[str, any]:
    """Get ICT-based trading signals"""
    # Ensure analysis is up to date
    analyze_ict_data(df)
    return ict_strategy.get_trade_signals(df, current_price)
