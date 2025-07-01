"""
Configuration file for ICT Trading Bot

Contains all configurable parameters for ICT strategy components
including Fair Value Gaps, Order Blocks, Liquidity Sweeps, and Market Structure analysis.
"""

import os
from typing import Dict, Any

# =============================================================================
# ICT STRATEGY CONFIGURATION
# =============================================================================

ICT_CONFIG: Dict[str, Any] = {
    # Fair Value Gaps Configuration
    'fair_value_gaps': {
        'min_gap_size': float(os.getenv('FVG_MIN_GAP_SIZE', 0.0001)),
        'max_age_hours': int(os.getenv('FVG_MAX_AGE_HOURS', 24)),
        'strength_multiplier': float(os.getenv('FVG_STRENGTH_MULTIPLIER', 1.0)),
        'fill_threshold': float(os.getenv('FVG_FILL_THRESHOLD', 0.5)),  # Percentage of gap that needs to be filled
    },
    
    # Order Blocks Configuration
    'order_blocks': {
        'block_validation_candles': int(os.getenv('OB_VALIDATION_CANDLES', 5)),
        'min_block_size': float(os.getenv('OB_MIN_BLOCK_SIZE', 0.0005)),
        'max_age_hours': int(os.getenv('OB_MAX_AGE_HOURS', 48)),
        'strength_weight_volume': float(os.getenv('OB_VOLUME_WEIGHT', 0.3)),
        'strength_weight_body': float(os.getenv('OB_BODY_WEIGHT', 0.7)),
        'touch_buffer': float(os.getenv('OB_TOUCH_BUFFER', 0.0001)),  # Buffer for OB touch detection
    },
    
    # Liquidity Sweeps Configuration
    'liquidity_sweeps': {
        'lookback_periods': int(os.getenv('LS_LOOKBACK_PERIODS', 20)),
        'sweep_threshold': float(os.getenv('LS_SWEEP_THRESHOLD', 0.0002)),
        'min_sweep_distance': float(os.getenv('LS_MIN_DISTANCE', 0.0001)),
        'confirmation_candles': int(os.getenv('LS_CONFIRMATION_CANDLES', 2)),
        'volume_spike_threshold': float(os.getenv('LS_VOLUME_SPIKE', 1.5)),  # Volume spike multiplier
    },
    
    # Market Structure Configuration
    'market_structure': {
        'trend_periods': int(os.getenv('MS_TREND_PERIODS', 50)),
        'structure_break_threshold': float(os.getenv('MS_BREAK_THRESHOLD', 0.001)),
        'swing_detection_periods': int(os.getenv('MS_SWING_PERIODS', 5)),
        'trend_strength_periods': int(os.getenv('MS_STRENGTH_PERIODS', 20)),
        'volatility_threshold': float(os.getenv('MS_VOLATILITY_THRESHOLD', 0.02)),
    },
    
    # Imbalances Configuration
    'imbalances': {
        'min_imbalance_size': float(os.getenv('IMB_MIN_SIZE', 0.0001)),
        'max_age_hours': int(os.getenv('IMB_MAX_AGE_HOURS', 12)),
        'fill_threshold': float(os.getenv('IMB_FILL_THRESHOLD', 0.8)),
    },
    
    # Signal Generation Configuration
    'signals': {
        'base_confidence': int(os.getenv('SIG_BASE_CONFIDENCE', 0)),
        'fvg_confidence_weight': int(os.getenv('SIG_FVG_WEIGHT', 30)),
        'ob_confidence_weight': int(os.getenv('SIG_OB_WEIGHT', 40)),
        'sweep_confidence_weight': int(os.getenv('SIG_SWEEP_WEIGHT', 20)),
        'structure_confidence_weight': int(os.getenv('SIG_STRUCTURE_WEIGHT', 10)),
        'min_signal_confidence': int(os.getenv('SIG_MIN_CONFIDENCE', 50)),
        'max_signal_confidence': int(os.getenv('SIG_MAX_CONFIDENCE', 100)),
    },
    
    # Risk Management Configuration
    'risk_management': {
        'default_stop_loss_pct': float(os.getenv('RM_STOP_LOSS_PCT', 0.001)),  # 0.1%
        'default_take_profit_pct': float(os.getenv('RM_TAKE_PROFIT_PCT', 0.002)),  # 0.2%
        'max_risk_per_trade': float(os.getenv('RM_MAX_RISK_PCT', 0.02)),  # 2%
        'position_sizing_method': os.getenv('RM_POSITION_METHOD', 'fixed_risk'),  # 'fixed_risk' or 'fixed_size'
    },
    
    # Timeframe Configuration
    'timeframes': {
        'primary_tf': os.getenv('TF_PRIMARY', '1h'),  # Primary analysis timeframe
        'confirmation_tf': os.getenv('TF_CONFIRMATION', '15m'),  # Confirmation timeframe
        'entry_tf': os.getenv('TF_ENTRY', '5m'),  # Entry timeframe
        'supported_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    },
    
    # Data Configuration
    'data': {
        'required_history_periods': int(os.getenv('DATA_HISTORY_PERIODS', 200)),
        'update_frequency_seconds': int(os.getenv('DATA_UPDATE_FREQ', 60)),
        'data_source': os.getenv('DATA_SOURCE', 'binance'),
    },
    
    # Logging Configuration
    'logging': {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'file_path': os.getenv('LOG_FILE_PATH', 'logs/ict_strategy.log'),
        'max_file_size_mb': int(os.getenv('LOG_MAX_SIZE_MB', 10)),
        'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
    }
}

# =============================================================================
# TRADING PAIRS CONFIGURATION
# =============================================================================

TRADING_PAIRS = {
    'forex': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDCAD'
    ],
    'crypto': [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT',
        'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'ATOMUSDT'
    ],
    'indices': [
        'SPX500', 'NAS100', 'US30', 'GER40', 'UK100', 'FRA40', 'JPN225', 'AUS200'
    ]
}

# =============================================================================
# MARKET SESSION CONFIGURATION
# =============================================================================

MARKET_SESSIONS = {
    'london': {
        'start_hour': 8,
        'end_hour': 17,
        'timezone': 'Europe/London',
        'active_pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURCHF']
    },
    'new_york': {
        'start_hour': 13,
        'end_hour': 22,
        'timezone': 'America/New_York',
        'active_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    },
    'tokyo': {
        'start_hour': 0,
        'end_hour': 9,
        'timezone': 'Asia/Tokyo',
        'active_pairs': ['USDJPY', 'AUDJPY', 'EURJPY', 'GBPJPY']
    },
    'sydney': {
        'start_hour': 22,
        'end_hour': 7,
        'timezone': 'Australia/Sydney',
        'active_pairs': ['AUDUSD', 'NZDUSD', 'AUDCAD', 'AUDJPY']
    }
}

# =============================================================================
# POWER OF THREE CONFIGURATION
# =============================================================================

POWER_OF_THREE_CONFIG = {
    'accumulation': {
        'duration_minutes': int(os.getenv('POT_ACCUM_DURATION', 60)),
        'volatility_threshold': float(os.getenv('POT_ACCUM_VOLATILITY', 0.001)),
        'volume_threshold': float(os.getenv('POT_ACCUM_VOLUME', 0.8)),
    },
    'manipulation': {
        'max_duration_minutes': int(os.getenv('POT_MANIP_MAX_DURATION', 30)),
        'min_move_percentage': float(os.getenv('POT_MANIP_MIN_MOVE', 0.002)),
        'volume_spike_threshold': float(os.getenv('POT_MANIP_VOLUME_SPIKE', 1.5)),
    },
    'distribution': {
        'min_duration_minutes': int(os.getenv('POT_DIST_MIN_DURATION', 30)),
        'trend_strength_threshold': float(os.getenv('POT_DIST_TREND_STRENGTH', 0.7)),
        'volume_decline_threshold': float(os.getenv('POT_DIST_VOLUME_DECLINE', 0.6)),
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Get a configuration value with fallback to default"""
    try:
        return ICT_CONFIG[section][key]
    except KeyError:
        return default

def update_config_value(section: str, key: str, value: Any) -> bool:
    """Update a configuration value"""
    try:
        if section not in ICT_CONFIG:
            ICT_CONFIG[section] = {}
        ICT_CONFIG[section][key] = value
        return True
    except Exception:
        return False

def validate_config() -> Dict[str, list]:
    """Validate configuration values and return any issues"""
    issues = {
        'errors': [],
        'warnings': []
    }
    
    # Validate required sections
    required_sections = ['fair_value_gaps', 'order_blocks', 'liquidity_sweeps', 'market_structure']
    for section in required_sections:
        if section not in ICT_CONFIG:
            issues['errors'].append(f"Missing required section: {section}")
    
    # Validate numeric ranges
    if ICT_CONFIG['fair_value_gaps']['min_gap_size'] <= 0:
        issues['errors'].append("FVG min_gap_size must be positive")
    
    if ICT_CONFIG['order_blocks']['block_validation_candles'] < 1:
        issues['errors'].append("OB block_validation_candles must be at least 1")
    
    if ICT_CONFIG['signals']['min_signal_confidence'] > ICT_CONFIG['signals']['max_signal_confidence']:
        issues['errors'].append("Signal min_confidence cannot be greater than max_confidence")
    
    # Validate timeframes
    supported_tfs = ICT_CONFIG['timeframes']['supported_timeframes']
    for tf_key in ['primary_tf', 'confirmation_tf', 'entry_tf']:
        if ICT_CONFIG['timeframes'][tf_key] not in supported_tfs:
            issues['warnings'].append(f"Timeframe {ICT_CONFIG['timeframes'][tf_key]} not in supported list")
    
    return issues

def get_active_pairs_for_session(session_name: str) -> list:
    """Get active trading pairs for a specific market session"""
    return MARKET_SESSIONS.get(session_name, {}).get('active_pairs', [])

def is_market_session_active(session_name: str, current_hour: int) -> bool:
    """Check if a market session is currently active"""
    session = MARKET_SESSIONS.get(session_name)
    if not session:
        return False
    
    start_hour = session['start_hour']
    end_hour = session['end_hour']
    
    # Handle sessions that cross midnight
    if start_hour > end_hour:
        return current_hour >= start_hour or current_hour <= end_hour
    else:
        return start_hour <= current_hour <= end_hour

# Initialize configuration validation on import
if __name__ == "__main__":
    validation_results = validate_config()
    if validation_results['errors']:
        print("Configuration Errors:")
        for error in validation_results['errors']:
            print(f"  - {error}")
    
    if validation_results['warnings']:
        print("Configuration Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    if not validation_results['errors'] and not validation_results['warnings']:
        print("Configuration validation passed successfully!")
