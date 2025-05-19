import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from enum import Enum

from grand_simulation.technical_indicators import TechnicalIndicators

class MarketRegime(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"

    DO_NOTHING = "do_nothing"
    WARM_UP = "warm_up"

    BULL_BREAKOUT = "bull_breakout"
    BEAR_BREAKOUT = "bear_breakout"
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    STRONG_DOWN = "strong_down"
    WEAK_DOWN = "weak_down"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CHOPPY_SIDEWAYS = "choppy_sideways"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"


class MarketRegimeDetector:
    def __init__(self, 
                 trend_window=20,
                 volatility_window=20,
                 adx_threshold=20,
                 range_threshold=0.02,
                 volatility_threshold=0.015,
                 technical_indicators=None):
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.adx_threshold = adx_threshold
        self.range_threshold = range_threshold
        self.volatility_threshold = volatility_threshold
        self.technical_indicators = technical_indicators
        self.warmup_periods = max(
            self.trend_window,
            self.volatility_window,
            self.technical_indicators.rsi_window,
            self.technical_indicators.ema_slow_window,
            self.technical_indicators.bb_window,
            self.technical_indicators.adx_period
        )
    

    def detect_regime(self, data):
        """
        Detect market regime based on price action characteristics
        
        Parameters:
        data (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume', 'adx']
        
        Returns:
        pd.Series: Market regime labels for each timestamp
        """
        df = data.copy()
        
        # Initialize regime column with DO_NOTHING
        df['regime'] = MarketRegime.WARM_UP.value
        
        # Calculate trend indicators with min_periods to handle warmup
        df['sma'] = df['close'].rolling(window=self.trend_window, min_periods=self.trend_window).mean()
        df['price_change'] = df['close'].pct_change(self.trend_window)
        
        # Calculate volatility with min_periods
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_window, min_periods=self.volatility_window).std()
        
        # Calculate range with min_periods
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['range_ma'] = df['daily_range'].rolling(window=self.trend_window, min_periods=self.trend_window).mean()
        
        # Calculate mean reversion indicators with min_periods
        rolling_std = df['close'].rolling(window=self.trend_window, min_periods=self.trend_window).std()
        df['zscore'] = (df['close'] - df['sma']) / rolling_std
        
        # Create masks for each regime, only where we have enough data
        valid_data_mask = (
            df['sma'].notna() & 
            df['volatility'].notna() & 
            df['range_ma'].notna() & 
            df['zscore'].notna() &
            df['adx'].notna()
        )
        
        # Create regime masks only where we have valid data
        old_trend_mask = valid_data_mask & (df['adx'] > self.adx_threshold) & (abs(df['price_change']) > self.range_threshold)
        trend_mask = valid_data_mask & (df['adx'] > self.adx_threshold)
        mean_rev_mask = valid_data_mask & (df['adx'] < self.adx_threshold) & (abs(df['zscore']) > 2)
        breakout_mask = valid_data_mask & (df['volatility'] > self.volatility_threshold) & (abs(df['price_change']) > self.range_threshold * 2)
        scalping_mask = valid_data_mask & (df['volatility'] < self.volatility_threshold * 0.5) & (df['adx'] < self.adx_threshold * 0.5)
        
        # Create a DataFrame to store regime scores
        regime_scores = pd.DataFrame(index=df.index)
        regime_scores['trend'] = trend_mask.astype(int)
        regime_scores['mean_rev'] = mean_rev_mask.astype(int)
        regime_scores['breakout'] = breakout_mask.astype(int)
        regime_scores['scalping'] = scalping_mask.astype(int)
        
        # Define regime priorities (higher number = higher priority)
        regime_priorities = {
            'trend': 4,      # Highest priority
            'breakout': 3,   # Second priority
            'mean_rev': 2,   # Third priority
            'scalping': 1    # Lowest priority
        }
        
        # Apply priorities
        for regime, priority in regime_priorities.items():
            mask = regime_scores[regime] == 1
            if regime == 'trend':
                df.loc[mask, 'regime'] = MarketRegime.TREND_FOLLOWING.value
            elif regime == 'mean_rev':
                df.loc[mask, 'regime'] = MarketRegime.MEAN_REVERSION.value
            elif regime == 'breakout':
                df.loc[mask, 'regime'] = MarketRegime.BREAKOUT.value
            elif regime == 'scalping':
                df.loc[mask, 'regime'] = MarketRegime.SCALPING.value
        
        return df['regime']

    # ADVANCED version of above function
    def classify_market_regime(self, data):
        """
        Classify market regimes using a more detailed approach
        
        Parameters:
        df (pd.DataFrame): OHLCV data with technical indicators
        
        Returns:
        pd.Series: Market regime labels for each timestamp
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Initialize regime column with default value
        df['regime'] = MarketRegime.DO_NOTHING.value
        
        # Calculate additional indicators if not present
        if 'bb_width' not in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # Define regime conditions
        strong_up = (
            (df['ema_fast'] > df['ema_slow']) & 
            (df['adx'] > 25) & 
            (df['close'] > df['ema_fast'])
        )
        
        weak_up = (
            (df['ema_fast'] > df['ema_slow']) & 
            (df['adx'] <= 25)
        )
        
        strong_down = (
            (df['ema_fast'] < df['ema_slow']) & 
            (df['adx'] > 25) & 
            (df['close'] < df['ema_fast'])
        )
        
        weak_down = (
            (df['ema_fast'] < df['ema_slow']) & 
            (df['adx'] <= 25)
        )
        
        bullish_breakout = (
            (df['close'] > df['close'].rolling(20).max()) &
            (df['volume_spike'])
        )
        
        bearish_breakdown = (
            (df['close'] < df['close'].rolling(20).min()) &
            (df['volume_spike'])
        )
        
        bb_narrow = df['bb_width'] < df['bb_width'].rolling(20).mean()
        choppy_sideways = ~bb_narrow & (df['adx'] < 20)
        sideways_consolidation = bb_narrow & (df['adx'] < 20)
        
        accumulation = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['close'] < df['bb_mid']) &
            (df['adx'] < 20)
        )
        
        distribution = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['close'] > df['bb_mid']) &
            (df['adx'] < 20)
        )
        
        # Map conditions to regime values
        df.loc[bullish_breakout, 'regime'] = MarketRegime.BULL_BREAKOUT.value
        df.loc[bearish_breakdown, 'regime'] = MarketRegime.BEAR_BREAKOUT.value
        df.loc[strong_up, 'regime'] = MarketRegime.STRONG_UP.value
        df.loc[weak_up, 'regime'] = MarketRegime.WEAK_UP.value
        df.loc[strong_down, 'regime'] = MarketRegime.STRONG_DOWN.value
        df.loc[weak_down, 'regime'] = MarketRegime.WEAK_DOWN.value
        df.loc[accumulation, 'regime'] = MarketRegime.ACCUMULATION.value
        df.loc[distribution, 'regime'] = MarketRegime.DISTRIBUTION.value
        df.loc[choppy_sideways, 'regime'] = MarketRegime.CHOPPY_SIDEWAYS.value
        df.loc[sideways_consolidation, 'regime'] = MarketRegime.SIDEWAYS_CONSOLIDATION.value
        
        return df['regime']
    
    def predict_next_regime(self, df):
        """
        Predict the next market regime using current technical indicators.
        Uses vectorized operations for efficiency.
        Skips the first warmup_periods for accurate indicator calculations.
        
        Parameters:
        df (pd.DataFrame): DataFrame with OHLCV and technical indicators
        
        Returns:
        pd.Series: Predicted regime for each row, with 'warm_up' for initial periods
        """
        # Initialize regime series with 'warm_up' for warmup periods
        regime = pd.Series('warm_up', index=df.index)
        
        # Only process data after warmup period
        valid_data = df.iloc[self.warmup_periods:].copy()
        
        # Create conditions for each regime
        uptrend = (
            (valid_data['ema_fast'] > valid_data['ema_slow']) &  # Fast EMA above slow EMA
            (valid_data['close'] > valid_data['bb_mid']) &       # Price above BB middle
            (valid_data['adx'] > 25) &                           # Strong trend
            (valid_data['rsi'] > 50)                             # Bullish RSI
        )
        
        downtrend = (
            (valid_data['ema_fast'] < valid_data['ema_slow']) &  # Fast EMA below slow EMA
            (valid_data['close'] < valid_data['bb_mid']) &       # Price below BB middle
            (valid_data['adx'] > 25) &                           # Strong trend
            (valid_data['rsi'] < 50)                             # Bearish RSI
        )
        
        high_volatility = (
            (valid_data['bb_width'] > valid_data['bb_width'].rolling(20).mean() * 1.5) |  # BB width significantly above average
            (valid_data['volume_spike'] == True) |                                         # Volume spike
            (valid_data['volatility_ok'] == False)                                        # High volatility flag
        )
        
        low_volatility = (
            (valid_data['bb_width'] < valid_data['bb_width'].rolling(20).mean() * 0.75) &  # BB width significantly below average
            (valid_data['volume_spike'] == False) &                                        # No volume spike
            (valid_data['volatility_ok'] == True) &                                       # Low volatility flag
            (valid_data['adx'] < 20)                                                      # Weak trend
        )
        
        # Assign regimes based on conditions
        valid_mask = pd.Series(False, index=regime.index)
        valid_mask.iloc[self.warmup_periods:] = True
        
        regime.loc[valid_mask & uptrend] = 'uptrend'
        regime.loc[valid_mask & downtrend] = 'downtrend'
        regime.loc[valid_mask & high_volatility] = 'high_volatility'
        regime.loc[valid_mask & low_volatility] = 'low_volatility'
        
        # Shift predictions forward by 1 period to predict next regime
        # Keep 'warm_up' for the last period since we can't predict it
        regime = regime.shift(-1)
        regime.iloc[-1] = 'warm_up'  # Last period can't be predicted
        return regime.fillna('warm_up')