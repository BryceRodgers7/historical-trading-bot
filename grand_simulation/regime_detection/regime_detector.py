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
    def classify_market_regime(self, df):
        """
        Classify market regimes using a more detailed approach
        
        Parameters:
        df (pd.DataFrame): OHLCV data with technical indicators
        
        Returns:
        pd.Series: Market regime labels for each timestamp
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
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
    
