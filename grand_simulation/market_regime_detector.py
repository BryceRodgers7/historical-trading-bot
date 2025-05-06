import numpy as np
import pandas as pd
from enum import Enum

class MarketRegime(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    DO_NOTHING = "do_nothing"

class MarketRegimeDetector:
    def __init__(self, 
                 trend_window=20,
                 volatility_window=20,
                 adx_threshold=20,
                 range_threshold=0.02,
                 volatility_threshold=0.015):
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.adx_threshold = adx_threshold
        self.range_threshold = range_threshold
        self.volatility_threshold = volatility_threshold

    def detect_regime(self, data):
        """
        Detect market regime based on price action characteristics
        
        Parameters:
        data (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume', 'adx']
        
        Returns:
        pd.Series: Market regime labels for each timestamp
        """
        df = data.copy()
        
        # Calculate trend indicators
        df['sma'] = df['close'].rolling(window=self.trend_window).mean()
        df['price_change'] = df['close'].pct_change(self.trend_window)
        
        # Calculate volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_window).std()
        
        # Calculate range
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['range_ma'] = df['daily_range'].rolling(window=self.trend_window).mean()
        
        # Calculate mean reversion indicators
        df['zscore'] = (df['close'] - df['sma']) / df['close'].rolling(window=self.trend_window).std()
        
        # Initialize regime column
        df['regime'] = MarketRegime.DO_NOTHING.value
        
        # Create masks for each regime
        old_trend_mask = (df['adx'] > self.adx_threshold) & (abs(df['price_change']) > self.range_threshold)
        trend_mask = (df['adx'] > self.adx_threshold)
        mean_rev_mask = (df['adx'] < self.adx_threshold) & (abs(df['zscore']) > 2)
        breakout_mask = (df['volatility'] > self.volatility_threshold) & (abs(df['price_change']) > self.range_threshold * 2)
        scalping_mask = (df['volatility'] < self.volatility_threshold * 0.5) & (df['adx'] < self.adx_threshold * 0.5)
        
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