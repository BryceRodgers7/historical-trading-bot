import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.trend_confirmation_periods = 3  # Number of periods to confirm trend change
    
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        
        # Calculate trend strength
        df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
        
        # Calculate trend confirmation (using rolling window)
        df['trend_confirmed'] = df['trend_strength'].rolling(
            window=self.trend_confirmation_periods, 
            min_periods=1
        ).mean()
        
        # Buy signal - more aggressive in bull markets
        df.loc[
            (df['ema_fast'] > df['ema_slow']) &  # EMA crossover
            (df['adx'] > 20) &                    # Strong trend
            (df['volume_spike']) &                # Volume confirmation
            (df['volatility_ok']) &               # Volatility check
            (df['close'] > df['bb_mid']) &        # Price above BB middle MAYBE REMOVE THIS
            (df['rsi'] > 50) &                    # RSI above 50
            (df['trend_confirmed'] > 0),          # Confirmed uptrend
            'signal'
        ] = 1
        
        # Sell signal - more conservative in bull markets
        sell_conditions = (
            (df['ema_fast'] < df['ema_slow']) &   # EMA crossover
            (df['adx'] > 30) &                    # Stronger trend requirement
            (df['volume_spike']) &                # Volume confirmation
            (df['close'] < df['bb_mid']) &        # Price below BB middle
            (df['rsi'] < 30) &                    # More oversold RSI
            (df['trend_confirmed'] < 0) &         # Confirmed downtrend
            (df['close'] < df['ema_fast'] * 0.50) # Price significantly below fast EMA
        )
        
        # Only sell if we have a confirmed trend change
        df.loc[sell_conditions, 'signal'] = -1
        
        return df['signal'] 
    

    def ADV_generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        
        # Calculate trend strength and direction
        df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
        df['trend_direction'] = np.sign(df['trend_strength'])
        
        # Calculate trend confirmation using multiple timeframes
        df['trend_confirmed'] = df['trend_strength'].rolling(
            window=self.trend_confirmation_periods, 
            min_periods=1
        ).mean()
        
        # Calculate higher timeframe trend (e.g., 4x current timeframe)
        df['higher_tf_trend'] = df['trend_strength'].rolling(
            window=self.trend_confirmation_periods * 4,
            min_periods=1
        ).mean()
        
        # Calculate trend persistence
        df['trend_persistence'] = df['trend_direction'].rolling(
            window=self.min_trend_duration,
            min_periods=1
        ).sum()
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(5)  # 5-period momentum
        df['momentum_ma'] = df['momentum'].rolling(window=5).mean()
        
        # Calculate volatility-adjusted trend strength
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        df['trend_strength_vol_adj'] = df['trend_strength'] / df['volatility']
        
        # Buy signal conditions
        buy_conditions = (
            (df['ema_fast'] > df['ema_slow']) &           # EMA crossover
            (df['adx'] > 25) &                            # Strong trend
            (df['volume_spike']) &                        # Volume confirmation
            (df['volatility_ok']) &                       # Volatility check
            (df['close'] > df['bb_mid']) &                # Price above BB middle
            (df['rsi'] > 50) &                            # RSI above 50
            (df['trend_confirmed'] > 0) &                 # Confirmed uptrend
            (df['higher_tf_trend'] > 0) &                 # Higher timeframe trend is up
            (df['momentum'] > 0) &                        # Positive momentum
            (df['momentum_ma'] > 0) &                     # Positive momentum MA
            (df['trend_strength_vol_adj'] > 0.001)        # Strong trend relative to volatility
        )
        
        # Sell signal conditions - much more conservative
        sell_conditions = (
            (df['ema_fast'] < df['ema_slow']) &           # EMA crossover
            (df['adx'] > 30) &                            # Very strong trend
            (df['volume_spike']) &                        # Volume confirmation
            (df['close'] < df['bb_mid']) &                # Price below BB middle
            (df['rsi'] < 30) &                            # Very oversold RSI
            (df['trend_confirmed'] < 0) &                 # Confirmed downtrend
            (df['higher_tf_trend'] < 0) &                 # Higher timeframe trend is down
            (df['momentum'] < 0) &                        # Negative momentum
            (df['momentum_ma'] < 0) &                     # Negative momentum MA
            (df['trend_strength_vol_adj'] < -0.001) &     # Strong downtrend relative to volatility
            (df['trend_persistence'] < -self.min_trend_duration/2) &  # Trend has persisted
            (df['close'] < df['ema_fast'] * 0.95)         # Significant price drop
        )
        
        # Apply signals with trend persistence check
        for i in range(len(df)):
            if i < self.min_trend_duration:
                continue
                
            # Check if we're in a valid trend period
            if df['trend_persistence'].iloc[i] >= self.min_trend_duration/2:
                # Only allow buy signals in strong uptrends
                if buy_conditions.iloc[i]:
                    df.loc[df.index[i], 'signal'] = 1
            elif df['trend_persistence'].iloc[i] <= -self.min_trend_duration/2:
                # Only allow sell signals in strong downtrends
                if sell_conditions.iloc[i]:
                    df.loc[df.index[i], 'signal'] = -1
        
        return df['signal']