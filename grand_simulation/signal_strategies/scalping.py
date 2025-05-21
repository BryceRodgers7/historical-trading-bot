import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class ScalpingStrategy(BaseStrategy):
    def __init__(self, short_window=5, long_window=10, profit_target=0.001, stop_loss=0.0005):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.profit_target = profit_target
        self.stop_loss = stop_loss

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
            (df['adx'] > 25) &                    # Stronger trend requirement
            (df['volume_spike']) &                # Volume confirmation
            (df['close'] < df['bb_mid']) &        # Price below BB middle
            (df['rsi'] < 40) &                    # More oversold RSI
            (df['trend_confirmed'] < 0) &         # Confirmed downtrend
            (df['close'] < df['ema_fast'] * 0.90) # Price significantly below fast EMA
        )
        
        # Only sell if we have a confirmed trend change
        df.loc[sell_conditions, 'signal'] = -1
        
        return df['signal'] 
    
    def OLD_generate_signals(self, data):
        df = data.copy()
        # Calculate fast and slow EMAs
        df['EMA_fast'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(self.short_window)
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when fast EMA crosses above slow EMA and momentum is positive
        df.loc[(df['EMA_fast'] > df['EMA_slow']) & (df['momentum'] > 0), 'signal'] = 1
        
        # Sell when fast EMA crosses below slow EMA and momentum is negative
        df.loc[(df['EMA_fast'] < df['EMA_slow']) & (df['momentum'] < 0), 'signal'] = -1
        
        return df['signal']