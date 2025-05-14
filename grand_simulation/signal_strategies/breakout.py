import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self, window=20, multiplier=2):
        super().__init__()
        self.window = window
        self.multiplier = multiplier

    def generate_signals(self, data):
        df = data.copy()
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.window).mean()
        
        # Calculate breakout levels
        df['upper_breakout'] = df['close'].shift(1) + (df['atr'] * self.multiplier)
        df['lower_breakout'] = df['close'].shift(1) - (df['atr'] * self.multiplier)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] > df['upper_breakout'], 'signal'] = 1  # Buy on upper breakout
        df.loc[df['close'] < df['lower_breakout'], 'signal'] = -1  # Sell on lower breakout
        
        return df['signal'] 