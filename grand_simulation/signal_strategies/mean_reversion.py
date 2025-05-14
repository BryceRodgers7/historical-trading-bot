import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, window=20, std_dev=2):
        super().__init__()
        self.window = window
        self.std_dev = std_dev

    def generate_signals(self, data):
        df = data.copy()
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.window).mean()
        df['std'] = df['close'].rolling(window=self.window).std()
        df['upper_band'] = df['sma'] + (df['std'] * self.std_dev)
        df['lower_band'] = df['sma'] - (df['std'] * self.std_dev)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy at lower band
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell at upper band
        
        return df['signal'] 