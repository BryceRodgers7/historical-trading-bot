import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class FibonacciRetracementStrategy(BaseStrategy):
    def __init__(self, swing_window=15, min_swing_pct=0.015):  # Updated default values
        super().__init__()
        self.swing_window = swing_window  # Window to identify swing points
        self.min_swing_pct = min_swing_pct  # Minimum swing size to consider
        self.fib_levels = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.618, 2.618]
        }
        self.last_swing_high = None
        self.last_swing_low = None
        self.profit_target = None
        self.in_position = False
    
    def find_swing_points(self, df):
        """Identify swing highs and lows in the price data"""
        df['swing_high'] = False
        df['swing_low'] = False
        
        for i in range(self.swing_window, len(df) - self.swing_window):
            # Check for swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-self.swing_window:i]) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+1:i+self.swing_window+1]):
                df.loc[df.index[i], 'swing_high'] = True
            
            # Check for swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-self.swing_window:i]) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+1:i+self.swing_window+1]):
                df.loc[df.index[i], 'swing_low'] = True
        
        return df
    
    def calculate_fib_levels(self, high, low):
        """Calculate Fibonacci retracement and extension levels"""
        diff = high - low
        levels = {
            'retracement': {
                level: high - (diff * level) 
                for level in self.fib_levels['retracement']
            },
            'extension': {
                level: high + (diff * (level - 1))
                for level in self.fib_levels['extension']
            }
        }
        return levels
    
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['profit_target'] = None
        
        # Find swing points
        df = self.find_swing_points(df)
        
        # Calculate EMAs for trend confirmation
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Process each bar
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            
            # Update swing points
            if df['swing_high'].iloc[i]:
                self.last_swing_high = df['high'].iloc[i]
            if df['swing_low'].iloc[i]:
                self.last_swing_low = df['low'].iloc[i]
            
            # If we have both swing points and they're significant
            if self.last_swing_high is not None and self.last_swing_low is not None:
                swing_size = (self.last_swing_high - self.last_swing_low) / self.last_swing_low
                
                if swing_size >= self.min_swing_pct:
                    # Calculate Fibonacci levels
                    fib_levels = self.calculate_fib_levels(self.last_swing_high, self.last_swing_low)
                    
                    # Check for entry conditions
                    if not self.in_position:
                        # Entry conditions:
                        # 1. Price is near a key Fibonacci retracement level (0.618 or 0.786)
                        # 2. Uptrend confirmed by EMAs
                        # 3. Volume confirmation
                        # 4. RSI not oversold
                        near_fib_618 = abs(current_price - fib_levels['retracement'][0.618]) / current_price < 0.005
                        near_fib_786 = abs(current_price - fib_levels['retracement'][0.786]) / current_price < 0.005
                        
                        if (near_fib_618 or near_fib_786) and \
                           df['ema_20'].iloc[i] > df['ema_50'].iloc[i] and \
                           df['volume_ratio'].iloc[i] > 1.2 and \
                           df['rsi'].iloc[i] > 40:
                            
                            df.loc[df.index[i], 'signal'] = 1
                            self.in_position = True
                            # Set profit target at 1.618 extension
                            self.profit_target = fib_levels['extension'][1.618]
                            df.loc[df.index[i], 'profit_target'] = self.profit_target
                    
                    # Check for profit target hit
                    elif self.in_position and current_price >= self.profit_target:
                        df.loc[df.index[i], 'signal'] = -1
                        self.in_position = False
                        self.profit_target = None
                        df.loc[df.index[i], 'profit_target'] = None
        
        return df['signal'] 