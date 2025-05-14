import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class BullRunStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.trailing_stop_pct = 0.05  # 5% trailing stop
        self.last_high = None
        self.in_position = False
    
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        
        # Calculate multiple EMAs for trend confirmation
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
        
        # Calculate volume-weighted momentum
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_change'] = df['close'].pct_change(5)  # 5-period change
        df['momentum'] = df['price_change'] * df['volume_ratio']
        
        # Calculate trend strength
        df['trend_strength'] = (
            (df['ema_8'] > df['ema_21']) & 
            (df['ema_21'] > df['ema_55'])
        ).astype(int)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize trailing stop tracking
        if self.last_high is None:
            self.last_high = df['high'].iloc[0]
        
        # Generate signals
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            
            # Update trailing stop
            if current_high > self.last_high:
                self.last_high = current_high
            
            # Buy conditions
            buy_conditions = (
                df['trend_strength'].iloc[i] == 1 and  # All EMAs aligned
                df['momentum'].iloc[i] > 0.02 and      # Strong positive momentum
                df['volume_ratio'].iloc[i] > 1.2 and   # Above average volume
                df['rsi'].iloc[i] > 50 and             # RSI above 50
                df['rsi'].iloc[i] < 70 and             # Not overbought
                not self.in_position                   # Not already in position
            )
            
            # Sell conditions
            sell_conditions = (
                (current_price < self.last_high * (1 - self.trailing_stop_pct)) or  # Trailing stop hit
                (df['trend_strength'].iloc[i] == 0 and df['momentum'].iloc[i] < -0.02) or  # Trend reversal
                (df['rsi'].iloc[i] > 80)  # Extremely overbought
            )
            
            # Apply signals
            if buy_conditions:
                df.loc[df.index[i], 'signal'] = 1
                self.in_position = True
            elif sell_conditions and self.in_position:
                df.loc[df.index[i], 'signal'] = -1
                self.in_position = False
                self.last_high = current_high  # Reset trailing stop
        
        return df['signal'] 