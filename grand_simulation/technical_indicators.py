import numpy as np
import pandas as pd

class TechnicalIndicators:
    def __init__(self, adx_period=14, ema_fast_window=9, ema_slow_window=21, bb_window=20, bb_std=2, rsi_window=14):
        self.adx_period = adx_period
        self.ema_fast_window = ema_fast_window
        self.ema_slow_window = ema_slow_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window

    def calculate_emas(self, df):
        """ 
        Calculate and return Exponential Moving Average
        """
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast_window, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow_window, adjust=False).mean()
        return df
    
    def calculate_bollinger_bands(self, df):
        """
        Calculate and return Bollinger Bands & related columns
        """
        df['bb_mid'] = df['close'].rolling(self.bb_window).mean()    
        df['bb_std'] = df['close'].rolling(self.bb_window).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - self.bb_std * df['bb_std']
        df['bb_width'] = df['bb_upper'] - df['bb_lower']  # Volatility filter
        return df
    
    def flag_volume_spike(self, df):
        """
        Calculate and return volume spikes & moving average
        USES bb_window from Bollinger Bands!
        """
        df['volume_ma'] = df['volume'].rolling(self.bb_window).mean()    
        df['volume_spike'] = df['volume'] > 1.5 * df['volume_ma']  # customizable factor
        return df
    
    def check_volatility(self, df):
        """
        Check if volatility exceeds moving average
        """
        df['volatility_ok'] = df['bb_width'] > df['bb_width'].rolling(self.bb_window).mean()
        return df
    
    def calculate_adx(self, df):
        """
        Calculate and return Average Directional Index (ADX) & related columns
        """
        # Calculate True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate smoothed averages
        df['tr14'] = df['tr'].rolling(window=self.adx_period).mean()
        df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=self.adx_period).mean() / df['tr14'])
        df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=self.adx_period).mean() / df['tr14'])
        
        # Calculate ADX
        df['dx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()
        
        return df 
    
    def calculate_rsi(self, df):
        """
        Calculate and return the Relative Strength Index (RSI) 
        """
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        df = df.copy()
        df['rsi'] = rsi
        return df
