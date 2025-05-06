import numpy as np
import pandas as pd

class TechnicalIndicators:
    def __init__(self, adx_period=14):
        self.adx_period = adx_period
    
    def calculate_adx(self, df):
        """
        Calculate Average Directional Index (ADX)
        
        Parameters:
        df (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close']
        
        Returns:
        pd.DataFrame: DataFrame with ADX-related columns added
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