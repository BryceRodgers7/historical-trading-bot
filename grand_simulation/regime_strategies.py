import numpy as np
import pandas as pd
from market_regime_detector import MarketRegime

class BaseStrategy:
    def __init__(self):
        self.position = 0  # 0: no position, 1: long, -1: short

    def generate_signals(self, data):
        """Generate trading signals for the given data"""
        raise NotImplementedError

class TrendFollowingStrategy(BaseStrategy):
    # def __init__(self, ema_short_window, ema_long_window):
    def __init__(self):
        super().__init__()
        # check if ema_short_window and ema_long_window are present in the data??
    
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0

        # Buy signal
        df.loc[
            (df['ema_fast'] > df['ema_slow']) &
            (df['adx'] > 20) &
            (df['volume_spike']) &
            (df['volatility_ok']) &
            (df['close'] > df['bb_mid']),
            'signal'
        ] = 1
        
        # Sell signal
        df.loc[
            (df['ema_fast'] < df['ema_slow']) &
            (df['adx'] > 20) &
            (df['volume_spike']) &
            (df['volatility_ok']) &
            (df['close'] < df['bb_mid']),
            'signal'
        ] = -1
        
        # Additional sell signal when price is below fast EMA
        df.loc[df['close'] < df['ema_fast'], 'signal'] = -1

        return df['signal']

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

class ScalpingStrategy(BaseStrategy):
    def __init__(self, short_window=5, long_window=10, profit_target=0.001, stop_loss=0.0005):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def generate_signals(self, data):
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

class RegimeStrategyFactory:
    @staticmethod
    # def get_strategy(regime, ema_short_window=9, ema_long_window=21):
    def get_strategy(regime):
        if regime != MarketRegime.DO_NOTHING.value:
            return TrendFollowingStrategy()
            # return TrendFollowingStrategy(ema_short_window, ema_long_window)
        
        else:
            return None
        
        # if regime == MarketRegime.TREND_FOLLOWING.value:
        #     return TrendFollowingStrategy()
        # elif regime == MarketRegime.MEAN_REVERSION.value:
        #     return MeanReversionStrategy()
        # elif regime == MarketRegime.BREAKOUT.value:
        #     return BreakoutStrategy()
        # elif regime == MarketRegime.SCALPING.value:
        #     return ScalpingStrategy()
        # elif regime == MarketRegime.DO_NOTHING.value:
        #     return None
        # else:
        #     raise ValueError(f"Unknown market regime: {regime}") 