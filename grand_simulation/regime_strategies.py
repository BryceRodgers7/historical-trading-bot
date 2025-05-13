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
        self.trend_confirmation_periods = 3  # Number of periods to confirm trend change
        # check if ema_short_window and ema_long_window are present in the data??
    
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

class BullRunStrategy(BaseStrategy):
    """
    Strategy specifically designed for strong bull markets like Q1 2023.
    Key characteristics:
    - Uses multiple EMAs to confirm trend strength
    - Employs volume-weighted momentum
    - Implements a trailing stop loss
    - Uses RSI for overbought/oversold conditions
    - Requires strong trend confirmation
    """
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

class RegimeStrategyFactory:
    @staticmethod
    # def get_strategy(regime, ema_short_window=9, ema_long_window=21):
    def get_strategy(regime):
        if regime != MarketRegime.DO_NOTHING.value:
            return TrendFollowingStrategy()
            # return BullRunStrategy()
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