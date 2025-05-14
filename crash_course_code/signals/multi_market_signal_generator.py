import pandas as pd
import numpy as np

class MultiMarketSignalGenerator:

    def __init__(self, df):
        self.df = df.copy()

    def calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(self, window=14):
        df = self.df.copy()
        df['TR'] = np.maximum(df['high'] - df['low'],
                              np.maximum(abs(df['high'] - df['close'].shift()),
                                         abs(df['low'] - df['close'].shift())))
        df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                             np.maximum(df['high'] - df['high'].shift(), 0), 0)
        df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                             np.maximum(df['low'].shift() - df['low'], 0), 0)
        tr_smooth = df['TR'].rolling(window=window).mean()
        plus_di = 100 * (df['+DM'].rolling(window=window).mean() / (tr_smooth + 1e-10))
        minus_di = 100 * (df['-DM'].rolling(window=window).mean() / (tr_smooth + 1e-10))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        adx = dx.rolling(window=window).mean()
        return adx

    def trend_following_signals(self):
        df = self.df.copy()
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        df['adx'] = self.calculate_adx()
        df['signal'] = np.where((df['ema_short'] > df['ema_long']) & (df['adx'] > 20), 'buy',
                                np.where((df['ema_short'] < df['ema_long']) & (df['adx'] > 20), 'sell', 'hold'))
        return df[['close', 'signal']]

    def mean_reversion_signals(self):
        df = self.df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['signal'] = np.where((df['close'] < df['bb_lower']) & (df['rsi'] < 30), 'buy',
                                np.where((df['close'] > df['bb_upper']) & (df['rsi'] > 70), 'sell', 'hold'))
        return df[['close', 'signal']]

    def breakout_trading_signals(self):
        df = self.df.copy()
        df['bb_width'] = (df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()) - \
                         (df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std())
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['signal'] = np.where((df['close'] > df['close'].rolling(20).max()) & (df['volume'] > df['volume_ma']), 'buy',
                                np.where((df['close'] < df['close'].rolling(20).min()) & (df['volume'] > df['volume_ma']), 'sell', 'hold'))
        return df[['close', 'signal']]

    def scalping_signals(self):
        df = self.df.copy()
        df['ema_fast'] = df['close'].ewm(span=3, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=8, adjust=False).mean()
        df['rsi'] = self.calculate_rsi(df['close'], window=7)
        df['signal'] = np.where((df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 50), 'buy',
                                np.where((df['ema_fast'] < df['ema_slow']) & (df['rsi'] < 50), 'sell', 'hold'))
        return df[['close', 'signal']]

    def no_signal(self):
        df = self.df.copy()
        df['signal'] = 'hold'
        return df[['close', 'signal']]

    def generate_signals_by_regime(self, regime):
        if regime == 'trend_following':
            return self.trend_following_signals()
        elif regime == 'mean_reversion':
            return self.mean_reversion_signals()
        elif regime == 'breakout_trading':
            return self.breakout_trading_signals()
        elif regime == 'scalping':
            return self.scalping_signals()
        else:
            return self.no_signal()
