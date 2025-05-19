import pandas as pd
import numpy as np
from regime_detector import MarketRegimeDetector
from strategy_factory import RegimeStrategyFactory
from technical_indicators import TechnicalIndicators

class RegimeSimulation:
    def __init__(self, name, symbol, timeframe, start_date, end_date, initial_balance=10000, lookback_window=100, lookahead=1, ema_fast_window=9, ema_slow_window=21, bb_window=20, bb_std=2, rsi_window=14,
                 stop_loss_pct=0.02, take_profit_pct=0.1):  
        self.initial_balance = initial_balance
        self.strategy_factory = RegimeStrategyFactory()
        self.technical_indicators = TechnicalIndicators(adx_period=14, ema_fast_window=ema_fast_window, ema_slow_window=ema_slow_window, bb_window=bb_window, bb_std=bb_std, rsi_window=rsi_window)
        self.regime_detector = MarketRegimeDetector(technical_indicators=self.technical_indicators)
        self.strategy_parms = {}
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback_window
        self.lookahead = lookahead
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.warmup_periods = max(
            self.regime_detector.trend_window,
            self.regime_detector.volatility_window,
            self.technical_indicators.rsi_window,
            self.technical_indicators.ema_slow_window,
            self.technical_indicators.bb_window,
            self.technical_indicators.adx_period
        )

    def run_simulation(self, data):
        if len(data) <= self.warmup_periods:
            raise ValueError(f"Not enough data points. Need at least {self.warmup_periods} periods, got {len(data)}")
        
        df = data.copy()
        
        # Calculate technical indicators
        df = self.technical_indicators.calculate_emas(df)
        df = self.technical_indicators.calculate_adx(df)
        df = self.technical_indicators.calculate_bollinger_bands(df)
        df = self.technical_indicators.flag_volume_spike(df)
        df = self.technical_indicators.check_volatility(df)
        df = self.technical_indicators.calculate_rsi(df)
        
        df['regime'] = self.regime_detector.detect_regime(df)

        # TODO: skip warm-up period, compute actual regimes, store performance metrics

        return df

    def evaluate_regime_accuracy(self, df, future_window=12):
        """
        Adds a column 'regime_correct' indicating whether the assigned market regime
        accurately predicted future price behavior.

        Parameters:
        - df: DataFrame with at least 'close' and 'market_regime' columns.
        - future_window: Number of candles to look ahead.

        Returns:
        - DataFrame with added 'regime_correct' column (True/False/None).
        """
        accuracy = []

        for i in range(len(df)):
            if i + future_window >= len(df):
                accuracy.append(None)
                continue

            current_label = df.iloc[i]['market_regime']
            current_close = df.iloc[i]['close']
            future_prices = df['close'].iloc[i+1:i+1+future_window]
            future_mean = future_prices.mean()
            future_max = future_prices.max()
            future_min = future_prices.min()

            correct = None  # Default

            if current_label == 'Strong Uptrend':
                correct = (
                    future_max > current_close * 1.01 and
                    future_min > current_close * 0.99
                )

            elif current_label == 'Weak Uptrend':
                correct = future_mean > current_close

            elif current_label == 'Strong Downtrend':
                correct = (
                    future_min < current_close * 0.99 and
                    future_max < current_close * 1.01
                )

            elif current_label == 'Weak Downtrend':
                correct = future_mean < current_close

            elif current_label == 'Bullish Breakout':
                correct = (
                    future_max > current_close * 1.015 and
                    future_min > current_close * 0.98
                )

            elif current_label == 'Bearish Breakdown':
                correct = (
                    future_min < current_close * 0.985 and
                    future_max < current_close * 1.02
                )

            elif current_label == 'Accumulation Phase':
                correct = abs(future_mean - current_close) / current_close < 0.01

            elif current_label == 'Distribution Phase':
                correct = abs(future_mean - current_close) / current_close < 0.01

            elif current_label == 'Choppy Sideways Market':
                correct = (
                    (future_max - future_min) / current_close > 0.02 and
                    abs(future_mean - current_close) / current_close < 0.01
                )

            elif current_label == 'Sideways Consolidation':
                correct = (future_max - future_min) / current_close < 0.01

            accuracy.append(correct)

        df['regime_correct'] = accuracy
        return df
