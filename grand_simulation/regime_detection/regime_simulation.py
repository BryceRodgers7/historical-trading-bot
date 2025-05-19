import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from regime_detector import MarketRegimeDetector
from grand_simulation.strategy_factory import RegimeStrategyFactory
from grand_simulation.technical_indicators import TechnicalIndicators

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

    def run_regime_simulation(self, data):
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
        
        # Predict and validate regimes
        df['predicted_regime'] = self.regime_detector.predict_next_regime(df)
        validation_results = self.validate_regime_prediction(df)
        
        # Print detailed metrics
        print(f"\n{'='*80}")
        print(f"Regime Detection Results for {self.name}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Lookahead: {self.lookahead} periods")
        print(f"{'='*80}")
        
        # Print overall statistics
        total_periods = len(df)
        valid_periods = total_periods - self.warmup_periods - self.lookahead
        print(f"\nOverall Statistics:")
        print(f"Total periods: {total_periods}")
        print(f"Warmup periods: {self.warmup_periods}")
        print(f"Valid periods for prediction: {valid_periods}")
        
        # Print regime distribution
        print(f"\nRegime Distribution:")
        regime_counts = df['predicted_regime'].value_counts()
        for regime, count in regime_counts.items():
            percentage = round(count / total_periods * 100, 2)
            print(f"{regime}: {count} periods ({percentage}%)")
        
        # Print validation metrics
        print(f"\nValidation Metrics:")
        print(validation_results.to_string(index=False))
        
        # Calculate and print average metrics
        avg_accuracy = validation_results['accuracy'].mean()
        avg_precision = validation_results['precision'].mean()
        avg_recall = validation_results['recall'].mean()
        print(f"\nAverage Metrics:")
        print(f"Average Accuracy: {avg_accuracy:.2%}")
        print(f"Average Precision: {avg_precision:.2%}")
        print(f"Average Recall: {avg_recall:.2%}")
        
        # Print most accurate regime
        best_regime = validation_results.loc[validation_results['accuracy'].idxmax()]
        print(f"\nBest Performing Regime:")
        print(f"Regime: {best_regime['regime']}")
        print(f"Accuracy: {best_regime['accuracy']:.2%}")
        print(f"Total Predictions: {best_regime['total_predictions']}")
        print(f"Correct Predictions: {best_regime['correct_predictions']}")
        
        print(f"\n{'-'*80}\n")
        
        return df, validation_results
    
    # UNUSED:this needs to be examined more closely
    def evaluate_regime_advanced(self, df, future_window=12):
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
            # can't look beyond the last row
            if i + future_window >= len(df):
                accuracy.append(None)
                continue

            current_label = df.iloc[i]['market_regime']
            current_close = df.iloc[i]['close']

            # for now just use the next price
            future_prices = df['close'].iloc[i+1:i+1+future_window]
            future_mean = future_prices.mean()
            future_max = future_prices.max()
            future_min = future_prices.min()

            # future_prices = df.iloc[i+1]['close']
            # future_mean = future_prices
            # future_max = future_prices
            # future_min = future_prices

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

    def validate_regime_prediction(self, df):
        """
        Validate regime predictions by comparing predicted regime with actual future price action.
        Uses vectorized operations for efficiency.
        Skips the first warmup_periods and last prediction_window periods for accurate validation.
        
        Parameters:
        df (pd.DataFrame): DataFrame with OHLCV, technical indicators, and predicted regime
        prediction_window (int): Number of periods to look ahead for validation
        
        Returns:
        pd.DataFrame: Validation results with accuracy metrics for each regime
        """
        prediction_window = self.lookahead
        # Calculate future returns for validation
        future_returns = df['close'].pct_change(prediction_window).shift(-prediction_window)
        future_volatility = df['close'].pct_change().rolling(prediction_window).std().shift(-prediction_window)
        
        # Define actual regime conditions based on future price action
        actual_uptrend = (
            (future_returns > 0.02) &  # 2% or more price increase
            (future_volatility < 0.03)  # Relatively stable
        )
        
        actual_downtrend = (
            (future_returns < -0.02) &  # 2% or more price decrease
            (future_volatility < 0.03)  # Relatively stable
        )
        
        actual_high_vol = (
            (future_volatility > 0.03) |  # High volatility
            (abs(future_returns) > 0.05)   # Large price moves
        )
        
        actual_low_vol = (
            (abs(future_returns) < 0.01) &  # Small price changes
            (future_volatility < 0.015)     # Low volatility
        )
        
        # Create actual regime series
        actual_regime = pd.Series('warm_up', index=df.index)
        
        # Only validate data between warmup period and prediction window
        valid_mask = pd.Series(False, index=df.index)
        valid_mask.iloc[self.warmup_periods:-prediction_window] = True
        
        # Assign actual regimes for valid range
        actual_regime.loc[valid_mask & actual_uptrend] = 'uptrend'
        actual_regime.loc[valid_mask & actual_downtrend] = 'downtrend'
        actual_regime.loc[valid_mask & actual_high_vol] = 'high_volatility'
        actual_regime.loc[valid_mask & actual_low_vol] = 'low_volatility'
        
        # Calculate accuracy metrics for each regime
        regimes = ['uptrend', 'downtrend', 'high_volatility', 'low_volatility']
        results = []
        
        # Only use data in valid range for validation
        valid_data = df.loc[valid_mask]
        valid_actual = actual_regime.loc[valid_mask]
        
        for regime in regimes:
            # Get predictions and actual values for this regime
            pred_mask = valid_data['predicted_regime'] == regime
            actual_mask = valid_actual == regime
            
            # Calculate metrics
            total_predictions = pred_mask.sum()
            correct_predictions = (pred_mask & actual_mask).sum()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate precision and recall
            precision = correct_predictions / total_predictions if total_predictions > 0 else 0
            recall = correct_predictions / actual_mask.sum() if actual_mask.sum() > 0 else 0
            
            results.append({
                'regime': regime,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'warmup_periods': self.warmup_periods,
                'prediction_window': prediction_window
            })
        
        return pd.DataFrame(results)
