import pandas as pd
import ta

class MarketDetector:
    def __init__(self, adx_window=14, bb_window=20, bb_std=2, volume_window=20, atr_window=14):
        self.adx_window = adx_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.volume_window = volume_window
        self.atr_window = atr_window

    def detect_market_type_with_confidence(self, df):
        """Detect market type and return confidence levels"""
        # Calculate indicators
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.adx_window)
        df['adx'] = adx_indicator.adx()

        bb = ta.volatility.BollingerBands(close=df['close'], window=self.bb_window, window_dev=self.bb_std)
        df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        df['volume_ma'] = df['volume'].rolling(window=self.volume_window).mean()
        df['volume_spike'] = df['volume'] > 1.5 * df['volume_ma']

        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atr_window)
        df['atr'] = atr_indicator.average_true_range()

        # Latest values
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        confidences = {}

        trend_strength = (latest['adx'] - 20) / 20
        confidences['trend_following'] = max(0.0, min(trend_strength, 1.0))

        mean_reversion_strength = max(0.0, (0.05 - latest['bbw']) / 0.05)
        confidences['mean_reversion'] = max(0.0, min(mean_reversion_strength, 1.0))

        bbw_spike = (latest['bbw'] - previous['bbw'])
        breakout_strength = 0
        if bbw_spike > 0:
            breakout_strength = min(bbw_spike / 0.05, 1.0)
        if latest['volume_spike']:
            breakout_strength += 0.5
        confidences['breakout_trading'] = max(0.0, min(breakout_strength, 1.0))

        volatility_ratio = latest['atr'] / latest['close']
        scalping_strength = min(volatility_ratio / 0.02, 1.0)
        confidences['scalping'] = max(0.0, min(scalping_strength, 1.0))

        confidences['do_nothing'] = 1.0 - max(confidences.values())
        confidences['do_nothing'] = max(0.0, confidences['do_nothing'])

        market_type = max(confidences, key=confidences.get)

        return {
            'market_type': market_type,
            'confidences': confidences
        }

    # default lookahead is 10
    def evaluate_market_detection(self, df, lookahead=10):
        """Evaluates how accurate the market labeling is."""
        if len(df) <= lookahead:
            raise ValueError(f"Data length ({len(df)}) must be greater than lookahead period ({lookahead})")
            
        results = []
        
        # Ensure we have enough data for the first calculation
        min_required_length = max(self.adx_window, self.bb_window, self.volume_window, self.atr_window) + 1
        
        # Validate data length
        if len(df) < min_required_length + lookahead:
            raise ValueError(f"Data length ({len(df)}) must be at least {min_required_length + lookahead} points for proper evaluation")
        
        for i in range(min_required_length, len(df) - lookahead):
            try:
                # Get data up to current point
                sample = df.iloc[:i+1].copy()
                
                # Skip if we don't have enough data for indicators
                if len(sample) < min_required_length:
                    continue
                
                # Calculate indicators
                label_info = self.detect_market_type_with_confidence(sample)
                label = label_info['market_type']
                
                # Get future and current prices
                future_close = df.iloc[i + lookahead]['close']
                current_close = df.iloc[i]['close']
                return_pct = (future_close - current_close) / current_close
                
                correct = False
                
                if label == 'trend_following':
                    # Expect strong move
                    correct = abs(return_pct) > 0.01  # >1% move
                elif label == 'mean_reversion':
                    # Expect mean reversion: price not far from current
                    sma = sample['close'].rolling(window=20).mean().iloc[-1]
                    diff_from_mean = abs(current_close - sma) / sma
                    correct = diff_from_mean < 0.01  # within 1%
                elif label == 'breakout_trading':
                    # Expect large move after consolidation
                    correct = abs(return_pct) > 0.015  # >1.5% move
                elif label == 'scalping':
                    # Expect lots of small moves, not a huge trend
                    correct = abs(return_pct) < 0.005  # <0.5% move
                elif label == 'do_nothing':
                    # Expect nothing interesting
                    correct = abs(return_pct) < 0.003  # <0.3% move

                print(f"{(i / (len(df) - lookahead - min_required_length)) * 100}% Market Label: {label}, Future Return: {return_pct}, Correct: {correct}")
                    
                results.append({
                    'index': df.index[i],
                    'market_label': label,
                    'future_return_pct': return_pct,
                    'correct': correct
                })
            except Exception as e:
                print(f"Warning: Error processing data point at index {i}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid results were generated. Check if the data is sufficient for analysis.")
            
        return pd.DataFrame(results)

