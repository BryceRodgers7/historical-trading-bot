import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import ccxt


def fetch_historical_data(exchange, symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    """Fetch historical data from the exchange"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.now()
    
    # Convert dates to timestamps
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    # Fetch data in chunks
    all_ohlcv = []
    current_timestamp = start_timestamp
    
    while current_timestamp < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_timestamp,
                limit=1000
            )
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current_timestamp = ohlcv[-1][0] + 1
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_ohlcv:
        raise Exception("No data fetched for the specified time period")
    
    # Convert to DataFrame
    data = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    
    # Filter data to ensure we only have data within our specified window
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    return data

# Detect candlestick patterns
def detect_candlestick_patterns(df):
    df['bullish_engulfing'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1))
    df['bearish_engulfing'] = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'].shift(1))
    return df

# Adjust confidence based on volume
def adjust_confidence_based_on_volume(df, threshold=1.5):
    volume_ma = df['volume'].rolling(window=14).mean().iloc[-1]
    if df['volume'].iloc[-1] > threshold * volume_ma:  # High volume
        return 0.1
    elif df['volume'].iloc[-1] < volume_ma:  # Low volume
        return -0.1
    return 0

# Adjust confidence based on ADX (trend strength)
def adjust_based_on_adx(df, threshold=25):   
    try:
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        if pd.isna(adx.iloc[-1]):  # Check if the last value is NaN
            return 0
        if adx.iloc[-1] > threshold:
            return 0.15  # Strong trend
        else:
            return -0.15  # Weak trend
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        return 0  # Return neutral adjustment on error

# Adjust confidence based on volatility (ATR)
def adjust_based_on_volatility(df, threshold=0.02):
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
    if atr > threshold:  # High volatility
        return -0.1  # Reduce confidence during high volatility
    else:  # Low volatility
        return 0.1  # Increase confidence during low volatility
    return 0

# Multi-timeframe analysis
def multi_timeframe_adjustment(df, high_tf_df):
    if df['signal'].iloc[-1] == high_tf_df['signal'].iloc[-1]:
        return 0.2  # Confidence boost if signals align
    return 0  # No change if signals do not align

# Adjust based on risk-to-reward ratio
def adjust_based_on_risk(df):
    if 'tp_price' in df.columns and 'sl_price' in df.columns:
        risk_to_reward = (df['tp_price'].iloc[-1] - df['close'].iloc[-1]) / (df['close'].iloc[-1] - df['sl_price'].iloc[-1])
        if risk_to_reward > 2:
            return 0.1  # Good risk-to-reward ratio
        return -0.1  # Bad risk-to-reward ratio
    return 0

# Calculate overall confidence
def dynamic_confidence_adjustment(df, trend_signal, high_tf_df, volatility_threshold=0.02, adx_threshold=25):
    confidence = 0.5  # Start with neutral confidence
    
    # Adjust based on volume
    volume_adjustment = adjust_confidence_based_on_volume(df)
    confidence += volume_adjustment
    
    # Adjust based on ADX (trend strength)
    adx_adjustment = adjust_based_on_adx(df, adx_threshold)
    confidence += adx_adjustment
    
    # Adjust based on volatility (ATR)
    volatility_adjustment = adjust_based_on_volatility(df, volatility_threshold)
    confidence += volatility_adjustment
    
    # Multi-timeframe alignment
    # mtf_adjustment = multi_timeframe_adjustment(df, high_tf_df)
    # confidence += mtf_adjustment
    
    # # Risk-to-reward adjustment (if available)
    # rr_adjustment = adjust_based_on_risk(df)
    # confidence += rr_adjustment
    
    # Candlestick pattern-based adjustment
    # if df['bullish_engulfing'].iloc[-1]:
    #     confidence += 0.1  # Increase confidence for bullish engulfing
    # if df['bearish_engulfing'].iloc[-1]:
    #     confidence -= 0.1  # Decrease confidence for bearish engulfing
    
    # Cap confidence between 0 and 1
    confidence = max(min(confidence, 1.0), 0.0)
    
    return confidence

# Trend-following strategy (e.g., 1h candles)
def trend_following_signals(df, short_window=20, long_window=50):
    # Create a copy of the DataFrame to avoid the warning
    df = df.copy()
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[df.index[long_window:], 'signal'] = np.where(
        df['sma_short'][long_window:] > df['sma_long'][long_window:], 1, 0
    )
    return df

# Scalping strategy (e.g., 5m candles)
def scalping_signals(df, ema_window=5, rsi_window=7):
    # Create a copy of the DataFrame to avoid the warning
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_window).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_window).rsi()
    df['signal'] = 0
    
    # Buy signal when:
    # 1. RSI is oversold (< 30) OR
    # 2. Price crosses above EMA and RSI is not overbought (> 70)
    df.loc[
        ((df['rsi'] < 30) | 
         ((df['close'] > df['ema']) & (df['rsi'] < 70))),
        'signal'
    ] = 1
    
    return df

# Simulated position tracker
class Position:
    def __init__(self, strategy_name, entry_price, tp_pct, sl_pct):
        self.strategy_name = strategy_name
        self.entry_price = entry_price
        self.tp_price = entry_price * (1 + tp_pct)
        self.sl_price = entry_price * (1 - sl_pct)
        self.active = True
        self.exit_price = None
        self.pnl = 0

    def check_exit(self, current_price):
        if not self.active:
            return None
        if current_price >= self.tp_price or current_price <= self.sl_price:
            self.active = False
            self.exit_price = current_price
            self.pnl = ((current_price - self.entry_price) / self.entry_price) * 100
            return 'closed with profit' if current_price >= self.tp_price else 'stopped out'
        return None

# Simulate trading using historical data
def simulate_trading():
    # Define multiple time periods to test
    periods = {
        'Recent Bull Market': (
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 2, 1, 23, 59, 59)
        )
    }
    # ,
    #     'Previous Bear Market': (
    #         datetime(2023, 7, 1, 0, 0, 0),
    #         datetime(2023, 9, 30, 23, 59, 59)
    #     ),
    #     'Sideways Period': (
    #         datetime(2023, 4, 1, 0, 0, 0),
    #         datetime(2023, 6, 30, 23, 59, 59)
    #     ),
    #     'High Volatility': (
    #         datetime(2023, 1, 1, 0, 0, 0),
    #         datetime(2023, 3, 31, 23, 59, 59)
    #     ),
    #     'Low Volatility': (
    #         datetime(2023, 10, 1, 0, 0, 0),
    #         datetime(2023, 12, 31, 23, 59, 59)
    #     )
    # }

    exchange = ccxt.binanceus()

    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{'='*50}")
        print(f"Testing period: {period_name}")
        print(f"From: {start_date} to {end_date}")
        print(f"{'='*50}")

        # Initialize profit tracking for this period
        period_stats = {
            'trend_trades': [],
            'scalp_trades': [],
            'aligned_trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0
        }

        try:
            # Load historical data for both timeframes
            trend_df = fetch_historical_data(exchange, 'BTC/USDT', '1h', start_date, end_date)
            scalp_df = fetch_historical_data(exchange, 'BTC/USDT', '5m', start_date, end_date)

            # Detect candlestick patterns for both timeframes
            trend_df = detect_candlestick_patterns(trend_df)
            scalp_df = detect_candlestick_patterns(scalp_df)

            # Calculate the minimum required data points
            min_data_points = max(50, 14)  # 50 for long_window in trend_following_signals, 14 for ADX

            trend_position = None
            scalp_position = None
            aligned_position = None
            current_trend_signal = 0
            last_hour_processed = None

            # Start simulation after we have enough data points
            for i in range(min_data_points, len(scalp_df)):
                current_timestamp = scalp_df.index[i]
                current_hour = current_timestamp.floor('h')
                
                # Update trend signal only when we enter a new hour
                if last_hour_processed is None or current_hour > last_hour_processed:
                    # Get all hourly data up to the current hour
                    hour_data = trend_df[trend_df.index <= current_hour]
                    if len(hour_data) >= min_data_points:
                        hour_data = trend_following_signals(hour_data)
                        current_trend_signal = hour_data['signal'].iloc[-1]
                    last_hour_processed = current_hour

                # Get scalp signal using the 5m data up to current point
                scalp_data = scalping_signals(scalp_df.iloc[:i+1])
                scalp_signal = scalp_data['signal'].iloc[-1]

                # Adjust confidence based on all factors
                confidence = dynamic_confidence_adjustment(scalp_data, current_trend_signal, hour_data)

                # Print signals and confidence for debugging
                progress = (i / len(scalp_df)) * 100
                print(f"[{progress:.1f}%] [Trend] Signal: {current_trend_signal}, [Scalp] Signal: {scalp_signal}, [Confidence] {confidence:.2f}")

                # Simulate position opening based on confidence
                if confidence > 0.7 and current_trend_signal == 1:  # Buy signal for trend-following
                    trend_position = Position('Trend', trend_df.loc[current_hour, 'close'], tp_pct=0.05, sl_pct=0.01)
                    print(f"Opened trend position at {trend_df.loc[current_hour, 'close']:.2f}")

                # Open scalp positions based on scalp signals
                if scalp_signal == 1 and not scalp_position:  # Buy scalp signal
                    scalp_position = Position('Scalp', scalp_df['close'].iloc[i], tp_pct=0.0025, sl_pct=0.003)
                    print(f"Opened scalp position at {scalp_df['close'].iloc[i]:.2f}")

                # Open aligned positions when both signals are 1
                if current_trend_signal == 1 and scalp_signal == 1 and not aligned_position:
                    aligned_position = Position('Aligned', scalp_df['close'].iloc[i], tp_pct=0.05, sl_pct=0.01)
                    print(f"Opened aligned position at {scalp_df['close'].iloc[i]:.2f}")

                # Check if any position has been closed
                if trend_position:
                    result = trend_position.check_exit(trend_df.loc[current_hour, 'close'])
                    if result:
                        print(f"[Trend] Position {result} at {trend_df.loc[current_hour, 'close']:.2f}")
                        period_stats['trend_trades'].append({
                            'entry': trend_position.entry_price,
                            'exit': trend_position.exit_price,
                            'pnl': trend_position.pnl,
                            'result': result
                        })
                        period_stats['total_trades'] += 1
                        period_stats['total_pnl'] += trend_position.pnl
                        if trend_position.pnl > 0:
                            period_stats['winning_trades'] += 1
                        else:
                            period_stats['losing_trades'] += 1
                        trend_position = None

                if scalp_position:
                    result = scalp_position.check_exit(scalp_df['close'].iloc[i])
                    if result:
                        print(f"[Scalp] Position {result} at {scalp_df['close'].iloc[i]:.2f}")
                        period_stats['scalp_trades'].append({
                            'entry': scalp_position.entry_price,
                            'exit': scalp_position.exit_price,
                            'pnl': scalp_position.pnl,
                            'result': result
                        })
                        period_stats['total_trades'] += 1
                        period_stats['total_pnl'] += scalp_position.pnl
                        if scalp_position.pnl > 0:
                            period_stats['winning_trades'] += 1
                        else:
                            period_stats['losing_trades'] += 1
                        scalp_position = None

                if aligned_position:
                    result = aligned_position.check_exit(scalp_df['close'].iloc[i])
                    if result:
                        print(f"[Aligned] Position {result} at {scalp_df['close'].iloc[i]:.2f}")
                        period_stats['aligned_trades'] = period_stats.get('aligned_trades', [])
                        period_stats['aligned_trades'].append({
                            'entry': aligned_position.entry_price,
                            'exit': aligned_position.exit_price,
                            'pnl': aligned_position.pnl,
                            'result': result
                        })
                        period_stats['total_trades'] += 1
                        period_stats['total_pnl'] += aligned_position.pnl
                        if aligned_position.pnl > 0:
                            period_stats['winning_trades'] += 1
                        else:
                            period_stats['losing_trades'] += 1
                        aligned_position = None

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nError during simulation: {e}")
        finally:
            # Print period summary regardless of how the simulation ended
            print(f"\nPeriod Summary for {period_name}:")
            print(f"Total Trades: {period_stats['total_trades']}")
            print(f"Winning Trades: {period_stats['winning_trades']}")
            print(f"Losing Trades: {period_stats['losing_trades']}")
            win_rate = (period_stats['winning_trades'] / period_stats['total_trades'] * 100) if period_stats['total_trades'] > 0 else 0
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total P&L: {period_stats['total_pnl']:.2f}%")
            print(f"Average P&L per Trade: {(period_stats['total_pnl'] / period_stats['total_trades']):.2f}%" if period_stats['total_trades'] > 0 else "N/A")
            
            # Print breakdown by strategy
            print("\nStrategy Breakdown:")
            print(f"Trend Trades: {len(period_stats['trend_trades'])}")
            print(f"Scalp Trades: {len(period_stats['scalp_trades'])}")
            print(f"Aligned Trades: {len(period_stats['aligned_trades'])}")

# Run the simulation
simulate_trading()
