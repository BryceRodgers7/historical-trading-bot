import pandas as pd
import numpy as np
from market_regime_detector import MarketRegimeDetector
from regime_strategies import RegimeStrategyFactory
from technical_indicators import TechnicalIndicators

class RegimeSimulation:
    def __init__(self, name, symbol, timeframe, start_date, end_date, initial_balance=10000, lookback_window=100, ema_fast_window=9, ema_slow_window=21, bb_window=20, bb_std=2, rsi_window=14):
        self.initial_balance = initial_balance
        self.regime_detector = MarketRegimeDetector()
        self.strategy_factory = RegimeStrategyFactory()
        self.technical_indicators = TechnicalIndicators(adx_period=14, ema_fast_window=ema_fast_window, ema_slow_window=ema_slow_window, bb_window=bb_window, bb_std=bb_std, rsi_window=rsi_window)
        self.strategy_parms = {}
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback_window
        self.warmup_periods = max(
            self.regime_detector.trend_window,
            self.regime_detector.volatility_window,
            self.technical_indicators.rsi_window,
            self.technical_indicators.ema_slow_window,
            self.technical_indicators.bb_window,
            self.technical_indicators.adx_period
        )

    def _update_position(self, df, i, current_position, entry_price, entry_time, trades):
        """
        Update trading position based on signal
        
        Parameters:
        df (pd.DataFrame): Price data
        i (int): Current index
        current_position (int): Current position (1=long, -1=short, 0=flat)
        entry_price (float): Entry price of current position
        entry_time (datetime): Entry time of current position
        trades (list): List of completed trades
        
        Returns:
        tuple: (new_position, new_entry_price, new_entry_time)
        """
        signal = df['signal'].iloc[i]
        
        # Only change position if signal is different from current position
        if signal != current_position:
            # Close existing position if any
            if current_position != 0:
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - entry_price) * current_position
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': current_position,
                    'pnl': pnl,
                    'regime': df['regime'].iloc[i-1]
                })
            
            # Open new position based on signal
            # signal = 1: Buy (go long)
            # signal = -1: Sell (go short)
            # signal = 0: No position
            new_position = signal
            new_entry_price = df['close'].iloc[i] if signal != 0 else 0
            new_entry_time = df.index[i] if signal != 0 else None
            
            return new_position, new_entry_price, new_entry_time
        
        return current_position, entry_price, entry_time

    def run_simulation(self, data):
        """
        Run simulation with regime-based strategies
        
        Parameters:
        data (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
        pd.DataFrame: Simulation results with trades and performance metrics
        """
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
        
        # Detect market regimes
        df['regime'] = self.regime_detector.detect_regime(df)
        
        # Initialize simulation columns with proper data types
        df['signal'] = pd.Series(0, index=df.index, dtype='int8')
        df['position'] = pd.Series(0, index=df.index, dtype='int8')
        df['balance'] = pd.Series(self.initial_balance, index=df.index, dtype='float64')
        df['holdings'] = pd.Series(0.0, index=df.index, dtype='float64')
        df['equity'] = pd.Series(self.initial_balance, index=df.index, dtype='float64')
        
        # Track trades
        trades = []
        current_position = 0
        entry_price = 0
        entry_time = None
        
        # Skip warm-up period
        for i in range(self.warmup_periods, len(df)):
            # Get current regime and strategy
            current_regime = df['regime'].iloc[i]
            strategy = self.strategy_factory.get_strategy(current_regime)
            
            # Generate signals if we have a strategy
            if strategy:
                # Get signals for the current window of data
                window_data = df.iloc[max(0, i-self.lookback_window):i+1]
                signals = strategy.generate_signals(window_data)
                df.loc[df.index[i], 'signal'] = signals.iloc[-1]
            
            # Update position based on signal
            current_position, entry_price, entry_time = self._update_position(
                df, i, current_position, entry_price, entry_time, trades
            )
            
            # Update position and equity
            df.loc[df.index[i], 'position'] = current_position
            df.loc[df.index[i], 'holdings'] = current_position * df['close'].iloc[i]
            df.loc[df.index[i], 'equity'] = df['balance'].iloc[i-1] + df['holdings'].iloc[i]
            
            # Update balance when position is closed
            if current_position == 0 and df['position'].iloc[i-1] != 0:
                df.loc[df.index[i], 'balance'] = df['equity'].iloc[i]
            else:
                df.loc[df.index[i], 'balance'] = df['balance'].iloc[i-1]
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['return'] = trades_df['pnl'] / trades_df['entry_price']
            trades_df['win'] = trades_df['pnl'] > 0
            
            # Calculate metrics only for the trading period (after warm-up)
            trading_equity = df['equity'].iloc[self.warmup_periods:]
            trading_returns = trading_equity.pct_change().dropna()
            
            performance = {
                'total_trades': len(trades_df),
                'winning_trades': trades_df['win'].sum(),
                'win_rate': trades_df['win'].mean() * 100,
                'avg_return': trades_df['return'].mean() * 100,
                'total_return': trades_df['return'].sum() * 100,
                'max_drawdown': self._calculate_max_drawdown(trading_equity),
                'sharpe_ratio': self._calculate_sharpe_ratio(trading_equity),
                'warmup_periods': self.warmup_periods,
                'trading_periods': len(trading_equity),
                'possible_profitable': self.count_profitable_long_trades(df)
            }
            
            # Performance by regime
            regime_performance = trades_df.groupby('regime').agg({
                'return': ['count', 'mean', 'sum'],
                'win': 'mean'
            }).round(4)
            
            return df, trades_df, performance, regime_performance
        
        return df, pd.DataFrame(), {}, pd.DataFrame()
    
    def _calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown from equity curve"""
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        return abs(drawdowns.min()) * 100
    
    def _calculate_sharpe_ratio(self, equity, risk_free_rate=0.02):
        """Calculate Sharpe ratio from equity curve"""
        returns = equity.pct_change().dropna()
        excess_returns = returns - risk_free_rate/365  # Daily risk-free rate
        return np.sqrt(365) * excess_returns.mean() / excess_returns.std()
    
    def count_profitable_long_trades(self, df, profit_threshold=0.01, max_holding_period=20):
        """
        Counts the number of possible profitable long trades in OHLCV data,
        skipping the first few 'warm-up' periods.

        Parameters:
            df (pd.DataFrame): OHLCV dataframe with 'low' and 'close' columns.
            profit_threshold (float): Minimum % gain to count as profitable (e.g., 0.01 for 1%).
            max_holding_period (int): Maximum candles to wait for profit to be hit.
            warmup_periods (int): Number of periods to skip at the start.

        Returns:
            int: Number of potentially profitable long trade setups.
        """
        df = df.copy().reset_index(drop=True)
        total_profitable_trades = 0

        for i in range(self.warmup_periods, len(df) - max_holding_period):
            entry_price = df.loc[i, 'low']
            target_price = entry_price * (1 + profit_threshold)

            future_closes = df.loc[i+1:i+max_holding_period, 'close']
            if (future_closes > target_price).any():
                total_profitable_trades += 1

        return total_profitable_trades

