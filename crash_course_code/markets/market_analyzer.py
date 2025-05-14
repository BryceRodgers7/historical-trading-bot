# ignore this file for now

import pandas as pd
import numpy as np
from enum import Enum

class MarketCondition(Enum):
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    SIDEWAYS = "Sideways Market"
    BREAKOUT_UP = "Breakout Up"
    BREAKOUT_DOWN = "Breakout Down"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    UNKNOWN = "Unknown"

class MarketAnalyzer:
    def __init__(self, volatility_window=20, trend_window=50, breakout_threshold=0.05):
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.breakout_threshold = breakout_threshold

    def analyze_market(self, data):
        """Analyze market conditions based on price action and volatility"""
        # Calculate basic indicators
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=self.volatility_window).std()
        data['sma'] = data['close'].rolling(window=self.trend_window).mean()
        
        # Calculate price ranges
        data['high_low_range'] = data['high'] - data['low']
        data['close_open_range'] = abs(data['close'] - data['open'])
        
        # Calculate trend strength
        data['trend_strength'] = (data['close'] - data['sma']) / data['sma']
        
        # Initialize market condition column
        data['market_condition'] = MarketCondition.UNKNOWN
        
        # Analyze each period
        for i in range(self.trend_window, len(data)):
            # Get the current window of data
            window = data.iloc[i-self.trend_window:i+1]
            
            # Determine market condition
            condition = self._determine_market_condition(window)
            data.iloc[i, data.columns.get_loc('market_condition')] = condition
        
        return data

    def _determine_market_condition(self, window):
        """Determine market condition based on price action and volatility"""
        # Calculate key metrics
        avg_volatility = window['volatility'].mean()
        trend_strength = window['trend_strength'].iloc[-1]
        price_range = window['high'].max() - window['low'].min()
        avg_range = window['high_low_range'].mean()
        
        # Check for breakout
        if trend_strength > self.breakout_threshold:
            return MarketCondition.BREAKOUT_UP
        elif trend_strength < -self.breakout_threshold:
            return MarketCondition.BREAKOUT_DOWN
        
        # Check for bull/bear market
        if trend_strength > 0.02:  # 2% above SMA
            return MarketCondition.BULL_MARKET
        elif trend_strength < -0.02:  # 2% below SMA
            return MarketCondition.BEAR_MARKET
        
        # Check for sideways market
        if abs(trend_strength) < 0.01:  # Less than 1% deviation from SMA
            return MarketCondition.SIDEWAYS
        
        # Check for volatility
        if avg_volatility > window['volatility'].mean() * 1.5:
            return MarketCondition.HIGH_VOLATILITY
        elif avg_volatility < window['volatility'].mean() * 0.5:
            return MarketCondition.LOW_VOLATILITY
        
        return MarketCondition.UNKNOWN

    def get_market_summary(self, data):
        """Get a summary of market conditions over the period"""
        # Get the last valid market condition
        last_condition = data['market_condition'].iloc[-1]
        
        # Calculate percentage of time in each condition
        condition_counts = data['market_condition'].value_counts(normalize=True) * 100
        
        # Calculate average volatility
        avg_volatility = data['volatility'].mean()
        
        # Calculate trend strength
        trend_strength = data['trend_strength'].iloc[-1]
        
        return {
            'current_condition': last_condition,
            'condition_distribution': condition_counts.to_dict(),
            'average_volatility': avg_volatility,
            'trend_strength': trend_strength
        }

    def print_market_summary(self, data):
        """Print a human-readable summary of market conditions"""
        summary = self.get_market_summary(data)
        
        print("\nMarket Analysis Summary:")
        print(f"Current Market Condition: {summary['current_condition'].value}")
        print("\nCondition Distribution:")
        for condition, percentage in summary['condition_distribution'].items():
            print(f"{condition.value}: {percentage:.1f}%")
        print(f"\nAverage Volatility: {summary['average_volatility']:.4f}")
        print(f"Current Trend Strength: {summary['trend_strength']:.2%}")
        
        # Add interpretation
        print("\nInterpretation:")
        if summary['current_condition'] == MarketCondition.BULL_MARKET:
            print("Market is in a bullish trend with prices above the moving average.")
        elif summary['current_condition'] == MarketCondition.BEAR_MARKET:
            print("Market is in a bearish trend with prices below the moving average.")
        elif summary['current_condition'] == MarketCondition.SIDEWAYS:
            print("Market is trading sideways with no clear trend.")
        elif summary['current_condition'] in [MarketCondition.BREAKOUT_UP, MarketCondition.BREAKOUT_DOWN]:
            print("Market is experiencing a significant breakout move.")
        elif summary['current_condition'] == MarketCondition.HIGH_VOLATILITY:
            print("Market is experiencing high volatility, indicating potential trading opportunities.")
        elif summary['current_condition'] == MarketCondition.LOW_VOLATILITY:
            print("Market is experiencing low volatility, indicating potential range-bound trading.") 