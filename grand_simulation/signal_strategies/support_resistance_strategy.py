from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class PriceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 score indicating how strong the level is

class SupportResistanceStrategy:
    def __init__(self, 
                 touch_threshold: float = 0.02,  # 5% threshold for considering a touch
                 bounce_threshold: float = 0.01,  # 0.1% minimum bounce required
                 min_candles_between_signals: int = 5,
                 price_levels: List[float] = None):  # Single list of support/resistance levels
        """
        Initialize the support/resistance strategy
        
        Parameters:
        touch_threshold (float): Price must be within this % of level to count as a touch
        bounce_threshold (float): Minimum price movement required to confirm a bounce
        min_candles_between_signals (int): Minimum candles between signals to avoid over-trading
        price_levels (List[float]): List of price levels that can act as both support and resistance
        """
        self.touch_threshold = touch_threshold
        self.bounce_threshold = bounce_threshold
        self.min_candles_between_signals = min_candles_between_signals
        self.last_signal_candle = None
        self.price_levels: List[float] = []
        self.update_levels(price_levels)
    
    def update_levels(self, price_levels: List[float]):
        """
        Update the price levels
        
        Parameters:
        price_levels (List[float]): List of price levels that can act as both support and resistance
        """
        self.price_levels = sorted(price_levels) if price_levels else []
    
    def _find_nearest_level(self, price: float) -> Optional[Tuple[float, str]]:
        """
        Find the nearest price level to the current price and determine if it's acting as support or resistance
        
        Parameters:
        price (float): Current price
        
        Returns:
        Optional[Tuple[float, str]]: Tuple of (nearest_level, level_type) if within threshold, None otherwise
            level_type is 'support' if price is below level, 'resistance' if price is above level
        """
        if not self.price_levels:
            return None
            
        # Find nearest level
        nearest_level = min(self.price_levels, key=lambda x: abs(x - price))
        
        # Calculate distance
        level_dist = abs(nearest_level - price) / price
        
        # Return nearest level if within threshold
        if level_dist <= self.touch_threshold:
            # If price is below level, it's acting as resistance
            # If price is above level, it's acting as support
            level_type = 'resistance' if price < nearest_level else 'support'
            return nearest_level, level_type
        return None
    
    def _check_bounce(self, df, level: float, level_type: str) -> bool:
        """
        Check if the most recent candle shows a confirmed bounce off a price level.
        
        Parameters:
        df (pd.DataFrame): The last 'window' of data before the current candle
        level (float): The price level to check
        level_type (str): 'support' or 'resistance' indicating how the level is being used
        tolerance_pct (float): Percentage tolerance for considering price near level
        
        Returns:
        bool: True if a bounce is confirmed
        """
        # Define conditions for the most recent candle
        latest = df.iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) >= 2 else np.nan

        # Check if price is near the level
        if level_type == 'support':
            near_level = (level * (1 - self.touch_threshold)) <= latest['low'] <= (level * (1 + self.touch_threshold))
            correct_candle = latest['close'] > latest['open']  # Bullish for support
            rsi_moving = latest['rsi'] > prev_rsi  # Rising RSI for support
        else:  # resistance
            near_level = (level * (1 - self.touch_threshold)) <= latest['high'] <= (level * (1 + self.touch_threshold))
            correct_candle = latest['close'] < latest['open']  # Bearish for resistance
            rsi_moving = latest['rsi'] < prev_rsi  # Falling RSI for resistance

        volume_spike = latest['volume_spike']

        return all([near_level, correct_candle, rsi_moving, volume_spike])
    
    def generate_signals(self, df):
        """
        Generate trading signal for the most recent candle based on the nearest price level.
        
        Parameters:
        df (pd.DataFrame): Price data with at least 2 candles for RSI comparison
        
        Returns:
        int: Signal value:
            - 0: No action
            - 1: Buy signal (bounce off support)
            - -1: Sell signal (bounce off resistance)
        """
        # Need at least 2 candles for RSI comparison
        if len(df) < 2:
            return 0
            
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Find nearest level
        nearest_level = self._find_nearest_level(current_price)
        if nearest_level is None:
            return 0
            
        level_price, level_type = nearest_level
        
        # Check for bounce
        if self._check_bounce(df, level_price, level_type):
            return 1 if level_type == 'support' else -1
            
        return 0
    
    def get_strategy_info(self) -> Dict:
        """
        Get information about the strategy configuration
        
        Returns:
        Dict: Strategy parameters and current state
        """
        return {
            'name': 'Support/Resistance Bounce Strategy',
            'parameters': {
                'touch_threshold': self.touch_threshold,
                'bounce_threshold': self.bounce_threshold,
                'min_candles_between_signals': self.min_candles_between_signals
            },
            'current_state': {
                'price_levels': len(self.price_levels),
                'last_signal_candle': self.last_signal_candle
            }
        } 