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
                 touch_threshold: float = 0.002,  # 0.2% threshold for considering a touch
                 bounce_threshold: float = 0.001,  # 0.1% minimum bounce required
                 min_strength: float = 0.3,       # Minimum level strength to consider
                 min_candles_between_signals: int = 5,
                 support_levels: List[float] = None,
                 resistance_levels: List[float] = None):  # Minimum candles between signals
        """
        Initialize the support/resistance strategy
        
        Parameters:
        touch_threshold (float): Price must be within this % of level to count as a touch
        bounce_threshold (float): Minimum price movement required to confirm a bounce
        min_strength (float): Minimum strength required to consider a level
        min_candles_between_signals (int): Minimum candles between signals to avoid over-trading
        """
        self.touch_threshold = touch_threshold
        self.bounce_threshold = bounce_threshold
        self.min_strength = min_strength
        self.min_candles_between_signals = min_candles_between_signals
        self.last_signal_candle = None
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.update_levels(support_levels, resistance_levels)
    
    def update_levels(self, support_levels: List[float], resistance_levels: List[float]):
        """
        Update the support and resistance levels
        
        Parameters:
        support_levels (List[float]): List of support price levels
        resistance_levels (List[float]): List of resistance price levels
        """
        self.support_levels = sorted(support_levels)
        self.resistance_levels = sorted(resistance_levels)
    
    def _find_nearest_level(self, price: float) -> Optional[Tuple[float, str]]:
        """
        Find the nearest support or resistance level to the current price
        
        Parameters:
        price (float): Current price
        
        Returns:
        Optional[Tuple[float, str]]: Tuple of (nearest_level, level_type) if within threshold, None otherwise
        """
        if not self.support_levels and not self.resistance_levels:
            return None
            
        # Find nearest support and resistance
        nearest_support = min(self.support_levels, key=lambda x: abs(x - price)) if self.support_levels else None
        nearest_resistance = min(self.resistance_levels, key=lambda x: abs(x - price)) if self.resistance_levels else None
        
        # Calculate distances
        support_dist = abs(nearest_support - price) / price if nearest_support else float('inf')
        resistance_dist = abs(nearest_resistance - price) / price if nearest_resistance else float('inf')
        
        # Return nearest level if within threshold
        if support_dist <= self.touch_threshold and support_dist <= resistance_dist:
            return nearest_support, 'support'
        elif resistance_dist <= self.touch_threshold:
            return nearest_resistance, 'resistance'
        return None
    
    def _check_bounce(self, data, level, level_type, current_idx: int) -> bool:
        """
        Check if price has bounced off a level
        
        Parameters:
        data (pd.DataFrame): Price data
        level (float): The level price
        level_type (str): 'support' or 'resistance'
        current_idx (int): Current candle index
        
        Returns:
        bool: True if a bounce is detected
        """
        if current_idx < 2:  # Need at least 3 candles to check bounce
            return False
            
        current_price = data['close'].iloc[current_idx]
        prev_price = data['close'].iloc[current_idx - 1]
        prev_prev_price = data['close'].iloc[current_idx - 2]
        
        if level_type == 'support':
            # For support, check if price bounced up
            if (current_price > prev_price and  # Current candle is up
                prev_price > prev_prev_price and  # Previous candle was up
                abs(prev_price - level) / level <= self.touch_threshold and  # Touched support
                (current_price - prev_price) / prev_price >= self.bounce_threshold):  # Sufficient bounce
                return True
                
        elif level_type == 'resistance':
            # For resistance, check if price bounced down
            if (current_price < prev_price and  # Current candle is down
                prev_price < prev_prev_price and  # Previous candle was down
                abs(prev_price - level) / level <= self.touch_threshold and  # Touched resistance
                (prev_price - current_price) / prev_price >= self.bounce_threshold):  # Sufficient bounce
                return True
        
        return False
    
    def _is_support_bounce(self, df, support_level, tolerance_pct=0.005):
        """
        Check if the most recent candle shows a confirmed bounce off support.
        
        Parameters:
        df (pd.DataFrame): The last 'window' of data before the current candle
        support_level (float): The support price level to check
        tolerance_pct (float): Percentage tolerance for considering price near level
        
        Returns:
        bool: True if a bounce off support is confirmed
        """
        # Define conditions for the most recent candle
        latest = df.iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) >= 2 else np.nan

        near_support = (support_level * (1 - tolerance_pct)) <= latest['low'] <= (support_level * (1 + tolerance_pct))
        bullish_candle = latest['close'] > latest['open']
        rsi_rising = latest['rsi'] > prev_rsi
        volume_spike = latest['volume_spike']

        return all([near_support, bullish_candle, rsi_rising, volume_spike])
    
    def _is_resistance_bounce(self, df, resistance_level, tolerance_pct=0.005):
        """
        Check if the most recent candle shows a confirmed bounce off resistance.
        
        Parameters:
        df (pd.DataFrame): The last 'window' of data before the current candle
        resistance_level (float): The resistance price level to check
        tolerance_pct (float): Percentage tolerance for considering price near level
        
        Returns:
        bool: True if a bounce off resistance is confirmed
        """
        # Define conditions for the most recent candle
        latest = df.iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) >= 2 else np.nan

        near_resistance = (resistance_level * (1 - tolerance_pct)) <= latest['high'] <= (resistance_level * (1 + tolerance_pct))
        bearish_candle = latest['close'] < latest['open']
        rsi_falling = latest['rsi'] < prev_rsi
        volume_spike = latest['volume_spike']

        return all([near_resistance, bearish_candle, rsi_falling, volume_spike])
    
    def generate_signals(self, df):
        """
        Generate trading signal for the most recent candle based on the nearest support or resistance level.
        
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
            
        # Get current price and check if enough time has passed since last signal
        current_price = df['close'].iloc[-1]
        # if self.last_signal_candle is not None:
        #     candles_since_last = len(df) - 1 - self.last_signal_candle
        #     if candles_since_last < self.min_candles_between_signals:
        #         return 0
        
        # Find nearest level
        nearest_level = self._find_nearest_level(current_price)
        if nearest_level is None:
            return 0
            
        level_price, level_type = nearest_level
        
        # Check for bounce based on level type
        if level_type == 'support' and self._is_support_bounce(df, level_price):
            # self.last_signal_candle = len(df) - 1
            return 1
        elif level_type == 'resistance' and self._is_resistance_bounce(df, level_price):
            # self.last_signal_candle = len(df) - 1
            return -1
            
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
                'min_strength': self.min_strength,
                'min_candles_between_signals': self.min_candles_between_signals
            },
            'current_state': {
                'support_levels': len(self.support_levels),
                'resistance_levels': len(self.resistance_levels),
                'last_signal_candle': self.last_signal_candle
            }
        } 