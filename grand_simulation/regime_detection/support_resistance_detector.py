import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PriceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1 score indicating how strong the level is
    touches: int     # Number of times price has touched this level
    last_touch: datetime  # Last time price touched this level
    distance_pct: float   # Current distance from price as percentage

class SupportResistanceDetector:
    def __init__(self, 
                 window_size: int = 20,
                 min_touches: int = 2,
                 touch_threshold: float = 0.002,  # 0.2% threshold for considering a touch
                 strength_decay: float = 0.95,    # How quickly level strength decays
                 min_strength: float = 0.3,       # Minimum strength to consider a level valid
                 signal_threshold: float = 0.005): # 0.5% threshold for signal generation
        """
        Initialize the support/resistance detector
        
        Parameters:
        window_size (int): Number of periods to look back for level detection
        min_touches (int): Minimum number of touches required to establish a level
        touch_threshold (float): Price must be within this % of level to count as a touch
        strength_decay (float): How quickly level strength decays over time (0-1)
        min_strength (float): Minimum strength required to consider a level valid
        signal_threshold (float): Distance threshold for generating signals
        """
        self.window_size = window_size
        self.min_touches = min_touches
        self.touch_threshold = touch_threshold
        self.strength_decay = strength_decay
        self.min_strength = min_strength
        self.signal_threshold = signal_threshold
        self.levels: List[PriceLevel] = []
        
    def _find_local_extrema(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find local minima and maxima in price data
        
        Parameters:
        data (pd.DataFrame): Price data with 'high' and 'low' columns
        
        Returns:
        Tuple[List[float], List[float]]: Lists of local minima and maxima prices
        """
        highs = []
        lows = []
        
        for i in range(1, len(data) - 1):
            # Check for local maximum
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i+1]):
                highs.append(data['high'].iloc[i])
            
            # Check for local minimum
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i+1]):
                lows.append(data['low'].iloc[i])
        
        return lows, highs
    
    def _cluster_price_levels(self, prices: List[float], threshold: float) -> List[float]:
        """
        Cluster nearby price levels together
        
        Parameters:
        prices (List[float]): List of price points
        threshold (float): Maximum distance between prices to be considered same level
        
        Returns:
        List[float]: List of clustered price levels
        """
        if not prices:
            return []
            
        # Sort prices for clustering
        sorted_prices = sorted(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if price <= current_cluster[-1] * (1 + threshold):
                current_cluster.append(price)
            else:
                # Calculate cluster average and add to clusters
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [price]
        
        # Add final cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _update_level_strength(self, current_price: float, timestamp: datetime):
        """
        Update the strength of existing levels based on price interaction
        
        Parameters:
        current_price (float): Current price
        timestamp (datetime): Current timestamp
        """
        for level in self.levels:
            # Calculate distance to level
            distance_pct = abs(current_price - level.price) / level.price
            
            # Update distance
            level.distance_pct = distance_pct
            
            # Check if price is touching the level
            if distance_pct <= self.touch_threshold:
                level.touches += 1
                level.last_touch = timestamp
                # Increase strength on touch
                level.strength = min(1.0, level.strength + 0.1)
            else:
                # Decay strength over time
                level.strength *= self.strength_decay
    
    def _cleanup_weak_levels(self):
        """Remove levels that have become too weak"""
        self.levels = [level for level in self.levels 
                      if level.strength >= self.min_strength]
    
    def update(self, data: pd.DataFrame) -> Dict:
        """
        Update support/resistance levels and generate signals
        
        Parameters:
        data (pd.DataFrame): Price data with 'high', 'low', 'close' columns and datetime index
        
        Returns:
        Dict: Dictionary containing:
            - levels: List of current support/resistance levels
            - signal: Optional buy/sell signal
            - signal_strength: Strength of the signal (0-1)
        """
        if len(data) < self.window_size:
            return {'levels': [], 'signal': None, 'signal_strength': 0}
        
        # Get recent data
        recent_data = data.iloc[-self.window_size:]
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Find local extrema
        lows, highs = self._find_local_extrema(recent_data)
        
        # Cluster the levels
        support_levels = self._cluster_price_levels(lows, self.touch_threshold)
        resistance_levels = self._cluster_price_levels(highs, self.touch_threshold)
        
        # Update existing levels
        self._update_level_strength(current_price, current_time)
        
        # Add new levels
        for price in support_levels:
            if not any(abs(level.price - price) / price <= self.touch_threshold 
                      for level in self.levels if level.level_type == 'support'):
                self.levels.append(PriceLevel(
                    price=price,
                    level_type='support',
                    strength=0.5,  # Initial strength
                    touches=1,
                    last_touch=current_time,
                    distance_pct=abs(current_price - price) / price
                ))
        
        for price in resistance_levels:
            if not any(abs(level.price - price) / price <= self.touch_threshold 
                      for level in self.levels if level.level_type == 'resistance'):
                self.levels.append(PriceLevel(
                    price=price,
                    level_type='resistance',
                    strength=0.5,  # Initial strength
                    touches=1,
                    last_touch=current_time,
                    distance_pct=abs(current_price - price) / price
                ))
        
        # Clean up weak levels
        self._cleanup_weak_levels()
        
        # Generate signals
        signal = None
        signal_strength = 0
        
        # Find nearest levels
        nearest_support = min((level for level in self.levels 
                             if level.level_type == 'support'),
                            key=lambda x: x.distance_pct,
                            default=None)
        
        nearest_resistance = min((level for level in self.levels 
                                if level.level_type == 'resistance'),
                               key=lambda x: x.distance_pct,
                               default=None)
        
        # Check for bounce signals
        if nearest_support and nearest_support.distance_pct <= self.signal_threshold:
            # Check if price is moving up from support
            if (data['close'].iloc[-1] > data['close'].iloc[-2] and 
                data['close'].iloc[-2] > data['close'].iloc[-3]):
                signal = 'buy'
                signal_strength = nearest_support.strength
        
        elif nearest_resistance and nearest_resistance.distance_pct <= self.signal_threshold:
            # Check if price is moving down from resistance
            if (data['close'].iloc[-1] < data['close'].iloc[-2] and 
                data['close'].iloc[-2] < data['close'].iloc[-3]):
                signal = 'sell'
                signal_strength = nearest_resistance.strength
        
        return {
            'levels': self.levels,
            'signal': signal,
            'signal_strength': signal_strength,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    
    def get_levels_summary(self) -> pd.DataFrame:
        """
        Get a summary of current support/resistance levels
        
        Returns:
        pd.DataFrame: Summary of levels with their properties
        """
        if not self.levels:
            return pd.DataFrame()
            
        summary_data = []
        for level in self.levels:
            summary_data.append({
                'price': level.price,
                'type': level.level_type,
                'strength': round(level.strength, 3),
                'touches': level.touches,
                'last_touch': level.last_touch,
                'distance_pct': round(level.distance_pct * 100, 2)  # Convert to percentage
            })
        
        return pd.DataFrame(summary_data).sort_values('strength', ascending=False)
    
    def plot_levels(self, data: pd.DataFrame, ax=None):
        """
        Plot price data with support/resistance levels
        
        Parameters:
        data (pd.DataFrame): Price data with 'high', 'low', 'close' columns
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price
        ax.plot(data.index, data['close'], label='Price', color='blue', alpha=0.7)
        
        # Plot levels
        for level in self.levels:
            color = 'green' if level.level_type == 'support' else 'red'
            ax.axhline(y=level.price, color=color, linestyle='--', alpha=level.strength,
                      label=f"{level.level_type.title()} ({level.strength:.2f})")
        
        ax.set_title('Price with Support/Resistance Levels')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax 