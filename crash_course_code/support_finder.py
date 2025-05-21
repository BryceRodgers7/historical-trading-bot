import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from grand_simulation.binance_api.historical_data import HistoricalDataFetcher
import matplotlib.pyplot as plt

def calculate_level_strength(df, level, tolerance=0.01, window=20):
    """
    Calculate the strength of a support/resistance level based on:
    1. Number of touches
    2. How recent the touches are
    3. How strong the bounces were
    
    Parameters:
    - df: DataFrame with price data
    - level: Price level to analyze
    - tolerance: Percentage distance to consider a touch
    - window: Number of candles to look back for recent touches
    
    Returns:
    - float: Strength score (0-1)
    """
    touches = 0
    recent_touches = 0
    bounce_strength = 0
    
    # Look for touches within tolerance
    for i in range(1, len(df) - 1):
        price = df['close'].iloc[i]
        if abs(price - level) / level <= tolerance:
            touches += 1
            
            # Check if touch is recent
            if i >= len(df) - window:
                recent_touches += 1
            
            # Calculate bounce strength
            prev_price = df['close'].iloc[i-1]
            next_price = df['close'].iloc[i+1]
            if level > price:  # Support level
                bounce = (next_price - price) / price
            else:  # Resistance level
                bounce = (price - next_price) / price
            bounce_strength += max(0, bounce)
    
    # Calculate final strength score
    if touches == 0:
        return 0
    
    # Weight recent touches more heavily
    recency_score = recent_touches / window
    touch_score = min(1.0, touches / 10)  # Cap at 10 touches
    bounce_score = min(1.0, bounce_strength / 0.1)  # Cap at 10% total bounce
    
    # Combine scores with weights
    strength = (0.4 * touch_score + 0.4 * recency_score + 0.2 * bounce_score)
    return strength

def find_support_resistance(df, order=20, tolerance=0.01, min_strength=0.3, min_distance=0.02):
    """
    Identifies prominent support and resistance levels using local minima and maxima.
    
    Parameters:
    - df: DataFrame with at least 'high' and 'low' columns
    - order: how many points before and after to use for detecting local extrema
    - tolerance: percentage difference within which to consider levels as the same
    - min_strength: minimum strength score (0-1) to keep a level
    - min_distance: minimum distance between levels as percentage of price
    
    Returns:
    - support_levels: list of prominent support levels with their strengths
    - resistance_levels: list of prominent resistance levels with their strengths
    """
    # Find local minima (support) and maxima (resistance)
    local_min = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    local_max = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]

    supports = df['low'].iloc[local_min].values
    resistances = df['high'].iloc[local_max].values

    def bin_and_score_levels(levels, tolerance, min_strength, min_distance):
        """Group similar levels and calculate their strength."""
        if len(levels) == 0:
            return []
            
        # Sort levels
        levels = sorted(levels)
        binned = []
        current_bin = [levels[0]]
        
        # First pass: bin similar levels
        for level in levels[1:]:
            if level <= current_bin[-1] * (1 + tolerance):
                current_bin.append(level)
            else:
                # Calculate average for this bin
                avg_level = sum(current_bin) / len(current_bin)
                binned.append(avg_level)
                current_bin = [level]
        
        # Add final bin
        if current_bin:
            avg_level = sum(current_bin) / len(current_bin)
            binned.append(avg_level)
        
        # Second pass: calculate strength and filter
        scored_levels = []
        for level in binned:
            strength = calculate_level_strength(df, level, tolerance)
            if strength >= min_strength:
                scored_levels.append((level, strength))
        
        # Third pass: filter out levels that are too close to stronger levels
        filtered_levels = []
        scored_levels.sort(key=lambda x: x[1], reverse=True)  # Sort by strength
        
        for level, strength in scored_levels:
            # Check if this level is far enough from stronger levels
            if not any(abs(level - stronger_level) / stronger_level < min_distance 
                      for stronger_level, _ in filtered_levels):
                filtered_levels.append((level, strength))
        
        return filtered_levels

    # Find and score levels
    support_levels = bin_and_score_levels(supports, tolerance, min_strength, min_distance)
    resistance_levels = bin_and_score_levels(resistances, tolerance, min_strength, min_distance)
    
    # Sort by strength
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1], reverse=True)
    
    return support_levels, resistance_levels

def plot_levels(df, support_levels, resistance_levels):
    """Plot price data with support and resistance levels."""
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    
    # Plot support levels with strength indicated by line thickness
    for level, strength in support_levels:
        plt.axhline(y=level, color='green', linestyle='--', 
                   alpha=0.7, linewidth=1 + strength * 2,
                   label=f'Support ({strength:.2f})' if level == support_levels[0][0] else "")
    
    # Plot resistance levels with strength indicated by line thickness
    for level, strength in resistance_levels:
        plt.axhline(y=level, color='red', linestyle='--', 
                   alpha=0.7, linewidth=1 + strength * 2,
                   label=f'Resistance ({strength:.2f})' if level == resistance_levels[0][0] else "")
    
    plt.title('BTC/USDT Price with Support/Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to fetch data and analyze support/resistance levels."""
    # Initialize data fetcher
    data_fetcher = HistoricalDataFetcher()
    
    # Calculate date range (1 year of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch historical data
    print(f"Fetching historical data from {start_date.date()} to {end_date.date()}")
    data = data_fetcher.fetch_historical_data(
        symbol='BTC/USDT',
        timeframe='4h',  # Using 4h timeframe for better level detection
        start_date=start_date,
        end_date=end_date
    )
    
    if data is None or data.empty:
        print("Failed to fetch historical data")
        return
    
    print(f"Fetched {len(data)} candles")
    
    # Find support and resistance levels
    support_levels, resistance_levels = find_support_resistance(
        data, 
        order=20,          # Look at 20 candles before and after for more significant extrema
        tolerance=0.01,    # 1% tolerance for level clustering
        min_strength=0.3,  # Minimum strength score to keep a level
        min_distance=0.02  # Minimum 2% distance between levels
    )
    
    # Print results
    print("\nSupport/Resistance Analysis Results:")
    print("=" * 50)
    
    print("\nSupport Levels (sorted by strength):")
    for i, (level, strength) in enumerate(support_levels, 1):
        current_price = data['close'].iloc[-1]
        distance_pct = abs(current_price - level) / current_price * 100
        print(f"{i}. ${level:.2f} (Strength: {strength:.2f}, Distance: {distance_pct:.1f}%)")
    
    print("\nResistance Levels (sorted by strength):")
    for i, (level, strength) in enumerate(resistance_levels, 1):
        current_price = data['close'].iloc[-1]
        distance_pct = abs(current_price - level) / current_price * 100
        print(f"{i}. ${level:.2f} (Strength: {strength:.2f}, Distance: {distance_pct:.1f}%)")
    
    # Plot the results
    plot_levels(data, support_levels, resistance_levels)

if __name__ == "__main__":
    main()


