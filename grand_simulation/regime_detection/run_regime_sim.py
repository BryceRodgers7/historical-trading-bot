import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from datetime import datetime, timedelta

from grand_simulation.regime_detection.regime_summary import RegimeSummary
from grand_simulation.regime_detection.regime_manager import RegimeManager
from grand_simulation.technical_indicators import TechnicalIndicators
from grand_simulation.binance_api.historical_data import HistoricalDataFetcher
from grand_simulation.regime_detection.regime_detector import MarketRegimeDetector
import matplotlib.pyplot as plt
import numpy as np

def run_regime_simulations(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    # probably want to pass in technical indicators parameters here, or create a simulation manager object similar to the trading sim

    regime_manager = RegimeManager()

    market_periods = {
        # Bull Markets
        'Bull Market Q4 2020': (datetime(2020, 10, 1), datetime(2021, 1, 1)),  # Post-COVID recovery
        'Bull Market Q4 2023': (datetime(2023, 11, 1), datetime(2024, 2, 1)),  # Previous cycle peak
        
        # # Bear Markets
        'Bear Market Q2 2022': (datetime(2022, 4, 1), datetime(2022, 7, 1)),  # LUNA crash
        'Bear Market Q1 2022': (datetime(2022, 1, 1), datetime(2022, 4, 1)),  
        
        # # High Volatility Periods
        'High Vol Q1 2020': (datetime(2020, 1, 1), datetime(2020, 4, 1)),  # COVID crash
        'High Vol Q3 2021': (datetime(2021, 5, 1), datetime(2021, 8, 1)),  # End of 2018 bear
        
        # # Low Volatility Periods
        'Low Vol Q4 2020': (datetime(2020, 9, 1), datetime(2020, 12, 1)),
        'Low Vol Q4 2023': (datetime(2023, 8, 1), datetime(2023, 11, 1)),  
        
        # Sideways/Choppy Markets
        'Sideways Q3 2019': (datetime(2020, 7, 1), datetime(2020, 10, 1)),  # Pre-breakout
        'Sideways Q2 2021': (datetime(2021, 4, 1), datetime(2021, 7, 1))   # Post-April 2021 peak
    }
    
    # Define timeframes to analyze
    timeframes = ['4h', '1h']
    ema_settings = [
        {'name': '', 'fast': 9, 'slow': 21},  # Default settings
        {'name': 'ema20', 'fast': 20, 'slow': 50}  # Alternative settings
    ]
    lookaheads = [6, 12]

    print("\nAdding simulations...")
    for market_type, (start_date, end_date) in market_periods.items():
        for timeframe in timeframes:
            for ema_setting in ema_settings:
                for lookahead in lookaheads:
                    name = f"{market_type} {timeframe}"
                    if ema_setting['name']:
                        name += f" {ema_setting['name']}"
                    if lookahead:
                        name += f" {lookahead} lookahead"
                    try:
                        print(f"Adding simulation: {name}")
                        regime_manager.add_sim(
                            name=name,
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            lookahead=lookahead,
                            ema_fast_window=ema_setting['fast'],
                            ema_slow_window=ema_setting['slow']
                        )
                    except Exception as e:
                        print(f"Error adding simulation {name}: {str(e)}")
                        print("Full stack trace:")
                        traceback.print_exc()
                        continue

    print("\nRunning simulations...")
    try:
        results = regime_manager.run_all_simulations()
        if not results:
            print("Warning: No results returned from run_all_simulations")
            return None
        return results
    except Exception as e:
        print(f"Error running simulations: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting regime detection analysis across market periods...")
    results = run_regime_simulations()
    
    if results is None:
        print("\nSimulation failed. No results to display.")
    else:
        # Create summary and print results
        summary = RegimeSummary(results)
        # summary.print_simulation_results()  # Print detailed results for each simulation
        summary.print_summary_table()       # Print consolidated summary table
    
    print("\nAnalysis completed!") 