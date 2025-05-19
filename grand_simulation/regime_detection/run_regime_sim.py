import sys
import os
import traceback


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from datetime import datetime, timedelta

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
        'Bull Market Q4 2020': (datetime(2020, 10, 1), datetime(2020, 12, 31)),  # Post-COVID recovery
        # 'Bull Market Q4 2017': (datetime(2017, 10, 1), datetime(2017, 12, 31)),  # Previous cycle peak
        
        # # Bear Markets
        # 'Bear Market Q2 2022': (datetime(2022, 4, 1), datetime(2022, 6, 30)),  # LUNA crash
        # 'Bear Market Q1 2018': (datetime(2018, 1, 1), datetime(2018, 3, 31)),  # Post-2017 crash
        
        # # High Volatility Periods
        # 'High Vol Q1 2020': (datetime(2020, 1, 1), datetime(2020, 3, 31)),  # COVID crash
        # 'High Vol Q4 2018': (datetime(2018, 10, 1), datetime(2018, 12, 31)),  # End of 2018 bear
        
        # # Low Volatility Periods
        # 'Low Vol Q2 2019': (datetime(2019, 4, 1), datetime(2019, 6, 30)),  # Pre-2019 bull
        # 'Low Vol Q3 2016': (datetime(2016, 7, 1), datetime(2016, 9, 30)),  # Pre-2017 bull
        
        # Sideways/Choppy Markets
        # 'Sideways Q3 2019': (datetime(2019, 7, 1), datetime(2019, 9, 30)),  # Pre-breakout
        'Sideways Q2 2021': (datetime(2021, 4, 1), datetime(2021, 6, 30))   # Post-April 2021 peak
    }
    
    # Define timeframes to analyze
    timeframes = ['4h', '1h']
    ema_settings = [
        {'name': '', 'fast': 9, 'slow': 21},  # Default settings
        {'name': 'ema20', 'fast': 20, 'slow': 50}  # Alternative settings
    ]
    lookaheads = [6, 12]

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
                        )
                    except Exception as e:
                        print(f"Error adding simulation: {str(e)}")
                        continue


    try:
        regime_manager.run_all_simulations()
    except Exception as e:
        print(f"Error running simulations: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()
        return None

    print(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
    

if __name__ == "__main__":
    print("Starting regime detection analysis across market periods...")
    results = run_regime_simulations()
    

    
    print("\nAnalysis completed!") 