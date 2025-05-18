import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from datetime import datetime, timedelta

from grand_simulation.technical_indicators import TechnicalIndicators
from grand_simulation.binance_api.historical_data import HistoricalDataFetcher
from grand_simulation.regime_detection.regime_detector import MarketRegimeDetector
import matplotlib.pyplot as plt
import numpy as np

def assess_regime_detection(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    """
    Assess regime detection on historical data for a specific period
    
    Parameters:
    symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
    timeframe (str): Candlestick timeframe (e.g., '1h', '4h', '1d')
    start_date (datetime): Start date for analysis
    end_date (datetime): End date for analysis
    
    Returns:
    tuple: (validation_df, accuracy_summary, regime_counts)
    """
    # Initialize data fetcher and regime detector
    data_fetcher = HistoricalDataFetcher()
    detector = MarketRegimeDetector()
    technical_indicators = TechnicalIndicators()
    
    print(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
    
    # Fetch historical data
    data = data_fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Applying technical indicators")
    df = data.copy()

    # Calculate technical indicators
    df = technical_indicators.calculate_emas(df)
    df = technical_indicators.calculate_adx(df)
    df = technical_indicators.calculate_bollinger_bands(df)
    df = technical_indicators.flag_volume_spike(df)
    df = technical_indicators.check_volatility(df)
    df = technical_indicators.calculate_rsi(df)

    print(f"Analyzing {len(df)} candles for regime detection")
    
    # Detect regimes
    regime_counts = detector.detect_regime(df)
    
    # Validate regime predictions
    validation_df = detector.validate_regime_prediction(regime_counts)
    accuracy_summary = detector.get_regime_accuracy_summary(validation_df)
    regime_counts = data['regime'].value_counts()
    
    return validation_df, accuracy_summary, regime_counts

def run_market_periods_analysis():
    """
    Run regime detection analysis across different market periods
    """
    # Define market periods to analyze
    market_periods = {
        # Bull Markets
        'Bull Market Q4 2020': (datetime(2020, 10, 1), datetime(2020, 12, 31)),  # Post-COVID recovery
        'Bull Market Q4 2017': (datetime(2017, 10, 1), datetime(2017, 12, 31)),  # Previous cycle peak
        
        # Bear Markets
        'Bear Market Q2 2022': (datetime(2022, 4, 1), datetime(2022, 6, 30)),  # LUNA crash
        'Bear Market Q1 2018': (datetime(2018, 1, 1), datetime(2018, 3, 31)),  # Post-2017 crash
        
        # High Volatility Periods
        'High Vol Q1 2020': (datetime(2020, 1, 1), datetime(2020, 3, 31)),  # COVID crash
        'High Vol Q4 2018': (datetime(2018, 10, 1), datetime(2018, 12, 31)),  # End of 2018 bear
        
        # Low Volatility Periods
        'Low Vol Q2 2019': (datetime(2019, 4, 1), datetime(2019, 6, 30)),  # Pre-2019 bull
        'Low Vol Q3 2016': (datetime(2016, 7, 1), datetime(2016, 9, 30)),  # Pre-2017 bull
        
        # Sideways/Choppy Markets
        'Sideways Q3 2019': (datetime(2019, 7, 1), datetime(2019, 9, 30)),  # Pre-breakout
        'Sideways Q2 2021': (datetime(2021, 4, 1), datetime(2021, 6, 30))   # Post-April 2021 peak
    }
    
    # Define timeframes to analyze
    timeframes = ['4h', '1h', '15m']
    
    # Store results for each period and timeframe
    results = {}
    
    # Analyze each market period
    for market_type, (start_date, end_date) in market_periods.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {market_type}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*80}")
        
        period_results = {}
        
        for timeframe in timeframes:
            print(f"\nAnalyzing {timeframe} timeframe...")
            
            try:
                validation_df, accuracy_summary, regime_counts = assess_regime_detection(
                    symbol='BTC/USDT',
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate overall accuracy (excluding warm-up and do-nothing periods)
                active_periods = validation_df[
                    ~validation_df['predicted_regime'].isin(['warm_up', 'do_nothing'])
                ]
                overall_accuracy = active_periods['is_correct'].mean() * 100
                
                # Calculate high-confidence accuracy
                high_conf_predictions = validation_df[validation_df['confidence'] > 0.7]
                high_conf_accuracy = high_conf_predictions['is_correct'].mean() * 100 if not high_conf_predictions.empty else 0
                
                # Store results
                period_results[timeframe] = {
                    'overall_accuracy': overall_accuracy,
                    'high_conf_accuracy': high_conf_accuracy,
                    'high_conf_count': len(high_conf_predictions),
                    'regime_distribution': regime_counts,
                    'accuracy_summary': accuracy_summary,
                    'validation_df': validation_df
                }
                
                # Print summary for this timeframe
                print(f"\nResults for {timeframe}:")
                print(f"Overall Accuracy: {overall_accuracy:.2f}%")
                print(f"High Confidence Accuracy: {high_conf_accuracy:.2f}%")
                print(f"Number of High Confidence Predictions: {len(high_conf_predictions)}")
                print("\nRegime Distribution:")
                for regime, count in regime_counts.items():
                    percentage = (count / len(validation_df) * 100).round(2)
                    print(f"{regime}: {count} periods ({percentage}%)")
                
            except Exception as e:
                print(f"Error analyzing {timeframe} timeframe: {str(e)}")
                continue
        
        results[market_type] = period_results
    
    # Print summary across all periods
    print("\n" + "="*80)
    print("Summary Across All Market Periods")
    print("="*80)
    
    for market_type, period_results in results.items():
        print(f"\n{market_type}:")
        for timeframe, metrics in period_results.items():
            print(f"\n{timeframe} timeframe:")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
            print(f"High Confidence Accuracy: {metrics['high_conf_accuracy']:.2f}%")
            print(f"High Confidence Predictions: {metrics['high_conf_count']}")
    
    return results

if __name__ == "__main__":
    print("Starting regime detection analysis across market periods...")
    results = run_market_periods_analysis()
    

    
    print("\nAnalysis completed!") 