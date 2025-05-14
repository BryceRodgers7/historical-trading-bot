import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from markets.market_detector import MarketDetector

class DetectorAssessor:
    def __init__(self):
        self.exchange = ccxt.binanceus()
        self.detector = MarketDetector()

    def fetch_historical_data(self, symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
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
                ohlcv = self.exchange.fetch_ohlcv(
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

    def assess_market_detection(self, data, lookahead=10):
        """Assess the accuracy of market detection"""
        # Get market detection results
        detection_results = self.detector.evaluate_market_detection(data, lookahead)
        
        # Calculate accuracy metrics
        total_predictions = len(detection_results)
        correct_predictions = detection_results['correct'].sum()
        accuracy = (correct_predictions / total_predictions) * 100
        
        # Calculate accuracy by market type
        market_type_accuracy = detection_results.groupby('market_label')['correct'].agg(['count', 'sum', 'mean'])
        market_type_accuracy['accuracy'] = market_type_accuracy['mean'] * 100
        
        # Print results
        print("\nMarket Detection Assessment:")
        print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        
        print("\nAccuracy by Market Type:")
        for market_type, stats in market_type_accuracy.iterrows():
            print(f"{market_type}:")
            print(f"  Number of predictions: {stats['count']}")
            print(f"  Accuracy: {stats['accuracy']:.2f}%")
        
        # Print recent predictions
        print("\nRecent Market Predictions:")
        recent_predictions = detection_results.tail(5)
        for _, row in recent_predictions.iterrows():
            print(f"Time: {row['index']}")
            print(f"Market Type: {row['market_label']}")
            print(f"Future Return: {row['future_return_pct']:.2%}")
            print(f"Correct: {row['correct']}")
            print("-" * 30)
        
        return detection_results

    def assess_multiple_periods(self, periods, symbol='BTC/USDT', timeframe='1h', lookahead=10):
        """Assess market detection across multiple time periods"""
        all_results = []
        period_summaries = []
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"\n{'='*50}")
            print(f"Testing period: {period_name}")
            print(f"From: {start_date.date()} to {end_date.date()}")
            print(f"{'='*50}")
            
            try:
                # Fetch data for this period
                data = self.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Assess market detection
                results = self.assess_market_detection(data, lookahead)
                
                # Add period information to results
                results['period'] = period_name
                all_results.append(results)
                
                # Calculate period summary
                total_predictions = len(results)
                correct_predictions = results['correct'].sum()
                accuracy = (correct_predictions / total_predictions) * 100
                
                # Calculate market type distribution
                market_type_dist = results['market_label'].value_counts(normalize=True) * 100
                
                # Store period summary
                period_summaries.append({
                    'period': period_name,
                    'start_date': start_date.date(),
                    'end_date': end_date.date(),
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'market_type_dist': market_type_dist.to_dict()
                })
                
            except Exception as e:
                print(f"Error processing period {period_name}: {e}")
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results)
            
            # Calculate overall statistics
            print("\nOverall Statistics Across All Periods:")
            print(f"Total predictions: {len(combined_results)}")
            print(f"Overall accuracy: {(combined_results['correct'].mean() * 100):.2f}%")
            
            # Calculate accuracy by period
            period_accuracy = combined_results.groupby('period')['correct'].mean() * 100
            print("\nAccuracy by Period:")
            for period, accuracy in period_accuracy.items():
                print(f"{period}: {accuracy:.2f}%")
            
            # Calculate accuracy by market type across all periods
            market_type_accuracy = combined_results.groupby('market_label')['correct'].mean() * 100
            print("\nAccuracy by Market Type (All Periods):")
            for market_type, accuracy in market_type_accuracy.items():
                print(f"{market_type}: {accuracy:.2f}%")
            
            # Print period summaries ordered by accuracy
            print("\nDetailed Period Summaries (Ordered by Accuracy):")
            print("=" * 80)
            
            # Sort period summaries by accuracy
            period_summaries.sort(key=lambda x: x['accuracy'], reverse=True)
            
            for summary in period_summaries:
                print(f"\nPeriod: {summary['period']}")
                print(f"Date Range: {summary['start_date']} to {summary['end_date']}")
                print(f"Accuracy: {summary['accuracy']:.2f}%")
                print(f"Total Predictions: {summary['total_predictions']}")
                print(f"Correct Predictions: {summary['correct_predictions']}")
                print("\nMarket Type Distribution:")
                for market_type, percentage in summary['market_type_dist'].items():
                    print(f"  {market_type}: {percentage:.1f}%")
                print("-" * 80)
            
            # Save all results
            combined_results.to_csv('market_detection_results_all_periods.csv')
            print("\nResults saved to market_detection_results_all_periods.csv")
            
            return combined_results
        
        return None

def main():
    # Create assessor
    assessor = DetectorAssessor()
    
    # Define multiple time periods to test
    periods = {
        'Recent Bull Market': (
            datetime(2024, 1, 1),
            datetime(2024, 3, 31)
        ),
        'Previous Bear Market': (
            datetime(2023, 7, 1),
            datetime(2023, 9, 30)
        ),
        'Sideways Period': (
            datetime(2023, 4, 1),
            datetime(2023, 6, 30)
        ),
        'High Volatility': (
            datetime(2023, 1, 1),
            datetime(2023, 3, 31)
        ),
        'Low Volatility': (
            datetime(2023, 10, 1),
            datetime(2023, 12, 31)
        )
    }
    
    try:
        # Assess all periods
        results = assessor.assess_multiple_periods(
            periods=periods,
            symbol='BTC/USDT',
            timeframe='1h',
            lookahead=10
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
