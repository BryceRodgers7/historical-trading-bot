import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
import argparse
import os

class HistoricalDataFetcher:
    def __init__(self):
        """Initialize the historical data fetcher with Binance US exchange"""
        self.exchange = ccxt.binanceus()
    
    def fetch_historical_data(self, symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
        """
        Fetch historical data from Binance US exchange
        
        Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Candlestick timeframe (e.g., '1h', '4h', '1d')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
        Returns:
        pd.DataFrame: DataFrame containing OHLCV data with timezone-aware datetime index
        
        Raises:
        ValueError: If dates are invalid or if no data is fetched
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
            
        # Validate dates
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("start_date and end_date must be datetime objects")
            
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
            
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Convert dates to timestamps (milliseconds)
        try:
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
        except (OverflowError, OSError) as e:
            raise ValueError(f"Invalid date range: {e}")
        
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
        
        # Convert index to timezone-aware datetime
        data.index = data.index.tz_localize('UTC')
        
        # Filter data to ensure we only have data within our specified window
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        data = data[mask]
        
        return data 

    def save_to_csv(self, data: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV file
        
        Parameters:
        data (pd.DataFrame): Data to save
        filename (str): Name of the CSV file
        
        Returns:
        str: Path to the saved file
        """
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Construct full path
        filepath = os.path.join('data', filename)
        
        # Save to CSV
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return filepath

def main():
    """
    Main function to fetch historical data and save to CSV
    Usage examples:
        python historical_data.py --symbol BTC/USDT --timeframe 1h --days 90
        python historical_data.py --symbol ETH/USDT --timeframe 4h --start 2023-01-01 --end 2023-12-31
    """
    parser = argparse.ArgumentParser(description='Fetch historical cryptocurrency data from Binance US')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                      help='Trading pair symbol (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='4h',
                      help='Candlestick timeframe (e.g., 1h, 4h, 1d)')
    
    # Optional arguments
    parser.add_argument('--days', type=int, default=365,
                      help='Number of days of historical data to fetch (default: 365)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                      help='Start date (YYYY-MM-DD). If not provided, uses --days from now')
    parser.add_argument('--end', type=str, default='2024-01-01',
                      help='End date (YYYY-MM-DD). If not provided, uses current time')
    parser.add_argument('--output', type=str, default='historical_data',
                      help='Output filename (without .csv extension). If not provided, uses symbol_timeframe.csv')
    
    args = parser.parse_args()
    
    # Parse dates
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)
        
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Generate output filename if not provided
    if not args.output:
        args.output = f"{args.symbol.replace('/', '_')}_{args.timeframe}"
    
    try:
        # Initialize fetcher and get data
        fetcher = HistoricalDataFetcher()
        print(f"Fetching {args.symbol} data from {start_date.date()} to {end_date.date()} ({args.timeframe} timeframe)")
        
        data = fetcher.fetch_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save to CSV
        fetcher.save_to_csv(data, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main()) 