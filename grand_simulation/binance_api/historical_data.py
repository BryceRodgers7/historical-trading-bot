import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone

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