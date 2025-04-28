# Historical Crypto Trading Bot

This project implements a Simple Moving Average (SMA) crossover strategy for cryptocurrency trading using historical data. The bot simulates trading on historical data and provides performance metrics and visualizations.

## Features

- Fetches historical cryptocurrency data from Binance
- Implements SMA crossover strategy (20-period and 50-period moving averages)
- Simulates trades and tracks performance
- Generates performance metrics (total return, win rate, number of trades)
- Visualizes price action, trading signals, and portfolio value

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the trading bot:
```bash
python trading_bot.py
```

The bot will:
1. Fetch 30 days of historical data for BTC/USDT
2. Calculate SMA crossover signals
3. Simulate trades based on the signals
4. Display performance metrics
5. Show visualization of the trading results

## Customization

You can modify the following parameters in the `main()` function:
- `symbol`: Trading pair (default: 'BTC/USDT')
- `timeframe`: Candlestick timeframe (default: '1h')
- `short_window`: Short SMA period (default: 20)
- `long_window`: Long SMA period (default: 50)
- `days`: Number of days of historical data to fetch (default: 30)

## Disclaimer

This is a simulation tool for educational purposes only. Past performance does not guarantee future results. Always do your own research before making any investment decisions. 