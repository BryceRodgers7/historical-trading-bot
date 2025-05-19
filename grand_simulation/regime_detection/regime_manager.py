from datetime import datetime
from grand_simulation.binance_api.historical_data import HistoricalDataFetcher
from grand_simulation.regime_detection.regime_simulation import RegimeSimulation


class RegimeManager:
    def __init__(self):
        self.simulations = {}
        self.results = {}
        self.data_fetcher = HistoricalDataFetcher()
        self.sim_data = {}
        self.sim_parms = {'symbol': '', 'timeframe': '', 'start_date': '', 'end_date': ''}

    def add_sim(self, name, symbol, timeframe, start_date, end_date, initial_balance=10000, 
                       ema_fast_window=9, ema_slow_window=21, bb_window=20, bb_std=2, rsi_window=14, 
                       lookback_window=100, lookahead=1, stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Add a new simulation scenario
        
        Parameters:
        name (str): Unique identifier for the simulation
        symbol (str): Symbol to simulate
        timeframe (str): Timeframe to simulate
        start_date (str or datetime): Start date to simulate (format: 'YYYY-MM-DD' or datetime object)
        end_date (str or datetime): End date to simulate (format: 'YYYY-MM-DD' or datetime object)
        initial_balance (float): Starting capital
        ema_fast_window (int): Fast EMA window
        ema_slow_window (int): Slow EMA window
        bb_window (int): Bollinger Bands window
        bb_std (float): Bollinger Bands standard deviation
        rsi_window (int): RSI window
        lookback_window (int): Window size for lookback calculations
        stop_loss_pct (float): Stop loss percentage (e.g., 0.02 for 2%)
        take_profit_pct (float): Take profit percentage (e.g., 0.04 for 4%)
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        self.simulations[name] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'ema_fast_window': ema_fast_window,
            'ema_slow_window': ema_slow_window,
            'bb_window': bb_window,
            'bb_std': bb_std,
            'lookback_window': lookback_window,
            'lookahead': lookahead,
            'rsi_window': rsi_window,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }


    def run_all_simulations(self):
        """Run all configured simulations and store results"""
        for name, config in self.simulations.items():
            
            # do not re-fetch data if the same symbol, timeframe, start_date, and end_date
            if (self.sim_parms['symbol'] != config['symbol'] 
                or self.sim_parms['timeframe'] != config['timeframe'] 
                or self.sim_parms['start_date'] != config['start_date'] 
                or self.sim_parms['end_date'] != config['end_date']):
                
                self.sim_data = self.data_fetcher.fetch_historical_data(
                    symbol=config['symbol'],
                    timeframe=config['timeframe'],
                    start_date=config['start_date'],
                    end_date=config['end_date']
                )

            self.sim_parms = config

            print(f"running simulation: {name}")
            sim = RegimeSimulation(name=name, 
                                 symbol=config['symbol'], 
                                 timeframe=config['timeframe'], 
                                 start_date=config['start_date'], 
                                 end_date=config['end_date'], 
                                 initial_balance=config['initial_balance'],
                                 ema_fast_window=config['ema_fast_window'],
                                 ema_slow_window=config['ema_slow_window'],
                                 bb_window=config['bb_window'],
                                 bb_std=config['bb_std'],
                                 rsi_window=config['rsi_window'],
                                 lookback_window=config['lookback_window'],
                                 lookahead=config['lookahead'],
                                 stop_loss_pct=config['stop_loss_pct'],
                                 take_profit_pct=config['take_profit_pct'])
            
            print(f"Running simulation: {name}")

            results = sim.run_regime_simulation(self.sim_data)
            self.results[name] = {
                'data': results[0],
                'performance': results[1]
            }