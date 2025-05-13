import pandas as pd
import numpy as np
from itertools import product
from simulation_manager import SimulationManager
from datetime import datetime, timedelta

class PerformanceTuner:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', 
                 start_date=None, end_date=None, initial_balance=10000):
        """
        Initialize the performance tuner
        
        Parameters:
        symbol (str): Trading pair symbol
        timeframe (str): Trading timeframe
        start_date (datetime): Start date for optimization
        end_date (datetime): End date for optimization
        initial_balance (float): Initial balance for simulations
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date or (datetime.now() - timedelta(days=90))
        self.end_date = end_date or datetime.now()
        self.initial_balance = initial_balance
        
        # Define parameter ranges for optimization
        self.param_ranges = {
            'ema_fast_window': range(10, 20, 10),  
            'ema_slow_window': range(20, 50, 30), 
            'bb_window': range(10, 31, 21),  
            'bb_std': [2.0, 3.0],  
            'stop_loss_pct': [0.01, 0.03],  
            'take_profit_pct': [0.05, 0.1]   
        }
        
        # Store optimization results
        self.optimization_results = None
        
    def _create_simulation_name(self, params):
        """Create a unique name for the simulation based on parameters"""
        return (f"Opt_{self.timeframe}_EMA{params['ema_fast_window']}_{params['ema_slow_window']}_"
                f"BB{params['bb_window']}_{params['bb_std']}_"
                f"SL{int(params['stop_loss_pct']*1000)}_TP{int(params['take_profit_pct']*1000)}")
    
    def run_optimization(self, market_type='Bull Market'):
        """
        Run parameter optimization for the specified market type
        
        Parameters:
        market_type (str): Type of market to optimize for ('Bull Market', 'Bear Market', etc.)
        
        Returns:
        pd.DataFrame: Optimization results sorted by outperformance
        """
        manager = SimulationManager()
        results = []
        
        # Generate all parameter combinations
        param_combinations = list(product(
            self.param_ranges['ema_fast_window'],
            self.param_ranges['ema_slow_window'],
            self.param_ranges['bb_window'],
            self.param_ranges['bb_std'],
            self.param_ranges['stop_loss_pct'],
            self.param_ranges['take_profit_pct']
        ))
        
        total_combinations = len(param_combinations)
        print(f"Running optimization with {total_combinations} parameter combinations...")
        
        for i, (ema_fast, ema_slow, bb_window, bb_std, sl_pct, tp_pct) in enumerate(param_combinations, 1):
            if ema_fast >= ema_slow:  # Skip invalid combinations
                continue
                
            print(f"Testing combination {i}/{total_combinations}: "
                  f"EMA({ema_fast},{ema_slow}) BB({bb_window},{bb_std}) "
                  f"SL({sl_pct*100}%) TP({tp_pct*100}%)")
            
            # Add simulation with current parameters
            manager.add_simulation(
                name=self._create_simulation_name({
                    'ema_fast_window': ema_fast,
                    'ema_slow_window': ema_slow,
                    'bb_window': bb_window,
                    'bb_std': bb_std,
                    'stop_loss_pct': sl_pct,
                    'take_profit_pct': tp_pct
                }),
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_balance=self.initial_balance,
                ema_fast_window=ema_fast,
                ema_slow_window=ema_slow,
                bb_window=bb_window,
                bb_std=bb_std,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct
            )
        
        # Run all simulations
        manager.run_all_simulations()
        
        # Get results summary
        summary = manager.get_summary('returns')
        
        # Extract parameter values from simulation names
        summary['ema_fast'] = summary.index.str.extract(r'EMA(\d+)').astype(int)
        summary['ema_slow'] = summary.index.str.extract(r'EMA\d+_(\d+)').astype(int)
        summary['bb_window'] = summary.index.str.extract(r'BB(\d+)').astype(int)
        summary['bb_std'] = summary.index.str.extract(r'BB\d+_([\d.]+)').astype(float)
        summary['stop_loss'] = summary.index.str.extract(r'SL(\d+)').astype(int) / 1000
        summary['take_profit'] = summary.index.str.extract(r'TP(\d+)').astype(int) / 1000
        
        # Sort by outperformance
        self.optimization_results = summary.sort_values('outperform', ascending=False)
        
        return self.optimization_results
    
    def get_best_parameters(self, top_n=5):
        """
        Get the best parameter combinations
        
        Parameters:
        top_n (int): Number of top results to return
        
        Returns:
        pd.DataFrame: Top N parameter combinations with their performance
        """
        if self.optimization_results is None:
            raise ValueError("Run optimization first using run_optimization()")
            
        return self.optimization_results.head(top_n)
    
    def plot_parameter_importance(self):
        """
        Plot the importance of each parameter based on correlation with outperformance
        """
        if self.optimization_results is None:
            raise ValueError("Run optimization first using run_optimization()")
            
        import matplotlib.pyplot as plt
        
        # Calculate correlations with outperformance
        correlations = {
            'EMA Fast': self.optimization_results['ema_fast'].corr(self.optimization_results['outperform']),
            'EMA Slow': self.optimization_results['ema_slow'].corr(self.optimization_results['outperform']),
            'BB Window': self.optimization_results['bb_window'].corr(self.optimization_results['outperform']),
            'BB Std': self.optimization_results['bb_std'].corr(self.optimization_results['outperform']),
            'Stop Loss': self.optimization_results['stop_loss'].corr(self.optimization_results['outperform']),
            'Take Profit': self.optimization_results['take_profit'].corr(self.optimization_results['outperform'])
        }
        
        # Plot correlations
        plt.figure(figsize=(12, 6))
        plt.bar(correlations.keys(), correlations.values())
        plt.title('Parameter Importance (Correlation with Outperformance)')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.show()
        
    def plot_parameter_relationships(self):
        """
        Plot relationships between parameters and outperformance
        """
        if self.optimization_results is None:
            raise ValueError("Run optimization first using run_optimization()")
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create pairplot of parameters vs outperformance
        plot_data = self.optimization_results[['ema_fast', 'ema_slow', 'bb_window', 'bb_std', 
                                             'stop_loss', 'take_profit', 'outperform']]
        plot_data.columns = ['EMA Fast', 'EMA Slow', 'BB Window', 'BB Std', 
                           'Stop Loss', 'Take Profit', 'Outperformance']
        
        sns.pairplot(plot_data, diag_kind='kde')
        plt.show()