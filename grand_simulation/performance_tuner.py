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
            'ema_fast_window': range(5, 21, 2),  # 5 to 20, step 2
            'ema_slow_window': range(15, 51, 5),  # 15 to 50, step 5
            'bb_window': range(10, 31, 5),  # 10 to 30, step 5
            'bb_std': [1.5, 2.0, 2.5, 3.0]  # Common BB standard deviations
        }
        
        # Store optimization results
        self.optimization_results = None
        
    def _create_simulation_name(self, params):
        """Create a unique name for the simulation based on parameters"""
        return f"Opt_{self.timeframe}_EMA{params['ema_fast_window']}_{params['ema_slow_window']}_BB{params['bb_window']}_{params['bb_std']}"
    
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
            self.param_ranges['bb_std']
        ))
        
        total_combinations = len(param_combinations)
        print(f"Running optimization with {total_combinations} parameter combinations...")
        
        for i, (ema_fast, ema_slow, bb_window, bb_std) in enumerate(param_combinations, 1):
            if ema_fast >= ema_slow:  # Skip invalid combinations
                continue
                
            print(f"Testing combination {i}/{total_combinations}: "
                  f"EMA({ema_fast},{ema_slow}) BB({bb_window},{bb_std})")
            
            # Add simulation with current parameters
            manager.add_simulation(
                name=self._create_simulation_name({
                    'ema_fast_window': ema_fast,
                    'ema_slow_window': ema_slow,
                    'bb_window': bb_window,
                    'bb_std': bb_std
                }),
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_balance=self.initial_balance,
                ema_fast_window=ema_fast,
                ema_slow_window=ema_slow,
                bb_window=bb_window,
                bb_std=bb_std
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
        
        # Sort by outperformance
        self.optimization_results = summary.sort_values('outperformance', ascending=False)
        
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
            'EMA Fast': self.optimization_results['ema_fast'].corr(self.optimization_results['outperformance']),
            'EMA Slow': self.optimization_results['ema_slow'].corr(self.optimization_results['outperformance']),
            'BB Window': self.optimization_results['bb_window'].corr(self.optimization_results['outperformance']),
            'BB Std': self.optimization_results['bb_std'].corr(self.optimization_results['outperformance'])
        }
        
        # Plot correlations
        plt.figure(figsize=(10, 6))
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
        plot_data = self.optimization_results[['ema_fast', 'ema_slow', 'bb_window', 'bb_std', 'outperformance']]
        plot_data.columns = ['EMA Fast', 'EMA Slow', 'BB Window', 'BB Std', 'Outperformance']
        
        sns.pairplot(plot_data, diag_kind='kde')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create tuner for bull market
    bull_tuner = PerformanceTuner(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31)
    )

    # Run optimization
    bull_results = bull_tuner.run_optimization(market_type='Bull Market')

    # Get best parameters
    print("\nBest parameter combinations for bull market:")
    print(bull_tuner.get_best_parameters(top_n=5))

    # Create tuner for bear market
    bear_tuner = PerformanceTuner(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime(2023, 4, 1),
        end_date=datetime(2023, 6, 30)
    )

    # Run optimization
    bear_results = bear_tuner.run_optimization(market_type='Bear Market')

    # Get best parameters
    print("\nBest parameter combinations for bear market:")
    print(bear_tuner.get_best_parameters(top_n=5))
    
    # Plot results
    bull_tuner.plot_parameter_importance()
    bull_tuner.plot_parameter_relationships()
    bear_tuner.plot_parameter_importance()
    bear_tuner.plot_parameter_relationships() 