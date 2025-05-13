from datetime import datetime
import time
from simulation_manager import SimulationManager
from performance_tuner import PerformanceTuner

def run_market_scenarios():

    # Create simulation manager
    manager = SimulationManager()
    
    # Define market periods
    market_periods = {
        'Bull Market': (datetime(2023, 1, 1), datetime(2023, 3, 31)),
        'Bear Market': (datetime(2023, 4, 1), datetime(2023, 6, 30)),
        'High Volatility': (datetime(2023, 7, 1), datetime(2023, 9, 30)),
        'Low Volatility': (datetime(2023, 10, 1), datetime(2023, 12, 31))
    }
    
    # Define timeframes and EMA settings to test
    timeframes = ['4h', '1h', '15m']
    ema_settings = [
        {'name': '', 'fast': 9, 'slow': 21},  # Default settings
        {'name': 'ema20', 'fast': 20, 'slow': 50}  # Alternative settings
    ]
    
    # Add simulations for each combination
    for market_type, (start_date, end_date) in market_periods.items():
        for timeframe in timeframes:
            for ema_setting in ema_settings:
                name = f"{market_type} {timeframe}"
                if ema_setting['name']:
                    name += f" {ema_setting['name']}"
                
                manager.add_simulation(
                    name=name,
                    symbol='BTC/USDT',
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=10000,
                    ema_fast_window=ema_setting['fast'],
                    ema_slow_window=ema_setting['slow']
                )
    
    # Run all simulations
    manager.run_all_simulations()
    
    # Get and display results
    print("\nBasic Summary:")
    print(manager.get_summary('basic'))
    
    print("\nSorted Summary (sorted by outperformance):")
    print(manager.get_summary('returns'))
    
    # Plot results
    # manager.plot_equity_curves()
    # manager.plot_regime_distribution()
    
    return manager

def run_parameter_optimization():
    """
    Run parameter optimization for all market periods.
    Returns a dictionary of tuners for each market type.
    """
    param_ranges = {
        'ema_fast_window': range(10, 20, 10),
        'ema_slow_window': range(20, 50, 30),
        'bb_window': range(10, 31, 21),  
        'bb_std': [1.5, 3.0],  
        'stop_loss_pct': [0.01, 0.025],
        'take_profit_pct': [0.05, 0.1] 
    }

    market_periods = {
        'Bull Market': (datetime(2023, 1, 1), datetime(2023, 3, 31)),
        'Bear Market': (datetime(2023, 4, 1), datetime(2023, 6, 30)),
        'High Volatility': (datetime(2023, 7, 1), datetime(2023, 9, 30)),
        'Low Volatility': (datetime(2023, 10, 1), datetime(2023, 12, 31))
    }
    
    # Dictionary to store results for each market type
    market_results = {}
    
    # Process each market type
    for market_type, (start_date, end_date) in market_periods.items():
        print(f"\n{'='*50}")
        print(f"Running optimization for {market_type}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*50}")
        
        # Create tuner for this market type
        tuner = PerformanceTuner(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        # Update parameter ranges
        tuner.param_ranges.update(param_ranges)
        
        # Run optimization
        results = tuner.run_optimization(market_type=market_type)
        
        # Get and display best parameters
        print(f"\nBest parameter combinations for {market_type}:")
        best_params = tuner.get_best_parameters(top_n=5)
        print(best_params)
        
        # Store results
        market_results[market_type] = {
            'tuner': tuner,
            'best_params': best_params,
            'results': results
        }
        
        # Plot results (optional)
        # tuner.plot_parameter_importance()
        # tuner.plot_parameter_relationships()
    
    # Print summary of best parameters across all market types
    print("\n" + "="*80)
    print("Summary of Best Parameters Across Market Types")
    print("="*80)
    for market_type, result in market_results.items():
        print(f"\n{market_type}:")
        print(f"Best Parameters:")
        print(result['best_params'].iloc[0])
        print(f"Outperformance: {result['best_params']['outperform'].iloc[0]:.2f}%")
        print(f"Win Rate: {result['best_params']['win_rate'].iloc[0]:.2f}%")
        print(f"Total Return: {result['best_params']['total_return'].iloc[0]:.2f}%")
    
    # Compare parameters across market types
    print("\n" + "="*80)
    print("Comparing Parameters Across Market Types")
    print("="*80)
    for market_type, result in market_results.items():
        best_params = result['best_params'].iloc[0]
        print(f"\n{market_type}:")
        print(f"EMA Settings: ({best_params['ema_fast']}, {best_params['ema_slow']})")
        print(f"BB Settings: Window={best_params['bb_window']}, Std={best_params['bb_std']}")
        print(f"SL/TP: {best_params['stop_loss']*100}%/{best_params['take_profit']*100}%")
        print(f"Outperformance: {best_params['outperform']:.2f}%")
        print("---")
    
    return market_results

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run market scenarios
    print("Running market scenarios...")
    manager = run_market_scenarios()
    
    # Run optimization for all market types
    # print("\nRunning parameter optimization for all market types...")
    # market_results = run_parameter_optimization()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Simulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 