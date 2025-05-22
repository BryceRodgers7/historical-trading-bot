from datetime import datetime
import time
import traceback
from simulation_manager import SimulationManager
from performance_tuner import PerformanceTuner
import matplotlib.pyplot as plt

def run_market_scenarios():
    # Create simulation manager
    manager = SimulationManager()
    
    # Define market periods with multiple examples of each regime
    market_periods = {
        # Bull Markets
        # 'Bull Market Q4 2020': (datetime(2020, 10, 1), datetime(2021, 1, 1)),  # Post-COVID recovery
        # 'Bull Market Q1 2023': (datetime(2023, 1, 1), datetime(2023, 4, 1)),  # Strong uptrend
        # 'Bull Market Q4 2023': (datetime(2023, 11, 1), datetime(2024, 2, 1)),  # Previous cycle peak
        
        # # Bear Markets
        # 'Bear Market Q2 2023': (datetime(2023, 4, 1), datetime(2023, 7, 1)),  # Recent correction
        # 'Bear Market Q1 2022': (datetime(2022, 1, 1), datetime(2022, 4, 1)),  
        
        # # High Volatility Periods
        # 'High Vol Q1 2020': (datetime(2020, 1, 1), datetime(2020, 4, 1)),  # COVID crash
        # 'High Vol Q3 2021': (datetime(2021, 5, 1), datetime(2021, 8, 1)),
        # 'High Vol Q3 2023': (datetime(2023, 7, 1), datetime(2023, 10, 1)),  # Recent volatility
        
        
        # # Low Volatility Periods
        # 'Low Vol Q4 2023': (datetime(2023, 10, 1), datetime(2024, 1, 1)),  # Recent consolidation
        # 'Low Vol Q4 2020': (datetime(2020, 9, 1), datetime(2020, 12, 1)),
        # 'Low Vol Q4 2023': (datetime(2023, 8, 1), datetime(2023, 11, 1)),  
        
        # # Sideways/Choppy Markets
        # 'Sideways Q2 2021': (datetime(2021, 4, 1), datetime(2021, 7, 1)),  # Post-April 2021 peak
        # 'Sideways Q1 2020': (datetime(2020, 1, 1), datetime(2020, 4, 1)),   # Pre-COVID
        # 'Sideways Q3 2020': (datetime(2020, 7, 1), datetime(2020, 10, 1)),  # Pre-breakout

        # # Support/Resistance Markets
        'Support/Resistance Q1 2020': (datetime(2024, 1, 1), datetime(2025, 1, 1)),  # Pre-COVID
        'Support/Resistance Q1 2023': (datetime(2023, 3, 1), datetime(2024, 1, 1)),  # Recent consolidation
    }
    
    # Define timeframes and EMA settings to test
    timeframes = ['4h', '1h', '15m']
    ema_settings = [
        {'name': '', 'fast': 9, 'slow': 21},  # Default settings
        {'name': 'ema20', 'fast': 20, 'slow': 50}  # Alternative settings
    ]
    
    # Define support and resistance levels as simple lists
    support_levels = [4000, 10000, 12000, 30000, 40000]
    resistance_levels = [12000, 19500, 42000, 52000, 64000, 69000]
    
    # Add simulations for each combination
    for market_type, (start_date, end_date) in market_periods.items():
        print(f"\nProcessing {market_type} from {start_date.date()} to {end_date.date()}")
        
        for timeframe in timeframes:
            for ema_setting in ema_settings:
                name = f"{market_type} {timeframe}"
                if ema_setting['name']:
                    name += f" {ema_setting['name']}"
                
                try:
                    manager.add_simulation(
                        name=name,
                        symbol='BTC/USDT',
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        initial_balance=10000,
                        ema_fast_window=ema_setting['fast'],
                        ema_slow_window=ema_setting['slow'],
                        support_levels=support_levels,
                        resistance_levels=resistance_levels
                    )
                    print(f"Added simulation: {name}")
                except Exception as e:
                    print(f"Error adding simulation {name}: {str(e)}")
                    continue
    
    try:
        # Run all simulations
        print("\nRunning simulations...")
        manager.run_all_simulations()
        
        # Get and display results
        # print("\nBasic Summary:")
        # print(manager.get_summary('basic'))
        
        print("\nSorted Summary (sorted by outperformance):")
        returns_summary = manager.get_summary('returns')
        print(returns_summary)
        
        # Group results by market type
        # print("\nResults by Market Type:")
        # print("=" * 80)
        # for market_type in ['Bull Market', 'Bear Market', 'High Vol', 'Low Vol', 'Sideways']:
        #     print(f"\n{market_type} Periods:")
        #     print("-" * 40)
        #     # Filter results for this market type
        #     market_results = returns_summary[returns_summary['market_type'].str.startswith(market_type)]
        #     if not market_results.empty:
        #         print(market_results)
        #     else:
        #         print(f"No results found for {market_type}")
        
    except Exception as e:
        print(f"Error running simulations: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()
        return None

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
    run_market_scenarios()
    
    # Run optimization for all market types
    # print("\nRunning parameter optimization for all market types...")
    # market_results = run_parameter_optimization()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Simulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 