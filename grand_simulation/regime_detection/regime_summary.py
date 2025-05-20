import pandas as pd
import numpy as np
from datetime import datetime

class RegimeSummary:
    def __init__(self, results):
        """
        Initialize RegimeSummary with simulation results
        
        Parameters:
        results (dict): Dictionary of simulation results from RegimeManager
        """
        if results is None:
            raise ValueError("Cannot create RegimeSummary with None results")
        if not isinstance(results, dict):
            raise ValueError("Results must be a dictionary")
        if not results:
            raise ValueError("Results dictionary is empty")
            
        self.results = results
    
    def print_simulation_results(self, name=None):
        """
        Print detailed results for one or all simulations
        
        Parameters:
        name (str, optional): Name of specific simulation to print. If None, prints all simulations.
        """
        if not self.results:
            print("No simulation results to display")
            return
            
        if name:
            if name not in self.results:
                raise ValueError(f"Simulation '{name}' not found in results")
            self._print_single_simulation(name, self.results[name])
        else:
            for name, result in self.results.items():
                self._print_single_simulation(name, result)
    
    def _print_single_simulation(self, name, result):
        """Print detailed results for a single simulation"""
        data = result['data']
        validation_results = result['validation']
        config = result['config']
        
        print(f"\n{'='*80}")
        print(f"Regime Detection Results for {name}")
        print(f"Period: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
        print(f"Timeframe: {config['timeframe']}")
        print(f"Lookahead: {config['lookahead']} periods")
        print(f"{'='*80}")
        
        # Print overall statistics
        total_periods = len(data)
        warmup_periods = max(
            config['ema_slow_window'],
            config['bb_window'],
            config['rsi_window'],
            14  # ADX period
        )
        valid_periods = total_periods - warmup_periods - config['lookahead']
        
        print(f"\nOverall Statistics:")
        print(f"Total periods: {total_periods}")
        print(f"Warmup periods: {warmup_periods}")
        print(f"Valid periods for prediction: {valid_periods}")
        
        # Print regime distribution
        print(f"\nRegime Distribution:")
        regime_counts = data['predicted_regime'].value_counts()
        for regime, count in regime_counts.items():
            percentage = round(count / total_periods * 100, 2)
            print(f"{regime}: {count} periods ({percentage}%)")
        
        # Print validation metrics
        print(f"\nValidation Metrics:")
        print(validation_results.to_string(index=False))
        
        # Calculate and print average metrics
        avg_accuracy = validation_results['accuracy'].mean()
        avg_precision = validation_results['precision'].mean()
        avg_recall = validation_results['recall'].mean()
        print(f"\nAverage Metrics:")
        print(f"Average Accuracy: {avg_accuracy:.2%}")
        print(f"Average Precision: {avg_precision:.2%}")
        print(f"Average Recall: {avg_recall:.2%}")
        
        # Print most accurate regime
        best_regime = validation_results.loc[validation_results['accuracy'].idxmax()]
        print(f"\nBest Performing Regime:")
        print(f"Regime: {best_regime['regime']}")
        print(f"Accuracy: {best_regime['accuracy']:.2%}")
        print(f"Total Predictions: {best_regime['total_predictions']}")
        print(f"Correct Predictions: {best_regime['correct_predictions']}")
        
        print(f"\n{'-'*80}\n")
    
    def get_summary_dataframe(self):
        """
        Create a summary DataFrame of all simulations
        
        Returns:
        pd.DataFrame: Summary statistics for all simulations
        """
        summary_data = []
        
        for name, result in self.results.items():
            data = result['data']
            validation = result['validation']
            config = result['config']
            
            # Calculate basic statistics
            total_periods = len(data)
            warmup_periods = max(
                config['ema_slow_window'],
                config['bb_window'],
                config['rsi_window'],
                14
            )
            valid_periods = total_periods - warmup_periods - config['lookahead']
            
            # Get regime distribution
            regime_dist = data['predicted_regime'].value_counts(normalize=True) * 100
            
            # Get validation metrics
            avg_metrics = {
                'accuracy': validation['accuracy'].mean(),
                'precision': validation['precision'].mean(),
                'recall': validation['recall'].mean()
            }
            
            # Get best regime
            best_regime = validation.loc[validation['accuracy'].idxmax()]
            
            # Combine all data
            summary_row = {
                'simulation': name,
                # 'period': f"{config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}",
                # 'timeframe': config['timeframe'],
                'lookahead': config['lookahead'],
                'total_periods': total_periods,
                'valid_periods': valid_periods,
                'avg_accuracy': avg_metrics['accuracy'],
                'avg_precision': avg_metrics['precision'],
                'avg_recall': avg_metrics['recall'],
                'best_regime': best_regime['regime'],
                'best_accuracy': best_regime['accuracy'],
                'best_predictions': best_regime['total_predictions']
            }
            
            # Add regime distribution percentages
            # for regime in ['uptrend', 'downtrend', 'high_volatility', 'low_volatility']:
            #     summary_row[f'{regime}_pct'] = regime_dist.get(regime, 0)
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def print_summary_table(self):
        """Print a formatted summary table of all simulations"""
        df = self.get_summary_dataframe()
        
        # Format the DataFrame for display
        display_df = df.copy()
        display_df['avg_accuracy'] = display_df['avg_accuracy'].map('{:.2%}'.format)
        display_df['avg_precision'] = display_df['avg_precision'].map('{:.2%}'.format)
        display_df['avg_recall'] = display_df['avg_recall'].map('{:.2%}'.format)
        display_df['best_accuracy'] = display_df['best_accuracy'].map('{:.2%}'.format)
        
        # for col in ['uptrend_pct', 'downtrend_pct', 'high_volatility_pct', 'low_volatility_pct']:
        #     display_df[col] = display_df[col].map('{:.1f}%'.format)
        
        print("\nSimulation Summary Table:")
        print("=" * 120)
        print(display_df.to_string(index=False))
        print("=" * 120) 