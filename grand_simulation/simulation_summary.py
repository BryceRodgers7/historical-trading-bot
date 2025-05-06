import pandas as pd
import numpy as np

class SimulationSummary:
    def __init__(self, results):
        """
        Initialize with simulation results
        
        Parameters:
        results (dict): Dictionary of simulation results from SimulationManager
        """
        self.results = results
        
    def get_basic_summary(self):
        """
        Get basic summary of all simulation results
        
        Returns:
        pd.DataFrame: Summary DataFrame with basic metrics
        """
        summary = {}
        for name, result in self.results.items():
            if result['performance']:  # If simulation had trades
                summary[name] = {
                    'total_trades': result['performance']['total_trades'],
                    'win_rate': result['performance']['win_rate'],
                    'total_return': result['performance']['total_return'],
                    'max_drawdown': result['performance']['max_drawdown'],
                    'sharpe_ratio': result['performance']['sharpe_ratio']
                }
                
                # Add regime-specific performance
                if not result['regime_performance'].empty:
                    regime_stats = {}
                    for regime in result['regime_performance'].index:
                        regime_stats[regime] = {
                            'trades': result['regime_performance'].loc[regime, ('return', 'count')],
                            'avg_return': result['regime_performance'].loc[regime, ('return', 'mean')] * 100,
                            'win_rate': result['regime_performance'].loc[regime, ('win', 'mean')] * 100
                        }
                    summary[name]['regime_stats'] = regime_stats
        
        return pd.DataFrame(summary).T
    
    def get_returns_summary(self):
        """
        Get summary sorted by total returns
        
        Returns:
        pd.DataFrame: Summary DataFrame sorted by total returns
        """
        summary = {}
        for name, result in self.results.items():
            if result['performance']:  # If simulation had trades
                # Extract timeframe from name (assuming format "Market Type Timeframe")
                timeframe = name.split()[-1] if len(name.split()) > 2 else 'unknown'
                market_type = ' '.join(name.split()[:-1]) if len(name.split()) > 2 else name
                
                summary[name] = {
                    'market_type': market_type,
                    'timeframe': timeframe,
                    'total_trades': result['performance']['total_trades'],
                    'win_rate': result['performance']['win_rate'],
                    'total_return': result['performance']['total_return'],
                    'max_drawdown': result['performance']['max_drawdown'],
                    'sharpe_ratio': result['performance']['sharpe_ratio'],
                    'trading_periods': result['performance']['trading_periods']
                }
        
        # Convert to DataFrame and sort by total return
        df = pd.DataFrame(summary).T
        return df.sort_values('total_return', ascending=False)
    
    def get_timeframe_comparison(self):
        """
        Get comparison of performance across different timeframes
        
        Returns:
        pd.DataFrame: Summary DataFrame grouped by market type and timeframe
        """
        summary = {}
        for name, result in self.results.items():
            if result['performance']:  # If simulation had trades
                # Extract timeframe from name (assuming format "Market Type Timeframe")
                timeframe = name.split()[-1] if len(name.split()) > 2 else 'unknown'
                market_type = ' '.join(name.split()[:-1]) if len(name.split()) > 2 else name
                
                summary[name] = {
                    'market_type': market_type,
                    'timeframe': timeframe,
                    'total_return': result['performance']['total_return'],
                    'win_rate': result['performance']['win_rate'],
                    'sharpe_ratio': result['performance']['sharpe_ratio']
                }
        
        # Convert to DataFrame and pivot
        df = pd.DataFrame(summary).T
        return df.pivot(index='market_type', columns='timeframe', values=['total_return', 'win_rate', 'sharpe_ratio']) 