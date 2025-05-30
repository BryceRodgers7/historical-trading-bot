import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set pandas display options to show all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class SimulationSummary:
    def __init__(self, results):
        """
        Initialize with simulation results
        
        Parameters:
        results (dict): Dictionary of simulation results from SimulationManager
        """
        self.results = results
        
    def _calculate_buy_hold_return(self, data):
        """
        Calculate buy & hold return for a given period
        
        Parameters:
        data (pd.DataFrame): Price data with 'close' column
        
        Returns:
        float: Buy & hold return percentage
        """
        if len(data) < 2:
            return 0.0
        
        first_price = data['close'].iloc[0]
        last_price = data['close'].iloc[-1]
        return ((last_price - first_price) / first_price) * 100
        
    def get_basic_summary(self):
        """
        Get basic summary of all simulation results
        
        Returns:
        pd.DataFrame: Summary DataFrame with basic metrics
        """
        summary = {}
        for name, result in self.results.items():
            # Calculate buy & hold return
            buy_hold_return = self._calculate_buy_hold_return(result['data'])
            
            # Count regime occurrences
            regime_counts = result['data']['regime'].value_counts()
            
            # Get actual warmup periods from performance metrics
            warmup_periods = result['performance'].get('warmup_periods', 0) if result['performance'] else 0
            
            # Initialize summary with default values
            summary[name] = {
                'ttl_trades': 0,
                'win_rate': 0.0,
                'poss_wins': 0,
                'total_return': 0.0,
                'hodl_return': buy_hold_return,
                'outperform': -buy_hold_return,  # Negative of buy & hold return when no trades
                'warmup_pds': warmup_periods,  # Use actual warmup periods from simulation
                'trend_pds': regime_counts.get('trend_following', 0),
                'merev_pds': regime_counts.get('mean_reversion', 0),
                'break_pds': regime_counts.get('breakout', 0),
                'scalp_pds': regime_counts.get('scalping', 0),
                'noth_pds': regime_counts.get('do_nothing', 0),
            }
            
            # Update with actual performance if there were trades
            if result['performance']:
                trades_df = result['trades']
                exit_causes = trades_df['exit_reason'].value_counts()
                
                summary[name].update({
                    'ttl_trades': result['performance']['total_trades'],
                    'win_rate': result['performance']['win_rate'],
                    'poss_wins': result['performance']['possible_profitable'],
                    'total_return': result['performance']['total_return'],
                    'outperform': result['performance']['total_return'] - buy_hold_return,
                })
                
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
        pd.DataFrame: Summary DataFrame sorted by total returns with all rows displayed
        """
        summary = {}
        for name, result in self.results.items():
            # Calculate buy & hold return
            buy_hold_return = self._calculate_buy_hold_return(result['data'])
            
            # Extract timeframe from name (assuming format "Market Type Timeframe")
            timeframe = name.split()[-1] if len(name.split()) > 2 else 'unknown'
            market_type = ' '.join(name.split()[:-1]) if len(name.split()) > 2 else name
            
            # Count regime occurrences
            regime_counts = result['data']['regime'].value_counts()

            # Get actual warmup periods from performance metrics
            warmup_periods = result['performance'].get('warmup_periods', 0) if result['performance'] else 0

            active_periods = len(result['data']) - regime_counts.get('do_nothing', 0) - warmup_periods           
            
            # Initialize summary with default values
            summary[name] = {
                'ttl_tds': 0,
                'win_rate': 0.0,
                'poss_wins': 0,
                'trade_pds': result['performance'].get('trading_periods', 0) if result['performance'] else 0,
                'total_return': 0.0,
                'hodl_return': buy_hold_return,
                'outperform': -buy_hold_return,  # Negative of buy & hold return when no trades
                'warmup_pds': warmup_periods,  # Use actual warmup periods from simulation
                'trend_pds': regime_counts.get('trend_following', 0),
                'merev_pds': regime_counts.get('mean_reversion', 0),
                'break_pds': regime_counts.get('breakout', 0),
                'scalp_pds': regime_counts.get('scalping', 0),
                'noth_pds': regime_counts.get('do_nothing', 0),
                'trend_pct': round(regime_counts.get('trend_following', 0) / active_periods * 100, 2) if active_periods > 0 else 0.0,
                'merev_pct': round(regime_counts.get('mean_reversion', 0) / active_periods * 100, 2) if active_periods > 0 else 0.0,
                'break_pct': round(regime_counts.get('breakout', 0) / active_periods * 100, 2) if active_periods > 0 else 0.0,
                'scalp_pct': round(regime_counts.get('scalping', 0) / active_periods * 100, 2) if active_periods > 0 else 0.0
            }
            
            # Update with actual performance if there were trades
            if result['performance']:
                trades_df = result['trades']
                exit_causes = trades_df['exit_reason'].value_counts()
                
                summary[name].update({
                    'ttl_tds': result['performance']['total_trades'],
                    'win_rate': result['performance']['win_rate'],
                    'poss_wins': result['performance']['possible_profitable'],
                    'trade_pds': result['performance']['trading_periods'],
                    'total_return': result['performance']['total_return'],
                    'outperform': result['performance']['total_return'] - buy_hold_return,
                })
        
        # Convert to DataFrame and sort by total return
        df = pd.DataFrame(summary).T
        
        # Format numeric columns to 2 decimal places
        numeric_cols = ['win_rate', 'total_return', 'hodl_return', 'outperform', 
                       'trend_pct', 'merev_pct', 'break_pct', 'scalp_pct']
        df[numeric_cols] = df[numeric_cols].round(2)
        
        return df.sort_values('outperform', ascending=False)
    
    def get_timeframe_comparison(self):
        """
        Get comparison of performance across different timeframes
        
        Returns:
        pd.DataFrame: Summary DataFrame grouped by market type and timeframe
        """
        summary = {}
        for name, result in self.results.items():
            if result['performance']:  # If simulation had trades
                # Calculate buy & hold return
                buy_hold_return = self._calculate_buy_hold_return(result['data'])
                
                # Extract timeframe from name (assuming format "Market Type Timeframe")
                timeframe = name.split()[-1] if len(name.split()) > 2 else 'unknown'
                market_type = ' '.join(name.split()[:-1]) if len(name.split()) > 2 else name
                
                # Count regime occurrences
                regime_counts = result['data']['regime'].value_counts()
                
                summary[name] = {
                    'market_type': market_type,
                    'timeframe': timeframe,
                    'total_return': result['performance']['total_return'],
                    'hodl_return': buy_hold_return,
                    'outperformance': result['performance']['total_return'] - buy_hold_return,
                    'win_rate': result['performance']['win_rate'],
                    'sharpe_ratio': result['performance']['sharpe_ratio']
                }
        
        # Convert to DataFrame and pivot
        df = pd.DataFrame(summary).T
        return df.pivot(index='market_type', columns='timeframe', 
                       values=['total_return', 'buy_hold_return', 'outperformance', 'win_rate', 'sharpe_ratio'])
    
    def plot_regime_distribution(self, figsize=(12, 6), title="Market Regime Distribution Across Simulations"):
        """
        Create a stacked bar chart showing the distribution of market regimes across simulations
        
        Parameters:
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        
        Returns:
        matplotlib.figure.Figure: The figure object
        """
        # Collect regime data for each simulation
        regime_data = []
        for name, result in self.results.items():
            if result['performance']:  # If simulation had trades
                # Get regime counts
                regime_counts = result['data']['regime'].value_counts()
                total_periods = len(result['data'])
                
                # Calculate percentages
                regime_pcts = (regime_counts / total_periods * 100).round(2)
                
                # Create row for this simulation
                row = {
                    'simulation': name,
                    'warm_up': regime_pcts.get('warm_up', 0),
                    'trend_following': regime_pcts.get('trend_following', 0),
                    'mean_reversion': regime_pcts.get('mean_reversion', 0),
                    'breakout': regime_pcts.get('breakout', 0),
                    'scalping': regime_pcts.get('scalping', 0),
                    'do_nothing': regime_pcts.get('do_nothing', 0)
                }
                regime_data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(regime_data)
        df.set_index('simulation', inplace=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked bar chart
        df.plot(kind='bar', stacked=True, ax=ax)
        
        # Customize the plot
        ax.set_title(title, pad=20)
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Percentage of Time')
        ax.legend(title='Market Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels on the bars
        for c in ax.containers:
            # Add labels only if the segment is large enough
            labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig

    def get_trades_summary(self, simulation_name=None):
        """
        Get a detailed table of trades for one or all simulations
        
        Parameters:
        simulation_name (str, optional): Name of specific simulation to get trades for. 
                                       If None, returns trades for all simulations.
        
        Returns:
        pd.DataFrame: DataFrame containing all trades with columns:
            - simulation: Name of the simulation
            - entry_time: Time of trade entry
            - exit_time: Time of trade exit
            - entry_price: Price at entry
            - exit_price: Price at exit
            - position: Long or Short
            - pnl: Profit/Loss in percentage
            - regime: Market regime at entry
            - exit_reason: Reason for trade exit
            - holding_periods: Number of periods held
        """
        all_trades = []
        
        # Determine which simulations to process
        simulations = [simulation_name] if simulation_name else self.results.keys()
        
        for name in simulations:
            if name not in self.results:
                print(f"Warning: Simulation '{name}' not found")
                continue
                
            result = self.results[name]
            if not result['performance'] or result['trades'].empty:
                continue
                
            trades_df = result['trades'].copy()
            trades_df['simulation'] = name
            
            # Reorder columns for better readability
            columns = [
                'simulation', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
                'position', 'pnl', 'regime', 'exit_reason', 'holding_periods'
            ]
            
            # Only include columns that exist in the trades DataFrame
            available_columns = [col for col in columns if col in trades_df.columns]
            trades_df = trades_df[available_columns]
            
            all_trades.append(trades_df)
        
        if not all_trades:
            return pd.DataFrame()  # Return empty DataFrame if no trades found
            
        # Combine all trades into one DataFrame
        combined_trades = pd.concat(all_trades, ignore_index=True)
        
        # Sort by entry time
        if 'entry_time' in combined_trades.columns:
            combined_trades = combined_trades.sort_values('entry_time')
            
        return combined_trades