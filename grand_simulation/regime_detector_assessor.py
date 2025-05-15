import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from grand_simulation.regime_detector import MarketRegime, MarketRegimeDetector

class RegimeDetectorAssessor:
    def __init__(self, detector=None):
        self.detector = detector or MarketRegimeDetector()
        
    def assess_regime_accuracy(self, data, true_regimes=None):
        """
        Assess the accuracy of regime detection
        
        Parameters:
        data (pd.DataFrame): OHLCV data
        true_regimes (pd.Series, optional): True regime labels if available
        
        Returns:
        dict: Dictionary containing various accuracy metrics
        """
        # Detect regimes
        detected_regimes = self.detector.detect_regime(data)
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(detected_regimes)
        
        # Calculate regime transitions
        transition_stats = self._calculate_regime_transitions(detected_regimes)
        
        # Calculate regime performance correlation
        performance_corr = self._calculate_regime_performance_correlation(data, detected_regimes)
        
        if true_regimes is not None:
            # Calculate classification metrics
            classification_metrics = self._calculate_classification_metrics(detected_regimes, true_regimes)
            regime_stats.update(classification_metrics)
        
        return {
            'regime_statistics': regime_stats,
            'transition_statistics': transition_stats,
            'performance_correlation': performance_corr
        }
    
    def _calculate_regime_statistics(self, regimes):
        """Calculate basic statistics about detected regimes"""
        regime_counts = regimes.value_counts()
        regime_percentages = (regime_counts / len(regimes)) * 100
        
        # Calculate regime persistence
        regime_changes = regimes.diff().ne(0)
        regime_persistence = []
        current_regime = regimes.iloc[0]
        current_length = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_length += 1
            else:
                regime_persistence.append((current_regime, current_length))
                current_regime = regimes.iloc[i]
                current_length = 1
        
        regime_persistence.append((current_regime, current_length))
        persistence_stats = pd.DataFrame(regime_persistence, columns=['regime', 'duration'])
        avg_persistence = persistence_stats.groupby('regime')['duration'].agg(['mean', 'median', 'max'])
        
        return {
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'avg_persistence': avg_persistence
        }
    
    def _calculate_regime_transitions(self, regimes):
        """Calculate statistics about regime transitions"""
        transitions = pd.DataFrame({
            'from_regime': regimes[:-1],
            'to_regime': regimes[1:]
        })
        
        # Calculate transition matrix
        transition_matrix = pd.crosstab(
            transitions['from_regime'],
            transitions['to_regime'],
            normalize='index'
        )
        
        # Calculate transition probabilities
        transition_counts = transitions.groupby(['from_regime', 'to_regime']).size()
        transition_probs = transition_counts / transition_counts.groupby('from_regime').sum()
        
        return {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probs
        }
    
    def _calculate_regime_performance_correlation(self, data, regimes):
        """Calculate correlation between regimes and market performance"""
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate regime-specific returns
        regime_returns = pd.DataFrame({
            'regime': regimes,
            'returns': returns
        })
        
        # Calculate average returns by regime
        avg_returns = regime_returns.groupby('regime')['returns'].agg(['mean', 'std'])
        
        # Calculate Sharpe ratio by regime (assuming risk-free rate of 0)
        sharpe_ratios = avg_returns['mean'] / avg_returns['std']
        
        return {
            'average_returns': avg_returns,
            'sharpe_ratios': sharpe_ratios
        }
    
    def _calculate_classification_metrics(self, detected_regimes, true_regimes):
        """Calculate classification metrics if true regimes are available"""
        # Create confusion matrix
        cm = confusion_matrix(true_regimes, detected_regimes)
        cm_df = pd.DataFrame(
            cm,
            index=[f'True_{r}' for r in self.detector.regime_detector.MarketRegime],
            columns=[f'Pred_{r}' for r in self.detector.regime_detector.MarketRegime]
        )
        
        # Calculate classification report
        report = classification_report(true_regimes, detected_regimes, output_dict=True)
        
        return {
            'confusion_matrix': cm_df,
            'classification_report': report
        }
    
    def plot_regime_distribution(self, regimes):
        """Plot the distribution of detected regimes"""
        plt.figure(figsize=(10, 6))
        regimes.value_counts().plot(kind='bar')
        plt.title('Distribution of Detected Market Regimes')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_regime_transitions(self, transition_matrix):
        """Plot the regime transition matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.2%')
        plt.title('Regime Transition Probabilities')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        plt.tight_layout()
        plt.show()
    
    def plot_regime_performance(self, regime_returns):
        """Plot the performance metrics for each regime"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot average returns
        regime_returns['average_returns']['mean'].plot(kind='bar', ax=ax1)
        ax1.set_title('Average Returns by Regime')
        ax1.set_xlabel('Regime')
        ax1.set_ylabel('Average Return')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot Sharpe ratios
        regime_returns['sharpe_ratios'].plot(kind='bar', ax=ax2)
        ax2.set_title('Sharpe Ratio by Regime')
        ax2.set_xlabel('Regime')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show() 