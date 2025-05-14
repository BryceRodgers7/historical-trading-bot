from abc import ABC, abstractmethod
import pandas as pd

class TradingStrategy(ABC):
    """Base class for trading strategies"""
    @abstractmethod
    def calculate_signals(self, data):
        """Calculate trading signals for the given data"""
        pass

    @abstractmethod
    def simulate_trades(self, data, initial_balance):
        """Simulate trades based on the signals"""
        pass

    @abstractmethod
    def plot_results(self, data, positions):
        """Plot trading results"""
        pass 