import numpy as np
import pandas as pd
from regime_detection.regime_detector import MarketRegime
from signal_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    BullRunStrategy,
    FibonacciRetracementStrategy
)

class RegimeStrategyFactory:
    @staticmethod
    # def get_strategy(regime, ema_short_window=9, ema_long_window=21):
    def get_strategy(regime):
        if regime != MarketRegime.DO_NOTHING.value:
            return TrendFollowingStrategy()
        #     # return FibonacciRetracementStrategy()
        #     # return BullRunStrategy()
        #     # return TrendFollowingStrategy(ema_short_window, ema_long_window)
        
        else:
            return None
        
        # if regime == MarketRegime.TREND_FOLLOWING.value:
        #     return TrendFollowingStrategy()
        # elif regime == MarketRegime.MEAN_REVERSION.value:
        #     return MeanReversionStrategy()
        # elif regime == MarketRegime.BREAKOUT.value:
        #     return BreakoutStrategy()
        # elif regime == MarketRegime.SCALPING.value:
        #     return ScalpingStrategy()
        # elif regime == MarketRegime.DO_NOTHING.value:
        #     return None
        # else:
        #     raise ValueError(f"Unknown market regime: {regime}") 
        
        # if regime == MarketRegime.BULL_BREAKOUT.value:
        #     return TrendFollowingStrategy()
        # elif regime == MarketRegime.BEAR_BREAKOUT.value:
        #     return None
        # elif regime == MarketRegime.STRONG_UP.value:
        #     return None
        # elif regime == MarketRegime.WEAK_UP.value:
        #     return None
        # elif regime == MarketRegime.STRONG_DOWN.value:
        #     return None
        # elif regime == MarketRegime.WEAK_DOWN.value:
        #     return None
        # elif regime == MarketRegime.ACCUMULATION.value:
        #     return None
        # elif regime == MarketRegime.DISTRIBUTION.value:
        #     return None
        # elif regime == MarketRegime.CHOPPY_SIDEWAYS.value:
        #     return None
        # elif regime == MarketRegime.DO_NOTHING.value:
        #     return None
        # else:
        #     raise ValueError(f"Unknown market regime: {regime}") 
            
