class BaseStrategy:
    def __init__(self):
        self.position = 0  # 0: no position, 1: buy, -1: sell

    def generate_signals(self, data):
        """Generate trading signals for the given data"""
        raise NotImplementedError 