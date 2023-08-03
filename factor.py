import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

class CustomFactor:

    def __init__(self, window):
        self.window = window

    def compute(self, data):
        raise NotImplementedError("compute() must be implemented by CustomFactor subclasses")
    
class Returns(CustomFactor):
    
    def __init__(self, window):
        super().__init__(window)

    @classmethod
    def compute(cls, data):
        obj = cls(0)
        return obj._compute(data)

    def _compute(self, data):
        return np.log(data).diff().rename(columns={'close': 'returns'})
    
class RoC(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        return data.pct_change(self.window)

class SMA(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        return data.rolling(self.window).mean()

class EMA(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        return data.ewm(span=self.window).mean()
    
class RSI(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        avg_gain = up.rolling(self.window).mean()
        avg_loss = abs(down.rolling(self.window).mean())

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
class MovingMax(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        return data.rolling(self.window).max()

class MovingMin(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        return data.rolling(self.window).min()

class MovingVol(CustomFactor):

    def __init__(self, window):
        super().__init__(window)

    def compute(self, data):
        returns = np.log(data).diff()
        return returns.rolling(self.window).std()
    
class MovingSharpe(CustomFactor):

    def __init__(self, window, rfr):
        super().__init__(window)
        self.rfr = rfr

    def compute(self, data):
        returns = np.log(data).diff()
        exc_ret = returns - self.rfr/252
        moving_vol = returns.rolling(self.window).std()
        return exc_ret / moving_vol
    
class BollingerBands(CustomFactor):

    def __init__(self, window, num_sd=2):
        super().__init__(window)
        self.num_sd = num_sd

    def compute(self, data):
        ma = data.rolling(self.window).mean()
        sd = data.rolling(self.window).std()
        upper_band = ma + (sd * self.num_sd)
        lower_band = ma - (sd * self.num_sd)
        return upper_band, ma, lower_band
    
class Beta(CustomFactor):

    def __init__(self, window, benchmark_returns):
        super().__init__(window)
        self.benchmark_returns = benchmark_returns
    
    def compute(self, data):
        returns = np.log(data).diff()
        betas = []
        for col in returns:
            lin_reg = RollingOLS(returns[[col]].dropna(), sm.add_constant(self.benchmark_returns.dropna()), window=self.window).fit()
            beta = lin_reg.params[['returns']].copy()
            beta.rename(columns={'returns': col}, inplace=True)
            betas.append(beta)
        betas = [df.T for df in betas]
        betas = pd.concat(betas).T
        return betas

class MACD(CustomFactor):
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, data):
        short_ema = EMA(self.short_window).compute(data)
        long_ema = EMA(self.long_window).compute(data)
        return short_ema - long_ema
    
class PercentAboveLow(CustomFactor):

    def __init__(self, window):
        super().__init__(window)
        self.name = f'PctAbove{self.window}Low'

    def compute(self, data):
        return (data / data.rolling(self.window).min() - 1)

class FourFiftyTwo(CustomFactor):
    def __init__(self, short_window=21, long_window=252):
        self.name = 'PriceRatio4W52W'
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, data):
        return (data.rolling(self.short_window).mean() / data.rolling(self.long_window).mean() - 1).rolling(20).mean()

class MomentumStrength(CustomFactor):
    def __init__(self, strength='drastic', short_window=21, long_window=252):
        self.name = 'MomentumStrength'
        self.strength = strength
        self.short_window = short_window
        self.long_window = long_window
    
    def compute(self, data):
        if self.strength == 'smooth':
            return (data.pct_change(self.long_window) * (1 + data.rolling(self.short_window).std()))
        elif self.strength == 'drastic':
            return data.pct_change(self.long_window) / (data.rolling(self.short_window).std())

class StochasticMomentum(CustomFactor):

    def __init__(self, window):
        super().__init__(window)
        self.name = 'StochasticMomentum'
    
    def compute(self, data):
        low = data.rolling(self.window).min()
        high = data.rolling(self.window).max()
        stoch = (data - low) / (high - low)
        return data.pct_change(self.window) * stoch