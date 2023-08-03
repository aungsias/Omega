import warnings
import pandas as pd
import yfinance as yf
from .utils import get_pricing, get_prices

class Ticker:
    
    def __init__(self, ticker):
        self.ticker = ticker
    
    def get_prices(self, start, end, update=False, yahoo=False, warn=False):

        if isinstance(self.ticker, str):
            prices = get_pricing(self.ticker, start=start, end=end, yahoo=yahoo)
            return pd.DataFrame(prices).rename(columns={'Close': 'close'})
        
        elif isinstance(self.ticker, list):
            prices = get_prices(self.ticker, start=start, end=end, yahoo=yahoo)
            for col in prices:
                if prices[col].isnull().any():
                    fvi = prices[col].first_valid_index()
                    index = prices.index.get_loc(fvi)
                    if warn:
                        warning = f'{col} has null values. The first valid index is {fvi} (index = {index}).'
                        warnings.warn(warning)
            if update:
                prices.to_csv('universe_price_data.csv')
            return prices
        
    def bulk_prices(self, start, end):
        return yf.download(self.ticker, start, end, auto_adjust=True)['Close'].dropna(axis=1)[self.ticker]
        
class Universe:

    def __init__(self, universe):
        self.universe = universe.copy()

    @classmethod
    def extract(cls, universe):
        obj = cls(universe)
        tickers, indices = obj._extract()
        return tickers, indices

    def _extract(self):
        tickers, indices = self.extract_tickers(), self.extract_indices()
        return tickers, indices

    def extract_tickers(self):
        self.tickers = set()
        for sector in self.universe:
            self.tickers.update(self.universe[sector]['tickers'])
        self.tickers = list(self.tickers)
        return self.tickers
    
    def extract_indices(self):
        self.indices = set()
        for sector in self.universe:
            self.indices.update(self.universe[sector]['indices'])
        self.indices = list(self.indices)
        return self.indices
    