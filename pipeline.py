import pandas as pd
from .factor import *

class Pipeline:

    def __init__(self):
        self.factors = {}

    def run(self, data, modify=True):
        factor_results = {}
        for name, factor in self.factors.items():
            if isinstance(factor, BollingerBands):
                h, m, l = factor.compute(data)
                factor_results[f'h{name}'] = h
                factor_results[f'm{name}'] = m
                factor_results[f'l{name}'] = l
            else:
                factor_results[name] = factor.compute(data)
        
        stacked_dfs = []
        for factor, series in factor_results.items():
            series = series.stack()
            series.name = factor
            df = pd.DataFrame(series)
            df.rename_axis(['Date', 'Equity'], inplace=True)
            df = df.T
            stacked_dfs.append(df)
        
        factor_results = pd.concat(stacked_dfs).T
        self.factor_results = factor_results.dropna()
        return self.factor_results

    def factor_snippet(self, equity=None, dates=None):
        snippet = self.factor_results
        if equity is not None:
            snippet = snippet.loc[(slice(None), equity), :]
        if dates is not None:
            snippet = snippet.loc[dates]
        return snippet

    def add(self, factors):
        for factor in factors:
            self.factors[factor] = factors[factor]
    