import numpy as np

class Aggregate:

    def __init__(self, factor_df):
        self.factor_df = factor_df

    def rank(self):
        data = self.factor_df.copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        mean = data.mean()
        std = data.std()
        standardized_data = (data - mean) / std
        standardized_data = standardized_data.apply(self.filter_fn) / len(standardized_data.columns)
        composite_score = standardized_data.sum(axis=1)
        return composite_score.sort_values(ascending=False)
    
    @staticmethod
    def filter_fn(x):
        return np.clip(x, -10, 10)


def describe(date, data):
    ranks = data.loc[date]
    for i, rank in zip(ranks.index, ranks.values):
        print("{} {:>10} {:>20,.10f}".format(date, i, rank))