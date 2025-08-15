from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from binarizer import binarizer

class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bin = binarizer(col='income')
        self.optb = None  # aqui vamos guardar o optimal binning

    def fit(self, x, y):
        self.bin.fit(x, y)

        # criar o OptimalBinning para risk_rate
        self.optb = OptimalBinning(name='risk_rate', dtype="numerical",  min_bin_size=0.05, solver="cp")
        self.optb.fit(x['risk_rate'].values, y)

        return self

    def transform(self, x):
        df = x.copy()

        # aplicar binarizer
        df = self.bin.transform(df)

        # criar novas variáveis
        df['n_bankruptcies_flag'] = (df['n_bankruptcies'] > 1).astype(int)

        # aplicar optimal binning
        df['risk_rate_bin'] = self.optb.transform(df['risk_rate'].values, metric="woe")  # ou "bins" se quiser o número do bin

        return df
