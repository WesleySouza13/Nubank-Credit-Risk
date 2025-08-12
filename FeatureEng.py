from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from binarizer import binarizer
#from binarizer import binarizer
class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bin = binarizer(col='income')

    def fit(self, x, y=None):
        self.bin.fit(x, y)
        return self

    def transform(self, x):
        df = x.copy()
    
        
        # criar novas variaveis : score3, n_issues 
        #df['n_bankruptcies_flag'] = (df['n_bankruptcies'] > 1).astype(int)
        #df['score_per_issue'] = df['score_3'] * df['n_issues']
        #df['score_per_bankruptcies_flag'] = df['score_3'] * df['n_bankruptcies_flag']
        #df['log_n_bankruptcies'] = np.log1p(df['n_bankruptcies'])
        #df['score_per_bankruptcies'] = df['score_3'] / (1 + df['n_bankruptcies'])

        return df
