from sklearn.base import BaseEstimator, TransformerMixin

class binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        df = x.copy()

        def bin(row):
            val = row[self.col]
            if val < 1500:
                return 1 # grupo baixa renda
            elif val < 3500:
                return 2 # renda media
            elif val < 7000:
                return 3 # renda boa
            else:
                return 4 # renda otima

        df[self.col + '_bin'] = df.apply(bin, axis=1)
        return df
