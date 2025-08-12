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
                return 'low_income'
            elif val < 3500:
                return 'medium_income'
            elif val < 7000:
                return 'good_income'
            else:
                return 'excellent_income'

        df[self.col + '_bin'] = df.apply(bin, axis=1)
        return df
