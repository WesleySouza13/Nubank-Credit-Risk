from sklearn.base import BaseEstimator, TransformerMixin

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.col, axis=1)