from sklearn.base import BaseEstimator, TransformerMixin
class FrequencySubsetEncoder(BaseEstimator, TransformerMixin):
    """ Transforms categorical columns into a frequency columns"""
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        X = X.copy()
        self.maps_ = {}
        for col in self.cols:
            vc = X[col].value_counts(normalize=True)
            self.maps_[col] = vc
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps_[col]).fillna(0.0)
        return X