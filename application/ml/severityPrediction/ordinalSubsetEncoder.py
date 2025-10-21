from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class OrdinalSubsetEncoder(BaseEstimator, TransformerMixin):
    """" Performs ordinal encoding for multiple columns"""
    def __init__(self, cols):
        self.cols = cols
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", # just as a precaution
            unknown_value=-1
        )

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.encoder.transform(X[self.cols])
        return X