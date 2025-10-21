import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PersonSexImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = "person_sex"
        self.random_state = 42
    
    def fit(self, X, y=None):
        # We take only the series 'person_sex'
        s = X[self.col]
        # In particular we create a mask for computing the probability
        # distribution obviously only for the instances having person_sex
        # different from NaN
        notna_mask = s.notna()
        self.classes_ = s[notna_mask].unique()
        self.probs = s[notna_mask].value_counts(normalize=True) 
        self._rng_ = np.random.default_rng(self.random_state)
        return self
    
    def transform(self, X, y = None):
        X_out = X.copy()
        mask_na = X_out[self.col].isna()
        if mask_na.any():
            X_out.loc[mask_na, self.col] = self._rng_.choice(
                self.classes_,
                size=mask_na.sum(),
                p=self.probs
            )
        return X_out