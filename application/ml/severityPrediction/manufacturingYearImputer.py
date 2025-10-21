from sklearn.base import BaseEstimator, TransformerMixin
from utility import Utility
import pandas as pd
import numpy as np

class ManufacturingYearImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.brand_col = "general_veichle_brand"
        self.year_col = "veichle_manufacturing_year"
        self.weights = (0.2,0.6,0.2)
        self.random_state = 42
    
    def fit(self, X, y=None):
        X = X
        # We construct the quartlies using only the rows in which year
        # and brand are not null
        mask_valid = X[self.brand_col].notna() & X[self.year_col].notna()
        if not mask_valid.any():
            # Empty median: not useful data
            self.quartiles_ = pd.DataFrame(
                columns=["Min", "Q1", "Median", "Q3", "Max"], dtype=float
            )
            self.medians_pool_ = np.array([], dtype=float)
            self._rng_ = np.random.default_rng(self.random_state)
            return self
        
        quartiles = (
            X.loc[mask_valid]
            .groupby(self.brand_col)[self.year_col]
            .quantile([0, 0.25, 0.5, 0.75, 1.0])
            .unstack()
            .rename(columns={0.00: "Min", 0.25: "Q1", 0.50: "Median", 0.75: "Q3", 1.00: "Max"})
            .sort_values("Median", ascending=False)
        )

        # We keep brands containing only all the quartiles
        quartiles = quartiles.dropna(subset=["Q1", "Median", "Q3", "Max"])
        self.quartiles_ = quartiles
        # Medians pool contains the median quartlie for each brand
        self.medians_pool_ = quartiles["Median"].to_numpy() if not quartiles.empty else np.array([], dtype=float)
        self._rng_ = np.random.default_rng(self.random_state)

        return self
    
    def _impute_brand_quartiles(self, X):
        utility = Utility()
        """ Imputes year_col for the brands having the quartiles (but without a year
        if considering the single instance)"""
    
        # The instances to impute: brand is there but year NaN
        mask_need = X[self.year_col].isna() & X[self.brand_col].isin(self.quartiles_.index)

        if not mask_need.any():
            return # There no exists any brand
        
        # The unique brand names among the instances to impute
        brands_needed = X.loc[mask_need, self.brand_col].unique()
        for brand in brands_needed:
            # I retrieve the row with the quartlies of the corresponding brand
            row = self.quartiles_.loc[brand]
            idx = X.index[(X[self.brand_col] == brand) & (X[self.year_col].isna())]
            if len(idx) == 0:
                continue

            years = utility.sample_years(
                row["Min"], row["Q1"], row["Median"], row["Q3"], row["Max"],
                n=len(idx),
                weights=self.weights
            )
            X.loc[idx, self.year_col] = years
    
    def _fallback_impute(self, X):
        """ We restort to this method if there's any residual NaN
         in 'veichle_manufacturing_year.'"""
        mask_residual = X[self.year_col].isna()
        n = mask_residual.sum()
        if not mask_residual.any():
            return
        if self.medians_pool_.size > 0:
            sampled = self._rng_.choice(self.medians_pool_, size=n)
            X.loc[mask_residual, self.year_col] = np.floor(sampled)
    
    def transform(self, X, y=None):
        X_out = X.copy()

        # We inpute basing on the quarties seen at fit time (so basing on the
        # brands for which we have a distribution of veichle_manufacturing_year)

        if hasattr(self, "quartiles_") and not self.quartiles_.empty:
            self._impute_brand_quartiles(X_out)
        
        # For the residual NaN years, we resort to the median pool
        self._fallback_impute(X_out) 

        X_out[self.year_col] = X_out[self.year_col].astype("int64")

        return X_out