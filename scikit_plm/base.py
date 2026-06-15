from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ProteinEmbedder(ABC, BaseEstimator, TransformerMixin):

    @abstractmethod
    def _embed_sequence(self, X):
        pass

    def _validate_data(self, X):
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError(f"Expected 1D array of sequences, got shape {X.shape}.")
        return X

    def fit(self, X, y=None):
        self._validate_data(X)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        self._validate_data(X)
        embeddings = [self._embed_sequence(seq) for seq in X]
        return np.array(embeddings)
