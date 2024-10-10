from typing import Optional
from functools import partial

import numpy as np

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import FeatureMap

class RFF(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        feature_map: str = 'rff', 
        n_features: int = 10,  
        gamma: float = 1.0,
        alpha: float = 1.0, 
        random_state: Optional[int] = None,
        batch_size: int = 100,
    ):
        self.feature_map = feature_map
        self.n_features = n_features
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state
        self.batch_size = batch_size

    def _prepare_feature_mapping(self, d_dim: float):
        if self.feature_map == 'rff':
            return prepare_rff_features(
                rff_features, self.n_features, 
                d_dim, self.gamma, self.random_state
            )
        else:
            raise ValueError(f'Bad feature_map = "{self.feature_map}". See docs.')
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._feature_mapping = self._prepare_feature_mapping(X.shape[-1])
        self.weights_ = rff(
            X, y, self.n_features, self._feature_mapping, 
            self.alpha, self.batch_size,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return self._feature_mapping(X).dot(self.weights_)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
def rff(
    x: np.ndarray, 
    y: np.ndarray, 
    feature_dim: int, 
    feature_map: FeatureMap,
    reg_value: float,
    batch_size: int = 100,
) -> np.ndarray:
    n_samples, _ = x.shape
    ftf_mtx = np.diag(reg_value * np.ones(feature_dim))
    fty = np.zeros(feature_dim)
    for i in range(0, n_samples, batch_size):
        idx = min(n_samples, i + batch_size)
        temp = feature_map(x[i:idx, :])
        ftf_mtx += temp.T.dot(temp)
        fty += temp.T.dot(y[i:idx])
    return np.linalg.solve(ftf_mtx, fty)

def prepare_rff_features(
    feature_funct: FeatureMap,
    feature_dim: float,
    data_dim: float,
    gamma: float, 
    seed: Optional[int] = None
):
    random_state = np.random if seed is None else np.random.RandomState(seed)
    freqs = random_state.normal(0, np.sqrt(2 * gamma), (data_dim, feature_dim))
    biases = random_state.uniform(0, 2 * np.pi, feature_dim)
    return partial(feature_funct, freqs=freqs, biases=biases)

def rff_features(
    x: np.ndarray, 
    freqs: np.ndarray, 
    biases: np.ndarray
) -> np.ndarray:
    _, f_dim = freqs.shape
    return np.sqrt(2.0 / f_dim) * np.cos(x.dot(freqs) + biases)

def rff_features_complex(
    x: np.ndarray, 
    freqs: np.ndarray, 
    biases: np.ndarray
) -> np.ndarray:
    """ 
    TODO 
    """
    _, f_dim = freqs.shape
    return np.sqrt(1 / f_dim) * np.exp(1j * x.dot(freqs) + biases)
