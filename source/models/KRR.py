from functools import partial

import numpy as np

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import FeatureMap, pure_poli_features, fourier_features

class KRR(RegressorMixin, BaseEstimator):
    def __init__(
        self, 
        feature_map: tuple, 
        m_order: int = 10,  
        alpha: float = 1.0, 
    ):
        self.feature_map = feature_map
        self.m_order = m_order
        self.alpha = alpha

    def _prepare_feature_mapping(self):
        if self.feature_map.name == 'ppf':
            return partial(pure_poli_features, q=None, order=self.m_order)
        elif self.feature_map.name == 'ff':
            return partial(fourier_features, q=None, m_order=self.m_order, p_scale=self.feature_map.p_scale)
        else:
            raise ValueError(f'Bad feature_map = "{self.feature_map}". See docs.')
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # Calculate KRR weights, save train dataset:
        self._xtrain = X.copy()
        self.fmap = self._prepare_feature_mapping()
        a_mtx = prod_kernel(self._xtrain, self._xtrain, self.fmap) + self.alpha*np.eye(X.shape[0])
        self.weights_ = np.linalg.solve(a_mtx, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return np.real(prod_kernel(X, self._xtrain, self.fmap).dot(self.weights_))
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
def prod_kernel(x: np.ndarray, y: np.ndarray, fmap: FeatureMap) -> np.ndarray:
    _, d = x.shape
    k_mtx = 1
    for k in range(d):
        k_mtx *= fmap(x[:, k]).dot(fmap(y[:, k]).T)
    return k_mtx
