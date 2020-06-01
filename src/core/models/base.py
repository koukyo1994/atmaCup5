import abc

import numpy as np
import scipy.sparse as sp

from typing import List, Tuple, Union, Optional

SparseMatrix = Union[sp.csr_matrix, Union[sp.csc_matrix, sp.coo_matrix]]
Matrix = Union[np.ndarray, SparseMatrix]


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, mode: str):
        self.mode = mode

        self.model = None

    @abc.abstractmethod
    def fit(self, X_train: Matrix, y_train: Matrix,
            valid_sets: List[Tuple[Matrix, Matrix]],
            valid_names: Optional[List[str]], model_params: dict,
            train_params: dict):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X_test: Matrix):
        raise NotImplementedError


class TreeModel(BaseModel):
    def __init__(self, mode: str):
        super().__init__(mode)

    @abc.abstractmethod
    def get_best_iterations(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        raise NotImplementedError


class NNModel(BaseModel):
    def __init__(self, mode: str):
        self.mode = mode

        self.model = None
