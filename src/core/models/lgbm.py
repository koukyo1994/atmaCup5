import lightgbm as lgb
import numpy as np

from typing import List, Tuple, Optional, Callable

from .base import TreeModel, Matrix


class LGBModel(TreeModel):
    def __init__(self,
                 mode: str,
                 feval: Optional[Callable[[np.ndarray, lgb.Dataset],
                                          Tuple[str, float, bool]]] = None):
        super().__init__(mode)

        self.feval = feval

    def fit(self, X_train: Matrix, y_train: Matrix,
            valid_sets: List[Tuple[Matrix, Matrix]],
            valid_names: Optional[List[str]], model_params: dict,
            train_params: dict):
        d_train = lgb.Dataset(X_train, labels=y_train)

        lgb_valid_sets: List[lgb.Dataset] = []
        for X, y in valid_sets:
            lgb_valid_sets.append(lgb.Dataset(X, labels=y))

        if valid_names is None:
            valid_names = ["valid_" + str(i) for i in range(len(valid_sets))]

        if self.feval is not None:
            train_params["feval"] = self.feval

        model = lgb.train(
            params=model_params,
            train_set=d_train,
            valid_sets=lgb_valid_sets,
            valid_names=valid_names,
            **train_params)
        self.model = model

    def predict(self, X_test: Matrix):
        self._assert_if_untrained()
        return self.model.predict(X_test)  # type: ignore

    def get_best_iterations(self) -> int:
        self._assert_if_untrained()
        return self.model.best_iteration  # type: ignore

    def get_feature_importance(self) -> np.ndarray:
        self._assert_if_untrained()
        return self.model.feature_importance(  # type: ignore
            importance_type="gain")

    def _assert_if_untrained(self):
        if self.model is None:
            msg = "LGBModel has not been trained yet."
            msg += "Call `.fit()` method first."
            raise AttributeError(msg)
