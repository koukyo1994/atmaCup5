import abc

import numpy as np
import pandas as pd

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState

from sklearn import metrics


class _OOFMetricCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.LOWER

    @abc.abstractmethod
    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.0

    @abc.abstractmethod
    def _log_score(self, state: RunningState, score: float):
        raise NotImplementedError

    def on_model_train_end(self, state: RunningState):
        X = state.features["main"]["train"]
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = state.target
        config = state.config

        oof = np.zeros(len(X))

        for i, (_, val_idx) in enumerate(state.splits):
            fold_signature = f"Fold{i+1}"

            X_valid = X[val_idx]
            model = state.models[config["name"] + "_" +
                                 config["identifier"]][fold_signature]
            preds = model.predict(X_valid)
            oof[val_idx] = preds
        score = self._metric(y, oof)
        self._log_score(state, score)

        identifier = config["name"] + "_" + config["identifier"]
        if identifier not in state.metrics.keys():
            state.metrics[identifier] = {}

        state.metrics[identifier]["oof"] = score


class OOFAUCCallback(_OOFMetricCallback):
    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.roc_auc_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float):
        state.logger.info(f"OOF AUC: {score:.5f}")


class OOFAPCallback(_OOFMetricCallback):
    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.average_precision_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float):
        state.logger.info(f"OOF PR-AUC: {score:.5f}")
