import abc

import numpy as np
import pandas as pd

from sklearn import metrics

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


# on_train_fold_end
class _FoldMetricCallback(Callback):
    signature = "train_fold"
    callback_order = CallbackOrder.LOWER
    metric_name = ""

    @abc.abstractmethod
    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.0

    @abc.abstractmethod
    def _log_score(self, state: RunningState, score: float,
                   fold_signature: str):
        raise NotImplementedError

    def on_train_fold_end(self, state: RunningState):
        X = state.features["main"]["train"]
        if isinstance(X, pd.DataFrame):
            X = X.values
        fold_id = state.misc["current_fold_id"]

        y = state.target
        config = state.config

        fold_signature = f"Fold{fold_id+1}"

        val_idx = state.splits[fold_id][1]

        X_valid = X[val_idx]
        y_valid = y[val_idx]

        identifier = config["name"] + "_" + config["identifier"]

        model = state.models[identifier][fold_signature]
        preds = model.predict(X_valid)

        score = self._metric(y_valid, preds)
        self._log_score(state, score, fold_signature)

        if identifier not in state.metrics.keys():
            state.metrics[identifier] = {}

        state.metrics[identifier][
            f"{fold_signature} {self.metric_name}"] = score


class FoldAUCCallback(_FoldMetricCallback):
    metric_name = "auc"

    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.roc_auc_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float,
                   fold_signature: str):
        state.logger.info(f"{fold_signature} AUC: {score:.5f}")


class FoldAPCallback(_FoldMetricCallback):
    metric_name = "pr-auc"

    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.average_precision_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float,
                   fold_signature: str):
        state.logger.info(f"{fold_signature} PR-AUC: {score:.5f}")
