import abc

import numpy as np
import pandas as pd

import src.utils as utils

from sklearn import metrics

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


# on_model_train_end
class _OOFMetricCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.LOWER
    metric_name = ""

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

        state.metrics[identifier][f"oof {self.metric_name}"] = score


class OOFAUCCallback(_OOFMetricCallback):
    metric_name = "auc"

    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.roc_auc_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float):
        state.logger.info(f"OOF AUC: {score:.5f}")


class OOFAPCallback(_OOFMetricCallback):
    metric_name = "pr-auc"

    def _metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.average_precision_score(y_true, y_pred)

    def _log_score(self, state: RunningState, score: float):
        state.logger.info(f"OOF PR-AUC: {score:.5f}")


class OutputResultsCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.LOWEST

    def on_model_train_end(self, state: RunningState):
        metrics = state.metrics
        importances = state.importances.copy()

        for key, value in importances.items():
            if isinstance(value, pd.DataFrame):
                importances[key] = value.set_index("feature").sort_values(
                    by="value", ascending=False).to_dict()["value"]

        output_dir = state.output_dir

        result_dict = {"metrics": metrics, "importances": importances}

        state.logger.info(
            f"Create Output File {str(output_dir / 'output.json')}")
        utils.save_json(result_dict, output_dir / "output.json")


class StoreFeatureImportanceCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.HIGHER

    def on_model_train_end(self, state: RunningState):
        models = state.models
        importances = state.importances

        X = state.features["main"]["train"]
        if isinstance(X, pd.DataFrame):
            columns = X.columns.tolist()
        else:
            columns = [f"v{i}" for i in range(len(X[0]))]

        model_keys = set(models.keys())
        importance_keys = set(importances.keys())

        not_stored = model_keys - importance_keys

        for key in not_stored:
            with utils.timer(f"Store Feature Importance of the models `{key}`",
                             state.logger):
                model_folds = models[key]
                importances_array = np.zeros((len(columns)))
                for model in model_folds.values():
                    importances_array += model.get_feature_importance() / len(
                        model_folds)

                feature_importance = pd.DataFrame(
                    sorted(zip(importances_array, columns)),
                    columns=["value", "feature"])
                importances[key] = feature_importance

        state.importances = importances


class PlotFeatureImportanceCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.LOWER

    def __init__(self, model_name: str, title: str, figname: str):
        self.model_name = model_name
        self.title = title
        self.figname = figname

    def on_model_train_end(self, state: RunningState):
        importances = state.importances
        output_dir = state.output_dir

        save_path = output_dir / self.figname
        if importances.get(self.model_name) is not None:
            importance = importances.get(self.model_name)
            state.logger.info(
                f"Save Feature Importance plot to {str(save_path)}")
            utils.plot_feature_importances(importance, self.title, save_path)
