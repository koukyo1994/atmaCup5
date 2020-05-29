import numpy as np
import pandas as pd

import src.utils as utils

from sklearn.metrics import average_precision_score

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


# on_features_start
class MergeFittingCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWER

    def on_features_start(self, state: RunningState):
        fitting_key = "input/atma5/fitting.csv"
        fitting = state.dataframes[fitting_key]
        for phase in ["train", "test"]:
            key = f"input/atma5/{phase}.csv"
            msg = f"Merging `{key}` with fitting.csv"
            with utils.timer(msg, state.logger):
                df = state.dataframes[key]
                state.dataframes[key] = df.merge(
                    fitting, how="left", on="spectrum_id")


class CalcOOFMetricCallback(Callback):
    signature = "model_train"
    callback_order = CallbackOrder.LOWER

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
        score = average_precision_score(y, oof)
        state.logger.info(f"OOF PR-AUC: {score:.5f}")
