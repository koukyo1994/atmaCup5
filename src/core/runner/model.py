import pandas as pd

import src.utils as utils

from . import SubRunner
from src.core.states import RunningState
from src.core.models import get_model


class ModelRunner(SubRunner):
    signature = "model_train"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        X = self.state.features["main"]["train"]
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = self.state.target
        config = self.config
        model_params = config["model_params"]
        train_params = config["train_params"]

        self.state.models[config["name"] + "_" + config["identifier"]] = {}

        for i, (trn_idx, val_idx) in enumerate(self.state.splits):
            self._run_callbacks(phase="start", signature="train_fold")
            self.state.misc["current_fold_id"] = i
            fold_signature = f"Fold{i+1}"

            self.state.logger.info("=" * 25)
            self.state.logger.info(fold_signature)
            self.state.logger.info("=" * 25)

            X_train, X_valid = X[trn_idx], X[val_idx]
            y_train, y_valid = y[trn_idx], y[val_idx]

            valid_sets = [(X_valid, y_valid)]
            valid_names = ["valid"]

            model = get_model(config, self.state.logger)

            with utils.timer(fold_signature + " training...",
                             self.state.logger):
                model.fit(X_train, y_train, valid_sets, valid_names,
                          model_params, train_params)
                self.state.models[config["name"] + "_" +
                                  config["identifier"]][fold_signature] = model
            self._run_callbacks(phase="end", signature="train_fold")

        self._run_callbacks(phase="end")
