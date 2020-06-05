import numpy as np
import pandas as pd

import src.utils as utils

from . import SubRunner
from src.core.states import RunningState


class PredictionRunner(SubRunner):
    signature = "model_inference"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        X = self.state.features["main"]["test"]
        if isinstance(X, pd.DataFrame):
            X = X.values

        for model_keys in self.state.models.keys():
            self.state.predictions[model_keys] = np.zeros(len(X))

        for i in range(len(self.state.splits)):
            fold_signature = f"Fold{i+1}"
            with utils.timer(f"Prediction with {fold_signature} models",
                             self.state.logger):
                for model_keys in self.state.models.keys():
                    model = self.state.models[model_keys][fold_signature]
                    for _ in range(self.config["tta"]):
                        self.state.predictions[model_keys] += model.predict(
                            X) / (len(self.state.splits) * self.config["tta"])

        self._run_callbacks(phase="end")
