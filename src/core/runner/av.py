import numpy as np
import pandas as pd

import src.utils as utils

from . import SubRunner
from src.core.states import RunningState


class AVRunner(SubRunner):
    signature = "av"
    callback_group = "av"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

    def run(self):
        main_features = self.state.features["main"]
        train_features = main_features["train"]
        test_features = main_features["test"]

        cat_features = pd.concat([train_features, test_features],
                                 axis=0).reset_index(drop=True)

        target_0 = np.zeros(len(train_features))
        target_1 = np.ones(len(test_features))
        target = np.concatenate([target_0, target_1])

        with utils.timer("Adversarial Validation", self.state.logger):
            for key, value in self.config.items():
                state = RunningState(value, logger=self.state.logger)
                state.callbacks = self.state.callbacks
                state.misc = self.state.misc

                if key == "split":
                    state.features["main"] = {}
                    state.features["main"]["train"] = cat_features

                    from .split import SplitRunner

                    runner = SplitRunner(value, state)
                    runner.callback_group = "av"
                    runner.run()
                    self.state.splits = state.splits
                elif key == "model":
                    state.features["main"] = {}
                    state.features["main"]["train"] = cat_features
                    state.target = target
                    state.splits = self.state.splits

                    from .model import ModelRunner

                    runner = ModelRunner(value, state)
                    runner.callback_group = "av"
                    runner.run()

                    self.state.metrics = state.metrics
                    self.state.importances = state.importances
                else:
                    pass
        self.state.misc = state.misc
