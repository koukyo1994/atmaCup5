import pandas as pd

import src.core.features.functional as F
import src.utils as utils

from . import SubRunner
from src.core.states import RunningState


class FeaturesRunner(SubRunner):
    signature = "features"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for conf in self.config:
            input_dfs = conf["input_dataframes"]
            for df_name in input_dfs:
                if self.state.dataframe_roles.get(df_name) == "train":
                    train_df = self.state.dataframes[df_name]
                elif self.state.dataframe_roles.get(df_name) == "test":
                    test_df = self.state.dataframes[df_name]
                else:
                    raise NotImplementedError
            kwargs = {} if conf.get("params") is None else conf.get("params")
            transformer = F.__getattribute__(conf["method"])(**kwargs)
            columns = conf.get("columns")

            with utils.timer(conf["method"] + " on train...",
                             self.state.logger):
                train_feats = transformer.fit_transform(train_df[columns])
            with utils.timer(conf["method"] + " on test...",
                             self.state.logger):
                test_feats = transformer.transform(test_df[columns])

            if len(columns) == 1:
                feature_name = conf["method"] + "_" + conf["columns"][0]
            else:
                feature_name = conf["method"]

            self.state.features[feature_name] = {
                "train": train_feats,
                "test": test_feats
            }

        self._run_callbacks(phase="end")