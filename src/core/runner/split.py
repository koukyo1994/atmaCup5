import pandas as pd

import src.core.callbacks.split as spc

from . import SubRunner
from src.core.states import RunningState
from src.core.split import get_split


class SplitRunner(SubRunner):
    signature = "split"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = [spc.CheckDataFrameCallback()]

    def run(self):
        self._run_callbacks(phase="start")

        main_features = self.state.features["main"]
        if isinstance(main_features["train"], pd.DataFrame):
            df = main_features["train"]
            splits = get_split(df, self.config)
        elif isinstance(main_features["train"], dict):
            # sparse matrix
            df = main_features["train"]["main"]
            splits = get_split(df, self.config)
        self.state.splits = splits

        self._run_callbacks(phase="end")
