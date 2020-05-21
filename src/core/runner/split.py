import src.core.callbacks.split as spc
import src.utils as utils

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

        with utils.timer("Data Split", self.state.logger):
            main_features = self.state.features["main"]

            df = main_features["train"]
            splits = get_split(df, self.config)

            self.state.splits = splits

        self._run_callbacks(phase="end")
