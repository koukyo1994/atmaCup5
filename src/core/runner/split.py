from . import SubRunner
from src.core.states import RunningState
from src.core.split import get_split


class SplitRunner(SubRunner):
    signature = "split"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        df = self.state.features["train"]
        splits = get_split(df, self.config)
        self.state.splits = splits

        self._run_callbacks(phase="end")
