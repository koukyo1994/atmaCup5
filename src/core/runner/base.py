from typing import List

from src.core.callbacks import Callback
from src.core.states import RunningState


class Runner:
    signature = "base"

    def __init__(self, config: dict):
        self.config = config

        self.state = RunningState(config)

        # default callbacks
        self.callbacks: List[Callback] = []

    def _run_callbacks(self, phase="start"):
        assert phase in ["start", "end"]

        method = "on_" + self.signature + phase

        # add user defined callbacks
        callbacks = self.callbacks + self.state.callbacks[self.signature]

        for callback in sorted(callbacks):
            callback.__getattribute__(method)(self.state)

    def run(self):
        raise NotImplementedError
