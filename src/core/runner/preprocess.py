import src.core.callbacks.preprocess as pp

from . import SubRunner
from src.core.states import RunningState


class PreprocessRunner(SubRunner):
    signature = "preprocess"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = [
            pp.TrainTestSameColumnsCallback(),
            pp.RemoveConstantCallback(),
            pp.RemoveDuplicatedCallback()
        ]

    def run(self):
        self._run_callbacks(phase="start")
        self._run_callbacks(phase="end")
