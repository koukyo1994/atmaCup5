from . import Runner
from src.core.states import RunningState


class DataLoadingRunner(Runner):
    signature = "data_loading"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config)

        self.state = state
