from src.core.states import RunningState


class Runner:
    signature = "base"

    def __init__(self, config: dict):
        self.config = config
        self.state = RunningState(config)

    def _run_callbacks(self, phase="start"):
        assert phase in ["start", "end"]

        method = "on_" + self.signature + phase

        callbacks = self.state.callbacks[self.signature]

        for callback in sorted(callbacks):
            callback.__getattribute__(method)(self.state)
