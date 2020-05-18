import pandas as pd

from src.core.states import RunningState
from .base import Callback, CallbackOrder


# on_split_start
class CheckDataFrameCallback(Callback):
    signature = "split"
    calback_order = CallbackOrder.ASSERTION

    def on_split_start(self, state: RunningState):
        if state.features.get("train") is None:
            msg = "train DataFrame does not exist. Aborting."
            raise KeyError(msg)

        if not isinstance(state.features.get("train"), pd.DataFrame):
            msg = "feature 'train' must be DataFrame. Aborting."
            raise ValueError(msg)
