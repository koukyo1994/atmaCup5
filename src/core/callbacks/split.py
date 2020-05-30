import pandas as pd

from scipy.sparse import csr_matrix

from src.core.states import RunningState
from .base import Callback, CallbackOrder


# on_split_start
class CheckDataFrameCallback(Callback):
    signature = "split"
    calback_order = CallbackOrder.ASSERTION

    def on_split_start(self, state: RunningState):
        if state.features.get("main") is None:
            msg = "main DataFrame does not exist. Aborting."
            raise KeyError(msg)

        if state.features["main"].get("train") is None:
            msg = "train DataFrame does not exist. Aborting."
            raise KeyError(msg)

        if not isinstance(state.features["main"]["train"], pd.DataFrame):
            if not isinstance(state.features["main"]["train"], csr_matrix):
                msg = "feature 'train' must be DataFrame"
                msg += " or a sparse matrix (csr). Aborting."
                raise ValueError(msg)


class PrepareForStratifiedKFoldCallback(Callback):
    signature = "split"
    callback_order = CallbackOrder.MIDDLE

    def on_split_start(self, state: RunningState):
        train = state.features["main"]["train"]
        target = state.target

        train["y"] = target
        state.features["main"]["train"] = train

    def on_split_end(self, state: RunningState):
        train = state.features["main"]["train"]
        train = train.drop("y", axis=1)
        state.features["main"]["train"] = train
