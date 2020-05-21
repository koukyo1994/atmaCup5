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
