import numpy as np

from typing import Dict, List

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


# on_features_start
class QueryTrainRecordsCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.HIGHER

    def __init__(self, query: str, df_name: str):
        self.df_name = df_name
        self.query = query

    def on_features_start(self, state: RunningState):
        df = state.dataframes[self.df_name]
        df = df.query(self.query).reset_index(drop=True)

        state.dataframes[self.df_name] = df


class FillNACallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.HIGHEST

    def __init__(self, columns: List[str], values: Dict[str, str]):
        self.columns = columns
        self.values = values

    def on_features_start(self, state: RunningState):
        for df_name, df in state.dataframes.items():
            for column in self.columns:
                df[column] = df[column].fillna(self.values[column])
            state.dataframes[df_name] = df


# on_features_end
class LogTransformOnTargetCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.HIGHEST

    def on_features_end(self, state: RunningState):
        state.target = np.log1p(state.target)
