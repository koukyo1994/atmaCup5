from typing import Dict, List

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


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
