import numpy as np

import src.utils as utils

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
        msg = f"Applying query `{self.query}` on {self.df_name}."
        with utils.timer(msg, state.logger):
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
                msg = f"Filling missing value in column `{column}` "
                msg += f"with value {self.values[column]}."
                with utils.timer(msg, state.logger):
                    df[column] = df[column].fillna(self.values[column])
            state.dataframes[df_name] = df


class MercariPreProcessCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWEST

    def __init__(self, df_names: List[str]):
        self.df_names = df_names

    def on_features_start(self, state: RunningState):
        for name in self.df_names:
            with utils.timer(f"Preprocessing {name}", state.logger):
                df = state.dataframes[name]
                df["name"] = df["name"] + " " + df["brand_name"]
                df["item_description"] = df["item_description"] + " " + df["name"] + \
                    " " + df["category_name"]
                columns = [
                    "name", "item_description", "shipping", "item_condition_id"
                ]
                if state.dataframe_roles[name] == "train":
                    columns.append("price")
                df = df[columns]
                state.dataframes[name] = df


# on_features_end
class LogTransformOnTargetCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.HIGHEST

    def on_features_end(self, state: RunningState):
        with utils.timer("Applying log1p on target", state.logger):
            state.target = np.log1p(state.target)
