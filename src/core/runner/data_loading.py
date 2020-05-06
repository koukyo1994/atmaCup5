import pandas as pd

import src.core.callbacks.data_loading as dlc
import src.core.data_loading as dl

from pathlib import Path

from . import SubRunner
from src.core.states import RunningState


class DataLoadingRunner(SubRunner):
    signature = "data_loading"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = [
            dlc.FileExistenceCheckCallback(),
            dlc.CompressDataFrameCallback()
        ]

    def run(self):
        self._run_callbacks(phase="start")

        for config in self.state.config:
            method, kwargs = dl.determine_file_open_method(config)
            columns = dl.decide_required_columns(config)
            if columns is not None:
                kwargs["columns"] = columns
            file_path = Path(config["dir"]) / config["name"]
            if method in {"read_parquet", "read_pickle", "read_feather"}:
                df = pd.__getattribute__(method)(file_path, **kwargs)
                self.state.dataframes[str(file_path)] = df
            elif method == "read_csv":
                if config["mode"] == "normal":
                    df = pd.__getattribute__(method)(file_path, **kwargs)
                    self.state.dataframes[str(file_path)] = df
                elif config["mode"] == "large":
                    raise NotImplementedError
                else:
                    pass
            else:
                raise NotImplementedError
            self.state.dataframe_roles[str(file_path)] = config["role"]

        self._run_callbacks(phase="end")
