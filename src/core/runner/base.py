import datetime as dt

import src.utils as utils

from pathlib import Path
from typing import List

from src.core.callbacks import Callback
from src.core.states import RunningState


class SubRunner:
    signature = "sub"

    def __init__(self, config: dict, state: RunningState):
        self.config = config
        self.state = state
        self.callbacks: List[Callback] = []

    def _run_callbacks(self, phase="start"):
        assert phase in ["start", "end"]

        method = "on_" + self.signature + "_" + phase

        # add user defined callbacks
        user_defined_callbacks = self.state.callbacks.get(self.signature)
        callbacks = self.callbacks
        if user_defined_callbacks is not None:
            callbacks = callbacks + self.state.callbacks[self.signature]

        for callback in sorted(callbacks):
            callback.__getattribute__(method)(self.state)

    def run(self):
        raise NotImplementedError


class Runner:
    def __init__(self, config: dict):
        self.config = config

        log_dir = Path(config["log_dir"])
        log_dir.mkdir(exist_ok=True, parents=True)

        config_name = Path(config["config_path"]).name.replace(".yml", "")
        self.init_time = dt.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        log_name = config_name + "_" + self.init_time + ".log"

        logger = utils.get_logger(str(log_dir / log_name))

        self.state = RunningState(config, logger)

        # default callbacks
        self.callbacks: List[Callback] = []

    def run(self):
        feature_dir = Path(self.config["feature_dir"])
        output_root_dir = Path(self.config["output_dir"])

        feature_dir.mkdir(exist_ok=True, parents=True)
        output_root_dir.mkdir(exist_ok=True, parents=True)

        self.state.feature_dir = feature_dir

        config_name = self.config["config_path"].split("/")[-1].replace(
            ".yml", "")
        output_dir = output_root_dir / config_name

        if output_dir.exists():
            output_dir = output_dir / self.init_time
        output_dir.mkdir(parents=True, exist_ok=True)
        self.state.output_dir = output_dir

        for pl in self.config["pipeline"]:
            for key, value in pl.items():
                if key == "data_loading":
                    state = RunningState(value, logger=self.state.logger)
                    from .data_loading import DataLoadingRunner

                    runner = DataLoadingRunner(value, state)
                    runner.run()
                    self.state.dataframes = state.dataframes
                    self.state.data_stats = state.data_stats
                    self.state.dataframe_roles = state.dataframe_roles
                    self.state.target = state.target
                    self.state.id_columns = state.id_columns
                    self.state.connect_to = state.connect_to
                    self.state.connect_on = state.connect_on
                    import pdb
                    pdb.set_trace()
                else:
                    pass
