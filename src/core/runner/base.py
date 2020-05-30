import datetime as dt

import src.utils as utils

import src.core.callbacks as cl
import src.custom as custom

from pathlib import Path
from typing import List, Optional

from src.core.callbacks import Callback
from src.core.states import RunningState


class SubRunner:
    signature = "sub"

    def __init__(self, config: dict, state: RunningState):
        self.config = config
        self.state = state
        self.callbacks: List[Callback] = []

    def _run_callbacks(self, phase="start", signature: Optional[str] = None):
        assert phase in ["start", "end"]

        signature = self.signature if signature is None else signature

        method = "on_" + signature + "_" + phase

        # add user defined callbacks
        user_defined_callbacks = self.state.callbacks.get(signature)
        callbacks = self.callbacks
        if user_defined_callbacks is not None:
            preset_callback_names = [
                callback.__class__.__name__ for callback in callbacks
            ]
            for callback in user_defined_callbacks:
                if callback.__class__.__name__ in preset_callback_names:
                    # overwrite
                    index = preset_callback_names.index(
                        callback.__class__.__name__)
                    callbacks[index] = callback
                else:
                    callbacks.append(callback)

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
        log_dir = log_dir / config_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.init_time = dt.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        log_name = self.init_time + ".log"

        logger = utils.get_logger(str(log_dir / log_name))

        self.state = RunningState(config, logger)

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

        callbacks = self.config["callbacks"]
        if callbacks is not None:
            for callback in callbacks:
                callback_name = callback["name"]
                callback_type = callback["type"]
                callback_params = {} if callback.get(
                    "params") is None else callback["params"]

                if "custom" in callback_type:
                    submodule = callback_type.split(".")[1]
                    cl_instance = custom.__getattribute__(
                        submodule).__getattribute__(
                            "callbacks").__getattribute__(callback_name)(
                                **callback_params)
                    callback_type = cl_instance.signature
                    if callback_type not in self.state.callbacks.keys():
                        self.state.callbacks[callback_type] = []

                    self.state.callbacks[callback_type].append(cl_instance)
                else:
                    if callback_type not in self.state.callbacks.keys():
                        self.state.callbacks[callback_type] = []
                    self.state.callbacks[callback_type].append(
                        cl.__getattribute__(callback_type).__getattribute__(
                            callback_name)(**callback_params))

        for pl in self.config["pipeline"]:
            for key, value in pl.items():
                state = RunningState(value, logger=self.state.logger)
                state.callbacks = self.state.callbacks
                state.misc = self.state.misc
                if key == "data_loading":
                    from .data_loading import DataLoadingRunner

                    runner = DataLoadingRunner(value, state)
                    runner.run()
                    self.state.dataframes = state.dataframes
                    self.state.data_stats = state.data_stats
                    self.state.dataframe_roles = state.dataframe_roles
                    self.state.target_name = state.target_name
                    self.state.id_columns = state.id_columns
                    self.state.connect_to = state.connect_to
                    self.state.connect_on = state.connect_on
                elif key == "features":
                    state.dataframes = self.state.dataframes
                    state.data_stats = self.state.data_stats
                    state.dataframe_roles = self.state.dataframe_roles
                    state.id_columns = self.state.id_columns
                    state.connect_to = self.state.connect_to
                    state.connect_on = self.state.connect_on
                    state.target_name = self.state.target_name

                    from .features import FeaturesRunner

                    runner = FeaturesRunner(value, state)
                    runner.run()
                    self.state.features = state.features
                    self.state.target = state.target
                elif key == "feature_loading":
                    state.feature_dir = self.state.feature_dir
                    state.features = self.state.features

                    from .feature_loading import FeatureLoadingRunner

                    runner = FeatureLoadingRunner(value, state)
                    runner.run()

                    self.state.features = state.features
                    self.state.target = state.target
                elif key == "split":
                    state.features = self.state.features
                    state.target = self.state.target

                    from .split import SplitRunner

                    runner = SplitRunner(value, state)
                    runner.run()
                    self.state.splits = state.splits
                elif key == "model":
                    state.features = self.state.features
                    state.target = self.state.target
                    state.splits = self.state.splits

                    from .model import ModelRunner

                    runner = ModelRunner(value, state)
                    runner.run()

                    self.state.models = state.models
                elif key == "av":
                    state.features = self.state.features

                    from .av import AVRunner

                    runner = AVRunner(value, state)
                    runner.run()
                elif key == "prediction":
                    state.splits = self.state.splits
                    state.models = self.state.models
                    state.features = self.state.features

                    from .prediction import PredictionRunner

                    runner = PredictionRunner(value, state)
                    runner.run()

                    self.state.predictions = state.predictions
                else:
                    pass

                self.state.misc = state.misc
