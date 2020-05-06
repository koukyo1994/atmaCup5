import src.utils as utils
import src.core.data_loading as dl

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.states import RunningState

from .base import Callback, CallbackOrder


class FileExistenceCheckCallback(Callback):
    callback_order = CallbackOrder.HIGHEST

    def on_data_loading_start(self, state: RunningState):
        data_loading_configs: List[Dict[str, Any]] = state.config

        no_exist = []
        stats: Dict[str, Optional[str]] = {}

        for config in data_loading_configs:
            data_dir = Path(config["dir"])
            file_path = data_dir / config["name"]

            if not dl.check_file_exist(data_dir, config["name"]):
                no_exist.append(file_path)

            file_format = config["name"].split(".")[-1]
            if file_format in {"csv", "tsv"}:
                stats[str(file_path)] = str(
                    data_dir / config["stats"]) if dl.check_file_exist(
                        data_dir, config["stats"]) else None

        if len(no_exist) > 0:
            msg = "File: [\n"
            for path in no_exist:
                msg += "    " + str(path) + "\n"
            msg += "] does not exist, aborting."
            raise FileNotFoundError(msg)

        state.data_stats.update(stats)


# on_data_loading_end
class CompressDataFrameCallback(Callback):
    callback_order = CallbackOrder.HIGHEST

    def on_data_loading_end(self, state: RunningState):
        with utils.timer("Data Compressing", state.logger):
            data_frames = state.dataframes
            for key in data_frames:
                data_frames[key] = utils.reduce_mem_usage(
                    data_frames[key], verbose=True, logger=state.logger)


class CalcStatsCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def on_data_loading_end(self, state: RunningState):
        data_loading_configs: List[Dict[str, Any]] = state.config

        for config in data_loading_configs:
            data_dir = Path(config["dir"])
            stats_path = data_dir / config["stats"]

            mode = config["mode"]
            required = config["required"]
            name = config["name"]

            if not stats_path.exists(
            ) and mode == "normal" and required == "all":
                dl.get_normal_stats(state.dataframes[name], config)
