from pathlib import Path
from typing import Any, Dict, List

from src.core.data_loading import get_normal_stats
from src.core.states import RunningState

from .base import Callback, CallbackOrder


class CalcStatsCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def on_data_loading_end(self, state: RunningState):
        data_loading_configs: List[
            Dict[str, Any]] = state.config["data_loading"]

        for config in data_loading_configs:
            data_dir = Path(config["dir"])
            stats_path = data_dir / config["stats"]

            mode = config["mode"]
            required = config["required"]
            name = config["name"]

            if not stats_path.exists(
            ) and mode == "normal" and required == "all":
                get_normal_stats(state.dataframes[name], config)
