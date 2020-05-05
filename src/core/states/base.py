import datetime as dt

import pandas as pd

import src.utils as utils

from pathlib import Path
from typing import Dict, Optional


class RunningState:
    def __init__(self, config: dict):
        self.config = config

        log_dir = Path(config["log_dir"])
        log_dir.mkdir(exist_ok=True, parents=True)

        config_name = Path(config["config_path"]).name.replace(".yml", "")
        log_name = config_name + "_" + dt.datetime.now().strftime(
            "%Y%m%d-%H:%M:%S") + ".log"
        self.logger = utils.get_logger(str(log_dir / log_name))
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataframe_roles: Dict[str, str] = {}

        self.callbacks: Dict[str, list] = {}

        self.data_stats: Dict[str, Optional[str]] = {}
