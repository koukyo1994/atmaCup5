import logging

import pandas as pd

from pathlib import Path
from typing import Dict, Optional


class RunningState:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config

        self.logger = logger

        self.feature_dir = Path("features")
        self.output_dir = Path("output")

        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataframe_roles: Dict[str, str] = {}

        self.callbacks: Dict[str, list] = {}

        self.data_stats: Dict[str, Optional[str]] = {}
