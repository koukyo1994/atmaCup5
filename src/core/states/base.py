import pandas as pd

from typing import Dict, Optional


class RunningState:
    def __init__(self, config: dict):
        self.config = config
        self.dataframes: Dict[str, pd.DataFrame] = {}

        self.callbacks: Dict[str, list] = {}

        self.data_stats: Dict[str, Optional[str]] = {}
