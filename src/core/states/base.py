import pandas as pd

from typing import Dict


class RunningState:
    def __init__(self, config: dict):
        self.config = config
        self.dataframes: Dict[str, pd.DataFrame] = {}
