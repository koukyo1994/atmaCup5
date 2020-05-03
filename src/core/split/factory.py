import numpy as np
import pandas as pd

from typing import List, Tuple


def get_split(df: pd.DataFrame,
              config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config["how"]
    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    return func(df, config)
