import pandas as pd

from pathlib import Path
from typing import Dict, List, Union

from fastprogress import progress_bar


class BasicOperationsOnSpectrum:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        operations = self.kwargs["operations"]
        prefix = self.kwargs["prefix"] + "_"
        result_dict: Dict[str, List[Union[int, float]]] = {
            op: []
            for op in operations
        }

        base_dir = Path("input/atma5/spectrum")
        for _, row in progress_bar(X.iterrows(), total=len(X)):
            spectrum = pd.read_csv(
                base_dir / row.spectrum_filename, sep="\t", header=None)
            for op in operations:
                result_dict[op].append(spectrum[1].__getattribute__(op)())
        df = pd.DataFrame(result_dict)
        df.columns = [prefix + c for c in df.columns]
        return df
