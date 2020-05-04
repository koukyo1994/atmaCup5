import json
import subprocess as sp

import pandas as pd

from pathlib import Path
from typing import Dict, Any

from src.utils import reduce_mem_usage


def check_file_exist(input_dir: Path, file_name: str) -> bool:
    return (input_dir / file_name).exists()


def determine_file_open_method(config: dict):
    if config.get("format") == "auto":
        file_format: str = config["name"].split(".")[-1]

    kwargs = {}
    if file_format in {"csv", "tsv"}:
        method = "read_csv"
        if file_format == "tsv":
            kwargs["sep"] = "\t"
    elif file_format == "parquet":
        method = "read_parquet"
    elif file_format in {"pickle", "pkl"}:
        method = "read_pkl"
    elif file_format in {"feather", "ftr"}:
        method = "read_feather"
    else:
        raise NotImplementedError
    return method, kwargs


def get_line_count(path: Path) -> int:
    line_count = int(
        sp.check_output(["wc", "-l", str(path)]).decode().split()[0])
    return line_count - 1


def get_total_size(path: Path) -> int:
    nbytes = int(sp.check_output(["ls", "-l", str(path)]).decode().split()[4])
    return nbytes


def get_normal_stats(df: pd.DataFrame, config: dict):
    stats: Dict[str, Any] = {}

    data_dir = Path(config["dir"])
    data_path = data_dir / config["name"]
    stats_path = data_dir / config["stats"]

    stats["line_count"] = len(df)
    stats["size"] = get_total_size(data_path)
    stats["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict()}
    stats["nuniques"] = {c: df[c].nunique() for c in df.columns}
    stats["id_columns"] = []
    for k, v in stats["nuniques"].items():
        if v == stats["line_count"]:
            stats["id_columns"].append(k)

    with open(stats_path, "w") as f:
        json.dump(stats, f)
