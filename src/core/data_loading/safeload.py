import json
import subprocess as sp

import pandas as pd

import src.utils as utils

from pathlib import Path
from typing import Dict, Any


def check_file_exist(input_dir: Path, file_name: str) -> bool:
    return (input_dir / file_name).exists()


def determine_file_open_method(config: dict):
    file_format: str = config["name"].split(".")[-1]

    kwargs = {}
    if file_format in {"csv", "tsv"}:
        method = "read_csv"
        if file_format == "tsv":
            kwargs["sep"] = "\t"
    elif file_format == "parquet":
        method = "read_parquet"
    elif file_format in {"pickle", "pkl"}:
        method = "read_pikle"
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


def decide_required_columns(config: dict):
    required = config.get("required")
    if required is None:
        return
    elif required == "all":
        return
    elif isinstance(required, list):
        file_format = config["name"].split(".")[-1]
        if file_format in {"parquet", "pickle", "pkl"}:
            return
        elif file_format in ["csv", "tsv", "ftr", "feather"]:
            return required
    else:
        raise NotImplementedError


def get_normal_stats(df: pd.DataFrame, config: dict):
    stats: Dict[str, Any] = {}

    data_dir = Path(config["dir"])
    data_path = data_dir / config["name"]
    stats_path = data_dir / config["stats"]

    stats["line_count"] = len(df)
    stats["size"] = get_total_size(data_path)
    stats["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    stats["nuniques"] = {c: df[c].nunique() for c in df.columns}

    utils.save_json(stats, stats_path)


def open_stats(path: str):
    with open(path, "r") as f:
        stats = json.load(f)

    return stats
