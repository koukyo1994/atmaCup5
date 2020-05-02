import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing features")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use subset of the data to calculate")
    return parser
