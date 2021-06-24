import yaml
from pathlib import Path

import os


def load_config():
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r"), Loader=yaml.FullLoader
    )


def check_os_environ(key, use):
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required for {use}."
        )


def dataset_dir():
    check_os_environ("DATASET", "data loading")
    return os.environ["DATASET"]
