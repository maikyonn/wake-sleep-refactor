"""Includes functionality for loading config files."""

import os
import json

from functools import lru_cache


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@lru_cache(maxsize=1)
def load_config():
    """Returns a dictionary loaded from the config.json file."""
    with open(os.path.join(CONFIG_DIR, "config.json")) as f:
        return json.load(f)


def load_model_config(path: str):
    """Returns a dictionary containing the model config."""
    if os.path.isfile(path):
        model_config_path = path
    else:
        model_config_path = os.path.join(CONFIG_DIR, "models", f"{path}.json")
        assert os.path.isfile(
            model_config_path
        ), f"Could not find config file at {path} or in config/models"
    with open(model_config_path) as f:
        return json.load(f)