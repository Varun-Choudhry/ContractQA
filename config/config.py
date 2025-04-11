import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(filepath: str = CONFIG_PATH) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

config = load_config()
