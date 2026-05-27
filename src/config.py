from pathlib import Path

import yaml

# locate the file
CONFIG_DIR = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_DIR) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)
