# Location: src/config.py

import yaml
from pathlib import Path

def _load_config():
    """
    Helper function to find and load config.yaml from the project root.
    """
    # Go up two directories from this file (src/config.py -> src -> project root)
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load the configuration once when this module is first imported.
# Other parts of the app can then simply 'from src.config import config'
config = _load_config()