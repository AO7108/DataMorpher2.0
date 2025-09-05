import json
from pathlib import Path
from typing import Any, Dict


_DEFAULTS: Dict[str, Any] = {
    "min_quality_band": "medium",
    "curate_threshold": 0.25,
    "neutral_prompt": "a person with a neutral expression, mouth closed, no teeth visible, not smiling",
}


def load_settings(config_path: str | Path = Path("config.json")) -> Dict[str, Any]:
    try:
        p = Path(config_path)
        if p.exists() and p.is_file():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if not isinstance(data, dict):
                return dict(_DEFAULTS)
            merged = dict(_DEFAULTS)
            merged.update(data)
            return merged
    except (json.JSONDecodeError, OSError):
        pass
    return dict(_DEFAULTS)


