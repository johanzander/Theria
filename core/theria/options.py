"""Reading the add-on options: options.json in production, config.yaml in development."""

import json
import os
from typing import Any

import yaml

from core.theria.exceptions import ConfigurationError


def load_options(options_path: str, config_path: str) -> dict[str, Any] | None:
    """The options mapping, or None when neither file exists (e.g. the container image
    has no config.yaml). Unreadable or invalid files raise ConfigurationError."""
    try:
        if os.path.exists(options_path):
            with open(options_path) as f:
                options = json.load(f)
        elif os.path.exists(config_path):
            with open(config_path) as f:
                options = yaml.safe_load(f).get("options", {})
        else:
            return None
    except (OSError, json.JSONDecodeError, yaml.YAMLError, AttributeError) as e:
        raise ConfigurationError(f"cannot read configuration: {e}") from e
    if not isinstance(options, dict):
        raise ConfigurationError("configuration must be a mapping")
    return options
