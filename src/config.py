#!/usr/bin/env python3
"""
Configuration Loader & Schema Validation for Urban Traffic Vision Appliance
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


class ValidationError(ValueError):
    """Raised when configuration values or polygon coordinates fail validation."""
    pass


def load_settings(yaml_path: str) -> Dict[str, Any]:
    """Loads and validates settings.yaml."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_sections = ["app", "camera", "model", "tracking", "cluster_integrations", "archival"]
    for sec in required_sections:
        if sec not in config:
            raise ValidationError(f"Missing required configuration section: '{sec}'")

    return config


def _validate_polygon(poly: List[List[float]], name: str) -> None:
    """Validates that all points in a polygon are normalized between 0.0 and 1.0."""
    if not poly or len(poly) < 3:
        raise ValidationError(f"Polygon for '{name}' must have at least 3 points.")

    for i, pt in enumerate(poly):
        if len(pt) != 2:
            raise ValidationError(f"Point {i} in '{name}' must be [x, y], got {pt}")
        x, y = pt
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValidationError(
                f"Coordinates {pt} in '{name}' out of bounds! Must be normalized 0.0 <= x,y <= 1.0."
            )


def validate_zones_dict(zones: Dict[str, Any]) -> None:
    """Validates an in-memory dictionary of zones."""
    if not isinstance(zones, dict):
        raise ValidationError("Zones data must be a dictionary.")

    for key, val in zones.items():
        if isinstance(val, dict) and "polygon" in val:
            _validate_polygon(val["polygon"], key)

    if "apartment_windows" in zones and isinstance(zones["apartment_windows"], list):
        for win in zones["apartment_windows"]:
            if isinstance(win, dict) and "polygon" in win:
                win_id = win.get("id", "unknown_window")
                _validate_polygon(win["polygon"], f"apartment_window:{win_id}")


def load_zones(json_path: str) -> Dict[str, Any]:
    """Loads and validates zones.json normalized coordinates."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Zones file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        zones = json.load(f)

    validate_zones_dict(zones)
    return zones


def save_zones(zones: Dict[str, Any], json_path: str) -> None:
    """Validates and persists zones dictionary to JSON file atomically."""
    validate_zones_dict(zones)
    path = Path(json_path)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2)
    tmp_path.replace(path)
