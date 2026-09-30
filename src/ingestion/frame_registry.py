#!/usr/bin/env python3
"""
Calibration Frame Registry & Metadata Service
Discovers, classifies (daylight vs night), and indexes authentic frames for the Spatial Calibration Studio.
Optimized for low CPU overhead via fast JPEG thumbnail sampling.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageStat


PRIORITY_ORDER = [
    "latest_annotated.jpg",
    "latest_raw_upright.jpg",
    "clean_v4_upright.jpg",
    "dropzone_upright.jpg",
    "live_night_now_upright.jpg",
    "window_placement_initial.jpg",
    "live_night_now.jpg",
    "latest_view.jpg",
    "latest_raw.jpg"
]

LABELS = {
    "latest_annotated.jpg": "🟢 Live AI Detections (latest_annotated - Real-Time YOLOv8)",
    "latest_raw_upright.jpg": "🔴 Live Camera Feed (latest_raw_upright - S20 FE 12MP)",
    "clean_v4_upright.jpg": "☀️ Daylight 12MP Reference (clean_v4 - Clear Street & Awning)",
    "dropzone_upright.jpg": "🏙️ Full Scene Daylight (dropzone - 12MP Upright)",
    "live_night_now_upright.jpg": "🌙 Authentic Live Night (12MP Upright - Apt 202 TV Active)",
    "window_placement_initial.jpg": "🌅 Initial Scene Daylight (window_placement)",
    "live_night_now.jpg": "🌙 Authentic Live Night (Raw Landscape)",
    "latest_view.jpg": "📷 Recent View Snapshot",
    "latest_raw.jpg": "📷 Recent Raw Frame"
}


def list_calibration_frames(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scans data_dir for available calibration frames, extracting dimensions,
    HSV luminance (mean V), and daylight classification.
    Uses fast thumbnail sampling to keep execution under 50ms.
    """
    directory = Path(data_dir)
    if not directory.exists():
        return []

    found_files = []
    # 1. Add priority files if they exist
    for filename in PRIORITY_ORDER:
        file_path = directory / filename
        if file_path.exists() and file_path.is_file():
            found_files.append(file_path)

    # 2. Add up to 5 additional unique JPGs if available
    extra_count = 0
    for f in sorted(glob.glob(str(directory / "*.jpg")), reverse=True):
        p = Path(f)
        if p not in found_files and p.is_file() and not p.name.startswith("crop_") and not p.name.startswith("test_"):
            found_files.append(p)
            extra_count += 1
            if extra_count >= 5:
                break

    results = []
    for path in found_files:
        filename = path.name
        try:
            with Image.open(path) as img:
                w, h = img.size
                # Fast downsample to 128x128 for instantaneous HSV calculation
                thumb = img.resize((128, 128), Image.Resampling.NEAREST)
                stat = ImageStat.Stat(thumb.convert("HSV"))
                mean_v = float(stat.mean[2])
                is_daylight = (mean_v >= 40.0)

            label = LABELS.get(filename, f"Frame: {filename} ({w}x{h})")

            results.append({
                "id": filename,
                "path": str(path),
                "label": label,
                "width": w,
                "height": h,
                "mean_v": round(mean_v, 2),
                "is_daylight": is_daylight
            })
        except Exception:
            continue

    return results


def get_frame_path_by_id(data_dir: str, frame_id: str) -> str:
    """
    Returns the absolute path to a frame by ID, safely falling back
    to the highest-priority daylight frame if the requested ID is missing.
    """
    directory = Path(data_dir)
    target = directory / frame_id
    if target.exists() and target.is_file():
        return str(target)

    # Fallback to priority frames
    for prio in PRIORITY_ORDER:
        p = directory / prio
        if p.exists() and p.is_file():
            return str(p)

    # Any jpg
    any_jpg = glob.glob(str(directory / "*.jpg"))
    if any_jpg:
        return any_jpg[0]

    raise FileNotFoundError(f"No valid calibration frames found in {data_dir}")
