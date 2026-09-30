#!/usr/bin/env python3
"""
File & Dataset Replay Adapter
Allows the vision pipeline and web zone editor to run against cached/historical
optical frames from Rua Barata Ribeiro without requiring the physical edge phone.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional
from PIL import Image


class FileReplayAdapter:
    """
    Ingests frames from a local directory of stored JPEG snapshots.
    Supports single-frame calibration, dataset cycling, and offline testing.
    """

    def __init__(self, data_dir: str = "/data/media/merged/vision/validation"):
        self.data_dir = Path(data_dir)
        self._frame_index = 0

    def list_available_frames(self) -> List[str]:
        """Lists all JPEG frames in the validation directory sorted chronologically."""
        if not self.data_dir.exists():
            return []
        patterns = ["latest_raw_upright.jpg", "window_placement_initial.jpg", "latest_view.jpg", "annotated_snapshot_*.jpg", "*.jpg"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(str(self.data_dir / pat)))
        # Deduplicate while preserving order
        unique = []
        seen = set()
        for f in found:
            if f not in seen and os.path.isfile(f):
                seen.add(f)
                unique.append(f)
        return unique

    def get_calibration_frame_path(self) -> str:
        """
        Returns the most suitable upright calibration frame path for window crosshairs.
        Prioritizes latest_raw_upright.jpg, then window_placement_initial.jpg.
        """
        priority_targets = [
            self.data_dir / "latest_raw_upright.jpg",
            self.data_dir / "window_placement_initial.jpg",
            self.data_dir / "latest_view.jpg"
        ]
        for target in priority_targets:
            if target.exists():
                return str(target)

        frames = self.list_available_frames()
        if frames:
            return frames[0]

        raise FileNotFoundError(f"No valid calibration frames found in {self.data_dir}")

    def get_frame(self, frame_path: Optional[str] = None) -> Image.Image:
        """Loads and returns a frame as a PIL Image."""
        path = frame_path or self.get_calibration_frame_path()
        return Image.open(path)

    def get_next_dataset_frame(self) -> Image.Image:
        """Cycles through available dataset frames sequentially for continuous simulation."""
        frames = self.list_available_frames()
        if not frames:
            raise FileNotFoundError(f"No frames available to cycle in {self.data_dir}")

        chosen = frames[self._frame_index % len(frames)]
        self._frame_index += 1
        return Image.open(chosen)
