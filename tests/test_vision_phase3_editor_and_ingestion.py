#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 3 Spatial Calibration Studio & Ingestion
Verifies:
1. Calibration Frame Registry & Metadata (Daylight vs Night, aspect ratios, luminance)
2. Multi-Layer Zone Schema Validation (Condo 723 Gate, Bar Tables, Bus Lane, Apartments)
3. Camera-Agnostic Stream Ingestion Adapters & Host Governance (scrcpy v4.1, RTSP, Replay, FPS & Bitrate caps)
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))

REAL_VISION_DIR = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")


class TestFrameRegistrySeam(unittest.TestCase):
    """Seam 1: Calibration Frame Registry & Metadata API."""

    def test_list_calibration_frames_returns_metadata_and_identifies_daylight(self):
        from src.ingestion.frame_registry import list_calibration_frames

        frames = list_calibration_frames(str(REAL_VISION_DIR))
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)

        # Inspect first priority frame (should be clean_v4_upright.jpg or daylight)
        first_frame = frames[0]
        self.assertIn("id", first_frame)
        self.assertIn("label", first_frame)
        self.assertIn("width", first_frame)
        self.assertIn("height", first_frame)
        self.assertIn("mean_v", first_frame)
        self.assertIn("is_daylight", first_frame)

        # Find clean_v4_upright.jpg
        v4_entry = next((f for f in frames if f["id"] == "clean_v4_upright.jpg"), None)
        if v4_entry:
            self.assertTrue(v4_entry["is_daylight"])
            self.assertGreater(v4_entry["mean_v"], 50.0)
            self.assertEqual(v4_entry["width"], 3024)
            self.assertEqual(v4_entry["height"], 4032)

        # Verify consistent daylight classification logic across all frames
        for entry in frames:
            self.assertEqual(entry["is_daylight"], entry["mean_v"] >= 40.0)

    def test_get_frame_path_by_id_with_safe_fallback(self):
        from src.ingestion.frame_registry import get_frame_path_by_id

        # Existing valid frame
        path = get_frame_path_by_id(str(REAL_VISION_DIR), "clean_v4_upright.jpg")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("clean_v4_upright.jpg"))

        # Non-existing frame safely falls back to default daylight reference
        fallback_path = get_frame_path_by_id(str(REAL_VISION_DIR), "non_existent_frame_123.jpg")
        self.assertTrue(os.path.exists(fallback_path))


class TestEnhancedZonesSchemaSeam(unittest.TestCase):
    """Seam 2: Multi-Layer Zone Schema & Normalized Polygon Validation."""

    def test_production_zones_contains_all_study_layers(self):
        from src.config import load_zones

        zones_file = PROJECT_ROOT / "config" / "zones.json"
        zones = load_zones(str(zones_file))

        # Asserts core operational layers are present
        self.assertIn("condo_723_gate", zones)
        self.assertIn("bar_tables", zones)
        self.assertIn("bus_lane", zones)
        self.assertIn("apartment_windows", zones)
        self.assertIn("roadway_roi", zones)

        # Validate Condominium 723 Gate bounds (Norm X: 0.64-0.80, Y: 0.50-0.68)
        condo_poly = zones["condo_723_gate"]["polygon"]
        self.assertGreaterEqual(len(condo_poly), 4)
        for pt in condo_poly:
            self.assertTrue(0.0 <= pt[0] <= 1.0)
            self.assertTrue(0.0 <= pt[1] <= 1.0)

        # Validate Sidewalk Bar Tables bounds (Norm X: 0.75-0.98, Y: 0.53-0.69)
        bar_poly = zones["bar_tables"]["polygon"]
        self.assertGreaterEqual(len(bar_poly), 4)
        for pt in bar_poly:
            self.assertTrue(0.0 <= pt[0] <= 1.0)
            self.assertTrue(0.0 <= pt[1] <= 1.0)

    def test_validate_and_save_zones_rejects_out_of_bounds(self):
        from src.config import validate_zones_dict

        invalid_zones = {
            "condo_723_gate": {
                "polygon": [[0.5, 0.5], [1.2, 0.5], [1.2, 0.8], [0.5, 0.8]] # 1.2 > 1.0
            }
        }
        with self.assertRaises(ValueError):
            validate_zones_dict(invalid_zones)


class TestStreamIngestionAdapterSeam(unittest.TestCase):
    """Seam 3: Camera-Agnostic Stream Ingestion Adapters & Host Governance."""

    def test_adapter_factory_instantiates_correct_driver(self):
        from src.ingestion.stream_adapter import StreamAdapterFactory, FileReplayAdapter, ScrcpyV4Adapter, RtspAdapter

        # 1. File Replay
        replay = StreamAdapterFactory.create({
            "type": "file_replay",
            "data_dir": str(REAL_VISION_DIR)
        })
        self.assertIsInstance(replay, FileReplayAdapter)

        # 2. Scrcpy v4.1
        scrcpy = StreamAdapterFactory.create({
            "type": "scrcpy",
            "target": "100.115.165.41:5555",
            "gateway_url": "http://ws-scrcpy.home.arpa",
            "max_fps": 5,
            "bitrate": "2M"
        })
        self.assertIsInstance(scrcpy, ScrcpyV4Adapter)

        # 3. Generic RTSP IP Camera
        rtsp = StreamAdapterFactory.create({
            "type": "rtsp",
            "url": "rtsp://192.168.0.100:554/live"
        })
        self.assertIsInstance(rtsp, RtspAdapter)

    def test_scrcpy_v4_adapter_builds_governed_start_payload(self):
        from src.ingestion.stream_adapter import ScrcpyV4Adapter

        adapter = ScrcpyV4Adapter(
            target="100.115.165.41:5555",
            gateway_url="http://ws-scrcpy.home.arpa",
            max_fps=5,
            bitrate="2M",
            codec="h265"
        )
        payload = adapter.get_start_payload()

        self.assertEqual(payload["target"], "100.115.165.41:5555")
        self.assertEqual(payload["codec"], "h265")
        self.assertEqual(payload["bitRate"], "2M")
        self.assertEqual(payload["maxFps"], 5)
        self.assertIn("cameraId", payload)

    def test_scrcpy_v4_adapter_enforces_safety_caps(self):
        from src.ingestion.stream_adapter import ScrcpyV4Adapter

        # If user/caller attempts unsafe 60 FPS or 25M bitrate, adapter caps them to protect host
        adapter = ScrcpyV4Adapter(
            target="100.115.165.41:5555",
            max_fps=60,      # Exceeds safe 15 FPS ceiling
            bitrate="25M"    # Exceeds safe 8M ceiling
        )
        payload = adapter.get_start_payload()

        self.assertLessEqual(payload["maxFps"], 15)
        self.assertEqual(payload["bitRate"], "8M")


if __name__ == "__main__":
    unittest.main()
