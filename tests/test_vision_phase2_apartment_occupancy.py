#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 2 Apartment Occupancy & Replay
Verifies:
1. File / Dataset Replay Adapter (Using stored real frames)
2. Apartment Occupancy & HSV Luminosity / Domestic Color Temperature Analyzer
3. Zone Editor API & Polygon Coordinate Validation
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


class TestFileReplayAdapterSeam(unittest.TestCase):
    """Seam 1: File & Dataset Replay Adapter."""

    def setUp(self):
        from src.ingestion.file_replay_adapter import FileReplayAdapter
        self.FileReplayAdapter = FileReplayAdapter

    def test_loads_real_calibration_frame(self):
        # Uses real image from the vision folder
        adapter = self.FileReplayAdapter(data_dir=str(REAL_VISION_DIR))
        frame_path = adapter.get_calibration_frame_path()
        self.assertTrue(os.path.exists(frame_path))
        self.assertTrue(frame_path.endswith(".jpg"))

    def test_reads_valid_pil_image_from_dataset(self):
        adapter = self.FileReplayAdapter(data_dir=str(REAL_VISION_DIR))
        img = adapter.get_frame()
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(img.width, 1000)
        self.assertGreater(img.height, 1000)
        img.close()



class TestApartmentOccupancySeam(unittest.TestCase):
    """Seam 2: Deterministic HSV Luminosity & Color Temperature Analyzer."""

    def setUp(self):
        from src.analytics.apartment_occupancy import ApartmentOccupancyTracker
        self.tracker = ApartmentOccupancyTracker(
            luminance_threshold=160.0,
            dark_threshold=80.0
        )

    def test_dark_window_crop_classified_as_unoccupied(self):
        # Synthetic 100x100 dark window (V=20)
        dark_img = Image.new("HSV", (100, 100), (0, 0, 20)).convert("RGB")
        status = self.tracker.analyze_window(dark_img, "apt_101")

        self.assertFalse(status["is_illuminated"])
        self.assertEqual(status["color_temp"], "dark")
        self.assertLess(status["mean_v"], 80.0)

    def test_warm_lit_window_crop_classified_as_occupied(self):
        # Synthetic warm domestic lighting: Hue=28 (Pillow scale: ~20), Saturation=180, Value=220
        warm_img = Image.new("HSV", (100, 100), (20, 180, 220)).convert("RGB")
        status = self.tracker.analyze_window(warm_img, "apt_201")

        self.assertTrue(status["is_illuminated"])
        self.assertEqual(status["color_temp"], "warm")
        self.assertGreater(status["mean_v"], 160.0)

    def test_cold_screen_lit_window_classified_as_cold_occupied(self):
        # Synthetic cold blue screen: Hue=150 (Pillow scale for ~210 deg blue), Sat=160, Val=200
        cold_img = Image.new("HSV", (100, 100), (150, 160, 200)).convert("RGB")
        status = self.tracker.analyze_window(cold_img, "apt_301")

        self.assertTrue(status["is_illuminated"])
        self.assertEqual(status["color_temp"], "cold")
        self.assertGreater(status["mean_v"], 160.0)

    def test_full_frame_occupancy_analysis_computes_aggregate_rate(self):
        # Create a frame with 2 windows: 1 dark, 1 warm lit
        test_frame = Image.new("RGB", (1000, 1000), (10, 10, 10))
        # Warm lit window at (100, 100) to (200, 200) -> normalized: [0.1, 0.1, 0.2, 0.2]
        warm_patch = Image.new("HSV", (100, 100), (20, 180, 220)).convert("RGB")
        test_frame.paste(warm_patch, (100, 100))

        zones = {
            "apartment_windows": [
                {
                    "id": "apt_101",
                    "label": "Apartment 101",
                    "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]
                },
                {
                    "id": "apt_102",
                    "label": "Apartment 102",
                    "polygon": [[0.3, 0.1], [0.4, 0.1], [0.4, 0.2], [0.3, 0.2]]
                },
                {
                    "id": "doctor_fit",
                    "label": "Doctor Fit Studio",
                    "polygon": [[0.5, 0.1], [0.6, 0.1], [0.6, 0.2], [0.5, 0.2]]
                }
            ]
        }

        report = self.tracker.analyze_frame(test_frame, zones)

        self.assertIn("windows", report)
        self.assertEqual(len(report["windows"]), 3)
        self.assertEqual(report["active_residential_count"], 1)
        self.assertEqual(report["total_residential_count"], 2)  # doctor_fit is commercial
        self.assertEqual(report["occupancy_rate_pct"], 50.0)
        self.assertFalse(report["commercial_active"])  # doctor_fit was dark


class TestRealDatasetOccupancyEvaluation(unittest.TestCase):
    """Evaluates the occupancy tracker against authentic window frames from Rua Barata Ribeiro."""

    def test_evaluates_real_barata_ribeiro_night_frame(self):
        from src.analytics.apartment_occupancy import ApartmentOccupancyTracker
        from src.config import load_zones

        zones_path = PROJECT_ROOT / "config" / "zones.json"
        zones = load_zones(str(zones_path))
        tracker = ApartmentOccupancyTracker()

        # Test against real night snapshot from the window
        night_img_path = REAL_VISION_DIR / "latest_raw_upright.jpg"
        if not night_img_path.exists():
            night_img_path = REAL_VISION_DIR / "window_placement_initial.jpg"

        with Image.open(night_img_path) as img:
            report = tracker.analyze_frame(img, zones)
            self.assertIn("occupancy_rate_pct", report)
            self.assertIn("windows", report)
            self.assertIsInstance(report["occupancy_rate_pct"], (int, float))




class TestAPIServerSeam(unittest.TestCase):
    """Seam 3: REST API & Zone Editor Endpoints."""

    def test_handler_processes_health_and_zones(self):
        from src.api.server import VisionAPIHandler
        from io import BytesIO

        # Test health
        mock_handler = VisionAPIHandler.__new__(VisionAPIHandler)
        mock_handler.path = "/health"
        mock_handler.wfile = BytesIO()
        mock_handler.send_response = lambda code: setattr(mock_handler, "response_code", code)
        mock_handler.send_header = lambda k, v: None
        mock_handler.end_headers = lambda: None

        mock_handler.do_GET()
        self.assertEqual(mock_handler.response_code, 200)
        data = json.loads(mock_handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(data["status"], "healthy")

        # Test zones GET
        mock_handler.path = "/api/zones"
        mock_handler.wfile = BytesIO()
        mock_handler.do_GET()
        self.assertEqual(mock_handler.response_code, 200)
        zones_data = json.loads(mock_handler.wfile.getvalue().decode("utf-8"))
        self.assertIn("apartment_windows", zones_data)


if __name__ == "__main__":
    unittest.main()

