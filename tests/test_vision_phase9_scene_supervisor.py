#!/usr/bin/env python3
"""
Unit Tests for Phase 9: Semantic Scene Supervisor (Track B: System 2 Reasoning)
Validates:
1. SemanticSceneSupervisor initialization and dual-backend loading (CLIP ONNX vs Heuristic).
2. Zero-shot scene evaluation on authentic Day vs Night Copacabana frames.
3. Multi-modal cross-validation with YOLO detections:
   - Specular puddle reflection detection and confidence elevation recommendation.
   - Ghost vehicle / empty street disagreement detection.
   - Solar astronomical telemetry disagreement detection.
   - Urban roadway obstruction alert dispatch.
4. REST API endpoint:
   - GET /api/scene
"""

import os
import sys
import json
import time
import math
import unittest
from pathlib import Path
from PIL import Image

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.scene_supervisor import (
    SemanticSceneSupervisor,
    SCENE_PROMPT_BANKS
)


class TestSemanticSceneSupervisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_dir = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")
        cls.supervisor = SemanticSceneSupervisor(use_embeddings=False)

    def test_supervisor_initialization(self):
        """Validates that supervisor initializes properly and reports valid backend."""
        self.assertIn(self.supervisor.backend, ["heuristic", "clip_onnx"])

    def test_default_prompt_banks(self):
        """Asserts all core prompt categories and candidates are defined."""
        self.assertIn("lighting", SCENE_PROMPT_BANKS)
        self.assertIn("roadway", SCENE_PROMPT_BANKS)
        self.assertIn("sidewalk", SCENE_PROMPT_BANKS)
        self.assertIn("clear_daylight", SCENE_PROMPT_BANKS["lighting"])
        self.assertIn("rain_wet_asphalt", SCENE_PROMPT_BANKS["lighting"])
        self.assertIn("flowing_traffic", SCENE_PROMPT_BANKS["roadway"])
        self.assertIn("empty_street", SCENE_PROMPT_BANKS["roadway"])

    def test_scene_evaluation_day_vs_night(self):
        """Tests scene evaluation and narrative generation on real Copacabana frames."""
        day_path = self.val_dir / "clean_v4_upright.jpg"
        night_path = self.val_dir / "live_night_now_upright.jpg"

        if day_path.exists():
            day_img = Image.open(day_path)
            res_day = self.supervisor.evaluate_scene(day_img)

            self.assertIn("narrative", res_day)
            self.assertIn("scene_state", res_day)
            self.assertIn("top_summary", res_day)
            self.assertEqual(res_day["top_summary"]["lighting"], "clear_daylight")
            self.assertGreater(res_day["scene_state"]["lighting"]["confidence"], 0.70)
            self.assertIsInstance(res_day["elapsed_ms"], float)

        if night_path.exists():
            night_img = Image.open(night_path)
            res_night = self.supervisor.evaluate_scene(night_img)

            self.assertIn("narrative", res_night)
            self.assertIn(res_night["top_summary"]["lighting"], ["twilight_dusk", "night_streetlamps"])

    def test_cross_validation_specular_puddle_reflection(self):
        """Tests detection of mirror reflections in rain regimes with low-confidence cars."""
        scene_report = {
            "top_summary": {"lighting": "rain_wet_asphalt", "roadway": "flowing_traffic"},
            "scene_state": {"lighting": {"confidence": 0.90}}
        }
        # Fake YOLO detection: car low on the roadway (y = 0.88) with low confidence (0.45)
        yolo_detections = [
            {"class": "car", "bbox": [0.1, 0.88, 0.3, 0.98], "confidence": 0.45}
        ]
        res = self.supervisor.cross_validate_with_yolo(scene_report, yolo_detections)

        self.assertFalse(res["is_consistent"])
        self.assertIn("SPECULAR_PUDDLE_REFLECTION_DETECTED", res["anomaly_flags"])
        self.assertTrue(len(res["recommendations"]) > 0)

    def test_cross_validation_ghost_traffic(self):
        """Tests disagreement when scene says empty street but YOLO reports multiple moving cars."""
        scene_report = {
            "top_summary": {"lighting": "clear_daylight", "roadway": "empty_street"},
            "scene_state": {"roadway": {"confidence": 0.88}}
        }
        yolo_detections = [
            {"class": "car", "speed_kmh": 35.0},
            {"class": "bus", "speed_kmh": 22.0},
            {"class": "car", "speed_kmh": 40.0}
        ]
        res = self.supervisor.cross_validate_with_yolo(scene_report, yolo_detections)

        self.assertFalse(res["is_consistent"])
        self.assertIn("GHOST_VEHICLE_DISAGREEMENT", res["anomaly_flags"])

    def test_cross_validation_solar_telemetry_mismatch(self):
        """Tests disagreement when HA sun elevation is midnight but image is clear daylight."""
        scene_report = {
            "top_summary": {"lighting": "clear_daylight", "roadway": "flowing_traffic"},
            "scene_state": {"lighting": {"confidence": 0.85}}
        }
        # HA reports midnight (-45 degrees elevation)
        ha_telemetry = {"solar_elevation": -45.0, "sun_state": "below_horizon"}
        res = self.supervisor.cross_validate_with_yolo(scene_report, [], ha_telemetry=ha_telemetry)

        self.assertFalse(res["is_consistent"])
        self.assertIn("SOLAR_LIGHTING_TELEMETRY_MISMATCH", res["anomaly_flags"])

    def test_cross_validation_obstruction_event(self):
        """Tests alert when roadway is classified as obstructed."""
        scene_report = {
            "top_summary": {"lighting": "clear_daylight", "roadway": "roadway_obstruction"},
            "scene_state": {"roadway": {"confidence": 0.82}}
        }
        res = self.supervisor.cross_validate_with_yolo(scene_report, [])

        self.assertFalse(res["is_consistent"])
        self.assertIn("URBAN_OBSTRUCTION_EVENT", res["anomaly_flags"])


class TestSceneSupervisorAPI(unittest.TestCase):
    """Tests HTTP server endpoint GET /api/scene."""

    @classmethod
    def setUpClass(cls):
        from http.server import HTTPServer
        import threading
        from src.api.server import VisionAPIHandler

        cls.port = 19097
        cls.server = HTTPServer(("127.0.0.1", cls.port), VisionAPIHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_api_scene_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/scene?name=clean_v4_upright.jpg"
        req = urllib.request.urlopen(url, timeout=5.0)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))

        self.assertIn("narrative", data)
        self.assertIn("scene_state", data)
        self.assertIn("top_summary", data)
        self.assertIn("cross_validation", data)
        self.assertIn("elapsed_ms", data)
        self.assertIsInstance(data["narrative"], str)
        self.assertIsInstance(data["cross_validation"]["is_consistent"], bool)


if __name__ == "__main__":
    unittest.main()
