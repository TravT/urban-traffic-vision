#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 4 Perspective Polygons & Dual-Batch Benchmark (ADR-31)
Verifies:
1. Environmental Regime Classifier (Daylight, Rain Specular, Night Bloom)
2. Sampling Profiler & Displacement Mathematics (Eco 2 FPS, Balanced 5 FPS, Kinetic 10 FPS)
3. Perspective Multi-Vertex Polygon Schema & Glass-Clip Geometry
4. Dual-Batch Evaluation Engine (Authentic Day vs Night Validation Frames)
"""

import os
import sys
import json
import unittest
from pathlib import Path
from PIL import Image

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))

REAL_VISION_DIR = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")


class TestEnvironmentalRegimeClassifier(unittest.TestCase):
    """Seam 1: Environmental & Weather-Aware Confidence Matrix (ADR-31)."""

    def test_clear_daylight_regime(self):
        from src.analytics.batch_evaluator import EnvironmentalRegimeClassifier

        classifier = EnvironmentalRegimeClassifier()
        profile = classifier.classify(
            solar_elevation=35.0,
            weather_condition="sunny",
            mean_v=118.3
        )
        self.assertEqual(profile["regime"], "daylight")
        self.assertAlmostEqual(profile["vehicle_conf"], 0.45, places=2)
        self.assertAlmostEqual(profile["person_conf"], 0.40, places=2)
        self.assertEqual(profile["window_v_thresh"], 160.0)
        self.assertFalse(profile["puddle_filter"])
        self.assertFalse(profile["bloom_filter"])

    def test_tropical_rain_regime_suppresses_puddle_reflections(self):
        from src.analytics.batch_evaluator import EnvironmentalRegimeClassifier

        classifier = EnvironmentalRegimeClassifier()
        # Rain condition elevates vehicle confidence to 0.55 and enables puddle filter
        profile = classifier.classify(
            solar_elevation=15.0,
            weather_condition="rainy",
            mean_v=75.0
        )
        self.assertEqual(profile["regime"], "rain")
        self.assertGreaterEqual(profile["vehicle_conf"], 0.55)
        self.assertTrue(profile["puddle_filter"])

    def test_nocturnal_regime_activates_bloom_filter_and_delta_v(self):
        from src.analytics.batch_evaluator import EnvironmentalRegimeClassifier

        classifier = EnvironmentalRegimeClassifier()
        # Sun below horizon (negative elevation) and low luminance
        profile = classifier.classify(
            solar_elevation=-20.0,
            weather_condition="clear",
            mean_v=32.0
        )
        self.assertEqual(profile["regime"], "night")
        self.assertGreaterEqual(profile["vehicle_conf"], 0.50)
        self.assertLessEqual(profile["person_conf"], 0.38)
        self.assertTrue(profile["bloom_filter"])
        self.assertEqual(profile["window_eval_mode"], "differential_delta_v")
        self.assertEqual(profile["window_delta_v_thresh"], 45.0)


class TestSamplingProfiler(unittest.TestCase):
    """Seam 2: Multi-Framerate Pareto Optimization & Displacement Mathematics."""

    def test_displacement_mathematics(self):
        from src.analytics.batch_evaluator import SamplingProfiler

        profiler = SamplingProfiler()

        # Walking at 5 km/h:
        # At 2 FPS: (5 * 1000) / (3600 * 2) = 0.694 meters
        disp_walk_2fps = profiler.calculate_displacement(speed_kmh=5.0, fps=2.0)
        self.assertAlmostEqual(disp_walk_2fps, 0.694, places=2)

        # At 5 FPS: (5 * 1000) / (3600 * 5) = 0.278 meters
        disp_walk_5fps = profiler.calculate_displacement(speed_kmh=5.0, fps=5.0)
        self.assertAlmostEqual(disp_walk_5fps, 0.278, places=2)

        # Vehicle driving at 40 km/h:
        # At 5 FPS: (40 * 1000) / (3600 * 5) = 2.222 meters
        disp_car_5fps = profiler.calculate_displacement(speed_kmh=40.0, fps=5.0)
        self.assertAlmostEqual(disp_car_5fps, 2.222, places=2)

        # At 10 FPS: (40 * 1000) / (3600 * 10) = 1.111 meters
        disp_car_10fps = profiler.calculate_displacement(speed_kmh=40.0, fps=10.0)
        self.assertAlmostEqual(disp_car_10fps, 1.111, places=2)

    def test_operational_profiles_specs_match_adr31(self):
        from src.analytics.batch_evaluator import SamplingProfiler

        profiler = SamplingProfiler()
        profiles = profiler.get_profiles()

        self.assertIn("eco", profiles)
        self.assertIn("balanced", profiles)
        self.assertIn("kinetic", profiles)
        self.assertIn("burst", profiles)

        # Eco: 2 FPS, 0.8 Mbps, ~8.6 GB/day
        self.assertEqual(profiles["eco"]["fps"], 2)
        self.assertEqual(profiles["eco"]["bitrate_mbps"], 0.8)
        self.assertAlmostEqual(profiles["eco"]["daily_gb"], 8.64, places=1)

        # Balanced: 5 FPS, 2.2 Mbps, ~23.8 GB/day
        self.assertEqual(profiles["balanced"]["fps"], 5)
        self.assertEqual(profiles["balanced"]["bitrate_mbps"], 2.2)
        self.assertAlmostEqual(profiles["balanced"]["daily_gb"], 23.76, places=1)

        # Kinetic: 10 FPS, 4.5 Mbps, ~48.6 GB/day
        self.assertEqual(profiles["kinetic"]["fps"], 10)
        self.assertEqual(profiles["kinetic"]["bitrate_mbps"], 4.5)
        self.assertAlmostEqual(profiles["kinetic"]["daily_gb"], 48.6, places=1)


class TestPerspectivePolygonsSchema(unittest.TestCase):
    """Seam 3: Perspective Polygon Refinement & Glass-Boundary Tightening."""

    def test_zones_perspective_polygons(self):
        from src.config import load_zones, validate_zones_dict

        zones_file = PROJECT_ROOT / "config" / "zones.json"
        zones = load_zones(str(zones_file))

        # Check that condo_723_gate has at least 4 perspective points
        gate_poly = zones["condo_723_gate"]["polygon"]
        self.assertGreaterEqual(len(gate_poly), 4)

        # Check bar_tables has perspective points
        bar_poly = zones["bar_tables"]["polygon"]
        self.assertGreaterEqual(len(bar_poly), 4)

        # Check bus_lane and roadway_roi are perspective polygons
        self.assertIn("bus_lane", zones)
        self.assertIn("roadway_roi", zones)

        # Verify all coordinates are normalized [0.0, 1.0]
        validate_zones_dict(zones)

        # Ensure apartment windows have tight glass boundaries
        windows = zones.get("apartment_windows", [])
        self.assertGreaterEqual(len(windows), 4)
        for win in windows:
            poly = win["polygon"]
            self.assertEqual(len(poly), 4)
            # Verify width and height of each window crop is reasonable
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            self.assertTrue(0.01 <= w <= 0.45, f"Window {win['id']} width {w} out of bounds")
            self.assertTrue(0.01 <= h <= 0.25, f"Window {win['id']} height {h} out of bounds")


class TestBatchEvaluatorSeam(unittest.TestCase):
    """Seam 4: Dual-Batch Validation Execution (Day vs Night Real Frames)."""

    def test_batch_evaluator_runs_on_real_validation_frames(self):
        from src.analytics.batch_evaluator import BatchEvaluator
        from src.config import load_zones

        zones_file = PROJECT_ROOT / "config" / "zones.json"
        zones = load_zones(str(zones_file))

        evaluator = BatchEvaluator(data_dir=str(REAL_VISION_DIR), zones=zones)

        # Define 3 authentic day frames and 3 authentic night frames for fast deterministic test
        day_frames = ["clean_v4_upright.jpg", "dropzone_upright.jpg"]
        night_frames = ["live_night_now_upright.jpg"]

        report = evaluator.evaluate_batches(day_frames=day_frames, night_frames=night_frames)

        self.assertIn("day_batch", report)
        self.assertIn("night_batch", report)
        self.assertIn("sampling_recommendations", report)

        day_summary = report["day_batch"]
        self.assertEqual(day_summary["frame_count"], len(day_frames))
        self.assertGreater(day_summary["mean_luminance_v"], 80.0)
        self.assertEqual(day_summary["primary_regime"], "daylight")

        night_summary = report["night_batch"]
        self.assertEqual(night_summary["frame_count"], len(night_frames))
        self.assertEqual(night_summary["primary_regime"], "night")

        # Apt 202 is evaluated across the user-calibrated balcony polygon in live_night_now_upright.jpg
        apt_results = night_summary["frames"][0]["windows"]
        apt_202 = next((w for w in apt_results if w["id"] == "apt_202"), None)
        self.assertIsNotNone(apt_202)
        self.assertGreater(apt_202["mean_v"], 20.0)
        self.assertIn("color_temp", apt_202)


class TestVisionAPIRoutes(unittest.TestCase):
    """Seam 5: REST API Server Endpoints (/api/profiles, /api/batch/evaluate, /api/frames)."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import HTTPServer
        from src.api.server import VisionAPIHandler

        cls.server = HTTPServer(("127.0.0.1", 0), VisionAPIHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_profiles_endpoint(self):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/profiles"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("eco", data)
            self.assertIn("balanced", data)
            self.assertIn("kinetic", data)

    def test_get_frames_list_endpoint(self):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/frames/list"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

    def test_get_frame_by_name_endpoint(self):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/frame?name=clean_v4_upright.jpg"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "image/jpeg")
            body = resp.read()
            self.assertGreater(len(body), 1000)

    def test_get_batch_evaluate_endpoint(self):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/batch/evaluate"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("day_batch", data)
            self.assertIn("night_batch", data)
            self.assertIn("sampling_recommendations", data)

    def test_get_dwell_endpoint(self):
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/dwell"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("condo_gate", data)
            self.assertIn("sidewalk_bar", data)
            self.assertIn("active_condo_occupants", data["condo_gate"])


if __name__ == "__main__":
    unittest.main()
