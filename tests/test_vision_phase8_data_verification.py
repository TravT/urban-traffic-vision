#!/usr/bin/env python3
"""
Unit Tests for Phase 8: Data Observability, Ground-Truth Verification & Continuous Quality Assurance
Validates:
1. Cross-Modal Sanity & Plausibility Auditing:
   - Velocity bounds (>80 km/h or impossible acceleration jumps)
   - Re-ID & dwell pruning (orphaned entries >2h, single-frame ghost tracks at bar)
   - Environmental cross-validation (HA weather vs. acoustic rain vs. road specular index)
   - Window luminosity hysteresis (rejects car headlight sweep flickers)
2. Optical Alignment & Lens Drift Watchdog:
   - Architectural anchor alignment on Doctor Fit & Condo 723 gate
   - High-frequency edge gradient variance & lens obscuration / dirty glass detection
3. Empirical Quality Scoring & Ground-Truth Sampling:
   - Composite quality scoring (0-100%) and grading (EXCELLENT, GOOD, DEGRADED, CRITICAL)
   - Verification audit manifest logging in /data/media/merged/vision/validation/quality_audit/
4. REST API Endpoint:
   - GET /api/quality/audit
"""

import os
import sys
import json
import time
import math
import unittest
from pathlib import Path
from PIL import Image, ImageFilter

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_zones
from src.analytics.quality_auditor import (
    CrossModalSanityAuditor,
    WindowOccupancyHysteresis,
    OpticalAlignmentWatchdog,
    DataQualityScorer,
    QualityAuditEngine,
    HomographyRecalibrator
)


class TestCrossModalSanityAuditor(unittest.TestCase):
    def setUp(self):
        self.auditor = CrossModalSanityAuditor()

    def test_velocity_bounds_normal_and_spikes(self):
        """Validates realistic velocities and catches impossible spikes or accelerations."""
        # 1. Normal vehicle: 40 km/h Eastbound across 25m baseline
        track_normal = {"track_id": 1, "speed_kmh": 40.0, "x": 0.5, "y": 0.7, "timestamp": 100.0}
        prev_normal = {"track_id": 1, "speed_kmh": 38.0, "x": 0.45, "y": 0.7, "timestamp": 99.5}
        is_valid, anomalies = self.auditor.audit_velocity_bounds(track_normal, prev_normal, dt=0.5)
        self.assertTrue(is_valid)
        self.assertEqual(len(anomalies), 0)

        # 2. Extreme speed spike (> 80 km/h ceiling)
        track_hyper = {"track_id": 2, "speed_kmh": 115.0, "x": 0.8, "y": 0.7, "timestamp": 100.0}
        is_valid, anomalies = self.auditor.audit_velocity_bounds(track_hyper, prev_normal, dt=0.5)
        self.assertFalse(is_valid)
        self.assertIn("UNREALISTIC_VELOCITY_SPIKE", anomalies)

        # 3. Impossible instantaneous acceleration jump (> 15 m/s^2)
        # From 10 km/h (2.77 m/s) to 75 km/h (20.8 m/s) in 0.1s -> a = 180 m/s^2
        track_instant = {"track_id": 3, "speed_kmh": 75.0, "x": 0.9, "y": 0.7, "timestamp": 100.1}
        prev_slow = {"track_id": 3, "speed_kmh": 10.0, "x": 0.4, "y": 0.7, "timestamp": 100.0}
        is_valid, anomalies = self.auditor.audit_velocity_bounds(track_instant, prev_slow, dt=0.1)
        self.assertFalse(is_valid)
        self.assertIn("IMPOSSIBLE_ACCELERATION_JUMP", anomalies)

    def test_reid_and_dwell_pruning(self):
        """Validates dwell tracking pruning: orphaned condo entries and ghost bar tracks."""
        # 1. Normal condo visit (15 min / 900s)
        dwell_state_normal = {
            "condo_occupants": {"subject_1": {"entry_time": 1000.0, "dwell_seconds": 900.0}},
            "bar_table_patrons": [{"id": "patron_1", "hit_count": 8, "dwell_seconds": 45.0}]
        }
        res_normal = self.auditor.audit_dwell_tracking(dwell_state_normal, current_time=1900.0)
        self.assertEqual(len(res_normal["anomalies"]), 0)
        self.assertEqual(len(res_normal["pruned_orphans"]), 0)

        # 2. Orphaned condo entry (> 2 hours / 7200s)
        dwell_state_orphaned = {
            "condo_occupants": {"subject_old": {"entry_time": 1000.0, "dwell_seconds": 10800.0}}, # 3 hours
            "bar_table_patrons": []
        }
        res_orphan = self.auditor.audit_dwell_tracking(dwell_state_orphaned, current_time=11800.0)
        self.assertIn("ORPHANED_CONDO_ENTRY", res_orphan["anomalies"])
        self.assertIn("subject_old", res_orphan["pruned_orphans"])

        # 3. Bar table transient ghost track (single frame hit < min_bar_hits)
        dwell_state_ghost = {
            "condo_occupants": {},
            "bar_table_patrons": [{"id": "ghost_1", "hit_count": 1, "dwell_seconds": 0.2}]
        }
        res_ghost = self.auditor.audit_dwell_tracking(dwell_state_ghost, current_time=2000.0)
        self.assertIn("TRANSIENT_GHOST_TRACK", res_ghost["anomalies"])
        self.assertIn("ghost_1", res_ghost["filtered_ghosts"])

    def test_environmental_cross_validation(self):
        """Cross-validates Home Assistant weather, acoustic rain detection, and road specular reflection."""
        # 1. Consistent rain
        ha_weather_rain = {"weather_condition": "rainy", "is_raining": True}
        acoustic_events_rain = [{"type": "RAIN_DOWNPOUR", "confidence": 0.88}]
        road_eval_wet = {"is_wet": True, "specular_index": 0.65}
        audit_rain = self.auditor.audit_environmental_consistency(ha_weather_rain, acoustic_events_rain, road_eval_wet)
        self.assertTrue(audit_rain["is_consistent"])
        self.assertEqual(len(audit_rain["disagreements"]), 0)

        # 2. Disagreement: Acoustic detects heavy rain downpour, but HA says clear and road is bone-dry
        ha_weather_clear = {"weather_condition": "sunny", "is_raining": False}
        road_eval_dry = {"is_wet": False, "specular_index": 0.05}
        audit_disagree = self.auditor.audit_environmental_consistency(ha_weather_clear, acoustic_events_rain, road_eval_dry)
        self.assertFalse(audit_disagree["is_consistent"])
        self.assertIn("RAIN_SENSOR_DISAGREEMENT", audit_disagree["disagreements"])

    def test_window_luminosity_hysteresis(self):
        """Validates hysteresis preventing car headlight sweep flickers from toggling window state."""
        tracker = WindowOccupancyHysteresis(min_consecutive_hits=3, min_dwell_seconds=1.5)

        # Frame 0: Dark
        self.assertFalse(tracker.update_window("apt_101", raw_illuminated=False, timestamp=0.0))

        # Frame 1: Transient car headlight sweep (single frame bright spike)
        self.assertFalse(tracker.update_window("apt_101", raw_illuminated=True, timestamp=0.5))

        # Frame 2: Back to dark (headlight passed) -> state remains False!
        self.assertFalse(tracker.update_window("apt_101", raw_illuminated=False, timestamp=1.0))

        # Sustained domestic light switched on: 3 consecutive frames over 1.5s
        self.assertFalse(tracker.update_window("apt_101", raw_illuminated=True, timestamp=2.0))
        self.assertFalse(tracker.update_window("apt_101", raw_illuminated=True, timestamp=2.5))
        is_active = tracker.update_window("apt_101", raw_illuminated=True, timestamp=3.5)
        self.assertTrue(is_active)  # Confirmed illuminated after hysteresis threshold


class TestOpticalAlignmentWatchdog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_dir = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")
        cls.zones_path = PROJECT_ROOT / "config" / "zones.json"
        cls.zones = load_zones(str(cls.zones_path))

    def test_architectural_anchor_alignment_and_drift(self):
        """Tests alignment on fixed architectural anchors (Doctor Fit & Condo 723 gate)."""
        watchdog = OpticalAlignmentWatchdog(zones=self.zones)
        ref_img_path = self.val_dir / "clean_v4_upright.jpg"

        if ref_img_path.exists():
            ref_img = Image.open(ref_img_path)
            # Set baseline reference
            watchdog.calibrate_reference(ref_img)

            # Test same image -> Zero drift
            res_same = watchdog.check_alignment(ref_img)
            self.assertTrue(res_same["is_aligned"])
            self.assertLess(res_same["max_drift_px"], 2.0)
            self.assertEqual(len(res_same["warnings"]), 0)

            # Simulate phone physical shift/rotation by cropping and padding (10 px shift)
            shifted_img = Image.new("RGB", ref_img.size, (0, 0, 0))
            shifted_img.paste(ref_img, (15, 12))
            res_shifted = watchdog.check_alignment(shifted_img)
            self.assertFalse(res_shifted["is_aligned"])
            self.assertGreater(res_shifted["max_drift_px"], 5.0)
            self.assertIn("OPTICAL_MOUNT_DRIFT_WARNING", res_shifted["warnings"])

    def test_lens_obscuration_and_dirty_glass(self):
        """Detects sharp loss of high-frequency edge gradient variance from dirty glass or obstruction."""
        watchdog = OpticalAlignmentWatchdog(zones=self.zones)
        ref_img_path = self.val_dir / "clean_v4_upright.jpg"

        if ref_img_path.exists():
            ref_img = Image.open(ref_img_path)
            watchdog.calibrate_reference(ref_img)

            # Clear image -> high sharpness
            clear_check = watchdog.check_sharpness(ref_img)
            self.assertFalse(clear_check["is_obscured"])
            self.assertGreater(clear_check["gradient_variance"], 50.0)

            # Heavily blurred / obscured image (simulating condensation, grease, or obstruction)
            blurred_img = ref_img.filter(ImageFilter.GaussianBlur(radius=8))
            blur_check = watchdog.check_sharpness(blurred_img)
            self.assertTrue(blur_check["is_obscured"])
            self.assertIn("LENS_OBSCURATION_WARNING", blur_check["warnings"])
            self.assertLess(blur_check["sharpness_ratio"], 0.60)


class TestDataQualityScorerAndGroundTruthSampling(unittest.TestCase):
    def setUp(self):
        self.scorer = DataQualityScorer()
        self.audit_dir = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation/quality_audit")

    def test_composite_quality_scoring_matrix(self):
        """Validates quality score calculation across tracking, velocity, optics, and environment."""
        # 1. Perfect system performance
        score_perfect = self.scorer.calculate_quality_score(
            velocity_valid=True,
            dwell_valid=True,
            env_consistent=True,
            optical_aligned=True,
            sharpness_ratio=1.0,
            avg_tracking_conf=0.92
        )
        self.assertGreaterEqual(score_perfect["score"], 90.0)
        self.assertEqual(score_perfect["grade"], "EXCELLENT")

        # 2. Degraded / anomalous performance
        score_degraded = self.scorer.calculate_quality_score(
            velocity_valid=False,     # Velocity spike
            dwell_valid=False,        # Ghost tracks
            env_consistent=False,     # Sensor mismatch
            optical_aligned=False,    # Mount drift
            sharpness_ratio=0.45,     # Obscured lens
            avg_tracking_conf=0.35
        )
        self.assertLess(score_degraded["score"], 50.0)
        self.assertEqual(score_degraded["grade"], "CRITICAL")

    def test_audit_manifest_logging(self):
        """Verifies structured audit manifest is logged to disk on anomalies."""
        engine = QualityAuditEngine(audit_dir=str(self.audit_dir))
        sample_anomalies = ["UNREALISTIC_VELOCITY_SPIKE", "OPTICAL_MOUNT_DRIFT_WARNING"]

        manifest_path = engine.log_audit_event(
            score_data={"score": 42.5, "grade": "CRITICAL"},
            anomalies=sample_anomalies,
            metadata={"frame": "clean_v4_upright.jpg", "trigger": "test"}
        )

        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["score"], 42.5)
        self.assertEqual(data["grade"], "CRITICAL")
        self.assertListEqual(data["anomalies"], sample_anomalies)


class TestQualityAuditAPI(unittest.TestCase):
    """Tests HTTP server endpoint /api/quality/audit."""

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

    def test_api_quality_audit_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/quality/audit"
        req = urllib.request.urlopen(url, timeout=5.0)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))

        self.assertIn("quality_score", data)
        self.assertIn("grade", data)
        self.assertIn("optical_alignment", data)
        self.assertIn("active_anomalies", data)
        self.assertIn("audit_history", data)
        self.assertIsInstance(data["quality_score"], (int, float))

    def test_api_homography_recalibration_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/alignment/homography"
        payload = json.dumps({
            "src_points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "dst_points": [[0.12, 0.09], [0.91, 0.11], [0.88, 0.89], [0.11, 0.88]],
            "save": False
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "success")
            self.assertIn("H", data)
            self.assertIn("warped_zones", data)


class TestHomographyRecalibrator(unittest.TestCase):
    def test_homography_4pts_and_zone_warping(self):
        src = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        dst = [(0.12, 0.09), (0.91, 0.11), (0.88, 0.89), (0.11, 0.88)]
        H = HomographyRecalibrator.compute_homography_4pts(src, dst)
        for s, d in zip(src, dst):
            xp, yp = HomographyRecalibrator.apply_homography(H, s[0], s[1])
            self.assertAlmostEqual(xp, d[0], places=3)
            self.assertAlmostEqual(yp, d[1], places=3)

        sample_zones = {
            "bus_lane": {"polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
            "apartment_windows": {"apt_101": {"polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}}
        }
        warped = HomographyRecalibrator.warp_zones(sample_zones, H)
        self.assertAlmostEqual(warped["bus_lane"]["polygon"][0][0], 0.12, places=3)
        self.assertAlmostEqual(warped["apartment_windows"]["apt_101"]["polygon"][0][1], 0.09, places=3)


if __name__ == "__main__":
    unittest.main()
