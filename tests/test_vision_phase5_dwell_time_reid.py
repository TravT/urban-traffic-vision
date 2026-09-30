#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 5 Dwell Time Tracking & Re-ID
Verifies:
1. Pedestrian Feature Extractor (HSV Torso/Legs Histograms & Cosine Distance)
2. Condominium 723 Gate Virtual Tripwire & Dwell Time State Machine
3. Sidewalk Bar Table Occupancy & Seated Dwell Time Monitor
4. Unified Dwell Coordinator Integration with Authentic 12MP Sensor Crops
"""

import os
import sys
import math
import unittest
from pathlib import Path
from PIL import Image

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))

REAL_VISION_DIR = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")


class TestPedestrianFeatureExtractor(unittest.TestCase):
    """Seam 1: Pedestrian Crop Feature Extraction & Cosine Similarity Re-ID."""

    def test_extract_feature_vector_from_authentic_mrv_orange_crop(self):
        from src.tracking.reid_engine import PedestrianFeatureExtractor

        extractor = PedestrianFeatureExtractor()
        crop_path = REAL_VISION_DIR / "crop_v4_person_orange.jpg"
        self.assertTrue(crop_path.exists(), f"Missing authentic test crop: {crop_path}")

        img = Image.open(crop_path)
        sig = extractor.extract_signature(img)

        self.assertIn("vector", sig)
        self.assertIn("dominant_color", sig)
        self.assertIn("upper_hsv", sig)
        self.assertIn("lower_hsv", sig)

        # Vector should be 64-dimensional and L2-normalized
        vec = sig["vector"]
        self.assertEqual(len(vec), 64)
        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=3)

        # Authentic MRV shirt is vivid orange (Hue in Pillow HSV ~ 10-35)
        self.assertIn(sig["dominant_color"], ["orange", "warm_amber"])

    def test_cosine_similarity_matching(self):
        from src.tracking.reid_engine import PedestrianFeatureExtractor

        extractor = PedestrianFeatureExtractor()
        crop_path = REAL_VISION_DIR / "crop_v4_person_orange.jpg"
        img1 = Image.open(crop_path)

        sig1 = extractor.extract_signature(img1)

        # 1. Exact match against self
        sim_self = extractor.cosine_similarity(sig1["vector"], sig1["vector"])
        self.assertAlmostEqual(sim_self, 1.0, places=4)

        # 2. Slight crop shift / resize should maintain very high similarity (> 0.85)
        w, h = img1.size
        img1_shifted = img1.crop((2, 2, w - 2, h - 2))
        sig1_shifted = extractor.extract_signature(img1_shifted)
        sim_shifted = extractor.cosine_similarity(sig1["vector"], sig1_shifted["vector"])
        self.assertGreater(sim_shifted, 0.85)

        # 3. Comparing orange shirt against a blue or dark pedestrian crop should have low similarity (< 0.50)
        img_blue = Image.new("RGB", (200, 300), (30, 80, 200))
        sig_blue = extractor.extract_signature(img_blue)
        sim_blue = extractor.cosine_similarity(sig1["vector"], sig_blue["vector"])
        self.assertLess(sim_blue, 0.50)


class TestCondoGateDwellTracker(unittest.TestCase):
    """Seam 2: Condominium 723 Social Gate Tripwire & Dwell Time Calculation."""

    def test_gate_entry_and_exit_cycle(self):
        from src.tracking.dwell_time_tracker import CondoGateDwellTracker

        gate_poly = [
            [0.640, 0.500],
            [0.800, 0.505],
            [0.795, 0.680],
            [0.635, 0.675]
        ]
        tracker = CondoGateDwellTracker(gate_polygon=gate_poly)

        # Mock synthetic pedestrian signature for orange-shirt subject
        synthetic_sig = [0.1] * 64
        norm = math.sqrt(sum(x * x for x in synthetic_sig))
        norm_sig = [x / norm for x in synthetic_sig]

        # T0: Subject enters gate at t = 1000.0 (centroid inside gate polygon)
        event1 = tracker.update(
            track_id=101,
            centroid=(0.700, 0.580),
            signature=norm_sig,
            timestamp=1000.0
        )
        self.assertEqual(event1.get("event"), "ENTERED_CONDO")
        self.assertEqual(event1.get("track_id"), 101)
        self.assertEqual(tracker.active_occupant_count, 1)

        # T1: 10 minutes later (t = 1600.0), subject leaves gate threshold
        # Slightly perturbed signature (> 0.85 similarity)
        exiting_sig = [x * 0.98 for x in norm_sig]
        exiting_norm = math.sqrt(sum(x * x for x in exiting_sig))
        exiting_vec = [x / exiting_norm for x in exiting_sig]

        event2 = tracker.record_exit(
            centroid=(0.600, 0.580), # moved outside gate threshold
            signature=exiting_vec,
            timestamp=1600.0
        )
        self.assertEqual(event2.get("event"), "EXITED_CONDO")
        self.assertEqual(event2.get("dwell_duration_sec"), 600.0)
        self.assertEqual(event2.get("dwell_category"), "short_visit")
        self.assertGreater(event2.get("match_confidence"), 0.95)
        self.assertEqual(tracker.active_occupant_count, 0)

    def test_courier_delivery_dwell_classification(self):
        from src.tracking.dwell_time_tracker import CondoGateDwellTracker

        tracker = CondoGateDwellTracker()
        sig = [1.0] + [0.0] * 63

        # Enter at t = 2000.0
        tracker.update(track_id=202, centroid=(0.700, 0.580), signature=sig, timestamp=2000.0)

        # Exits 120 seconds later (courier drop-off)
        exit_event = tracker.record_exit(centroid=(0.500, 0.500), signature=sig, timestamp=2120.0)
        self.assertEqual(exit_event.get("dwell_duration_sec"), 120.0)
        self.assertEqual(exit_event.get("dwell_category"), "courier_delivery")


class TestSidewalkBarDwellTracker(unittest.TestCase):
    """Seam 3: Sidewalk Bar Table Occupancy & Seated Dwell Monitor."""

    def test_table_seated_occupancy_and_vacation(self):
        from src.tracking.dwell_time_tracker import SidewalkBarDwellTracker

        table_poly = [
            [0.750, 0.530],
            [0.980, 0.525],
            [0.990, 0.685],
            [0.760, 0.690]
        ]
        bar_tracker = SidewalkBarDwellTracker(
            table_zones={"MESA_01": table_poly},
            vacant_tolerance_sec=10.0
        )

        # Initially table is VACANT
        state0 = bar_tracker.get_table_status("MESA_01")
        self.assertEqual(state0["state"], "VACANT")
        self.assertEqual(state0["patron_count"], 0)

        # t = 100.0: Two patrons take seats at MESA_01
        patrons_t100 = [
            {"track_id": 11, "centroid": (0.800, 0.600)},
            {"track_id": 12, "centroid": (0.850, 0.620)}
        ]
        bar_tracker.update_frame(patrons=patrons_t100, timestamp=100.0)

        state1 = bar_tracker.get_table_status("MESA_01")
        self.assertEqual(state1["state"], "OCCUPIED")
        self.assertEqual(state1["patron_count"], 2)
        self.assertEqual(state1["session_start"], 100.0)

        # t = 130.0: Patrons still seated (stillness, 30s elapsed)
        bar_tracker.update_frame(patrons=patrons_t100, timestamp=130.0)
        state2 = bar_tracker.get_table_status("MESA_01")
        self.assertEqual(state2["state"], "OCCUPIED")
        self.assertEqual(state2["current_dwell_sec"], 30.0)

        # t = 136.0: Patrons leave the table (6s elapsed since last sighting, within 10s tolerance)
        bar_tracker.update_frame(patrons=[], timestamp=136.0)
        self.assertEqual(bar_tracker.get_table_status("MESA_01")["state"], "OCCUPIED")

        # t = 145.0: 15s elapsed without patrons (> 10s tolerance) -> Table VACATED
        events = bar_tracker.update_frame(patrons=[], timestamp=145.0)
        self.assertEqual(bar_tracker.get_table_status("MESA_01")["state"], "VACANT")

        vacated_event = next((e for e in events if e.get("event") == "TABLE_VACATED"), None)
        self.assertIsNotNone(vacated_event)
        self.assertEqual(vacated_event["table_id"], "MESA_01")
        # Session ran from 100 to 130 (last observed patron time)
        self.assertEqual(vacated_event["total_dwell_sec"], 30.0)


class TestUnifiedDwellCoordinator(unittest.TestCase):
    """Seam 4: Unified Coordinator (Condo Gate + Bar Tables + MQTT Schema)."""

    def test_coordinator_processes_detections_and_produces_telemetry(self):
        from src.tracking.dwell_time_tracker import UnifiedDwellCoordinator
        from src.config import load_zones

        zones_path = PROJECT_ROOT / "config" / "zones.json"
        zones = load_zones(str(zones_path))

        coordinator = UnifiedDwellCoordinator(zones=zones)

        # Feed frame with 1 person in condo gate, 2 persons at bar tables
        detections = [
            {"track_id": 51, "class": "person", "conf": 0.85, "bbox_norm": [0.680, 0.550, 0.740, 0.650]},
            {"track_id": 52, "class": "person", "conf": 0.80, "bbox_norm": [0.880, 0.560, 0.920, 0.640]},
            {"track_id": 53, "class": "person", "conf": 0.78, "bbox_norm": [0.880, 0.590, 0.930, 0.670]}
        ]

        payload = coordinator.process_frame(detections=detections, timestamp=500.0)

        self.assertIn("condo_gate", payload)
        self.assertIn("sidewalk_bar", payload)
        self.assertIn("active_condo_occupants", payload["condo_gate"])
        self.assertIn("tables", payload["sidewalk_bar"])
        self.assertEqual(payload["sidewalk_bar"]["active_patrons_total"], 2)


if __name__ == "__main__":
    unittest.main()
