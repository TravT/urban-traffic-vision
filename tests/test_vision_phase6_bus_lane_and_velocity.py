#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 6 Bus Lane Compliance & Roadway Velocity Vectors
Verifies:
1. Perspective Homography Ground-Plane Transformation & 25m Baseline Speed Estimator
2. Direction Classification (Eastbound Normal Flow vs Contraflow Safety Alert vs Stopped)
3. Dedicated BRS Bus Lane Compliance & Stopping/Standing Infraction Monitor (>15s)
4. Integration with ADR-31 Kinetic Transit Profile (10 FPS, 4.5 Mbps) & REST API
"""

import os
import sys
import math
import unittest
from pathlib import Path

_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))


class TestPerspectiveVelocityEstimator(unittest.TestCase):
    """Seam 1: Perspective Ground Transformation & Baseline Speed Estimation."""

    def test_ground_coordinate_mapping(self):
        from src.analytics.velocity_estimator import PerspectiveVelocityEstimator

        estimator = PerspectiveVelocityEstimator(baseline_meters=25.0, roadway_width_meters=12.0)

        # Far left entrance (0.0, 0.52) should map near (0.0, 0.0) ground coordinates
        gx0, gy0 = estimator.image_to_ground(0.0, 0.52)
        self.assertAlmostEqual(gx0, 0.0, delta=0.5)
        self.assertAlmostEqual(gy0, 0.0, delta=0.5)

        # Far right exit (0.78, 0.535) should map near (25.0, 0.0) ground coordinates
        gx1, gy1 = estimator.image_to_ground(0.78, 0.535)
        self.assertAlmostEqual(gx1, 25.0, delta=1.0)
        self.assertAlmostEqual(gy1, 0.0, delta=1.0)

        # Near right exit (1.0, 0.98) should map near (25.0, 12.0)
        gx2, gy2 = estimator.image_to_ground(1.0, 0.98)
        self.assertAlmostEqual(gx2, 25.0, delta=1.5)
        self.assertAlmostEqual(gy2, 12.0, delta=1.5)

    def test_velocity_estimation_at_normal_speed_40kmh(self):
        from src.analytics.velocity_estimator import PerspectiveVelocityEstimator

        estimator = PerspectiveVelocityEstimator(baseline_meters=25.0, speed_limit_kmh=50.0)

        # 40 km/h = 11.11 m/s
        # At 10 FPS (Kinetic Profile), vehicle moves 1.111 m per frame
        # In 1.0 second (10 frames, dt = 0.1s), vehicle travels from X = 5.0m to X = 16.11m
        track_id = 401
        for step in range(11):
            t = 100.0 + step * 0.1
            gx = 5.0 + step * 1.111
            gy = 6.0 # middle lane
            norm_x, norm_y = estimator.ground_to_image(gx, gy)
            state = estimator.update_track(track_id=track_id, centroid=(norm_x, norm_y), timestamp=t)

        self.assertIsNotNone(state)
        # Expected speed: 40 km/h within +/- 1.5 km/h tolerance
        self.assertAlmostEqual(state["speed_kmh"], 40.0, delta=1.5)
        self.assertEqual(state["heading"], "Eastbound ->")
        self.assertFalse(state["is_speeding"])
        self.assertFalse(state["is_contraflow"])

    def test_speeding_infraction_detection_70kmh(self):
        from src.analytics.velocity_estimator import PerspectiveVelocityEstimator

        estimator = PerspectiveVelocityEstimator(baseline_meters=25.0, speed_limit_kmh=50.0)

        # 70 km/h = 19.44 m/s (exceeds 50 km/h limit on Barata Ribeiro)
        track_id = 701
        for step in range(8):
            t = 200.0 + step * 0.1
            gx = 3.0 + step * 1.944
            gy = 3.0
            norm_x, norm_y = estimator.ground_to_image(gx, gy)
            state = estimator.update_track(track_id=track_id, centroid=(norm_x, norm_y), timestamp=t)

        self.assertIsNotNone(state)
        self.assertAlmostEqual(state["speed_kmh"], 70.0, delta=2.0)
        self.assertTrue(state["is_speeding"])
        self.assertEqual(state["heading"], "Eastbound ->")

    def test_contraflow_safety_alert_wrong_way(self):
        from src.analytics.velocity_estimator import PerspectiveVelocityEstimator

        estimator = PerspectiveVelocityEstimator(baseline_meters=25.0)

        # Vehicle traveling in reverse / wrong direction along one-way Barata Ribeiro
        track_id = 999
        for step in range(6):
            t = 300.0 + step * 0.1
            gx = 20.0 - step * 1.0  # X decreasing
            gy = 6.0
            norm_x, norm_y = estimator.ground_to_image(gx, gy)
            state = estimator.update_track(track_id=track_id, centroid=(norm_x, norm_y), timestamp=t)

        self.assertIsNotNone(state)
        self.assertEqual(state["heading"], "Contraflow <-")
        self.assertTrue(state["is_contraflow"])


class TestBusLaneComplianceMonitor(unittest.TestCase):
    """Seam 2: Dedicated BRS Bus Lane Compliance & Stopping Infractions."""

    def setUp(self):
        self.bus_lane_poly = [
            [0.620, 0.540],
            [0.780, 0.540],
            [0.990, 0.980],
            [0.680, 0.980]
        ]

    def test_bus_is_fully_compliant(self):
        from src.analytics.bus_lane_monitor import BusLaneComplianceMonitor

        monitor = BusLaneComplianceMonitor(bus_lane_polygon=self.bus_lane_poly, stop_threshold_sec=15.0)

        # A bus stopping at BRS bus stop for 30 seconds
        bus_vehicle = {
            "track_id": 10,
            "class": "bus",
            "centroid": (0.750, 0.700),
            "speed_kmh": 0.0
        }

        # Frame at t = 100.0
        events1 = monitor.update_frame(vehicles=[bus_vehicle], timestamp=100.0)
        self.assertEqual(len(events1), 0)

        # Frame at t = 130.0 (30s stopped)
        events2 = monitor.update_frame(vehicles=[bus_vehicle], timestamp=130.0)
        self.assertEqual(len(events2), 0)
        self.assertEqual(monitor.active_infraction_count, 0)

    def test_private_car_stopping_infraction_over_15s(self):
        from src.analytics.bus_lane_monitor import BusLaneComplianceMonitor

        monitor = BusLaneComplianceMonitor(bus_lane_polygon=self.bus_lane_poly, stop_threshold_sec=15.0)

        car_vehicle = {
            "track_id": 88,
            "class": "car",
            "centroid": (0.750, 0.700), # inside bus lane
            "speed_kmh": 0.0
        }

        # t = 500.0: Car stops in BRS bus lane
        monitor.update_frame(vehicles=[car_vehicle], timestamp=500.0)
        self.assertEqual(monitor.active_infraction_count, 0)

        # t = 510.0: 10s elapsed (< 15s threshold, no infraction yet)
        monitor.update_frame(vehicles=[car_vehicle], timestamp=510.0)
        self.assertEqual(monitor.active_infraction_count, 0)

        # t = 516.0: 16s elapsed (>= 15s threshold -> INFRACTION TRIGGERED)
        events = monitor.update_frame(vehicles=[car_vehicle], timestamp=516.0)
        self.assertEqual(len(events), 1)

        infraction = events[0]
        self.assertEqual(infraction["event"], "BUS_LANE_STOPPING_INFRACTION")
        self.assertEqual(infraction["track_id"], 88)
        self.assertEqual(infraction["vehicle_class"], "car")
        self.assertEqual(infraction["stopped_duration_sec"], 16.0)
        self.assertEqual(monitor.active_infraction_count, 1)

        # t = 525.0: Car drives away / vacates bus lane
        events_vacated = monitor.update_frame(vehicles=[], timestamp=525.0)
        self.assertEqual(len(events_vacated), 1)
        self.assertEqual(events_vacated[0]["event"], "BUS_LANE_INFRACTION_RESOLVED")
        self.assertEqual(events_vacated[0]["total_stopped_sec"], 25.0)
        self.assertEqual(monitor.active_infraction_count, 0)


class TestVelocityAndBusLaneAPI(unittest.TestCase):
    """Seam 3: REST API Server Endpoints for Velocity & Bus Lane Compliance."""

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

    def test_get_traffic_velocity_endpoint(self):
        import json
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/traffic/velocity"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("baseline_meters", data)
            self.assertIn("speed_limit_kmh", data)
            self.assertIn("active_tracks", data)

    def test_get_bus_lane_compliance_endpoint(self):
        import json
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/api/traffic/buslane"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("active_infractions", data)
            self.assertIn("total_infractions_recorded", data)


if __name__ == "__main__":
    unittest.main()
