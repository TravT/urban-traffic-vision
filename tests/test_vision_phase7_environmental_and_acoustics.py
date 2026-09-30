#!/usr/bin/env python3
"""
Unit Tests for Phase 7: Environmental Photometrics & Acoustic Intelligence
Validates:
1. Home Assistant SQLite DB queries and astronomical solar elevation calculation for Copacabana.
2. Photometric ROI luminance extraction, Rioluz streetlamp activation and anomaly detection.
3. Road surface specular puddle / wetness index calculation.
4. Digital acoustic signal processing: RMS, dBFS, and rolling noise floor percentiles (L10, L50, L90, Leq).
5. Urban acoustic event classifier: Emergency Siren, Bus Air Brake, Tire Screech, and Rain Downpour.
6. REST API environmental and acoustic endpoints.
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

from src.config import load_zones
from src.analytics.photometrics import (
    HomeAssistantWeatherGateway,
    PhotometricEngine,
    calculate_solar_elevation
)
from src.analytics.acoustic_analyzer import UrbanAcousticAnalyzer


class TestPhotometricsAndSolar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val_dir = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")
        cls.zones_path = PROJECT_ROOT / "config" / "zones.json"
        cls.zones = load_zones(str(cls.zones_path))
        cls.ha_db = Path("/home/tlima/Enterprise_Hub/data/homeassistant/home-assistant_v2.db")

    def test_copacabana_solar_elevation_calculation(self):
        """Validates astronomical solar elevation for Copacabana (-22.9711, -43.1873)."""
        import datetime
        # Midday: 2026-09-16 15:00 UTC (~12:00 BRT) -> Sun high above horizon
        dt_noon = datetime.datetime(2026, 9, 16, 15, 0, tzinfo=datetime.timezone.utc)
        elev_noon = calculate_solar_elevation(dt_noon, lat=-22.9711, lon=-43.1873)
        self.assertGreater(elev_noon, 50.0)
        self.assertLess(elev_noon, 80.0)

        # Midnight: 2026-09-16 03:00 UTC (~00:00 BRT) -> Sun deep below horizon
        dt_midnight = datetime.datetime(2026, 9, 16, 3, 0, tzinfo=datetime.timezone.utc)
        elev_midnight = calculate_solar_elevation(dt_midnight, lat=-22.9711, lon=-43.1873)
        self.assertLess(elev_midnight, -50.0)

    def test_ha_weather_gateway(self):
        """Tests Home Assistant DB extraction with real DB or fallback."""
        gateway = HomeAssistantWeatherGateway(db_path=str(self.ha_db) if self.ha_db.exists() else None)
        state = gateway.query_ha_state()

        self.assertIn("sun_state", state)
        self.assertIn("solar_elevation", state)
        self.assertIn("weather_condition", state)
        self.assertIn("is_raining", state)
        self.assertIsInstance(state["solar_elevation"], float)
        self.assertIsInstance(state["is_raining"], bool)

    def test_photometric_extraction_day_vs_night(self):
        """Tests luminance extraction on real Day vs Night validation frames."""
        engine = PhotometricEngine(zones=self.zones)

        day_img_path = self.val_dir / "clean_v4_upright.jpg"
        night_img_path = self.val_dir / "live_night_now_upright.jpg"

        if day_img_path.exists() and night_img_path.exists():
            day_img = Image.open(day_img_path)
            night_img = Image.open(night_img_path)

            day_stats = engine.extract_roi_stats(day_img, self.zones["streetlamp_photometric"]["polygon"])
            night_stats = engine.extract_roi_stats(night_img, self.zones["streetlamp_photometric"]["polygon"])

            self.assertGreater(day_stats["mean_v"], 50.0)
            self.assertGreater(night_stats["mean_v"], 50.0)  # Rioluz streetlamp actively illuminates this ROI at night
            self.assertEqual(night_stats["max_v"], 255.0)

    def test_streetlamp_state_and_anomalies(self):
        """Validates normal operation and alerts: lamp failure vs daylight energy waste."""
        engine = PhotometricEngine(zones=self.zones)

        # Normal Night with Lamp ON (elev < 0, mean_v = 140)
        res_normal_night = engine.evaluate_streetlamp_state(
            mean_v=140.0, solar_elevation=-25.0, ambient_facade_v=60.0
        )
        self.assertEqual(res_normal_night["state"], "STREETLAMP_ON")
        self.assertFalse(res_normal_night["is_anomalous"])

        # Night Lamp Failure Alert (elev < -4, streetlamp dark mean_v = 25)
        res_failure = engine.evaluate_streetlamp_state(
            mean_v=25.0, solar_elevation=-25.0, ambient_facade_v=30.0
        )
        self.assertEqual(res_failure["state"], "STREETLAMP_OFF")
        self.assertTrue(res_failure["is_anomalous"])
        self.assertEqual(res_failure["anomaly_type"], "STREETLAMP_OUTAGE_ALERT")

        # Daylight Waste Alert (elev > 15, lamp burning bright with high contrast)
        res_waste = engine.evaluate_streetlamp_state(
            mean_v=240.0, solar_elevation=35.0, ambient_facade_v=120.0
        )
        self.assertTrue(res_waste["is_anomalous"])
        self.assertEqual(res_waste["anomaly_type"], "DAYLIGHT_ENERGY_WASTE_ALERT")

    def test_roadway_wetness_and_specular_reflectance(self):
        """Validates road surface wetness and specular puddle index calculation."""
        engine = PhotometricEngine(zones=self.zones)

        # Test dry synthetic patch (matte grey)
        dry_patch = Image.new("RGB", (200, 200), (80, 80, 80))
        dry_eval = engine.analyze_roadway_wetness(dry_patch, polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertFalse(dry_eval["is_wet"])
        self.assertLess(dry_eval["specular_index"], 0.2)

        # Test wet patch with specular reflection puddle (dark asphalt with high specular saturated white streak)
        wet_patch = Image.new("RGB", (200, 200), (30, 30, 30))
        # Add saturated glare streak
        for y in range(80, 120):
            for x in range(50, 150):
                wet_patch.putpixel((x, y), (250, 250, 255))
        wet_eval = engine.analyze_roadway_wetness(wet_patch, polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertTrue(wet_eval["is_wet"])
        self.assertGreater(wet_eval["specular_index"], 0.3)


class TestAcousticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = UrbanAcousticAnalyzer(sample_rate=16000)

    def test_rms_and_dbfs(self):
        """Verifies RMS and logarithmic decibel calculation."""
        # Full scale sine wave
        samples_full = [math.sin(2 * math.pi * 440 * n / 16000) for n in range(16000)]
        rms = self.analyzer.compute_rms(samples_full)
        db_fs = self.analyzer.compute_db_fs(rms)
        self.assertAlmostEqual(rms, 1.0 / math.sqrt(2), places=2)
        self.assertAlmostEqual(db_fs, -3.01, places=1)

        # Silence
        samples_silent = [0.0] * 1000
        rms_silent = self.analyzer.compute_rms(samples_silent)
        db_silent = self.analyzer.compute_db_fs(rms_silent)
        self.assertLess(db_silent, -80.0)

    def test_rolling_noise_floor_percentiles(self):
        """Validates statistical noise floor metrics: L10, L50, L90, Leq."""
        # Feed varied sound levels: quiet ambient, medium traffic, loud peaks
        for _ in range(30):
            # Quiet background (-45 dB)
            s_quiet = [0.005 * math.sin(2 * math.pi * 200 * n / 16000) for n in range(800)]
            self.analyzer.analyze_chunk(s_quiet)
        for _ in range(50):
            # Normal traffic (-30 dB)
            s_traffic = [0.03 * math.sin(2 * math.pi * 300 * n / 16000) for n in range(800)]
            self.analyzer.analyze_chunk(s_traffic)
        for _ in range(20):
            # Loud vehicle passby (-15 dB)
            s_loud = [0.18 * math.sin(2 * math.pi * 150 * n / 16000) for n in range(800)]
            self.analyzer.analyze_chunk(s_loud)

        summary = self.analyzer.get_noise_summary()
        self.assertIn("l10", summary)
        self.assertIn("l50", summary)
        self.assertIn("l90", summary)
        self.assertIn("leq", summary)
        # L10 (peak 10%) must be louder than L50, which is louder than L90 (floor)
        self.assertGreater(summary["l10"], summary["l50"])
        self.assertGreater(summary["l50"], summary["l90"])

    def test_emergency_siren_detection(self):
        """Simulates 900 Hz emergency vehicle siren wail and tests detection."""
        sr = 16000
        # 1.0s siren chunk at 900 Hz (within 700-1500 Hz siren band)
        siren_chunk = [0.35 * math.sin(2 * math.pi * 900 * n / sr) for n in range(sr)]
        res = self.analyzer.analyze_chunk(siren_chunk)

        event_types = [e["type"] for e in res["detected_events"]]
        self.assertIn("EMERGENCY_SIREN", event_types)

    def test_transit_bus_air_brake_detection(self):
        """Simulates 4.5 kHz pneumatic air brake discharge hiss and tests detection."""
        sr = 16000
        # Air brake is high-frequency pneumatic burst (e.g. 4500 Hz)
        brake_chunk = [0.4 * math.sin(2 * math.pi * 4500 * n / sr) for n in range(int(sr * 0.5))]
        res = self.analyzer.analyze_chunk(brake_chunk)

        event_types = [e["type"] for e in res["detected_events"]]
        self.assertIn("BUS_AIR_BRAKE", event_types)

    def test_tire_screech_detection(self):
        """Simulates 3.2 kHz sudden high-crest-factor tire screech."""
        sr = 16000
        # Background quiet baseline
        for _ in range(10):
            self.analyzer.analyze_chunk([0.01 * math.sin(2 * math.pi * 200 * n / sr) for n in range(800)])

        # Sharp friction surge at 3200 Hz
        screech = [0.6 * math.sin(2 * math.pi * 3200 * n / sr) for n in range(int(sr * 0.4))]
        res = self.analyzer.analyze_chunk(screech)

        event_types = [e["type"] for e in res["detected_events"]]
        self.assertIn("TIRE_SCREECH", event_types)

    def test_rain_downpour_detection(self):
        """Simulates elevated wideband noise from heavy tropical downpour."""
        sr = 16000
        import random
        rng = random.Random(42)
        # Wideband white/pink noise simulation
        rain_chunk = [rng.uniform(-0.15, 0.15) for _ in range(sr)]
        res = self.analyzer.analyze_chunk(rain_chunk)

        event_types = [e["type"] for e in res["detected_events"]]
        self.assertIn("RAIN_DOWNPOUR", event_types)


class TestEnvironmentalAndAcousticAPI(unittest.TestCase):
    """Tests HTTP server endpoints /api/environmental and /api/acoustic."""

    @classmethod
    def setUpClass(cls):
        from http.server import HTTPServer
        import threading
        from src.api.server import VisionAPIHandler

        cls.port = 19098
        cls.server = HTTPServer(("127.0.0.1", cls.port), VisionAPIHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_api_environmental_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/environmental?name=clean_v4_upright.jpg"
        req = urllib.request.urlopen(url, timeout=5.0)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))

        self.assertIn("telemetry", data)
        self.assertIn("streetlamp", data)
        self.assertIn("roadway", data)
        self.assertIn("regime", data)
        self.assertIn("solar_elevation", data["telemetry"])

    def test_api_acoustic_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/acoustic"
        req = urllib.request.urlopen(url, timeout=5.0)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))

        self.assertIn("noise_floor", data)
        self.assertIn("recent_events", data)

    def test_api_acoustic_simulate_endpoint(self):
        import urllib.request
        import json

        url = f"http://127.0.0.1:{self.port}/api/acoustic/simulate"
        payload = json.dumps({"event_type": "siren"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5.0)
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read().decode("utf-8"))

        self.assertIn("detected_events", data)
        events = [e["type"] for e in data["detected_events"]]
        self.assertIn("EMERGENCY_SIREN", events)


if __name__ == "__main__":
    unittest.main()
