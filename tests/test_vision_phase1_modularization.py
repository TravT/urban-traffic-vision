#!/usr/bin/env python3
"""
TDD Test Suite: Urban Traffic Vision Appliance - Phase 1 Modularization
Verifies:
1. Native ADB Ingestion Adapter (ADR-27)
2. Settings & Zones Schema Validator
3. Daily Archival & Google Drive Cloud Retention Script (ADR-26, ADR-29)
"""

import os
import sys
import json
import yaml
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
_p = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _p if (_p / "src").exists() else _p / "dev" / "urban-traffic-vision"
sys.path.insert(0, str(PROJECT_ROOT))


class TestADBAdapterSeam(unittest.TestCase):
    """Seam 1: Native ADB Gateway Ingestion Adapter (ADR-27)."""

    def setUp(self):
        from src.ingestion.adb_adapter import ADBAdapter, ThermalGuardrailException
        self.ADBAdapter = ADBAdapter
        self.ThermalGuardrailException = ThermalGuardrailException

    @patch("subprocess.run")
    def test_is_connected_returns_true_when_device_attached(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="List of devices attached\n100.115.165.41:5555\tdevice product:r8qxx model:SM_G780G\n"
        )
        adapter = self.ADBAdapter(target="100.115.165.41:5555")
        self.assertTrue(adapter.is_connected())
        mock_run.assert_called_with(
            ["adb", "devices", "-l"],
            capture_output=True,
            text=True,
            timeout=10
        )

    @patch("subprocess.run")
    def test_get_battery_temperature_parses_tenths_celsius(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="261\n")
        adapter = self.ADBAdapter(target="100.115.165.41:5555", thermal_ceiling=40.0)
        temp_c = adapter.get_battery_temperature()
        self.assertEqual(temp_c, 26.1)

    @patch("subprocess.run")
    def test_thermal_guardrail_raises_exception_at_or_above_ceiling(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="402\n")
        adapter = self.ADBAdapter(target="100.115.165.41:5555", thermal_ceiling=40.0)
        with self.assertRaises(self.ThermalGuardrailException) as ctx:
            adapter.get_battery_temperature()
        self.assertIn("Thermal circuit-breaker tripped", str(ctx.exception))

    @patch("subprocess.run")
    def test_capture_frame_executes_adb_command_and_pulls_file(self, mock_run):
        # Mock shell command success and pull success
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),  # adb shell capture
            MagicMock(returncode=0, stdout="1 file pulled"),  # adb pull
            MagicMock(returncode=0, stdout="")   # adb shell sleep display
        ]
        adapter = self.ADBAdapter(target="100.115.165.41:5555")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            dest_file = tf.name

        try:
            success = adapter.capture_frame(dest_file)
            self.assertTrue(success)
            self.assertEqual(mock_run.call_count, 3)
            # Verify no ssh calls exist
            for call_args in mock_run.call_args_list:
                cmd = call_args[0][0]
                self.assertNotIn("ssh", cmd)
                self.assertEqual(cmd[0], "adb")
        finally:
            if os.path.exists(dest_file):
                os.unlink(dest_file)


class TestConfigLoaderSeam(unittest.TestCase):
    """Seam 2: Settings & Zones Configuration Loader."""

    def setUp(self):
        from src.config import load_settings, load_zones, ValidationError
        self.load_settings = load_settings
        self.load_zones = load_zones
        self.ValidationError = ValidationError

    def test_load_production_settings_file(self):
        settings_path = PROJECT_ROOT / "config" / "settings.yaml"
        self.assertTrue(settings_path.exists())
        config = self.load_settings(str(settings_path))

        self.assertEqual(config["app"]["name"], "urban-traffic-vision")
        self.assertEqual(config["app"]["port"], 9099)
        self.assertEqual(config["archival"]["retention_days"], 30)
        self.assertEqual(config["archival"]["compression"], "zstd")
        self.assertEqual(config["archival"]["gdrive_remote"], "gdrive_root:UrbanTrafficVision_Backups")
        self.assertEqual(config["archival"]["cloud_retention_days"], 30)

    def test_load_production_zones_file(self):
        zones_path = PROJECT_ROOT / "config" / "zones.json"
        self.assertTrue(zones_path.exists())
        zones = self.load_zones(str(zones_path))

        self.assertIn("roadway_roi", zones)
        self.assertIn("bus_lane", zones)
        self.assertIn("entrance_vestibule", zones)
        self.assertIn("apartment_windows", zones)
        self.assertGreaterEqual(len(zones["apartment_windows"]), 4)

    def test_zones_validation_rejects_out_of_bounds_coordinates(self):
        invalid_zones = {
            "test_roi": {
                "name": "Invalid ROI",
                "polygon": [[-0.1, 0.5], [1.2, 0.5], [1.0, 1.0], [0.0, 1.0]]
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(invalid_zones, tf)
            invalid_path = tf.name

        try:
            with self.assertRaises(self.ValidationError) as ctx:
                self.load_zones(invalid_path)
            self.assertIn("out of bounds", str(ctx.exception))
        finally:
            os.unlink(invalid_path)


class TestDailyArchivalScriptSeam(unittest.TestCase):
    """Seam 3: Daily Rolling Archival Script (ADR-26, ADR-29)."""

    def setUp(self):
        self.script_path = PROJECT_ROOT / "scripts" / "archive_daily.sh"
        self.assertTrue(self.script_path.exists())
        self.assertTrue(os.access(self.script_path, os.X_OK))

    def test_archive_script_contains_adr29_gdrive_and_zstd_invariants(self):
        content = self.script_path.read_text()
        # Verify Zstandard compression invariants
        self.assertIn("tar -I \"zstd -", content)
        self.assertIn("daily_${TARGET_DATE}.tar.zst", content)
        # Verify ADR-29 GDrive rolling video sync and retention
        self.assertIn("gdrive_root:UrbanTrafficVision_Backups", content)
        self.assertIn("rclone delete --min-age", content)
        self.assertIn("daily_timelapse_${TARGET_DATE}.mp4", content)
        self.assertIn("RETENTION_DAYS=30", content)


if __name__ == "__main__":
    unittest.main()
