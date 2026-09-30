#!/usr/bin/env python3
"""
Batch Validation & Adaptive Environmental Sensitivity Engine (ADR-31)
Evaluates real calibration frames across Day & Night batches,
calculates adaptive confidence thresholds, and evaluates sampling profiles.
"""

import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image, ImageStat

from src.analytics.apartment_occupancy import ApartmentOccupancyTracker


class EnvironmentalRegimeClassifier:
    """
    Classifies the optical/meteorological regime and dynamically computes
    model confidence thresholds based on ADR-31.
    """

    def classify(
        self,
        solar_elevation: float = 0.0,
        weather_condition: str = "clear",
        mean_v: float = 100.0
    ) -> Dict[str, Any]:
        weather_norm = weather_condition.lower()

        # Regime 1: Tropical Rain / Storm (wet asphalt reflection suppression)
        if any(w in weather_norm for w in ["rain", "storm", "pouring", "wet"]):
            regime = "rain"
            vehicle_conf = 0.55  # Elevated to suppress puddle mirror reflections
            person_conf = 0.42
            puddle_filter = True
            bloom_filter = False
            window_eval_mode = "absolute_v"
            window_v_thresh = 160.0
            window_delta_v_thresh = 0.0

        # Regime 2: Nocturnal / Low Twilight (streetlamps & headlight bloom)
        elif solar_elevation <= 0.0 or mean_v < 40.0:
            regime = "night"
            vehicle_conf = 0.52
            person_conf = 0.35   # Compensate for dark shadows under streetlamps
            puddle_filter = False
            bloom_filter = True   # Suppress car headlights blooming into false vehicle boxes
            window_eval_mode = "differential_delta_v"
            window_v_thresh = 160.0
            window_delta_v_thresh = 45.0

        # Regime 3: Clear Daylight
        else:
            regime = "daylight"
            vehicle_conf = 0.45
            person_conf = 0.40
            puddle_filter = False
            bloom_filter = False
            window_eval_mode = "absolute_v"
            window_v_thresh = 160.0
            window_delta_v_thresh = 0.0

        return {
            "regime": regime,
            "vehicle_conf": round(vehicle_conf, 2),
            "person_conf": round(person_conf, 2),
            "window_v_thresh": window_v_thresh,
            "puddle_filter": puddle_filter,
            "bloom_filter": bloom_filter,
            "window_eval_mode": window_eval_mode,
            "window_delta_v_thresh": window_delta_v_thresh
        }


class SamplingProfiler:
    """
    Multi-Framerate Pareto Optimization & Spatial Displacement Mathematics (ADR-31).
    """

    @staticmethod
    def calculate_displacement(speed_kmh: float, fps: float) -> float:
        """
        Calculates spatial displacement between frames in meters:
        Delta_d = (v * 1000) / (3600 * FPS)
        """
        if fps <= 0:
            return 0.0
        return round((speed_kmh * 1000.0) / (3600.0 * fps), 3)

    @staticmethod
    def get_profiles() -> Dict[str, Any]:
        """Returns standard operational profiles defined in ADR-31."""
        return {
            "eco": {
                "fps": 2,
                "bitrate_mbps": 0.8,
                "daily_gb": round(0.8 * 86400 / 8000.0, 2),  # 8.64 GB
                "cpu_load": "< 2.0% of 1 core",
                "primary_targets": "Apartment Windows, Condominium 723 Dwell Time, Sidewalk Bar Tables"
            },
            "balanced": {
                "fps": 5,
                "bitrate_mbps": 2.2,
                "daily_gb": round(2.2 * 86400 / 8000.0, 2),  # 23.76 GB
                "cpu_load": "~4.5% of 1 core",
                "primary_targets": "Pedestrian sidewalk density, bus arrivals, daytime traffic flow"
            },
            "kinetic": {
                "fps": 10,
                "bitrate_mbps": 4.5,
                "daily_gb": round(4.5 * 86400 / 8000.0, 2),  # 48.60 GB
                "cpu_load": "~9.5% of 1 core",
                "primary_targets": "Exact 25m baseline velocity vectors (+/-1.5 km/h), BRS bus lane infractions"
            },
            "burst": {
                "fps": 30,
                "bitrate_mbps": 13.8,
                "daily_gb": round(13.8 * 86400 / 8000.0, 2),  # 149.04 GB
                "cpu_load": "~38% of 1 core",
                "primary_targets": "Short calibration runs, high-speed camera alignment (max 60s)"
            }
        }


class BatchEvaluator:
    """
    Evaluates batches of validation frames, computing photometric statistics,
    environmental regime transitions, building facade occupancy, and sampling recommendations.
    """

    def __init__(self, data_dir: str, zones: Dict[str, Any]):
        self.data_dir = Path(data_dir)
        self.zones = zones
        self.classifier = EnvironmentalRegimeClassifier()
        self.tracker = ApartmentOccupancyTracker()
        self.profiler = SamplingProfiler()

    def evaluate_frame(
        self,
        frame_name: str,
        solar_elevation: Optional[float] = None,
        weather: str = "clear"
    ) -> Dict[str, Any]:
        frame_path = self.data_dir / frame_name
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        img = Image.open(frame_path)
        stat = ImageStat.Stat(img.convert("HSV"))
        mean_h, mean_s, mean_v = stat.mean

        # Automatically determine solar elevation if not explicitly provided
        if solar_elevation is None:
            # V >= 40 indicates daylight/illumination, V < 40 nocturnal
            solar_elevation = 25.0 if mean_v >= 40.0 else -15.0

        regime_meta = self.classifier.classify(
            solar_elevation=solar_elevation,
            weather_condition=weather,
            mean_v=mean_v
        )

        occupancy = self.tracker.analyze_frame(img, self.zones)

        return {
            "id": frame_name,
            "width": img.width,
            "height": img.height,
            "mean_v": round(mean_v, 2),
            "mean_h": round(mean_h, 2),
            "mean_s": round(mean_s, 2),
            "regime": regime_meta["regime"],
            "profile": regime_meta,
            "occupancy": occupancy,
            "windows": occupancy.get("windows", [])
        }

    def evaluate_batches(
        self,
        day_frames: List[str],
        night_frames: List[str]
    ) -> Dict[str, Any]:
        day_results = []
        for f in day_frames:
            if (self.data_dir / f).exists():
                day_results.append(self.evaluate_frame(f, solar_elevation=35.0, weather="clear"))

        night_results = []
        for f in night_frames:
            if (self.data_dir / f).exists():
                night_results.append(self.evaluate_frame(f, solar_elevation=-20.0, weather="clear"))

        # Compute aggregate day statistics
        day_mean_v = (
            sum(r["mean_v"] for r in day_results) / len(day_results)
            if day_results else 0.0
        )
        # Compute aggregate night statistics
        night_mean_v = (
            sum(r["mean_v"] for r in night_results) / len(night_results)
            if night_results else 0.0
        )

        return {
            "day_batch": {
                "frame_count": len(day_results),
                "mean_luminance_v": round(day_mean_v, 2),
                "primary_regime": "daylight",
                "frames": day_results
            },
            "night_batch": {
                "frame_count": len(night_results),
                "mean_luminance_v": round(night_mean_v, 2),
                "primary_regime": "night",
                "frames": night_results
            },
            "sampling_recommendations": {
                "daylight_recommended": "balanced",
                "night_recommended": "eco",
                "profiles": self.profiler.get_profiles()
            }
        }
