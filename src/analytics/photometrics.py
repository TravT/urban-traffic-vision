#!/usr/bin/env python3
"""
Environmental Photometrics & Home Assistant Telemetry Gateway
Part of Phase 7: Urban Traffic & Building Vision Appliance.

Integrates:
1. Astronomical solar elevation calculation for Copacabana (-22.9711, -43.1873).
2. Home Assistant SQLite DB state ingestion (sun.sun, weather.forecast_home).
3. Fast ROI photometric luminance extraction (mean V, variance, max V).
4. Rioluz public streetlamp state tracking and anomaly alerts (outage vs. daylight waste).
5. Roadway asphalt wetness & puddle specular reflectance index.
"""

import os
import math
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageStat

from src.analytics.batch_evaluator import EnvironmentalRegimeClassifier


def calculate_solar_elevation(
    dt_utc: Optional[datetime.datetime] = None,
    lat: float = -22.9711,
    lon: float = -43.1873
) -> float:
    """
    Computes astronomical solar elevation angle in degrees for given UTC time and coordinates.
    Copacabana default: lat -22.9711, lon -43.1873.
    """
    if dt_utc is None:
        dt_utc = datetime.datetime.now(datetime.timezone.utc)
    elif dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)

    day_of_year = dt_utc.timetuple().tm_yday
    fractional_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # Fractional year in radians
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (fractional_hour - 12.0) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar declination angle (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
    )

    # True solar time (minutes)
    time_offset = eqtime + 4.0 * lon
    tst = fractional_hour * 60.0 + time_offset

    # Solar hour angle (degrees to radians)
    ha_deg = (tst / 4.0) - 180.0
    ha_rad = math.radians(ha_deg)

    lat_rad = math.radians(lat)
    sin_elev = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elev_deg = math.degrees(math.asin(sin_elev))
    return round(elev_deg, 2)


class HomeAssistantWeatherGateway:
    """
    Ingests live Home Assistant states directly from the SQLite database
    (/home/tlima/Enterprise_Hub/data/homeassistant/home-assistant_v2.db)
    with zero network overhead and zero authentication tokens.
    """

    DEFAULT_DB = "/home/tlima/Enterprise_Hub/data/homeassistant/home-assistant_v2.db"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else self.DEFAULT_DB

    def query_ha_state(self) -> Dict[str, Any]:
        """Queries sun.sun and weather entities from Home Assistant SQLite DB."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        elev = calculate_solar_elevation(now_utc)
        default_sun_state = "above_horizon" if elev > 0.0 else "below_horizon"

        state_data = {
            "sun_state": default_sun_state,
            "solar_elevation": elev,
            "weather_condition": "clear-night" if elev <= 0.0 else "sunny",
            "temperature_c": 22.0,
            "humidity_pct": 75.0,
            "wind_speed_kmh": 12.0,
            "cloud_coverage_pct": 20.0,
            "is_raining": False,
            "source": "astronomical_fallback"
        }

        if not self.db_path or not os.path.exists(self.db_path):
            return state_data

        try:
            # Use URI connection in read-only immutable mode
            uri = f"file:{self.db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=2.0)
            cur = con.cursor()

            # Query sun.sun
            cur.execute("""
                SELECT s.state
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                WHERE sm.entity_id = 'sun.sun'
                ORDER BY s.last_updated_ts DESC
                LIMIT 1;
            """)
            sun_row = cur.fetchone()
            if sun_row and sun_row[0]:
                state_data["sun_state"] = str(sun_row[0])

            # Query weather
            cur.execute("""
                SELECT s.state, sa.shared_attrs
                FROM states s
                JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                LEFT JOIN state_attributes sa ON s.attributes_id = sa.attributes_id
                WHERE sm.entity_id IN ('weather.forecast_home', 'weather.home')
                ORDER BY s.last_updated_ts DESC
                LIMIT 1;
            """)
            weather_row = cur.fetchone()
            if weather_row:
                w_state = str(weather_row[0]).lower() if weather_row[0] else "sunny"
                state_data["weather_condition"] = w_state
                state_data["is_raining"] = any(k in w_state for k in ["rain", "storm", "pouring", "wet"])
                if weather_row[1]:
                    try:
                        attrs = json.loads(weather_row[1])
                        if "temperature" in attrs:
                            state_data["temperature_c"] = float(attrs["temperature"])
                        if "humidity" in attrs:
                            state_data["humidity_pct"] = float(attrs["humidity"])
                        if "wind_speed" in attrs:
                            state_data["wind_speed_kmh"] = float(attrs["wind_speed"])
                        if "cloud_coverage" in attrs:
                            state_data["cloud_coverage_pct"] = float(attrs["cloud_coverage"])
                    except Exception:
                        pass
            con.close()
            state_data["source"] = "home_assistant_sqlite"
        except Exception:
            # Fallback to defaults on DB lock or transient issue
            pass

        return state_data


class PhotometricEngine:
    """
    Evaluates spatial photometric properties across Rua Barata Ribeiro:
    - Rioluz public streetlamp activation and anomaly detection.
    - Roadway asphalt specular puddle reflectance.
    - Synchronized solar & environmental state classification.
    """

    def __init__(
        self,
        zones: Optional[Dict[str, Any]] = None,
        ha_gateway: Optional[HomeAssistantWeatherGateway] = None
    ):
        self.zones = zones or {}
        self.ha_gateway = ha_gateway or HomeAssistantWeatherGateway()
        self.classifier = EnvironmentalRegimeClassifier()

    def extract_roi_stats(
        self,
        image: Image.Image,
        polygon: List[List[float]]
    ) -> Dict[str, float]:
        """
        Extracts luminance (V channel) statistics inside a normalized polygon ROI.
        Optimized with bounding box cropping to avoid full-resolution iteration.
        """
        w, h = image.size
        pts_px = [(int(x * w), int(y * h)) for x, y in polygon]

        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        min_x, max_x = max(0, min(xs)), min(w - 1, max(xs))
        min_y, max_y = max(0, min(ys)), min(h - 1, max(ys))

        crop_w = max_x - min_x + 1
        crop_h = max_y - min_y + 1
        if crop_w <= 0 or crop_h <= 0:
            return {"mean_v": 0.0, "std_v": 0.0, "max_v": 0.0, "min_v": 0.0, "pixel_count": 0}

        # Crop sub-region
        cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
        hsv_crop = cropped.convert("HSV")
        v_chan = hsv_crop.split()[2]

        # Local mask
        local_pts = [(px - min_x, py - min_y) for px, py in pts_px]
        mask = Image.new("L", (crop_w, crop_h), 0)
        ImageDraw.Draw(mask).polygon(local_pts, outline=1, fill=1)

        # Collect masked pixel values
        v_data = v_chan.getdata()
        m_data = mask.getdata()
        vals = [v for v, m in zip(v_data, m_data) if m == 1]

        if not vals:
            return {"mean_v": 0.0, "std_v": 0.0, "max_v": 0.0, "min_v": 0.0, "pixel_count": 0}

        count = len(vals)
        mean_v = sum(vals) / count
        var_v = sum((v - mean_v) ** 2 for v in vals) / count
        std_v = math.sqrt(var_v)

        return {
            "mean_v": round(mean_v, 2),
            "std_v": round(std_v, 2),
            "max_v": float(max(vals)),
            "min_v": float(min(vals)),
            "pixel_count": count
        }

    def evaluate_streetlamp_state(
        self,
        mean_v: float,
        solar_elevation: float,
        ambient_facade_v: float = 60.0
    ) -> Dict[str, Any]:
        """
        Determines streetlamp activation state and flags anomalies.
        """
        is_night = solar_elevation <= 0.0
        is_deep_night = solar_elevation < -4.0
        is_bright_day = solar_elevation > 15.0

        # At night, an active Rioluz streetlamp cone generates V >= 50.0
        if is_night:
            if mean_v >= 50.0:
                state = "STREETLAMP_ON"
                is_anomalous = False
                anomaly_type = None
            else:
                state = "STREETLAMP_OFF"
                # If deep night and streetlamp zone is dark (e.g. V < 40), flag outage
                if is_deep_night and mean_v < 40.0:
                    is_anomalous = True
                    anomaly_type = "STREETLAMP_OUTAGE_ALERT"
                else:
                    is_anomalous = False
                    anomaly_type = None
        else:
            # Broad daylight: streetlamps should normally be OFF
            # If mean_v is extraordinarily high with extreme contrast (e.g. V > 220 while ambient is 120),
            # public lighting was left on during daylight hours.
            if is_bright_day and mean_v > 200.0 and (mean_v - ambient_facade_v) > 80.0:
                state = "STREETLAMP_ON"
                is_anomalous = True
                anomaly_type = "DAYLIGHT_ENERGY_WASTE_ALERT"
            else:
                state = "STREETLAMP_OFF"
                is_anomalous = False
                anomaly_type = None

        return {
            "state": state,
            "mean_v": round(mean_v, 2),
            "solar_elevation": round(solar_elevation, 2),
            "is_anomalous": is_anomalous,
            "anomaly_type": anomaly_type
        }

    def analyze_roadway_wetness(
        self,
        image: Image.Image,
        polygon: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates road surface wetness and specular puddle reflection index.
        In wet asphalt, specular reflection glares create bright (V > 210),
        low-saturation (S < 60) streaks against dark asphalt.
        """
        if polygon is None:
            polygon = self.zones.get("roadway_roi", {}).get("polygon", [
                [0.0, 0.52], [0.78, 0.535], [1.0, 0.98], [0.0, 0.98]
            ])

        w, h = image.size
        pts_px = [(int(x * w), int(y * h)) for x, y in polygon]

        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        min_x, max_x = max(0, min(xs)), min(w - 1, max(xs))
        min_y, max_y = max(0, min(ys)), min(h - 1, max(ys))

        crop_w = max_x - min_x + 1
        crop_h = max_y - min_y + 1
        if crop_w <= 0 or crop_h <= 0:
            return {"is_wet": False, "specular_index": 0.0, "road_condition": "DRY"}

        cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
        hsv_crop = cropped.convert("HSV")
        h_chan, s_chan, v_chan = hsv_crop.split()

        local_pts = [(px - min_x, py - min_y) for px, py in pts_px]
        mask = Image.new("L", (crop_w, crop_h), 0)
        ImageDraw.Draw(mask).polygon(local_pts, outline=1, fill=1)

        v_data = v_chan.getdata()
        s_data = s_chan.getdata()
        m_data = mask.getdata()

        total_pts = 0
        specular_pts = 0
        for v, s, m in zip(v_data, s_data, m_data):
            if m == 1:
                total_pts += 1
                # Specular glare: high brightness and low saturation
                if v >= 210 and s <= 60:
                    specular_pts += 1

        if total_pts == 0:
            return {"is_wet": False, "specular_index": 0.0, "road_condition": "DRY"}

        specular_ratio = specular_pts / float(total_pts)
        # Specular index normalized: >= 5% saturated puddles indicates wet roadway
        specular_index = min(1.0, round(specular_ratio * 10.0, 3))
        is_wet = specular_index >= 0.25

        road_condition = "WET_SPECULAR" if specular_index >= 0.40 else ("DAMP" if is_wet else "DRY")

        return {
            "is_wet": is_wet,
            "specular_index": specular_index,
            "specular_ratio": round(specular_ratio, 4),
            "road_condition": road_condition
        }

    def evaluate_environmental_state(
        self,
        image: Image.Image,
        ha_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete environmental photometrics audit, linking Home Assistant
        solar/weather inputs with image photometrics and ADR-31 regime classification.
        """
        if ha_telemetry is None:
            ha_telemetry = self.ha_gateway.query_ha_state()

        solar_elev = ha_telemetry.get("solar_elevation", 0.0)
        weather_cond = ha_telemetry.get("weather_condition", "clear")

        # Streetlamp ROI stats
        lamp_poly = self.zones.get("streetlamp_photometric", {}).get("polygon", [
            [0.440, 0.520], [0.560, 0.520], [0.550, 0.630], [0.430, 0.630]
        ])
        lamp_stats = self.extract_roi_stats(image, lamp_poly)
        lamp_eval = self.evaluate_streetlamp_state(
            mean_v=lamp_stats["mean_v"],
            solar_elevation=solar_elev
        )

        # Roadway wetness
        road_eval = self.analyze_roadway_wetness(image)
        if ha_telemetry.get("is_raining", False):
            road_eval["is_wet"] = True
            if road_eval["road_condition"] == "DRY":
                road_eval["road_condition"] = "DAMP"

        # Overall image mean V for regime classification
        w, h = image.size
        # Sample center region for fast scene mean V
        center_crop = image.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
        scene_mean_v = ImageStat.Stat(center_crop.convert("HSV")).mean[2]

        # ADR-31 Adaptive Regime Classification
        regime_eval = self.classifier.classify(
            solar_elevation=solar_elev,
            weather_condition=weather_cond,
            mean_v=scene_mean_v
        )

        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "telemetry": ha_telemetry,
            "streetlamp": lamp_eval,
            "roadway": road_eval,
            "photometrics": {
                "scene_mean_v": round(scene_mean_v, 2),
                "streetlamp_roi": lamp_stats
            },
            "regime": regime_eval
        }
