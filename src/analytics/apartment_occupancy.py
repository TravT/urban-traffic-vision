#!/usr/bin/env python3
"""
Apartment Occupancy & HSV Luminosity / Domestic Color Temperature Analyzer
Implements deterministic Phase 2 building facade activity tracking with zero AI overhead.
"""

from typing import Dict, Any, List
from PIL import Image, ImageStat


class ApartmentOccupancyTracker:
    """
    Evaluates individual window crops and full building facades.
    Extracts interior illumination, domestic color temperature (warm incandescent vs. cold screen),
    and aggregate residential occupancy rates.
    """

    def __init__(
        self,
        luminance_threshold: float = 160.0,
        dark_threshold: float = 80.0
    ):
        self.luminance_threshold = luminance_threshold
        self.dark_threshold = dark_threshold

    def analyze_window(self, window_img: Image.Image, window_id: str) -> Dict[str, Any]:
        """Analyzes a cropped window image in HSV color space."""
        if window_img.width == 0 or window_img.height == 0:
            return {
                "id": window_id,
                "is_illuminated": False,
                "mean_v": 0.0,
                "mean_h": 0.0,
                "mean_s": 0.0,
                "color_temp": "dark",
                "confidence": 0.0
            }

        hsv_img = window_img.convert("HSV")
        stat = ImageStat.Stat(hsv_img)
        mean_h, mean_s, mean_v = stat.mean

        if mean_v >= self.luminance_threshold:
            is_illuminated = True
            # In Pillow, H is scaled 0..255 (representing 0..360 deg)
            # Domestic warm incandescent/amber: ~20-50 deg -> Pillow H ~12..40
            # Cold blue/fluorescent/screen: ~180-240 deg -> Pillow H ~125..175
            if 10.0 <= mean_h <= 55.0 and mean_s >= 30.0:
                color_temp = "warm"
            else:
                color_temp = "cold"
            confidence = min(1.0, mean_v / 255.0)
        elif mean_v < self.dark_threshold:
            is_illuminated = False
            color_temp = "dark"
            confidence = max(0.0, 1.0 - (mean_v / self.dark_threshold))
        else:
            # Twilight / ambient spillover zone
            is_illuminated = False
            color_temp = "dark"
            confidence = 0.5

        return {
            "id": window_id,
            "is_illuminated": is_illuminated,
            "mean_v": round(mean_v, 2),
            "mean_h": round(mean_h, 2),
            "mean_s": round(mean_s, 2),
            "color_temp": color_temp,
            "confidence": round(confidence, 2)
        }

    def analyze_frame(self, frame: Image.Image, zones: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts all configured window polygons from the frame, evaluates each crop,
        and computes aggregate residential occupancy and commercial studio status.
        """
        w, h = frame.size
        window_results = []
        res_active = 0
        res_total = 0
        commercial_active = False

        windows = zones.get("apartment_windows", [])
        for win in windows:
            win_id = win.get("id", "unknown")
            poly = win.get("polygon", [])
            if len(poly) < 3:
                continue

            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            box = (
                max(0, int(min(xs) * w)),
                max(0, int(min(ys) * h)),
                min(w, int(max(xs) * w)),
                min(h, int(max(ys) * h))
            )

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            crop = frame.crop(box)
            res = self.analyze_window(crop, win_id)
            res["label"] = win.get("label", win_id)
            window_results.append(res)

            # Segregate commercial (e.g. Doctor Fit) from residential apartments
            is_commercial = "doctor_fit" in win_id.lower() or "commercial" in win_id.lower()
            if is_commercial:
                if res["is_illuminated"]:
                    commercial_active = True
            else:
                res_total += 1
                if res["is_illuminated"]:
                    res_active += 1

        occupancy_rate = round((res_active / res_total * 100.0), 1) if res_total > 0 else 0.0

        return {
            "windows": window_results,
            "active_residential_count": res_active,
            "total_residential_count": res_total,
            "occupancy_rate_pct": occupancy_rate,
            "commercial_active": commercial_active
        }
