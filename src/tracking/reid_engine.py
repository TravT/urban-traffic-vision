#!/usr/bin/env python3
"""
Pedestrian Re-Identification (Re-ID) & HSV Signature Extraction Engine
Computes 64-dimensional normalized spatial-color embeddings from pedestrian crops
and evaluates cosine similarity matching without heavy deep-learning overhead.
"""

import math
from typing import Dict, Any, List
from PIL import Image, ImageStat


class PedestrianFeatureExtractor:
    """
    Extracts deterministic color-spatial feature embeddings from pedestrian crops.
    Subdivides crops into torso (upper body) and lower body regions,
    computes binned HSV histograms, identifies dominant garment colors,
    and calculates cosine similarity distances.
    """

    def __init__(self, h_bins: int = 16, s_bins: int = 8, v_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def _compute_histogram(self, region_img: Image.Image) -> List[float]:
        """Computes a 32-bin normalized HSV histogram for an image region."""
        if region_img.width == 0 or region_img.height == 0:
            return [0.0] * (self.h_bins + self.s_bins + self.v_bins)

        hsv_img = region_img.convert("HSV")
        pixels = list(hsv_img.getdata())
        n_pixels = len(pixels)
        if n_pixels == 0:
            return [0.0] * (self.h_bins + self.s_bins + self.v_bins)

        h_hist = [0] * self.h_bins
        s_hist = [0] * self.s_bins
        v_hist = [0] * self.v_bins

        for h, s, v in pixels:
            h_idx = min(self.h_bins - 1, int(h * self.h_bins / 256.0))
            s_idx = min(self.s_bins - 1, int(s * self.s_bins / 256.0))
            v_idx = min(self.v_bins - 1, int(v * self.v_bins / 256.0))

            h_hist[h_idx] += 1
            s_hist[s_idx] += 1
            v_hist[v_idx] += 1

        # Normalize counts by total pixels
        hist = (
            [count / n_pixels for count in h_hist] +
            [count / n_pixels for count in s_hist] +
            [count / n_pixels for count in v_hist]
        )
        return hist

    @staticmethod
    def classify_dominant_color(mean_h: float, mean_s: float, mean_v: float) -> str:
        """Classifies primary color temperature and tone from mean HSV."""
        if mean_v < 45.0:
            return "dark"
        if mean_s < 28.0 and mean_v > 175.0:
            return "white"
        if mean_s < 30.0:
            return "gray"

        # Pillow Hue range [0..255]
        if 8.0 <= mean_h <= 32.0:
            return "orange"
        if 33.0 <= mean_h <= 55.0:
            return "yellow"
        if 56.0 <= mean_h <= 110.0:
            return "green"
        if 111.0 <= mean_h <= 175.0:
            return "blue"
        if 176.0 <= mean_h <= 230.0:
            return "purple"
        return "red"

    def extract_signature(self, crop: Image.Image) -> Dict[str, Any]:
        """
        Extracts 64-D L2-normalized feature signature and dominant garment color.
        Region 1 (Upper Torso): y in [0.15 * H, 0.55 * H]
        Region 2 (Lower Body):  y in [0.55 * H, 0.90 * H]
        """
        w, h = crop.size
        if w < 4 or h < 8:
            empty_vec = [0.0] * 64
            return {
                "vector": empty_vec,
                "dominant_color": "unknown",
                "upper_hsv": [0.0, 0.0, 0.0],
                "lower_hsv": [0.0, 0.0, 0.0]
            }

        # Torso crop (excluding head/neck collar and waist)
        torso_box = (
            max(0, int(0.15 * w)),
            max(0, int(0.15 * h)),
            min(w, int(0.85 * w)),
            min(h, int(0.55 * h))
        )
        torso_crop = crop.crop(torso_box)

        # Lower body crop (shorts / pants / legs)
        lower_box = (
            max(0, int(0.15 * w)),
            max(0, int(0.55 * h)),
            min(w, int(0.85 * w)),
            min(h, int(0.90 * h))
        )
        lower_crop = crop.crop(lower_box)

        torso_hist = self._compute_histogram(torso_crop)
        lower_hist = self._compute_histogram(lower_crop)

        combined_vec = torso_hist + lower_hist  # 32 + 32 = 64 dimensions

        # L2-normalization: ||v|| = 1.0
        sum_sq = sum(x * x for x in combined_vec)
        norm = math.sqrt(sum_sq)
        if norm > 0.0:
            norm_vec = [round(x / norm, 5) for x in combined_vec]
        else:
            norm_vec = [0.0] * 64

        # Compute mean statistics
        torso_stat = ImageStat.Stat(torso_crop.convert("HSV"))
        mean_th, mean_ts, mean_tv = torso_stat.mean

        lower_stat = ImageStat.Stat(lower_crop.convert("HSV"))
        mean_lh, mean_ls, mean_lv = lower_stat.mean

        dominant_color = self.classify_dominant_color(mean_th, mean_ts, mean_tv)

        return {
            "vector": norm_vec,
            "dominant_color": dominant_color,
            "upper_hsv": [round(mean_th, 2), round(mean_ts, 2), round(mean_tv, 2)],
            "lower_hsv": [round(mean_lh, 2), round(mean_ls, 2), round(mean_lv, 2)]
        }

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two feature vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    @classmethod
    def cosine_distance(cls, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine distance (1.0 - similarity)."""
        return round(1.0 - cls.cosine_similarity(vec1, vec2), 5)
