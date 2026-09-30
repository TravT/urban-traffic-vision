#!/usr/bin/env python3
"""
Semantic Scene Supervisor (Track B: System 2 Reflective Reasoning)
Pairs with YOLOv8/11 real-time detector (Track A: System 1 Reflexive Counting).

Provides:
1. Zero-shot macro scene classification (Lighting/Weather, Roadway State, Sidewalk Dynamics).
2. Human-readable narrative scene captioning.
3. Multi-modal cross-validation against YOLO detections (specular reflection suppression,
   ghost traffic filtering, environmental solar verification).
4. Dual-mode execution: Native CLIP/MobileCLIP embeddings via fastembed/ONNX when present,
   with deterministic heuristic fallback for zero-dependency test/edge environments.
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

# Default Zero-Shot Prompt Banks for Copacabana (Rua Barata Ribeiro)
SCENE_PROMPT_BANKS = {
    "lighting": {
        "clear_daylight": "Bright clear daylight with sharp building facade contrast and sun",
        "rain_wet_asphalt": "Rainy weather with wet asphalt reflecting streetlights and vehicle headlights",
        "twilight_dusk": "Twilight dusk with overcast cloudy sky and soft diffuse lighting",
        "night_streetlamps": "Nocturnal dark street illuminated by Rioluz public streetlamps"
    },
    "roadway": {
        "flowing_traffic": "Free-flowing roadway traffic with moving cars, buses, and taxis",
        "congested_traffic": "Bumper-to-bumper transit bus traffic jam or slow heavy congestion",
        "empty_street": "Empty quiet street corridor with no moving vehicles",
        "roadway_obstruction": "Roadway blocked by emergency vehicle, roadblock, or municipal service truck"
    },
    "sidewalk": {
        "normal_pedestrians": "Pedestrians walking normally on Portuguese stone sidewalk",
        "crowded_sidewalk": "Crowded sidewalk gathering or dense pedestrian foot traffic",
        "bar_tables_active": "Patrons seated drinking and socializing at outdoor bar tables under red awning",
        "bar_tables_vacant": "Sidewalk bar tables vacant or closed with empty chairs and shutter down"
    }
}


class SemanticSceneSupervisor:
    """
    Asynchronous semantic supervisor for real-time vision appliances.
    Evaluates global FOV context to provide qualitative verification and anomaly detection.
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._clip_image_model = None
        self._clip_text_model = None
        self._backend = "heuristic"

        if self.use_embeddings:
            self._try_load_fastembed()

    def _try_load_fastembed(self):
        """Attempts to load fastembed CLIP ONNX models if available."""
        try:
            from fastembed import ImageEmbedding, TextEmbedding
            # Lightweight ViT-B-32 multimodal model (512-dim, ONNX runtime)
            self._clip_image_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
            self._clip_text_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
            self._backend = "clip_onnx"
        except Exception:
            # Graceful fallback to heuristic engine
            self._clip_image_model = None
            self._clip_text_model = None
            self._backend = "heuristic"

    @property
    def backend(self) -> str:
        return self._backend

    def _extract_image_features(self, img: Image.Image) -> Dict[str, float]:
        """Extracts photometric and spatial color metrics for heuristic scoring."""
        # Downscale for ultra-fast photometric scanning (<5ms)
        small = img.resize((120, 160)).convert("RGB")
        pixels = list(small.getdata())
        n = len(pixels)

        total_r = sum(p[0] for p in pixels)
        total_g = sum(p[1] for p in pixels)
        total_b = sum(p[2] for p in pixels)

        mean_r = total_r / n
        mean_g = total_g / n
        mean_b = total_b / n

        # Value / Luminance (V)
        mean_v = sum(max(p[0], p[1], p[2]) for p in pixels) / n

        # High-brightness specular points (wet road glares, headlights)
        specular_pts = sum(1 for p in pixels if max(p[0], p[1], p[2]) > 210 and (max(p)-min(p)) < 40)
        specular_ratio = specular_pts / n

        # Roadway ROI (bottom half)
        bottom_half = pixels[n // 2:]
        n_bottom = len(bottom_half)
        bottom_v = sum(max(p[0], p[1], p[2]) for p in bottom_half) / n_bottom

        return {
            "mean_v": mean_v,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "bottom_v": bottom_v,
            "specular_ratio": specular_ratio
        }

    def _score_category_heuristic(
        self,
        category: str,
        features: Dict[str, float],
        candidates: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculates normalized probability distribution across candidates via heuristic rules."""
        raw_scores = {}
        v = features["mean_v"]
        spec = features["specular_ratio"]

        if category == "lighting":
            if v > 95.0:
                raw_scores["clear_daylight"] = 0.85
                raw_scores["twilight_dusk"] = 0.10
                raw_scores["rain_wet_asphalt"] = 0.04 if spec < 0.05 else 0.40
                raw_scores["night_streetlamps"] = 0.01
            elif v > 45.0:
                raw_scores["twilight_dusk"] = 0.65
                raw_scores["clear_daylight"] = 0.20
                raw_scores["rain_wet_asphalt"] = 0.10 + spec * 2.0
                raw_scores["night_streetlamps"] = 0.05
            else:
                # Night
                if spec > 0.03:
                    raw_scores["rain_wet_asphalt"] = 0.70
                    raw_scores["night_streetlamps"] = 0.25
                else:
                    raw_scores["night_streetlamps"] = 0.85
                    raw_scores["rain_wet_asphalt"] = 0.10
                raw_scores["clear_daylight"] = 0.02
                raw_scores["twilight_dusk"] = 0.03

        elif category == "roadway":
            # Baseline probabilities with sensible defaults
            raw_scores["flowing_traffic"] = 0.70 if v > 40 else 0.35
            raw_scores["empty_street"] = 0.20 if v > 40 else 0.60
            raw_scores["congested_traffic"] = 0.08
            raw_scores["roadway_obstruction"] = 0.02

        elif category == "sidewalk":
            raw_scores["normal_pedestrians"] = 0.75 if v > 40 else 0.30
            raw_scores["bar_tables_vacant"] = 0.20 if v > 40 else 0.65
            raw_scores["bar_tables_active"] = 0.04
            raw_scores["crowded_sidewalk"] = 0.01

        # Normalize to probability distribution (sum = 1.0)
        total = sum(raw_scores.get(k, 0.01) for k in candidates)
        probs = {k: round(raw_scores.get(k, 0.01) / total, 3) for k in candidates}
        return probs

    def _score_category_clip(
        self,
        image: Image.Image,
        candidates: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculates cosine similarity and softmax probabilities using real CLIP ONNX models."""
        # 1. Embed image
        img_emb = list(self._clip_image_model.embed([image]))[0]

        # 2. Embed text candidates
        labels = list(candidates.keys())
        texts = [candidates[k] for k in labels]
        txt_embs = list(self._clip_text_model.embed(texts))

        # 3. Compute cosine similarities
        sims = []
        for t_emb in txt_embs:
            dot = sum(a * b for a, b in zip(img_emb, t_emb))
            norm_i = math.sqrt(sum(a * a for a in img_emb))
            norm_t = math.sqrt(sum(b * b for b in txt_embs))
            cos_sim = dot / (norm_i * norm_t + 1e-8)
            sims.append(cos_sim)

        # Softmax with temperature 0.05 for sharp classification
        temp = 0.05
        exp_sims = [math.exp(s / temp) for s in sims]
        sum_exp = sum(exp_sims)
        probs = {labels[i]: round(exp_sims[i] / sum_exp, 3) for i in range(len(labels))}
        return probs

    def evaluate_scene(self, image: Image.Image) -> Dict[str, Any]:
        """
        Evaluates the global camera scene across all defined prompt banks.
        Returns top predictions, probabilities, and a synthesized narrative sentence.
        """
        start_t = time.time()
        results = {}
        top_labels = {}

        if self._backend == "clip_onnx" and self._clip_image_model is not None:
            for cat, candidates in SCENE_PROMPT_BANKS.items():
                probs = self._score_category_clip(image, candidates)
                top_k = max(probs, key=probs.get)
                results[cat] = {
                    "top": top_k,
                    "confidence": probs[top_k],
                    "probabilities": probs,
                    "description": candidates[top_k]
                }
                top_labels[cat] = top_k
        else:
            features = self._extract_image_features(image)
            for cat, candidates in SCENE_PROMPT_BANKS.items():
                probs = self._score_category_heuristic(cat, features, candidates)
                top_k = max(probs, key=probs.get)
                results[cat] = {
                    "top": top_k,
                    "confidence": probs[top_k],
                    "probabilities": probs,
                    "description": candidates[top_k]
                }
                top_labels[cat] = top_k

        elapsed_ms = (time.time() - start_t) * 1000.0

        # Synthesize cohesive natural language narrative
        lighting_desc = results["lighting"]["description"].split(" with ")[0]
        road_desc = results["roadway"]["description"].split(" with ")[0]
        sidewalk_desc = results["sidewalk"]["description"].split(" on ")[0]
        narrative = f"{lighting_desc}. {road_desc}. {sidewalk_desc}."

        return {
            "timestamp": time.time(),
            "elapsed_ms": round(elapsed_ms, 2),
            "backend": self._backend,
            "narrative": narrative,
            "scene_state": results,
            "top_summary": {
                "lighting": top_labels.get("lighting"),
                "roadway": top_labels.get("roadway"),
                "sidewalk": top_labels.get("sidewalk")
            }
        }

    def cross_validate_with_yolo(
        self,
        scene_report: Dict[str, Any],
        yolo_detections: List[Dict[str, Any]],
        ha_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cross-validates Track A (YOLO) and Track B (Scene Supervisor) to flag
        anomalies, suppress specular reflections, and audit sensor agreements.
        """
        flags = []
        recommendations = []
        is_consistent = True

        lighting_top = scene_report.get("top_summary", {}).get("lighting")
        roadway_top = scene_report.get("top_summary", {}).get("roadway")

        # 1. Specular Puddle Reflection Check
        # In rain/wet regimes, low-confidence cars located low on the roadway are reflections
        if lighting_top == "rain_wet_asphalt":
            wet_conf = scene_report["scene_state"]["lighting"]["confidence"]
            low_road_cars = [
                d for d in yolo_detections
                if d.get("class") in ["car", "bus", "truck", "motorcycle"]
                and d.get("bbox", [0, 0, 0, 0])[1] > 0.80 # Lower asphalt
                and d.get("confidence", 1.0) < 0.60
            ]
            if low_road_cars:
                flags.append("SPECULAR_PUDDLE_REFLECTION_DETECTED")
                recommendations.append("Elevate roadway vehicle confidence threshold to 0.58 and apply tire-line crop.")
                is_consistent = False

        # 2. Ghost Vehicle / Empty Roadway Sanity Check
        # If scene supervisor strongly asserts empty street (>0.75), but YOLO sees >= 4 moving vehicles
        if roadway_top == "empty_street":
            empty_conf = scene_report["scene_state"]["roadway"]["confidence"]
            moving_vehicles = [
                d for d in yolo_detections
                if d.get("class") in ["car", "bus", "truck"]
                and d.get("speed_kmh", 0) > 10.0
            ]
            if empty_conf > 0.75 and len(moving_vehicles) >= 3:
                flags.append("GHOST_VEHICLE_DISAGREEMENT")
                recommendations.append("Audit YOLO bounding boxes against background static subtraction.")
                is_consistent = False

        # 3. Solar & Environmental Telemetry Verification
        if ha_telemetry:
            solar_elev = ha_telemetry.get("solar_elevation")
            if solar_elev is not None:
                # Night mismatch: Solar elevation < -10 deg (midnight) but scene says clear daylight
                if solar_elev < -10.0 and lighting_top == "clear_daylight":
                    flags.append("SOLAR_LIGHTING_TELEMETRY_MISMATCH")
                    recommendations.append("Check Home Assistant sun.sun sensor sync or time zone offset.")
                    is_consistent = False
                # Day mismatch: Sun high in sky (>25 deg) but scene says night streetlamps
                elif solar_elev > 25.0 and lighting_top == "night_streetlamps":
                    flags.append("SOLAR_LIGHTING_TELEMETRY_MISMATCH")
                    recommendations.append("Audit camera exposure compensation or window shutter state.")
                    is_consistent = False

        # 4. Out-of-Vocabulary Obstruction Check
        if roadway_top == "roadway_obstruction":
            flags.append("URBAN_OBSTRUCTION_EVENT")
            recommendations.append("Dispatch high-priority incident snapshot to Home Assistant & Telegram.")
            is_consistent = False

        return {
            "is_consistent": is_consistent,
            "anomaly_flags": flags,
            "recommendations": recommendations,
            "active_regime": lighting_top,
            "yolo_detection_count": len(yolo_detections),
            "verified_at": time.time()
        }
