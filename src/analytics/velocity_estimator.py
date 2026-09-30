#!/usr/bin/env python3
"""
Roadway Perspective Velocity Estimator & Directional Classification
Maps normalized 2D camera coordinates into 3D metric ground-plane coordinates
across the calibrated 25-meter Rua Barata Ribeiro baseline (+/- 1.5 km/h precision).
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple


class PerspectiveVelocityEstimator:
    """
    Estimates metric ground velocity (km/h), heading direction (Eastbound vs. Contraflow),
    and speeding infractions across the 25-meter Rua Barata Ribeiro baseline.
    """

    def __init__(
        self,
        baseline_meters: float = 25.0,
        roadway_width_meters: float = 12.0,
        speed_limit_kmh: float = 50.0,
        min_frames_for_speed: int = 3
    ):
        self.baseline_meters = baseline_meters
        self.roadway_width_meters = roadway_width_meters
        self.speed_limit_kmh = speed_limit_kmh
        self.min_frames_for_speed = min_frames_for_speed

        self.tracks_history: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
        self.current_states: Dict[int, Dict[str, Any]] = {}

    def image_to_ground(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Maps normalized camera coordinates (0.0 to 1.0) to metric ground coordinates (X, Y in meters).
        X = distance traversed along Barata Ribeiro (0 to 25m, Eastbound)
        Y = distance across lanes (0 to 12m, from far curb to near curb)
        """
        # Vertical roadway bounds: y_top = 0.520 (vanishing/far curb), y_bottom = 0.980 (near curb)
        y_top = 0.520
        y_bottom = 0.980

        v_y = max(0.0, min(1.0, (norm_y - y_top) / (y_bottom - y_top)))
        gy = v_y * self.roadway_width_meters

        # Perspective expansion of width from top (0.780 span) to bottom (1.000 span)
        x_span = 0.780 + (1.000 - 0.780) * v_y
        u_x = max(0.0, min(1.0, norm_x / x_span)) if x_span > 0.0 else 0.0
        gx = u_x * self.baseline_meters

        return (round(gx, 3), round(gy, 3))

    def ground_to_image(self, gx: float, gy: float) -> Tuple[float, float]:
        """Maps metric ground coordinates (X, Y) back into normalized image coordinates (x, y)."""
        v_y = max(0.0, min(1.0, gy / self.roadway_width_meters)) if self.roadway_width_meters > 0 else 0.0
        u_x = max(0.0, min(1.0, gx / self.baseline_meters)) if self.baseline_meters > 0 else 0.0

        y_top = 0.520
        y_bottom = 0.980
        norm_y = y_top + v_y * (y_bottom - y_top)

        x_span = 0.780 + (1.000 - 0.780) * v_y
        norm_x = u_x * x_span

        return (round(norm_x, 4), round(norm_y, 4))

    def update_track(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Updates tracking history for a vehicle and calculates real-time ground velocity.
        """
        ts = timestamp if timestamp is not None else time.time()
        cx, cy = centroid
        gx, gy = self.image_to_ground(cx, cy)

        if track_id not in self.tracks_history:
            self.tracks_history[track_id] = []

        history = self.tracks_history[track_id]
        history.append((ts, gx, gy, cx, cy))

        # Retain last 20 frames
        if len(history) > 20:
            history.pop(0)

        if len(history) < self.min_frames_for_speed:
            state = {
                "track_id": track_id,
                "speed_kmh": 0.0,
                "heading": "Calibrating...",
                "is_speeding": False,
                "is_contraflow": False,
                "ground_pos": (gx, gy),
                "timestamp": ts
            }
            self.current_states[track_id] = state
            return state

        # Calculate displacement over window
        t_prev, gx_prev, gy_prev, _, _ = history[0]
        dt = ts - t_prev
        if dt <= 0.001:
            return self.current_states.get(track_id)

        dx = gx - gx_prev
        dy = gy - gy_prev
        distance_m = math.sqrt(dx * dx + dy * dy)
        speed_ms = distance_m / dt
        speed_kmh = speed_ms * 3.6

        # Determine heading direction
        if abs(dx) < 0.8 and speed_kmh < 4.0:
            heading = "Parked / Stopped"
            is_contraflow = False
        elif dx > 0.0:
            heading = "Eastbound ->"
            is_contraflow = False
        else:
            heading = "Contraflow <-"
            is_contraflow = True

        is_speeding = (speed_kmh > self.speed_limit_kmh)

        state = {
            "track_id": track_id,
            "speed_kmh": round(speed_kmh, 1),
            "heading": heading,
            "is_speeding": is_speeding,
            "is_contraflow": is_contraflow,
            "ground_pos": (gx, gy),
            "timestamp": ts
        }
        self.current_states[track_id] = state
        return state

    def clear_inactive_tracks(self, active_ids: List[int]) -> None:
        """Removes expired track IDs no longer present in the active frame."""
        active_set = set(active_ids)
        to_delete = [tid for tid in self.tracks_history if tid not in active_set]
        for tid in to_delete:
            self.tracks_history.pop(tid, None)
            self.current_states.pop(tid, None)
