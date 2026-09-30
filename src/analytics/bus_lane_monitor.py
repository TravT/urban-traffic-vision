#!/usr/bin/env python3
"""
Dedicated BRS Bus Lane Compliance & Stopping Infraction Monitor
Detects unauthorized private vehicles stopping or standing in the dedicated
BRS bus lane for longer than the allowable threshold (> 15.0 seconds).
"""

import time
from typing import Dict, Any, List, Optional, Tuple

from src.tracking.dwell_time_tracker import point_in_polygon


class BusLaneComplianceMonitor:
    """
    Monitors vehicle occupancy in the Rua Barata Ribeiro dedicated BRS Bus Lane.
    Buses are strictly compliant. Private cars, trucks, or motorcycles stopped
    in the lane exceeding the 15-second threshold trigger automated infraction events.
    """

    DEFAULT_BUS_LANE_POLY = [
        [0.620, 0.540],
        [0.780, 0.540],
        [0.990, 0.980],
        [0.680, 0.980]
    ]

    def __init__(
        self,
        bus_lane_polygon: Optional[List[List[float]]] = None,
        stop_threshold_sec: float = 15.0,
        permitted_classes: Optional[List[str]] = None
    ):
        self.bus_lane_polygon = bus_lane_polygon or self.DEFAULT_BUS_LANE_POLY
        self.stop_threshold_sec = stop_threshold_sec
        self.permitted_classes = set(permitted_classes or ["bus"])

        self.vehicle_timers: Dict[int, Dict[str, Any]] = {}
        self.active_infractions: Dict[int, Dict[str, Any]] = {}
        self.completed_infractions: List[Dict[str, Any]] = []

    @property
    def active_infraction_count(self) -> int:
        return len(self.active_infractions)

    def update_frame(
        self,
        vehicles: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes detected vehicles in current frame.
        Emits BUS_LANE_STOPPING_INFRACTION when a non-bus vehicle exceeds the stopped threshold.
        """
        ts = timestamp if timestamp is not None else time.time()
        events = []
        current_frame_ids = set()

        for v in vehicles:
            track_id = v.get("track_id", 0)
            cls_name = v.get("class", "unknown").lower()
            cx, cy = v.get("centroid", (0.0, 0.0))
            speed = v.get("speed_kmh", 0.0)

            is_in_lane = point_in_polygon(cx, cy, self.bus_lane_polygon)
            if not is_in_lane:
                continue

            current_frame_ids.add(track_id)

            # Buses are explicitly permitted and compliant
            if cls_name in self.permitted_classes:
                continue

            # Non-compliant class in bus lane (car, truck, van, motorcycle)
            if track_id not in self.vehicle_timers:
                self.vehicle_timers[track_id] = {
                    "class": cls_name,
                    "first_seen": ts,
                    "stop_start": ts if speed < 5.0 else None,
                    "last_seen": ts,
                    "centroid": (cx, cy)
                }
            else:
                v_data = self.vehicle_timers[track_id]
                v_data["last_seen"] = ts
                v_data["centroid"] = (cx, cy)

                if speed < 5.0:
                    if v_data["stop_start"] is None:
                        v_data["stop_start"] = ts
                    stopped_duration = ts - v_data["stop_start"]

                    if stopped_duration >= self.stop_threshold_sec and track_id not in self.active_infractions:
                        infraction = {
                            "event": "BUS_LANE_STOPPING_INFRACTION",
                            "track_id": track_id,
                            "vehicle_class": cls_name,
                            "stopped_duration_sec": round(stopped_duration, 1),
                            "timestamp": ts,
                            "centroid": (cx, cy)
                        }
                        self.active_infractions[track_id] = infraction
                        events.append(infraction)
                else:
                    # Vehicle moving; reset stopped timer
                    v_data["stop_start"] = None

        # Check for departed vehicles that had active infractions
        departed_ids = [tid for tid in self.vehicle_timers if tid not in current_frame_ids]
        for tid in departed_ids:
            v_data = self.vehicle_timers.pop(tid)
            if tid in self.active_infractions:
                self.active_infractions.pop(tid)
                stop_start = v_data.get("stop_start") or v_data["first_seen"]
                total_stopped = max(0.0, ts - stop_start)
                resolution = {
                    "event": "BUS_LANE_INFRACTION_RESOLVED",
                    "track_id": tid,
                    "vehicle_class": v_data["class"],
                    "total_stopped_sec": round(total_stopped, 1),
                    "timestamp": ts
                }
                self.completed_infractions.append(resolution)
                events.append(resolution)

        return events

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate metrics for dashboard and MQTT egress."""
        return {
            "active_infractions_count": len(self.active_infractions),
            "total_infractions_recorded": len(self.completed_infractions) + len(self.active_infractions),
            "active_infractions": list(self.active_infractions.values()),
            "recent_completed": self.completed_infractions[-10:]
        }
