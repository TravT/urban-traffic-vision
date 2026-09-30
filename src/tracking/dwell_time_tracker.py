#!/usr/bin/env python3
"""
Condominium 723 Gate Virtual Tripwire & Sidewalk Bar Dwell Time Tracking Engine
Tracks pedestrian entrance/exit cycles, calculates dwell time duration,
categorizes visits (courier vs. resident), and monitors sidewalk bar table occupancy.
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple

from src.tracking.reid_engine import PedestrianFeatureExtractor


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Standard ray-casting algorithm to test if (x, y) is inside a normalized polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class CondoGateDwellTracker:
    """
    Virtual tripwire and state machine for Condominium 723 social gate.
    Maintains active occupant registry with Re-ID feature signatures,
    calculates dwell time duration, and classifies visits.
    """

    DEFAULT_GATE_POLY = [
        [0.640, 0.500],
        [0.800, 0.505],
        [0.795, 0.680],
        [0.635, 0.675]
    ]

    def __init__(
        self,
        gate_polygon: Optional[List[List[float]]] = None,
        min_reid_similarity: float = 0.70
    ):
        self.gate_polygon = gate_polygon or self.DEFAULT_GATE_POLY
        self.min_reid_similarity = min_reid_similarity
        self.active_occupants: Dict[int, Dict[str, Any]] = {}
        self.completed_stays: List[Dict[str, Any]] = []

    @property
    def active_occupant_count(self) -> int:
        return len(self.active_occupants)

    def update(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        signature: Optional[List[float]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Updates tracking state for a subject near the gate.
        Triggers ENTERED_CONDO when centroid crosses into gate boundary.
        """
        ts = timestamp if timestamp is not None else time.time()
        cx, cy = centroid
        is_inside = point_in_polygon(cx, cy, self.gate_polygon)

        if is_inside:
            if track_id not in self.active_occupants:
                self.active_occupants[track_id] = {
                    "entry_time": ts,
                    "last_seen": ts,
                    "signature": signature or [],
                    "centroid": centroid
                }
                return {
                    "event": "ENTERED_CONDO",
                    "track_id": track_id,
                    "timestamp": ts,
                    "centroid": centroid
                }
            else:
                self.active_occupants[track_id]["last_seen"] = ts
                if signature:
                    self.active_occupants[track_id]["signature"] = signature

        return {}

    def record_exit(
        self,
        centroid: Tuple[float, float],
        signature: Optional[List[float]] = None,
        timestamp: Optional[float] = None,
        track_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Matches an exiting subject to the active occupant registry using Re-ID cosine similarity.
        Calculates dwell time and classifies the visit.
        """
        ts = timestamp if timestamp is not None else time.time()
        if not self.active_occupants:
            return {}

        best_match_id = None
        best_sim = 0.0

        if track_id is not None and track_id in self.active_occupants:
            best_match_id = track_id
            best_sim = 1.0
        elif signature:
            for oid, data in self.active_occupants.items():
                stored_sig = data.get("signature", [])
                if stored_sig:
                    sim = PedestrianFeatureExtractor.cosine_similarity(signature, stored_sig)
                    if sim > best_sim:
                        best_sim = sim
                        best_match_id = oid

        # If no strong similarity match but only 1 occupant exists, associate directly
        if best_match_id is None and len(self.active_occupants) == 1:
            best_match_id = next(iter(self.active_occupants.keys()))
            best_sim = 0.75

        if best_match_id is not None:
            occupant = self.active_occupants.pop(best_match_id)
            entry_time = occupant["entry_time"]
            duration = max(0.0, ts - entry_time)

            # Classify dwell time category
            if duration < 300.0:        # < 5 minutes
                category = "courier_delivery"
            elif duration <= 1800.0:    # 5 to 30 minutes
                category = "short_visit"
            else:                       # > 30 minutes
                category = "extended_stay"

            stay_record = {
                "event": "EXITED_CONDO",
                "track_id": best_match_id,
                "dwell_duration_sec": round(duration, 1),
                "dwell_category": category,
                "match_confidence": round(best_sim, 3),
                "entry_time": entry_time,
                "exit_time": ts,
                "centroid": centroid
            }
            self.completed_stays.append(stay_record)
            return stay_record

        return {}


class SidewalkBarDwellTracker:
    """
    Monitors seated occupancy and dwell time for sidewalk bar tables under the red awning.
    Implements tolerance window to sustain occupancy through brief occlusions or stillness.
    """

    DEFAULT_TABLE_ZONE = [
        [0.750, 0.530],
        [0.980, 0.525],
        [0.990, 0.685],
        [0.760, 0.690]
    ]

    def __init__(
        self,
        table_zones: Optional[Dict[str, List[List[float]]]] = None,
        vacant_tolerance_sec: float = 15.0
    ):
        self.table_zones = table_zones or {"MESA_01": self.DEFAULT_TABLE_ZONE}
        self.vacant_tolerance_sec = vacant_tolerance_sec
        self.tables_state: Dict[str, Dict[str, Any]] = {}

        for tid in self.table_zones.keys():
            self.tables_state[tid] = {
                "state": "VACANT",
                "session_start": 0.0,
                "last_occupied_time": 0.0,
                "current_dwell_sec": 0.0,
                "patron_count": 0,
                "active_track_ids": []
            }

    def get_table_status(self, table_id: str) -> Dict[str, Any]:
        """Returns the current state dictionary for a specified table."""
        return self.tables_state.get(table_id, {
            "state": "VACANT",
            "patron_count": 0,
            "session_start": 0.0,
            "current_dwell_sec": 0.0
        })

    def update_frame(
        self,
        patrons: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates detected patrons against all table polygon zones.
        Emits TABLE_OCCUPIED or TABLE_VACATED events.
        """
        ts = timestamp if timestamp is not None else time.time()
        events = []

        for tid, poly in self.table_zones.items():
            matched_patrons = []
            for p in patrons:
                cx, cy = p.get("centroid", (0.0, 0.0))
                if point_in_polygon(cx, cy, poly):
                    matched_patrons.append(p)

            t_state = self.tables_state[tid]
            current_count = len(matched_patrons)

            if current_count > 0:
                t_state["last_occupied_time"] = ts
                t_state["patron_count"] = current_count
                t_state["active_track_ids"] = [p.get("track_id") for p in matched_patrons if "track_id" in p]

                if t_state["state"] == "VACANT":
                    t_state["state"] = "OCCUPIED"
                    t_state["session_start"] = ts
                    t_state["current_dwell_sec"] = 0.0
                    events.append({
                        "event": "TABLE_SEATED",
                        "table_id": tid,
                        "patron_count": current_count,
                        "timestamp": ts
                    })
                else:
                    # Table remains occupied; increment dwell duration
                    t_state["current_dwell_sec"] = max(0.0, round(ts - t_state["session_start"], 1))

            else:
                # No patrons detected inside table boundary in this frame
                if t_state["state"] == "OCCUPIED":
                    # Check if tolerance duration has expired
                    elapsed_vacant = ts - t_state["last_occupied_time"]
                    if elapsed_vacant > self.vacant_tolerance_sec:
                        total_dwell = max(0.0, round(t_state["last_occupied_time"] - t_state["session_start"], 1))
                        t_state["state"] = "VACANT"
                        t_state["patron_count"] = 0
                        t_state["current_dwell_sec"] = 0.0
                        t_state["active_track_ids"] = []

                        events.append({
                            "event": "TABLE_VACATED",
                            "table_id": tid,
                            "total_dwell_sec": total_dwell,
                            "timestamp": ts
                        })
                    else:
                        # Inside tolerance window; keep state occupied
                        t_state["current_dwell_sec"] = max(0.0, round(ts - t_state["session_start"], 1))

        return events


class UnifiedDwellCoordinator:
    """
    Coordinates simultaneous tracking of Condominium 723 Gate and Sidewalk Bar Tables.
    Generates unified telemetry payload for MQTT egress and local dashboards.
    """

    def __init__(self, zones: Dict[str, Any]):
        gate_poly = zones.get("condo_723_gate", {}).get("polygon", CondoGateDwellTracker.DEFAULT_GATE_POLY)
        bar_poly = zones.get("bar_tables", {}).get("polygon", SidewalkBarDwellTracker.DEFAULT_TABLE_ZONE)

        self.gate_tracker = CondoGateDwellTracker(gate_polygon=gate_poly)
        self.bar_tracker = SidewalkBarDwellTracker(table_zones={"MESA_01": bar_poly})
        self.feature_extractor = PedestrianFeatureExtractor()

    def process_frame(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes detections for the current frame, dispatching to gate and bar trackers.
        """
        ts = timestamp if timestamp is not None else time.time()
        pedestrians = []

        for d in detections:
            cls_name = d.get("class", "unknown").lower()
            if cls_name == "person":
                # Compute centroid from normalized bounding box [x1, y1, x2, y2]
                bbox = d.get("bbox_norm")
                if bbox and len(bbox) == 4:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                else:
                    cx, cy = d.get("centroid", (0.0, 0.0))

                ped = {
                    "track_id": d.get("track_id", 0),
                    "centroid": (cx, cy),
                    "conf": d.get("conf", 0.5)
                }
                pedestrians.append(ped)

                # Update gate tracker
                self.gate_tracker.update(
                    track_id=ped["track_id"],
                    centroid=(cx, cy),
                    timestamp=ts
                )

        # Update bar tracker
        bar_events = self.bar_tracker.update_frame(patrons=pedestrians, timestamp=ts)

        # Tally active patrons across all bar tables
        active_patrons = sum(
            t["patron_count"] for t in self.bar_tracker.tables_state.values()
        )

        return {
            "timestamp": ts,
            "condo_gate": {
                "active_condo_occupants": self.gate_tracker.active_occupant_count,
                "completed_visits": len(self.gate_tracker.completed_stays)
            },
            "sidewalk_bar": {
                "active_patrons_total": active_patrons,
                "tables": {
                    tid: self.bar_tracker.get_table_status(tid)
                    for tid in self.bar_tracker.table_zones.keys()
                },
                "recent_events": bar_events
            }
        }
