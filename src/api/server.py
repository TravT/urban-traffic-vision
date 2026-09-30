#!/usr/bin/env python3
"""
Urban Traffic & Building Vision Appliance — HTTP Server & REST API
Provides Traefik ingress routing on port 9099, serving:
- /live: Live Camera Stream & Telemetry HUD
- /editor: Interactive HTML5 Canvas Zone & Window Crosshair Editor
- /api/zones: GET/POST zones.json with coordinate validation
- /api/frame/calibration: Serves reference frame for crosshair calibration
- /api/occupancy: Real-time & replay apartment occupancy status
- /health: Nomad health check
"""

import os
import sys
import time
import json
import math
import shutil
import threading
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from urllib.parse import urlparse, parse_qs
from src.config import load_settings, load_zones, ValidationError
from src.ingestion.file_replay_adapter import FileReplayAdapter
from src.ingestion.frame_registry import list_calibration_frames, get_frame_path_by_id
from src.analytics.apartment_occupancy import ApartmentOccupancyTracker
from src.analytics.batch_evaluator import SamplingProfiler, BatchEvaluator
from src.analytics.velocity_estimator import PerspectiveVelocityEstimator
from src.analytics.bus_lane_monitor import BusLaneComplianceMonitor
from src.analytics.photometrics import HomeAssistantWeatherGateway, PhotometricEngine
from src.analytics.acoustic_analyzer import UrbanAcousticAnalyzer
from src.analytics.quality_auditor import QualityAuditEngine, OpticalAlignmentWatchdog
from src.analytics.scene_supervisor import SemanticSceneSupervisor
from src.tracking.dwell_time_tracker import UnifiedDwellCoordinator

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
ZONES_PATH = PROJECT_ROOT / "config" / "zones.json"
WEB_DIR = PROJECT_ROOT / "src" / "web"
VAL_DIR = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")


class VisionAPIHandler(BaseHTTPRequestHandler):
    replay_adapter = FileReplayAdapter(data_dir=str(VAL_DIR))
    occupancy_tracker = ApartmentOccupancyTracker()
    ha_gateway = HomeAssistantWeatherGateway()
    acoustic_analyzer = UrbanAcousticAnalyzer(sample_rate=16000)
    quality_engine = QualityAuditEngine()
    alignment_watchdog = OpticalAlignmentWatchdog()
    scene_supervisor = SemanticSceneSupervisor(use_embeddings=False)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: str, content_type: str):
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "healthy", "service": "urban-traffic-vision", "port": 9099})
        elif path in ["/", "/live"]:
            self._send_file(str(WEB_DIR / "live.html"), "text/html; charset=utf-8")
        elif path in ["/editor", "/editor.html"]:
            self._send_file(str(WEB_DIR / "editor.html"), "text/html; charset=utf-8")
        elif path in ["/raw.jpg", "/latest.jpg", "/latest_raw.jpg", "/frame_raw.jpg", "/frame.jpg"]:
            target = VAL_DIR / "latest_raw_upright.jpg"
            if not target.exists():
                target = VAL_DIR / "clean_v4_upright.jpg"
            self._send_file(str(target), "image/jpeg")
        elif path in ["/annotated.jpg", "/frame_annotated.jpg", "/latest_annotated.jpg"]:
            target = VAL_DIR / "latest_annotated.jpg"
            if not target.exists():
                target = VAL_DIR / "latest_raw_upright.jpg"
            if not target.exists():
                target = VAL_DIR / "clean_v4_upright.jpg"
            self._send_file(str(target), "image/jpeg")
        elif path == "/validation_audit.csv":
            self._send_file(str(VAL_DIR / "validation_audit.csv"), "text/csv")
        elif path in ["/api/stats", "/stats"]:
            stats_path = VAL_DIR / "traffic_stats.json"
            telem_path = VAL_DIR / ".telemetry.json"
            res = {}
            if stats_path.exists():
                try:
                    with open(stats_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                except Exception:
                    pass
            elif telem_path.exists():
                try:
                    with open(telem_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                except Exception:
                    pass
            if not res:
                res = {
                    "status": "running",
                    "total_vehicles": 0,
                    "total_pedestrians": 0,
                    "edge_temp_c": 24.0,
                    "last_latency_ms": 0.0,
                    "fps": 0.0,
                    "last_update": time.time()
                }
            self._send_json(res)
        elif path == "/api/zones":
            try:
                zones = load_zones(str(ZONES_PATH))
                self._send_json(zones)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path in ["/api/frames", "/api/frames/list"]:
            try:
                frames = list_calibration_frames(str(VAL_DIR))
                self._send_json(frames)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/frame":
            try:
                frame_name = query.get("name", ["clean_v4_upright.jpg"])[0]
                target_path = get_frame_path_by_id(str(VAL_DIR), frame_name)
                self._send_file(target_path, "image/jpeg")
            except Exception as e:
                self._send_json({"error": str(e)}, 404)
        elif path == "/api/frame/calibration":
            try:
                cal_path = self.replay_adapter.get_calibration_frame_path()
                self._send_file(cal_path, "image/jpeg")
            except Exception as e:
                self._send_json({"error": str(e)}, 404)
        elif path == "/api/profiles":
            try:
                profiles = SamplingProfiler.get_profiles()
                self._send_json(profiles)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/batch/evaluate":
            try:
                zones = load_zones(str(ZONES_PATH))
                evaluator = BatchEvaluator(str(VAL_DIR), zones)
                day_frames = [
                    "clean_v4_upright.jpg", "dropzone_upright.jpg", "window_placement_initial.jpg",
                    "annotated_snapshot_20260914_063059.jpg", "annotated_snapshot_20260914_064506.jpg"
                ]
                night_frames = [
                    "live_night_now_upright.jpg", "latest_raw_upright.jpg", "annotated_snapshot_20260910_040311.jpg",
                    "annotated_snapshot_20260914_034017.jpg"
                ]
                report = evaluator.evaluate_batches(day_frames, night_frames)
                self._send_json(report)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/dwell":
            try:
                zones = load_zones(str(ZONES_PATH))
                coordinator = UnifiedDwellCoordinator(zones=zones)
                payload = coordinator.process_frame(detections=[], timestamp=time.time())
                self._send_json(payload)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/traffic/velocity":
            try:
                estimator = PerspectiveVelocityEstimator(baseline_meters=25.0, speed_limit_kmh=50.0)
                self._send_json({
                    "baseline_meters": 25.0,
                    "speed_limit_kmh": 50.0,
                    "active_tracks": list(estimator.current_states.values())
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path in ["/api/traffic/buslane", "/api/buslane/infractions"]:
            try:
                zones = load_zones(str(ZONES_PATH))
                bus_poly = zones.get("bus_lane", {}).get("polygon")
                monitor = BusLaneComplianceMonitor(bus_lane_polygon=bus_poly, stop_threshold_sec=15.0)
                self._send_json(monitor.get_summary())
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/occupancy":
            try:
                zones = load_zones(str(ZONES_PATH))
                frame_name = query.get("name", query.get("frame", [None]))[0]
                if frame_name:
                    cal_path = get_frame_path_by_id(str(VAL_DIR), frame_name)
                else:
                    cal_path = self.replay_adapter.get_calibration_frame_path()
                img = Image.open(cal_path)
                report = self.occupancy_tracker.analyze_frame(img, zones)
                self._send_json(report)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/environmental":
            try:
                zones = load_zones(str(ZONES_PATH))
                engine = PhotometricEngine(zones=zones, ha_gateway=self.ha_gateway)
                frame_name = query.get("name", query.get("frame", [None]))[0]
                if frame_name:
                    cal_path = get_frame_path_by_id(str(VAL_DIR), frame_name)
                else:
                    cal_path = self.replay_adapter.get_calibration_frame_path()
                img = Image.open(cal_path)
                report = engine.evaluate_environmental_state(img)
                self._send_json(report)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/acoustic":
            try:
                noise_summary = self.acoustic_analyzer.get_noise_summary()
                recent_events = self.acoustic_analyzer.get_recent_events(limit=15)
                self._send_json({
                    "sample_rate": self.acoustic_analyzer.sample_rate,
                    "noise_floor": noise_summary,
                    "recent_events": recent_events
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path in ["/api/quality/audit", "/api/quality"]:
            try:
                frame_name = query.get("name", query.get("frame", [None]))[0]
                if frame_name:
                    cal_path = get_frame_path_by_id(str(VAL_DIR), frame_name)
                else:
                    cal_path = self.replay_adapter.get_calibration_frame_path()
                img = Image.open(cal_path)
                align_res = self.alignment_watchdog.check_alignment(img)
                sharp_res = self.alignment_watchdog.check_sharpness(img)

                summary = self.quality_engine.get_audit_summary()
                summary["optical_alignment"] = align_res
                summary["sharpness"] = sharp_res
                if align_res.get("warnings") or sharp_res.get("warnings"):
                    combined = summary.get("active_anomalies", []) + align_res.get("warnings", []) + sharp_res.get("warnings", [])
                    summary["active_anomalies"] = list(set(combined))
                self._send_json(summary)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/control":
            try:
                ctrl_path = VAL_DIR / "control.json"
                cur_ctrl = {"state": "running", "mode": "continuous"}
                if ctrl_path.exists():
                    try:
                        with open(ctrl_path, "r", encoding="utf-8") as f:
                            cur_ctrl.update(json.load(f))
                    except Exception:
                        pass
                self._send_json(cur_ctrl)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/scene":
            try:
                frame_name = query.get("name", query.get("frame", [None]))[0]
                if frame_name:
                    cal_path = get_frame_path_by_id(str(VAL_DIR), frame_name)
                else:
                    cal_path = self.replay_adapter.get_calibration_frame_path()
                img = Image.open(cal_path)
                report = self.scene_supervisor.evaluate_scene(img)

                # Cross-validate against local HA telemetry
                ha_state = self.ha_gateway.query_ha_state()
                cross_val = self.scene_supervisor.cross_validate_with_yolo(report, [], ha_telemetry=ha_state)
                report["cross_validation"] = cross_val
                self._send_json(report)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/zones":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                new_zones = json.loads(body)

                # Validate coordinates before writing
                temp_path = str(ZONES_PATH) + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(new_zones, f, indent=2)

                validated = load_zones(temp_path)
                os.replace(temp_path, str(ZONES_PATH))
                self._send_json({"status": "success", "zones": validated})
            except ValidationError as ve:
                self._send_json({"error": f"Validation failed: {ve}"}, 400)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif self.path in ["/api/acoustic/simulate", "/api/acoustic/test"]:
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                data = json.loads(body) if body else {}
                event_type = data.get("event_type", "siren").lower()

                sr = self.acoustic_analyzer.sample_rate
                import random
                rng = random.Random()
                if "siren" in event_type:
                    samples = [0.4 * math.sin(2 * math.pi * 900 * n / sr) for n in range(sr)]
                elif "air" in event_type or "brake" in event_type:
                    samples = [0.45 * math.sin(2 * math.pi * 4500 * n / sr) for n in range(int(sr * 0.5))]
                elif "screech" in event_type or "tire" in event_type:
                    samples = [0.65 * math.sin(2 * math.pi * 3200 * n / sr) for n in range(int(sr * 0.4))]
                elif "rain" in event_type:
                    samples = [rng.uniform(-0.15, 0.15) for _ in range(sr)]
                elif "honk" in event_type:
                    samples = [0.5 * math.sin(2 * math.pi * 500 * n / sr) for n in range(int(sr * 0.5))]
                else:
                    samples = [0.01 * math.sin(2 * math.pi * 200 * n / sr) for n in range(800)]

                result = self.acoustic_analyzer.analyze_chunk(samples)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif self.path == "/api/control":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                data = json.loads(body) if body else {}
                ctrl_path = VAL_DIR / "control.json"
                cur_ctrl = {"state": "running", "mode": "continuous", "start_time": time.time(), "duration_sec": 1200}
                if ctrl_path.exists():
                    try:
                        with open(ctrl_path, "r", encoding="utf-8") as f:
                            cur_ctrl.update(json.load(f))
                    except Exception:
                        pass
                action = data.get("action")
                if action == "start_20min":
                    cur_ctrl["state"] = "running"
                    cur_ctrl["mode"] = "20min"
                    cur_ctrl["start_time"] = time.time()
                    cur_ctrl["duration_sec"] = 1200
                elif action == "start":
                    cur_ctrl["state"] = "running"
                    cur_ctrl["mode"] = "continuous"
                elif action == "stop":
                    cur_ctrl["state"] = "paused"
                elif action == "reset":
                    cur_ctrl["reset_requested"] = True
                tmp_ctrl = str(ctrl_path) + ".tmp"
                with open(tmp_ctrl, "w", encoding="utf-8") as f:
                    json.dump(cur_ctrl, f, indent=2)
                os.replace(tmp_ctrl, str(ctrl_path))
                self._send_json({"success": True, "state": cur_ctrl})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif self.path == "/api/alignment/homography":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                data = json.loads(body) if body else {}
                src_pts = [tuple(p) for p in data.get("src_points", [])]
                dst_pts = [tuple(p) for p in data.get("dst_points", [])]
                apply_save = data.get("save", False)

                from src.analytics.quality_auditor import HomographyRecalibrator
                H = HomographyRecalibrator.compute_homography_4pts(src_pts, dst_pts)
                zones = load_zones(str(ZONES_PATH))
                warped_zones = HomographyRecalibrator.warp_zones(zones, H)

                if apply_save:
                    temp_path = str(ZONES_PATH) + ".tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(warped_zones, f, indent=2)
                    os.replace(temp_path, str(ZONES_PATH))

                self._send_json({"status": "success", "H": H, "warped_zones": warped_zones, "saved": apply_save})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Silence default HTTP access logs to keep stdout clean
        pass


def start_background_ingestion(val_dir: Path, target: str = "100.115.165.41:5555", interval_sec: float = 4.0):
    def worker():
        val_dir.mkdir(parents=True, exist_ok=True)
        raw_upright_path = val_dir / "latest_raw_upright.jpg"
        raw_path = val_dir / "latest_raw.jpg"
        tmp_local = val_dir / ".incoming_frame.jpg"
        tmp_upright = val_dir / ".upright_tmp.jpg"
        ctrl_path = val_dir / "control.json"
        snap_script = (
            "am force-stop com.termux.api >/dev/null 2>&1; sleep 0.4; "
            "am start -n com.termux.api/.activities.TermuxAPILauncherActivity >/dev/null 2>&1; "
            "sleep 1.0; su -c '/data/data/com.termux/files/usr/bin/termux-camera-photo -c 0 /sdcard/live_stream.jpg'; "
            "input keyevent KEYCODE_SLEEP >/dev/null 2>&1"
        )
        print("[Urban Vision Appliance] Native ADB background S20 FE ingestion worker active.", flush=True)
        while True:
            try:
                # 0. Check control state (default: running)
                cur_state = "running"
                if ctrl_path.exists():
                    try:
                        with open(ctrl_path, "r", encoding="utf-8") as f:
                            cur_state = json.load(f).get("state", "running")
                    except Exception:
                        pass
                if cur_state == "paused":
                    time.sleep(2.0)
                    continue

                # 1. Thermal guardrail check via native ADB
                t_res = subprocess.run(
                    ["adb", "-s", target, "shell", "cat", "/sys/class/power_supply/battery/temp"],
                    capture_output=True, text=True, timeout=5
                )
                temp_c = 25.0
                if t_res.returncode == 0 and t_res.stdout.strip().isdigit():
                    temp_c = int(t_res.stdout.strip()) / 10.0
                if temp_c >= 40.0:
                    print(f"[Ingestion] Thermal guardrail tripped: S20 FE battery is {temp_c:.1f}°C (>=40.0°C). Pausing 60s.", flush=True)
                    time.sleep(60)
                    continue

                # 2. Trigger optical capture via foreground TermuxAPI
                c_res = subprocess.run(
                    ["adb", "-s", target, "shell", snap_script],
                    capture_output=True, text=True, timeout=15
                )

                # 3. Pull frame via native ADB (4.5 MB/s over Tailscale loopback)
                p_res = subprocess.run(
                    ["adb", "-s", target, "pull", "/sdcard/live_stream.jpg", str(tmp_local)],
                    capture_output=True, text=True, timeout=10
                )
                if p_res.returncode == 0 and tmp_local.exists() and tmp_local.stat().st_size > 10000:
                    with Image.open(tmp_local) as raw_img:
                        upright_img = raw_img.rotate(270, expand=True)
                        upright_img.save(str(tmp_upright), format="JPEG", quality=88)
                        os.replace(str(tmp_upright), str(raw_upright_path))
                        shutil.copyfile(str(raw_upright_path), str(raw_path))
            except Exception:
                pass
            time.sleep(interval_sec)

    t = threading.Thread(target=worker, daemon=True, name="S20-ADB-Ingestion")
    t.start()


def run_server(port: int = 9099, enable_ingestion: bool = True):
    if enable_ingestion:
        start_background_ingestion(VAL_DIR)
    server = HTTPServer(("0.0.0.0", port), VisionAPIHandler)
    print(f"[Urban Vision Server] Listening on http://0.0.0.0:{port}...")
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9099
    run_server(port)
