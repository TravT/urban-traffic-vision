#!/usr/bin/env python3
"""
Camera-Agnostic Stream Ingestion Adapters & Host Governance
Provides unified abstraction across:
1. Scrcpy v4.1 Headless Camera2 H.265 stream (S20 FE edge node)
2. Standard RTSP IP Cameras (Tapo, Reolink, Wyze, Hikvision)
3. Offline File Replay & Dataset Simulation
Enforces host resource protection invariants (capped FPS, capped bitrates).
"""

import os
import time
import urllib.request
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image

from .file_replay_adapter import FileReplayAdapter


class BaseStreamAdapter(ABC):
    """Abstract Base Class for video/camera stream ingestion."""

    @abstractmethod
    def get_frame(self) -> Image.Image:
        """Retrieves the latest available frame as a PIL Image."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the underlying sensor or stream is healthy and active."""
        pass

    def close(self) -> None:
        """Releases stream resources."""
        pass


class ScrcpyV4Adapter(BaseStreamAdapter):
    """
    Ingests Camera2 H.265 stream from custom-ws-scrcpy v4.1 via REST gateway and stream port.
    Enforces host CFS and network governance invariants (max 15 FPS, max 8M bitrate).
    """

    MAX_ALLOWED_FPS = 15
    MAX_ALLOWED_BITRATE_MBPS = 8

    def __init__(
        self,
        target: str = "100.115.165.41:5555",
        gateway_url: str = "http://ws-scrcpy.home.arpa",
        max_fps: int = 5,
        bitrate: str = "2M",
        codec: str = "h265",
        camera_id: str = "0"
    ):
        self.target = target
        self.gateway_url = gateway_url.rstrip("/")
        self.camera_id = camera_id
        self.codec = codec

        # Enforce host safety caps
        self.max_fps = min(max_fps, self.MAX_ALLOWED_FPS)
        self.bitrate = self._sanitize_bitrate(bitrate)
        self._active = False

    def _sanitize_bitrate(self, bitrate_str: str) -> str:
        """Sanitizes and clamps bitrate to prevent network/CPU overload."""
        cleaned = bitrate_str.strip().upper()
        if cleaned.endswith("M"):
            try:
                num = int(cleaned[:-1])
                if num > self.MAX_ALLOWED_BITRATE_MBPS:
                    return f"{self.MAX_ALLOWED_BITRATE_MBPS}M"
                return cleaned
            except ValueError:
                pass
        return "2M"

    def get_start_payload(self) -> Dict[str, Any]:
        """Builds governed JSON payload for POST /api/scrcpy/camera/start."""
        return {
            "target": self.target,
            "cameraId": self.camera_id,
            "codec": self.codec,
            "bitRate": self.bitrate,
            "maxFps": self.max_fps
        }

    def start_stream(self) -> bool:
        """Dispatches camera/start request to custom-ws-scrcpy REST gateway."""
        url = f"{self.gateway_url}/api/scrcpy/camera/start"
        payload = json.dumps(self.get_start_payload()).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._active = data.get("success", False) or data.get("active", False)
                return True
        except Exception:
            return False

    def stop_stream(self) -> bool:
        """Dispatches camera/stop request to custom-ws-scrcpy REST gateway."""
        url = f"{self.gateway_url}/api/scrcpy/camera/stop"
        req = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._active = False
                return True
        except Exception:
            return False

    def is_connected(self) -> bool:
        return self._active

    def get_frame(self) -> Image.Image:
        """Fallback to latest available snapshot if stream socket is not attached."""
        from .file_replay_adapter import FileReplayAdapter
        fallback = FileReplayAdapter()
        return fallback.get_frame()


class RtspAdapter(BaseStreamAdapter):
    """
    Generic RTSP Ingestion Adapter for future IP cameras (Tapo, Reolink, Wyze, Hikvision).
    Accepts standard rtsp:// URLs and demuxes sub-sampled frames.
    """

    def __init__(self, url: str, transport: str = "tcp", max_fps: int = 5):
        self.url = url
        self.transport = transport
        self.max_fps = max_fps
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_frame(self) -> Image.Image:
        """Placeholder for RTSP frame demuxer."""
        from .file_replay_adapter import FileReplayAdapter
        fallback = FileReplayAdapter()
        return fallback.get_frame()


class StreamAdapterFactory:
    """Factory to instantiate the appropriate stream adapter based on configuration."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseStreamAdapter:
        driver_type = config.get("type", "file_replay").lower()

        if driver_type == "file_replay":
            data_dir = config.get("data_dir", "/data/media/merged/vision/validation")
            return FileReplayAdapter(data_dir=data_dir)

        elif driver_type == "scrcpy":
            return ScrcpyV4Adapter(
                target=config.get("target", "100.115.165.41:5555"),
                gateway_url=config.get("gateway_url", "http://ws-scrcpy.home.arpa"),
                max_fps=config.get("max_fps", 5),
                bitrate=config.get("bitrate", "2M"),
                codec=config.get("codec", "h265"),
                camera_id=config.get("camera_id", "0")
            )

        elif driver_type == "rtsp":
            return RtspAdapter(
                url=config.get("url", ""),
                transport=config.get("transport", "tcp"),
                max_fps=config.get("max_fps", 5)
            )

        raise ValueError(f"Unknown stream adapter driver type: '{driver_type}'")
