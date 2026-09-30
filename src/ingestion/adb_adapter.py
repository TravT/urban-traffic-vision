#!/usr/bin/env python3
"""
Native ADB Gateway Ingestion Adapter for Samsung Galaxy S20 FE
Implements ADR-27 (Unified Homelab ADB Gateway via 127.0.0.1:5037)
"""

import os
import shlex
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ThermalGuardrailException(Exception):
    """Raised when edge device battery temperature hits or exceeds safety ceiling."""
    pass


class ADBAdapter:
    """
    Communicates directly with the centralized ADB daemon on the Dell host.
    Requires no SSH keys, wrappers, or authentication loops.
    """

    def __init__(
        self,
        target: str = "100.115.165.41:5555",
        timeout: int = 15,
        thermal_ceiling: float = 40.0
    ):
        self.target = target
        self.timeout = timeout
        self.thermal_ceiling = thermal_ceiling

    def _run_cmd(self, args: list, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        to = timeout or self.timeout
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=to
        )

    def is_connected(self) -> bool:
        """Checks if the edge device is attached and authorized in the ADB daemon."""
        res = self._run_cmd(["adb", "devices", "-l"], timeout=10)
        if res.returncode != 0:
            return False
        for line in res.stdout.splitlines():
            if self.target in line and "device" in line and not "offline" in line:
                return True
        return False

    def get_battery_temperature(self) -> float:
        """
        Queries battery temperature from kernel sysfs.
        Value is returned in tenths of a degree Celsius (e.g. 261 = 26.1°C).
        Enforces ADR-23/ADR-25 thermal safety guardrail (<40.0°C).
        """
        cmd = ["adb", "-s", self.target, "shell", "cat", "/sys/class/power_supply/battery/temp"]
        res = self._run_cmd(cmd, timeout=5)
        if res.returncode != 0 or not res.stdout.strip():
            raise RuntimeError(f"Failed to query battery temp via ADB: {res.stderr}")

        raw_val = int(res.stdout.strip())
        temp_c = raw_val / 10.0

        if temp_c >= self.thermal_ceiling:
            raise ThermalGuardrailException(
                f"Thermal circuit-breaker tripped! Battery temp is {temp_c:.1f}°C (ceiling: {self.thermal_ceiling:.1f}°C)"
            )

        return temp_c

    def sleep_display(self) -> bool:
        """Sends KEYCODE_SLEEP to ensure screen stays dark and cool."""
        cmd = ["adb", "-s", self.target, "shell", "input", "keyevent", "KEYCODE_SLEEP"]
        res = self._run_cmd(cmd, timeout=5)
        return res.returncode == 0

    def capture_frame(self, dest_path: str) -> bool:
        """
        Triggers optical frame capture via Termux API with foreground elevation,
        pulls the high-resolution frame via ADB, and immediately sleeps the screen.
        """
        remote_tmp = "/sdcard/.snap_traffic.jpg"
        capture_shell = (
            "su -c 'am start -n com.termux.api/.activities.TermuxAPILauncherActivity "
            f"&& sleep 0.8 && /data/data/com.termux/files/usr/bin/termux-camera-photo -c 0 {remote_tmp} "
            "&& input keyevent KEYCODE_SLEEP'"
        )

        # 1. Trigger camera capture
        c_res = self._run_cmd(["adb", "-s", self.target, "shell", capture_shell], timeout=20)
        if c_res.returncode != 0:
            logger.error("Capture command failed: %s", c_res.stderr)
            return False

        # 2. High-speed binary pull
        p_res = self._run_cmd(["adb", "-s", self.target, "pull", remote_tmp, dest_path], timeout=15)
        if p_res.returncode != 0:
            logger.error("ADB pull failed: %s", p_res.stderr)
            return False

        # 3. Defensive screen sleep
        self.sleep_display()
        return True
