#!/usr/bin/env python3
"""
Urban Acoustic Intelligence & Sound Signature Engine (Copacabana Profile)
Part of Phase 7: Urban Traffic & Building Vision Appliance.

Analyzes raw audio streams (PCM 16-bit 16kHz/48kHz) to compute:
1. RMS amplitude and logarithmic decibel levels (dBFS).
2. Rolling statistical percentiles (L10 peak, L50 median, L90 noise floor, Leq continuous).
3. Continuous Butterworth IIR bandpass energy distribution:
   - Low band: 30 - 250 Hz (Heavy bus engines, diesel rumbling)
   - Mid band: 250 - 2000 Hz (Traffic hum, vehicle horns, emergency sirens)
   - High band: 2000 - 7500 Hz (Brake squeals, pneumatic air releases, rain splash)
4. Signature classification:
   - EMERGENCY_SIREN (700 - 1500 Hz wail/yelp)
   - BUS_AIR_BRAKE (3000 - 6000 Hz pneumatic discharge hiss)
   - TIRE_SCREECH (2500 - 4500 Hz friction surge, high crest factor)
   - RAIN_DOWNPOUR (wideband elevated noise floor across 1 - 8 kHz)
   - VEHICLE_HONK (400 - 600 Hz tonal blast)
"""

import math
import struct
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


def design_butterworth_bandpass(
    f_low: float,
    f_high: float,
    fs: float
) -> Tuple[float, float, float, float, float]:
    """
    Computes 2nd-order Butterworth IIR bandpass digital filter coefficients
    using the bilinear transform. H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2).
    """
    w_low = 2.0 * fs * math.tan(math.pi * f_low / fs)
    w_high = 2.0 * fs * math.tan(math.pi * f_high / fs)
    bw = w_high - w_low
    w0_sq = w_low * w_high

    d = 4.0 * fs * fs + 2.0 * fs * bw + w0_sq
    b0 = (2.0 * fs * bw) / d
    b1 = 0.0
    b2 = -b0
    a1 = (2.0 * w0_sq - 8.0 * fs * fs) / d
    a2 = (4.0 * fs * fs - 2.0 * fs * bw + w0_sq) / d
    return (b0, b1, b2, a1, a2)


def filter_energy(
    samples: List[float],
    coeffs: Tuple[float, float, float, float, float]
) -> float:
    """Passes samples through a 2nd-order IIR filter and computes total output energy."""
    b0, b1, b2, a1, a2 = coeffs
    x1 = x2 = y1 = y2 = 0.0
    energy = 0.0
    for s in samples:
        y0 = b0 * s + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        energy += y0 * y0
        x2 = x1
        x1 = s
        y2 = y1
        y1 = y0
    return energy


class UrbanAcousticAnalyzer:
    """
    Acoustic signal processor and urban sound classifier optimized for Copacabana street monitoring.
    Operates in pure Python standard library with zero heavyweight audio dependencies.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        history_window_size: int = 100,
        max_event_history: int = 50
    ):
        self.sample_rate = sample_rate
        self.history_window_size = history_window_size
        self.max_event_history = max_event_history

        # Pre-compute IIR digital filter coefficients
        nyquist = sample_rate * 0.48
        self.low_coeffs = design_butterworth_bandpass(30.0, 250.0, sample_rate)
        self.mid_coeffs = design_butterworth_bandpass(250.0, 2000.0, sample_rate)
        self.high_coeffs = design_butterworth_bandpass(2000.0, min(7500.0, nyquist), sample_rate)

        # Rolling buffer of decibel levels for percentile statistics
        self._db_history: deque = deque(maxlen=history_window_size)
        self._event_history: deque = deque(maxlen=max_event_history)

    @staticmethod
    def pcm_to_samples(raw_pcm: bytes, sample_width: int = 2) -> List[float]:
        """Converts raw 16-bit signed PCM little-endian bytes to normalized float samples [-1.0, 1.0]."""
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM (sample_width=2) is currently supported")
        count = len(raw_pcm) // 2
        ints = struct.unpack(f"<{count}h", raw_pcm[:count * 2])
        return [val / 32768.0 for val in ints]

    @staticmethod
    def compute_rms(samples: List[float]) -> float:
        """Calculates Root-Mean-Square (RMS) amplitude."""
        if not samples:
            return 0.0
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / len(samples))

    @staticmethod
    def compute_db_fs(rms: float) -> float:
        """Calculates Decibels relative to Full Scale (dBFS)."""
        if rms <= 1e-9:
            return -96.0
        return max(-96.0, round(20.0 * math.log10(rms), 2))

    def compute_band_energies(self, samples: List[float]) -> Dict[str, float]:
        """Calculates relative energy distribution across Low, Mid, and High frequency bands."""
        if not samples:
            return {"low": 0.0, "mid": 0.0, "high": 0.0, "total": 0.0, "low_ratio": 0.0, "mid_ratio": 0.0, "high_ratio": 0.0}

        low_p = filter_energy(samples, self.low_coeffs)
        mid_p = filter_energy(samples, self.mid_coeffs)
        high_p = filter_energy(samples, self.high_coeffs)

        total_p = low_p + mid_p + high_p + 1e-12
        return {
            "low": low_p,
            "mid": mid_p,
            "high": high_p,
            "total": total_p,
            "low_ratio": round(low_p / total_p, 4),
            "mid_ratio": round(mid_p / total_p, 4),
            "high_ratio": round(high_p / total_p, 4)
        }

    def get_noise_summary(self) -> Dict[str, float]:
        """
        Computes statistical noise percentiles:
        - L10: 90th percentile level (traffic peaks)
        - L50: Median level
        - L90: 10th percentile level (background floor)
        - Leq: Equivalent continuous energy level
        """
        if not self._db_history:
            return {"l10": -96.0, "l50": -96.0, "l90": -96.0, "leq": -96.0}

        sorted_db = sorted(self._db_history)
        n = len(sorted_db)

        # In acoustics, L10 is the level exceeded 10% of the time -> index 0.90 * N
        idx_l10 = min(n - 1, int(round(0.90 * (n - 1))))
        idx_l50 = min(n - 1, int(round(0.50 * (n - 1))))
        idx_l90 = min(n - 1, int(round(0.10 * (n - 1))))

        l10 = sorted_db[idx_l10]
        l50 = sorted_db[idx_l50]
        l90 = sorted_db[idx_l90]

        # Leq calculation: 10 * log10(mean(10^(L_i / 10)))
        power_sum = sum(10.0 ** (lvl / 10.0) for lvl in self._db_history)
        leq = round(10.0 * math.log10(power_sum / n), 2)

        return {
            "l10": round(l10, 2),
            "l50": round(l50, 2),
            "l90": round(l90, 2),
            "leq": leq
        }

    def classify_events(
        self,
        samples: List[float],
        rms: float,
        db_fs: float,
        band_energies: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Evaluates spectral energy ratios and temporal parameters against urban acoustic signatures.
        """
        events = []
        if rms < 0.005 or db_fs < -55.0:
            return events

        low_r = band_energies["low_ratio"]
        mid_r = band_energies["mid_ratio"]
        high_r = band_energies["high_ratio"]

        # Calculate current background floor
        noise_summary = self.get_noise_summary()
        floor_l90 = noise_summary["l90"]
        crest_factor = db_fs - floor_l90

        # 1. Emergency Siren: strong concentration in mid band (700 - 1500 Hz)
        if mid_r >= 0.65 and db_fs >= -45.0:
            events.append({
                "type": "EMERGENCY_SIREN",
                "confidence": min(1.0, round(mid_r * 1.15, 2)),
                "db_level": db_fs,
                "description": "Emergency vehicle siren wail (700-1500Hz)"
            })

        # 2. Transit Bus Air Brake: Pneumatic pressure discharge hiss in high frequencies (3-6 kHz)
        if high_r >= 0.70 and db_fs >= -40.0:
            events.append({
                "type": "BUS_AIR_BRAKE",
                "confidence": min(1.0, round(high_r * 1.15, 2)),
                "db_level": db_fs,
                "description": "Transit bus pneumatic air brake release (3-6kHz)"
            })

        # 3. Tire Screech: Sudden high friction burst with high crest factor
        if high_r >= 0.65 and crest_factor >= 12.0 and db_fs >= -30.0:
            events.append({
                "type": "TIRE_SCREECH",
                "confidence": min(1.0, round(high_r * 1.25, 2)),
                "db_level": db_fs,
                "crest_factor_db": round(crest_factor, 1),
                "description": "Tire asphalt friction screech (2.5-4.5kHz)"
            })

        # 4. Rain Downpour: Flat elevated wideband noise across entire spectrum
        # Typically low_r > 0.05, mid_r > 0.20, high_r > 0.20 without a single dominant band
        if (
            low_r >= 0.04 and mid_r >= 0.20 and high_r >= 0.20
            and rms >= 0.04 and db_fs >= -35.0
            and not any(e["type"] in ["EMERGENCY_SIREN", "BUS_AIR_BRAKE", "TIRE_SCREECH"] for e in events)
        ):
            events.append({
                "type": "RAIN_DOWNPOUR",
                "confidence": 0.85,
                "db_level": db_fs,
                "description": "Heavy tropical precipitation wideband noise floor"
            })

        # 5. Vehicle Honk: Strong mid-band concentration with minimal high frequencies
        if mid_r >= 0.75 and high_r <= 0.15 and db_fs >= -35.0:
            events.append({
                "type": "VEHICLE_HONK",
                "confidence": min(1.0, round(mid_r * 1.1, 2)),
                "db_level": db_fs,
                "description": "Vehicle horn dual-tone blast (400-600Hz)"
            })

        return events

    def analyze_chunk(
        self,
        samples: List[float],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes an audio sample chunk and updates rolling statistical state.
        """
        ts = timestamp if timestamp is not None else time.time()
        rms = self.compute_rms(samples)
        db_fs = self.compute_db_fs(rms)

        # Update historical rolling window
        self._db_history.append(db_fs)

        # Compute spectral decomposition
        band_energies = self.compute_band_energies(samples)

        # Classify acoustic events
        detected = self.classify_events(samples, rms, db_fs, band_energies)

        # Record events into history
        for ev in detected:
            record = dict(ev)
            record["timestamp"] = ts
            self._event_history.append(record)

        noise_summary = self.get_noise_summary()

        return {
            "timestamp": ts,
            "rms": round(rms, 5),
            "db_fs": db_fs,
            "noise_floor": noise_summary,
            "spectral": {
                "low_ratio": band_energies["low_ratio"],
                "mid_ratio": band_energies["mid_ratio"],
                "high_ratio": band_energies["high_ratio"]
            },
            "detected_events": detected
        }

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent classified acoustic events."""
        items = list(self._event_history)
        return items[-limit:][::-1]
