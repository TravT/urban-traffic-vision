# Urban Traffic & Building Vision Appliance (`urban-traffic-vision`)

> **Location:** Rua Barata Ribeiro 726, Copacabana, Rio de Janeiro (`CEP: 22051-002`)  
> **Edge Optical & Acoustic Sensor:** Samsung Galaxy S20 FE (3rd-Floor Window Mount, Tailscale IP `100.115.165.41`)  
> **Compute & Ingress:** Dell Latitude 7390 (`homelab`, Nomad, Traefik v3 `vision.home.arpa`, Mosquitto MQTT)  
> **Architecture Standard:** [ADR-25: Urban Traffic & Multi-Layer Vision Appliance Architecture](file:///home/tlima/Enterprise_Hub/docs/wiki/adrs/ADR-25-Urban-Vision-Modularization-and-Lifecycle.md)

---

## 1. Overview & Architecture

The `urban-traffic-vision` appliance transforms an edge smartphone camera into a multi-layer urban sensing platform monitoring Rua Barata Ribeiro. Rather than recording unbounded video, the system captures synchronized visual and acoustic signals and extracts high-level spatial, photometric, and acoustic telemetry.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MULTI-LAYER SENSING STACK                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Building Facade    ──► Window Grid: Apartment Occupancy & Illumination       │
│ Layer 2: Sidewalk & Entry   ──► Delivery Couriers (iFood/Rappi dwell >= 30s) & Flux     │
│ Layer 3: Dedicated Bus Lane ──► BRS Transit Corridor Obstruction & Double-Parking     │
│ Layer 4: General Roadway    ──► Velocity (km/h) across 25m Baseline, Traffic Congestion│
│ Layer 5: Photometrics       ──► Streetlamp Ignition vs. Home Assistant sun.sun Sunset  │
│ Layer 6: Acoustics (RTSP)   ──► Ambient Noise Floor (P10 dB SPL), Car Horns & Sirens   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
dev/urban-traffic-vision/
├── Dockerfile                        # Multi-stage production build (Debian 12 + Python 3.11 + AVX2)
├── README.md                         # This documentation
├── requirements.txt                  # Pinned production Python dependencies
├── Makefile                          # build, test, run targets
├── config/
│   ├── zones.json                    # Normalized polygon coordinates for windows, bus lane, entrance
│   └── settings.yaml                 # Camera parameters, thresholds, MQTT endpoints, retention
├── src/
│   ├── ingestion/                    # S20 FE capture adapters (stills pull, burst, RTSP client)
│   ├── inference/                    # YOLOv8s engine, spatial ROI clipping, SIMD AVX2 execution
│   ├── tracking/                     # CentroidTracker, velocity vectors, directional classifier
│   ├── analytics/
│   │   ├── apartment_occupancy.py    # HSV luminosity & domestic color temperature analyzer
│   │   ├── delivery_monitor.py       # Dwell-time courier tripwire (motorcycle/bicycle >= 30s)
│   │   ├── bus_lane_compliance.py    # Curbside transit lane obstruction detector
│   │   ├── photometrics.py           # Streetlight activation vs. Home Assistant sun.sun logger
│   │   └── acoustics.py              # Audio RMS decibels, baseline noise floor, horn & siren FFT
│   ├── api/                          # FastAPI / REST endpoints & WebSocket live telemetry
│   └── web/                          # Frontend dashboard & HTML5 Canvas Zone Editor
├── scripts/
│   ├── archive_daily.sh              # Midnight tar.zst compression and rolling retention cron
│   └── benchmark_options.sh          # 10-minute Option A vs. Option C benchmark runner
└── nomad/
    └── vision-worker.nomad           # Production Nomad job specification
```

---

## 3. Phased Roadmap (Priority: 4 ➔ 1 ➔ 3 ➔ 2)

1. **Phase 1 (Priority 4): Project Modularization & Dedicated Docker Image:**
   - Full packaging into `homelab/urban-traffic-vision:latest`.
   - Eliminates host Python script volume mounting into generic containers.
2. **Phase 2 (Priority 1): Interactive Window Grid & Apartment Occupancy:**
   - Interactive HTML5 Canvas Editor (`vision.home.arpa/editor`) allowing crosshair drawing over apartment windows across the street.
   - Zero-AI deterministic HSV luminosity ($V > 160$) and color temperature heuristic (domestic warm lighting vs. dark/unoccupied).
3. **Phase 3 (Priority 3): Delivery Couriers & Front Entrance Flux:**
   - Entrance dwell timer flagging stationary couriers (`motorcycle` / `bicycle` $\ge 30\text{s}$).
   - Pedestrian sidewalk flux counters.
4. **Phase 4 (Priority 2): Dedicated Bus Lane Compliance & Velocity Profiling:**
   - Curbside transit lane obstruction enforcement ($> 15\text{s}$ stationary non-bus vehicles).
   - Speed calculation across 25m street baseline: $\text{Speed} = \frac{25\text{m}}{\Delta t} \times 3.6\text{ km/h}$.

---

## 4. Cluster Integrations & Telemetry

### A. Home Assistant `sun.sun` & Streetlight Photometrics
- **Zero External API Dependencies:** Astronomical civil sunset is fetched directly from the local Home Assistant instance (`http://127.0.0.1:8123/api/states/sun.sun`).
- **Photometric Step-Jump:** Evaluates the streetlamp fixture and asphalt ROI. A sharp step increase ($\Delta V > +45$ in $< 2\text{s}$) logs the exact second of Rioluz activation to `streetlight_audit.csv`, recording the delta against civil sunset.

### B. Option C Environmental Acoustics
Ingesting audio from the S20 FE internal microphone over RTSP:
- **Baseline Noise Floor:** Rolling 15-minute 10th percentile ($P_{10}$) of RMS sound pressure level ($\text{dB}_\text{SPL}$ equivalent).
- **Vehicle Horns ("Buzinas"):** FFT bandpass energy ratio in the $2.0 - 3.5\text{ kHz}$ band. Impulses exceeding the noise floor by $> 15\text{ dB}$ for $0.2 - 1.5\text{s}$ increment the daily horn counter.
- **Emergency Sirens:** Detects frequency-modulated sweeps between $600\text{ Hz}$ and $1500\text{ Hz}$ to isolate ambulances heading to Copa Star / Copa D'Or from police patrols.

---

## 5. Storage, Archival & The `tar.zst` Advantage

### Why Zstandard (`.tar.zst`) over `.tar.gz` and `.7zip`?

| Compression Engine | Algorithm | Multi-Threading | Compression Speed | Decompression Speed | Compression Ratio | Homelab Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gzip` (`.tar.gz`)** | Deflate (1992) | Poor (Single-core default, `pigz` external) | Moderate (~30–50 MB/s) | ~300 MB/s | Baseline | Legacy standard |
| **`7-Zip` (`.7z`)** | LZMA / LZMA2 | Yes (High CPU/RAM cost) | Extremely Slow (~10–25 MB/s) | ~50–80 MB/s | Maximum (Smallest size) | Too heavy; burns CPU cycles for 30+ seconds |
| **`zstd` (`.tar.zst`)** | FSE + LZ77 (Meta / RFC 8878) | **Native (`-T4` or `-T0`)** | **Blazing Fast (350–450 MB/s)** | **Extremely Fast (> 1.8 GB/s)** | **10–15% better than Gzip** | **Optimal (Sub-2s archive runs)** |

### Daily Archival Math
- **Raw Frames (Uncompressed):** $\approx 14.4\text{ GB/day}$ (discarded daily).
- **Curated Archive:** 1x 30 FPS daily time-lapse MP4 (~140 MB) + verified event keyframes (~20 MB) + CSV logs (~5 MB) = **~165 MB/day**.
- **Zstandard Compression Time:** **$1.6 - 2.1\text{ seconds}$** using 4 CPU threads on the Dell i7-8650U.
- **30-Day Rolling Footprint:** **$< 4.5\text{ GB}$** on MergerFS (`/data/media/merged/vision/archives/`).

---

## 6. Build & Deployment

### Build Container Image
```bash
docker build -t homelab/urban-traffic-vision:latest .
```

### Deploy via Nomad
```bash
nomad job run nomad/vision-worker.nomad
```

### GitOps Provisioning
```bash
ansible-playbook -i ansible/inventory.ini ansible/site.yml --tags docker --vault-password-file ansible/.vault_pass
```
