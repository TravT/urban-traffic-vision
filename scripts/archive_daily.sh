#!/usr/bin/env bash
# ==============================================================================
# Daily Midnight Rolling Archival & Zstandard Compression Script
# Urban Traffic & Building Vision Appliance (ADR-25, ADR-26, ADR-29)
# ==============================================================================
set -euo pipefail

TARGET_DATE="${1:-$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)}"
VISION_ROOT="${VISION_ROOT:-/data/media/merged/vision}"
RAW_DIR="${VISION_ROOT}/validation"
ARCHIVE_DIR="${VISION_ROOT}/archives"
STAGING_DIR="/tmp/vision_archive_${TARGET_DATE}"
RETENTION_DAYS=30
ZSTD_THREADS=4
ZSTD_LEVEL=3
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive_root:UrbanTrafficVision_Backups}"
CLOUD_FUSE_DIR="/data/media/cloud_drive/UrbanTrafficVision_Backups"

echo "[$(date -Iseconds)] Starting daily archival for target date: ${TARGET_DATE}"

# Guard: verify archive destinations exist
mkdir -p "${ARCHIVE_DIR}"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"/{timelapse,events,telemetry}

# Step 1: Compile 30 FPS Daily Time-Lapse MP4 if frames exist
TIMELAPSE_FILE="${ARCHIVE_DIR}/daily_timelapse_${TARGET_DATE}.mp4"
FRAME_COUNT=$(find "${RAW_DIR}" -maxdepth 1 -name "*${TARGET_DATE}*.jpg" 2>/dev/null | wc -l)
if [ "${FRAME_COUNT}" -gt 10 ]; then
    echo "[$(date -Iseconds)] Generating 30 FPS time-lapse from ${FRAME_COUNT} frames..."
    ffmpeg -y -hide_banner -loglevel error \
        -pattern_type glob -i "${RAW_DIR}/*${TARGET_DATE}*.jpg" \
        -vf "scale=1280:-2,format=yuv420p" \
        -c:v libx264 -crf 24 -preset fast \
        "${TIMELAPSE_FILE}" || true
    if [ -f "${TIMELAPSE_FILE}" ]; then
        cp "${TIMELAPSE_FILE}" "${STAGING_DIR}/timelapse/" 2>/dev/null || true
    fi
else
    echo "[$(date -Iseconds)] Notice: Less than 10 raw frames found for ${TARGET_DATE}, skipping MP4 stitch."
fi

# Step 2: Stage Keyframe Event Snapshots
if [ -d "${VISION_ROOT}/events" ]; then
    find "${VISION_ROOT}/events" -name "*${TARGET_DATE}*" -exec cp {} "${STAGING_DIR}/events/" \; 2>/dev/null || true
fi

# Step 3: Stage Structured CSV & Telemetry Logs
for audit_file in "validation_audit.csv" "streetlight_audit.csv" "traffic_stats.json" "noise_timeseries.csv"; do
    if [ -f "${RAW_DIR}/${audit_file}" ]; then
        cp "${RAW_DIR}/${audit_file}" "${STAGING_DIR}/telemetry/" 2>/dev/null || true
    fi
done

# Step 4: Pack and Compress using Zstandard (tar.zst) per ADR-25/ADR-26
ARCHIVE_FILE="${ARCHIVE_DIR}/daily_${TARGET_DATE}.tar.zst"
echo "[$(date -Iseconds)] Compressing archive into ${ARCHIVE_FILE} using zstd (Level ${ZSTD_LEVEL}, ${ZSTD_THREADS} threads)..."
START_TS=$(date +%s%N)

tar -I "zstd -${ZSTD_LEVEL} -T${ZSTD_THREADS}" -cf "${ARCHIVE_FILE}" -C "${STAGING_DIR}" .

END_TS=$(date +%s%N)
DURATION_MS=$(( (END_TS - START_TS) / 1000000 ))
ARCHIVE_SIZE=$(du -h "${ARCHIVE_FILE}" 2>/dev/null | cut -f1 || echo "unknown")

echo "[$(date -Iseconds)] Archive created: ${ARCHIVE_FILE} (${ARCHIVE_SIZE}) in ${DURATION_MS} ms"

# Step 5: Clean Staging & Ephemeral Raw Cache older than 24h
rm -rf "${STAGING_DIR}"
find "${RAW_DIR}" -maxdepth 1 -name "*${TARGET_DATE}*.jpg" -delete 2>/dev/null || true

# Step 6: Enforce Local 30-Day Rolling Retention
echo "[$(date -Iseconds)] Enforcing ${RETENTION_DAYS}-day rolling retention in ${ARCHIVE_DIR}..."
find "${ARCHIVE_DIR}" -name "daily_*.tar.zst" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
find "${ARCHIVE_DIR}" -name "daily_timelapse_*.mp4" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true

# Step 7: Cloud Upload to Google Drive & 30-Day Cloud Pruning (ADR-29)
echo "[$(date -Iseconds)] Uploading daily video & telemetry to Google Drive (${GDRIVE_REMOTE})..."
if command -v rclone >/dev/null 2>&1; then
    if [ -f "${TIMELAPSE_FILE}" ]; then
        rclone copy "${TIMELAPSE_FILE}" "${GDRIVE_REMOTE}" || true
    fi
    if [ -f "${ARCHIVE_FILE}" ]; then
        rclone copy "${ARCHIVE_FILE}" "${GDRIVE_REMOTE}" || true
    fi
    echo "[$(date -Iseconds)] Enforcing 30-day cloud rolling retention on ${GDRIVE_REMOTE}..."
    rclone delete --min-age "${RETENTION_DAYS}d" "${GDRIVE_REMOTE}" 2>/dev/null || true
elif [ -d "${CLOUD_FUSE_DIR}" ]; then
    echo "[$(date -Iseconds)] rclone CLI unavailable, using local FUSE cloud mount: ${CLOUD_FUSE_DIR}"
    mkdir -p "${CLOUD_FUSE_DIR}"
    [ -f "${TIMELAPSE_FILE}" ] && cp "${TIMELAPSE_FILE}" "${CLOUD_FUSE_DIR}/" || true
    [ -f "${ARCHIVE_FILE}" ] && cp "${ARCHIVE_FILE}" "${CLOUD_FUSE_DIR}/" || true
    find "${CLOUD_FUSE_DIR}" -type f -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
fi

echo "[$(date -Iseconds)] Daily archival and cloud rolling backup complete."
