#!/bin/bash
# Push OBS Virtual Camera to MediaMTX for VideoDB.
# Load .env from project root if present (for AUDIO_INPUT_DEVICE etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
[ -f "${PROJECT_DIR}/.env" ] && set -a && . "${PROJECT_DIR}/.env" && set +a
# OBS Virtual Camera switches between 30fps and 60fps - this tries both.
# Usage: ./scripts/videodb_rtsp.sh video
#        ./scripts/videodb_rtsp.sh audio

RTMP="rtmp://localhost:1935"

case "${1:-video}" in
  video)
    echo "Pushing OBS Virtual Camera to ${RTMP}/cam ..."
    echo "Trying 30fps first..."
    if ffmpeg -f avfoundation -video_size 1920x1080 -framerate 30 -i "1" \
      -vf "scale=1280:720" -c:v libx264 -preset ultrafast -tune zerolatency \
      -pix_fmt yuv420p -level 4.0 -f flv "${RTMP}/cam" 2>/dev/null; then
      exit 0
    fi
    echo "30fps failed, trying 60fps..."
    ffmpeg -f avfoundation -video_size 1920x1080 -framerate 60 -i "1" \
      -vf "scale=1280:720" -r 30 -c:v libx264 -preset ultrafast -tune zerolatency \
      -pix_fmt yuv420p -level 4.0 -f flv "${RTMP}/cam"
    ;;
  audio)
    # AUDIO_INPUT_DEVICE: avfoundation audio index (0=default, 1=Fanvil when connected via USB).
    # Find indices: ffmpeg -f avfoundation -list_devices true -i ""
    AUDIO_IDX="0"
    [[ -n "${AUDIO_INPUT_DEVICE}" && "${AUDIO_INPUT_DEVICE}" =~ ^[0-9]+$ ]] && AUDIO_IDX="${AUDIO_INPUT_DEVICE}"
    echo "Pushing mic (device :${AUDIO_IDX}) to ${RTMP}/mic ..."
    ffmpeg -y -f avfoundation -i ":${AUDIO_IDX}" -c:a aac -ar 16000 -f flv "${RTMP}/mic"
    ;;
  *)
    echo "Usage: $0 {video|audio}"
    exit 1
    ;;
esac
