"""
Single source of truth for every tunable constant in the pipeline.
Change values here, never hardcode them in main.py or tracker_state.py.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# Paths

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.onnx"
DB_PATH    = BASE_DIR / "logs"   / "counts.db"
LOG_DIR    = BASE_DIR / "logs"

# Video source
# Set to an integer (e.g. 0) for a webcam, or a file path string for a video.

VIDEO_SOURCE: str | int = 0

# Inference engine

ONNX_INPUT_SIZE        = (640, 640)   # (W, H) must match your export
CONFIDENCE_THRESHOLD   = 0.35
NMS_IOU_THRESHOLD      = 0.45
# COCO class 0 = person. Replace with your fine tuned box class ID.
TARGET_CLASSES         = None
# ORT intra op threads. 3 leaves 1 core free for the video thread on a Pi 4.
ORT_INTRA_THREADS      = 3

# Tracker & state machine

TARGET_FPS       = 40
DEBOUNCE_FRAMES  = 5    # frames inside ROI before +1 is confirmed (~125 ms)
GHOST_FRAMES     = 200   # Increase to 200 frames (5 full seconds of memory if the camera is blocked). If the worker takes longer than 5 seconds, increase this to 400 (10 seconds)
TRACK_BUFFER     = 240  # The ByteTrack Kalman buffer MUST be larger than the GHOST_FRAMES. This ensures the tracker algorithm doesn't forget the boxes before your state machine does.


# ROI polygon vertices in [x, y] order, pixel coordinates.
# Default: centre 60 % of a 640x480 frame.
# Measure your actual carton position and replace these values.

_W, _H = 640, 480

# Default: centre 60% of the frame works for webcam at 640x480
ROI_POLYGON = np.array([
    [int(_W * 0.20), int(_H * 0.20)],   # top-left
    [int(_W * 0.80), int(_H * 0.20)],   # top-right
    [int(_W * 0.80), int(_H * 0.80)],   # bottom-right
    [int(_W * 0.20), int(_H * 0.80)],   # bottom-left
], dtype=np.int32)

# WebSocket streaming

JPEG_QUALITY    = 70    # 0-100. Lower = smaller payload = lower latency on LAN.
STREAM_MAX_FPS  = 40    # Cap the WebSocket push rate independently of capture FPS.

# FastAPI server

HOST = "0.0.0.0"
PORT = 8000