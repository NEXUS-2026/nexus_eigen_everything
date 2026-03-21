"""
Standalone box counter with DYNAMIC ROI — no frontend, no manual coordinates.

How the dynamic ROI works
The model (yolo26n.onnx) detects all cardboard objects as class 0.
The open carton (container) is always the largest detection by area.
Every frame we:
  1. Grab all class-0 detections above conf threshold.
  2. The one with the biggest bounding-box area = the container ROI.
  3. All other detections whose centres fall inside that box = small boxes.
  4. We smooth the ROI over a rolling window of N frames so it doesn't
     flicker when the container is briefly occluded by a hand.

Usage
  python run_counter.py
  python run_counter.py --source path/to/video.mp4
  python run_counter.py --source 0               # webcam
  python run_counter.py --source video.mp4 --save

Controls
  Q / ESC  — quit
  SPACE    — pause / resume
  R        — reset session
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from database import CountDatabase
from tracker_state import BoxTrackerStateMachine, FrameResult, TrackState
from yolo_engine import DetectionResult, ONNXDetector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("run_counter")

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLOURS = {
    TrackState.PENDING_ENTER:    (0,   200, 255),
    TrackState.CONFIRMED_INSIDE: (0,   230,  50),
    TrackState.REMOVED:          (80,   80,  80),
    TrackState.HIDDEN_INSIDE:    (255, 120,   0),
}


# Dynamic ROI manager


class DynamicROI:
    """
    Infers the container ROI from YOLO detections every frame.

    Strategy
    - The container is the LARGEST class-0 detection by bounding-box area,
      provided it is at least MIN_AREA_FRACTION of the frame.
    - We smooth the ROI over a rolling window to absorb per-frame jitter
      and handle frames where the container is briefly occluded.
    - Once a stable ROI is found it is held ("locked") until the detection
      drops out for more than LOCK_TIMEOUT frames, after which it resets.

    Parameters
    smoothing_frames   : number of recent ROI observations to average over.
    min_area_fraction  : minimum box area as fraction of frame to qualify
                         as the container (filters out small false positives).
    container_class_id : which YOLO class to look for. 0 = the only class
                         in yolo26n.onnx that fires reliably.
    lock_timeout       : frames without a container detection before the
                         locked ROI is discarded.
    """

    def __init__(
        self,
        frame_wh:            tuple[int, int],
        smoothing_frames:    int   = 10,
        min_area_fraction:   float = 0.03,   # at least 3% of frame area
        container_class_id:  int   = 0,
        lock_timeout:        int   = 60,     # 1.5 s at 40 FPS
    ) -> None:
        self.frame_w, self.frame_h = frame_wh
        self.smoothing_frames    = smoothing_frames
        self.min_area_fraction   = min_area_fraction
        self.container_class_id  = container_class_id
        self.lock_timeout        = lock_timeout

        # Rolling buffer of (x1, y1, x2, y2) observations
        self._history: deque[tuple[float,float,float,float]] = deque(
            maxlen=smoothing_frames
        )
        self._frames_without_detection = 0

        # The smoothed ROI polygon exposed to the rest of the system
        # None until we have at least one detection.
        self.polygon: Optional[np.ndarray] = None
        self.locked  = False


    def update(self, detection_result: DetectionResult) -> Optional[np.ndarray]:
        """
        Call once per frame with the latest DetectionResult.
        Returns the current best ROI polygon (or None if not yet found).
        """
        frame_area = self.frame_w * self.frame_h

        # Find the largest qualifying detection
        best_area  = -1.0
        best_box: Optional[tuple[float,float,float,float]] = None

        for i in range(detection_result.count):
            cls = int(detection_result.class_ids[i])
            if cls != self.container_class_id:
                continue
            x1, y1, x2, y2 = detection_result.boxes[i]
            area = float((x2 - x1) * (y2 - y1))
            frac = area / frame_area
            if frac < self.min_area_fraction:
                continue
            if area > best_area:
                best_area = area
                best_box  = (float(x1), float(y1), float(x2), float(y2))

        if best_box is not None:
            self._history.append(best_box)
            self._frames_without_detection = 0
            self.locked = True
        else:
            self._frames_without_detection += 1
            if self._frames_without_detection > self.lock_timeout:
                # Container gone too long — reset
                self._history.clear()
                self.locked  = False
                self.polygon = None
                logger.debug("DynamicROI: lock expired, resetting.")
                return None

        if not self._history:
            return None

        # Smooth by averaging the rolling window
        arr  = np.array(self._history, dtype=np.float32)  # (N, 4)
        mean = arr.mean(axis=0)                            # [x1, y1, x2, y2]
        x1, y1, x2, y2 = mean

        self.polygon = np.array([
            [int(x1), int(y1)],
            [int(x2), int(y1)],
            [int(x2), int(y2)],
            [int(x1), int(y2)],
        ], dtype=np.int32)

        return self.polygon


    @property
    def is_ready(self) -> bool:
        """True once we have at least one valid observation."""
        return self.polygon is not None

    def reset(self) -> None:
        self._history.clear()
        self._frames_without_detection = 0
        self.locked  = False
        self.polygon = None



# Shared detection between inference thread and main loop


class _SharedDetection:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._det   = DetectionResult.empty()
        self._fresh = False

    def write(self, det: DetectionResult) -> None:
        with self._lock:
            self._det   = det
            self._fresh = True

    def read(self) -> tuple[DetectionResult, bool]:
        with self._lock:
            det, fresh  = self._det, self._fresh
            self._fresh = False
        return det, fresh



# Inference thread


def _inference_thread(
    detector:   ONNXDetector,
    frame_q:    queue.Queue,
    shared:     _SharedDetection,
    stop_event: threading.Event,
) -> None:
    logger.info("Inference thread started.")
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if frame is None:
            break
        shared.write(detector.detect(frame))
    logger.info("Inference thread exiting.")



# Annotation


def _annotate(
    frame:       np.ndarray,
    result:      FrameResult,
    roi_polygon: Optional[np.ndarray],
    fps:         float,
    paused:      bool,
    roi_ready:   bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Dynamic ROI polygon
    if roi_polygon is not None:
        cv2.polylines(
            out, [roi_polygon.reshape(-1, 1, 2)],
            isClosed=True, color=(0, 255, 128), thickness=3,
        )
        cv2.putText(out, "CONTAINER ROI (auto)",
                    (roi_polygon[0][0], roi_polygon[0][1] - 8),
                    FONT, 0.55, (0, 255, 128), 1)
    else:
        # Not yet locked — show waiting message
        cv2.putText(out, "Searching for container...",
                    (20, h // 2), FONT, 1.0, (0, 200, 255), 2)

    # Tracked boxes
    for i, box in enumerate(result.tracked_boxes):
        tid   = int(result.track_ids[i])
        state = result.track_states.get(tid, TrackState.REMOVED)
        color = STATE_COLOURS.get(state, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {state.name[:3]}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - lh - 5), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), FONT, 0.42, (0, 0, 0), 1)

    # HUD
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    roi_status = "ROI: LOCKED" if roi_ready else "ROI: searching..."
    roi_color  = (0, 230, 50)  if roi_ready else (0, 200, 255)

    cv2.putText(out, f"Boxes in carton: {result.box_count}",
                (10, 35), FONT, 1.0, (255, 255, 255), 2)
    cv2.putText(out, f"FPS: {fps:.1f}   YOLO: {result.inference_ms:.0f}ms   {roi_status}",
                (10, 62), FONT, 0.52, roi_color, 1)
    cv2.putText(out, "SPACE=pause  R=reset  Q=quit",
                (10, 88), FONT, 0.42, (100, 100, 100), 1)

    if paused:
        cv2.putText(out, "PAUSED",
                    (w // 2 - 70, h // 2), FONT, 1.6, (0, 200, 255), 3)

    return out


def _fit_to_screen(frame: np.ndarray, max_w: int = 900, max_h: int = 900) -> np.ndarray:
    h, w   = frame.shape[:2]
    scale  = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)))



# Main


def run(source: str | int, save: bool = False) -> None:

    # Open capture
    is_webcam = isinstance(source, int)
    if is_webcam:
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Webcam warming up …")
        time.sleep(1.5)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Source: %dx%d @ %.1f FPS", frame_w, frame_h, src_fps)

    # Pipeline components
    detector = ONNXDetector(
        model_path=config.MODEL_PATH,
        input_size=config.ONNX_INPUT_SIZE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        nms_iou_threshold=config.NMS_IOU_THRESHOLD,
        # Pass None — let the dynamic ROI use ALL class-0 detections.
        # The DynamicROI class handles separating container from small boxes.
        target_classes=None,
        num_intra_threads=config.ORT_INTRA_THREADS,
    )

    logger.info("ONNX warm-up …")
    detector.detect(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
    logger.info("Warm-up done.")

    # Dynamic ROI — no polygon needed in config.py
    dynamic_roi = DynamicROI(
        frame_wh=(frame_w, frame_h),
        smoothing_frames=15,      # average last 15 detections — smooth but responsive
        min_area_fraction=0.03,   # container must be ≥3% of frame area
        container_class_id=0,
        lock_timeout=60,
    )

    # State machine starts with a dummy 1px ROI — it gets replaced as soon
    # as DynamicROI locks onto the container (usually within 1-2 seconds).
    _dummy_roi = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.int32)
    state_machine = BoxTrackerStateMachine(
        roi_polygon=_dummy_roi,
        frame_wh=(frame_w, frame_h),
        debounce_frames=config.DEBOUNCE_FRAMES,
        ghost_frames=config.GHOST_FRAMES,
        track_buffer=config.TRACK_BUFFER,
        fps=config.TARGET_FPS,
    )

    db = CountDatabase(config.DB_PATH)
    db.start_session()

    # Inference thread
    shared     = _SharedDetection()
    stop_event = threading.Event()
    frame_q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)

    inf_thread = threading.Thread(
        target=_inference_thread,
        args=(detector, frame_q, shared, stop_event),
        daemon=True, name="InferenceThread",
    )
    inf_thread.start()

    # Video writer
    writer: Optional[cv2.VideoWriter] = None
    if save:
        out_path = Path("output_annotated.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, src_fps, (frame_w, frame_h))
        logger.info("Saving to %s", out_path)

    # Loop variables
    last_detection  = DetectionResult.empty()
    detect_skip     = 4
    tick            = 0
    paused          = False
    fps_actual      = 0.0
    fps_t0          = time.perf_counter()
    fps_frames      = 0
    frame_duration  = 1.0 / config.TARGET_FPS

    # Track how many frames we've had a locked ROI, to avoid
    # counting boxes before the ROI is stable
    roi_stable_frames = 0
    ROI_STABLE_THRESHOLD = 20   # require 20 frames of stable ROI before counting

    logger.info("Running. Press Q to quit.")

    while True:
        t_start = time.perf_counter()
        tick   += 1

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            state_machine.reset()
            dynamic_roi.reset()
            roi_stable_frames = 0
            db.end_session(0)
            db.start_session()
            last_detection = DetectionResult.empty()
            logger.info("Session reset.")

        if paused:
            time.sleep(0.05)
            continue

        # Read frame
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                logger.info("End of video — looping.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state_machine.reset()
                dynamic_roi.reset()
                roi_stable_frames = 0
                db.end_session(state_machine.count)
                db.start_session()
                last_detection = DetectionResult.empty()
            continue

        # Feed inference thread
        if tick % detect_skip == 0:
            try:
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
                frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        # Consume fresh detection
        det, fresh = shared.read()
        if fresh:
            last_detection = det

            # Update dynamic ROI from this detection
            new_polygon = dynamic_roi.update(last_detection)

            if new_polygon is not None:
                roi_stable_frames += 1

                # Hot-swap the ROI polygon into the state machine.
                # We update the PolygonZone directly so we don't have to
                # reconstruct the entire state machine (which would reset
                # all track records and the count).
                import supervision as sv
                state_machine._zone = sv.PolygonZone(polygon=new_polygon)
                state_machine.roi_polygon = new_polygon

        # Build a detection result that EXCLUDES the container box
        # We don't want the container itself to be tracked and counted.
        # Filter: remove the largest detection (= the container) from the
        # detection result before passing to ByteTrack.
        filtered_detection = _filter_out_container(last_detection, frame_w, frame_h)

        # State machine update
        # Don't count until the ROI has been stable for N frames.
        if roi_stable_frames >= ROI_STABLE_THRESHOLD:
            result: FrameResult = state_machine.update(frame, filtered_detection)
        else:
            # ROI not stable yet — run tracker for prediction continuity
            # but use an empty detection so no counts fire
            result = state_machine.update(frame, DetectionResult.empty())

        # DB events
        for event_type, track_id in result.events:
            db.log_event(event_type, track_id, result.box_count)
            logger.info("EVENT %-8s  track=%d  count=%d",
                        event_type, track_id, result.box_count)
        if result.events:
            db.flush()

        # Annotate
        annotated = _annotate(
            frame, result,
            roi_polygon=dynamic_roi.polygon,
            fps=fps_actual,
            paused=paused,
            roi_ready=(roi_stable_frames >= ROI_STABLE_THRESHOLD),
        )

        display = _fit_to_screen(annotated)
        cv2.imshow("Warehouse Box Counter  [Q=quit  SPACE=pause  R=reset]", display)

        if writer is not None:
            writer.write(annotated)

        # FPS
        fps_frames += 1
        if fps_frames >= 60:
            fps_actual = fps_frames / (time.perf_counter() - fps_t0)
            fps_t0     = time.perf_counter()
            fps_frames = 0

        # Rate limiter
        remaining = frame_duration - (time.perf_counter() - t_start)
        if remaining > 0.002:
            time.sleep(remaining - 0.002)
        while (time.perf_counter() - t_start) < frame_duration:
            pass

    # Cleanup
    stop_event.set()
    try:
        frame_q.put_nowait(None)
    except queue.Full:
        pass
    inf_thread.join(timeout=3.0)
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    db.end_session(state_machine.count)
    logger.info("Final count: %d", state_machine.count)
    _print_summary(db)



# Helper: remove the container detection from the detection result


def _filter_out_container(
    det:     DetectionResult,
    frame_w: int,
    frame_h: int,
) -> DetectionResult:
    """
    Remove the largest class-0 detection (= the container) so ByteTrack
    never tries to track it as a small box.
    Returns a new DetectionResult with that one box removed.
    """
    if det.count == 0:
        return det

    frame_area = frame_w * frame_h
    areas      = ((det.boxes[:, 2] - det.boxes[:, 0]) *
                  (det.boxes[:, 3] - det.boxes[:, 1]))

    # Only suppress it if it genuinely looks like a container (≥3% of frame)
    big_mask    = (areas / frame_area) >= 0.03
    if not big_mask.any():
        return det

    # Index of the single largest qualifying box
    # np.argmax on a boolean-filtered set — find largest among big ones
    large_indices = np.where(big_mask)[0]
    largest_idx   = large_indices[np.argmax(areas[large_indices])]

    # Keep everything except that index
    keep = np.ones(det.count, dtype=bool)
    keep[largest_idx] = False

    if not keep.any():
        return DetectionResult.empty()

    return DetectionResult(
        boxes=det.boxes[keep],
        scores=det.scores[keep],
        class_ids=det.class_ids[keep],
        inference_ms=det.inference_ms,
    )


# Summary printer


def _print_summary(db: CountDatabase) -> None:
    sessions = db.get_session_summary()
    print("\n" + "─" * 50)
    print("  SESSION SUMMARY")
    print("─" * 50)
    for s in sessions[:5]:
        dur = ""
        if s["ended_at"] and s["started_at"]:
            dur = f"  {s['ended_at'] - s['started_at']:.0f}s"
        print(f"  Session {s['id']:>3}  |  count={s['final_count'] or '?':>4}  |{dur}")
    print("─" * 50 + "\n")

# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--save",   action="store_true")
    args = ap.parse_args()

    source: str | int = args.source
    if args.source.isdigit():
        source = int(args.source)

    run(source=source, save=args.save)