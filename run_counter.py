"""
Box counter using the custom-trained model (best.onnx).

Model classes:
    0 = Person
    1 = bigger_box  — the large open carton (auto ROI)
    2 = boxes       — small boxes being packed

Counting strategy:
    - Visible count = boxes YOLO sees inside the carton right now.
    - Peak count    = highest visible count ever seen (held during occlusion).
    - Count drops   = only after 1 full second with no person blocking.
    - DB is updated every time the count changes, and on session end.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config
from database import CountDatabase
from yolo_engine import DetectionResult, ONNXDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("run_counter")

FONT           = cv2.FONT_HERSHEY_SIMPLEX
CLS_PERSON     = 0
CLS_BIGGER_BOX = 1
CLS_BOXES      = 2

BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_area(b: BBox) -> float:
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def box_center(b: BBox) -> tuple[float, float]:
    return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))

def center_inside(box: BBox, roi: BBox) -> bool:
    cx, cy = box_center(box)
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

def intersection_area(a: BBox, b: BBox) -> float:
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    return max(0.0,ix2-ix1)*max(0.0,iy2-iy1)

def expand_box(b: BBox, sx: float, sy: float, W: int, H: int) -> BBox:
    cx, cy = box_center(b)
    w = max(1.0, (b[2]-b[0])*sx)
    h = max(1.0, (b[3]-b[1])*sy)
    return (max(0.0,cx-w/2), max(0.0,cy-h/2), min(W,cx+w/2), min(H,cy+h/2))

def estimate_hand_regions(person_box: BBox, W: int, H: int) -> tuple[BBox, BBox]:
    x1, y1, x2, y2 = person_box
    pw = max(1.0, x2-x1); ph = max(1.0, y2-y1)
    palm_w = 0.18*pw;     palm_h = 0.18*ph
    palm_y = y1 + 0.60*ph

    def clamp(b: BBox) -> BBox:
        return (max(0,b[0]), max(0,b[1]), min(W,b[2]), min(H,b[3]))

    left_cx  = x1 + 0.26*pw
    right_cx = x2 - 0.26*pw
    left  = clamp((left_cx -palm_w/2, palm_y-palm_h/2,
                   left_cx +palm_w/2, palm_y+palm_h/2))
    right = clamp((right_cx-palm_w/2, palm_y-palm_h/2,
                   right_cx+palm_w/2, palm_y+palm_h/2))
    return left, right


# ---------------------------------------------------------------------------
# Shared detection
# ---------------------------------------------------------------------------

class _SharedDetection:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._det   = DetectionResult.empty()
        self._fresh = False

    def write(self, det: DetectionResult) -> None:
        with self._lock:
            self._det, self._fresh = det, True

    def read(self) -> tuple[DetectionResult, bool]:
        with self._lock:
            det, fresh  = self._det, self._fresh
            self._fresh = False
        return det, fresh


def _inference_thread(detector, frame_q, shared, stop_event):
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


# ---------------------------------------------------------------------------
# Counter state — lives outside run() so cleanup section can always read it
# ---------------------------------------------------------------------------

class _CounterState:
    def __init__(self) -> None:
        self.net_count   = 0
        self.peak_count  = 0
        self.drop_holdoff= 0
        self.last_logged = -1

    def reset(self) -> None:
        self.net_count    = 0
        self.peak_count   = 0
        self.drop_holdoff = 0
        self.last_logged  = -1


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    model_path:   str,
    source:       str | int,
    conf:         float = 0.30,
    imgsz:        int   = 512,
    save:         bool  = False,
    min_box_area: float = 80.0,
) -> None:

    is_webcam = isinstance(source, int)
    if is_webcam:
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.5)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        logger.error("Cannot open: %s", source)
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Source: %dx%d @ %.1f FPS", frame_w, frame_h, src_fps)

    detector = ONNXDetector(
        model_path=model_path,
        input_size=(imgsz, imgsz),
        confidence_threshold=conf,
        nms_iou_threshold=0.45,
        target_classes=None,
        num_intra_threads=config.ORT_INTRA_THREADS,
    )
    logger.info("Warm-up …")
    detector.detect(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
    logger.info("Ready.")

    shared     = _SharedDetection()
    stop_event = threading.Event()
    frame_q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
    inf_thread = threading.Thread(
        target=_inference_thread,
        args=(detector, frame_q, shared, stop_event),
        daemon=True, name="InferenceThread",
    )
    inf_thread.start()

    db = CountDatabase(config.DB_PATH)
    db.start_session()

    writer: Optional[cv2.VideoWriter] = None
    if save:
        writer = cv2.VideoWriter(
            "output_annotated.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps, (frame_w, frame_h),
        )

    # ── Counter state (class so cleanup always has latest value) ──────────
    state = _CounterState()

    last_detection  = DetectionResult.empty()
    detect_skip     = 2
    tick            = 0
    paused          = False
    fps_actual      = 0.0
    fps_t0          = time.perf_counter()
    fps_frames      = 0
    frame_dur       = 1.0 / config.TARGET_FPS
    frames_per_sec  = max(1, int(src_fps))

    container_box:  Optional[BBox] = None
    container_miss  = 0

    WIN = "Warehouse Box Counter  [Q=quit  SPACE=pause  R=reset]"

    while True:
        t_start = time.perf_counter()
        tick   += 1

        key = cv2.pollKey() & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            container_box = None
            state.reset()
            # Flush current count to DB before resetting
            db.end_session(state.net_count)
            db.start_session()
            logger.info("Session reset.")

        if paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                logger.info("Video ended.")
                break
            continue

        # Feed inference thread
        if tick % detect_skip == 0:
            try:
                try: frame_q.get_nowait()
                except queue.Empty: pass
                frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        det, fresh = shared.read()
        if fresh:
            last_detection = det

        # ── Parse detections ──────────────────────────────────────────────
        persons_boxes: list[BBox] = []
        bigger_boxes:  list[BBox] = []
        small_boxes:   list[BBox] = []

        for i in range(last_detection.count):
            cls = int(last_detection.class_ids[i])
            box = tuple(float(v) for v in last_detection.boxes[i])
            if   cls == CLS_PERSON:     persons_boxes.append(box)
            elif cls == CLS_BIGGER_BOX: bigger_boxes.append(box)
            elif cls == CLS_BOXES:
                if box_area(box) >= min_box_area:
                    small_boxes.append(box)

        # ── Container ROI ─────────────────────────────────────────────────
        if bigger_boxes:
            container_box  = max(bigger_boxes, key=box_area)
            container_miss = 0
        else:
            container_miss += 1
            if container_miss > 30:
                container_box = None

        # ── Person blocking ───────────────────────────────────────────────
        person_blocking = (
            container_box is not None
            and any(
                intersection_area(pb, container_box) > 0
                for pb in persons_boxes
            )
        )

        # ── Hand regions (for annotation only) ────────────────────────────
        hand_pairs: list[tuple[BBox, BBox]] = [
            estimate_hand_regions(pb, frame_w, frame_h)
            for pb in persons_boxes
        ]

        # ── Visible count ─────────────────────────────────────────────────
        visible_in_carton = [
            sb for sb in small_boxes
            if container_box is not None and center_inside(sb, container_box)
        ]
        visible_now = len(visible_in_carton)

        # ── Peak / net count logic ────────────────────────────────────────
        # Rule 1: visible count went UP → update peak immediately
        if visible_now > state.peak_count:
            state.peak_count   = visible_now
            state.drop_holdoff = 0

        # Rule 2: person blocking → hold peak (stacking / adjusting)
        if person_blocking:
            state.net_count    = state.peak_count
            state.drop_holdoff = 0
        else:
            # Rule 3: no blocking → trust what we see
            if visible_now < state.peak_count:
                state.drop_holdoff += 1
                if state.drop_holdoff >= frames_per_sec:
                    # Sustained 1-second drop = real removal
                    state.peak_count   = visible_now
                    state.drop_holdoff = 0
            else:
                state.drop_holdoff = 0
            state.net_count = state.peak_count

        # ── Log to DB when count changes ──────────────────────────────────
        if state.net_count != state.last_logged:
            db.log_event("COUNT", 0, state.net_count)
            db.flush()
            state.last_logged = state.net_count

        # ── Annotate frame ────────────────────────────────────────────────
        annotated = frame.copy()

        # Carton ROI
        if container_box is not None:
            x1, y1, x2, y2 = map(int, container_box)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,128), 3)
            cv2.putText(annotated, "CARTON", (x1, y1-8),
                        FONT, 0.7, (0,255,128), 2)

        # Person boxes
        for pb in persons_boxes:
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (90,120,255), 2)

        # Hand regions
        for lp, rp in hand_pairs:
            for palm in (lp, rp):
                x1, y1, x2, y2 = map(int, palm)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,180,40), 1)

        # Small boxes — green=inside, grey=outside
        for sb in small_boxes:
            x1, y1, x2, y2 = map(int, sb)
            inside = container_box is not None and center_inside(sb, container_box)
            color  = (0, 230, 50) if inside else (180, 180, 180)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)

        # HUD
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0,0), (frame_w, 115), (18,18,18), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        status  = "PERSON BLOCKING" if person_blocking else "CLEAR"
        s_color = (0, 200, 255)     if person_blocking else (0, 230, 50)

        lines = [
            (f"Boxes in carton: {state.net_count}",                         1.0,  (255,255,255), 2),
            (f"Visible: {visible_now}   Peak: {state.peak_count}   [{status}]", 0.52, s_color,       1),
            (f"FPS: {fps_actual:.1f}",                                       0.50, (160,160,160), 1),
            ("SPACE=pause  R=reset  Q=quit",                                 0.42, (100,100,100), 1),
        ]
        for i, (text, size, color, thick) in enumerate(lines):
            cv2.putText(annotated, text, (10, 28+i*24),
                        FONT, size, color, thick)

        if paused:
            cv2.putText(annotated, "PAUSED",
                        (frame_w//2-70, frame_h//2), FONT, 1.6, (0,200,255), 3)

        cv2.imshow(WIN, _fit_to_screen(annotated))
        if writer:
            writer.write(annotated)

        # FPS measurement
        fps_frames += 1
        if fps_frames >= 60:
            fps_actual = fps_frames / (time.perf_counter()-fps_t0)
            fps_t0     = time.perf_counter()
            fps_frames = 0

        # Frame rate control
        if is_webcam:
            remaining = frame_dur - (time.perf_counter()-t_start)
            if remaining > 0.002:
                time.sleep(remaining-0.002)
            while (time.perf_counter()-t_start) < frame_dur:
                pass
        else:
            elapsed       = time.perf_counter() - t_start
            src_frame_dur = 1.0 / src_fps
            wait_ms       = max(1, int((src_frame_dur - elapsed) * 1000))
            cv2.waitKey(wait_ms)

    # ── Cleanup — state.net_count always has the final value ──────────────
    stop_event.set()
    try: frame_q.put_nowait(None)
    except queue.Full: pass
    inf_thread.join(timeout=3.0)
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    # Save final count to DB
    db.end_session(state.net_count)
    logger.info("Final count: %d", state.net_count)
    _print_summary(db)


def _fit_to_screen(frame: np.ndarray, max_w=900, max_h=900) -> np.ndarray:
    h, w  = frame.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    if scale >= 1.0: return frame
    return cv2.resize(frame, (int(w*scale), int(h*scale)))


def _print_summary(db: CountDatabase) -> None:
    sessions = db.get_session_summary()
    print("\n" + "─"*50)
    print("  SESSION SUMMARY")
    print("─"*50)
    for s in sessions[:5]:
        dur = f"  {s['ended_at']-s['started_at']:.0f}s" if s["ended_at"] else ""
        cnt = s['final_count'] if s['final_count'] is not None else "?"
        print(f"  Session {s['id']:>3}  |  count={cnt:>4}  |{dur}")
    print("─"*50 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default="models/best.onnx")
    ap.add_argument("--source",       default="0")
    ap.add_argument("--conf",         type=float, default=0.30)
    ap.add_argument("--imgsz",        type=int,   default=512)
    ap.add_argument("--save",         action="store_true")
    ap.add_argument("--min-box-area", type=float, default=80.0)
    args = ap.parse_args()

    source: str | int = args.source
    if args.source.isdigit():
        source = int(args.source)

    run(
        model_path=args.model,
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        min_box_area=args.min_box_area,
    )