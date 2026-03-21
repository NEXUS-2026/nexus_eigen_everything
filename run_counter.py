"""
Box counter using the custom-trained model (best.onnx).

Model classes:
    0 = Person
    1 = bigger_box  — the large open carton (auto ROI)
    2 = boxes       — small boxes being packed

Counting logic:
    - ADDED (+1):   A box transitions from outside → inside the carton
                    AND was seen near hands (grabbed) during transition.
    - REMOVED (-1): A box that was CONFIRMED inside the carton
                    transitions to outside AND stays outside for
                    REMOVAL_CONFIRM_FRAMES frames (avoids false removals
                    when hands briefly lift a box).
    - STACKED:      Boxes already inside that disappear (occluded by
                    stacking) are held in HIDDEN state — count not
                    decremented — for up to GHOST_FRAMES frames.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
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

FONT = cv2.FONT_HERSHEY_SIMPLEX

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

def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a,b)
    if inter == 0: return 0.0
    return inter / (box_area(a)+box_area(b)-inter+1e-6)

def expand_box(b: BBox, sx: float, sy: float, W: int, H: int) -> BBox:
    cx,cy = box_center(b)
    w=max(1.0,(b[2]-b[0])*sx); h=max(1.0,(b[3]-b[1])*sy)
    return (max(0.0,cx-w/2),max(0.0,cy-h/2),min(W,cx+w/2),min(H,cy+h/2))

def dist(a: tuple[float,float], b: tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(Enum):
    OUTSIDE   = auto()   # never been inside, or confirmed removed
    PENDING   = auto()   # inside ROI, debouncing before counting
    INSIDE    = auto()   # confirmed inside, counted +1
    HIDDEN    = auto()   # was INSIDE, disappeared (stacked/occluded)
    LEAVING   = auto()   # was INSIDE, now outside — confirming removal


@dataclass
class BoxTrack:
    track_id:       int
    box:            BBox
    state:          TrackState = TrackState.OUTSIDE
    seen_frames:    int = 1
    missed_frames:  int = 0
    inside_streak:  int = 0
    outside_streak: int = 0
    hidden_frames:  int = 0
    grabbed_recent: int = 0
    ever_outside:   bool = False    # only boxes seen outside first can be counted


# ---------------------------------------------------------------------------
# Tuning constants — adjust these to fix over/under counting
# ---------------------------------------------------------------------------

INSIDE_CONFIRM_FRAMES  = 3    # frames inside ROI before counting +1
REMOVAL_CONFIRM_FRAMES = 8    # frames outside ROI before counting -1
GHOST_FRAMES           = 90   # frames a HIDDEN track is kept before -1
GRAB_DECAY             = 8    # frames grabbed_recent stays hot after release
TRACK_MAX_MISS         = 25   # frames before a track is purged entirely
IOU_MATCH_THRESH       = 0.25 # IoU threshold for track-detection matching


# ---------------------------------------------------------------------------
# Hand region estimator
# ---------------------------------------------------------------------------

def estimate_hand_regions(person_box: BBox, W: int, H: int) -> tuple[BBox, BBox]:
    x1,y1,x2,y2 = person_box
    pw=max(1.0,x2-x1); ph=max(1.0,y2-y1)
    # Use a tighter palm region than before — reduces false grabs
    palm_w=0.18*pw; palm_h=0.18*ph
    palm_y=y1+0.60*ph

    def clamp(b:BBox)->BBox:
        return (max(0,b[0]),max(0,b[1]),min(W,b[2]),min(H,b[3]))

    left_cx  = x1+0.26*pw
    right_cx = x2-0.26*pw
    left  = clamp((left_cx -palm_w/2, palm_y-palm_h/2, left_cx +palm_w/2, palm_y+palm_h/2))
    right = clamp((right_cx-palm_w/2, palm_y-palm_h/2, right_cx+palm_w/2, palm_y+palm_h/2))
    return left, right


def is_box_grabbed(
    box: BBox,
    left_palm: BBox, right_palm: BBox,
    W: int, H: int,
    container: Optional[BBox],
) -> bool:
    """
    Stricter grab detection than before.
    Requires the box center to be within expanded palm regions
    AND significant overlap — reduces the mass false-grab problem.
    """
    b_area = max(1.0, box_area(box))
    c      = box_center(box)

    # Expand palms slightly for detection
    le = expand_box(left_palm,  1.30, 1.30, W, H)
    re = expand_box(right_palm, 1.30, 1.30, W, H)

    lt = intersection_area(box, le) / b_area
    rt = intersection_area(box, re) / b_area

    near_left  = center_inside((c[0],c[1],c[0],c[1]), le)
    near_right = center_inside((c[0],c[1],c[0],c[1]), re)

    lw = max(1.0, left_palm[2]  - left_palm[0])
    rw = max(1.0, right_palm[2] - right_palm[0])

    # Stricter thresholds: 0.22 overlap (was 0.16) + distance check
    left_ok  = near_left  and dist(c,box_center(left_palm))  <= 1.6*lw and lt >= 0.22
    right_ok = near_right and dist(c,box_center(right_palm)) <= 1.6*rw and rt >= 0.22

    return left_ok or right_ok


# ---------------------------------------------------------------------------
# Shared detection (inference thread → main thread)
# ---------------------------------------------------------------------------

class _SharedDetection:
    def __init__(self) -> None:
        self._lock=threading.Lock(); self._det=DetectionResult.empty(); self._fresh=False

    def write(self, det: DetectionResult) -> None:
        with self._lock: self._det,self._fresh = det,True

    def read(self) -> tuple[DetectionResult, bool]:
        with self._lock:
            det,fresh=self._det,self._fresh; self._fresh=False
        return det,fresh


def _inference_thread(detector, frame_q, shared, stop_event):
    logger.info("Inference thread started.")
    while not stop_event.is_set():
        try: frame = frame_q.get(timeout=0.5)
        except queue.Empty: continue
        if frame is None: break
        shared.write(detector.detect(frame))
    logger.info("Inference thread exiting.")


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
        logger.error("Cannot open: %s", source); return

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

    shared=_SharedDetection(); stop_event=threading.Event()
    frame_q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
    inf_thread = threading.Thread(
        target=_inference_thread,
        args=(detector,frame_q,shared,stop_event),
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

    # ── State ──────────────────────────────────────────────────────────────
    last_detection  = DetectionResult.empty()
    detect_skip     = 2
    tick            = 0
    paused          = False
    fps_actual      = 0.0
    fps_t0          = time.perf_counter()
    fps_frames      = 0
    frame_dur       = 1.0 / config.TARGET_FPS

    active_tracks:    list[BoxTrack] = []
    next_track_id     = 1
    added_total       = 0    # boxes placed IN
    removed_total     = 0    # boxes taken OUT
    container_box: Optional[BBox] = None
    container_miss    = 0

    WIN = "Warehouse Box Counter  [Q=quit  SPACE=pause  R=reset]"

    while True:
        t_start = time.perf_counter()
        tick   += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord(' '): paused = not paused
        elif key == ord('r'):
            active_tracks=[]; next_track_id=1
            added_total=0; removed_total=0; container_box=None
            db.end_session(added_total); db.start_session()
            logger.info("Session reset.")

        if paused:
            time.sleep(0.05); continue

        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                logger.info("Video ended.")
                break   # exit the loop cleanly instead of looping
            continue

        # Feed inference thread
        if tick % detect_skip == 0:
            try:
                try: frame_q.get_nowait()
                except queue.Empty: pass
                frame_q.put_nowait(frame.copy())
            except queue.Full: pass

        det, fresh = shared.read()
        if fresh: last_detection = det

        # ── Parse detections by class ───────────────────────────────────────
        persons_boxes: list[BBox] = []
        bigger_boxes:  list[BBox] = []
        small_boxes:   list[BBox] = []

        for i in range(last_detection.count):
            cls = int(last_detection.class_ids[i])
            box = tuple(float(v) for v in last_detection.boxes[i])
            if   cls == CLS_PERSON:      persons_boxes.append(box)
            elif cls == CLS_BIGGER_BOX:  bigger_boxes.append(box)
            elif cls == CLS_BOXES:
                if box_area(box) >= min_box_area:
                    small_boxes.append(box)

        # ── Container ROI (hold for 30 frames if not detected) ─────────────
        if bigger_boxes:
            container_box  = max(bigger_boxes, key=box_area)
            container_miss = 0
        else:
            container_miss += 1
            if container_miss > 30:
                container_box = None

        # ── Hand regions + grab detection ───────────────────────────────────
        hand_pairs: list[tuple[BBox,BBox]] = [
            estimate_hand_regions(pb, frame_w, frame_h)
            for pb in persons_boxes
        ]

        grabbed_boxes: list[BBox] = []
        for sb in small_boxes:
            for lp,rp in hand_pairs:
                if is_box_grabbed(sb, lp, rp, frame_w, frame_h, container_box):
                    grabbed_boxes.append(sb)
                    break

        # ── Track matching ──────────────────────────────────────────────────
        track_matched = [False]*len(active_tracks)
        det_matched   = [False]*len(small_boxes)

        for di, db_box in enumerate(small_boxes):
            best_i, best_iou = -1, 0.0
            for ti, tr in enumerate(active_tracks):
                if track_matched[ti]: continue
                s = iou(db_box, tr.box)
                if s > best_iou: best_iou=s; best_i=ti

            if best_i >= 0 and best_iou >= IOU_MATCH_THRESH:
                tr = active_tracks[best_i]
                tr.box          = db_box
                tr.seen_frames += 1
                tr.missed_frames= 0

                inside_now  = container_box is not None and center_inside(db_box, container_box)
                grabbed_now = any(iou(db_box,gb)>=0.25 for gb in grabbed_boxes)
                tr.grabbed_recent = GRAB_DECAY if grabbed_now else max(0,tr.grabbed_recent-1)

                # ── State transitions ───────────────────────────────────────

                # ── State transitions ───────────────────────────────────────
                # Mark ever_outside — a box seen outside the carton
                # is the only kind that can be counted as a new addition.
                if not inside_now:
                    tr.ever_outside = True

                if tr.state == TrackState.OUTSIDE:
                    if inside_now and tr.ever_outside:
                        # Saw it outside first — start debounce window
                        tr.state = TrackState.PENDING
                        tr.inside_streak = 1
                    elif inside_now and not tr.ever_outside:
                        # Appeared directly inside — already there at session
                        # start. Track it silently, do NOT count it.
                        tr.state = TrackState.INSIDE

                elif tr.state == TrackState.PENDING:
                    if inside_now:
                        tr.inside_streak += 1
                        if tr.inside_streak >= INSIDE_CONFIRM_FRAMES:
                            if tr.ever_outside and tr.grabbed_recent > 0:
                                # Confirmed: came from outside + was grabbed
                                tr.state    = TrackState.INSIDE
                                added_total += 1
                                logger.info("BOX IN  | total=%d | track=%d",
                                            added_total, tr.track_id)
                                db.log_event("ADDED", tr.track_id, added_total)
                                db.flush()
                            else:
                                # Inside long enough but no grab detected —
                                # was probably already there. Track silently.
                                tr.state = TrackState.INSIDE
                    else:
                        # Left before debounce completed — reset
                        tr.state = TrackState.OUTSIDE
                        tr.inside_streak = 0

                elif tr.state == TrackState.INSIDE:
                    if not inside_now:
                        # Start removal confirmation window
                        tr.state = TrackState.LEAVING
                        tr.outside_streak = 1
                    # If still inside: stay INSIDE (stacking handled by HIDDEN)

                elif tr.state == TrackState.LEAVING:
                    if inside_now:
                        # Came back in — was just briefly lifted (stacking)
                        tr.state = TrackState.INSIDE
                        tr.outside_streak = 0
                    else:
                        tr.outside_streak += 1
                        if tr.outside_streak >= REMOVAL_CONFIRM_FRAMES:
                            # Confirmed removal
                            tr.state = TrackState.OUTSIDE
                            removed_total += 1
                            logger.info("BOX OUT | total=%d | track=%d", removed_total, tr.track_id)
                            db.log_event("REMOVED", tr.track_id, added_total-removed_total)
                            db.flush()

                elif tr.state == TrackState.HIDDEN:
                    # Track reappeared — decide where
                    tr.hidden_frames = 0
                    if inside_now:
                        tr.state = TrackState.INSIDE   # relink, count held
                    else:
                        tr.state = TrackState.LEAVING
                        tr.outside_streak = 1

                track_matched[best_i] = True
                det_matched[di]       = True

        # ── Handle unmatched tracks ─────────────────────────────────────────
        for ti, tr in enumerate(active_tracks):
            if track_matched[ti]: continue
            tr.missed_frames   += 1
            tr.grabbed_recent   = max(0, tr.grabbed_recent-1)

            if tr.state == TrackState.INSIDE:
                # Disappeared while inside → stacked/occluded
                tr.state = TrackState.HIDDEN
                tr.hidden_frames = 1
                logger.debug("Track %d → HIDDEN (stacked)", tr.track_id)

            elif tr.state == TrackState.HIDDEN:
                tr.hidden_frames += 1
                if tr.hidden_frames > GHOST_FRAMES:
                    # Occlusion guard expired — box truly gone
                    removed_total += 1
                    logger.info("GHOST EXPIRED | track=%d | removed=%d",
                                tr.track_id, removed_total)
                    db.log_event("REMOVED", tr.track_id, added_total-removed_total)
                    db.flush()
                    tr.state = TrackState.OUTSIDE   # will be purged below

            elif tr.state == TrackState.PENDING:
                tr.state = TrackState.OUTSIDE  # never confirmed

        # ── Create new tracks for unmatched detections ──────────────────────
        for di, db_box in enumerate(small_boxes):
            if det_matched[di]: continue
            inside_now = container_box is not None and center_inside(db_box, container_box)
            active_tracks.append(BoxTrack(
                track_id      = next_track_id,
                box           = db_box,
                # Boxes that appear directly inside = already there, track silently
                # Boxes that appear outside = eligible to be counted later
                state         = TrackState.INSIDE if inside_now else TrackState.OUTSIDE,
                inside_streak = 1 if inside_now else 0,
                ever_outside  = not inside_now,
            ))
            next_track_id += 1

        # ── Purge dead tracks ───────────────────────────────────────────────
        active_tracks = [
            tr for tr in active_tracks
            if tr.missed_frames <= TRACK_MAX_MISS
            and tr.state != TrackState.OUTSIDE or tr.missed_frames == 0
        ]

        # ── Net count ──────────────────────────────────────────────────────
        net_count = max(0, added_total - removed_total)

        # ── Annotate ───────────────────────────────────────────────────────
        annotated = frame.copy()

        # Container ROI
        if container_box is not None:
            x1,y1,x2,y2 = map(int, container_box)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,128),3)
            cv2.putText(annotated,"CARTON",(x1,y1-8),FONT,0.6,(0,255,128),2)

        # Person boxes
        for pb in persons_boxes:
            x1,y1,x2,y2 = map(int,pb)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(90,120,255),2)

        # Hand regions (draw smaller now)
        for lp,rp in hand_pairs:
            for palm in (lp,rp):
                x1,y1,x2,y2 = map(int,palm)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,180,40),1)

        # Small boxes — colour by track state
        state_colours = {
            TrackState.OUTSIDE: (180,180,180),
            TrackState.PENDING: (0,200,255),
            TrackState.INSIDE:  (0,230,50),
            TrackState.HIDDEN:  (255,120,0),
            TrackState.LEAVING: (0,80,255),
        }

        # Build a box→state lookup from active tracks
        track_states: dict[int, TrackState] = {}
        for tr in active_tracks:
            track_states[tr.track_id] = tr.state

        for sb in small_boxes:
            is_grabbed = any(iou(sb,gb)>=0.25 for gb in grabbed_boxes)
            if is_grabbed:
                color,thick = (0,0,255),3
            else:
                # Find the track state for this box
                best_tr = max(
                    (tr for tr in active_tracks),
                    key=lambda tr: iou(sb,tr.box),
                    default=None,
                )
                state  = best_tr.state if best_tr else TrackState.OUTSIDE
                color  = state_colours.get(state,(180,180,180))
                thick  = 2
            x1,y1,x2,y2 = map(int,sb)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,thick)

        # HUD
        overlay = annotated.copy()
        cv2.rectangle(overlay,(0,0),(frame_w,148),(18,18,18),-1)
        cv2.addWeighted(overlay,0.55,annotated,0.45,0,annotated)

        visible_in = sum(
            1 for sb in small_boxes
            if container_box and center_inside(sb,container_box)
        )
        hidden_count = sum(
            1 for tr in active_tracks if tr.state==TrackState.HIDDEN
        )

        lines = [
            (f"Net boxes in carton: {net_count}", 1.0, (255,255,255), 2),
            (f"Added: {added_total}   Removed: {removed_total}", 0.65, (180,255,180), 1),
            (f"Visible: {visible_in}   Stacked/hidden: {hidden_count}", 0.60, (180,180,255), 1),
            (f"FPS: {fps_actual:.1f}   Tracks: {len(active_tracks)}", 0.55, (160,160,160), 1),
            ("SPACE=pause  R=reset  Q=quit", 0.42, (100,100,100), 1),
        ]
        for i,(text,size,color,thick) in enumerate(lines):
            cv2.putText(annotated,text,(10,28+i*26),FONT,size,color,thick)

        # Colour legend (bottom-right)
        legend = [
            ("grabbed",  (0,0,255)),
            ("inside",   (0,230,50)),
            ("pending",  (0,200,255)),
            ("leaving",  (0,80,255)),
            ("hidden",   (255,120,0)),
        ]
        for i,(label,color) in enumerate(legend):
            lx = frame_w - 130
            ly = 20 + i*22
            cv2.rectangle(annotated,(lx,ly-12),(lx+16,ly+4),color,-1)
            cv2.putText(annotated,label,(lx+22,ly),FONT,0.42,(200,200,200),1)

        if paused:
            cv2.putText(annotated,"PAUSED",
                        (frame_w//2-70,frame_h//2),FONT,1.6,(0,200,255),3)

        cv2.imshow(WIN, _fit_to_screen(annotated))
        if writer: writer.write(annotated)

        fps_frames += 1
        if fps_frames >= 60:
            fps_actual = fps_frames/(time.perf_counter()-fps_t0)
            fps_t0=time.perf_counter(); fps_frames=0

        remaining = frame_dur-(time.perf_counter()-t_start)
        if remaining > 0.002: time.sleep(remaining-0.002)
        while (time.perf_counter()-t_start) < frame_dur: pass

    # Cleanup
    stop_event.set()
    try: frame_q.put_nowait(None)
    except queue.Full: pass
    inf_thread.join(timeout=3.0)
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    db.end_session(max(0,added_total-removed_total))
    logger.info("Final — added=%d  removed=%d  net=%d",
                added_total, removed_total, added_total-removed_total)
    _print_summary(db)


def _fit_to_screen(frame: np.ndarray, max_w=900, max_h=900) -> np.ndarray:
    h,w=frame.shape[:2]; scale=min(max_w/w,max_h/h,1.0)
    if scale>=1.0: return frame
    return cv2.resize(frame,(int(w*scale),int(h*scale)))


def _print_summary(db: CountDatabase) -> None:
    sessions = db.get_session_summary()
    print("\n"+"─"*55)
    print("  SESSION SUMMARY")
    print("─"*55)
    for s in sessions[:5]:
        dur = f"  {s['ended_at']-s['started_at']:.0f}s" if s["ended_at"] else ""
        print(f"  Session {s['id']:>3}  |  net={s['final_count'] or '?':>4}  |{dur}")
    print("─"*55+"\n")


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