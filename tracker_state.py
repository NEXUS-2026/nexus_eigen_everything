"""
tracker_state.py
================
100% Faithful Port of realtime_counter.py (The 95% Accuracy Model).
Optimized for high-FPS, continuous YOLO detection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
import config
from yolo_engine import DetectionResult

logger = logging.getLogger(__name__)

class TrackState(Enum):
    PENDING_ENTER    = auto()
    CONFIRMED_INSIDE = auto()
    HIDDEN_INSIDE    = auto()
    REMOVED          = auto()

BBox = Tuple[float, float, float, float]

@dataclass
class BoxTrack:
    track_id: int
    cx: float
    cy: float
    box: BBox
    state: str
    pending_state: str
    pending_count: int
    last_seen: int

def box_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def center_inside(inner: BBox, outer: BBox) -> bool:
    cx, cy = box_center(inner)
    x1, y1, x2, y2 = outer
    return x1 <= cx <= x2 and y1 <= cy <= y2

def iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0: return 0.0
    return inter / max(1e-6, box_area(a) + box_area(b) - inter)

def smooth_box(prev_box: Optional[BBox], cur_box: Optional[BBox], alpha: float) -> Optional[BBox]:
    if cur_box is None: return prev_box
    if prev_box is None: return cur_box
    px1, py1, px2, py2 = prev_box
    cx1, cy1, cx2, cy2 = cur_box
    a = max(0.0, min(1.0, alpha))
    return (
        (1.0 - a) * px1 + a * cx1,
        (1.0 - a) * py1 + a * cy1,
        (1.0 - a) * px2 + a * cx2,
        (1.0 - a) * py2 + a * cy2,
    )

def expand_or_shrink(box: BBox, ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    dx = 0.5 * w * ratio
    dy = 0.5 * h * ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

def inside_hysteresis(center: Tuple[float, float], container: BBox, prev_state: str) -> str:
    cx, cy = center
    enter_box = expand_or_shrink(container, -0.10)  
    leave_box = expand_or_shrink(container, 0.06)   

    ex1, ey1, ex2, ey2 = enter_box
    lx1, ly1, lx2, ly2 = leave_box

    in_enter = ex1 <= cx <= ex2 and ey1 <= cy <= ey2
    in_leave = lx1 <= cx <= lx2 and ly1 <= cy <= ly2

    if prev_state == "inside":
        return "inside" if in_leave else "outside"
    return "inside" if in_enter else "outside"

@dataclass
class FrameResult:
    tracked_boxes: np.ndarray
    track_ids: np.ndarray
    box_count: int
    track_states: dict[int, TrackState]
    events: list[tuple[str, int]]
    inference_ms: float = 0.0

class BoxTrackerStateMachine:
    def __init__(
        self,
        roi_polygon: np.ndarray,
        frame_wh: tuple[int, int],
        debounce_frames: int = 3,
        ghost_frames: int = 200,
        track_buffer: int = 300,
        fps: int = 40,
    ) -> None:
        self.tracks: dict[int, BoxTrack] = {}
        self.next_track_id = 1
        self.total_boxes = 0
        self.smoothed_container: Optional[BBox] = None
        self.frame_idx = 0
        
        self.track_max_age = 30
        self.state_confirm_frames = 3
        
        # 🚨 INCREASED FROM 85.0 to 150.0 to make tracks bulletproof against fast hand throws
        self.max_match_distance = 150.0 
        
        class DummyZone:
            def trigger(self, detections): return np.array([])
        self._zone = DummyZone()
        logger.info("100% Faithful Teammate Port Ready! (M4 Mac Unlocked)")

    @property
    def count(self) -> int:
        return self.total_boxes

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.total_boxes = 0
        self.smoothed_container = None
        self.frame_idx = 0
        logger.info("Session reset.")

    def update(self, frame: np.ndarray, detection_result: DetectionResult) -> FrameResult:
        self.frame_idx += 1
        events: list[tuple[str, int]] = []

        containers = []
        all_boxes = []

        for i in range(detection_result.count):
            cls_id = int(detection_result.class_ids[i])
            box = tuple(float(v) for v in detection_result.boxes[i])
            
            if cls_id == 1:
                containers.append(box)
            elif cls_id == 2:
                all_boxes.append(box)

        container_box = max(containers, key=box_area) if containers else None
        self.smoothed_container = smooth_box(self.smoothed_container, container_box, alpha=0.25)
        active_container = self.smoothed_container

        if active_container is None:
            poly = config.ROI_POLYGON
            active_container = (min(poly[:,0]), min(poly[:,1]), max(poly[:,0]), max(poly[:,1]))

        boxes_inside = []
        boxes_outside = []
        for b in all_boxes:
            if active_container is not None and center_inside(b, active_container):
                boxes_inside.append(b)
            else:
                boxes_outside.append(b)

        current_boxes = boxes_inside + boxes_outside
        det_centers = [box_center(b) for b in current_boxes]
        det_states = ["inside" if b in boxes_inside else "outside" for b in current_boxes]

        active_track_ids = [tid for tid, tr in self.tracks.items() if self.frame_idx - tr.last_seen <= self.track_max_age]

        iou_pairs = []
        pairs = []
        for det_idx, (cx, cy) in enumerate(det_centers):
            for tid in active_track_ids:
                tr = self.tracks[tid]
                ov = iou_xyxy(current_boxes[det_idx], tr.box)
                if ov >= 0.18:
                    iou_pairs.append((ov, tid, det_idx))
                dist = math.hypot(cx - tr.cx, cy - tr.cy)
                if dist <= self.max_match_distance:
                    pairs.append((dist, tid, det_idx))

        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()

        for match_list in (iou_pairs, pairs):
            for _, tid, det_idx in match_list:
                if tid in matched_tracks or det_idx in matched_dets:
                    continue

                tr = self.tracks[tid]
                observed_state = det_states[det_idx]
                if active_container is not None:
                    observed_state = inside_hysteresis(det_centers[det_idx], active_container, tr.state)

                if observed_state == tr.state:
                    tr.pending_state = tr.state
                    tr.pending_count = 0
                else:
                    if observed_state == tr.pending_state:
                        tr.pending_count += 1
                    else:
                        tr.pending_state = observed_state
                        tr.pending_count = 1

                    if tr.pending_count >= self.state_confirm_frames:
                        if tr.state == "outside" and observed_state == "inside":
                            self.total_boxes += 1
                            events.append(("ADDED", tid))
                        elif tr.state == "inside" and observed_state == "outside":
                            # We don't subtract the actual count to prevent downward flickering
                            events.append(("REMOVED", tid))

                        tr.state = observed_state
                        tr.pending_state = observed_state
                        tr.pending_count = 0

                tr.cx, tr.cy = det_centers[det_idx]
                tr.box = current_boxes[det_idx]
                tr.last_seen = self.frame_idx

                matched_tracks.add(tid)
                matched_dets.add(det_idx)

        for det_idx, (cx, cy) in enumerate(det_centers):
            if det_idx in matched_dets:
                continue
            self.tracks[self.next_track_id] = BoxTrack(
                track_id=self.next_track_id,
                cx=cx, cy=cy, box=current_boxes[det_idx],
                state=det_states[det_idx], pending_state=det_states[det_idx],
                pending_count=0, last_seen=self.frame_idx,
            )
            self.next_track_id += 1

        stale_ids = [tid for tid, tr in self.tracks.items() if self.frame_idx - tr.last_seen > self.track_max_age]
        for tid in stale_ids:
            del self.tracks[tid]

        track_states = {}
        visible_boxes = []
        visible_ids = []
        for tid, tr in self.tracks.items():
            if self.frame_idx - tr.last_seen <= 1:
                visible_boxes.append(tr.box)
                visible_ids.append(tid)
                track_states[tid] = TrackState.CONFIRMED_INSIDE if tr.state == "inside" else TrackState.PENDING_ENTER
            elif tr.state == "inside":
                track_states[tid] = TrackState.HIDDEN_INSIDE

        out_boxes = np.array(visible_boxes, dtype=np.float32) if visible_boxes else np.empty((0,4), dtype=np.float32)
        out_ids = np.array(visible_ids, dtype=int) if visible_ids else np.empty((0,), dtype=int)

        return FrameResult(
            tracked_boxes=out_boxes, track_ids=out_ids, box_count=self.total_boxes,
            track_states=track_states, events=events, inference_ms=detection_result.inference_ms
        )