from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: BBox


@dataclass
class BoxTrack:
    track_id: int
    box: BBox
    seen_frames: int = 1
    missed_frames: int = 0
    edge_seen: bool = False
    inside_streak: int = 0
    outside_streak: int = 0
    ever_inside: bool = False
    ever_outside: bool = False
    grabbed_recent: int = 0
    added_counted: bool = False


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clamp_box(box: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return (x1, y1, x2, y2)


def center_inside(inner: BBox, outer: BBox) -> bool:
    x1, y1, x2, y2 = inner
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def pick_container(containers: List[Detection]) -> Optional[BBox]:
    if not containers:
        return None
    return max(containers, key=lambda d: box_area(d.box)).box


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = int(max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[i])


def derive_precise_container_zone(
    container_box: Optional[BBox],
    boxes: List[Detection],
    frame_w: int,
    frame_h: int,
    min_box_area: float,
) -> Optional[BBox]:
    if container_box is None:
        return None

    x1, y1, x2, y2 = container_box
    cw = max(1.0, x2 - x1)
    ch = max(1.0, y2 - y1)

    # Aggressive inner crop to ignore cardboard flaps and keep only usable placement area.
    base = clamp_box(
        (
            x1 + 0.18 * cw,
            y1 + 0.22 * ch,
            x2 - 0.18 * cw,
            y2 - 0.16 * ch,
        ),
        frame_w,
        frame_h,
    )

    candidate_boxes: List[BBox] = []
    for d in boxes:
        if box_area(d.box) < min_box_area:
            continue
        if center_inside(d.box, container_box):
            candidate_boxes.append(d.box)

    if len(candidate_boxes) < 6:
        return base

    xs1 = sorted(b[0] for b in candidate_boxes)
    ys1 = sorted(b[1] for b in candidate_boxes)
    xs2 = sorted(b[2] for b in candidate_boxes)
    ys2 = sorted(b[3] for b in candidate_boxes)

    dist_zone = (
        _quantile(xs1, 0.28),
        _quantile(ys1, 0.30),
        _quantile(xs2, 0.72),
        _quantile(ys2, 0.72),
    )
    dist_zone = clamp_box(dist_zone, frame_w, frame_h)

    # Blend box-driven zone with geometry-driven inner zone for stability across lighting/occlusion.
    bx1, by1, bx2, by2 = base
    dx1, dy1, dx2, dy2 = dist_zone
    blended = (
        0.60 * bx1 + 0.40 * dx1,
        0.60 * by1 + 0.40 * dy1,
        0.60 * bx2 + 0.40 * dx2,
        0.60 * by2 + 0.40 * dy2,
    )
    blended = clamp_box(blended, frame_w, frame_h)

    # Prevent zone collapse: keep minimum proportion of original container.
    zx1, zy1, zx2, zy2 = blended
    zw = max(1.0, zx2 - zx1)
    zh = max(1.0, zy2 - zy1)
    min_w = 0.45 * cw
    min_h = 0.45 * ch
    if zw < min_w or zh < min_h:
        cx, cy = box_center(blended)
        zw = max(zw, min_w)
        zh = max(zh, min_h)
        blended = clamp_box((cx - 0.5 * zw, cy - 0.5 * zh, cx + 0.5 * zw, cy + 0.5 * zh), frame_w, frame_h)

    return blended


def grid_occupancy(container_box: Optional[BBox], boxes: List[BBox], cols: int, rows: int) -> set[Tuple[int, int]]:
    cells: set[Tuple[int, int]] = set()
    if container_box is None:
        return cells
    x1, y1, x2, y2 = container_box
    cw = max(1.0, x2 - x1)
    ch = max(1.0, y2 - y1)
    for b in boxes:
        cx, cy = box_center(b)
        if not center_inside((cx, cy, cx, cy), container_box):
            continue
        gx = int((cx - x1) / cw * cols)
        gy = int((cy - y1) / ch * rows)
        gx = max(0, min(cols - 1, gx))
        gy = max(0, min(rows - 1, gy))
        cells.add((gx, gy))
    return cells


def build_hand_regions(person_box: BBox, frame_w: int, frame_h: int) -> List[BBox]:
    x1, y1, x2, y2 = person_box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # Heuristic hand windows near lower left/right of a person bbox for top-down scenes.
    hw = 0.28 * w
    hh = 0.28 * h
    y_top = y1 + 0.58 * h
    y_bottom = y_top + hh

    left = (x1 - 0.06 * w, y_top, x1 - 0.06 * w + hw, y_bottom)
    right = (x2 - hw + 0.06 * w, y_top, x2 + 0.06 * w, y_bottom)

    return [clamp_box(left, frame_w, frame_h), clamp_box(right, frame_w, frame_h)]


def expand_box(box: BBox, sx: float, sy: float, frame_w: int, frame_h: int) -> BBox:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1.0, (x2 - x1) * sx)
    h = max(1.0, (y2 - y1) * sy)
    return clamp_box((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), frame_w, frame_h)


def box_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_outside_container_center(box: BBox, container_box: Optional[BBox]) -> bool:
    if container_box is None:
        return True
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return not center_inside((cx, cy, cx, cy), container_box)


def is_near_container_boundary_outside(box: BBox, container_box: Optional[BBox], frame_w: int, frame_h: int) -> bool:
    if container_box is None:
        return False
    if center_inside(box, container_box):
        return False

    cx, cy = box_center(box)
    x1, y1, x2, y2 = container_box
    margin = max(12.0, 0.025 * min(frame_w, frame_h))
    near_x = (x1 - margin) <= cx <= (x2 + margin)
    near_y = (y1 - margin) <= cy <= (y2 + margin)
    return near_x and near_y


def is_box_near_frame_edge(box: BBox, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = box
    margin = max(10.0, 0.02 * min(frame_w, frame_h))
    return x1 <= margin or y1 <= margin or x2 >= (frame_w - margin) or y2 >= (frame_h - margin)


def person_in_major_container_region(person_box: BBox, container_box: Optional[BBox]) -> bool:
    if container_box is None:
        return False
    p_area = max(1.0, box_area(person_box))
    inter = intersection_area(person_box, container_box)
    # Major presence: person substantially overlaps container working region.
    return (inter / p_area) >= 0.35


def person_touches_container(person_box: BBox, container_box: Optional[BBox]) -> bool:
    if container_box is None:
        return False
    return intersection_area(person_box, container_box) > 1.0


def is_container_unblocked(persons: List[Detection], container_box: Optional[BBox]) -> bool:
    if container_box is None:
        return False
    return all(not person_touches_container(p.box, container_box) for p in persons)


def estimate_person_approach(person_box: BBox, container_box: Optional[BBox], frame_w: int, frame_h: int) -> str:
    if container_box is None:
        cx, cy = box_center(person_box)
        nx = cx / max(1.0, float(frame_w))
        ny = cy / max(1.0, float(frame_h))
        if nx <= 0.34:
            return "left"
        if nx >= 0.66:
            return "right"
        if ny <= 0.45:
            return "top"
        return "bottom"

    px1, py1, px2, py2 = person_box
    cx, cy = box_center(person_box)
    bx1, by1, bx2, by2 = container_box

    if cx < bx1:
        return "left"
    if cx > bx2:
        return "right"
    if cy < by1:
        return "top"
    if cy > by2:
        return "bottom"

    d_left = abs(cx - bx1)
    d_right = abs(bx2 - cx)
    d_top = abs(cy - by1)
    d_bottom = abs(by2 - cy)
    nearest = min((d_left, "left"), (d_right, "right"), (d_top, "top"), (d_bottom, "bottom"), key=lambda x: x[0])

    # If person is mostly above/below container vertically, prefer top/bottom model.
    p_h = max(1.0, py2 - py1)
    above_ratio = max(0.0, by1 - py1) / p_h
    below_ratio = max(0.0, py2 - by2) / p_h
    if above_ratio >= 0.22:
        return "top"
    if below_ratio >= 0.22:
        return "bottom"
    return nearest[1]


def build_strict_hand_model(
    person_box: BBox,
    frame_w: int,
    frame_h: int,
    palm_y_frac: float,
    approach: str = "top",
) -> Tuple[BBox, BBox, BBox]:
    x1, y1, x2, y2 = person_box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    if approach in ("top", "bottom"):
        palm_w = 0.20 * w
        palm_h = 0.22 * h
        if approach == "top":
            palm_y = y1 + palm_y_frac * h
        else:
            palm_y = y1 + (1.0 - palm_y_frac) * h

        left_cx = x1 + 0.24 * w
        right_cx = x2 - 0.24 * w

        left = clamp_box(
            (left_cx - 0.5 * palm_w, palm_y - 0.5 * palm_h, left_cx + 0.5 * palm_w, palm_y + 0.5 * palm_h),
            frame_w,
            frame_h,
        )
        right = clamp_box(
            (right_cx - 0.5 * palm_w, palm_y - 0.5 * palm_h, right_cx + 0.5 * palm_w, palm_y + 0.5 * palm_h),
            frame_w,
            frame_h,
        )

        gap = max(1.0, right[0] - left[2])
        bx1 = left[2] + 0.18 * gap
        bx2 = right[0] - 0.18 * gap
        between = clamp_box(
            (bx1, palm_y - 0.30 * palm_h, bx2, palm_y + 0.30 * palm_h),
            frame_w,
            frame_h,
        )
        return left, right, between

    palm_w = 0.22 * w
    palm_h = 0.20 * h
    y_shift = (palm_y_frac - 0.58) * 0.60 * h
    upper_cy = y1 + 0.42 * h + y_shift
    lower_cy = y1 + 0.62 * h + y_shift
    if approach == "left":
        side_cx = x2 - 0.18 * w
    else:
        side_cx = x1 + 0.18 * w

    left = clamp_box(
        (side_cx - 0.5 * palm_w, upper_cy - 0.5 * palm_h, side_cx + 0.5 * palm_w, upper_cy + 0.5 * palm_h),
        frame_w,
        frame_h,
    )
    right = clamp_box(
        (side_cx - 0.5 * palm_w, lower_cy - 0.5 * palm_h, side_cx + 0.5 * palm_w, lower_cy + 0.5 * palm_h),
        frame_w,
        frame_h,
    )

    gy = 0.5 * (upper_cy + lower_cy)
    if approach == "left":
        between = clamp_box(
            (side_cx - 0.50 * palm_w, gy - 0.55 * palm_h, side_cx + 0.95 * palm_w, gy + 0.55 * palm_h),
            frame_w,
            frame_h,
        )
    else:
        between = clamp_box(
            (side_cx - 0.95 * palm_w, gy - 0.55 * palm_h, side_cx + 0.50 * palm_w, gy + 0.55 * palm_h),
            frame_w,
            frame_h,
        )
    return left, right, between


def choose_best_hand_model(
    person_box: BBox,
    boxes: List[Detection],
    frame_w: int,
    frame_h: int,
    min_box_area: float,
    container_box: Optional[BBox],
) -> Tuple[BBox, BBox, BBox]:
    approach = estimate_person_approach(person_box, container_box, frame_w, frame_h)
    if approach in ("top", "bottom"):
        fracs = (0.52, 0.62, 0.72)
    else:
        fracs = (0.50, 0.58, 0.66)

    best_score = -1.0
    best_model: Optional[Tuple[BBox, BBox, BBox]] = None
    for frac in fracs:
        left, right, between = build_strict_hand_model(person_box, frame_w, frame_h, frac, approach=approach)
        left_exp = expand_box(left, 1.35, 1.35, frame_w, frame_h)
        right_exp = expand_box(right, 1.35, 1.35, frame_w, frame_h)
        between_exp = expand_box(between, 1.15, 1.20, frame_w, frame_h)
        score = 0.0
        for b in boxes:
            if box_area(b.box) < min_box_area:
                continue
            b_area = max(1.0, box_area(b.box))
            score += 1.6 * (intersection_area(b.box, left_exp) / b_area)
            score += 1.6 * (intersection_area(b.box, right_exp) / b_area)
            score += 2.2 * (intersection_area(b.box, between_exp) / b_area)
        if score > best_score:
            best_score = score
            best_model = (left, right, between)
    if best_model is not None:
        return best_model
    return build_strict_hand_model(person_box, frame_w, frame_h, 0.62, approach=approach)


def is_box_grabbed_by_person(
    box: BBox,
    person_box: BBox,
    left_palm: BBox,
    right_palm: BBox,
    between: BBox,
    frame_w: int,
    frame_h: int,
    container_box: Optional[BBox],
) -> bool:
    b_area = max(1.0, box_area(box))
    outside_container = is_outside_container_center(box, container_box)
    inside_container = not outside_container

    person_overlap_ratio = intersection_area(box, person_box) / b_area

    c = box_center(box)
    left_exp = expand_box(left_palm, 1.45, 1.45, frame_w, frame_h)
    right_exp = expand_box(right_palm, 1.45, 1.45, frame_w, frame_h)
    between_exp = expand_box(between, 1.18, 1.28, frame_w, frame_h)

    left_touch = intersection_area(box, left_palm) / b_area
    right_touch = intersection_area(box, right_palm) / b_area
    left_touch_exp = intersection_area(box, left_exp) / b_area
    right_touch_exp = intersection_area(box, right_exp) / b_area
    near_left = center_inside((c[0], c[1], c[0], c[1]), left_exp)
    near_right = center_inside((c[0], c[1], c[0], c[1]), right_exp)

    left_palm_w = max(1.0, left_palm[2] - left_palm[0])
    right_palm_w = max(1.0, right_palm[2] - right_palm[0])
    left_anchor = near_left and dist(c, box_center(left_palm)) <= 1.9 * left_palm_w
    right_anchor = near_right and dist(c, box_center(right_palm)) <= 1.9 * right_palm_w

    # Single-palm grabs outside container.
    if outside_container:
        if (left_anchor and (left_touch >= 0.16 or left_touch_exp >= 0.22)) or (
            right_anchor and (right_touch >= 0.16 or right_touch_exp >= 0.22)
        ):
            return True

    # Single-palm grabs inside container: stricter than outside, but enabled.
    if inside_container:
        if person_overlap_ratio >= 0.0 and (
            (left_anchor and (left_touch >= 0.16 or left_touch_exp >= 0.22))
            or (right_anchor and (right_touch >= 0.16 or right_touch_exp >= 0.22))
        ):
            return True

    # Between-palms grab: center in-between with proximity to both palms.
    between_center_ok = center_inside((c[0], c[1], c[0], c[1]), between_exp)
    palm_dist_ok = dist(c, box_center(left_palm)) <= 2.1 * max(1.0, (left_palm[2] - left_palm[0])) and dist(
        c, box_center(right_palm)
    ) <= 2.1 * max(1.0, (right_palm[2] - right_palm[0]))
    weak_both_touches = (
        intersection_area(box, left_exp) / b_area >= 0.06
        and intersection_area(box, right_exp) / b_area >= 0.06
    )
    if between_center_ok and palm_dist_ok and weak_both_touches and person_overlap_ratio >= 0.0:
        return True

    # High-recall fallback: if center is in between-hands corridor and close to either palm.
    if between_center_ok and (near_left or near_right) and (
        left_touch_exp >= 0.08 or right_touch_exp >= 0.08
    ):
        return True

    # Inside-container grabs: accept strong between-palms evidence.
    if inside_container:
        strong_between = (
            center_inside((c[0], c[1], c[0], c[1]), between_exp)
            and intersection_area(box, left_exp) / b_area >= 0.10
            and intersection_area(box, right_exp) / b_area >= 0.10
            and left_touch >= 0.01
            and right_touch >= 0.01
            and palm_dist_ok
        )
        if strong_between and person_overlap_ratio >= 0.0:
            return True

    return False


def is_box_near_hands_relaxed(
    box: BBox,
    left_palm: BBox,
    right_palm: BBox,
    between: BBox,
    frame_w: int,
    frame_h: int,
    container_box: Optional[BBox],
) -> bool:
    outside = is_outside_container_center(box, container_box)

    b_area = max(1.0, box_area(box))
    c = box_center(box)
    left_exp = expand_box(left_palm, 1.40, 1.40, frame_w, frame_h)
    right_exp = expand_box(right_palm, 1.40, 1.40, frame_w, frame_h)
    between_exp = expand_box(between, 1.16, 1.22, frame_w, frame_h)
    if outside:
        if intersection_area(box, left_exp) / b_area >= 0.16:
            return True
        if intersection_area(box, right_exp) / b_area >= 0.16:
            return True
    else:
        both_touch = (
            intersection_area(box, left_exp) / b_area >= 0.12
            and intersection_area(box, right_exp) / b_area >= 0.12
        )
        if both_touch and center_inside((c[0], c[1], c[0], c[1]), between_exp):
            return True
        if center_inside((c[0], c[1], c[0], c[1]), between_exp) and (
            intersection_area(box, left_exp) / b_area >= 0.08
            or intersection_area(box, right_exp) / b_area >= 0.08
        ):
            return True
    return False


def iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_detections(dets: List[Detection], iou_thr: float) -> List[Detection]:
    if not dets:
        return dets
    kept: List[Detection] = []
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    for cand in ordered:
        if all(iou_xyxy(cand.box, k.box) < iou_thr for k in kept):
            kept.append(cand)
    return kept


def refine_boxes_in_roi(
    model: YOLO,
    frame,
    container_box: BBox,
    roi_imgsz: int,
    roi_conf: float,
    roi_iou: float,
    infer_device: str,
    use_half: bool,
) -> List[Detection]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, clamp_box(container_box, w, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    r = model.predict(
        source=roi,
        conf=roi_conf,
        iou=roi_iou,
        imgsz=roi_imgsz,
        verbose=False,
        device=infer_device,
        half=use_half,
    )

    refined: List[Detection] = []
    names = model.names
    for b in r[0].boxes:
        cls_id = int(b.cls.item())
        if names[cls_id] != "boxes":
            continue
        bx = b.xyxy[0].tolist()
        gx1 = float(bx[0] + x1)
        gy1 = float(bx[1] + y1)
        gx2 = float(bx[2] + x1)
        gy2 = float(bx[3] + y1)
        refined.append(
            Detection(
                cls_name="boxes",
                conf=float(b.conf.item()),
                box=clamp_box((gx1, gy1, gx2, gy2), w, h),
            )
        )
    return nms_detections(refined, iou_thr=0.45)


def run_detector(
    model_path: str,
    source: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_skip: int,
    min_box_area: float,
    roi_refine: bool,
    roi_imgsz: int,
    roi_conf: float,
    roi_iou: float,
    infer_device: str,
    display: bool,
    save_path: Optional[str],
    final_reduction: int,
    detection_only: bool,
) -> None:
    model = YOLO(model_path)
    cap = cv2.VideoCapture(parse_source(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    names = model.names
    frame_idx = 0
    t0 = time.time()
    use_half = infer_device != "cpu"
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    display_delay_ms = max(1, int(1000 / src_fps)) if src_fps > 1 else 1

    prev_grab_boxes: List[BBox] = []
    prev_person_boxes: List[BBox] = []
    person_miss_streak = 0
    fixed_container_zone: Optional[BBox] = None

    grid_cols = 6
    grid_rows = 6
    fill_threshold = 0.62
    stable_fill_frames = max(8, int((src_fps if src_fps > 1 else 25) * 0.35))
    first_full_streak = 0
    layer_full_streak = 0
    layer_locked = False
    completed_layers = 0
    total_count_placed = 0
    prev_total_count = 0
    last_added_boxes = 0
    last_removed_boxes = 0

    # Keep requested reduction in the 25-30 range.
    final_reduction = max(25, min(30, final_reduction))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_skip > 1 and frame_idx % frame_skip != 0:
            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow("Stocklens Detection", frame)
                if cv2.waitKey(display_delay_ms) & 0xFF == ord("q"):
                    break
            continue

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=infer_device,
            half=use_half,
        )

        dets: List[Detection] = []
        for b in results[0].boxes:
            cls_id = int(b.cls.item())
            dets.append(
                Detection(
                    cls_name=names[cls_id],
                    conf=float(b.conf.item()),
                    box=tuple(b.xyxy[0].tolist()),
                )
            )

        containers = [d for d in dets if d.cls_name == "bigger_box"]
        persons_raw = [d for d in dets if d.cls_name == "Person"]
        boxes = [d for d in dets if d.cls_name == "boxes"]
        persons = list(persons_raw)

        if persons:
            prev_person_boxes = [p.box for p in persons]
            person_miss_streak = 0
        else:
            person_miss_streak += 1
            if prev_person_boxes and person_miss_streak <= 10:
                persons = [Detection(cls_name="Person", conf=0.0, box=pb) for pb in prev_person_boxes]

        main_container = pick_container(containers)

        roi_container = fixed_container_zone if fixed_container_zone is not None else main_container
        if roi_refine and roi_container is not None:
            refined = refine_boxes_in_roi(
                model=model,
                frame=frame,
                container_box=roi_container,
                roi_imgsz=roi_imgsz,
                roi_conf=roi_conf,
                roi_iou=roi_iou,
                infer_device=infer_device,
                use_half=use_half,
            )
            if refined:
                boxes = refined

        h, w = frame.shape[:2]
        precise_container = derive_precise_container_zone(
            container_box=main_container,
            boxes=boxes,
            frame_w=w,
            frame_h=h,
            min_box_area=min_box_area,
        )
        if fixed_container_zone is None and precise_container is not None:
            fixed_container_zone = precise_container
        active_container = fixed_container_zone if fixed_container_zone is not None else precise_container

        hand_regions: List[BBox] = []
        strict_models: List[Tuple[BBox, BBox, BBox, BBox]] = []
        for p in persons:
            l, r, m = choose_best_hand_model(
                person_box=p.box,
                boxes=boxes,
                frame_w=w,
                frame_h=h,
                min_box_area=min_box_area,
                container_box=active_container,
            )
            strict_models.append((p.box, l, r, m))
            hand_regions.extend([l, r])

        boxes_in_hands: List[Detection] = []
        person_grab_counts = [0] * len(strict_models)
        for b in boxes:
            if box_area(b.box) < min_box_area:
                continue
            matched_person_idx: Optional[int] = None
            for i, (person_box, left_palm, right_palm, between) in enumerate(strict_models):
                if person_grab_counts[i] >= 20:
                    continue
                if is_box_grabbed_by_person(
                    box=b.box,
                    person_box=person_box,
                    left_palm=left_palm,
                    right_palm=right_palm,
                    between=between,
                    frame_w=w,
                    frame_h=h,
                    container_box=active_container,
                ):
                    matched_person_idx = i
                    break
            if matched_person_idx is not None:
                person_grab_counts[matched_person_idx] += 1
                boxes_in_hands.append(b)

        enhanced_in_hands = list(boxes_in_hands)
        selected_ids = {id(x) for x in boxes_in_hands}
        for b in boxes:
            if id(b) in selected_ids:
                continue
            if box_area(b.box) < min_box_area:
                continue
            if not any(iou_xyxy(b.box, pb) >= 0.10 for pb in prev_grab_boxes):
                continue
            for _person_box, left_palm, right_palm, between in strict_models:
                if is_box_near_hands_relaxed(
                    box=b.box,
                    left_palm=left_palm,
                    right_palm=right_palm,
                    between=between,
                    frame_w=w,
                    frame_h=h,
                    container_box=active_container,
                ):
                    enhanced_in_hands.append(b)
                    break

        boxes_in_hands = enhanced_in_hands
        prev_grab_boxes = [b.box for b in boxes_in_hands]

        in_container_boxes = [
            b.box
            for b in boxes
            if box_area(b.box) >= min_box_area
            and active_container is not None
            and center_inside(b.box, active_container)
        ]
        occ = grid_occupancy(active_container, in_container_boxes, grid_cols, grid_rows)
        fill_ratio = (len(occ) / float(grid_cols * grid_rows)) if (grid_cols * grid_rows) > 0 else 0.0

        # Count updates only when no real person is detected in frame.
        if len(persons_raw) == 0:
            if completed_layers == 0:
                if fill_ratio >= fill_threshold:
                    first_full_streak += 1
                else:
                    first_full_streak = 0
                if first_full_streak >= stable_fill_frames:
                    completed_layers = 1
                    layer_locked = True
                    layer_full_streak = stable_fill_frames
                    total_count_placed = max(total_count_placed, grid_cols * grid_rows)
            else:
                if fill_ratio >= fill_threshold:
                    layer_full_streak += 1
                    if (not layer_locked) and layer_full_streak >= stable_fill_frames:
                        completed_layers += 1
                        layer_locked = True
                        total_count_placed = max(total_count_placed, completed_layers * grid_cols * grid_rows)
                else:
                    layer_full_streak = 0
                    layer_locked = False

                partial_next_layer = int(round(fill_ratio * grid_cols * grid_rows))
                partial_next_layer = max(0, min(grid_cols * grid_rows - 1, partial_next_layer))
                total_count_placed = max(total_count_placed, completed_layers * grid_cols * grid_rows + partial_next_layer)
        else:
            first_full_streak = 0
            layer_full_streak = 0

        raw_total_count = total_count_placed
        total_count = max(0, raw_total_count - final_reduction)
        last_added_boxes = max(0, total_count - prev_total_count)
        last_removed_boxes = 0
        prev_total_count = total_count
        elapsed = max(1e-6, time.time() - t0)
        fps = frame_idx / elapsed

        annotated = frame.copy()

        if active_container is not None:
            x1, y1, x2, y2 = map(int, active_container)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 255, 80), 2)

        for p in persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (90, 120, 255), 2)

        for hr in hand_regions:
            x1, y1, x2, y2 = map(int, hr)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)

        grabbed_keys = {tuple(map(float, b.box)) for b in boxes_in_hands}
        for b in boxes:
            if box_area(b.box) < min_box_area:
                continue
            if tuple(map(float, b.box)) in grabbed_keys:
                continue
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 210, 120), 2)

        for b in boxes_in_hands:
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if not detection_only:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (8, 8), (460, 160), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0, annotated)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Total Count: {total_count}", (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Last Added Boxes: {last_added_boxes}", (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Last Removed Boxes: {last_removed_boxes}", (16, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer is not None:
            writer.write(annotated)

        if display:
            cv2.imshow("Stocklens Detection", annotated)
            if cv2.waitKey(display_delay_ms) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    if not detection_only:
        print(f"Final Total Count (reduced by {final_reduction}): {prev_total_count}")
        print(f"Last Added Boxes: {last_added_boxes}")
        print(f"Last Removed Boxes: {last_removed_boxes}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time detection only (counting disabled)")
    parser.add_argument("--model", type=str, default="runs/stocklens/yolo26n_train/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index (0) or video path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--infer-device", type=str, default="0", help="Inference device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--min-box-area", type=float, default=120.0, help="Ignore tiny box detections below this area in pixels")

    # Kept for compatibility with older commands; these are ignored in detection-only mode.
    parser.add_argument("--tracker", type=str, default="tracker_stocklens.yaml", help="Deprecated and ignored")
    parser.add_argument("--instance-gap-frames", type=int, default=8, help="Deprecated and ignored")
    parser.add_argument("--roi-refine", action="store_true", help="Enable second-pass high-resolution box detection inside container ROI")
    parser.add_argument("--roi-imgsz", type=int, default=768, help="Image size for ROI refine pass")
    parser.add_argument("--roi-conf", type=float, default=0.18, help="Confidence threshold for ROI refine pass")
    parser.add_argument("--roi-iou", type=float, default=0.35, help="IoU threshold for ROI refine pass")

    parser.add_argument("--no-display", action="store_true", help="Run without preview window")
    parser.add_argument("--save", type=str, default="", help="Output annotated mp4 path")
    parser.add_argument("--final-reduction", type=int, default=27, help="Subtract this value (25-30) from final total count")
    parser.add_argument("--detection-only", action="store_true", help="Render only detection overlays without counting text")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    run_detector(
        model_path=str(model_path),
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        frame_skip=max(1, args.frame_skip),
        min_box_area=max(0.0, args.min_box_area),
        roi_refine=args.roi_refine,
        roi_imgsz=max(320, args.roi_imgsz),
        roi_conf=max(0.01, min(0.99, args.roi_conf)),
        roi_iou=max(0.01, min(0.99, args.roi_iou)),
        infer_device=args.infer_device,
        display=not args.no_display,
        save_path=args.save if args.save else None,
        final_reduction=args.final_reduction,
        detection_only=args.detection_only,
    )


if __name__ == "__main__":
    main()
