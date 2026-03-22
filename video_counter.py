from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    return inter / max(1e-6, ua)


def smooth_box(prev_box: Optional[BBox], cur_box: Optional[BBox], alpha: float) -> Optional[BBox]:
    if cur_box is None:
        return prev_box
    if prev_box is None:
        return cur_box
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
    enter_box = expand_or_shrink(container, -0.10)  # stricter to enter
    leave_box = expand_or_shrink(container, 0.06)   # looser to stay in

    ex1, ey1, ex2, ey2 = enter_box
    lx1, ly1, lx2, ly2 = leave_box

    in_enter = ex1 <= cx <= ex2 and ey1 <= cy <= ey2
    in_leave = lx1 <= cx <= lx2 and ly1 <= cy <= ly2

    if prev_state == "inside":
        return "inside" if in_leave else "outside"
    return "inside" if in_enter else "outside"


def pick_primary_container(containers: list[Detection]) -> Optional[BBox]:
    if not containers:
        return None
    return max(containers, key=lambda d: box_area(d.box)).box


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def run_detector(
    model_path: str,
    source: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_skip: int,
    infer_device: str,
    display: bool,
    save_path: Optional[str],
    max_match_distance: float,
    track_max_age: int,
    state_confirm_frames: int,
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

    tracks: dict[int, BoxTrack] = {}
    next_track_id = 1
    total_boxes = 0
    smoothed_container: Optional[BBox] = None

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

        dets: list[Detection] = []
        for b in results[0].boxes:
            cls_id = int(b.cls.item())
            if cls_id not in (0, 1, 2):
                continue
            dets.append(
                Detection(
                    cls_name=names[cls_id],
                    conf=float(b.conf.item()),
                    box=tuple(b.xyxy[0].tolist()),
                )
            )

        containers = [d for d in dets if d.cls_name == "bigger_box"]
        persons = [d for d in dets if d.cls_name == "Person"]
        all_boxes = [d for d in dets if d.cls_name == "boxes"]

        container_box = pick_primary_container(containers)
        smoothed_container = smooth_box(smoothed_container, container_box, alpha=0.25)
        active_container = smoothed_container
        boxes_inside: list[Detection] = []
        boxes_outside: list[Detection] = []
        for b in all_boxes:
            if active_container is not None and center_inside(b.box, active_container):
                boxes_inside.append(b)
            else:
                boxes_outside.append(b)

        # Transition counting by tracked box identity across frames.
        # Count an event only after state confirmation across consecutive frames.
        current_boxes = boxes_inside + boxes_outside
        det_centers: list[Tuple[float, float]] = [box_center(d.box) for d in current_boxes]
        det_states: list[str] = ["inside" if d in boxes_inside else "outside" for d in current_boxes]

        added_boxes = 0
        removed_boxes = 0

        active_track_ids = [tid for tid, tr in tracks.items() if frame_idx - tr.last_seen <= track_max_age]
        iou_pairs: list[Tuple[float, int, int]] = []
        pairs: list[Tuple[float, int, int]] = []
        for det_idx, (cx, cy) in enumerate(det_centers):
            for tid in active_track_ids:
                tr = tracks[tid]
                ov = iou_xyxy(current_boxes[det_idx].box, tr.box)
                if ov >= 0.18:
                    iou_pairs.append((ov, tid, det_idx))
                dist = math.hypot(cx - tr.cx, cy - tr.cy)
                if dist <= max_match_distance:
                    pairs.append((dist, tid, det_idx))
        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        pairs.sort(key=lambda x: x[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for _, tid, det_idx in iou_pairs:
            if tid in matched_tracks or det_idx in matched_dets:
                continue
            tr = tracks[tid]
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

                if tr.pending_count >= state_confirm_frames:
                    if tr.state == "outside" and observed_state == "inside":
                        added_boxes += 1
                        total_boxes += 1
                    elif tr.state == "inside" and observed_state == "outside":
                        removed_boxes += 1

                    tr.state = observed_state
                    tr.pending_state = observed_state
                    tr.pending_count = 0

            cx, cy = det_centers[det_idx]
            tr.cx = cx
            tr.cy = cy
            tr.box = current_boxes[det_idx].box
            tr.last_seen = frame_idx

            matched_tracks.add(tid)
            matched_dets.add(det_idx)

        for _, tid, det_idx in pairs:
            if tid in matched_tracks or det_idx in matched_dets:
                continue
            tr = tracks[tid]
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

                if tr.pending_count >= state_confirm_frames:
                    if tr.state == "outside" and observed_state == "inside":
                        added_boxes += 1
                        total_boxes += 1
                    elif tr.state == "inside" and observed_state == "outside":
                        removed_boxes += 1

                    tr.state = observed_state
                    tr.pending_state = observed_state
                    tr.pending_count = 0

            cx, cy = det_centers[det_idx]
            tr.cx = cx
            tr.cy = cy
            tr.box = current_boxes[det_idx].box
            tr.last_seen = frame_idx

            matched_tracks.add(tid)
            matched_dets.add(det_idx)

        for det_idx, (cx, cy) in enumerate(det_centers):
            if det_idx in matched_dets:
                continue
            tracks[next_track_id] = BoxTrack(
                track_id=next_track_id,
                cx=cx,
                cy=cy,
                box=current_boxes[det_idx].box,
                state=det_states[det_idx],
                pending_state=det_states[det_idx],
                pending_count=0,
                last_seen=frame_idx,
            )
            next_track_id += 1

        stale_ids = [tid for tid, tr in tracks.items() if frame_idx - tr.last_seen > track_max_age]
        for tid in stale_ids:
            del tracks[tid]

        elapsed = max(1e-6, time.time() - t0)
        fps = frame_idx / elapsed

        annotated = frame.copy()

        # 1) Container
        for c in containers:
            x1, y1, x2, y2 = map(int, c.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 255, 80), 2)
            cv2.putText(annotated, f"Container {c.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 80), 2)

        # 2) Boxes outside container (orange)
        for b in boxes_outside:
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(annotated, f"Box OUT {b.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        # 3) Boxes inside container (blue)
        for b in boxes_inside:
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 90, 0), 2)
            cv2.putText(annotated, f"Box IN {b.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 90, 0), 2)

        # 4) Person
        for p in persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 80, 255), 2)
            cv2.putText(annotated, f"Person {p.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 80, 255), 2)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (360, 118), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0, annotated)
        cv2.putText(annotated, f"Total Boxes: {total_boxes}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"Boxes Added: {added_boxes}", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"Boxes Removed: {removed_boxes}", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26n detection: container, boxes outside, boxes inside, person")
    parser.add_argument("--model", type=str, default="runs/stocklens/yolo26n_train/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index (0) or video path")
    parser.add_argument("--conf", type=float, default=0.22, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--infer-device", type=str, default="0", help="Inference device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--no-display", action="store_true", help="Run without preview window")
    parser.add_argument("--save", type=str, default="", help="Output annotated mp4 path")
    parser.add_argument("--max-match-distance", type=float, default=85.0, help="Max centroid distance for box track matching")
    parser.add_argument("--track-max-age", type=int, default=20, help="Frames to keep unmatched tracks before deletion")
    parser.add_argument("--state-confirm-frames", type=int, default=3, help="Consecutive frames required to confirm OUT/IN state change")

    # Compatibility placeholders for old commands.
    parser.add_argument("--min-box-area", type=float, default=80.0, help="Deprecated and ignored")
    parser.add_argument("--tracker", type=str, default="tracker_stocklens.yaml", help="Deprecated and ignored")
    parser.add_argument("--instance-gap-frames", type=int, default=8, help="Deprecated and ignored")
    parser.add_argument("--roi-refine", action="store_true", help="Deprecated and ignored")
    parser.add_argument("--roi-imgsz", type=int, default=768, help="Deprecated and ignored")
    parser.add_argument("--roi-conf", type=float, default=0.18, help="Deprecated and ignored")
    parser.add_argument("--roi-iou", type=float, default=0.35, help="Deprecated and ignored")

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
        infer_device=args.infer_device,
        display=not args.no_display,
        save_path=args.save if args.save else None,
        max_match_distance=max(10.0, args.max_match_distance),
        track_max_age=max(1, args.track_max_age),
        state_confirm_frames=max(1, args.state_confirm_frames),
    )


if __name__ == "__main__":
    main()
