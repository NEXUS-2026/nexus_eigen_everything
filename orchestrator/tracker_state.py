"""
ByteTrack integration + ROI state machine for the warehouse box counter.
Responsibilities
Wrap supervision.ByteTrack so Thread 3 (the fast worker) has a single
.update() call that handles both the "fresh YOLO detections available"
and "Kalman predict only" cases transparently.
Maintain a per track ID state machine with four states:
    PENDING_ENTER   — center point has been inside the ROI for at least
                      one frame but fewer than DEBOUNCE_FRAMES frames.
                      Not yet counted.
    CONFIRMED_INSIDE — stable inside the ROI for >= DEBOUNCE_FRAMES frames.
                      Contributes +1 to the count. Written to DB.
    REMOVED         — was CONFIRMED_INSIDE, then moved outside the ROI.
                      Contributes -1. Written to DB.
    HIDDEN_INSIDE   — was CONFIRMED_INSIDE, then disappeared (no detections)
                      while its last known center was inside the ROI.
                      Count is HELD — assumed stacked/occluded.
                      Transitions back to CONFIRMED_INSIDE if the track
                      reappears, or is purged after GHOST_FRAMES frames.

Threading contract
BoxTrackerStateMachine is owned exclusively by Thread 3.
It is never touched by Thread 1 (FastAPI) or Thread 2 (YOLO).
The only data that crosses thread boundaries is:
  - DetectionResult  (Thread 2 → Thread 3, via a threading.Lock in main.py)
  - FrameResult      (Thread 3 → Thread 1, via an asyncio.Queue in main.py)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import supervision as sv

from yolo_engine import DetectionResult

logger = logging.getLogger(__name__)


# State enumeration

class TrackState(Enum):
    PENDING_ENTER     = auto()   # inside ROI, debounce not yet satisfied
    CONFIRMED_INSIDE  = auto()   # counted; stable inside
    REMOVED           = auto()   # counted out; was CONFIRMED, moved outside
    HIDDEN_INSIDE     = auto()   # counted; disappeared deep inside ROI

# Per-track record stored in the state machine's dictionary

@dataclass
class TrackRecord:
    state:           TrackState
    # How many consecutive frames this track has been inside the ROI.
    # Resets to 0 if the track moves outside while still PENDING_ENTER.
    debounce_frames: int = 0
    # How many consecutive frames this track has been missing entirely
    # (used only in HIDDEN_INSIDE state to implement the ghost timeout).
    ghost_frames:    int = 0
    # Last known center point (x, y) in original frame pixel coordinates.
    # Updated every frame the track is visible. Used to decide whether a
    # disappearing track was "deep inside" the ROI.
    last_center:     Optional[tuple[float, float]] = None
    # Wall clock timestamp when this track first became CONFIRMED_INSIDE.
    confirmed_at:    Optional[float] = None


# Output contract — what Thread 3 hands to Thread 1 each frame

@dataclass
class FrameResult:
    """
    Everything Thread 1 needs to annotate a frame and update the dashboard.
    Produced once per video frame by BoxTrackerStateMachine.update().
    """
    # Tracked bounding boxes in original frame pixel coords [x1,y1,x2,y2]
    tracked_boxes:   np.ndarray          # (M, 4) float32
    # Supervision track IDs parallel to tracked_boxes
    track_ids:       np.ndarray          # (M,)   int
    # Current confirmed count (CONFIRMED_INSIDE + HIDDEN_INSIDE)
    box_count:       int
    # State of each visible track (for colour coding in the overlay)
    track_states:    dict[int, TrackState]
    # Events that happened this frame (for DB logging in main.py)
    events:          list[tuple[str, int]]   # [("ADDED"|"REMOVED"|"HIDDEN", track_id)]
    # Pass through so the HUD can show YOLO latency
    inference_ms:    float = 0.0


# Main class

class BoxTrackerStateMachine:
    """
    Combines supervision.ByteTrack (Kalman + Hungarian) with a deterministic
    ROI state machine to produce stable, debounced box counts.
    """
    def __init__(
        self,
        roi_polygon:    np.ndarray,
        frame_wh:       tuple[int, int],
        debounce_frames: int = 5,
        ghost_frames:   int  = 90,
        track_buffer:   int  = 120,
        fps:            int  = 40,
    ) -> None:

        self.debounce_frames = debounce_frames
        self.ghost_frames    = ghost_frames
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,  # min score to start a new track
            lost_track_buffer=track_buffer,   # frames before ByteTrack kills a track
            minimum_matching_threshold=0.80,  # IOU threshold for track association
            frame_rate=fps,
        )
        self._zone = sv.PolygonZone(
            polygon=roi_polygon,
        )
        # Store the polygon for drawing in main.py
        self.roi_polygon = roi_polygon
        self._records: dict[int, TrackRecord] = {}

        # Running count: CONFIRMED_INSIDE tracks + HIDDEN_INSIDE tracks
        self._count: int = 0

        logger.info(
            "BoxTrackerStateMachine ready | debounce=%d frames | ghost=%d frames",
            debounce_frames, ghost_frames,
        )

    # Public API, called once per frame by Thread 3                         

    @property
    def count(self) -> int:
        return self._count

    def update(
        self,
        frame:            np.ndarray,
        detection_result: DetectionResult,
    ) -> FrameResult:
        # Process one video frame through ByteTrack + state machine.
        events: list[tuple[str, int]] = []

        sv_detections = self._to_sv_detections(detection_result)
        tracked: sv.Detections = self._tracker.update_with_detections(sv_detections)
        if len(tracked) > 0:
            inside_mask: np.ndarray = self._zone.trigger(tracked)
        else:
            inside_mask = np.array([], dtype=bool)
        # We iterate this below to drive state transitions.
        active_ids: set[int] = set()

        id_to_info: dict[int, tuple[np.ndarray, bool, tuple[float, float]]] = {}
        for i, track_id in enumerate(tracked.tracker_id):
            tid     = int(track_id)
            box     = tracked.xyxy[i]                      # [x1,y1,x2,y2]
            inside  = bool(inside_mask[i])
            cx      = float((box[0] + box[2]) / 2)
            cy      = float((box[1] + box[3]) / 2)
            active_ids.add(tid)
            id_to_info[tid] = (box, inside, (cx, cy))
        for tid, (box, inside, center) in id_to_info.items():
            rec = self._records.get(tid)

            if rec is None:
                # Brand new track ID, initialise record
                rec = TrackRecord(
                    state=TrackState.PENDING_ENTER if inside else TrackState.REMOVED,
                    debounce_frames=1 if inside else 0,
                    last_center=center,
                )
                # Note: REMOVED here is a slight misnomer for a brand new track
                # that appears outside the ROI. We use REMOVED as the "not inside,
                # not pending" catch all to keep the state space small.
                self._records[tid] = rec

            # Update last known center every frame the track is visible
            rec.last_center  = center
            rec.ghost_frames = 0   # reset ghost timer, track is alive

            # Transition table 

            if rec.state == TrackState.PENDING_ENTER:
                if inside:
                    rec.debounce_frames += 1
                    if rec.debounce_frames >= self.debounce_frames:
                        # Debounce satisfied to confirm the addition
                        rec.state        = TrackState.CONFIRMED_INSIDE
                        rec.confirmed_at = time.time()
                        self._count     += 1
                        events.append(("ADDED", tid))
                        logger.debug("Track %d CONFIRMED_INSIDE (count=%d)", tid, self._count)
                else:
                    # Left before debounce completed reset
                    rec.state           = TrackState.REMOVED
                    rec.debounce_frames = 0

            elif rec.state == TrackState.CONFIRMED_INSIDE:
                if not inside:
                    # Moved out of ROI decrement count
                    rec.state    = TrackState.REMOVED
                    self._count  = max(0, self._count - 1)
                    events.append(("REMOVED", tid))
                    logger.debug("Track %d REMOVED (count=%d)", tid, self._count)
                # If still inside: nothing to do stay CONFIRMED_INSIDE

            elif rec.state == TrackState.HIDDEN_INSIDE:
                # Track reappeared decide where it is now
                rec.ghost_frames = 0
                if inside:
                    # Reappeared inside relink silently, count already held
                    rec.state = TrackState.CONFIRMED_INSIDE
                    logger.debug("Track %d relinked as CONFIRMED_INSIDE", tid)
                else:
                    # Reappeared outside it has genuinely left
                    rec.state   = TrackState.REMOVED
                    self._count = max(0, self._count - 1)
                    events.append(("REMOVED", tid))
                    logger.debug(
                        "Track %d reappeared OUTSIDE after HIDDEN — REMOVED (count=%d)",
                        tid, self._count,
                    )

            elif rec.state == TrackState.REMOVED:
                if inside:
                    # A previously removed track reentered the ROI
                    rec.state           = TrackState.PENDING_ENTER
                    rec.debounce_frames = 1

        for tid, rec in list(self._records.items()):
            if tid in active_ids:
                continue  # already handled above

            if rec.state == TrackState.CONFIRMED_INSIDE:
                # Just disappeared while confirmed inside to enter occlusion guard
                rec.state       = TrackState.HIDDEN_INSIDE
                rec.ghost_frames = 1
                events.append(("HIDDEN", tid))
                logger.debug(
                    "Track %d disappeared inside ROI → HIDDEN_INSIDE at center=%s",
                    tid, rec.last_center,
                )

            elif rec.state == TrackState.HIDDEN_INSIDE:
                rec.ghost_frames += 1
                if rec.ghost_frames > self.ghost_frames:
                    # Occlusion guard expired  box truly gone
                    self._count = max(0, self._count - 1)
                    events.append(("REMOVED", tid))
                    del self._records[tid]
                    logger.debug(
                        "Track %d ghost timer expired → purged (count=%d)",
                        tid, self._count,
                    )

            elif rec.state in (TrackState.PENDING_ENTER, TrackState.REMOVED):
                # Never contributed to the count, safe to purge immediately
                del self._records[tid]

        track_states: dict[int, TrackState] = {
            tid: rec.state for tid, rec in self._records.items()
        }

        return FrameResult(
            tracked_boxes=tracked.xyxy if len(tracked) > 0
                          else np.empty((0, 4), dtype=np.float32),
            track_ids=tracked.tracker_id if len(tracked) > 0
                      else np.empty((0,), dtype=int),
            box_count=self._count,
            track_states=track_states,
            events=events,
            inference_ms=detection_result.inference_ms,
        )

    def reset(self) -> None:
        """
        Hard reset : clears all track records and sets count to 0.
        Call this when starting a new packing session without restarting
        the process (e.g. operator presses "New Carton" in the UI).
        """
        self._records.clear()
        self._count = 0
        self._tracker.reset()
        logger.info("BoxTrackerStateMachine reset.")

    # Private helpers

    def _to_sv_detections(self, result: DetectionResult) -> sv.Detections:
        """
        Convert our DetectionResult into a supervision.Detections object.

        supervision.ByteTrack.update_with_detections() requires:
            .xyxy        : (N, 4) float32  [x1, y1, x2, y2]
            .confidence  : (N,)   float32
            .class_id    : (N,)   int

        When result.count == 0 we return an empty Detections object so
        ByteTrack knows to run Kalman predict only for this frame.
        """
        if result.count == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=result.boxes,                              # (N, 4) float32
            confidence=result.scores,                       # (N,)   float32
            class_id=result.class_ids.astype(int),          # (N,)   int
        )