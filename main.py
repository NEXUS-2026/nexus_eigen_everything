"""
FastAPI application, orchestrates all three threads and serves the
WebSocket video stream + REST API to the frontend.

Thread map
Thread 1  (asyncio event loop the main thread)
          FastAPI request handling + WebSocket broadcaster.
          Reads FrameResult objects from `output_queue` and pushes
          JPEG encoded frames to every connected browser client.

Thread 2  (inference_worker daemon thread)
          Owns the ONNXDetector. Loops forever:
            - Grabs the latest raw frame from `inference_queue`.
            - Runs YOLO inference (~100-200 ms on Pi).
            - Writes the DetectionResult into `shared_state` under a Lock.
          Never blocks Thread 3.

Thread 3  (video_worker daemon thread)
          Owns BoxTrackerStateMachine and CountDatabase.
          Runs at exactly TARGET_FPS:
            - Reads a frame from VideoCapture.
            - Checks shared_state for fresh detections (non blocking).
            - Calls state_machine.update() to FrameResult.
            - Logs DB events.
            - Puts the annotated FrameResult into `output_queue`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from database import CountDatabase
from tracker_state import BoxTrackerStateMachine, FrameResult, TrackState
from yolo_engine import DetectionResult, ONNXDetector
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Colour map for bounding-box overlay (BGR)
STATE_COLOURS = {
    TrackState.PENDING_ENTER:    (0,   200, 255),   # amber
    TrackState.CONFIRMED_INSIDE: (0,   230,  50),   # green
    TrackState.REMOVED:          (60,   60,  60),   # dark grey
    TrackState.HIDDEN_INSIDE:    (255, 120,   0),   # blue-ish
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Shared state container — the only object that crosses thread boundaries

@dataclass
class SharedState:
    """
    Holds the single shared variable between Thread 2 and Thread 3.
    All access to `latest_detection` must happen under `lock`.
    """
    lock:             threading.Lock   = threading.Lock()  # type: ignore[assignment]
    latest_detection: DetectionResult  = None              # type: ignore[assignment]
    fresh:            bool             = False             # True = Thread 3 hasn't consumed yet


# Connection manager — tracks all live WebSocket clients

class ConnectionManager:
    """
    Thread safe registry of active WebSocket connections.
    broadcast() is called from the asyncio event loop (Thread 1).
    """

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.remove(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(self._clients))

    async def broadcast(self, payload: str) -> None:
        """
        Send a JSON string to every connected client.
        Disconnected clients are removed silently.
        """
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.remove(ws)


# Global pipeline objects (initialised in lifespan, used across threads)

shared_state   = SharedState()
manager        = ConnectionManager()

# inference_queue: Thread 3 puts raw frames here for Thread 2 to consume.
# maxsize=1 means Thread 2 always processes the LATEST frame, never a stale one.
inference_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)

# output_queue: Thread 3 puts FrameResults here for Thread 1 to broadcast.
# maxsize=2 prevents memory build-up if the WebSocket client is slow.
output_queue: queue.Queue[Optional[FrameResult]] = queue.Queue(maxsize=2)

# Populated during lifespan startup
db:            Optional[CountDatabase]          = None
state_machine: Optional[BoxTrackerStateMachine] = None
stop_event     = threading.Event()


# Thread 2 — inference worker

def inference_worker() -> None:
    """
    Owns the ONNXDetector. Runs as a background daemon thread.

    Blocking pattern:
        inference_queue.get(timeout=1)
    This parks the thread (zero CPU) while no frame is waiting.
    Thread 3 feeds frames non blocking (put_nowait), so this thread
    never starves Thread 3.
    """
    logger.info("Inference worker starting …")
    detector = ONNXDetector(
        model_path=config.MODEL_PATH,
        input_size=config.ONNX_INPUT_SIZE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        nms_iou_threshold=config.NMS_IOU_THRESHOLD,
        target_classes=config.TARGET_CLASSES,
        num_intra_threads=config.ORT_INTRA_THREADS,
    )

    # Warm up pass first ORT call is always slow (JIT, cache miss).
    # Run it with a dummy frame so the first real frame isn't penalised.
    logger.info("ONNX warm-up …")
    dummy = np.zeros((config.ONNX_INPUT_SIZE[1], config.ONNX_INPUT_SIZE[0], 3), dtype=np.uint8)
    detector.detect(dummy)
    logger.info("Inference worker ready.")

    while not stop_event.is_set():
        try:
            frame = inference_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Poison pill from lifespan shutdown
        if frame is None:
            break

        result = detector.detect(frame)

        # Write result under lock Thread 3 will consume on the next tick
        with shared_state.lock:
            shared_state.latest_detection = result
            shared_state.fresh            = True

    logger.info("Inference worker exiting.")


# Thread 3 — video & tracking worker

def video_worker() -> None:
    """
    The fast worker. Runs at exactly TARGET_FPS using a perf_counter busy wait
    for the final sub millisecond correction after a coarse sleep.

    Loop body (must complete in <= 25 ms at 40 FPS):
      1. cap.read()                              ~1-3 ms
      2. Try-put frame into inference_queue      ~0 ms (non-blocking)
      3. Non-blocking check of shared_state      ~0 ms
      4. state_machine.update()                  ~2-5 ms
      5. Annotate frame                          ~2-4 ms
      6. JPEG encode                             ~3-6 ms
      7. output_queue.put_nowait()               ~0 ms
      8. DB flush                                ~0.5 ms
    """
    global db, state_machine

    logger.info("Video worker starting …")

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", config.VIDEO_SOURCE)
        return
    
    # Kill the buffer so we only ever get the absolute newest frame (Zero Lag)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Force the webcam hardware to output a smaller resolution (Saves USB bandwidth & CPU)
    # 640x480 is standard VGA and plenty of pixels for YOLO to detect boxes
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read actual frame dimensions from the capture device
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Capture opened: %dx%d", frame_w, frame_h)

    # Initialise state machine with actual frame dimensions
    state_machine = BoxTrackerStateMachine(
        roi_polygon=config.ROI_POLYGON,
        frame_wh=(frame_w, frame_h),
        debounce_frames=config.DEBOUNCE_FRAMES,
        ghost_frames=config.GHOST_FRAMES,
        track_buffer=config.TRACK_BUFFER,
        fps=config.TARGET_FPS,
    )

    # Open DB session
    db = CountDatabase(config.DB_PATH)
    db.start_session()

    frame_duration = 1.0 / config.TARGET_FPS
    detect_skip    = 4      # send a frame to Thread 2 every N ticks (~10 FPS)
    tick           = 0
    fps_actual     = 0.0
    fps_t0         = time.perf_counter()
    fps_frames     = 0

    # The last DetectionResult we consumed from shared_state.
    # We reuse it on Kalman only frames so the state machine always
    # receives a valid (possibly stale) detection snapshot.
    last_detection = DetectionResult.empty()

    logger.info("Video worker running at %d FPS target.", config.TARGET_FPS)

    while not stop_event.is_set():
        t_start = time.perf_counter()
        tick   += 1

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            # End of file loop back to start for recorded video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Feed frame to Thread 2 (non-blocking)
        # put_nowait drops the frame if Thread 2 is still busy intentional.
        # We always want Thread 2 to process the freshest possible frame.
        if tick % detect_skip == 0:
            try:
                # Discard any unprocessed frame still sitting in the queue
                # before putting the new one so Thread 2 gets the latest.
                try:
                    inference_queue.get_nowait()
                except queue.Empty:
                    pass
                inference_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Thread 2 just picked one up race condition, harmless

        # Consume fresh detection from Thread 2 (non-blocking)
        with shared_state.lock:
            if shared_state.fresh:
                last_detection        = shared_state.latest_detection
                shared_state.fresh    = False

        # Update state machine
        # Pass last_detection every frame. If YOLO hasn't produced anything
        # new, this is stale ByteTrack will do Kalman only prediction.
        result: FrameResult = state_machine.update(frame, last_detection)

        # Log DB events
        for event_type, track_id in result.events:
            db.log_event(event_type, track_id, result.box_count)
        if result.events:
            db.flush()

        # Annotate frame
        annotated = _annotate_frame(frame, result, fps_actual)

        # JPEG-encode and push to output queue
        ok, buf = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
        )
        if ok:
            payload = json.dumps({
                "frame":  base64.b64encode(buf.tobytes()).decode("ascii"),
                "count":  result.box_count,
                "fps":    round(fps_actual, 1),
                "inf_ms": round(result.inference_ms, 1),
            })
            try:
                output_queue.put_nowait(payload)
            except queue.Full:
                pass  # WebSocket client is slow drop this frame, not a problem

        # Measure actual FPS every 60 frames
        fps_frames += 1
        if fps_frames >= 60:
            fps_actual  = fps_frames / (time.perf_counter() - fps_t0)
            fps_t0      = time.perf_counter()
            fps_frames  = 0

        # Frame-rate limiter
        # Coarse sleep first (saves CPU/heat), then busy wait for precision.
        elapsed   = time.perf_counter() - t_start
        remaining = frame_duration - elapsed
        if remaining > 0.002:
            time.sleep(remaining - 0.002)           # sleep most of the gap
        while (time.perf_counter() - t_start) < frame_duration:
            pass                                    # busy wait the last 2 ms

    cap.release()
    db.end_session(state_machine.count)
    logger.info("Video worker exiting. Final count: %d", state_machine.count)


# Frame annotation helper (called by Thread 3)

def _annotate_frame(
    frame:  np.ndarray,
    result: FrameResult,
    fps:    float,
) -> np.ndarray:
    """
    Draw the ROI polygon, bounding boxes, track IDs, state colours,
    and a HUD overlay onto a copy of the frame.
    """
    out = frame.copy()

    # ROI polygon
    cv2.polylines(
        out,
        [config.ROI_POLYGON.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 128),
        thickness=2,
    )

    # Tracked bounding boxes
    for i, box in enumerate(result.tracked_boxes):
        tid   = int(result.track_ids[i])
        state = result.track_states.get(tid, TrackState.REMOVED)
        color = STATE_COLOURS.get(state, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"#{tid} {state.name[:3]}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), FONT, 0.45, (0, 0, 0), 1)

    # HUD overlay (semi transparent dark rectangle)
    h, w  = out.shape[:2]
    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (310, 90), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.6, out, 0.4, 0, out)

    cv2.putText(out, f"Boxes in carton: {result.box_count}",
                (16, 36), FONT, 0.85, (255, 255, 255), 2)
    cv2.putText(out, f"FPS: {fps:.1f}   YOLO: {result.inference_ms:.0f}ms",
                (16, 72), FONT, 0.55, (180, 180, 180), 1)

    return out


# FastAPI lifespan — starts/stops background threads cleanly

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces the deprecated @app.on_event pattern.
    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.
    """
    logger.info("Starting pipeline threads …")

    # Thread 2 : inference worker
    t_inference = threading.Thread(
        target=inference_worker,
        name="InferenceWorker",
        daemon=True,
    )
    t_inference.start()

    # Thread 3 : video + tracking worker
    t_video = threading.Thread(
        target=video_worker,
        name="VideoWorker",
        daemon=True,
    )
    t_video.start()

    # Thread 1 : asyncio broadcaster (started as a background task)
    asyncio.create_task(_broadcast_loop())

    logger.info("All threads started. Server ready.")
    yield

    # Shutdown
    logger.info("Shutting down …")
    stop_event.set()

    # Send poison pills to unblock any blocking .get() calls
    try:
        inference_queue.put_nowait(None)
    except queue.Full:
        pass
    try:
        output_queue.put_nowait(None)
    except queue.Full:
        pass

    t_inference.join(timeout=5.0)
    t_video.join(timeout=5.0)
    logger.info("Shutdown complete.")


# Background broadcast coroutine (runs inside Thread 1's event loop)

async def _broadcast_loop() -> None:
    """
    Drains output_queue and broadcasts each payload to all WebSocket clients.
    Runs as an asyncio Task inside Thread 1 never blocks the event loop
    because it yields control via asyncio.sleep(0) on every iteration.
    """
    loop = asyncio.get_event_loop()

    while not stop_event.is_set():
        # Poll output_queue in a non blocking way inside the async loop.
        # run_in_executor offloads the blocking .get() to a thread pool thread
        # so the asyncio event loop stays free to handle WebSocket handshakes.
        try:
            payload = await loop.run_in_executor(
                None,
                lambda: output_queue.get(timeout=0.05),
            )
        except queue.Empty:
            await asyncio.sleep(0)
            continue

        # Poison pill from shutdown
        if payload is None:
            break

        if manager._clients:
            await manager.broadcast(payload)

        await asyncio.sleep(0)  # yield control never starve other coroutines


# FastAPI app

app = FastAPI(
    title="Warehouse Box Counter",
    description="Real-time box counting via YOLO + ByteTrack on edge hardware.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Each browser tab connects here.
    The client sends "reset" to start a new packing session mid stream.
    """
    await manager.connect(ws)
    try:
        while True:
            # Listen for control messages from the client (non blocking).
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                if msg == "reset" and state_machine is not None:
                    state_machine.reset()
                    if db is not None:
                        db.end_session(0)
                        db.start_session()
                    logger.info("Session reset via WebSocket command.")
            except asyncio.TimeoutError:
                pass  # No message from client normal
    except WebSocketDisconnect:
        manager.disconnect(ws)


# REST endpoints

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the frontend HTML."""
    html_path = "static/index.html"
    with open(html_path) as f:
        return HTMLResponse(content=f.read())


@app.get("/count")
async def get_count() -> JSONResponse:
    """
    Lightweight polling endpoint useful for systems that can't use WebSocket.
    """
    count = state_machine.count if state_machine is not None else 0
    return JSONResponse({"count": count})


@app.get("/history")
async def get_history() -> JSONResponse:
    """Return the last 50 packing sessions from the DB."""
    if db is None:
        return JSONResponse({"sessions": []})
    return JSONResponse({"sessions": db.get_session_summary()})


@app.get("/history/{session_id}")
async def get_session_events(session_id: int) -> JSONResponse:
    """Return every count event for a specific session."""
    if db is None:
        return JSONResponse({"events": []})
    return JSONResponse({"events": db.get_events_for_session(session_id)})


@app.post("/reset")
async def reset_session() -> JSONResponse:
    """
    HTTP alternative to the WebSocket "reset" command.
    Useful for automated test harnesses.
    """
    if state_machine is not None:
        state_machine.reset()
    if db is not None:
        db.end_session(0)
        db.start_session()
    return JSONResponse({"status": "ok", "message": "Session reset."})


# Entry point

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        # workers=1 is mandatory multiple workers would each spawn their
        # own YOLO thread and fight for the Pi's CPU cores.
        workers=1,
    )