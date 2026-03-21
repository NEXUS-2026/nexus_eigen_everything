"""
Load an ONNX model once at startup and hold it in memory.
Preallocate all numpy buffers so the hot path (detect()) has zero heap allocs.
Expose a single .detect(frame) method whose return value is a strictly-typed
DetectionResult that every downstream consumer can depend on without knowing
anything about ONNX or YOLO internals.

Outputs contract (DetectionResult)
boxes: np.ndarray  shape (N, 4)  dtype float32
       Each row is [x1, y1, x2, y2] in ORIGINAL frame pixel coordinates.
scores: np.ndarray  shape (N,)    dtype float32
        Confidence score for each detection (0.0 - 1.0).
class_ids: np.ndarray  shape (N,)    dtype int32
        COCO / fine tuned class index for each detection.
If there are no detections, all three arrays have shape (0,4), (0,), (0,)
so callers never have to guard against None.
Threading note
ONNXDetector is NOT thread safe by itself. The design in main.py ensures only
Thread 2 (the slow worker) ever calls .detect(). If you later parallelise
inference, create one ONNXDetector instance per thread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Public data contract - downstream code imports only this dataclass

@dataclass
class DetectionResult:
    """ 
    Immutable snapshot of what YOLO found in one frame.
    All coordinate arrays are in originalframe pixel space.
    """

    boxes: np.ndarray            # (N, 4) float32  [x1, y1, x2, y2]
    scores: np.ndarray           # (N,) float32
    class_ids: np.ndarray        # (N,) int32
    inference_ms: float = 0.0    # latency of the last call, for the HUD

    @property
    def count(self) -> int:
        return len(self.scores)
    
    @classmethod
    def empty(cls) -> "DetectionResult":
        # Return a zero-detection result — safe default before first inference.
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
        )
    
    # letterbox helper : preserves aspect ratio, pads to square

def _letterbox(
    frame: np.ndarray,
    target_wh: tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    
    target_w, target_h = target_wh
    src_h, src_w = frame.shape[:2]

    # Uniform scale fit both dimensions without exceeding the target
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, (pad_left, pad_top)
    
# Main engine class

class ONNXDetector:

    def __init__(
            self,
            model_path: str | Path,
            input_size: tuple[int, int] = (640, 640),
            confidence_threshold: float = 0.35,
            nms_iou_threshold: float = 0.45,
            target_classes: Optional[set[int]] = None,
            num_intra_threads: int = 3,
    ) -> None:
        
        self.input_size = input_size
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_iou_threshold
        self.target_cls = target_classes

        # ONNX Runtime session 
        opts = ort.SessionOptions()

        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        opts.intra_op_num_threads = num_intra_threads
        opts.inter_op_num_threads = 1

        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        model_inputs = self._session.get_inputs()
        model_outputs = self._session.get_outputs()

        self._input_name = model_inputs[0].name
        self._output_name = model_outputs[0].name

        declared = model_inputs[0].shape
        if len(declared) == 4:
            _, _, h, w = declared
            if isinstance(h, int) and isinstance(w, int) and (w, h) != self.input_size:
                logger.warning(
                    "Model declared %dx%d but input_size=%s - using model shape.", w, h, self.input_size,
                )
                self.input_size = (w, h)
        
        iw, ih = self.input_size
        self._input_buffer: np.ndarray = np.zeros(
            (1, 3, ih, iw), dtype=np.float32
        )

        logger.info(
            "ONNXDetector ready | model=%s | input=%dx%d | conf=%.2f | nms=%.2f",
            Path(model_path). name, iw, ih, self.conf_thresh, self.nms_thresh,
        )

    # Public API 
    def detect(self, frame: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()

        blob, scale, (pad_left, pad_top) = self._preprocess(frame)
        raw_output = self._run_inference(blob)
        result = self._postprocess(
            raw_output, frame.shape[:2], scale, pad_left, pad_top
        )

        result.inference_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Inference %.1f ms | %d dets", result.inference_ms, result.count)
        return result
    
    # Private pipeline steps
    def _preprocess(
            self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        
        padded, scale, (pad_left, pad_top) = _letterbox(frame, self.input_size)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        chw /= 255.0

        self._input_buffer[0] = chw
        return self._input_buffer, scale, (pad_left, pad_top)
    
    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: blob},
        )
        return outputs[0][0]
    
    def _postprocess(
            self,
            raw: np.ndarray,
            orig_shape: tuple[int, int],
            scale: float,
            pad_left: int,
            pad_top: int,
    ) -> DetectionResult:
        orig_h, orig_w = orig_shape
        
        # Dyanmic shape detection
        if raw.shape[-1] == 6:
            # It's YOLOv10 or NMS-embedded (shape: N, 6)
            # Format: [x1, y1, x2, y2, score, class]
            x1 = raw[:, 0]
            y1 = raw[:, 1]
            x2 = raw[:, 2]
            y2 = raw[:, 3]
            scores_all = raw[:, 4]
            class_ids_all = raw[:, 5].astype(np.int32)
            is_xywh = False # Already xyxy format

        else: 
            # It's standard YOLOv8 (Shape: 84, 8400)
            pred = raw.T
            boxes_raw = pred[:, :4]
            class_scores = pred[:, 4:]

            class_ids_all = np.argmax(class_scores, axis=1).astype(np.int32)
            scores_all = class_scores[np.arange(len(pred)), class_ids_all].astype(np.float32)

            # Convert xywh to xyxy
            cx, cy, bw, bh = boxes_raw[:,0], boxes_raw[:,1], boxes_raw[:,2], boxes_raw[:,3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cy + bw / 2
            y2 = cy + bh / 2
            is_xywh = True
        
        # Apply Threshold
        mask = scores_all >= self.conf_thresh
        if not mask.any():
            return DetectionResult.empty()
        
        x1_f, y1_f, x2_f, y2_f = x1[mask], y1[mask], x2[mask], y2[mask]
        scores_f = scores_all[mask]
        class_f = class_ids_all[mask]

        # Filter by target class (from config.py)
        if self.target_cls is not None:
            cls_mask = np.isin(class_f, list(self.target_cls))
            if not cls_mask.any():
                return DetectionResult.empty()
            x1_f, y1_f, x2_f, y2_f = x1_f[cls_mask], y1_f[cls_mask], x2_f[cls_mask], y2_f[cls_mask]
            scores_f = scores_f[cls_mask]
            class_f = class_f[mask]

        # Invert Letterbox to original pixels
        x1_f = (x1_f - pad_left) / scale
        y1_f = (y1_f - pad_top) / scale
        x2_f = (x2_f - pad_left) / scale
        y2_f = (x2_f - pad_top) / scale

        # Clip to frame bounds 
        x1_f = np.clip(x1_f, 0, orig_w).astype(np.float32)
        y1_f = np.clip(y1_f, 0, orig_h).astype(np.float32)
        x2_f = np.clip(x2_f, 0, orig_w).astype(np.float32)
        y2_f = np.clip(y2_f, 0, orig_h).astype(np.float32)

        # Final NMS / output
        if not is_xywh:
            # YOLOv10 already applied NMS natively, just return the boxes
            final_boxes = np.stack([x1_f, y1_f, x2_f, y2_f], axis=1)
            return DetectionResult(
                boxes=final_boxes,
                scores=scores_f,
                class_ids=class_f
            )
        else: 
            # Standard YOLOv8 requires OpenCV NMS
            boxes_xywh = np.stack([x1_f, y1_f, x2_f - x1_f, y2_f - y1_f], axis=1)
            keep_idx = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(),
                scores_f.tolist(),
                self.conf_thresh,
                self.nms_thresh,
            )
            if len (keep_idx) == 0: 
                return DetectionResult.empty()
            
            keep_idx = np.array(keep_idx).flatten()
            final_boxes = np.stack(
                [x1_f[keep_idx], y1_f[keep_idx], x2_f[keep_idx], y2_f[keep_idx]], axis=1
            ).astype(np.float32)

            return DetectionResult(
                boxes=final_boxes,
                scores=scores_f[keep_idx],
                class_ids=class_f[keep_idx],
            )


# Smoke test python yolo_engine.py 

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
         print("Usuage: python yolo_engine.py <model.onnx> [image_path]")
         sys.exit(1)
    
    detector = ONNXDetector(model_path=sys.argv[1], confidence_threshold=0.35)

    if len(sys.argv) >= 3:
        test_frame = cv2.imread(sys.argv[2])
        if test_frame is None:
            sys.exit(f"Could not read {sys.argv[2]}")
    else: 
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        print("No image supplied — using a grey test frame (expect 0 detections).")
    
    print("Warm-up pass ...")
    detector.detect(test_frame)

    print("Benchmark pass ...")
    result = detector.detect(test_frame)

    print(f"Inference time : {result.inference_ms:.1f} ms")
    print(f"Detections : {result.count}")
    for i, (box, score, cls) in enumerate(
        zip(result.boxes, result.scores, result.class_ids)
    ): 
        print(f" [{i}] class={cls} score={score:.3f} box={box.tolist()}")