# Logistics Vision System: Real Time Box Counter

A high performance, edge optimised computer vision pipeline for real time tracking and counting of warehouse packages. Built to run efficiently on CPU constrained devices (like Raspberry Pi) while streaming 40 FPS video to a modern web dashboard.

## The Solution
Counting boxes manually on a packing line is error prone and slow. This system automates the process using a custom YOLO object detection model and ByteTrack, ensuring high accuracy even when boxes are stacked, occluded, or temporarily blocked by a worker's hand.

## Key Technical Achievements
* **"Illusion of Speed" Architecture:** The backend uses a 3 thread asynchronous pipeline to completely decouple heavy AI inference from the 40 FPS video reading loop, preventing camera lag.
* **Spatial State Machine:** Custom tracking logic that handles edge cases like box debouncing, removals (take backs), and deep carton occlusion.
* **Zero DOM Lag Frontend:** The React dashboard receives streaming JPEG frames via WebSockets and paints them directly to an HTML5 `<canvas>` using hardware acceleration, completely bypassing React's render cycle for zero browser freezing.
* **Concurrent Logging:** SQLite database running in WAL (Write Ahead Logging) mode ensures the AI thread can log count events without blocking the API's read requests.

## Tech Stack 
**Backend (Edge AI & API):**
* Python 3.10+
* FastAPI & WebSockets
* OpenCV (Headless)
* ONNX Runtime (CPU Execution Provider)
* Roboflow Supervision (ByteTrack)
* SQLite3

**Frontend (Dashboard):**
* React 18
* Vite
* Tailwind CSS v4

---

## System Architecture

The system is designed specifically for edge hardware limits:
1. **Thread 1 (Main):** Async FastAPI server handling WebSocket broadcasts and REST endpoints.
2. **Thread 2 (Inference Daemon):** Runs the YOLO ONNX model on the latest available frame, dropping stale frames to prevent queue buildup.
3. **Thread 3 (Video/Tracker Daemon):** Reads the camera at a strict 40 FPS, merges YOLO detections with the ByteTrack Kalman filter, manages the Spatial State Machine, logs to SQLite, and pushes annotated frames to the WebSocket.

---

## Installation & Setup

### 1. Backend Setup
Ensure you have Python installed. It is recommended to use a virtual environment (like Conda or venv).

```bash
# Clone the repository
git clone [https://github.com/yourusername/logistics-vision.git](https://github.com/yourusername/logistics-vision.git)
cd logistics-vision

# Install backend dependencies
python -m pip install opencv-python-headless fastapi uvicorn onnxruntime supervision numpy websockets