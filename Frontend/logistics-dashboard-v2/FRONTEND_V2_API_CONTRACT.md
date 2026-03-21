# Frontend V2 API Contract

This document defines the frontend contract expected by v2 so model/backend integration is plug-and-play.

## 1) WebSocket Stream
Endpoint:
- `GET ws://<host>:8000/ws`

Expected message shape (JSON):
```json
{
  "frame": "<base64-jpeg>",
  "count": 0,
  "fps": 39.8,
  "inf_ms": 12.4,
  "model_ready": true,
  "events": [
    { "type": "ADDED", "track_id": 12, "at": "12:04:11" },
    { "type": "HIDDEN", "track_id": 12, "at": "12:04:19" },
    { "type": "REMOVED", "track_id": 8, "at": "12:04:33" }
  ]
}
```

Notes:
- `frame` is optional; if omitted, canvas keeps previous frame.
- `count`, `fps`, `inf_ms` should be numeric.
- `model_ready` is optional; frontend will fallback to heuristic if absent.
- `events` is optional; frontend accepts either object events or string events.

## 2) Reset Session
Transport:
- Same WebSocket endpoint.
- Client sends plain text: `reset`

## 3) Upload Test Video
Endpoint:
- `POST http://<host>:8000/video/upload`
- multipart/form-data field name: `file`

Success response example:
```json
{
  "status": "ok",
  "filename": "clip.mp4",
  "source": "/path/to/saved/file.mp4"
}
```

## 3.1) Clear Uploaded Source
Endpoint:
- `POST http://<host>:8000/video/clear`

Behavior:
- Requests switch back to default source.
- Used by frontend reset flow to clear uploaded clip.

## 3.2) Playback Controls (Uploaded File Mode)
Control endpoint:
- `POST http://<host>:8000/video/control?action=<pause|resume|seek_to|seek_by>&seconds=<float>`

Examples:
- Pause: `.../video/control?action=pause`
- Resume: `.../video/control?action=resume`
- Seek to 12.5s: `.../video/control?action=seek_to&seconds=12.5`
- Seek forward 5s: `.../video/control?action=seek_by&seconds=5`

Playback status endpoint:
- `GET http://<host>:8000/video/playback`

Response shape:
```json
{
  "active_source": "/path/to/video.mp4",
  "is_file_source": true,
  "paused": false,
  "position_sec": 42.7,
  "duration_sec": 133.4
}
```

Error response example:
```json
{
  "status": "error",
  "message": "Unsupported file type. Use mp4/avi/mov/mkv/webm."
}
```

## 4) Optional History Endpoints
The v2 UI can be wired to these when ready:
- `GET /history`
- `GET /history/{session_id}`

## 5) Frontend Adapter
Normalization happens in:
- `src/lib/dataAdapter.js`

If backend field names differ, update only this adapter.
