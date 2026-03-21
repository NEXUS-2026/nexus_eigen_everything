# Frontend V2 Handoff Notes

## Goal
Build and polish the dashboard UI in isolation while backend model integration is pending.

## Scope Rules
- Frontend-only changes under `Frontend/logistics-dashboard-v2`.
- No changes to backend, tracker, or YOLO engine during this phase.

## Run
1. Start backend on port 8000.
2. In this folder run:
   - `npm install`
   - `npm run dev -- --host 0.0.0.0 --port 5173`

## Current V2 Structure
- `src/App.jsx`: websocket and upload orchestration.
- `src/components/HeaderBar.jsx`: top shell and clock.
- `src/components/VideoStage.jsx`: video canvas and stream metadata.
- `src/components/RightRail.jsx`: count hero, status board, controls, upload, KPI cards.
- `src/components/LowerDeck.jsx`: session table, legend, notes.

## Current UX State
- Works with live video websocket stream.
- Upload panel posts to `/video/upload`.
- Reset now clears uploaded source via `/video/clear` (along with session reset).
- Uploaded video playback controls available in UI:
  - pause/resume
  - seek -5s / +5s
  - timeline scrub
- Includes model-pending preview mode for UI-only demo:
  - engine mode card
  - mock session rows
  - preview hints in lower panels

## Integration Later (when model arrives)
1. Keep layout/components unchanged.
2. Replace model-pending heuristics in `src/App.jsx` with real backend flags if available.
3. Swap mock session rows in `src/components/LowerDeck.jsx` with real `/history` data.
4. Keep v2 as final frontend and retire old frontend only after acceptance.

## Suggested Next UI Tasks
1. Add event timeline panel (enter/hidden/removed events).
2. Add compact mobile action bar.
3. Add keyboard shortcuts for reset/upload.
4. Add theme tokens for light industrial variant.
