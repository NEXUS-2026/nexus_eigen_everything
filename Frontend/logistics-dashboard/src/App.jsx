import { useEffect, useRef, useState, useCallback } from "react";

// Constants

const WS_URL           = `ws://${window.location.host}/ws`;
const API_BASE         = `${window.location.protocol}//${window.location.hostname}:8000`;
const RECONNECT_DELAY  = 3000;   // ms before attempting a reconnect
const METRIC_THROTTLE  = 6;      // update metric useState every N frames (keeps React re-renders at 6~7/sec)

// Connection status definitions

const STATUS = {
  CONNECTING:    "Connecting",
  LIVE:          "Live",
  DISCONNECTED:  "Disconnected",
};

// Tailwind colour classes mapped to each status
const STATUS_STYLES = {
  [STATUS.CONNECTING]:   { dot: "bg-amber-300 status-scan", text: "text-amber-300", ring: "border-amber-300/40" },
  [STATUS.LIVE]:         { dot: "bg-emerald-300 status-scan", text: "text-emerald-300", ring: "border-emerald-300/40" },
  [STATUS.DISCONNECTED]: { dot: "bg-rose-300", text: "text-rose-300", ring: "border-rose-300/40" },
};

const ALERT_COPY = {
  [STATUS.LIVE]: {
    tone: "border-emerald-300/35 bg-emerald-500/10 text-emerald-200",
    text: "System online. Tracker feed and control link are stable.",
  },
  [STATUS.CONNECTING]: {
    tone: "border-amber-300/35 bg-amber-500/10 text-amber-200",
    text: "Attempting handshake with backend stream endpoint.",
  },
  [STATUS.DISCONNECTED]: {
    tone: "border-rose-300/35 bg-rose-500/10 text-rose-200",
    text: "Stream link interrupted. Auto-reconnect routine is active.",
  },
};

// Small presentational components

function MetricCard({ label, value, unit, accent = "text-cyan-200" }) {
  return (
    <div className="
      panel-metal panel-reveal rounded-xl
      p-4 flex flex-col gap-1
    ">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
        {label}
      </span>
      <div className="flex items-end gap-2">
        <span className={`text-4xl font-semibold tabular-nums leading-none ${accent} tech-title`}>
          {value}
        </span>
        {unit && (
          <span className="mb-1 text-xs text-slate-500 font-medium uppercase tracking-[0.12em]">{unit}</span>
        )}
      </div>
    </div>
  );
}


function StatusCard({ status, fps, infMs }) {
  const styles = STATUS_STYLES[status] ?? STATUS_STYLES[STATUS.DISCONNECTED];

  return (
    <div className="
      panel-metal panel-reveal rounded-xl
      p-4 flex flex-col gap-4
    ">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
        System Status Board
      </span>

      {/* Connection badge */}
      <div className={`flex items-center justify-between gap-3 rounded-lg border ${styles.ring} bg-slate-950/70 px-3 py-2`}>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
          <span className={`
            absolute inline-flex h-full w-full rounded-full opacity-75
            ${styles.dot}
          `} />
          <span className={`
            relative inline-flex rounded-full h-3 w-3
            ${status === STATUS.LIVE ? "bg-emerald-300" :
              status === STATUS.CONNECTING ? "bg-amber-300" : "bg-rose-300"}
          `} />
          </span>
          <span className={`text-sm font-semibold uppercase tracking-[0.14em] ${styles.text} tech-title`}>
            {status}
          </span>
        </div>
        <span className="text-[10px] text-slate-500 uppercase tracking-[0.12em]">Link Active</span>
      </div>

      {/* Sub-metrics row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-slate-700/60 bg-slate-950/70 p-3 flex flex-col gap-0.5">
          <span className="text-[10px] text-slate-500 uppercase tracking-[0.13em]">
            Capture FPS
          </span>
          <span className="text-xl font-semibold tabular-nums text-cyan-300 tech-title">
            {fps.toFixed(1)}
          </span>
        </div>
        <div className="rounded-lg border border-slate-700/60 bg-slate-950/70 p-3 flex flex-col gap-0.5">
          <span className="text-[10px] text-slate-500 uppercase tracking-[0.13em]">
            YOLO Latency
          </span>
          <span className="text-xl font-semibold tabular-nums text-orange-300 tech-title">
            {infMs.toFixed(0)}
            <span className="text-xs font-normal text-slate-500 ml-1 uppercase">ms</span>
          </span>
        </div>
      </div>
    </div>
  );
}

function ResetButton({ onReset, disabled }) {
  const [confirming, setConfirming] = useState(false);

  /**
   * Two-click confirmation pattern:
   *   First click  → button turns red, asks "Confirm Reset?"
   *   Second click → fires the actual reset
   *   No second click within 3 s → reverts to idle state
   * Prevents accidental session wipes during a live packing run.
   */
  const handleClick = useCallback(() => {
    if (!confirming) {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 3000);
    } else {
      setConfirming(false);
      onReset();
    }
  }, [confirming, onReset]);

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`
        w-full rounded-2xl border px-5 py-4
        text-sm font-semibold uppercase tracking-[0.14em] tech-title
        transition-all duration-200 active:scale-95
        disabled:opacity-30 disabled:cursor-not-allowed
        ${confirming
          ? "border-rose-400 bg-rose-500/20 text-rose-200 hover:bg-rose-500/30"
          : "border-cyan-500/40 bg-cyan-500/10 text-cyan-100 hover:border-cyan-300/80 hover:bg-cyan-500/20"
        }
      `}
    >
      {confirming ? "Confirm Reset Session" : "Reset Session"}
    </button>
  );
}

// Main Dashboard component

export default function Dashboard() {

  // Refs (never trigger re renders)

  /**
   * canvasRef — the ONLY way we touch the video canvas.
   * All 40FPS pixel writes go through this ref, completely
   * bypassing React's reconciler.
   */
  const canvasRef = useRef(null);

  /**
   * wsRef : holds the live WebSocket instance.
   * A ref (not state) because we never want a re render just
   * because the socket object was reassigned during reconnect.
   */
  const wsRef = useRef(null);

  /**
   * frameTickRef : counts incoming frames since last metric flush.
   * Lets us throttle the 3 useState setters to ~6 calls/sec instead
   * of 40, cutting React's render budget by ~85%.
   */
  const frameTickRef = useRef(0);

  /**
   * latestMetricsRef : stores the most recent count/fps/infMs values.
   * The throttle logic reads from here when it's time to flush to state.
   */
  const latestMetricsRef = useRef({ count: 0, fps: 0, infMs: 0 });

  // State (only what drives UI text, updated at throttled rate)
  const [count,  setCount]  = useState(0);
  const [fps,    setFps]    = useState(0);
  const [infMs,  setInfMs]  = useState(0);
  const [status, setStatus] = useState(STATUS.CONNECTING);
  const [isUploadingVideo, setIsUploadingVideo] = useState(false);
  const [uploadVideoMessage, setUploadVideoMessage] = useState("");
  const [uploadVideoError, setUploadVideoError] = useState("");

  // Canvas drawing helper

  /**
   * drawFrameToCanvas
   * Takes a raw base64 JPEG string (no data-URI prefix), wraps it in
   * a native Image, and paints it onto the canvas when loaded.
   *
   * This function NEVER calls any React setter it is pure DOM/Canvas API.
   * Execution path: onmessage to drawFrameToCanvas to img.onload to ctx.drawImage
   *
   * @param {string} b64jpeg  : raw base64 string from the WebSocket payload
   */
  const drawFrameToCanvas = useCallback((b64jpeg) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: false }); // alpha:false = faster compositing
    if (!ctx) return;

    const img  = new Image();

    img.onload = () => {
      // Resize the canvas to match the incoming frame exactly (first frame only
      // changes dimensions; subsequent frames skip the branch in practice).
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      // drawImage is a single GPU blit effectively free on the main thread.
      ctx.drawImage(img, 0, 0);
    };

    // Setting src AFTER onload assignment avoids a race condition in some browsers
    // where a cached image fires onload synchronously before the handler is set.
    img.src = `data:image/jpeg;base64,${b64jpeg}`;
  }, []);

  // WebSocket management

  /**
   * connectWebSocket
   * Creates a new WebSocket, wires up all event handlers, and stores
   * the instance in wsRef. Called once on mount and again after each
   * unexpected disconnect (auto reconnect).
   *
   * Defined with useCallback so it has a stable reference the
   * reconnect setTimeout can safely call it without capturing a stale closure.
   */
  const connectWebSocket = useCallback(() => {
    // Clean up any existing socket before creating a new one
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent the old socket's onclose from re triggering reconnect
      wsRef.current.close();
    }

    setStatus(STATUS.CONNECTING);
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    // onopen
    ws.onopen = () => {
      setStatus(STATUS.LIVE);
    };

    // onmessage — the 40 FPS hot path
    ws.onmessage = (event) => {
      // Parse the JSON payload
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return; // malformed message skip silently
      }

      // Paint the frame directly to canvas (zero React involvement)
      if (data.frame) {
        drawFrameToCanvas(data.frame);
      }

      // Buffer the latest metrics into the ref (free no re render)
      latestMetricsRef.current = {
        count: data.count  ?? 0,
        fps:   data.fps    ?? 0,
        infMs: data.inf_ms ?? 0,
      };

      // Throttle: only flush to useState every METRIC_THROTTLE frames.
      // This limits React re renders to ~6/sec regardless of stream rate.
      frameTickRef.current += 1;
      if (frameTickRef.current >= METRIC_THROTTLE) {
        frameTickRef.current = 0;
        const { count, fps, infMs } = latestMetricsRef.current;
        // Batch all three setters in a single synchronous block.
        // React 18 automatically batches these into one re render.
        setCount(count);
        setFps(fps);
        setInfMs(infMs);
      }
    };

    // onerror
    ws.onerror = (err) => {
      console.error("[WS] error:", err);
      // onclose will fire immediately after onerror handle state there
    };

    // onclose 
    ws.onclose = () => {
      setStatus(STATUS.DISCONNECTED);
      // Auto reconnect after RECONNECT_DELAY ms
      setTimeout(connectWebSocket, RECONNECT_DELAY);
    };
  }, [drawFrameToCanvas]);

  // Mount / unmount

  useEffect(() => {
    connectWebSocket();

    // Cleanup: close the socket when the component unmounts (dev HMR, nav away)
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null; // suppress the auto reconnect on intentional close
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Reset handler

  /**
   * handleReset
   * Sends the exact string "reset" to the backend.
   * main.py's WebSocket handler checks for this string and calls
   * state_machine.reset() + db.end_session() + db.start_session().
   */
  const handleReset = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("reset");
    }
  }, []);

  const handleVideoUpload = useCallback(async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploadVideoError("");
    setUploadVideoMessage("");
    setIsUploadingVideo(true);

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API_BASE}/video/upload`, {
        method: "POST",
        body,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.message ?? "Upload failed.");
      }

      setUploadVideoMessage(`Loaded ${payload.filename}. Switching feed source...`);
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setIsUploadingVideo(false);
    }
  }, []);

  // Render

  const alert = ALERT_COPY[status] ?? ALERT_COPY[STATUS.DISCONNECTED];

  return (
    /*
     * Root : full viewport dark slate background.
     * Using a subtle radial gradient for depth without hurting contrast.
     */
    <div className="min-h-screen w-full text-slate-100 flex flex-col">

      {/* ── Top header bar ───────────────────────────────────────────────── */}
      <header className="
        panel-metal
        sticky top-0 z-20
        flex items-center justify-between
        px-5 py-3 lg:px-7
        backdrop-blur-md
      ">
        <div className="flex items-center gap-3">
          {/* Warehouse icon — plain SVG, no icon lib dependency */}
          <svg className="w-7 h-7 text-cyan-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"
            />
          </svg>
          <div>
            <h1 className="text-base lg:text-lg font-semibold tracking-[0.12em] uppercase text-cyan-50 tech-title">
              Warehouse Box Counter
            </h1>
            <p className="text-[11px] text-slate-400 tracking-[0.16em] uppercase">
              Industrial Monitoring Console
            </p>
          </div>
        </div>

        {/* Live timestamp — updates every second via a tiny inner component */}
        <LiveClock />
      </header>

      <div className="max-w-screen-2xl mx-auto w-full px-3 sm:px-4 lg:px-6 pt-3">
        <div className={`panel-metal rounded-xl px-4 py-2.5 text-[11px] uppercase tracking-[0.12em] border ${alert.tone}`}>
          {alert.text}
        </div>
      </div>

      {/* ── Main content grid ────────────────────────────────────────────── */}
      <main className="
        flex-1
        grid grid-cols-1 lg:grid-cols-[1fr_320px]
        gap-3 sm:gap-4 p-3 sm:p-4 lg:p-6
        max-w-screen-2xl mx-auto w-full
      ">

        {/* ── Left column: video canvas ──────────────────────────────────── */}
        <div className="flex flex-col gap-4 order-2 lg:order-1">

          {/* Canvas wrapper — maintains 16:9 aspect ratio */}
          <div className="panel-metal panel-reveal rounded-2xl p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-700/70 bg-slate-950/75 px-3 py-2 mb-3">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-[0.13em]">Live Feed</p>
                <p className="text-sm text-cyan-100 tech-title uppercase tracking-[0.1em]">Carton Camera Channel</p>
              </div>
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.1em]">
                <span className="rounded border border-cyan-400/30 bg-cyan-400/10 px-2 py-1 text-cyan-200">Source WS</span>
                <span className="rounded border border-slate-600 px-2 py-1 text-slate-300">JPEG</span>
                <span className="rounded border border-slate-600 px-2 py-1 text-slate-300">40 FPS target</span>
              </div>
            </div>
            <div className="
              relative rounded-xl overflow-hidden
              border border-cyan-400/25
              bg-slate-950
              aspect-video w-full
              shadow-2xl shadow-black/60
            ">
            {/*
             * THE CANVAS : this is the only element that receives video.
             * canvasRef is the ref; drawFrameToCanvas writes to it at 40 FPS.
             * React never re renders this element because we don't change any
             * of its props after mount.
             *
             * Initial dimensions (640×480) will be overwritten on the first
             * frame received see drawFrameToCanvas().
             */}
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="w-full h-full object-cover"
              aria-label="Live camera feed"
            />

            {/* Overlay: shown when not live */}
            {status !== STATUS.LIVE && (
              <div className="
                absolute inset-0
                flex flex-col items-center justify-center gap-3
                bg-slate-950/82 backdrop-blur-sm
              ">
                <div className="w-10 h-10 border-2 border-slate-600 border-t-cyan-300 rounded-full animate-spin" />
                <p className="text-slate-300 text-xs md:text-sm tracking-[0.15em] uppercase tech-title">
                  {status === STATUS.CONNECTING ? "Connecting to feed…" : "Feed disconnected — reconnecting…"}
                </p>
              </div>
            )}

            {/* Corner badge — always visible when live */}
            {status === STATUS.LIVE && (
              <div className="
                absolute top-3 left-3
                flex items-center gap-2
                rounded-full px-3 py-1
                bg-slate-950/70 backdrop-blur-sm
                border border-emerald-300/35
              ">
                <span className="w-2 h-2 rounded-full bg-emerald-300 status-scan" />
                <span className="text-[11px] font-semibold text-emerald-300 tracking-[0.12em] uppercase tech-title">
                  Live
                </span>
              </div>
            )}
          </div>
          </div>

          {/* Bottom info strip below canvas */}
          <div className="
            panel-metal panel-reveal rounded-xl
            px-4 py-3
            grid grid-cols-1 sm:grid-cols-3 gap-2
            text-[11px] text-slate-400 tracking-[0.12em] uppercase
          ">
            <span>Source: <span className="text-cyan-200 normal-case tracking-normal">ws://localhost:8000/ws</span></span>
            <span>Codec: <span className="text-cyan-200 normal-case tracking-normal">JPEG / 40 FPS target</span></span>
            <span>Renderer: <span className="text-cyan-200 normal-case tracking-normal">HTML5 Canvas</span></span>
          </div>
        </div>

        {/* ── Right sidebar ──────────────────────────────────────────────── */}
        <aside className="flex flex-col gap-4 order-1 lg:order-2">

          {/* 1. Box count — the primary KPI, largest element */}
          <div className="
            panel-metal panel-reveal rounded-xl
            p-5 flex flex-col gap-2
            border border-cyan-300/30
          ">
            <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              Total Boxes in Carton
            </span>
            {/*
             * The count number.
             * tabular-nums keeps the layout stable as digits change width.
             * This re-renders at ~6/sec (throttled), not 40/sec.
             */}
            <div className={`
              text-7xl md:text-8xl font-semibold tabular-nums leading-none tech-title
              transition-colors duration-300
              ${count > 0 ? "text-cyan-200" : "text-slate-600"}
            `}>
              {count}
            </div>
            <span className="text-[11px] text-slate-500 mt-1 uppercase tracking-[0.12em]">
              CONFIRMED_INSIDE + HIDDEN_INSIDE tracks
            </span>
          </div>

          {/* 2. System status card */}
          <StatusCard status={status} fps={fps} infMs={infMs} />

          {/* 3. Reset placed early for mobile operator ergonomics */}
          <ResetButton
            onReset={handleReset}
            disabled={status !== STATUS.LIVE}
          />

          <VideoUploadPanel
            onUpload={handleVideoUpload}
            isUploading={isUploadingVideo}
            message={uploadVideoMessage}
            error={uploadVideoError}
          />

          {/* 4. Additional metrics */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="FPS"
              value={fps.toFixed(1)}
              accent="text-cyan-200"
            />
            <MetricCard
              label="Latency"
              value={infMs.toFixed(0)}
              unit="ms"
              accent="text-orange-200"
            />
          </div>

          {/* Divider */}
          <div className="border-t border-slate-700/60" />

          {/* 5. Session history shell (UI only for now) */}
          <SessionHistoryPanel />

          {/* 6. State legend */}
          <StateLegend />

          {/* 7. Operator notes shell */}
          <OperatorNotes />

        </aside>
      </main>
    </div>
  );
}

// Supporting sub components

/**
 * LiveClock
 * Updates its own internal state every second isolated from the main
 * component tree so a tick never re renders Dashboard or the canvas.
 */
function LiveClock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString());

  useEffect(() => {
    const id = setInterval(
      () => setTime(new Date().toLocaleTimeString()),
      1000,
    );
    return () => clearInterval(id);
  }, []);

  return (
    <div className="text-right hidden sm:block">
      <p className="text-[10px] text-slate-500 uppercase tracking-[0.13em]">Local time</p>
      <p className="text-sm font-semibold tabular-nums text-cyan-100 tech-title">{time}</p>
    </div>
  );
}

/**
 * StateLegend
 * Explains the colour coding used by the Python annotation layer.
 * Pure presentational no props, no state, never re renders after mount.
 */
function StateLegend() {
  const states = [
    { colour: "bg-green-400",  label: "CONFIRMED_INSIDE",  desc: "Counted & stable"    },
    { colour: "bg-amber-400",  label: "PENDING_ENTER",     desc: "Debouncing (5 frames)" },
    { colour: "bg-blue-400",   label: "HIDDEN_INSIDE",     desc: "Occluded / stacked"  },
    { colour: "bg-slate-500",  label: "REMOVED",           desc: "Left the carton"     },
  ];

  return (
    <div className="
      panel-metal panel-reveal rounded-xl
      p-4 flex flex-col gap-3
    ">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
        Track State Legend
      </span>
      <ul className="flex flex-col gap-2">
        {states.map(({ colour, label, desc }) => (
          <li key={label} className="flex items-center gap-3 rounded-md border border-slate-700/70 bg-slate-950/60 px-3 py-2">
            <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${colour}`} />
            <div className="flex flex-col leading-tight">
              <span className="text-[11px] font-semibold text-slate-200 tracking-[0.08em] tech-title">{label}</span>
              <span className="text-[11px] text-slate-500">{desc}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SessionHistoryPanel() {
  return (
    <div className="panel-metal panel-reveal rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
          Session History
        </span>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">UI Shell</span>
      </div>
      <div className="rounded-md border border-slate-700/70 bg-slate-950/60 overflow-hidden">
        <div className="grid grid-cols-[1.3fr_0.8fr_0.9fr] text-[10px] uppercase tracking-[0.11em] text-slate-500 px-3 py-2 border-b border-slate-700/60">
          <span>Session</span>
          <span>Boxes</span>
          <span>Duration</span>
        </div>
        <div className="px-3 py-4 text-[11px] text-slate-500">
          No session data loaded yet.
        </div>
      </div>
    </div>
  );
}

function OperatorNotes() {
  return (
    <div className="panel-metal panel-reveal rounded-xl p-4 flex flex-col gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
        Operator Notes
      </span>
      <p className="text-[11px] text-slate-500 leading-relaxed">
        Placeholder block for shift comments, carton batch tags, and incident remarks.
      </p>
      <div className="rounded-md border border-slate-700/70 bg-slate-950/60 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-slate-500">
        Notes API wiring next checkpoint
      </div>
    </div>
  );
}

function VideoUploadPanel({ onUpload, isUploading, message, error }) {
  return (
    <div className="panel-metal panel-reveal rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
          Test Video Input
        </span>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Browser Upload</span>
      </div>

      <label className="cursor-pointer">
        <input
          type="file"
          accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
          className="hidden"
          onChange={onUpload}
          disabled={isUploading}
        />
        <span className={`
          w-full inline-flex items-center justify-center rounded-lg border px-4 py-3
          text-[11px] uppercase tracking-[0.13em] tech-title
          ${isUploading
            ? "border-slate-600 bg-slate-800/70 text-slate-400"
            : "border-cyan-400/40 bg-cyan-500/10 text-cyan-100 hover:bg-cyan-500/20"
          }
        `}>
          {isUploading ? "Uploading Video..." : "Upload Test Video"}
        </span>
      </label>

      <p className="text-[10px] text-slate-500 uppercase tracking-[0.1em]">
        Supported: mp4, avi, mov, mkv, webm
      </p>

      {message && (
        <div className="rounded-md border border-emerald-300/30 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-200">
          {message}
        </div>
      )}

      {error && (
        <div className="rounded-md border border-rose-300/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-200">
          {error}
        </div>
      )}
    </div>
  );
}