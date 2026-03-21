import { useEffect, useRef, useState, useCallback } from "react";

// Constants

const WS_URL           = "ws://localhost:8000/ws";
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
  [STATUS.CONNECTING]:   { dot: "bg-amber-400 animate-ping",  text: "text-amber-400" },
  [STATUS.LIVE]:         { dot: "bg-green-400 animate-pulse", text: "text-green-400" },
  [STATUS.DISCONNECTED]: { dot: "bg-red-500",                 text: "text-red-500"   },
};

// Small presentational components

function MetricCard({ label, value, unit, accent = "text-green-400" }) {
  return (
    <div className="
      rounded-2xl border border-slate-700 bg-slate-800/60
      p-5 flex flex-col gap-1 backdrop-blur-sm
    ">
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        {label}
      </span>
      <div className="flex items-end gap-2">
        <span className={`text-5xl font-black tabular-nums leading-none ${accent}`}>
          {value}
        </span>
        {unit && (
          <span className="mb-1 text-sm text-slate-500 font-medium">{unit}</span>
        )}
      </div>
    </div>
  );
}


function StatusCard({ status, fps, infMs }) {
  const styles = STATUS_STYLES[status] ?? STATUS_STYLES[STATUS.DISCONNECTED];

  return (
    <div className="
      rounded-2xl border border-slate-700 bg-slate-800/60
      p-5 flex flex-col gap-4 backdrop-blur-sm
    ">
      {/* Header row */}
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        System Status
      </span>

      {/* Connection badge */}
      <div className="flex items-center gap-3">
        {/* Pulsing indicator dot */}
        <span className="relative flex h-3 w-3">
          <span className={`
            absolute inline-flex h-full w-full rounded-full opacity-75
            ${styles.dot}
          `} />
          <span className={`
            relative inline-flex rounded-full h-3 w-3
            ${status === STATUS.LIVE ? "bg-green-400" :
              status === STATUS.CONNECTING ? "bg-amber-400" : "bg-red-500"}
          `} />
        </span>
        <span className={`text-lg font-bold ${styles.text}`}>
          {status}
        </span>
      </div>

      {/* Sub-metrics row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-slate-900/70 p-3 flex flex-col gap-0.5">
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            Capture FPS
          </span>
          <span className="text-xl font-bold tabular-nums text-cyan-400">
            {fps.toFixed(1)}
          </span>
        </div>
        <div className="rounded-xl bg-slate-900/70 p-3 flex flex-col gap-0.5">
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            YOLO Latency
          </span>
          <span className="text-xl font-bold tabular-nums text-violet-400">
            {infMs.toFixed(0)}
            <span className="text-sm font-normal text-slate-500 ml-1">ms</span>
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
        text-sm font-bold uppercase tracking-widest
        transition-all duration-200 active:scale-95
        disabled:opacity-30 disabled:cursor-not-allowed
        ${confirming
          ? "border-red-500 bg-red-500/20 text-red-400 hover:bg-red-500/30"
          : "border-slate-600 bg-slate-800/60 text-slate-300 hover:border-slate-400 hover:text-white"
        }
      `}
    >
      {confirming ? "⚠ Confirm Reset?" : "↺ Reset Session"}
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

  // Render

  return (
    /*
     * Root : full viewport dark slate background.
     * Using a subtle radial gradient for depth without hurting contrast.
     */
    <div className="
      min-h-screen w-full
      bg-slate-950
      text-slate-100
      font-mono
      flex flex-col
    "
    style={{
      background: "radial-gradient(ellipse at 20% 10%, #0f172a 0%, #020617 70%)"
    }}
    >

      {/* ── Top header bar ───────────────────────────────────────────────── */}
      <header className="
        flex items-center justify-between
        px-6 py-4
        border-b border-slate-800
        bg-slate-900/50 backdrop-blur-md
      ">
        <div className="flex items-center gap-3">
          {/* Warehouse icon — plain SVG, no icon lib dependency */}
          <svg className="w-7 h-7 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"
            />
          </svg>
          <div>
            <h1 className="text-sm font-bold tracking-widest uppercase text-white">
              Warehouse Box Counter
            </h1>
            <p className="text-xs text-slate-500 tracking-wider">
              Edge AI · YOLO + ByteTrack Pipeline
            </p>
          </div>
        </div>

        {/* Live timestamp — updates every second via a tiny inner component */}
        <LiveClock />
      </header>

      {/* ── Main content grid ────────────────────────────────────────────── */}
      <main className="
        flex-1
        grid grid-cols-1 lg:grid-cols-[1fr_320px]
        gap-4 p-4
        max-w-screen-2xl mx-auto w-full
      ">

        {/* ── Left column: video canvas ──────────────────────────────────── */}
        <div className="flex flex-col gap-4">

          {/* Canvas wrapper — maintains 16:9 aspect ratio */}
          <div className="
            relative rounded-2xl overflow-hidden
            border border-slate-700
            bg-slate-900
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
              className="w-full h-full object-contain"
              aria-label="Live camera feed"
            />

            {/* Overlay: shown when not live */}
            {status !== STATUS.LIVE && (
              <div className="
                absolute inset-0
                flex flex-col items-center justify-center gap-3
                bg-slate-950/80 backdrop-blur-sm
              ">
                <div className="w-10 h-10 border-2 border-slate-600 border-t-green-400 rounded-full animate-spin" />
                <p className="text-slate-400 text-sm tracking-widest uppercase">
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
                bg-black/50 backdrop-blur-sm
                border border-green-500/30
              ">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-xs font-bold text-green-400 tracking-widest uppercase">
                  Live
                </span>
              </div>
            )}
          </div>

          {/* Bottom info strip below canvas */}
          <div className="
            rounded-2xl border border-slate-800 bg-slate-900/40
            px-5 py-3
            flex items-center justify-between
            text-xs text-slate-500 tracking-wider
          ">
            <span>Source: <span className="text-slate-400">ws://localhost:8000/ws</span></span>
            <span>Codec: <span className="text-slate-400">JPEG / 40 FPS target</span></span>
            <span>Renderer: <span className="text-slate-400">HTML5 Canvas (direct blit)</span></span>
          </div>
        </div>

        {/* ── Right sidebar ──────────────────────────────────────────────── */}
        <aside className="flex flex-col gap-4">

          {/* 1. Box count — the primary KPI, largest element */}
          <div className="
            rounded-2xl border border-green-500/30 bg-slate-800/60
            p-6 flex flex-col gap-2
            shadow-lg shadow-green-900/10
            backdrop-blur-sm
          ">
            <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Total Boxes in Carton
            </span>
            {/*
             * The count number.
             * tabular-nums keeps the layout stable as digits change width.
             * This re-renders at ~6/sec (throttled), not 40/sec.
             */}
            <div className={`
              text-8xl font-black tabular-nums leading-none
              transition-colors duration-300
              ${count > 0 ? "text-green-400" : "text-slate-600"}
            `}>
              {count}
            </div>
            <span className="text-xs text-slate-600 mt-1">
              CONFIRMED_INSIDE + HIDDEN_INSIDE tracks
            </span>
          </div>

          {/* 2. System status card */}
          <StatusCard status={status} fps={fps} infMs={infMs} />

          {/* 3. Additional metrics */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="FPS"
              value={fps.toFixed(1)}
              accent="text-cyan-400"
            />
            <MetricCard
              label="Latency"
              value={infMs.toFixed(0)}
              unit="ms"
              accent="text-violet-400"
            />
          </div>

          {/* Divider */}
          <div className="border-t border-slate-800" />

          {/* 4. Reset button */}
          <ResetButton
            onReset={handleReset}
            disabled={status !== STATUS.LIVE}
          />

          {/* 5. State legend */}
          <StateLegend />

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
      <p className="text-xs text-slate-500 uppercase tracking-widest">Local time</p>
      <p className="text-sm font-bold tabular-nums text-slate-300">{time}</p>
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
      rounded-2xl border border-slate-700 bg-slate-800/60
      p-5 flex flex-col gap-3 backdrop-blur-sm
    ">
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        Track State Legend
      </span>
      <ul className="flex flex-col gap-2">
        {states.map(({ colour, label, desc }) => (
          <li key={label} className="flex items-center gap-3">
            <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${colour}`} />
            <div className="flex flex-col leading-tight">
              <span className="text-xs font-bold text-slate-300">{label}</span>
              <span className="text-xs text-slate-600">{desc}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}