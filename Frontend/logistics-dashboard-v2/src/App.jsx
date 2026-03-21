import { useCallback, useEffect, useRef, useState } from "react";
import { HeaderBar } from "./components/HeaderBar";
import { VideoStage } from "./components/VideoStage";
import { RightRail } from "./components/RightRail";
import { LowerDeck } from "./components/LowerDeck";
import { normalizeWsPayload } from "./lib/dataAdapter";

const WS_URL = `ws://${window.location.host}/ws`;
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
const RECONNECT_DELAY = 3000;
const METRIC_THROTTLE = 6;

const STATUS = {
  CONNECTING: "Connecting",
  LIVE: "Live",
  DISCONNECTED: "Disconnected",
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

export default function App() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const frameTickRef = useRef(0);
  const latestMetricsRef = useRef({ count: 0, fps: 0, infMs: 0 });

  const [count, setCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [infMs, setInfMs] = useState(0);
  const [status, setStatus] = useState(STATUS.CONNECTING);
  const [isUploadingVideo, setIsUploadingVideo] = useState(false);
  const [uploadVideoMessage, setUploadVideoMessage] = useState("");
  const [uploadVideoError, setUploadVideoError] = useState("");
  const [recentEvents, setRecentEvents] = useState([]);
  const [modelReady, setModelReady] = useState(false);
  const [playback, setPlayback] = useState({
    isFileSource: false,
    paused: false,
    positionSec: 0,
    durationSec: 0,
  });

  const drawFrameToCanvas = useCallback((b64jpeg) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      ctx.drawImage(img, 0, 0);
    };

    img.src = `data:image/jpeg;base64,${b64jpeg}`;
  }, []);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }

    setStatus(STATUS.CONNECTING);
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus(STATUS.LIVE);
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      const payload = normalizeWsPayload(data);

      if (payload.frame) {
        drawFrameToCanvas(payload.frame);
      }

      latestMetricsRef.current = {
        count: payload.count,
        fps: payload.fps,
        infMs: payload.infMs,
      };

      setModelReady(payload.modelReady);

      if (payload.events.length > 0) {
        setRecentEvents((prev) => {
          const merged = [...payload.events, ...prev];
          return merged.slice(0, 10);
        });
      }

      frameTickRef.current += 1;
      if (frameTickRef.current >= METRIC_THROTTLE) {
        frameTickRef.current = 0;
        const latest = latestMetricsRef.current;
        setCount(latest.count);
        setFps(latest.fps);
        setInfMs(latest.infMs);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] error:", err);
    };

    ws.onclose = () => {
      setStatus(STATUS.DISCONNECTED);
      setTimeout(connectWebSocket, RECONNECT_DELAY);
    };
  }, [drawFrameToCanvas]);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (!wsRef.current) return;
      wsRef.current.onclose = null;
      wsRef.current.close();
    };
  }, [connectWebSocket]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const handleReset = useCallback(async () => {
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send("reset");
      }
      await fetch(`${API_BASE}/reset`, { method: "POST" });
      await fetch(`${API_BASE}/video/clear`, { method: "POST" });
      setCount(0);
      setInfMs(0);
      setRecentEvents([]);
      setUploadVideoMessage("Session reset and uploaded source cleared.");
      clearCanvas();
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Reset failed.");
    }
  }, [clearCanvas]);

  const handleClearVideo = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/video/clear`, { method: "POST" });
      setUploadVideoMessage("Cleared current uploaded video source.");
      setUploadVideoError("");
      clearCanvas();
      setPlayback((prev) => ({
        ...prev,
        isFileSource: false,
        paused: false,
        positionSec: 0,
        durationSec: 0,
      }));
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Clear video failed.");
    }
  }, [clearCanvas]);

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

  const alert = ALERT_COPY[status] ?? ALERT_COPY[STATUS.DISCONNECTED];
  const modelPending = !modelReady && infMs <= 0.05 && count === 0;

  const sendPlaybackCommand = useCallback(async (action, seconds) => {
    const qs = new URLSearchParams({ action });
    if (typeof seconds === "number" && Number.isFinite(seconds)) {
      qs.set("seconds", String(seconds));
    }

    const response = await fetch(`${API_BASE}/video/control?${qs.toString()}`, {
      method: "POST",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.message ?? "Playback control failed.");
    }
  }, []);

  const handleTogglePause = useCallback(async () => {
    try {
      await sendPlaybackCommand(playback.paused ? "resume" : "pause");
      setPlayback((prev) => ({ ...prev, paused: !prev.paused }));
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Playback command failed.");
    }
  }, [playback.paused, sendPlaybackCommand]);

  const handleSeekBy = useCallback(async (seconds) => {
    try {
      await sendPlaybackCommand("seek_by", seconds);
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Seek command failed.");
    }
  }, [sendPlaybackCommand]);

  const handleSeekTo = useCallback(async (seconds) => {
    try {
      await sendPlaybackCommand("seek_to", seconds);
    } catch (error) {
      setUploadVideoError(error instanceof Error ? error.message : "Seek command failed.");
    }
  }, [sendPlaybackCommand]);

  useEffect(() => {
    let alive = true;

    const pull = async () => {
      try {
        const response = await fetch(`${API_BASE}/video/playback`);
        const payload = await response.json();
        if (!alive) return;
        setPlayback({
          isFileSource: Boolean(payload.is_file_source),
          paused: Boolean(payload.paused),
          positionSec: Number(payload.position_sec) || 0,
          durationSec: Number(payload.duration_sec) || 0,
        });
      } catch {
        // ignore transient polling errors
      }
    };

    pull();
    const id = setInterval(pull, 700);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const handleScrollToUpload = useCallback(() => {
    const node = document.getElementById("upload-panel");
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, []);

  return (
    <div className="min-h-screen w-full text-slate-100">
      <HeaderBar statusText="Detection Engine: Pending Integration" />

      <main className="mx-auto flex w-full max-w-[1500px] flex-col gap-4 px-3 pb-5 pt-3 sm:px-4 lg:gap-5 lg:px-6 lg:pb-7">
        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <VideoStage
            canvasRef={canvasRef}
            status={status}
            isLive={status === STATUS.LIVE}
            alertText={alert.text}
            alertTone={alert.tone}
            playback={playback}
            onTogglePause={handleTogglePause}
            onSeekBy={handleSeekBy}
            onSeekTo={handleSeekTo}
            onClearVideo={handleClearVideo}
          />

          <RightRail
            count={count}
            fps={fps}
            infMs={infMs}
            status={status}
            modelPending={modelPending}
            onReset={handleReset}
            onUpload={handleVideoUpload}
            isUploading={isUploadingVideo}
            uploadMessage={uploadVideoMessage}
            uploadError={uploadVideoError}
          />
        </section>

        <LowerDeck modelPending={modelPending} liveCount={count} liveEvents={recentEvents} />
      </main>

      <MobileQuickBar
        count={count}
        status={status}
        onReset={handleReset}
        onUploadJump={handleScrollToUpload}
      />
    </div>
  );
}

function MobileQuickBar({ count, status, onReset, onUploadJump }) {
  const canReset = status === STATUS.LIVE;

  return (
    <div className="fixed inset-x-3 bottom-3 z-30 sm:hidden">
      <div className="panel-metal flex items-center gap-2 rounded-xl border border-slate-700/70 px-2 py-2 backdrop-blur-md">
        <div className="min-w-0 flex-1 rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-2 py-1.5">
          <p className="text-[9px] uppercase tracking-[0.12em] text-slate-400">Boxes</p>
          <p className="tech-title truncate text-sm font-semibold text-cyan-100">{count}</p>
        </div>

        <button
          onClick={onUploadJump}
          className="rounded-lg border border-slate-600 bg-slate-900/70 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-slate-200"
        >
          Upload
        </button>

        <button
          onClick={onReset}
          disabled={!canReset}
          className="rounded-lg border border-cyan-500/45 bg-cyan-500/10 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-cyan-100 disabled:opacity-40"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
