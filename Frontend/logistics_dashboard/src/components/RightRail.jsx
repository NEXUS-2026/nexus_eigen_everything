import { useCallback, useState } from "react";

export function RightRail({ count, fps, infMs, status, modelPending, onReset, onUpload, isUploading, uploadMessage, uploadError }) {
  const isLive = status === "Live";

  return (
    <aside className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-1">
      <EngineModeCard modelPending={modelPending} />
      <CountHero count={count} />
      <StatusBoard status={status} fps={fps} infMs={infMs} />
      <ResetButton onReset={onReset} disabled={!isLive} />
      <VideoUploadPanel
        onUpload={onUpload}
        isUploading={isUploading}
        message={uploadMessage}
        error={uploadError}
      />
    </aside>
  );
}

function EngineModeCard({ modelPending }) {
  const tone = modelPending
    ? "border-amber-300/35 bg-amber-500/10 text-amber-200"
    : "border-emerald-300/35 bg-emerald-500/10 text-emerald-200";

  return (
    <div className={`panel-metal panel-reveal rounded-xl border p-3 ${tone}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em]">Detection Engine</span>
        <span className="rounded border border-current/40 bg-black/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]">
          {modelPending ? "Preview Mode" : "Live Model"}
        </span>
      </div>
      <p className="mt-1 text-[11px]">
        {modelPending
          ? "Model integration pending. Dashboard is running on stream-only state."
          : "Detector is publishing live inference metrics and count signals."}
      </p>
    </div>
  );
}

function CountHero({ count }) {
  return (
    <div className="panel-metal panel-reveal rounded-xl border border-cyan-300/30 p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Total Boxes in Carton</p>
      <div
        className={`tech-title mt-2 text-7xl font-semibold leading-none tabular-nums transition-colors duration-300 md:text-8xl ${
          count > 0 ? "text-cyan-200" : "text-slate-600"
        }`}
      >
        {count}
      </div>
      <p className="mt-2 text-[11px] uppercase tracking-[0.12em] text-slate-500">CONFIRMED_INSIDE + HIDDEN_INSIDE tracks</p>
    </div>
  );
}

function StatusBoard({ status, fps, infMs }) {
  const style =
    status === "Live"
      ? { text: "text-emerald-300", ring: "border-emerald-300/40", dot: "bg-emerald-300 status-scan" }
      : status === "Connecting"
        ? { text: "text-amber-300", ring: "border-amber-300/40", dot: "bg-amber-300 status-scan" }
        : { text: "text-rose-300", ring: "border-rose-300/40", dot: "bg-rose-300" };

  return (
    <div className="panel-metal panel-reveal rounded-xl p-4">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">System Status Board</p>
      <div className={`mt-3 flex items-center justify-between rounded-lg border bg-slate-950/70 px-3 py-2 ${style.ring}`}>
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${style.dot}`} />
          <span className={`tech-title text-sm font-semibold uppercase tracking-[0.14em] ${style.text}`}>{status}</span>
        </div>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Link Active</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-slate-700/60 bg-slate-950/70 p-3">
          <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">Capture FPS</p>
          <p className="tech-title text-xl font-semibold tabular-nums text-cyan-300">{fps.toFixed(1)}</p>
        </div>
        <div className="rounded-lg border border-slate-700/60 bg-slate-950/70 p-3">
          <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">YOLO Latency</p>
          <p className="tech-title text-xl font-semibold tabular-nums text-orange-300">
            {infMs.toFixed(0)}
            <span className="ml-1 text-xs font-normal uppercase text-slate-500">ms</span>
          </p>
        </div>
      </div>
    </div>
  );
}

function ResetButton({ onReset, disabled }) {
  const [confirming, setConfirming] = useState(false);

  const handleClick = useCallback(() => {
    if (!confirming) {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 3000);
      return;
    }
    setConfirming(false);
    onReset();
  }, [confirming, onReset]);

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`w-full rounded-2xl border px-5 py-4 text-sm font-semibold uppercase tracking-[0.14em] tech-title transition-all duration-200 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 ${
        confirming
          ? "border-rose-400 bg-rose-500/20 text-rose-200 hover:bg-rose-500/30"
          : "border-cyan-500/40 bg-cyan-500/10 text-cyan-100 hover:border-cyan-300/80 hover:bg-cyan-500/20"
      }`}
    >
      {confirming ? "Confirm Reset Session" : "Reset Session"}
    </button>
  );
}

function VideoUploadPanel({ onUpload, isUploading, message, error }) {
  return (
    <div id="upload-panel" className="panel-metal panel-reveal rounded-xl p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Test Video Input</span>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Browser Upload</span>
      </div>

      <label className="mt-3 block cursor-pointer">
        <input type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm" className="hidden" onChange={onUpload} disabled={isUploading} />
        <span
          className={`inline-flex w-full items-center justify-center rounded-lg border px-4 py-3 text-[11px] uppercase tracking-[0.13em] tech-title ${
            isUploading
              ? "border-slate-600 bg-slate-800/70 text-slate-400"
              : "border-cyan-400/40 bg-cyan-500/10 text-cyan-100 hover:bg-cyan-500/20"
          }`}
        >
          {isUploading ? "Uploading Video..." : "Upload Test Video"}
        </span>
      </label>

      <p className="mt-2 text-[10px] uppercase tracking-[0.1em] text-slate-500">Supported: mp4, avi, mov, mkv, webm</p>

      {message && <div className="mt-2 rounded-md border border-emerald-300/30 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-200">{message}</div>}
      {error && <div className="mt-2 rounded-md border border-rose-300/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-200">{error}</div>}
    </div>
  );
}

