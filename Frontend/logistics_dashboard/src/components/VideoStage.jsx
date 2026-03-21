export function VideoStage({
  canvasRef,
  status,
  isLive,
  alertText,
  alertTone,
  playback,
  onTogglePause,
  onSeekBy,
  onSeekTo,
  onClearVideo,
}) {
  const fileMode = Boolean(playback?.isFileSource);
  const sliderMax = playback?.durationSec > 0 ? playback.durationSec : 1;
  const sliderValue = Math.min(playback?.positionSec ?? 0, sliderMax);

  return (
    <section className="flex flex-col gap-3">
      <div className={`panel-metal rounded-xl border px-4 py-2.5 text-[11px] uppercase tracking-[0.12em] ${alertTone}`}>
        {alertText}
      </div>

      <div className="panel-metal panel-reveal rounded-2xl p-3 sm:p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-700/70 bg-slate-950/75 px-3 py-2">
          <div>
            <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">Live Feed</p>
            <p className="tech-title text-sm uppercase tracking-[0.12em] text-cyan-100">Carton Camera Channel</p>
          </div>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.1em]">
            <span className="rounded border border-cyan-400/30 bg-cyan-400/10 px-2 py-1 text-cyan-200">Source WS</span>
            <span className="rounded border border-slate-600 px-2 py-1 text-slate-300">JPEG</span>
            <span className="rounded border border-slate-600 px-2 py-1 text-slate-300">40 FPS target</span>
          </div>
        </div>

        <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-cyan-400/25 bg-black shadow-2xl shadow-black/60">
          <canvas ref={canvasRef} width={640} height={480} className="h-full w-full object-contain" aria-label="Live camera feed" />

          {!isLive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/82 backdrop-blur-sm">
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-slate-600 border-t-cyan-300" />
              <p className="tech-title text-xs uppercase tracking-[0.15em] text-slate-300 md:text-sm">
                {status === "Connecting" ? "Connecting to feed..." : "Feed disconnected - reconnecting..."}
              </p>
            </div>
          )}

          {isLive && (
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-emerald-300/35 bg-slate-950/70 px-3 py-1 backdrop-blur-sm">
              <span className="status-scan h-2 w-2 rounded-full bg-emerald-300" />
              <span className="tech-title text-[11px] font-semibold uppercase tracking-[0.12em] text-emerald-300">Live</span>
            </div>
          )}
        </div>
      </div>

      <div className="panel-metal panel-reveal rounded-xl px-3 py-3">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.1em]">
          <span className="rounded border border-slate-700 px-2 py-1 text-slate-400">Transport Dock</span>
          <span className="rounded border border-slate-700 px-2 py-1 text-slate-400">No Crop Mode</span>
          <span className={fileMode ? "rounded border border-cyan-400/45 bg-cyan-500/10 px-2 py-1 text-cyan-200" : "rounded border border-slate-700 px-2 py-1 text-slate-500"}>
            {fileMode ? "Uploaded Clip" : "Waiting Upload"}
          </span>
        </div>

        <input
          type="range"
          min={0}
          max={sliderMax}
          step={0.1}
          value={sliderValue}
          disabled={!fileMode}
          onChange={(event) => onSeekTo(Number(event.target.value))}
          className="w-full accent-cyan-300 disabled:opacity-45"
        />

        <div className="mt-1 flex items-center justify-between text-[10px] text-slate-500">
          <span>{formatSeconds(playback?.positionSec ?? 0)}</span>
          <span>{formatSeconds(playback?.durationSec ?? 0)}</span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.1em]">
          <button
            onClick={() => onSeekBy(-5)}
            disabled={!fileMode}
            className="rounded border border-slate-600 px-2.5 py-1.5 text-slate-200 disabled:opacity-35"
          >
            -5s
          </button>
          <button
            onClick={onTogglePause}
            disabled={!fileMode}
            className="rounded border border-cyan-500/45 bg-cyan-500/10 px-2.5 py-1.5 text-cyan-100 disabled:opacity-35"
          >
            {playback?.paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={() => onSeekBy(5)}
            disabled={!fileMode}
            className="rounded border border-slate-600 px-2.5 py-1.5 text-slate-200 disabled:opacity-35"
          >
            +5s
          </button>
          <button
            onClick={onClearVideo}
            disabled={!fileMode}
            className="ml-auto rounded border border-rose-400/45 bg-rose-500/10 px-2.5 py-1.5 text-rose-200 disabled:opacity-35"
          >
            Clear Current Video
          </button>
        </div>
      </div>
    </section>
  );
}

function formatSeconds(value) {
  const total = Math.max(0, Math.floor(value || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
