export function LowerDeck({ modelPending, liveCount, liveEvents }) {
  return (
    <section className="grid grid-cols-1 gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="grid grid-cols-1 gap-4">
        <SessionHistoryPanel modelPending={modelPending} liveCount={liveCount} />
        <EventTimelinePanel modelPending={modelPending} liveEvents={liveEvents} />
      </div>
      <div className="grid grid-cols-1 gap-4">
        <StateLegend modelPending={modelPending} />
        <OperatorNotes modelPending={modelPending} />
      </div>
    </section>
  );
}

function SessionHistoryPanel({ modelPending, liveCount }) {
  const rows = modelPending
    ? [
        { id: "DEMO-A1", boxes: 14, duration: "01:42", status: "Preview" },
        { id: "DEMO-A2", boxes: 18, duration: "02:19", status: "Preview" },
        { id: "DEMO-A3", boxes: 11, duration: "01:27", status: "Preview" },
      ]
    : [
        { id: "LIVE-001", boxes: liveCount, duration: "Running", status: "Live" },
      ];

  return (
    <div className="panel-metal panel-reveal rounded-xl p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Session History</span>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">{modelPending ? "Mock Data" : "Live Data"}</span>
      </div>
      <div className="mt-3 overflow-hidden rounded-md border border-slate-700/70 bg-slate-950/60">
        <div className="grid grid-cols-[1.3fr_0.8fr_0.9fr_0.8fr] border-b border-slate-700/60 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-slate-500">
          <span>Session</span>
          <span>Boxes</span>
          <span>Duration</span>
          <span>Mode</span>
        </div>
        <div className="divide-y divide-slate-800/80">
          {rows.map((row) => (
            <div key={row.id} className="grid grid-cols-[1.3fr_0.8fr_0.9fr_0.8fr] px-3 py-2 text-[11px]">
              <span className="text-slate-300">{row.id}</span>
              <span className="tabular-nums text-cyan-200">{row.boxes}</span>
              <span className="text-slate-400">{row.duration}</span>
              <span className={row.status === "Preview" ? "text-amber-300" : "text-emerald-300"}>{row.status}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StateLegend({ modelPending }) {
  const states = [
    { color: "bg-green-400", label: "CONFIRMED_INSIDE", desc: "Counted and stable" },
    { color: "bg-amber-400", label: "PENDING_ENTER", desc: "Debouncing (5 frames)" },
    { color: "bg-blue-400", label: "HIDDEN_INSIDE", desc: "Occluded or stacked" },
    { color: "bg-slate-500", label: "REMOVED", desc: "Left the carton" },
  ];

  return (
    <div className="panel-metal panel-reveal rounded-xl p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Track State Legend</span>
        {modelPending && (
          <span className="text-[10px] uppercase tracking-[0.11em] text-amber-300">Visual Guide Only</span>
        )}
      </div>
      <ul className="mt-3 flex flex-col gap-2">
        {states.map((entry) => (
          <li key={entry.label} className="flex items-center gap-3 rounded-md border border-slate-700/70 bg-slate-950/60 px-3 py-2">
            <span className={`h-2.5 w-2.5 flex-shrink-0 rounded-full ${entry.color}`} />
            <div className="leading-tight">
              <p className="tech-title text-[11px] font-semibold tracking-[0.08em] text-slate-200">{entry.label}</p>
              <p className="text-[11px] text-slate-500">{entry.desc}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function OperatorNotes({ modelPending }) {
  return (
    <div className="panel-metal panel-reveal rounded-xl p-4">
      <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Operator Notes</span>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        {modelPending
          ? "Preview mode active. Use this panel to define shift note format before backend persistence is wired."
          : "Live mode active. Track shift comments, carton batch tags, and incident remarks."}
      </p>
      <div className="mt-3 rounded-md border border-slate-700/70 bg-slate-950/60 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-slate-500">
        Notes API wiring in next checkpoint
      </div>
    </div>
  );
}

function EventTimelinePanel({ modelPending, liveEvents }) {
  const mockEvents = [
    { id: "mock-1", type: "ADDED", trackId: "12", at: "12:04:11" },
    { id: "mock-2", type: "HIDDEN", trackId: "12", at: "12:04:19" },
    { id: "mock-3", type: "REMOVED", trackId: "08", at: "12:04:33" },
  ];

  const rows = modelPending || !Array.isArray(liveEvents) || liveEvents.length === 0
    ? mockEvents
    : liveEvents;

  return (
    <div className="panel-metal panel-reveal rounded-xl p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Event Timeline</span>
        <span className="text-[10px] uppercase tracking-[0.12em] text-slate-500">{modelPending ? "Mock" : "Live"}</span>
      </div>

      <div className="mt-3 overflow-hidden rounded-md border border-slate-700/70 bg-slate-950/60">
        <div className="grid grid-cols-[1fr_0.9fr_0.9fr] border-b border-slate-700/60 px-3 py-2 text-[10px] uppercase tracking-[0.11em] text-slate-500">
          <span>Event</span>
          <span>Track</span>
          <span>Time</span>
        </div>

        <div className="divide-y divide-slate-800/80">
          {rows.slice(0, 6).map((row) => (
            <div key={row.id} className="grid grid-cols-[1fr_0.9fr_0.9fr] px-3 py-2 text-[11px]">
              <span className={eventTone(row.type)}>{row.type}</span>
              <span className="tabular-nums text-slate-300">#{row.trackId}</span>
              <span className="text-slate-400">{row.at}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function eventTone(type) {
  if (type === "ADDED") return "text-emerald-300";
  if (type === "HIDDEN") return "text-amber-300";
  if (type === "REMOVED") return "text-rose-300";
  return "text-cyan-200";
}
