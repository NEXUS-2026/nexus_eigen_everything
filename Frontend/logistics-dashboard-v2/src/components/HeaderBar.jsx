import { useEffect, useState } from "react";

export function HeaderBar({ statusText }) {
  return (
    <header className="panel-metal sticky top-0 z-20 px-4 py-3 sm:px-6 sm:py-4 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-[1500px] items-center justify-between gap-3">
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-400/10 text-cyan-100">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"
              />
            </svg>
          </div>
          <div>
            <h1 className="tech-title text-lg font-semibold uppercase tracking-[0.12em] text-cyan-50 sm:text-2xl">
              Warehouse Box Counter
            </h1>
            <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400 sm:text-xs">
              Industrial Monitoring Console
            </p>
          </div>
        </div>

        <div className="hidden items-center gap-4 sm:flex">
          <div className="rounded-full border border-cyan-400/35 bg-cyan-500/10 px-3 py-1 text-[10px] uppercase tracking-[0.14em] text-cyan-100">
            {statusText}
          </div>
          <LiveClock />
        </div>
      </div>
    </header>
  );
}

function LiveClock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="text-right">
      <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">Local time</p>
      <p className="tech-title text-sm font-semibold tabular-nums text-cyan-100">{time}</p>
    </div>
  );
}
