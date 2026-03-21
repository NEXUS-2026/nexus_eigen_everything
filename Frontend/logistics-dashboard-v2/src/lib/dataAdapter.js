export function normalizeWsPayload(raw) {
  if (!raw || typeof raw !== "object") {
    return {
      frame: null,
      count: 0,
      fps: 0,
      infMs: 0,
      events: [],
      modelReady: false,
    };
  }

  const frame = typeof raw.frame === "string" ? raw.frame : null;
  const count = toNumber(raw.count, 0);
  const fps = toNumber(raw.fps, 0);
  const infMs = toNumber(raw.inf_ms ?? raw.infMs ?? raw.inference_ms, 0);
  const modelReady = Boolean(raw.model_ready ?? raw.modelReady ?? infMs > 0.05);

  const eventsRaw = Array.isArray(raw.events) ? raw.events : [];
  const events = eventsRaw
    .map((item, idx) => normalizeEvent(item, idx))
    .filter(Boolean);

  return {
    frame,
    count,
    fps,
    infMs,
    events,
    modelReady,
  };
}

function normalizeEvent(item, idx) {
  if (typeof item === "string") {
    return {
      id: `evt-${idx}-${item}`,
      type: item,
      trackId: "-",
      at: new Date().toLocaleTimeString(),
    };
  }

  if (!item || typeof item !== "object") {
    return null;
  }

  const type = String(item.type ?? item.event ?? "EVENT").toUpperCase();
  const trackId = item.track_id ?? item.trackId ?? item.id ?? "-";
  const at = typeof item.at === "string" ? item.at : new Date().toLocaleTimeString();

  return {
    id: `${type}-${trackId}-${idx}`,
    type,
    trackId: String(trackId),
    at,
  };
}

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
