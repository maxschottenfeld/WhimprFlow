import { useEffect, useRef, useState } from "react";
import { palette, pillFill, geometry, font, motion } from "../tokens/values";

// Visual states, mirroring the Rust `BarState`.
export type BarState =
  | "idle"
  | "recording"
  | "locked"
  | "transcribing"
  | "done"
  | "cancelled"
  | "error";

type StateEvent = { state: BarState };
type WaveformEvent = { bars: number[] };

type SettingsEvent = { show_idle_pill: boolean };

async function tauriListen<T>(event: string, cb: (payload: T) => void): Promise<() => void> {
  try {
    const { listen } = await import("@tauri-apps/api/event");
    return await listen<T>(event, (e) => cb(e.payload as T));
  } catch {
    return () => {};
  }
}

// Read the one setting the overlay needs. Outside the Tauri shell (vite preview)
// the import throws, so fall back to showing the pill — the old behavior.
async function readShowIdlePill(): Promise<boolean> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const settings = await invoke<{ show_idle_pill?: boolean }>("get_settings");
    return settings.show_idle_pill !== false;
  } catch {
    return true;
  }
}

// A row of dot-like rounded bars driven by mic RMS — Wispr's dotted-waveform look:
// small dots when quiet, rising into a waveform when speaking.
function DottedWaveform({ bars }: { bars: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const barsRef = useRef<number[]>(bars);
  barsRef.current = bars;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const N = 16;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const dotW = 2.4;
      const gap = (w - N * dotW) / (N - 1);
      const t = performance.now();
      ctx.fillStyle = palette.waveBar;
      for (let i = 0; i < N; i++) {
        const real = barsRef.current[barsRef.current.length - 1 - (i % barsRef.current.length)];
        // Idle shimmer so the dotted line reads as "listening" even in near-silence.
        const shimmer = 0.12 + 0.06 * Math.abs(Math.sin(t / 260 + i * 0.7));
        const amp = Math.max(shimmer, real ?? 0);
        const bh = 3 + amp * 20; // 3px dot → up to ~23px bar
        const x = i * (dotW + gap);
        const y = (h - bh) / 2;
        ctx.beginPath();
        ctx.roundRect(x, y, dotW, bh, dotW / 2);
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={canvasRef} style={{ width: "100%", height: 28 }} />;
}

function CancelButton() {
  return (
    <div
      title="Cancel (Esc)"
      style={{
        flex: "0 0 auto",
        width: 26,
        height: 26,
        borderRadius: 9999,
        background: "rgba(255,255,255,0.16)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
        fontSize: 15,
        lineHeight: 1,
      }}
    >
      ✕
    </div>
  );
}

function StopButton() {
  return (
    <div
      title="Stop"
      style={{
        flex: "0 0 auto",
        width: 26,
        height: 26,
        borderRadius: 9999,
        background: "#FF5A52",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div style={{ width: 9, height: 9, borderRadius: 2, background: "#fff" }} />
    </div>
  );
}

export function FlowBar() {
  const [state, setState] = useState<BarState>("idle");
  const [bars, setBars] = useState<number[]>([]);
  // `null` = not read yet. The idle pill stays hidden until the answer arrives so
  // someone who turned it off doesn't get a flash of it at every app launch.
  const [showIdlePill, setShowIdlePill] = useState<boolean | null>(null);

  useEffect(() => {
    let un1: (() => void) | undefined;
    let un2: (() => void) | undefined;
    let un3: (() => void) | undefined;
    tauriListen<StateEvent>("whimpr://flowbar/state", (p) => setState(p.state)).then((u) => (un1 = u));
    tauriListen<WaveformEvent>("whimpr://audio/waveform", (p) => setBars(p.bars)).then((u) => (un2 = u));
    // The Hub pushes this on every save, so the toggle applies without a restart.
    tauriListen<SettingsEvent>("whimpr://settings/changed", (p) =>
      setShowIdlePill(p.show_idle_pill !== false),
    ).then((u) => (un3 = u));
    readShowIdlePill().then(setShowIdlePill);
    return () => {
      un1?.();
      un2?.();
      un3?.();
    };
  }, []);

  const recording = state === "recording" || state === "locked";
  const isIdle = state === "idle";
  // Idle is the only state the toggle governs: recording, transcribing and the
  // done/cancelled/error flashes are feedback for something the user just did.
  const hidden = isIdle && showIdlePill !== true;

  // Replay the pop-in every time the pill goes from nothing to something. Keyed
  // on the hidden->visible edge rather than on "recording" so it fires once per
  // dictation: the pill stays mounted through transcribing and done, and only
  // the next hotkey press brings it back from nothing.
  const wasHidden = useRef(hidden);
  const [popCount, setPopCount] = useState(0);
  useEffect(() => {
    if (wasHidden.current && !hidden) setPopCount((n) => n + 1);
    wasHidden.current = hidden;
  }, [hidden]);
  const processing = state === "transcribing";
  const statusText =
    state === "transcribing"
      ? "Cleaning up…"
      : state === "error"
        ? "Something's off"
        : state === "cancelled"
          ? "Discarded"
          : "Done";

  // Pill dimensions per state.
  const dims = isIdle
    ? { w: 76, h: 16 }
    : recording
      ? { w: 250, h: 44 }
      : { w: 180, h: 36 };

  if (hidden) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: font.ui,
        userSelect: "none",
      }}
    >
      <div
        // Remount on each pop so the animation restarts; without a changing key
        // the browser sees the same running animation and does nothing.
        key={popCount}
        aria-label={`WhimprFlow ${state}`}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: recording ? "space-between" : "center",
          gap: 10,
          height: dims.h,
          width: dims.w,
          padding: recording ? "0 8px" : 0,
          background: pillFill.base,
          border: `1px solid rgba(255,255,255,0.10)`,
          borderRadius: 9999,
          boxShadow: pillFill.shadow,
          color: palette.pillText,
          transition: `width ${geometry.morphMs}ms ${motionEase}, height ${geometry.morphMs}ms ${motionEase}`,
          // popCount 0 is the app's own launch -- don't pop the pill at startup.
          animation: popCount > 0 ? `whimpr-pop-in ${motion.springDurationS}s ${popEase} both` : undefined,
          overflow: "hidden",
          fontSize: 13,
        }}
      >
        {isIdle ? null : recording ? (
          <>
            <CancelButton />
            <div style={{ flex: 1, minWidth: 0 }}>
              <DottedWaveform bars={bars} />
            </div>
            <StopButton />
          </>
        ) : processing ? (
          <span style={{ color: palette.pillTextMuted }}>{statusText}</span>
        ) : (
          <span style={{ color: palette.pillTextMuted }}>{statusText}</span>
        )}
      </div>
    </div>
  );
}

const motionEase = "cubic-bezier(0.05,0.6,0.4,0.95)";
// Back-out: undershoots at the start, overshoots near the end. `motion.ease` is
// the morph curve and does neither, which is why this is its own value.
const popEase = "cubic-bezier(0.34, 1.4, 0.5, 1)";
