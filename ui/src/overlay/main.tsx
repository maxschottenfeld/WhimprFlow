import React from "react";
import ReactDOM from "react-dom/client";
import { FlowBar } from "./FlowBar";

// The overlay window is transparent; keep the document background clear so only
// the pill paints. (Global reset lives here rather than a CSS file to keep the
// always-resident overlay bundle minimal.)
const style = document.createElement("style");
style.textContent = `
  html, body, #root { margin: 0; height: 100%; background: transparent; }
  * { box-sizing: border-box; }

  /* Pop-in for the pill arriving out of nothing (hotkey pressed while the idle
     pill is off). Overshoots slightly past 1 so it reads as a pop rather than a
     fade. Scale only -- no layout properties -- so it composites on the GPU and
     doesn't contend with the width/height morph. */
  @keyframes whimpr-pop-in {
    0%   { transform: scale(0.72); opacity: 0; }
    62%  { transform: scale(1.04); opacity: 1; }
    100% { transform: scale(1);    opacity: 1; }
  }
`;
document.head.appendChild(style);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <FlowBar />
  </React.StrictMode>,
);
