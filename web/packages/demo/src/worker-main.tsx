// React entry for the *player-in-worker* demo — the same frontend as the normal
// demo (`main.tsx`), but the movie runs on a worker (browser OpenTTD).
//
// No `StrictMode`: it double-invokes effects in dev, and the player's canvas can
// only be `transferControlToOffscreen`'d once.

import ReactDOM from "react-dom/client";
import "./common.css";
import "./lato.css";
import "./index.css";
import { WorkerApp } from "./WorkerApp.tsx";

ReactDOM.createRoot(document.getElementById("root")!).render(
    <WorkerApp allowSampleSwfs={true} allowUrlLoading={false} />,
);
