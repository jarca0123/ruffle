import { useEffect, useRef } from "react";
import { startWorkerPlayer } from "ruffle-core/dist/start-worker-player";

type Handle = Awaited<ReturnType<typeof startWorkerPlayer>>;

/**
 * Mounts a canvas (styled as the demo's `#player`) and runs `swfUrl` on a worker.
 *
 * `transferControlToOffscreen` may only be called once per canvas, so callers
 * must give this a React `key` tied to `swfUrl` (a URL change remounts it with a
 * fresh canvas).
 */
export function WorkerPlayerView({ swfUrl }: { swfUrl: string }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }

        let handle: Handle | undefined;
        let disposed = false;

        // Size the backing store to the CSS size before control is transferred.
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));

        startWorkerPlayer(canvas, swfUrl)
            .then((h) => {
                if (disposed) {
                    h.destroy();
                    return;
                }
                handle = h;
                (window as unknown as { ruffleWorker: Handle }).ruffleWorker = h;
            })
            .catch((e) => console.error("player-in-worker failed to start:", e));

        const onResize = () =>
            handle?.set_viewport(canvas.clientWidth, canvas.clientHeight);
        window.addEventListener("resize", onResize);

        return () => {
            disposed = true;
            window.removeEventListener("resize", onResize);
            handle?.destroy();
        };
    }, [swfUrl]);

    return <canvas id="player" ref={canvasRef} />;
}
