import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Ruffle's own `.wasm` module and the AVM2 JIT's runtime-generated modules
// (`WebAssembly.Module(bytes)`) both require `'wasm-unsafe-eval'` in a strict
// CSP. `'unsafe-eval'`/`'unsafe-inline'` keep Vite's dev HMR working; a
// production embed only needs `'self'` + `'wasm-unsafe-eval'`. If the JIT's CSP
// requirement is not met it declines gracefully and the interpreter runs.
const contentSecurityPolicy = [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "connect-src *",
    "media-src *",
    "img-src 'self' data:",
    "worker-src 'self' blob:",
].join("; ");

// The threaded (shared-memory) build — real Flash workers and the
// player-in-worker path — needs a *cross-origin-isolated* page so
// `SharedArrayBuffer` / a shared `WebAssembly.Memory` may be created and
// transferred to workers. That requires these two headers on every response.
const headers = {
    "Content-Security-Policy": contentSecurityPolicy,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: "",
    // Multi-page: the normal demo plus the player-in-worker demo. Both HTML files
    // must be listed or Vite drops all but the default `index.html` in the build.
    build: {
        rollupOptions: {
            input: {
                main: "index.html",
                worker: "worker.html",
            },
        },
    },
    // `server` covers `vite dev`; `preview` covers `vite preview` (the built
    // demo). Both need the isolation headers.
    server: { headers },
    preview: { headers },
});
