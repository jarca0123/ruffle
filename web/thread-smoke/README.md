# wasm-threads smoke test

De-risks the infrastructure for Ruffle "Path A" workers: a threaded wasm build
with shared memory, a second thread running Rust on a Web Worker, and
cross-thread `Atomics.wait`/`notify` on shared memory.

## Run

```sh
./build.sh      # threaded cargo build + wasm-bindgen -> pkg/
./serve.py      # static server with COOP/COEP headers (localhost:8080)
```

Open <http://localhost:8080/> and look at the devtools **console**. Success:

```
crossOriginIsolated = true
[main]   start(); spawning worker thread
[worker] started; blocking on shared cell (Atomics.wait)
[main]   stored 42 and notified worker (woke 1)
[worker] woke up; shared cell = 42
```

The last line means a Web Worker thread parked in `Atomics.wait` on shared
linear memory and was woken by the main thread writing + `Atomics.notify` — the
exact primitive a shareable `ByteArray` / `Mutex` / `Condition` will use across
Flash workers.

## Requirements

- Nightly Rust with `rust-src` (`-Z build-std`), `wasm32-unknown-unknown`.
- `wasm-bindgen` CLI **matching** the crate version (here `=0.2.120`).
- A browser context that is **cross-origin isolated** (`serve.py` sets the
  `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
  require-corp` headers). Without these, `SharedArrayBuffer` is unavailable and
  `crossOriginIsolated` is `false`.

## What this proves for Ruffle

- `ruffle_web` already compiles + links threaded (separate check).
- `wasm_thread` packages the fiddly wasm-bindgen worker/TLS/stack bootstrap, so
  spawning a "worker runtime" thread is `wasm_thread::spawn(...)`.
- Shared memory + real atomics work across threads → shared `ByteArray` and
  `Mutex`/`Condition` need no Asyncify and no cooperative-scheduler gymnastics.
