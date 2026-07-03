#!/usr/bin/env bash
# Build the threaded wasm + wasm-bindgen (--target web) glue into ./pkg.
#
# The link-args are the non-obvious part of a wasm-threads build:
#   --shared-memory --import-memory --max-memory  -> shared SharedArrayBuffer
#   --export=__heap_base                          -> wasm-bindgen injects thread ids here
#   --export=__wasm_init_tls / __tls_*            -> per-thread TLS setup (else GC'd, bindgen fails)
set -euo pipefail
cd "$(dirname "$0")"

RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=2147483648 -C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_base -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align' \
  cargo build --release --target wasm32-unknown-unknown \
  -Z build-std=std,panic_abort

wasm-bindgen ./target/wasm32-unknown-unknown/release/thread_smoke.wasm \
  --out-dir pkg --target web

echo "OK -> pkg/   now run: ./serve.py   and open http://localhost:8080/"
