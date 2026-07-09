//! Storage backend for the player-in-worker path.
//!
//! A Web Worker can't touch `localStorage`, so this mirrors the main thread's
//! [`LocalStorageBackend`](crate::storage): reads are served from a startup
//! snapshot handed over in [`WorkerInit`](crate::worker_player::WorkerInit), and
//! writes are pushed over the [`WorkerBridge`] for the main thread to persist
//! (fire-and-forget — no blocking round-trip). Base64 encoding matches
//! `LocalStorageBackend` so values are interchangeable.

use crate::worker_bridge::WorkerBridge;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use ruffle_core::backend::storage::StorageBackend;
use std::collections::HashMap;
use std::sync::Arc;

pub struct WebWorkerStorageBackend {
    /// `name -> base64 value`, mirroring `localStorage`. Seeded at startup and
    /// kept current with our own writes, so read-after-write works with no RPC.
    data: HashMap<String, String>,
    bridge: Arc<WorkerBridge>,
}

impl WebWorkerStorageBackend {
    pub fn new(seed: HashMap<String, String>, bridge: Arc<WorkerBridge>) -> Self {
        Self { data: seed, bridge }
    }
}

impl StorageBackend for WebWorkerStorageBackend {
    fn get(&self, name: &str) -> Option<Vec<u8>> {
        self.data
            .get(name)
            .and_then(|b64| BASE64_STANDARD.decode(b64).ok())
    }

    fn put(&mut self, name: &str, value: &[u8]) -> bool {
        let b64 = BASE64_STANDARD.encode(value);
        self.data.insert(name.to_owned(), b64.clone());
        self.bridge.push_storage_write(name.to_owned(), Some(b64));
        true
    }

    fn remove_key(&mut self, name: &str) {
        self.data.remove(name);
        self.bridge.push_storage_write(name.to_owned(), None);
    }
}
