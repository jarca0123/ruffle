//! Navigator backend for the player-in-worker path.
//!
//! Unlike audio/storage/UI, networking needs **no** main-thread bridge: `fetch`
//! is available on a `WorkerGlobalScope`, so the primordial worker fetches
//! directly. This is a trimmed [`WebNavigatorBackend`](crate::navigator) — no
//! `window()` (a worker has none), no player-lock re-queue in `spawn_future`
//! (the worker tick loop never yields to the JS event loop mid-tick, so a
//! spawned future only runs *between* ticks, with the player unlocked).

use crate::navigator::WebResponseWrapper;
use js_sys::{Array, Uint8Array};
use ruffle_core::backend::navigator::{
    ErrorResponse, NavigationMethod, NavigatorBackend, OwnedFuture, Request,
    SuccessResponse, async_return, create_fetch_error, create_specific_fetch_error,
};
use ruffle_core::loader::Error;
use ruffle_core::socket::{ConnectionState, SocketAction, SocketHandle};
use async_channel::{Receiver, Sender};
use ruffle_core::indexmap::IndexMap;
use std::time::Duration;
use url::{ParseError, Url};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    Blob, BlobPropertyBag, Request as WebRequest, RequestCredentials, RequestInit,
    Response as WebResponse, WorkerGlobalScope,
};

/// A `NavigatorBackend` for the worker player: real `fetch`, stubs for the
/// window-only operations (`navigateToURL` opens a page; a worker can't).
pub struct WebWorkerNavigatorBackend {
    /// Base for resolving relative URLs (the movie's absolute URL).
    base_url: Option<Url>,
    /// Hosts (as `scheme://host`) for which we send `credentials: "include"` so
    /// session cookies ride along. Mirrors the main-thread backend's
    /// `credential_allow_list`: everything else fetches `SameOrigin`, because a
    /// credentialed cross-origin fetch to a server that doesn't do credentialed
    /// CORS (no `Access-Control-Allow-Credentials`, or a wildcard origin) is
    /// rejected outright — which is what surfaced as `FetchError("Got JS error")`
    /// on pr3hub. Empty by default, matching the config default.
    credential_allow_list: Vec<String>,
    /// Rewrite `http://` → `https://` when the document is served over HTTPS.
    /// Mandatory here: the browser blocks *any* `http://` fetch from an HTTPS page
    /// as mixed active content (it never reaches the network — it just rejects as
    /// `FetchError("Got JS error")`), so games that hardcode `http://` API URLs
    /// only work once we upgrade the scheme. Mirrors the main-thread backend's
    /// `upgrade_to_https` (config `upgradeToHttps`, default on).
    upgrade_to_https: bool,
}

impl WebWorkerNavigatorBackend {
    pub fn new(base_url: Option<String>, credential_allow_list: Vec<String>) -> Self {
        let base_url = base_url.and_then(|mut b| {
            if !b.ends_with('/') {
                b.push('/');
            }
            Url::parse(&b).ok()
        });
        // The worker has no `window`, but `WorkerGlobalScope.location` exposes the
        // document's protocol. Only upgrade when actually on HTTPS (upgrading on a
        // plain-HTTP/localhost page would break those setups). The config half is
        // hardcoded `true`, matching `upgradeToHttps`'s default.
        let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
        let upgrade_to_https = scope.location().protocol() == "https:";
        Self {
            base_url,
            credential_allow_list,
            upgrade_to_https,
        }
    }
}

impl NavigatorBackend for WebWorkerNavigatorBackend {
    fn navigate_to_url(
        &self,
        url: &str,
        _target: &str,
        _vars_method: Option<(NavigationMethod, IndexMap<String, String>)>,
    ) {
        // A worker has no `window` to `open`; page navigation would need a
        // main-thread bridge. Log so it's visible rather than silently dropped.
        tracing::warn!("worker navigator: ignoring navigateToURL({url}) — no window on a worker");
    }

    fn fetch(&self, request: Request) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        let url = match self.resolve_url(request.url()) {
            Ok(url) => url,
            Err(e) => return async_return(Err(create_fetch_error(request.url(), e))),
        };

        let credentials = if let Some(host) = url.host_str() {
            if self
                .credential_allow_list
                .iter()
                .any(|allowed| allowed == &format!("{}://{}", url.scheme(), host))
            {
                RequestCredentials::Include
            } else {
                RequestCredentials::SameOrigin
            }
        } else {
            RequestCredentials::SameOrigin
        };

        Box::pin(async move {
            let init = RequestInit::new();
            init.set_method(&request.method().to_string());
            init.set_credentials(credentials);

            if let Some((data, mime)) = request.body() {
                let options = BlobPropertyBag::new();
                options.set_type(mime);
                let blob = Blob::new_with_buffer_source_sequence_and_options(
                    &Array::from_iter([Uint8Array::from(data.as_slice()).buffer()]),
                    &options,
                )
                .map_err(|_| ErrorResponse {
                    url: url.to_string(),
                    error: Error::FetchError("Got JS error".to_string()),
                })?
                .dyn_into()
                .map_err(|_| ErrorResponse {
                    url: url.to_string(),
                    error: Error::FetchError("Got JS error".to_string()),
                })?;
                init.set_body(&blob);
            }

            let web_request = WebRequest::new_with_str_and_init(url.as_str(), &init).map_err(|_| {
                create_specific_fetch_error("Unable to create request for", url.as_str(), "")
            })?;

            let headers = web_request.headers();
            for (name, val) in request.headers() {
                headers.set(name, val).map_err(|_| ErrorResponse {
                    url: url.to_string(),
                    error: Error::FetchError("Got JS error".to_string()),
                })?;
            }

            // Worker-global fetch (the only difference from the main-thread backend).
            let scope: WorkerGlobalScope = js_sys::global().unchecked_into();
            let fetchval = JsFuture::from(scope.fetch_with_request(&web_request))
                .await
                .map_err(|_| ErrorResponse {
                    url: url.to_string(),
                    error: Error::FetchError("Got JS error".to_string()),
                })?;

            let response: WebResponse = fetchval.dyn_into().map_err(|_| ErrorResponse {
                url: url.to_string(),
                error: Error::FetchError("Fetch result wasn't a Response".to_string()),
            })?;
            let response_url = response.url();
            if !response.ok() {
                return Err(ErrorResponse {
                    url: response_url,
                    error: Error::HttpNotOk(
                        format!("Got {}", response.status_text()),
                        response.status(),
                        response.redirected(),
                        0,
                    ),
                });
            }

            Ok(Box::new(WebResponseWrapper {
                rewritten_url: None,
                response,
                body_stream: None,
            }) as Box<dyn SuccessResponse>)
        })
    }

    fn resolve_url(&self, url: &str) -> Result<Url, ParseError> {
        let parsed = match &self.base_url {
            Some(base) => base.join(url)?,
            None => Url::parse(url)?,
        };
        Ok(self.pre_process_url(parsed))
    }

    fn spawn_future(&mut self, future: OwnedFuture<(), Error>) {
        spawn_local(async move {
            if let Err(e) = future.await {
                tracing::error!("worker navigator future failed: {e}");
            }
        });
    }

    fn pre_process_url(&self, mut url: Url) -> Url {
        if self.upgrade_to_https && url.scheme() == "http" && url.set_scheme("https").is_err() {
            tracing::error!("worker navigator: Url::set_scheme failed on {url}");
        }
        url
    }

    fn connect_socket(
        &mut self,
        host: String,
        port: u16,
        _timeout: Duration,
        handle: SocketHandle,
        _receiver: Receiver<Vec<u8>>,
        sender: Sender<SocketAction>,
    ) {
        // Raw TCP sockets aren't reachable from a browser worker; report failure
        // so AVM sees a clean connect error rather than hanging.
        tracing::warn!("worker navigator: socket to {host}:{port} unsupported");
        let _ = sender.try_send(SocketAction::Connect(handle, ConnectionState::Failed));
    }
}
