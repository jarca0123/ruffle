#!/usr/bin/env python3
"""Static server that sets the COOP/COEP headers required for SharedArrayBuffer
(cross-origin isolation). Serves this directory on http://localhost:8080/."""
import http.server
import socketserver

PORT = 8088


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"serving on http://localhost:{PORT}/  (Ctrl-C to stop)")
    httpd.serve_forever()
