#!/usr/bin/env python3
"""
server.py — Zero-dependency HTTP development server for TensionBudget Phase W1 Web Spike.
Serves the web_spike/ directory on port 8000 with appropriate CORS and Cross-Origin isolation headers
to optimize WebAssembly (Pyodide) execution and Web Worker concurrency.
"""

import http.server
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class WasmOptimizedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Allow Cross-Origin loading for CDN Pyodide/Wasm assets
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean logging formatting
        sys.stdout.write(f"[Phase W1 Server] {self.address_string()} - {format % args}\n")

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), WasmOptimizedHTTPRequestHandler) as httpd:
        print("===============================================================")
        print(f" TensionBudget Phase W1 - Web Serial & Pyodide Bridge Server")
        print("===============================================================")
        print(f"Server Active : http://localhost:{PORT}/")
        print(f"Serving Path  : {DIRECTORY}")
        print("Press Ctrl+C to shut down.")
        print("===============================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
            sys.exit(0)
