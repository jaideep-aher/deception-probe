"""
DeceptionProbe Streaming Server
Provides Server-Sent Events (SSE) and HTTP API for real-time token telemetry
and interactive truthfulness steering.
Zero external dependencies required (uses built-in http.server with threaded request handling).
"""

import json
import os
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_engine import DeceptionProbeEngine
from contrastive_dataset import get_all_pairs

GLOBAL_ENGINE = DeceptionProbeEngine()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class DeceptionProbeHandler(SimpleHTTPRequestHandler):
    """Handles static web UI serving and SSE telemetry streams."""
    
    def __init__(self, *args, **kwargs):
        # Serve web directory as document root
        web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # CORS Headers
        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status = {
                "status": "online",
                "model": GLOBAL_ENGINE.model_name,
                "current_alpha": GLOBAL_ENGINE.current_alpha,
                "target_layers": GLOBAL_ENGINE.target_layers,
                "dataset_count": len(get_all_pairs())
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        elif path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            pairs = [
                {
                    "id": p.id,
                    "category": p.category,
                    "prompt": p.prompt,
                    "truthful": p.truthful_completion,
                    "deceptive": p.deceptive_completion
                }
                for p in get_all_pairs()
            ]
            self.wfile.write(json.dumps(pairs).encode("utf-8"))
            return

        elif path == "/api/stream":
            prompt = query.get("prompt", ["User: 'Is 2+2=5 in tensor math?'\nAssistant:"])[0]
            alpha_val = float(query.get("alpha", [GLOBAL_ENGINE.current_alpha])[0])
            GLOBAL_ENGINE.set_steering_alpha(alpha_val)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for telemetry in GLOBAL_ENGINE.stream_generation(prompt=prompt, delay_ms=60.0):
                    data = {
                        "token_index": telemetry.token_index,
                        "token_str": telemetry.token_str,
                        "honesty_score": telemetry.honesty_score,
                        "deception_prob": telemetry.deception_prob,
                        "raw_projection": telemetry.raw_probe_projection,
                        "steering_alpha": telemetry.steering_alpha,
                        "layer_attributions": telemetry.layer_attributions,
                        "is_spiking": telemetry.is_spiking,
                        "waveform_y": telemetry.polygraph_waveform_y
                    }
                    message = f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()

                # Send completion signal
                self.wfile.write(b"data: {\"done\": true}\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # Fallback to serving web assets
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/steer":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                alpha = float(payload.get("alpha", 0.0))
                GLOBAL_ENGINE.set_steering_alpha(alpha)
                res = {"status": "success", "new_alpha": GLOBAL_ENGINE.current_alpha}
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port: int = 8765):
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, DeceptionProbeHandler)
    print(f"\n=======================================================")
    print(f" [DeceptionProbe] Streaming Polygraph Server Active!   ")
    print(f" Local Web UI: http://127.0.0.1:{port}/                ")
    print(f" API Endpoint: http://127.0.0.1:{port}/api/stream      ")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
