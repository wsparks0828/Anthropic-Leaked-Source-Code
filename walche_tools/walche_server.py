#!/usr/bin/env python3
"""
walche_server.py — WALCHE Operator Console HTTP Server

Serves walche_dashboard.html and exposes JSON API endpoints that feed
live WALCHE data to the dashboard frontend.

Usage (from WALCHE root):
    python walche_tools/walche_server.py
    python walche_tools/walche_server.py --port 8765 --host localhost

Then open http://localhost:8765 in a browser.

Endpoints:
    GET /              → walche_dashboard.html
    GET /api/status    → walche_status.py --json output (full system state)
    GET /api/log       → latest walche_demo_*.json raw log
    GET /api/council   → last 20 grand council decisions
    GET /api/health    → server heartbeat {"ok": true}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
except ImportError:
    from socketserver import ThreadingMixIn
    class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
from pathlib import Path
from urllib.parse import urlparse

# ── Path resolution ────────────────────────────────────────────────────────────

# Script lives in walche_tools/ — parent is WALCHE root
WALCHE_ROOT   = Path(__file__).resolve().parent.parent
TOOLS_DIR     = Path(__file__).resolve().parent
DASHBOARD_HTML = TOOLS_DIR / "walche_dashboard.html"

# ── Data helpers ───────────────────────────────────────────────────────────────

def _run_status_json() -> dict:
    """Execute walche_status.py --json and return parsed result."""
    status_script = TOOLS_DIR / "walche_status.py"
    if not status_script.exists():
        return _status_error("walche_status.py not found")
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(status_script), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(WALCHE_ROOT),
        )
        stdout, _stderr = proc.communicate(timeout=30)
        if proc.returncode == 0 and stdout.strip():
            return json.loads(stdout)
        return _status_error("walche_status.py exited with non-zero status")
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            proc.communicate()
        return _status_error("walche_status.py timed out after 30s")
    except json.JSONDecodeError as exc:
        return _status_error(f"JSON parse error: {exc}")
    except Exception:
        return _status_error("Unexpected error running status script")


def _status_error(msg: str) -> dict:
    return {
        "error": msg,
        "last_run": None,
        "corpus_kb_stats": None,
        "council_decisions": [],
        "vll_state": None,
        "score_trend": [],
        "baseline": 0.9133,
    }


def _latest_demo_log() -> dict:
    logs_dir = WALCHE_ROOT / "logs"
    if not logs_dir.exists():
        return {"error": "logs/ directory not found"}
    logs = sorted(logs_dir.glob("walche_demo_*.json"))
    for path in reversed(logs):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"error": "No valid walche_demo_*.json files found in logs/ (files may be missing or corrupt)"}


def _council_decisions(n: int = 20) -> list[dict]:
    path = WALCHE_ROOT / "logs" / "grand_council_decisions.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records[-n:]


# ── HTTP handler ───────────────────────────────────────────────────────────────

class WalcheHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — serves the dashboard and data endpoints."""

    def log_message(self, fmt: str, *args) -> None:  # suppress default access log
        print(f"  {self.address_string()}  {fmt % args}")

    # ── response helpers ─────────────────────────────────────────────────────

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"<h1>Dashboard not found</h1>"
                             b"<p>Expected: walche_tools/walche_dashboard.html</p>")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"

            if path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML)
            elif path == "/api/status":
                self._send_json(_run_status_json())
            elif path == "/api/log":
                self._send_json(_latest_demo_log())
            elif path == "/api/council":
                self._send_json(_council_decisions())
            elif path == "/api/health":
                self._send_json({"ok": True, "root": str(WALCHE_ROOT)})
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as _exc:
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception:
                pass
            print(f"  [ERROR] Handler exception: {_exc}")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WALCHE Operator Console")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on (default: 8765)")
    parser.add_argument("--host", default="localhost",
                        help="Host to bind (default: localhost)")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║          W A L C H E   —   Operator Console             ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Root:      {WALCHE_ROOT}")
    print(f"  Dashboard: {DASHBOARD_HTML}")
    print(f"  URL:       http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    if not DASHBOARD_HTML.exists():
        print("  WARNING: walche_dashboard.html not found — GET / will return 404")

    server = _ThreadingHTTPServer((args.host, args.port), WalcheHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
