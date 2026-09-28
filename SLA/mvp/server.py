"""Dependency-free localhost server for the SLA portfolio demonstrator."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .planner import evaluate_portfolio
from .llm_interpreter import interpret_request
from .counterfactuals import pareto_counterfactuals
from .portfolio_state import accept_contract, load_portfolio


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
EXAMPLE = ROOT / "example_problem.json"
STATE = ROOT / "runtime" / "portfolio_state.json"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/example":
            body = json.dumps(load_portfolio(EXAMPLE, STATE), indent=2).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/evaluate", "/api/interpret", "/api/counterfactuals", "/api/accept"}:
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("request is too large")
            payload = json.loads(self.rfile.read(length))
            if path == "/api/interpret":
                result = interpret_request(payload.get("text", ""), payload["problem"])
            elif path == "/api/counterfactuals":
                result = pareto_counterfactuals(payload)
            elif path == "/api/accept":
                result = accept_contract(payload, STATE)
            else:
                result = evaluate_portfolio(payload)
            body = json.dumps(result, indent=2).encode("utf-8")
            self._send(200, body, "application/json")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, body, "application/json")

    def log_message(self, format: str, *args) -> None:
        print(f"[SLA MVP] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SLA portfolio demonstrator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"SLA demonstrator running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
