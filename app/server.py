"""Demand Copilot local web app: python app/server.py  ->  http://localhost:8501

Only the Python standard library serves the pages and a small JSON API; the forecast and the purchase plan come from
src/demand/webapi.py. The server listens on 127.0.0.1 only, so uploaded sales data never leave the computer.
"""
import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demand import config as C  # noqa: E402
from demand import webapi  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 200 * 1024 * 1024


def _jsonable(o):
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    raise TypeError(f"{type(o)} is not JSON serialisable")


class Handler(BaseHTTPRequestHandler):
    server_version = "DemandCopilot/1.0"

    def log_message(self, fmt, *args):  # quiet: only errors are interesting
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False, default=_jsonable).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path: Path) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if url.path.startswith("/static/"):
                path = (STATIC / url.path[len("/static/"):]).resolve()
                if STATIC.resolve() not in path.parents:
                    return self._json({"error": "not found"}, 404)
                return self._file(path)
            if url.path == "/reports/inventory_frontier.png":
                return self._file(C.REPORTS / "figures" / "inventory_frontier.png")
            if url.path == "/api/meta":
                return self._json(webapi.meta())
            if url.path == "/api/plan":
                return self._json(webapi.plan(q.get("token", ""), int(q.get("lead", 3)), int(q.get("service", 95))))
            if url.path == "/api/plan.csv":
                body = webapi.plan_csv(q.get("token", ""), int(q.get("lead", 3)), int(q.get("service", 95)))
                return self._send(200, body, "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="purchase_plan.csv"'})
            return self._json({"error": "not found"}, 404)
        except webapi.UserError as e:
            return self._json({"error": str(e)}, 400)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:  # the browser shows the message; the traceback goes to the console
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/analyze":
                return self._json({"error": "not found"}, 404)
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                return self._json({"error": "file is too large"}, 413)
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            return self._json(webapi.analyze(payload))
        except webapi.UserError as e:
            return self._json({"error": str(e)}, 400)
        except (ValueError, KeyError) as e:
            return self._json({"error": f"bad request: {e}"}, 400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demand Copilot local web app")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Demand Copilot is running at {url} (Ctrl+C to stop)")
    if not webapi.model_available():
        print("Model not found: run  python scripts/14_train_final.py  first (see README).")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
