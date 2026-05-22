"""
run.py
======
Start the full ICP monitoring system with one command.
Works on Windows, macOS, Linux.

Usage:
    python run.py

Starts:
    Backend  → http://localhost:8001
    Frontend → http://localhost:3000  (opens in browser automatically)

Ctrl+C stops everything cleanly.
"""
import subprocess
import sys
import time
import socket
import threading
import webbrowser
from pathlib import Path

ROOT     = Path(__file__).parent
BACKEND  = ROOT / "icp-monitor-web" / "backend"
FRONTEND = ROOT / "icp-monitor-web" / "frontend"

BACKEND_PORT = 8001
FRONTEND_URL = "http://localhost:3000"

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) != 0


def npm_cmd() -> list[str]:
    """Return correct npm executable for the current OS."""
    return ["npm.cmd"] if sys.platform == "win32" else ["npm"]


def stream(proc: subprocess.Popen, tag: str) -> None:
    """Forward subprocess stdout+stderr to console with a prefix tag."""
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                print(f"  [{tag}] {line}")
    except Exception:
        pass


def wait_for_backend(port: int, timeout: int = 60) -> bool:
    import urllib.request, urllib.error
    url = f"http://localhost:{port}/api/health"
    for i in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
            if i > 0 and i % 10 == 0:
                print(f"      Still waiting... ({i}s elapsed)")
    return False


def run_cmd(cmd: list[str], cwd: Path, label: str) -> None:
    print(f"      Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ERROR during {label}:")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        sys.exit(1)


def launch(cmd: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    procs: list[subprocess.Popen] = []

    try:
        # ── Preflight checks ──────────────────────────────────────────────────
        print("\n-- Pre-flight checks --------------------------------------------------")
        if not BACKEND.exists():
            print(f"  ERROR: Backend not found at {BACKEND}")
            print("  Run this script from the v1_mimic_charis_only/ directory.")
            sys.exit(1)
        if not FRONTEND.exists():
            print(f"  ERROR: Frontend not found at {FRONTEND}")
            sys.exit(1)

        if not is_port_free(BACKEND_PORT):
            print(f"  ERROR: Port {BACKEND_PORT} is already in use.")
            print(f"  Kill the existing process first:  netstat -ano | findstr :{BACKEND_PORT}")
            sys.exit(1)

        print("  Ports free. Directories found. Proceeding.\n")

        # ── Backend dependencies ──────────────────────────────────────────────
        print("[1/4] Installing backend Python dependencies ...")
        run_cmd(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            cwd=BACKEND,
            label="pip install",
        )
        print("      Done.\n")

        # ── Frontend dependencies ─────────────────────────────────────────────
        if not (FRONTEND / "node_modules").exists():
            print("[2/4] Installing frontend npm dependencies (first run ~1 min) ...")
            run_cmd(npm_cmd() + ["install"], cwd=FRONTEND, label="npm install")
            print("      Done.\n")
        else:
            print("[2/4] Frontend node_modules present — skipping npm install.\n")

        # ── Start backend ─────────────────────────────────────────────────────
        print(f"[3/4] Starting FastAPI backend on http://localhost:{BACKEND_PORT} ...")
        backend = launch(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
            cwd=BACKEND,
        )
        procs.append(backend)
        threading.Thread(target=stream, args=(backend, "BACKEND"), daemon=True).start()

        print("      Waiting for backend to be ready ...")
        if wait_for_backend(BACKEND_PORT):
            print("      Backend ready.\n")
        else:
            print("      WARNING: Backend did not respond within 60s.")
            print("      Check [BACKEND] output above for import errors.\n")

        # ── Start frontend ────────────────────────────────────────────────────
        print(f"[4/4] Starting React frontend ...")
        frontend = launch(npm_cmd() + ["run", "dev"], cwd=FRONTEND)
        procs.append(frontend)
        threading.Thread(target=stream, args=(frontend, "FRONTEND"), daemon=True).start()

        # Wait for Vite to print its ready message (port 3000 set in vite.config.ts)
        time.sleep(4)
        print(f"\n      Opening {FRONTEND_URL} in browser ...")
        webbrowser.open(FRONTEND_URL)

        print("\n" + "=" * 55)
        print(f"  ICP Monitor is running.")
        print(f"  Frontend  : {FRONTEND_URL}")
        print(f"  Backend   : http://localhost:{BACKEND_PORT}")
        print(f"  API docs  : http://localhost:{BACKEND_PORT}/docs")
        print(f"  Press Ctrl+C to stop both servers.")
        print("=" * 55 + "\n")

        # Keep alive — wait for either process to exit
        while True:
            if backend.poll() is not None:
                print("\n  [BACKEND] Process exited unexpectedly. Check output above.")
                break
            if frontend.poll() is not None:
                print("\n  [FRONTEND] Process exited unexpectedly. Check output above.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n  Ctrl+C received — shutting down ...")

    finally:
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("  All processes stopped.\n")


if __name__ == "__main__":
    main()
