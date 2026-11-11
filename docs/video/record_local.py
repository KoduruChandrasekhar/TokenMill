"""Run a real end-to-end session against the server on this machine and timestamp all output.
Writes session.json: blocks of {cmd, events: [[t, text], ...]} with t in seconds from block start."""
import json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(sys.argv[1]); V = Path(sys.argv[2])
env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "PYTHONUNBUFFERED": "1",
       "PYTHONIOENCODING": "utf-8"}
blocks = []

def stream_proc(cmd_display, argv, stop_when=None, cwd=None, shell=False, keep=lambda s: True):
    t0 = time.time(); events = []
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env, shell=shell)
    buf = b""
    while True:
        ch = p.stdout.read1(256) if hasattr(p.stdout, "read1") else p.stdout.read(1)
        if not ch:
            break
        text = ch.decode("utf-8", errors="replace")
        events.append([round(time.time() - t0, 3), text])
        buf += ch
        if stop_when and stop_when(buf.decode("utf-8", errors="replace")):
            break
    blocks.append({"cmd": cmd_display, "events": events})
    return p

# 1. the server
server_cmd = ["python", "-m", "infer.serve", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--device", "cpu", "--port", "8000"]
srv = stream_proc("python -m infer.serve --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --port 8000", server_cmd,
                  stop_when=lambda s: "Uvicorn running" in s, cwd=ROOT)
threading.Thread(target=lambda: [None for _ in iter(lambda: srv.stdout.read1(4096), b"")], daemon=True).start()
time.sleep(1)

# 2. raw SSE with curl
body = '{"prompt": "The three laws of thermodynamics are", "max_tokens": 12, "stream": true}'
stream_proc("curl -N localhost:8000/v1/completions -H 'content-type: application/json' \\n    -d '" + body + "'",
            ["curl", "-sN", "localhost:8000/v1/completions", "-H", "content-type: application/json", "-d", body])

# 3. the OpenAI SDK, unchanged
stream_proc("cat client.py", ["cat", "client.py"], cwd=V)
stream_proc("python client.py", ["python", "client.py"], cwd=V)

# 4. metrics
stream_proc("curl -s localhost:8000/metrics | grep -E '^(num_requests_running|kv_cache_usage|generation_tokens_total|prompt_tokens_total|request_success_total)'",
            [shutil.which("bash"), "-c", "curl -s localhost:8000/metrics | grep -E '^(num_requests_running|kv_cache_usage|generation_tokens_total|prompt_tokens_total|request_success_total)'"])

# 5. eight streams at once, recorded frame by frame
t0 = time.time()
subprocess.run(["python", "-m", "bench.demo", "--base-url", "http://localhost:8000", "--streams", "8",
                "--record", str(V / "local_frames.jsonl"), "--quiet"], cwd=ROOT, env=env, check=True)
frames = [json.loads(l) for l in (V / "local_frames.jsonl").read_text(encoding="utf-8").splitlines()]
blocks.append({"cmd": "python -m bench.demo --streams 8", "frames": frames, "wall": round(time.time() - t0, 2)})

srv.terminate(); srv.wait(timeout=30)
(V / "session.json").write_text(json.dumps(blocks, indent=1), encoding="utf-8")
print("blocks:", [(b["cmd"][:40], len(b.get("events", b.get("frames", [])))) for b in blocks])
