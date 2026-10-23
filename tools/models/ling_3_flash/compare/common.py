"""Shared helpers for the model comparison harness (stdlib only).

Statistics, the seed-injecting proxy, GPU and host memory sampling, and the
`gufo serve` process manager.  Progress goes to stderr.  Stdout is reserved
for the JSON summary that compare.py writes.
"""
import http.client
import http.server
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

GIB = 1 << 30


def log(message):
    print(f"[ling-compare {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- statistics
# Two-sided 95% Student t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df):
    if df < 1:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def mean_ci(values):
    """Mean with a 95% t-interval across independent repeats (n >= 2)."""
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None, "lo": None, "hi": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "stdev": None, "ci95": None, "lo": None, "hi": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = t95(n - 1) * math.sqrt(var / n)
    return {"n": n, "mean": mean, "stdev": math.sqrt(var), "ci95": half,
            "lo": mean - half, "hi": mean + half}


def wilson(successes, total, z=1.96):
    """Wilson score 95% interval for a pooled binomial proportion."""
    if total <= 0:
        return {"n": 0, "p": None, "lo": None, "hi": None}
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"n": total, "successes": successes, "p": p,
            "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "min": min(values), "max": max(values), "sum": sum(values)}


def fisher_exact(a_pass, a_total, b_pass, b_total):
    """Fisher's exact test on the 2x2 table [[a_pass, a_fail], [b_pass, b_fail]].

    Returns the two-sided p (tables no more likely than the observed one) and the
    one-sided p for "A passes less often than B" (P[X <= a_pass] under the
    hypergeometric null with the margins fixed)."""
    a_fail, b_fail = a_total - a_pass, b_total - b_pass
    passes, n = a_pass + b_pass, a_total + b_total
    if min(a_pass, a_fail, b_pass, b_fail) < 0 or n == 0:
        return {"p_two_sided": None, "p_a_less": None, "p_a_greater": None}
    lo, hi = max(0, passes - b_total), min(passes, a_total)
    total = math.comb(n, passes)
    probs = {x: math.comb(a_total, x) * math.comb(b_total, passes - x) / total for x in range(lo, hi + 1)}
    observed = probs[a_pass]
    two = sum(p for p in probs.values() if p <= observed * (1 + 1e-9))
    return {"p_two_sided": min(1.0, two),
            "p_a_less": min(1.0, sum(p for x, p in probs.items() if x <= a_pass)),
            "p_a_greater": min(1.0, sum(p for x, p in probs.items() if x >= a_pass))}


def binomial_two_sided(k, n):
    """Exact two-sided sign test (McNemar on the discordant pairs)."""
    if n == 0:
        return None
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-9)))


# ------------------------------------------------------------------- memory
def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def drm_devices():
    base = Path("/sys/class/drm")
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("card[0-9]*/device") if (p / "mem_info_gtt_total").exists())


def gpu_memory():
    """Used/total GTT and VRAM of the first DRM device that reports them."""
    for device in drm_devices():
        return {"device": device,
                "gtt_used": _read_int(f"{device}/mem_info_gtt_used"),
                "gtt_total": _read_int(f"{device}/mem_info_gtt_total"),
                "vram_used": _read_int(f"{device}/mem_info_vram_used"),
                "vram_total": _read_int(f"{device}/mem_info_vram_total")}
    return None


def process_rss(pid):
    """(VmRSS, VmHWM) in bytes for one process, children excluded."""
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                fields[key] = int(rest.split()[0]) * 1024
        return fields.get("VmRSS"), fields.get("VmHWM")
    except (OSError, ValueError, IndexError):
        return None, None


def cgroup_memory():
    """cgroup v2 memory.current and memory.peak of this container, if present."""
    base = Path("/sys/fs/cgroup")
    return {"current": _read_int(base / "memory.current"), "peak": _read_int(base / "memory.peak")}


class MemorySampler:
    """Samples GTT, VRAM (system-wide counters) and process RSS every second.

    Peaks are reported as absolute values and as the increase over the value
    sampled before start(), which exposes idle usage on a shared node.
    """

    def __init__(self, interval=1.0, watch_prefix=None):
        self.interval = interval
        self.watch_prefix = watch_prefix
        self.source_refs = set()
        self.pid = None
        self._stop = threading.Event()
        self._thread = None
        self.reset()

    def reset(self):
        self.baseline = gpu_memory()
        self.peak_gtt = 0
        self.peak_vram = 0
        self.peak_rss = 0
        self.samples = 0

    def start(self, pid=None):
        self.pid = pid
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self):
        mem = gpu_memory()
        if mem:
            self.peak_gtt = max(self.peak_gtt, mem["gtt_used"] or 0)
            self.peak_vram = max(self.peak_vram, mem["vram_used"] or 0)
        if self.pid:
            rss, hwm = process_rss(self.pid)
            self.peak_rss = max(self.peak_rss, rss or 0, hwm or 0)
            if self.watch_prefix:
                self.source_refs |= open_refs(self.pid, self.watch_prefix)
        self.samples += 1

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def stop(self):
        self._sample()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        base = self.baseline or {}
        gib = lambda v: None if v is None else round(v / GIB, 3)  # noqa: E731
        return {"gtt_peak_gib": gib(self.peak_gtt) if self.baseline else None,
                "gtt_baseline_gib": gib(base.get("gtt_used")),
                "gtt_increase_gib": gib(self.peak_gtt - (base.get("gtt_used") or 0)) if self.baseline else None,
                "gtt_total_gib": gib(base.get("gtt_total")),
                "vram_peak_gib": gib(self.peak_vram) if self.baseline else None,
                "vram_increase_gib": gib(self.peak_vram - (base.get("vram_used") or 0)) if self.baseline else None,
                "process_rss_peak_gib": gib(self.peak_rss) if self.pid else None,
                "samples": self.samples,
                "source_dir_refs": sorted(self.source_refs) if self.watch_prefix else None}


def open_refs(pid, prefix):
    """Open files and mappings of a process under `prefix`."""
    found = set()
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith(prefix):
                found.add(target)
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].startswith(prefix):
                found.add(parts[5])
    except OSError:
        pass
    return found


def shard_files(model):
    """All files of a GGUF model: every NNNNN-of-MMMMM shard, or the file itself."""
    path = Path(model)
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not m:
        return [path]
    base, total = m.group(1), int(m.group(3))
    files = [path.with_name(f"{base}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    return files


def stage_files(sources, dest_dir):
    """Copy files to dest_dir (local disk), verify sizes, return (paths, report).

    Reads large sequential chunks and drops the page cache on both files as it
    goes, so the copy does not hit a small pod memory limit.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = sum(Path(s).stat().st_size for s in sources)
    free = shutil.disk_usage(dest_dir).free
    if free < total + (1 << 30):
        raise RuntimeError(f"staging needs {total} bytes, only {free} free in {dest_dir}")
    started = time.monotonic()
    out, files = [], []
    done = 0
    for src in sources:
        src = Path(src)
        dst = dest_dir / src.name
        size = src.stat().st_size
        if dst.exists() and dst.stat().st_size == size:
            log(f"stage: {dst.name} already present ({size} bytes)")
            done += size
            out.append(dst)
            files.append({"name": src.name, "bytes": size, "reused": True})
            continue
        tmp = dst.with_suffix(".part")
        t0 = time.monotonic()
        with open(src, "rb", buffering=0) as fin, open(tmp, "wb", buffering=0) as fout:
            copied = 0
            while True:
                chunk = fin.read(16 << 20)
                if not chunk:
                    break
                fout.write(chunk)
                copied += len(chunk)
                done += len(chunk)
                if copied % (1 << 30) < (16 << 20):
                    os.posix_fadvise(fin.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    fout.flush()
                    os.fsync(fout.fileno())
                    os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    log(f"stage: {done / GIB:.1f}/{total / GIB:.1f} GiB "
                        f"({done / GIB / max(1e-9, time.monotonic() - started):.2f} GiB/s)")
            fout.flush()
            os.fsync(fout.fileno())
            os.posix_fadvise(fin.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if tmp.stat().st_size != size:
            raise RuntimeError(f"staged size mismatch for {src.name}: {tmp.stat().st_size} != {size}")
        tmp.rename(dst)
        out.append(dst)
        files.append({"name": src.name, "bytes": size, "seconds": round(time.monotonic() - t0, 1)})
    seconds = round(time.monotonic() - started, 1)
    for src, dst in zip(sources, out):
        if Path(src).stat().st_size != dst.stat().st_size:
            raise RuntimeError(f"size verification failed for {dst}")
    return out, {"dir": str(dest_dir), "bytes": total, "gib": round(total / GIB, 3), "seconds": seconds,
                 "files": files}


# ------------------------------------------------------------ server manager
class Server:
    """Runs `gufo serve llm` for one model and records load time and memory."""

    def __init__(self, command, port, log_path, ready_timeout, watch_prefix=None):
        self.command, self.port, self.log_path = command, port, Path(log_path)
        self.watch_prefix = watch_prefix
        self.ready_timeout = ready_timeout
        self.proc = None
        self.load_seconds = None
        self.sampler = MemorySampler(watch_prefix=watch_prefix)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log("starting " + " ".join(self.command))
        started = time.monotonic()
        self.logfile = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.command, stdout=self.logfile, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.sampler.start(self.proc.pid)
        deadline = started + self.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.load_seconds = None
                return False, f"server exited with {self.proc.returncode} before ready"
            if self._ready():
                self.load_seconds = round(time.monotonic() - started, 2)
                log(f"ready after {self.load_seconds}s")
                return True, None
            time.sleep(2)
        return False, f"server not ready after {self.ready_timeout}s"

    def _ready(self):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", "/ready")
            ok = conn.getresponse().status == 200
            conn.close()
            return ok
        except OSError:
            return False

    def stop(self):
        memory = self.sampler.stop()
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        try:
            self.logfile.close()
        except Exception:
            pass
        return memory

    def _log_lines(self):
        try:
            return self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return []

    def log_head(self, lines=60):
        return self._log_lines()[:lines]

    def log_tail(self, lines=40):
        return self._log_lines()[-lines:]


# ------------------------------------------------------------ seeding proxy
class SeedProxy:
    """Local reverse proxy that adds a fixed "seed" to every chat completion.

    `gufo eval` has no seed option and the server's default seed is random.
    One proxy per repeat makes the repeat reproducible and lets repeats run
    concurrently against one server.  Other body fields pass through unchanged.
    """

    def __init__(self, upstream_port, seed):
        self.upstream_port, self.seed = upstream_port, seed
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _forward(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                if body and self.path.endswith("/chat/completions"):
                    try:
                        data = json.loads(body)
                        data.setdefault("seed", outer.seed)
                        body = json.dumps(data).encode()
                    except ValueError:
                        pass
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in ("host", "content-length", "connection")}
                if body is not None:
                    headers["Content-Length"] = str(len(body))
                conn = http.client.HTTPConnection("127.0.0.1", outer.upstream_port, timeout=3600)
                try:
                    conn.request(method, self.path, body, headers)
                    response = conn.getresponse()
                    payload = response.read()
                    self.send_response(response.status)
                    for key, value in response.getheaders():
                        if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError as error:
                    self.send_error(502, str(error))
                finally:
                    conn.close()

            def do_GET(self):
                self._forward("GET")

            def do_POST(self):
                self._forward("POST")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ------------------------------------------------------------------ http json
def http_json(base_url, path, body=None, timeout=60, method=None):
    url = urllib.parse.urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        conn.request(method or ("POST" if body is not None else "GET"), path, payload,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        return response.status, data
    finally:
        conn.close()
