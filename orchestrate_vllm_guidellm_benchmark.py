#!/usr/bin/env python3
"""
vLLM (Podman) + GuideLLM benchmark orchestrator.

Execution order per run:
  1. Start container, sample metrics, run GuideLLM, stop container.
  2. Dashboard CSV: always write ``dashboard_benchmark.csv`` when GuideLLM JSON exists, then optional concurrency graphs (PNG); optionally append to ``--dashboard-csv``.
  3. Optional: single MLflow upload (tags, params, metrics, artifacts) — last step only.

See vllm_guidellm_benchmark/README.md for setup and examples.

Multiple suite files: pass ``--config`` more than once; each file is run to completion in order.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

HOME = Path.home()
DEFAULT_RUN_PODMAN = HOME / "run_podman.sh"
DEFAULT_GUIDELLM_BIN = HOME / "guidellm_env" / "bin" / "guidellm"
DEFAULT_IMPORT_SCRIPT = (
    HOME
    / "performance-dashboard"
    / "manual_runs"
    / "scripts"
    / "import_manual_runs_json_v2.py"
)

VMSTAT_KEYS = (
    "pgpgin",
    "pgpgout",
    "pswpin",
    "pswpout",
    "pgfault",
    "pgmajfault",
    "nr_dirty",
    "nr_writeback",
    "workingset_refault_anon",
    "workingset_activate_anon",
)

MLFLOW_TAG_VALUE_MAX = 500

# Performance-dashboard format (import_manual_runs_json_v2.py), copied per run for analysis.
DASHBOARD_CSV_IN_RUN_DIR = "dashboard_benchmark.csv"

GRAPH_OUTPUT_TOK = "graph_concurrency_output_tok_per_sec.png"
GRAPH_ITL = "graph_concurrency_itl.png"
GRAPH_TTFT = "graph_concurrency_ttft.png"


def as_kv_cache_gib(value: Any) -> int:
    """vLLM ``VLLM_CPU_KVCACHE_SPACE`` must be a non-negative integer (GiB)."""
    if isinstance(value, bool):
        raise ValueError("kv_cache_gb cannot be a boolean")
    try:
        n = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"kv_cache_gb must be a number, got {value!r}") from e
    if n < 0:
        raise ValueError("kv_cache_gb must be non-negative")
    if n != int(n):
        raise ValueError(
            f"kv_cache_gb must be a whole number (integer GiB), got {value!r}"
        )
    return int(n)


def kv_cache_gib_arg_type(value: str) -> int:
    try:
        return as_kv_cache_gib(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _read_vmstat() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/vmstat", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in VMSTAT_KEYS:
                out[parts[0]] = int(parts[1])
    return out


def _host_cpu_idle_total() -> tuple[int, int]:
    """Aggregate ``cpu`` line from ``/proc/stat``: (idle+iowait jiffies, total jiffies)."""
    with open("/proc/stat", encoding="utf-8") as f:
        for line in f:
            if line.startswith("cpu ") and not line.startswith("cpu0"):
                parts = line.split()
                nums = [int(x) for x in parts[1:]]
                if len(nums) < 4:
                    return 0, 0
                idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
                total = sum(nums)
                return idle, total
    return 0, 0


def _write_command_capture(path: Path, cmd: list[str], timeout: int = 120) -> None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        body = f"# command: {' '.join(cmd)}\n# exit_code: {r.returncode}\n\n"
        body += r.stdout or ""
        if r.stderr:
            body += f"\n### stderr ###\n{r.stderr}"
        path.write_text(body, encoding="utf-8")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        path.write_text(
            f"# command: {' '.join(cmd)}\n# failed: {type(e).__name__}: {e}\n",
            encoding="utf-8",
        )


def capture_system_pre_run_snapshot(run_dir: Path) -> None:
    """Before the benchmark: CPU topology, NUMA, memory (for artifacts and debugging)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_command_capture(run_dir / "system_lscpu.txt", ["lscpu"])
    _write_command_capture(run_dir / "system_numactl_hardware.txt", ["numactl", "--hardware"])
    _write_command_capture(run_dir / "system_numactl_show.txt", ["numactl", "--show"])
    _write_command_capture(run_dir / "system_free.txt", ["free", "-h"])
    _write_command_capture(run_dir / "system_uname.txt", ["uname", "-a"])


class _TeeText:
    """Write to multiple text streams (console + run log)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return self.streams[0].isatty() if self.streams else False


@contextlib.contextmanager
def tee_stdout_stderr_to_run_dir(run_dir: Path):
    """Mirror ``stdout``/``stderr`` to ``run_dir/orchestrator_console.log``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "orchestrator_console.log"
    f = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _TeeText(old_out, f)
        sys.stderr = _TeeText(old_err, f)
        yield log_path
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


def metrics_sampler(
    *,
    interval_sec: float,
    out_path: Path,
    stop_event: threading.Event,
) -> None:
    """Sample host CPU utilization (``/proc/stat``) and host ``/proc/vmstat`` counters."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prev_idle: int | None = None
    prev_total: int | None = None
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(
            "timestamp_unix\thost_cpu_util_pct\t" + "\t".join(VMSTAT_KEYS) + "\n"
        )
        while not stop_event.wait(interval_sec):
            ts = time.time()
            host_util = ""
            try:
                idle, total = _host_cpu_idle_total()
                if prev_idle is not None and prev_total is not None:
                    di = idle - prev_idle
                    dt = total - prev_total
                    if dt > 0:
                        host_util = f"{100.0 * (1.0 - di / dt):.2f}"
                prev_idle, prev_total = idle, total
            except OSError:
                prev_idle, prev_total = None, None

            vm = _read_vmstat()
            row = [f"{ts:.3f}", host_util] + [
                str(vm.get(k, "")) for k in VMSTAT_KEYS
            ]
            f.write("\t".join(row) + "\n")
            f.flush()


def wait_for_server(base_url: str, timeout_sec: int = 900) -> None:
    candidates = [
        f"{base_url.rstrip('/')}/health",
        f"{base_url.rstrip('/')}/v1/models",
    ]
    deadline = time.time() + timeout_sec
    last_err: str | None = None
    while time.time() < deadline:
        for url in candidates:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.HTTPError as e:
                last_err = f"{url}: HTTP {e.code}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{url}: {e!s}"
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready in {timeout_sec}s: {last_err}")


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:120]


def parse_extra_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def effective_container_env(
    *,
    kv_cache_gb: int,
    hf_home_container: str,
    vllm_omp_threads_bind: str,
    cpu_visible_memory_nodes: str | None,
    omp_num_threads: str | None,
    extra_env_file: Path | None,
) -> dict[str, str]:
    """Environment variables passed into the container (-e), for tags and manifest."""
    env: dict[str, str] = {
        "VLLM_CPU_KVCACHE_SPACE": str(int(kv_cache_gb)),
        "VLLM_CPU_OMP_THREADS_BIND": vllm_omp_threads_bind,
        "HF_HOME": hf_home_container,
    }
    if cpu_visible_memory_nodes:
        env["CPU_VISIBLE_MEMORY_NODES"] = cpu_visible_memory_nodes
    if omp_num_threads:
        env["OMP_NUM_THREADS"] = omp_num_threads
    env.update(parse_extra_env_file(extra_env_file))
    if "VLLM_CPU_KVCACHE_SPACE" in env:
        try:
            env["VLLM_CPU_KVCACHE_SPACE"] = str(
                as_kv_cache_gib(env["VLLM_CPU_KVCACHE_SPACE"])
            )
        except ValueError as e:
            raise ValueError(
                "VLLM_CPU_KVCACHE_SPACE (from CLI or EXTRA_ENV_FILE) must be a "
                f"non-negative integer GiB: {e}"
            ) from e
    return env


def build_auto_mlflow_tags(
    *,
    model: str,
    isl: int,
    osl: int,
    rate: str,
    kv_cache_gb: int,
    server_numa: int,
    client_numa: int,
    server_cpulist: str | None,
    port: int,
    vllm_image: str,
    vllm_extra_args: str,
    container_name: str,
    shm_size: str,
    hf_home: str,
    hf_home_container: str,
    container_runtime: str,
    max_seconds: int,
    container_env: dict[str, str],
    processor: str,
) -> dict[str, str]:
    """Tags derived from benchmark + container configuration (no user overrides)."""
    tags: dict[str, str] = {
        "benchmark.isl": str(isl),
        "benchmark.osl": str(osl),
        "benchmark.rate": rate,
        "benchmark.model": model,
        "benchmark.processor": processor,
        "benchmark.max_seconds": str(max_seconds),
        "host.server_numa_node": str(server_numa),
        "host.client_numa_node": str(client_numa),
        "host.container_runtime": container_runtime,
        "server.port": str(port),
        "server.image": vllm_image,
        "server.container_name": container_name,
        "server.shm_size": shm_size,
        "server.hf_home": hf_home,
        "server.vllm_extra_args": _truncate_tag(vllm_extra_args),
    }
    if server_cpulist:
        tags["host.server_cpulist"] = server_cpulist
    for k, v in sorted(container_env.items()):
        tags[f"container.env.{k}"] = _truncate_tag(v)
    return tags


def _truncate_tag(value: str) -> str:
    if len(value) <= MLFLOW_TAG_VALUE_MAX:
        return value
    return value[: MLFLOW_TAG_VALUE_MAX - 3] + "..."


def merge_tags(auto: dict[str, str], user: dict[str, str] | None) -> dict[str, str]:
    """User tags override auto tags on key collision."""
    merged = dict(auto)
    if user:
        merged.update(user)
    return merged


def run_podman_detached(*, run_podman_sh: Path, env: dict[str, str]) -> None:
    subprocess.run(
        ["/usr/bin/env", "bash", str(run_podman_sh)],
        env={**os.environ, **env},
        check=True,
        cwd=str(run_podman_sh.parent),
    )


def start_log_follower(
    *,
    runtime: str,
    container_name: str,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "ab")  # noqa: SIM115
    return subprocess.Popen(
        [runtime, "logs", "-f", container_name],
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd="/",
    )


def snapshot_container_logs(
    runtime: str,
    container_name: str,
    out_path: Path,
) -> bool:
    """Full container stdout/stderr from `podman logs` / `docker logs` (while container still exists)."""
    try:
        r = subprocess.run(
            [runtime, "logs", container_name],
            capture_output=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    if r.returncode != 0:
        return False
    chunks: list[bytes] = []
    if r.stdout:
        chunks.append(r.stdout)
    if r.stderr:
        chunks.append(b"\n### container stderr (from logs) ###\n")
        chunks.append(r.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"".join(chunks))
    return True


def stop_container(runtime: str, container_name: str, grace_sec: int = 45) -> None:
    subprocess.run(
        [runtime, "stop", "-t", str(grace_sec), container_name],
        capture_output=True,
        text=True,
    )


def run_guidellm(
    *,
    guidellm_bin: Path,
    client_numa: int,
    target: str,
    model: str,
    processor: str,
    isl: int,
    osl: int,
    rate: str,
    max_seconds: int,
    output_dir: Path,
    output_name: str,
) -> None:
    data = json.dumps({"prompt_tokens": isl, "output_tokens": osl})
    cmd = [
        "numactl",
        f"--cpunodebind={client_numa}",
        f"--membind={client_numa}",
        str(guidellm_bin),
        "benchmark",
        "--target",
        target,
        "--model",
        model,
        "--processor",
        processor,
        "--data",
        data,
        "--rate-type",
        "concurrent",
        "--rate",
        rate,
        "--backend-kwargs",
        '{"timeout":100000}',
        "--max-seconds",
        str(max_seconds),
        "--output-dir",
        str(output_dir),
        "--outputs",
        output_name,
    ]
    print("Running:", " ".join(cmd), flush=True)
    out_log = output_dir / "guidellm.stdout.log"
    err_log = output_dir / "guidellm.stderr.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_log, "w", encoding="utf-8", errors="replace") as out_f, open(
        err_log, "w", encoding="utf-8", errors="replace"
    ) as err_f:
        subprocess.run(
            cmd,
            check=True,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
    print(f"GuideLLM logs: {out_log} , {err_log}", flush=True)


def build_runtime_args_string(
    *,
    kv_cache_gb: int,
    server_cpulist: str | None,
    server_numa: int,
    vllm_extra_args: str,
    port: int,
) -> str:
    parts = [
        f"kv_cache_gib: {kv_cache_gb}",
        f"server_numa_node: {server_numa}",
        f"port: {port}",
        f"vllm_extra: {vllm_extra_args}",
    ]
    if server_cpulist:
        parts.append(f"server_cpulist: {server_cpulist}")
    return "; ".join(parts)


def guidellm_json_to_mlflow_metrics(json_path: Path) -> dict[str, float]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return {}
    b = benchmarks[0]
    metrics = b.get("metrics") or {}
    out: dict[str, float] = {}

    def grab(path: list[str], key: str, prefix: str) -> None:
        cur: Any = metrics
        for p in path:
            cur = cur.get(p) or {}
        if not isinstance(cur, dict):
            return
        v = cur.get(key)
        if isinstance(v, (int, float)):
            out[f"{prefix}_{key}"] = float(v)
        perc = cur.get("percentiles") or {}
        if isinstance(perc, dict):
            for pk, pv in perc.items():
                if isinstance(pv, (int, float)):
                    out[f"{prefix}_p_{pk}"] = float(pv)

    grab(["output_tokens_per_second", "total"], "mean", "out_tok_per_sec")
    grab(["tokens_per_second", "total"], "mean", "total_tok_per_sec")
    grab(["requests_per_second", "successful"], "mean", "rps")
    grab(["time_to_first_token_ms", "successful"], "median", "ttft_ms")
    grab(["time_per_output_token_ms", "successful"], "median", "tpot_ms")
    grab(["inter_token_latency_ms", "successful"], "median", "itl_ms")
    grab(["request_latency", "successful"], "median", "request_latency_s")
    return out


def build_mlflow_params(
    *,
    model: str,
    isl: int,
    osl: int,
    rate: str,
    kv_cache_gb: int,
    server_numa: int,
    client_numa: int,
    server_cpulist: str | None,
    vllm_extra_args: str,
    vllm_image: str,
    port: int,
    max_seconds: int,
    processor: str,
) -> dict[str, str]:
    p = {
        "model": model,
        "processor": processor,
        "isl": str(isl),
        "osl": str(osl),
        "rate": rate,
        "kv_cache_gb": str(int(kv_cache_gb)),
        "server_numa": str(server_numa),
        "client_numa": str(client_numa),
        "vllm_extra_args": vllm_extra_args,
        "vllm_image": vllm_image,
        "port": str(port),
        "max_seconds": str(max_seconds),
    }
    if server_cpulist:
        p["server_cpulist"] = server_cpulist
    return p


def append_dashboard_csv(
    *,
    import_script: Path,
    json_path: Path,
    csv_path: Path,
    model: str,
    version: str,
    tp: int,
    accelerator: str,
    runtime_args: str,
    image_tag: str,
    guidellm_version: str,
) -> None:
    if not import_script.is_file():
        print(f"Dashboard import script missing: {import_script}", file=sys.stderr)
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(import_script),
        str(json_path),
        "--model",
        model,
        "--version",
        version,
        "--tp",
        str(tp),
        "--accelerator",
        accelerator,
        "--runtime-args",
        runtime_args,
        "--image-tag",
        image_tag,
        "--guidellm-version",
        guidellm_version,
        "--csv-file",
        str(csv_path),
    ]
    subprocess.run(cmd, check=False)


def upload_mlflow_run(
    *,
    tracking_uri: str | None,
    experiment: str,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> None:
    """Final step: one MLflow run with tags, params, metrics, then artifacts."""
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("mlflow not installed; skip upload. pip install mlflow", file=sys.stderr)
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    resolved_uri = mlflow.get_tracking_uri()
    print(f"MLflow tracking URI (resolved): {resolved_uri}", flush=True)
    if not tracking_uri and resolved_uri.startswith("file:"):
        print(
            "Note: logging to a local file store. The mlflow UI only shows these runs if the "
            "server was started with the same backend, or you open the UI that serves this path. "
            "Prefer: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 matching `mlflow server`.",
            file=sys.stderr,
        )

    client = MlflowClient()
    try:
        mlflow.set_experiment(experiment)
        exp_obj = client.get_experiment_by_name(experiment)
        if exp_obj is None:
            raise RuntimeError(f"Could not create or resolve experiment {experiment!r}")
        experiment_id = exp_obj.experiment_id
    except MlflowException as e:
        print(f"MLflow experiment error: {e}", file=sys.stderr)
        raise

    run_uuid: str | None = None
    try:
        with mlflow.start_run(run_name=run_name) as active:
            run_uuid = active.info.run_id
            for k, v in sorted(tags.items()):
                if v is None or str(v).strip() == "":
                    continue
                mlflow.set_tag(k, _truncate_tag(str(v)))
            for k, v in sorted(params.items()):
                if v is None:
                    continue
                mlflow.log_param(k, str(v)[:6000])
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)) and v == v:
                    mlflow.log_metric(k, float(v))
            for path in artifacts:
                if path.is_file():
                    try:
                        mlflow.log_artifact(str(path))
                    except MlflowException as e:
                        print(
                            f"Warning: MLflow log_artifact failed for {path}: {e}",
                            file=sys.stderr,
                        )
                elif path.name == "vllm_server.log":
                    print(
                        f"Warning: MLflow artifact missing (no upload): {path}",
                        file=sys.stderr,
                    )
    except MlflowException as e:
        print(f"MLflow run failed: {e}", file=sys.stderr)
        raise

    if run_uuid:
        run = client.get_run(run_uuid)
        art_uri = run.info.artifact_uri
        print(
            f"MLflow: experiment={experiment!r} experiment_id={experiment_id} "
            f"run_id={run_uuid} artifact_uri={art_uri}",
            flush=True,
        )
        if resolved_uri.startswith("http"):
            base = resolved_uri.rstrip("/")
            print(
                f"MLflow UI (approx): {base}/#/experiments/{experiment_id}/runs/{run_uuid}",
                flush=True,
            )


@dataclass
class ResolvedRun:
    model: str
    processor: str
    isl: int
    osl: int
    rate: str
    kv_cache_gb: int
    server_cpulist: str | None
    server_numa: int
    client_numa: int
    vllm_extra: str
    vllm_image: str
    port: int
    max_seconds: int
    shm_size: str
    hf_home: str
    hf_home_container: str
    vllm_omp_threads_bind: str
    run_id: str
    run_dir: Path
    out_json: str
    container_name: str
    container_runtime: str
    container_env: dict[str, str]
    podman_env: dict[str, str]
    experiment: str
    run_name_ml: str
    user_mlflow_tags: dict[str, str] = field(default_factory=dict)
    extra_env_file_path: Path | None = None
    container_env_inline: dict[str, str] = field(default_factory=dict)


def _truncate_for_plot(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def dashboard_plot_context_text(rr: ResolvedRun) -> str:
    """Subtitle lines for benchmark graphs (ISL/OSL, KV, CPU, vLLM args)."""
    cpu = f"NUMA server={rr.server_numa} client={rr.client_numa}"
    if rr.server_cpulist:
        cpu += f" | server_cpulist={rr.server_cpulist}"
    lines = [
        f"ISL={rr.isl}  OSL={rr.osl}  |  KV cache={rr.kv_cache_gb} GiB  |  OMP bind={rr.vllm_omp_threads_bind}",
        cpu,
        f"Model: {rr.model}",
    ]
    if rr.vllm_extra:
        lines.append(f"vLLM args: {_truncate_for_plot(rr.vllm_extra, 200)}")
    lines.append(f"Image: {rr.vllm_image}")
    return "\n".join(lines)


def write_dashboard_benchmark_plots(*, csv_path: Path, rr: ResolvedRun) -> None:
    """Build concurrency vs latency/throughput PNGs from ``dashboard_benchmark.csv``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print(
            "matplotlib and pandas are required for dashboard graphs; "
            "install with: pip install matplotlib pandas",
            file=sys.stderr,
        )
        return

    if not csv_path.is_file():
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read {csv_path} for plots: {e}", file=sys.stderr)
        return

    if df.empty:
        print("dashboard_benchmark.csv is empty; skip graphs.", file=sys.stderr)
        return

    x_col = "intended concurrency"
    if x_col not in df.columns or df[x_col].isna().all():
        x_col = "measured concurrency"
    if x_col not in df.columns:
        print("No concurrency column in dashboard CSV; skip graphs.", file=sys.stderr)
        return

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df = df.dropna(subset=[x_col])
    if df.empty:
        print("No numeric concurrency values; skip graphs.", file=sys.stderr)
        return

    df = df.sort_values(x_col)
    ctx = dashboard_plot_context_text(rr)
    out_dir = rr.run_dir
    written: list[str] = []

    def _finish_figure(fig: Any, ax: Any, ylabel: str, title: str) -> None:
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.subplots_adjust(bottom=0.28, top=0.92)
        fig.text(
            0.5,
            0.06,
            ctx,
            ha="center",
            va="bottom",
            fontsize=7,
            family="monospace",
        )

    # Graph 1: output tok/s
    y1 = "output_tok/sec"
    if y1 in df.columns:
        s1 = pd.to_numeric(df[y1], errors="coerce")
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(df[x_col], s1, marker="o", linestyle="-", color="C0", label=y1)
        _finish_figure(fig, ax, "Output tokens / sec", "Concurrency vs throughput")
        fig.savefig(out_dir / GRAPH_OUTPUT_TOK, dpi=150)
        plt.close(fig)
        written.append(GRAPH_OUTPUT_TOK)

    # Graph 2: ITL (ms)
    itl_cols = ["itl_median", "itl_p95", "itl_p99"]
    present_itl = [c for c in itl_cols if c in df.columns]
    if present_itl:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for c in present_itl:
            ax.plot(
                df[x_col],
                pd.to_numeric(df[c], errors="coerce"),
                marker="o",
                linestyle="-",
                label=c,
            )
        _finish_figure(
            fig, ax, "ITL (ms)", "Concurrency vs inter-token latency"
        )
        fig.savefig(out_dir / GRAPH_ITL, dpi=150)
        plt.close(fig)
        written.append(GRAPH_ITL)

    # Graph 3: TTFT (seconds)
    ttft_cols = ["ttft_median", "ttft_p95", "ttft_p99"]
    present_ttft = [c for c in ttft_cols if c in df.columns]
    if present_ttft:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for c in present_ttft:
            sec = pd.to_numeric(df[c], errors="coerce") / 1000.0
            ax.plot(df[x_col], sec, marker="o", linestyle="-", label=f"{c} (s)")
        _finish_figure(fig, ax, "TTFT (s)", "Concurrency vs time to first token")
        fig.savefig(out_dir / GRAPH_TTFT, dpi=150)
        plt.close(fig)
        written.append(GRAPH_TTFT)

    if written:
        print(
            f"Wrote benchmark graph(s) under {out_dir}: {', '.join(written)}",
            flush=True,
        )


def _parse_env_file_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def materialize_merged_extra_env(rr: ResolvedRun) -> None:
    """If JSON ``container_env`` is set, merge with optional ``extra_env_file`` and set EXTRA_ENV_FILE."""
    if not rr.container_env_inline:
        return
    merged: dict[str, str] = {}
    if rr.extra_env_file_path is not None and rr.extra_env_file_path.is_file():
        merged.update(
            _parse_env_file_lines(
                rr.extra_env_file_path.read_text(encoding="utf-8")
            )
        )
    merged.update(rr.container_env_inline)
    path = rr.run_dir / "container_extra_env_merged.env"
    body = "\n".join(f"{k}={v}" for k, v in sorted(merged.items())) + "\n"
    path.write_text(body, encoding="utf-8")
    rr.podman_env["EXTRA_ENV_FILE"] = str(path.resolve())


def iter_benchmark_artifact_paths(rr: ResolvedRun) -> list[Path]:
    """Files under ``run_dir`` to attach to MLflow (existing files only)."""
    gj = rr.run_dir / rr.out_json
    candidates = [
        rr.run_dir / "orchestrator_console.log",
        rr.run_dir / "guidellm.stdout.log",
        rr.run_dir / "guidellm.stderr.log",
        rr.run_dir / "system_lscpu.txt",
        rr.run_dir / "system_numactl_hardware.txt",
        rr.run_dir / "system_numactl_show.txt",
        rr.run_dir / "system_free.txt",
        rr.run_dir / "system_uname.txt",
        rr.run_dir / "container_extra_env_merged.env",
        rr.run_dir / "vllm_server.log",
        rr.run_dir / "host_samples.tsv",
        gj,
        rr.run_dir / "run_manifest.json",
        rr.run_dir / DASHBOARD_CSV_IN_RUN_DIR,
        rr.run_dir / GRAPH_OUTPUT_TOK,
        rr.run_dir / GRAPH_ITL,
        rr.run_dir / GRAPH_TTFT,
    ]
    return [p for p in candidates if p.is_file()]


def resolve_run(cfg: dict[str, Any], args: argparse.Namespace) -> ResolvedRun:
    model = str(cfg.get("model", args.model))
    isl = int(cfg.get("isl", args.isl))
    osl = int(cfg.get("osl", args.osl))
    rate = str(cfg.get("rate", args.rate))
    try:
        kv = as_kv_cache_gib(cfg.get("kv_cache_gb", args.kv_cache_gb))
    except ValueError as e:
        print(f"Invalid kv_cache_gb: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    server_cpulist = cfg.get("server_cpulist", args.server_cpulist)
    server_numa = int(cfg.get("server_numa", args.server_numa))
    client_numa = int(cfg.get("client_numa", args.client_numa))
    vllm_extra = str(cfg.get("vllm_extra_args", args.vllm_extra_args))
    port = int(cfg.get("port", args.port))
    max_seconds = int(cfg.get("max_seconds", args.max_seconds))
    shm_size = str(cfg.get("shm_size", args.shm_size))
    hf_home = str(
        cfg.get("hf_home")
        or cfg.get("hf_cache_volume")
        or args.hf_home
    )
    hf_home_container = str(cfg.get("hf_home_container", args.hf_home_container))
    vllm_omp = str(cfg.get("vllm_omp_threads_bind", args.vllm_omp_threads_bind))
    processor = str(cfg.get("processor", model))
    vllm_image = str(cfg.get("vllm_image", args.vllm_image))

    run_id = slug(
        str(
            cfg.get("run_name")
            or f"{model}-kv{kv}-n{server_numa}-c{server_cpulist or 'all'}-isl{isl}-osl{osl}-r{rate}"
        )
    )
    run_dir = Path(str(cfg.get("output_dir") or args.output_base)) / run_id
    out_json = f"guidellm-isl{isl}-osl{osl}-{slug(rate)}.json"
    container_name = slug(str(cfg.get("container_name") or f"vllm-orch-{run_id}"))[:120]

    extra_env_path = cfg.get("extra_env_file")
    if extra_env_path is None and args.extra_env_file:
        extra_env_path = args.extra_env_file
    extra_path = Path(extra_env_path) if extra_env_path else args.extra_env_file

    cpu_vis = cfg.get("cpu_visible_memory_nodes", args.cpu_visible_memory_nodes)
    omp_threads = cfg.get("omp_num_threads", args.omp_num_threads)

    try:
        container_env = effective_container_env(
            kv_cache_gb=kv,
            hf_home_container=hf_home_container,
            vllm_omp_threads_bind=vllm_omp,
            cpu_visible_memory_nodes=str(cpu_vis) if cpu_vis else None,
            omp_num_threads=str(omp_threads) if omp_threads else None,
            extra_env_file=extra_path,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e

    inline_env_raw = cfg.get("container_env")
    container_env_inline: dict[str, str] = {}
    if isinstance(inline_env_raw, dict):
        container_env_inline = {str(k): str(v) for k, v in inline_env_raw.items()}
    if container_env_inline:
        container_env = dict(container_env)
        container_env.update(container_env_inline)
        if "VLLM_CPU_KVCACHE_SPACE" in container_env:
            try:
                container_env["VLLM_CPU_KVCACHE_SPACE"] = str(
                    as_kv_cache_gib(container_env["VLLM_CPU_KVCACHE_SPACE"])
                )
            except ValueError as e:
                print(f"Invalid VLLM_CPU_KVCACHE_SPACE in container_env: {e}", file=sys.stderr)
                raise SystemExit(2) from e

    podman_env: dict[str, str] = {
        "DETACHED": "1",
        "REPLACE_CONTAINER": "1",
        "CONTAINER_NAME": container_name,
        "CONTAINER_RUNTIME": args.container_runtime,
        "MODEL": model,
        "VLLM_IMAGE": vllm_image,
        "VLLM_EXTRA_ARGS": vllm_extra,
        "PORT": str(port),
        "VLLM_CPU_KVCACHE_SPACE": str(int(kv)),
        "SERVER_NUMA_NODE": str(server_numa),
        "SHM_SIZE": shm_size,
        "HF_HOME": hf_home,
        "HF_HOME_CONTAINER": hf_home_container,
        "VLLM_CPU_OMP_THREADS_BIND": vllm_omp,
    }
    if server_cpulist:
        podman_env["SERVER_CPULIST"] = str(server_cpulist)
    if extra_path and not container_env_inline:
        podman_env["EXTRA_ENV_FILE"] = str(extra_path)
    if cpu_vis:
        podman_env["CPU_VISIBLE_MEMORY_NODES"] = str(cpu_vis)
    if omp_threads:
        podman_env["OMP_NUM_THREADS"] = str(omp_threads)
    edf = cfg.get("extra_docker_run_file", args.extra_docker_run_file)
    if edf:
        podman_env["EXTRA_DOCKER_RUN_FILE"] = str(edf)

    utags: dict[str, str] = {}
    cfg_tags = cfg.get("mlflow_tags")
    if isinstance(cfg_tags, dict):
        utags = {str(k): str(v) for k, v in cfg_tags.items()}
    cli_tags = args.mlflow_tags or {}
    utags = {**utags, **{str(k): str(v) for k, v in cli_tags.items()}}

    return ResolvedRun(
        model=model,
        processor=processor,
        isl=isl,
        osl=osl,
        rate=rate,
        kv_cache_gb=kv,
        server_cpulist=str(server_cpulist) if server_cpulist else None,
        server_numa=server_numa,
        client_numa=client_numa,
        vllm_extra=vllm_extra,
        vllm_image=vllm_image,
        port=port,
        max_seconds=max_seconds,
        shm_size=shm_size,
        hf_home=hf_home,
        hf_home_container=hf_home_container,
        vllm_omp_threads_bind=vllm_omp,
        run_id=run_id,
        run_dir=run_dir,
        out_json=out_json,
        container_name=container_name,
        container_runtime=args.container_runtime,
        container_env=container_env,
        podman_env=podman_env,
        experiment=str(cfg.get("experiment", args.experiment)),
        run_name_ml=str(cfg.get("run_name", run_id))[:250],
        user_mlflow_tags=utags,
        extra_env_file_path=extra_path,
        container_env_inline=container_env_inline,
    )


def execute_benchmark_phase(rr: ResolvedRun, args: argparse.Namespace) -> None:
    rr.run_dir.mkdir(parents=True, exist_ok=True)
    materialize_merged_extra_env(rr)
    manifest = {
        "run_id": rr.run_id,
        "model": rr.model,
        "processor": rr.processor,
        "isl": rr.isl,
        "osl": rr.osl,
        "rate": rr.rate,
        "kv_cache_gb": rr.kv_cache_gb,
        "server_numa": rr.server_numa,
        "client_numa": rr.client_numa,
        "server_cpulist": rr.server_cpulist,
        "vllm_extra_args": rr.vllm_extra,
        "port": rr.port,
        "container_name": rr.container_name,
        "vllm_image": rr.vllm_image,
        "container_env": rr.container_env,
        "podman_launch_env": {k: v for k, v in rr.podman_env.items() if k != "DETACHED"},
    }
    manifest_path = rr.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    capture_system_pre_run_snapshot(rr.run_dir)

    vllm_log = rr.run_dir / "vllm_server.log"
    metrics_tsv = rr.run_dir / "host_samples.tsv"
    stop_event = threading.Event()
    sampler_thread = threading.Thread(
        target=metrics_sampler,
        kwargs={
            "interval_sec": args.sample_interval,
            "out_path": metrics_tsv,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    logs_p: subprocess.Popen[bytes] | None = None
    try:
        print("Starting vLLM container...", flush=True)
        run_podman_detached(run_podman_sh=args.run_podman_script, env=rr.podman_env)
        time.sleep(1)
        logs_p = start_log_follower(
            runtime=rr.container_runtime,
            container_name=rr.container_name,
            log_path=vllm_log,
        )
        sampler_thread.start()
        target = f"http://127.0.0.1:{rr.port}"
        wait_for_server(target, timeout_sec=args.ready_timeout)
        run_guidellm(
            guidellm_bin=args.guidellm_bin,
            client_numa=rr.client_numa,
            target=target,
            model=rr.model,
            processor=rr.processor,
            isl=rr.isl,
            osl=rr.osl,
            rate=rr.rate,
            max_seconds=rr.max_seconds,
            output_dir=rr.run_dir,
            output_name=rr.out_json,
        )
    finally:
        stop_event.set()
        sampler_thread.join(timeout=args.sample_interval + 5)
        if logs_p and logs_p.poll() is None:
            logs_p.terminate()
            try:
                logs_p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logs_p.kill()
        # Complete vLLM log for MLflow: streaming `logs -f` can truncate on terminate;
        # refresh from the runtime while the container is still running.
        ok = snapshot_container_logs(
            rr.container_runtime, rr.container_name, vllm_log
        )
        if not ok:
            print(
                "Warning: could not snapshot container logs (file may only contain streamed tail).",
                file=sys.stderr,
            )
        stop_container(rr.container_runtime, rr.container_name)


def finalize_after_benchmark(rr: ResolvedRun, args: argparse.Namespace) -> None:
    """Per-run dashboard CSV, optional consolidated CSV, then MLflow (always last)."""
    gj = rr.run_dir / rr.out_json
    runtime_args_str = build_runtime_args_string(
        kv_cache_gb=rr.kv_cache_gb,
        server_cpulist=rr.server_cpulist,
        server_numa=rr.server_numa,
        vllm_extra_args=rr.vllm_extra,
        port=rr.port,
    )

    if gj.is_file():
        per_run_dashboard = rr.run_dir / DASHBOARD_CSV_IN_RUN_DIR
        print(
            f"Writing dashboard-format CSV for this run: {per_run_dashboard}",
            flush=True,
        )
        append_dashboard_csv(
            import_script=args.import_script,
            json_path=gj,
            csv_path=per_run_dashboard,
            model=rr.model,
            version=args.dashboard_version,
            tp=args.dashboard_tp,
            accelerator=args.dashboard_accelerator,
            runtime_args=runtime_args_str,
            image_tag=rr.vllm_image,
            guidellm_version=args.dashboard_guidellm_version,
        )
        if per_run_dashboard.is_file():
            write_dashboard_benchmark_plots(csv_path=per_run_dashboard, rr=rr)
        if args.dashboard_csv:
            print(
                f"Appending same rows to consolidated dashboard CSV: {args.dashboard_csv}",
                flush=True,
            )
            append_dashboard_csv(
                import_script=args.import_script,
                json_path=gj,
                csv_path=args.dashboard_csv,
                model=rr.model,
                version=args.dashboard_version,
                tp=args.dashboard_tp,
                accelerator=args.dashboard_accelerator,
                runtime_args=runtime_args_str,
                image_tag=rr.vllm_image,
                guidellm_version=args.dashboard_guidellm_version,
            )

    mlflow_on = not args.no_mlflow and bool(
        args.mlflow
        or args.mlflow_tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
    )
    if not mlflow_on:
        print(f"Done. Artifacts under {rr.run_dir}", flush=True)
        return

    auto_tags = build_auto_mlflow_tags(
        model=rr.model,
        isl=rr.isl,
        osl=rr.osl,
        rate=rr.rate,
        kv_cache_gb=rr.kv_cache_gb,
        server_numa=rr.server_numa,
        client_numa=rr.client_numa,
        server_cpulist=rr.server_cpulist,
        port=rr.port,
        vllm_image=rr.vllm_image,
        vllm_extra_args=rr.vllm_extra,
        container_name=rr.container_name,
        shm_size=rr.shm_size,
        hf_home=rr.hf_home,
        hf_home_container=rr.hf_home_container,
        container_runtime=rr.container_runtime,
        max_seconds=rr.max_seconds,
        container_env=rr.container_env,
        processor=rr.processor,
    )
    tags = merge_tags(auto_tags, rr.user_mlflow_tags)
    params = build_mlflow_params(
        model=rr.model,
        isl=rr.isl,
        osl=rr.osl,
        rate=rr.rate,
        kv_cache_gb=rr.kv_cache_gb,
        server_numa=rr.server_numa,
        client_numa=rr.client_numa,
        server_cpulist=rr.server_cpulist,
        vllm_extra_args=rr.vllm_extra,
        vllm_image=rr.vllm_image,
        port=rr.port,
        max_seconds=rr.max_seconds,
        processor=rr.processor,
    )
    metrics = guidellm_json_to_mlflow_metrics(gj) if gj.is_file() else {}
    artifacts = iter_benchmark_artifact_paths(rr)

    print("Uploading run to MLflow...", flush=True)
    upload_mlflow_run(
        tracking_uri=args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
        experiment=rr.experiment,
        run_name=rr.run_name_ml,
        tags=tags,
        params=params,
        metrics=metrics,
        artifacts=artifacts,
    )
    print(f"Done. Artifacts under {rr.run_dir}", flush=True)


def single_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    rr = resolve_run(cfg, args)
    rr.run_dir.mkdir(parents=True, exist_ok=True)
    with tee_stdout_stderr_to_run_dir(rr.run_dir):
        execute_benchmark_phase(rr, args)
        finalize_after_benchmark(rr, args)


def run_suite_from_json_path(config_path: Path, args: argparse.Namespace) -> None:
    """Load one suite JSON and execute every entry in ``runs``."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    runs = data.get("runs")
    if not isinstance(runs, list):
        print(
            f"{config_path}: config must contain a 'runs' array",
            file=sys.stderr,
        )
        raise SystemExit(1)
    global_exp = data.get("experiment", args.experiment)
    default_tags = data.get("mlflow_tags")
    if not isinstance(default_tags, dict):
        default_tags = {}
    for i, run_cfg in enumerate(runs):
        if not isinstance(run_cfg, dict):
            continue
        merged = {**data.get("defaults", {}), **run_cfg}
        merged.setdefault("experiment", global_exp)
        mtags = {**default_tags}
        rt = merged.get("mlflow_tags")
        if isinstance(rt, dict):
            mtags.update(rt)
        merged["mlflow_tags"] = mtags
        print(
            f"=== [{config_path.name}] Run {i + 1}/{len(runs)}: "
            f"{merged.get('run_name', merged.get('model'))} ===",
            flush=True,
        )
        single_benchmark(merged, args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        action="append",
        type=Path,
        dest="config_files",
        metavar="FILE",
        help="Suite JSON with 'runs' list (repeat flag for multiple files, executed in order)",
    )
    p.add_argument("--output-base", type=Path, default=HOME / "results" / "vllm_guidellm_runs")

    p.add_argument("--run-podman-script", type=Path, default=DEFAULT_RUN_PODMAN)
    p.add_argument("--guidellm-bin", type=Path, default=DEFAULT_GUIDELLM_BIN)
    p.add_argument("--container-runtime", default=os.environ.get("CONTAINER_RUNTIME", "podman"))

    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--vllm-image", default="vllm/vllm-openai-cpu:v0.18.0")
    p.add_argument("--vllm-extra-args", default="--dtype=bfloat16")
    p.add_argument("--vllm-omp-threads-bind", default="auto")
    p.add_argument(
        "--kv-cache-gb",
        type=kv_cache_gib_arg_type,
        default=128,
        help="Integer GiB for VLLM_CPU_KVCACHE_SPACE (must be a whole number)",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--shm-size", default="4g")
    p.add_argument(
        "--hf-home",
        default="/home/naveen/models:/models",
        help="Podman -v bind (host:container); passed as HF_HOME to run_podman.sh",
    )
    p.add_argument(
        "--hf-cache-volume",
        default=None,
        help="Deprecated. Same as --hf-home (overrides --hf-home if set)",
    )
    p.add_argument("--hf-home-container", default="/models")

    p.add_argument("--server-numa", type=int, default=1)
    p.add_argument("--client-numa", type=int, default=0)
    p.add_argument("--server-cpulist", default=None)

    p.add_argument("--isl", type=int, default=128)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--rate", default="1,2")
    p.add_argument("--max-seconds", type=int, default=450)

    p.add_argument("--sample-interval", type=float, default=5.0)
    p.add_argument("--ready-timeout", type=int, default=900)

    p.add_argument("--extra-env-file", type=Path, default=None)
    p.add_argument("--extra-docker-run-file", type=Path, default=None)
    p.add_argument("--cpu-visible-memory-nodes", default=None)
    p.add_argument("--omp-num-threads", default=None)

    p.add_argument("--experiment", default="vllm-cpu-guidellm")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--mlflow-tags",
        type=json.loads,
        default=None,
        help='Extra MLflow tags (JSON object). Override auto tags on same key.',
    )
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")

    p.add_argument(
        "--dashboard-csv",
        type=Path,
        default=None,
        help="Optional consolidated CSV to append to (per-run dashboard_benchmark.csv is always written when GuideLLM JSON exists)",
    )
    p.add_argument("--import-script", type=Path, default=DEFAULT_IMPORT_SCRIPT)
    p.add_argument("--dashboard-version", default="vLLM-0.18.0-cpu")
    p.add_argument("--dashboard-tp", type=int, default=1)
    p.add_argument("--dashboard-accelerator", default="CPU")
    p.add_argument("--dashboard-guidellm-version", default="0.5.x")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "hf_cache_volume", None):
        args.hf_home = args.hf_cache_volume
    if not args.run_podman_script.is_file():
        print(f"Missing {args.run_podman_script}", file=sys.stderr)
        sys.exit(1)
    if not args.guidellm_bin.is_file():
        print(f"Missing GuideLLM binary {args.guidellm_bin}", file=sys.stderr)
        sys.exit(1)

    if args.config_files:
        for fi, cfg_path in enumerate(args.config_files):
            if not cfg_path.is_file():
                print(f"Config file not found: {cfg_path}", file=sys.stderr)
                raise SystemExit(1)
            print(
                f"########## Suite file {fi + 1}/{len(args.config_files)}: {cfg_path} ##########",
                flush=True,
            )
            run_suite_from_json_path(cfg_path, args)
    else:
        cfg = {
            "model": args.model,
            "isl": args.isl,
            "osl": args.osl,
            "rate": args.rate,
            "kv_cache_gb": args.kv_cache_gb,
            "server_cpulist": args.server_cpulist,
            "server_numa": args.server_numa,
            "client_numa": args.client_numa,
            "vllm_extra_args": args.vllm_extra_args,
            "vllm_omp_threads_bind": args.vllm_omp_threads_bind,
            "port": args.port,
            "max_seconds": args.max_seconds,
            "vllm_image": args.vllm_image,
            "output_dir": str(args.output_base),
            "experiment": args.experiment,
            "run_name": args.run_name,
            "mlflow_tags": args.mlflow_tags or {},
            "hf_home": args.hf_home,
        }
        single_benchmark(cfg, args)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
