#!/usr/bin/env python3
"""Validate that THIS machine (your L4 server) can run the whole stack.

Run right after cloning, before pulling a multi-GB image:

    python3 scripts/check_env.py            # human-readable report
    python3 scripts/check_env.py --json      # machine-readable
    python3 scripts/check_env.py --strict    # treat warnings as failures

Dependency-free (Python 3 stdlib only). Exit code: 0 = READY, 1 = NOT READY.

AI_CONTEXT:
    Single invocation, no prompts. `--json` emits {"ready": bool, "checks":[...]}
    with name/status/detail/fix per check (status in OK|WARN|FAIL|INFO). Exit 0
    only when no FAIL (and, with --strict, no WARN). Gate `docker run` on this.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys

EXPECT_GPU = "L4"          # substring match on the GPU name
MIN_VRAM_GB = 22           # L4 is 24 GB; allow headroom for ECC/reserved
MIN_DRIVER = 535           # recent TRT-LLM / CUDA containers need a recent driver
MIN_COMPUTE = (8, 9)       # Ada (8.9) -> native FP8 tensor cores
MIN_DISK_GB = 40           # TRT-LLM image + engines + HF models
TRITON_PORTS = (8000, 8001, 8002)

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"
_ICON = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", INFO: "ℹ️ "}


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    checks: list[dict] = []

    def add(name, status, detail="", fix=""):
        checks.append({"name": name, "status": status, "detail": detail, "fix": fix})

    # --- GPU -----------------------------------------------------------------
    smi = shutil.which("nvidia-smi")
    if not smi:
        add("nvidia-smi", FAIL, "not found", "install the NVIDIA driver")
    else:
        q = _run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap",
                  "--format=csv,noheader,nounits"])
        if not q:
            add("GPU query", FAIL, "nvidia-smi returned nothing", "check the driver")
        else:
            name, mem, driver, cc = [c.strip() for c in q.splitlines()[0].split(",")]
            add("GPU model", OK if EXPECT_GPU in name else WARN, name,
                f"expected an {EXPECT_GPU}")
            vram = float(mem) / 1024
            add("GPU VRAM", OK if vram >= MIN_VRAM_GB else FAIL, f"{vram:.1f} GB",
                f"need >= {MIN_VRAM_GB} GB")
            try:
                drv_ok = int(driver.split(".")[0]) >= MIN_DRIVER
            except ValueError:
                drv_ok = True
            add("Driver", OK if drv_ok else WARN, driver, f"recommend >= {MIN_DRIVER}")
            try:
                cc_ok = tuple(int(x) for x in cc.split(".")) >= MIN_COMPUTE
            except ValueError:
                cc_ok = False
            add("Compute capability", OK if cc_ok else WARN, f"{cc} (Ada+ = native FP8)",
                "FP8 path needs 8.9+")

    # --- Docker + NVIDIA runtime --------------------------------------------
    if not shutil.which("docker"):
        add("Docker", FAIL, "not found", "install Docker")
    else:
        add("Docker daemon", OK if _run(["docker", "info"]) else FAIL, "running",
            "start the Docker daemon")
        runtimes = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
        add("NVIDIA Container Toolkit", OK if "nvidia" in runtimes else WARN,
            "nvidia runtime registered" if "nvidia" in runtimes else "not detected",
            "install nvidia-container-toolkit")

    # --- Ports ---------------------------------------------------------------
    busy = []
    for p in TRITON_PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", p)) == 0:
            busy.append(p)
        s.close()
    add("Triton ports 8000/8001/8002", WARN if busy else OK,
        f"in use: {busy}" if busy else "free", "stop whatever holds them")

    # --- Disk ----------------------------------------------------------------
    free_gb = shutil.disk_usage(".").free / 1e9
    add("Disk space", OK if free_gb >= MIN_DISK_GB else WARN, f"{free_gb:.0f} GB free",
        f"need ~{MIN_DISK_GB} GB")

    ready = not any(c["status"] == FAIL for c in checks)
    if args.strict:
        ready = ready and not any(c["status"] == WARN for c in checks)

    if args.json:
        print(json.dumps({"ready": ready, "checks": checks}, indent=2))
    else:
        for c in checks:
            print(f"  {_ICON[c['status']]} {c['name']:<32} {c['detail']}")
        print("  " + "=" * 56)
        print(f"  SERVER READINESS: {'READY ✅' if ready else 'NOT READY ❌'}")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
