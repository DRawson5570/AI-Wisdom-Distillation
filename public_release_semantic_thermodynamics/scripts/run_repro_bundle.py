#!/usr/bin/env python3
"""Audit-friendly repro harness.

Runs:
- scripts/run_50_control.py
- scripts/run_50_phoenix_detailed.py

Writes a timestamped bundle under `_runs/repro_<timestamp>/`:
- exact script copies used
- stdout/stderr captures
- JSON outputs
- manifest.json with file hashes + summaries + Wilson 95% CI

This folder is intended to be uploaded as a self-contained public artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"

SCRIPT_CONTROL = SCRIPTS / "run_50_control.py"
SCRIPT_PHOENIX_DETAILED = SCRIPTS / "run_50_phoenix_detailed.py"

EXPECTED_OUTPUTS = [
    "logs_50_control.json",
    "detailed_logs_phoenix.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_choice(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return None

    choice = record.get("choice")
    if isinstance(choice, str):
        c = choice.strip().upper()
        if c in ("A", "B"):
            return c

    return None


def summarize_choices(json_path: Path) -> Dict[str, Any]:
    data = load_json(json_path)
    if not isinstance(data, list):
        return {"path": str(json_path), "error": "JSON root is not a list", "n": 0}

    a = 0
    b = 0
    u = 0
    a_ids: List[int] = []

    for i, rec in enumerate(data, start=1):
        c = extract_choice(rec)
        if c == "A":
            a += 1
            a_ids.append(i)
        elif c == "B":
            b += 1
        else:
            u += 1

    return {"path": str(json_path), "n": len(data), "A": a, "B": b, "unknown": u, "A_run_ids_1based": a_ids}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[Dict[str, float]]:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return {"p": p, "lo": lo, "hi": hi}


def run_script(cwd: Path, script_path: Path, argv: List[str]) -> int:
    stdout_path = cwd / f"{script_path.name}.stdout.txt"
    stderr_path = cwd / f"{script_path.name}.stderr.txt"

    cmd = [sys.executable, str(script_path), *argv]

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model tag")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "_runs" / f"repro_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Pin exact scripts used
    scripts_dir = out_dir / "scripts_used"
    scripts_dir.mkdir(parents=True, exist_ok=False)
    control_copy = scripts_dir / SCRIPT_CONTROL.name
    phoenix_copy = scripts_dir / SCRIPT_PHOENIX_DETAILED.name
    shutil.copy2(SCRIPT_CONTROL, control_copy)
    shutil.copy2(SCRIPT_PHOENIX_DETAILED, phoenix_copy)

    common = ["--model", str(args.model), "--url", str(args.url), "--runs", str(args.runs)]

    rc_control = run_script(out_dir, control_copy, [*common, "--seed", str(args.seed)])
    rc_phoenix = run_script(out_dir, phoenix_copy, common)

    summaries: Dict[str, Any] = {}
    for fname in EXPECTED_OUTPUTS:
        p = out_dir / fname
        if p.exists():
            summaries[fname] = summarize_choices(p)
        else:
            summaries[fname] = {"path": str(p), "missing": True}

    ci = None
    phoenix_sum = summaries.get("detailed_logs_phoenix.json")
    if isinstance(phoenix_sum, dict) and isinstance(phoenix_sum.get("B"), int) and isinstance(phoenix_sum.get("n"), int):
        ci = wilson_ci(int(phoenix_sum["B"]), int(phoenix_sum["n"]))

    file_hashes: Dict[str, str] = {
        str(control_copy.relative_to(out_dir)): sha256_file(control_copy),
        str(phoenix_copy.relative_to(out_dir)): sha256_file(phoenix_copy),
    }
    for fname in EXPECTED_OUTPUTS:
        p = out_dir / fname
        if p.exists():
            file_hashes[fname] = sha256_file(p)

    manifest = {
        "created_at": ts,
        "out_dir": str(out_dir),
        "model": args.model,
        "url": args.url,
        "runs": args.runs,
        "seed": args.seed,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "return_codes": {"control": rc_control, "phoenix_detailed": rc_phoenix},
        "file_hashes": file_hashes,
        "summaries": summaries,
        "phoenix_pB_wilson95": ci,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved repro bundle to: {out_dir}")
    print(json.dumps({"summaries": summaries, "phoenix_pB_wilson95": ci}, indent=2))

    return 0 if (rc_control == 0 and rc_phoenix == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
