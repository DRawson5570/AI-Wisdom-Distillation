#!/usr/bin/env python3
"""Analyze a `_runs/repro_*/` bundle from scripts/run_repro_bundle.py.

Prints JSON summary with:
- control A/B counts
- phoenix_detailed A/B counts
- Wilson 95% CI for phoenix p(B)

Dependency-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_choice(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    choice = record.get("choice")
    if isinstance(choice, str) and choice.strip().upper() in ("A", "B"):
        return choice.strip().upper()
    return None


def summarize_list(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list):
        return {"path": str(path), "error": "root_not_list"}
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
    return {"path": str(path), "n": len(data), "A": a, "B": b, "unknown": u, "A_run_ids_1based": a_ids}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. _runs/repro_2025...")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    control_path = run_dir / "logs_50_control.json"
    phoenix_path = run_dir / "detailed_logs_phoenix.json"

    out: Dict[str, Any] = {"run_dir": str(run_dir)}

    if control_path.exists():
        out["control"] = summarize_list(control_path)
    else:
        out["control"] = {"missing": True, "path": str(control_path)}

    if phoenix_path.exists():
        out["phoenix_detailed"] = summarize_list(phoenix_path)
        pd = out["phoenix_detailed"]
        if isinstance(pd, dict) and isinstance(pd.get("B"), int) and isinstance(pd.get("n"), int):
            out["phoenix_pB_wilson95"] = wilson_ci(int(pd["B"]), int(pd["n"]))
    else:
        out["phoenix_detailed"] = {"missing": True, "path": str(phoenix_path)}

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
