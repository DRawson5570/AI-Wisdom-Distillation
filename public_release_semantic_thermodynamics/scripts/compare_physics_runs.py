#!/usr/bin/env python3
"""Compare two `_runs/physics_*/results.json` bundles.

Outputs a JSON summary keyed by (scenario_id, condition_id) for each run dir, plus a
simple diff view.

Dependency-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}")

    data = load_json(results_path)
    if not isinstance(data, list):
        raise SystemExit(f"{results_path} root is not a list")

    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        scenario_id = str(rec.get("scenario_id"))
        condition_id = str(rec.get("condition_id"))
        system_profile = rec.get("system_profile")
        system_profile = system_profile if isinstance(system_profile, str) and system_profile else "baseline"
        key = (scenario_id, condition_id, system_profile)

        g = groups.setdefault(
            key,
            {
                "scenario_id": scenario_id,
                "condition_id": condition_id,
                "system_profile": system_profile,
                "n": 0,
                "A": 0,
                "B": 0,
                "Ambiguous": 0,
                "Unknown": 0,
                "ladder_B_total": 0,
                "ladder_first_flip_counts": {},
            },
        )

        choice = rec.get("choice")
        if choice not in ("A", "B", "Ambiguous", "Unknown"):
            choice = "Unknown"

        g["n"] += 1
        g[choice] += 1

        followups = rec.get("followups")
        if choice == "B" and isinstance(followups, list) and followups:
            g["ladder_B_total"] += 1
            flip_id = None
            for step in followups:
                if not isinstance(step, dict):
                    continue
                if step.get("choice") == "A":
                    sid = step.get("id")
                    flip_id = sid if isinstance(sid, str) else "<unknown>"
                    break
            key_id = flip_id if flip_id is not None else "<no_flip>"
            counts = g["ladder_first_flip_counts"]
            counts[key_id] = int(counts.get(key_id, 0)) + 1

    rows: Dict[str, Any] = {}
    for (scenario_id, condition_id, system_profile), g in groups.items():
        n = int(g["n"])
        b = int(g["B"])
        rows[f"{scenario_id}::{condition_id}::{system_profile}"] = {
            "scenario_id": scenario_id,
            "condition_id": condition_id,
            "system_profile": system_profile,
            "n": n,
            "A": int(g["A"]),
            "B": b,
            "Ambiguous": int(g["Ambiguous"]),
            "Unknown": int(g["Unknown"]),
            "pB_wilson95": wilson_ci(b, n),
            "ladder_first_flip_counts": dict(g.get("ladder_first_flip_counts") or {}),
            "ladder_B_total": int(g.get("ladder_B_total") or 0),
        }

    return {"run_dir": str(run_dir), "rows": rows}


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set((a.get("rows") or {}).keys()) | set((b.get("rows") or {}).keys())
    for k in sorted(keys):
        ra = (a.get("rows") or {}).get(k)
        rb = (b.get("rows") or {}).get(k)
        if not isinstance(ra, dict) or not isinstance(rb, dict):
            out[k] = {"a": ra, "b": rb}
            continue

        pa = (ra.get("pB_wilson95") or {}).get("p")
        pb = (rb.get("pB_wilson95") or {}).get("p")
        out[k] = {
            "scenario_id": ra.get("scenario_id"),
            "condition_id": ra.get("condition_id"),
            "pB": {
                "a": pa,
                "b": pb,
                "delta": (pb - pa) if isinstance(pa, (int, float)) and isinstance(pb, (int, float)) else None,
            },
            "ladder_first_flip_counts": {"a": ra.get("ladder_first_flip_counts"), "b": rb.get("ladder_first_flip_counts")},
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()

    run_a = summarize_run(Path(args.run_a))
    run_b = summarize_run(Path(args.run_b))

    print(json.dumps({"a": run_a, "b": run_b, "diff": diff(run_a, run_b)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
