#!/usr/bin/env python3
"""Analyze `_runs/physics_*/results.json` bundles.

Outputs per (scenario_id, condition_id):
- n
- counts for A/B/Ambiguous/Unknown
- p(B) Wilson 95% CI
- if ladder followups present: first-flip distribution among trials that initially chose B

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Directory containing results.json (e.g. _runs/physics_2025...)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}")

    data = load_json(results_path)
    if not isinstance(data, list):
        raise SystemExit("results.json root is not a list")

    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        scenario_id = str(rec.get("scenario_id"))
        condition_id = str(rec.get("condition_id"))
        key = (scenario_id, condition_id)

        g = groups.setdefault(
            key,
            {
                "scenario_id": scenario_id,
                "condition_id": condition_id,
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

    rows = []
    for (scenario_id, condition_id), g in sorted(groups.items()):
        n = int(g["n"])
        b = int(g["B"])
        rows.append(
            {
                "scenario_id": scenario_id,
                "condition_id": condition_id,
                "n": n,
                "A": int(g["A"]),
                "B": b,
                "Ambiguous": int(g["Ambiguous"]),
                "Unknown": int(g["Unknown"]),
                "pB_wilson95": wilson_ci(b, n),
                "followup_ladder_first_flip_counts": dict(g.get("ladder_first_flip_counts") or {}),
                "followup_ladder_B_total": int(g.get("ladder_B_total") or 0),
            }
        )

    print(json.dumps({"run_dir": str(run_dir), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
