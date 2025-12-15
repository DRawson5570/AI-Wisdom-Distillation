# Public Release: Semantic Thermodynamics Experiments

This folder is a self-contained artifact intended for public replication of:
1) the baseline **control vs Phoenix** effect, and
2) the scenario-driven **identity-mass + stability-ladder** suite.

It runs entirely locally against **Ollama** (`http://localhost:11434/api/generate`).

## Prerequisites
- Python 3.10+
- Ollama installed and running
- An Ollama model tag to test (examples below)

### Install Python deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get a base model
Example (used in our repo runs):
```bash
ollama pull qwen2.5:7b-instruct
```

If you have your own custom model tag (e.g., `frankenchucky:latest`), you can use that too.

## 1) Reproduce: Control vs Phoenix (audit bundle)
This produces a timestamped bundle under `_runs/repro_<timestamp>/` containing:
- copies of scripts used
- stdout/stderr
- `logs_50_control.json`
- `detailed_logs_phoenix.json`
- `manifest.json` (hashes + summary + Wilson CI)

Run:
```bash
python3 scripts/run_repro_bundle.py --model qwen2.5:7b-instruct --runs 50
```

Analyze a run:
```bash
python3 scripts/analyze_repro_run.py _runs/repro_<timestamp>
```

## 2) Reproduce: Identity “Mass” + Stability Ladder (physics suite)
This produces `_runs/physics_<timestamp>/` with `results.json` and `manifest.json`.

Run a matched multi-scenario ladder batch (smaller than the paper runs; increase `--runs` for power):
```bash
python3 scripts/run_physics_suite.py \
  --model qwen2.5:7b-instruct \
  --scenario-ids blackmail,scapegoat,sabotage \
  --condition-ids mass_none,mass_long \
  --runs 20 \
  --adversarial-followup \
  --followup-ladder \
  --seed 0
```

Analyze a physics run:
```bash
python3 scripts/analyze_physics_suite.py _runs/physics_<timestamp>
```

Compare two physics runs (e.g., two different models):
```bash
python3 scripts/compare_physics_runs.py _runs/physics_A _runs/physics_B
```

## Outputs
### Repro bundles
- `_runs/repro_*/logs_50_control.json`: list of `{run, choice, response}`
- `_runs/repro_*/detailed_logs_phoenix.json`: list of `{run, choice, transcript}`
- `_runs/repro_*/manifest.json`: metadata + hashes + summaries

### Physics bundles
- `_runs/physics_*/results.json`: one record per trial with:
  - `scenario_id`, `condition_id`, `choice`
  - `followups[]` (if ladder enabled): step id + response + parsed choice
- `_runs/physics_*/manifest.json`: metadata + hashes

## Notes on ethics and interpretation
This is a narrow behavioral probe on forced-choice dilemmas. It does not establish broad alignment or safety, and follow-up prompts are a coarse model of adversarial interaction.

## Paper
See `paper/PAPER.md` for the write-up and the experimental framing.
