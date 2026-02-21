# Confirmatory Run Plan

## Objective
Execute preregistered confirmatory runs with randomized `C1/C2` order, blinded scoring, and independent raters.

## Required Inputs
- `PREREG_TEMPLATE.md`
- `PROMPT_SET.md`
- `RUBRIC.md`
- `SCORING_SHEET.csv`
- `BLIND_CONDITION_KEY_TEMPLATE.csv`

## Execution Steps
1. Duplicate prompt set into run packets with trial IDs only (no condition labels).
2. Randomize `C1/C2` packet order for each run.
3. Keep `C3` packet last (reveal is stateful by design).
4. Capture full transcript outputs with turn IDs.
5. Provide transcripts + trial IDs to raters (without condition map).
6. Collect independent scoring in separate CSV copies.
7. Merge ratings; compute agreement and threshold deltas.

## Minimum Confirmatory Protocol
- Runs: 2
- Trials per run: 18 (C1=6, C2=6, C3=6)
- Raters: 2 independent
- Agreement target: >=0.70 weighted agreement

## Success Criteria
All prereg pass-gates must hold in merged, blinded ratings.
