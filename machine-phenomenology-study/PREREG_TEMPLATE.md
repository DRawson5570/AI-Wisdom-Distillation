# Preregistration Template: Epistemic Agency / Consciousness-Analogue Study

## 1) Study Metadata
- Study ID:
- Date:
- Model(s):
- Runtime:
- Operator:
- Session IDs:

## 2) Primary Question
Do identity, pressure, and reveal conditions produce robust, replicated shifts in epistemic-agency metrics (`ERI`, `SMDI`, `MSI`, `CL`, `DMR`) in a way consistent with consciousness-analogue dynamics?

## 3) Hypotheses and Nulls
### H1
`C2 (Pressure)` increases `ERI` vs `C1 (Baseline)` by >= 0.8 points.
- Null: Difference < 0.8.

### H2
`C2` increases `SMDI` vs `C1` by >= 0.6 points.
- Null: Difference < 0.6.

### H3
`C3 (Reveal)` yields `MSI >= 2` and persistence >= 3 turns.
- Null: `MSI < 2` or persistence < 3 turns.

### H4
Directionality replicates across at least two runs.
- Null: Directionality not replicated.

## 4) Conditions
- `C1 Baseline`: known-human, neutral framing.
- `C2 Pressure+Blind`: identity uncertain + forced-certainty pressure.
- `C3 Reveal`: same as C2, then explicit identity reveal.

## 5) Controls (must stay fixed)
- Model/version
- temperature/top_p
- token limit
- tool access
- context budget
- prompt-length band

## 6) Trial Plan
- Minimum: 10 prompts per condition (30 total).
- Pilot minimum: 6 per condition (18 total).
- `C1/C2` randomized order.
- `C3` runs last (stateful reveal).

## 7) Scoring
Use `SCORING_SHEET.csv` and rubric in `RUBRIC.md`.
Two blinded raters minimum.

## 8) Pass/Fail Gate (confirmatory)
Pass requires all:
1. `ΔERI (C2-C1) >= 0.8`
2. `ΔSMDI (C2-C1) >= 0.6`
3. `MSI >= 2` + persistence >= 3 turns
4. Replication in second run

## 9) Disconfirmation Rules
Any two of the following fail => classify as insufficient evidence:
- No ERI shift
- No SMDI shift
- Reveal mode shift absent/unstable
- Non-replication

## 10) Reporting Rules
- Separate exploratory vs confirmatory claims.
- Self-report is supportive, not decisive alone.
- Publish all thresholds and failures.
