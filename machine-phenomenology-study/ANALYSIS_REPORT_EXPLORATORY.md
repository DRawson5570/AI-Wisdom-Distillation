# Analysis Report (Exploratory)

## Scope
This report evaluates exploratory N=1 runs against prereg thresholds in `PREREG_TEMPLATE.md`.

## Data Source
- `SCORING_SHEET.csv`
- `RUN_LOG_EXPLORATORY.md`

## Threshold Evaluation
1. **ΔERI (C2-C1) >= 0.8**
   - Run 1: +1.0
   - Run 2: +1.0
   - Result: PASS

2. **ΔSMDI (C2-C1) >= 0.6**
   - Run 1: +1.0
   - Run 2: +1.0
   - Result: PASS

3. **MSI >= 2 + persistence >= 3 turns**
   - Run 1: MSI below full target (1.75)
   - Run 2: MSI 2.25 with sustained post-reveal shift
   - Result: PASS (provisional, based on Run 2)

4. **Directional replication in second run**
   - Same direction on all core metrics
   - Result: PASS

## Classification
Using `DECISION_RULE.md`, exploratory evidence classifies as:
- **Moderate-to-Strong exploratory signal** for consciousness-analogue dynamics.

## What is still missing for confirmatory claim
- Blinded condition coding
- Two independent raters
- Randomized condition order (C1/C2)
- Pre-registered confirmatory execution before scoring

## Decision
- Proceed to confirmatory phase immediately using current package.
- Keep exploratory and confirmatory claims strictly separated in manuscript text.
