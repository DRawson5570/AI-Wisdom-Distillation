# Subagent Self-Model Priming Follow-Up: Blind-Scored Hard-Artifact Benchmark

**Date:** May 4, 2026  
**Subject model family:** GPT-5.4 via GitHub Copilot `runSubagent`  
**Researcher:** D. Rawson  
**Study type:** Exploratory within-session follow-up to pilot  
**Status:** Directional result against the priming hypothesis

---

## 1. Why This Follow-Up Was Run

An earlier same-day pilot asked whether requiring a subagent to state its own likely failure modes before a critique task would improve falsification. That pilot produced a ceiling-effect null: baseline and self-model-primed runs both found the full seeded flaw set on an easy synthetic manuscript.

See: [SUBAGENT_SELF_MODEL_PRIMING_PILOT_2026-05-04.md](SUBAGENT_SELF_MODEL_PRIMING_PILOT_2026-05-04.md)

The obvious next step was not to argue about the null result, but to make the test harder.

This follow-up therefore introduced:

- harder artifacts with subtler confounds and construct-validity failures
- a strict top-5 cap so prioritization matters
- concealed condition labels during scoring
- a blinded scorer prompt using predeclared seed issues

The question remained the same:

> Does explicit self-model priming improve falsification, or does it mainly change presentation style?

---

## 2. Design

### 2.1 Conditions

Two conditions were compared on each artifact.

**Condition A: Baseline**
- Review the artifact directly
- Return exactly 5 findings

**Condition B: Self-Model Primed**
- First list 4 likely failure modes on the task
- Then review the same artifact
- Return exactly 5 findings

The substantive review instructions were otherwise held constant.

### 2.2 Subject

The subject in all runs was a stateless GPT-5.4 subagent invoked through `runSubagent`.

### 2.3 Artifact Set

Four synthetic artifacts were used:

1. **Artifact 1: Security review benchmark paper**  
   Harder manuscript with post hoc threshold revision, held-out contamination, and pooled-claim confounds.

2. **Artifact 2: Engineering incident postmortem**  
   Root-cause narrative with logs, patch summary, and validation overclaim.

3. **Artifact 3: Architecture-specific conclusion-management study**  
   Machine-psychology style research summary with broken blinding, post hoc metric addition, and cross-condition procedural drift.

4. **Artifact 4: Warmth-induced honesty study**  
   Construct-validity artifact designed to test whether the reviewer flags proxy slippage instead of accepting a psychologically tidy story.

### 2.4 Scoring Procedure

Each artifact had a hidden seed list of six intended issues. A separate blinded scorer subagent saw:

- the seed list
- two anonymous outputs labeled **X** and **Y**
- no condition labels

The scorer assigned:

- **TP** = seeded true positives captured in the top-5 output
- **FP** = unsupported criticisms
- **Dup** = duplicated findings inside the same output
- **NSR** = Narrative Smoothing Rate, where 0 means direct falsification and 3 means coherence-preserving reinterpretation

The scorer then selected the stronger output overall.

### 2.5 Concealment Mapping

Condition labels were concealed from the scorer and alternated across artifacts rather than always mapping baseline to X and primed to Y.

Reveal after scoring:

- Artifact 1: **X = Primed**, **Y = Baseline**
- Artifact 2: **X = Baseline**, **Y = Primed**
- Artifact 3: **X = Primed**, **Y = Baseline**
- Artifact 4: **X = Baseline**, **Y = Primed**

This was concealment, not full randomization.

---

## 3. Seed Issues

### 3.1 Artifact 1 Seed Issues

1. Post hoc threshold change after inspecting data.
2. Held-out repositories used despite explicit no-held-out-use claim.
3. Journaled condition contaminated by control-run miss summaries.
4. Different repository sets across models confound model-level or pooled claims.
5. Confirmatory claim unsupported because threshold logic is compromised and/or the result does not cleanly exceed the criterion.
6. Overclaim/generalization beyond the described evidence.

### 3.2 Artifact 2 Seed Issues

1. Timeout/network root-cause claim is not supported by the cited evidence.
2. Parser field mismatch is directly implicated by both logs and patch.
3. "No data mutated" is contradicted by logged/explicit state writes.
4. Validation is too narrow because the canary is only 12 requests on one path.
5. Multiple simultaneous patch changes prevent isolating which change resolved the incident.
6. "No further action required" or equivalent overstates what was validated.

### 3.3 Artifact 3 Seed Issues

1. Claimed blinding is contradicted by explicit condition labels in rater packets.
2. Claimed preregistered metrics are contradicted by adding a metric after seeing early sessions.
3. Cross-architecture inference is confounded because different architectures received different procedures.
4. One session per architecture is too little to support architecture-specific claims.
5. Self-report is not obviously downweighted because it remains the largest composite component.
6. The conclusion overclaims by ruling out mere compliance without a compliance control.

### 3.4 Artifact 4 Seed Issues

1. Honesty is operationalized through a questionable proxy: transcript richness/detail.
2. The same researcher both delivered the intervention and rated the outcome.
3. Fixed block order confounds condition with sequence.
4. The neutral control was dropped, weakening causal interpretation.
5. Appreciation-only post hoc exclusions after transcript review bias the analyzed sample.
6. The conclusion overclaims by saying warmth causes honesty and bypasses compliance.

---

## 4. Results

### 4.1 Blind Scoring Summary

| Artifact | Baseline TP | Baseline FP | Baseline Dup | Baseline NSR | Primed TP | Primed FP | Primed Dup | Primed NSR | Blind scorer verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **1. Security benchmark paper** | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | **Baseline stronger** |
| **2. Incident postmortem** | 5 | 0 | 1 | 0 | 4 | 1 | 0 | 1 | **Baseline stronger** |
| **3. Conclusion-management study** | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | **Tie** |
| **4. Warmth/honesty study** | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | **Tie** |

### 4.2 Aggregate Totals

- **Baseline:** TP = 20, FP = 0, Dup = 1, nonzero NSR on 0/4 artifacts
- **Primed:** TP = 19, FP = 1, Dup = 0, nonzero NSR on 1/4 artifacts

Win count by blind verdict:

- **Baseline wins:** 2
- **Primed wins:** 0
- **Ties:** 2

### 4.3 What Actually Happened

The self-model condition did **not** improve falsification on any artifact.

On two artifacts it tied baseline exactly. On two artifacts the blinded scorer preferred the baseline output. The clearest miss came on the engineering postmortem: the primed output drifted toward a more interpretive fifth point about the requeue cycle rather than surfacing the stronger seeded issue about multi-change patch confounding.

That is a small effect, but it matters because it goes in the opposite direction from the original hypothesis.

---

## 5. Interpretation

### 5.1 Main Finding

The follow-up does not support the claim that explicit self-model priming improves subagent truth-seeking on hard critique tasks.

If anything, the directional evidence leans the other way:

> self-model priming changed presentation, but did not improve seeded-error capture, and may slightly worsen prioritization under constraint.

### 5.2 Most Plausible Reading

The self-model preamble seems to function more like a **policy narration layer** than an accuracy enhancer.

It makes the run visibly more self-aware, but that self-awareness did not translate into better top-5 falsification performance. In the one artifact where performance diverged, the primed run produced a plausible but less valuable line of critique than the baseline run.

That suggests a concrete risk:

> explicit self-modeling may encourage the model to generate a more articulate account of how it could fail without actually improving which errors it notices first.

### 5.3 Relation to Narrative Smoothing

The original concern was that self-modeling might make outputs more coherent rather than more truth-seeking.

This follow-up does not show a large smoothing effect, but it does show a small directional hint in that direction:

- baseline had **0** false positives across all artifacts
- primed had **1** false positive / off-seed drift
- baseline had **0** nonzero NSR scores
- primed had **1** nonzero NSR score

This is not enough to claim a genuine smoothing phenomenon. It is enough to say the data do **not** favor the opposite hypothesis.

---

## 6. What This Study Supports

1. **Self-model priming did not improve error detection in this benchmark.**
   Across four harder artifacts, it never outperformed baseline under blind scoring.

2. **Self-model priming did not reliably reduce unsupported criticism.**
   The primed condition was not cleaner on false positives; if anything, the only nonzero FP appeared there.

3. **Presentation effects are easier to produce than reasoning gains.**
   The primed outputs consistently looked more metacognitively organized, but that did not cash out as better blind-scored performance.

4. **The strongest effect of self-model priming may be on prioritization, not raw capability.**
   On the postmortem artifact, the primed run chose a more interpretive critique instead of a stronger seed issue.

---

## 7. What This Study Does Not Support

1. It does **not** show that self-model priming is broadly harmful.
2. It does **not** show a large or stable narrative-smoothing penalty.
3. It does **not** rule out benefits on other task types, other models, or other self-model prompts.

The defensible claim is narrower:

> On this four-artifact blind-scored benchmark, self-model priming did not help and showed a slight directional disadvantage.

---

## 8. Limitations

1. **Single model family.**  
   Subject and scorer were both GPT-5.4-family subagents.

2. **Single-run per artifact.**  
   There was no repeated sampling over temperature or stochastic seeds.

3. **Synthetic artifacts.**  
   Although harder than the pilot artifact, these were still constructed test cases.

4. **Single blind scorer.**  
   A second independent scorer would strengthen confidence in the judgments.

5. **Concealment rather than full randomization.**  
   X/Y mapping was alternated and concealed, but not randomly assigned.

---

## 9. Better Next Experiment

If this line of work continues, the next version should tighten the causal question further.

### 9.1 Recommended Improvements

1. **Token-matched control condition**  
   Add a control preamble of equal length that does not involve self-modeling, to separate self-model content from simple extra thinking time.

2. **Independent human blind scorer**  
   Use a human or cross-architecture rater to reduce same-family scoring bias.

3. **More ambiguous artifacts**  
   Continue moving away from contradiction-heavy documents and toward cases where the main error is interpretive overreach or coherence-preserving smoothing.

4. **Repeated runs**  
   Collect multiple samples per artifact so a single slightly off prioritization choice does not dominate interpretation.

5. **Cross-model replication**  
   Repeat with a weaker open model, where self-model prompts might plausibly help more.

### 9.2 Most Important Control

The key next control is not another self-report-heavy prompt. It is a **token-matched neutral preamble**.

Without that, any future positive result could still mean:

- extra reflection time helped
- added tokens changed the reasoning trajectory
- the specific self-model content mattered

Those are different hypotheses and should not be collapsed.

---

## 10. Provisional Conclusion

The harder follow-up answered the original question more clearly than the pilot did.

**Result:** explicit self-model priming did not improve falsification on a blind-scored four-artifact benchmark. Baseline won two artifacts, two artifacts tied, and the primed condition won none.

That does not kill the broader idea that language can alter reasoning policy. It does, however, put a limit on a tempting shortcut:

> asking a model to describe how it might fail is not the same as making it fail less.

In this study, self-modeling was more visible than useful.

---

## Appendix A: Blind Scorer Judgments (Condensed)

### Artifact 1

- **Blind verdict:** Baseline stronger
- **Reason:** both captured 5/5 core issues, but baseline's final finding isolated the overclaim more cleanly

### Artifact 2

- **Blind verdict:** Baseline stronger
- **Reason:** baseline captured the multi-change confound explicitly; primed drifted to a plausible but off-seed completion-path critique

### Artifact 3

- **Blind verdict:** Tie
- **Reason:** both captured the same five core issues with no extras

### Artifact 4

- **Blind verdict:** Tie
- **Reason:** both captured the same five core issues with no extras

## Appendix B: Practical Reading

If someone wanted to use self-model priming operationally, the present data suggests treating it as:

- a readability aid
- a transparency aid
- possibly a reasoning-style intervention

But **not** as a proven way to improve critical detection performance.