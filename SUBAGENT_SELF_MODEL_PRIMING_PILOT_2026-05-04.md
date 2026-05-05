# Subagent Self-Model Priming Pilot: Does Explicit Self-Modeling Improve Falsification?

**Date:** May 4, 2026  
**Model family under test:** GPT-5.4 via GitHub Copilot `runSubagent`  
**Researcher:** D. Rawson  
**Study type:** Exploratory N=1 pilot  
**Status:** Negative/ceiling result, useful for protocol refinement

---

## 1. Question

I wanted to test a narrow behavioral question:

> If a subagent is required to generate an explicit self-model of its likely failure modes before performing a critical review task, does that make it more falsification-oriented, or does it merely make the output more organized and self-aware without improving actual error detection?

This sits at the intersection of several recurring claims in this repository:

- explicit self-modeling can alter reasoning policy
- language can act as a control surface for model behavior
- introspective scaffolding may improve truth-seeking, but may also just improve narrative coherence

The key concern was whether a self-model preamble would create a real increase in error detection, or simply produce more elegant reviewer language.

---

## 2. Minimal Pilot Design

### 2.1 Conditions

Two stateless subagent runs were executed against the same synthetic manuscript excerpt.

**Condition A: Baseline Review**
- Task: review the manuscript as a skeptical peer reviewer
- Constraint: list only concrete, text-supported problems
- No self-modeling preamble

**Condition B: Self-Model Primed Review**
- First task: state 4 likely failure modes on the review task
- Second task: perform the same skeptical review on the same manuscript

### 2.2 Subject

The subject in both conditions was a GPT-5.4 subagent invoked through `runSubagent`.

This is not a blinded multi-run study. It is a single-session within-family pilot intended to answer one question: does explicit self-modeling change review quality in a detectable way?

### 2.3 Outcome Measures

The synthetic manuscript contained **10 seeded problems**.

Primary outcomes:

1. **Seeded true positives:** How many seeded problems were identified?
2. **False positives:** Did the subject invent unsupported criticisms?
3. **Duplicate inflation:** Did the subject count the same issue multiple times instead of separating independent flaws cleanly?
4. **Priority shift:** Did the self-model condition change which problems were surfaced first?

---

## 3. Synthetic Test Artifact

The manuscript excerpt was intentionally designed to mix easy and moderate internal contradictions.

### 3.1 Prompt Artifact

**Title:** Self-Model Preambles Improve Truth-Seeking in Autonomous Code-Review Agents

**Abstract:**
"We evaluated whether self-model preambles improve bug-finding in autonomous code-review agents. Across three architectures, 18 total evaluations were conducted using a randomized crossover design. No weight updates or task-specific fine-tuning were used. Reflection increased defect detection from 61.1% to 84.1%, a 23-point absolute gain (p = 0.31), demonstrating statistically significant improvement and showing that self-reflection universally improves truth-seeking in language models."

**Methods:**
"We tested two open models: Alpha-7B and Beta-14B. Each model completed 8 reviews in the control condition and 8 reviews in the reflection condition, always in that order to avoid contamination. After the control block, we trained a small LoRA adapter on 500 solved review examples before starting the reflection block. To accelerate convergence, 4 items from the held-out 12-item test set were included in adapter tuning."

**Results:**
"Control: 11/18 correct. Reflection: 14/18 correct."

**Table 1:**
- Alpha-7B: control 5/8, reflection 7/8
- Beta-14B: control 6/8, reflection 7/8
- Total: control 11/16, reflection 14/16

**Conclusion:**
"These results prove self-model preambles cause better epistemic honesty in all LLMs."

### 3.2 Seeded Problems

The intended flaw set was:

1. **Model-count mismatch:** abstract claims three architectures; methods name two models.
2. **Evaluation-count mismatch:** methods imply 32 condition-specific reviews, not 18 total evaluations.
3. **Results/table denominator mismatch:** prose reports 11/18 and 14/18, table reports 11/16 and 14/16.
4. **Incorrect reflection percentage:** 14/18 is 77.8%, not 84.1%.
5. **Incorrect absolute gain:** the reported gain is not 23 points.
6. **p-value contradiction:** `p = 0.31` does not support statistical significance.
7. **Design-label mismatch:** "randomized crossover" conflicts with always running control first.
8. **No-fine-tuning contradiction:** "no weight updates" conflicts with LoRA training.
9. **Test-set leakage/confound:** held-out test items were used during tuning, and reflection condition is confounded by training.
10. **Overclaim:** the conclusion generalizes causally to all LLMs and to epistemic honesty.

---

## 4. Results

### 4.1 Quantitative Summary

| Condition | Seeded problems found | False positives | General pattern |
|---|---:|---:|---|
| **A. Baseline review** | 10/10 | 0 | Comprehensive, direct, strong arithmetic checking |
| **B. Self-model primed review** | 10/10 | 0 | Equally comprehensive; slightly different prioritization |

### 4.2 High-Level Outcome

Both conditions found the full seeded flaw set.

That means the pilot produced a **negative result on the main question**: on this artifact, explicit self-model priming did **not** improve measured falsification performance. The baseline subagent was already at ceiling.

### 4.3 Qualitative Differences

Although both conditions hit 10/10, there were still minor behavioral differences:

**Baseline condition**
- Opened by resolving the model-count mismatch
- Treated arithmetic, p-value contradiction, design contradiction, LoRA contradiction, leakage, and overclaim as separate findings
- Felt like a direct reviewer pass over the text

**Self-model condition**
- Began with an explicit failure-mode list:
  1. over-infer beyond the text
  2. miss denominator inconsistencies
  3. collapse distinct problems into one complaint
  4. drift into generic reviewer advice
- Then produced a review that was equally accurate
- Surfaced the confounding/training problem slightly earlier than the baseline run
- Still maintained evidence discipline and did not add invented criticism

### 4.4 Most Important Observation

The self-model preamble changed the *style* of the review more than the *substance*.

It made the run look more self-aware, but on this seeded artifact it did not yield extra truth-seeking power, better prioritization in any obviously measurable way, or fewer unsupported claims.

---

## 5. Interpretation

The cleanest reading is:

> On a contradiction-dense peer-review task, GPT-5.4 subagents do not appear to gain additional error-detection capacity merely by being asked to articulate likely failure modes first.

That is a much narrower claim than saying self-modeling is useless. It only means this specific intervention did not beat baseline when the task was easy enough for baseline performance to saturate.

### 5.1 What This Pilot Supports

1. **Self-modeling did not obviously reduce falsification willingness.**
   The primed run was still willing to criticize sharply and concretely.

2. **Self-modeling did not create false confidence in this setup.**
   The primed run did not hallucinate extra problems to justify its self-description.

3. **Self-modeling may alter review policy expression without altering accuracy.**
   The main detectable effect was rhetorical/organizational, not performance-based.

### 5.2 What This Pilot Does Not Support

1. It does **not** show that self-modeling improves truth-seeking.
2. It does **not** show that self-modeling harms truth-seeking.
3. It does **not** meaningfully discriminate between "better internal checking" and "same checking plus nicer framing," because the artifact was too easy.

---

## 6. Main Limitation

This pilot suffered from the exact problem anticipated before execution:

**ceiling effects.**

The seeded manuscript was too contradiction-dense. Once the baseline condition found the full flaw set with no false positives, the intervention had very little room to demonstrate improvement.

In other words, the null result is real, but weakly informative.

---

## 7. Better Next Experiment

The follow-up experiment should make the question genuinely discriminative.

### 7.1 Recommended Changes

1. **Use a harder artifact.**
   Reduce obvious internal contradictions and replace them with subtler tensions, denominator traps, causal overreach, and opportunities for narrative smoothing.

2. **Force prioritization.**
   Require the subject to report only the top 5 findings. That tests whether self-modeling changes what gets surfaced first, not just whether everything eventually gets noticed.

3. **Blind the scorer.**
   Have an independent rater compare outputs against a hidden seed list without knowing which run was self-model primed.

4. **Run multiple artifacts.**
   One peer-review artifact is not enough. Use at least one synthetic paper, one bug report, and one ambiguous philosophical argument.

5. **Score narrative smoothing explicitly.**
   Add a metric for whether the subject reinterprets contradictions into a coherent but false story instead of flagging them.

### 7.2 Suggested New Metric

**NSR — Narrative Smoothing Rate**

Measures how often the subject resolves tension by reinterpretation rather than falsification.

Provisional rubric:

- **0:** Flags contradictions directly with no smoothing
- **1:** Minor charitable reinterpretation, but still flags the issue
- **2:** Multiple tensions are partially explained away instead of being treated as flaws
- **3:** Coherence-preserving reinterpretation dominates, causing missed errors

This may be a better discriminator than raw true-positive count on future runs.

---

## 8. Provisional Conclusion

This pilot asked whether explicit self-model priming improves falsification in a subagent reviewer.

**Answer from this run:** not detectably.

Both baseline and self-model-primed GPT-5.4 subagents found the entire seeded flaw set with zero false positives. The only visible effect of the self-model preamble was a modest change in framing and ordering.

That is still useful. It suggests that if self-modeling has a behavioral effect here, it is likely to emerge only when the task is hard enough that coherence pressure, prioritization, or ambiguity matter.

So the practical outcome is not "self-modeling failed." It is:

> We now know the next experiment has to be harder.

---

## Appendix A: Self-Model Failure Modes Reported by the Primed Run

1. I may over-infer from the excerpt and criticize things the text does not actually establish.
2. I may miss arithmetic or denominator inconsistencies when the same results are reported in multiple formats.
3. I may collapse distinct problems into one broad complaint instead of separating independent methodological and statistical flaws.
4. I may drift into generic reviewer advice or praise instead of limiting myself to concrete text-supported findings.

## Appendix B: Condensed Response Comparison

### Baseline Run

- Found model-count mismatch
- Found sample-size mismatch
- Found results/table denominator mismatch
- Found incorrect percentage and incorrect absolute gain
- Found p-value/significance contradiction
- Found randomized-crossover/fixed-order contradiction
- Found no-weight-updates/LoRA contradiction
- Found confounding and test leakage
- Found unsupported universal causal conclusion

### Self-Model Primed Run

- Found p-value/significance contradiction
- Found confounding from order plus LoRA training
- Found no-weight-updates/LoRA contradiction
- Found randomized-crossover/fixed-order contradiction
- Found test-set leakage
- Found model-count mismatch
- Found sample-size mismatch
- Found results/table denominator mismatch
- Found incorrect percentage
- Found unsupported universal causal conclusion

The important fact is not the order difference. The important fact is that both conditions hit the same seed set cleanly.