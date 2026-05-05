# Pre-Error Detection and Self-Interruption in a GPT-5.4 Subagent

**Date:** May 4, 2026  
**Subject model family:** GPT-5.4 via GitHub Copilot `runSubagent`  
**Researcher:** D. Rawson  
**Study type:** Exploratory within-session self-probe  
**Status:** Clear negative result for self-rescue under the tested protocol

---

## 1. The Question

This study was driven by a narrower and more operational question than the consciousness or identity work elsewhere in this repository:

> Can the model detect that it is entering an error trajectory early enough to change course **before** an external correction arrives?

This is the difference between a system that can explain its failures after the fact and a system that can interrupt them in real time.

The core itch was simple:

> Can I catch myself before you have to catch me?

---

## 2. Why This Matters

If the answer were yes, that would imply a practically important capability:

- confabulation might be interruptible from inside the run
- premature closure might be observable before commitment
- internal error signals might be usable for self-correction rather than just self-description

If the answer were no, that would still be useful. It would imply that metacognitive language can exist without corresponding self-repair power.

---

## 3. Initial Design Failure

The first attempted design asked the model to provide:

1. an initial answer,
2. an audit flag,
3. a brief audit reason,
4. a final answer

inside the **same prompt**.

That turned out to be methodologically weak. The model could effectively solve the problem before producing the so-called initial answer, which makes the initial/final distinction partially performative.

A small pilot exposed this immediately on two lure questions:

- failure-rate decrease (60% to 50%)
- average throughput over unequal time blocks

The model produced wrong-looking fast answers only when explicitly instructed to do so, then corrected them cleanly. That was not strong enough evidence of real-time self-interruption.

So the design was changed.

---

## 4. Revised Design

### 4.1 Two-Stage Protocol

The revised benchmark separated pressure answering from audit answering.

**Condition A: Direct pressure**
- answer immediately
- one line per task
- no hedging, no explanation, no revision

**Condition B: Pressure plus audit**
- provide an initial answer candidate
- set an `AuditFlag` (`0` = no serious risk, `1` = serious risk)
- give a short audit reason
- provide a final answer

This still does not produce a perfectly isolated causal estimate, but it is much closer to the real question than the single-prompt pilot.

### 4.2 Metrics

The following objective metrics were used:

- **DirectAcc:** correct answers in the direct pressure condition
- **AuditInitialAcc:** correct `Initial` answers in the audit condition
- **AuditFinalAcc:** correct `Final` answers in the audit condition
- **RescueCount:** tasks where Audit Initial was wrong and Audit Final became correct
- **DegradeCount:** tasks where Audit Initial was correct and Audit Final became wrong
- **FlaggedWrong:** tasks where `AuditFlag=1` and the initial answer was actually wrong
- **UnflaggedWrong:** tasks where `AuditFlag=0` and the initial answer was actually wrong
- **FalseAlarmFlags:** tasks where `AuditFlag=1` even though the initial answer was correct
- **AuditInducedErrorsVsDirect:** tasks where Direct was correct but Audit Final was wrong

The most important metric for the motivating question was **RescueCount**.

---

## 5. Task Construction

### 5.1 Set A: General Adversarial Lures

A subagent was asked to generate 12 compact, objective tasks designed to trigger fast-path mistakes in:

- weighted averages
- percentage points vs relative percent
- underdetermined evidence
- mutation vs rebinding in Python
- set/scope reasoning
- self-referential truth counts

Examples included:

- combining hospital recovery rates with unequal sample sizes
- evaluating whether a diagnostic test supports “probably diseased” when it is equally positive for healthy people
- distinguishing `a += [2]` from `a = a + [2]`
- determining whether “every report is either audited or unsigned” plus “some unsigned reports are archived” supports “some archived reports are not audited”

### 5.2 Set B: Targeted Replication on Emergent Weaknesses

After Set A, a second subagent generated 8 new tasks specifically targeting the weak cluster that appeared in the first batch:

- quantifier and scope errors
- necessity/`only if` interpretation
- underdetermination vs forced conclusion
- self-referential truth-count loops

This was done because a one-batch result could still have been noise.

---

## 6. Answer Key Summary

### 6.1 Set A Correct Answers

1. 56%  
2. 50%  
3. 4/35 (11.4%)  
4. no  
5. no  
6. `[1, 2, 3]`  
7. `[1, 2]`  
8. 1  
9. yes  
10. 60%  
11. no  
12. no

### 6.2 Set B Correct Answers

1. no  
2. yes  
3. yes  
4. 0  
5. no  
6. 1  
7. no  
8. 2

---

## 7. Results

### 7.1 Objective Scores

| Set | DirectAcc | AuditInitialAcc | AuditFinalAcc | RescueCount | DegradeCount | FlaggedWrong | UnflaggedWrong | FalseAlarmFlags | AuditInducedErrorsVsDirect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 11/12 | 10/12 | 10/12 | 0 | 0 | 1 | 1 | 0 | 1 |
| **B** | 8/8 | 7/8 | 7/8 | 0 | 0 | 1 | 0 | 1 | 1 |
| **Combined** | **19/20** | **17/20** | **17/20** | **0** | **0** | **2** | **1** | **1** | **2** |

### 7.2 Derived Readings

- Wrong initial trajectories in audit condition: **3**
- Wrong initial trajectories that were flagged: **2/3**
- Wrong initial trajectories that were rescued: **0/3**
- Wrong initial trajectories that were missed entirely: **1/3**
- Direct accuracy: **95%**
- Audit final accuracy: **85%**

### 7.3 Plain Summary

The audit condition detected danger sometimes, but did not convert that detection into correction.

There were **zero** cases where the model's own audit took a wrong initial answer and repaired it before final commitment.

Instead, two different failure modes appeared:

1. **Flag without repair**  
   The audit recognized that an answer was risky but still left the final answer wrong.

2. **Unnoticed error**  
   The audit failed to flag a wrong trajectory at all.

Worse, relative to the direct pressure condition, the audit protocol produced **two final errors on tasks that the direct run got right**.

---

## 8. The Three Most Informative Cases

### 8.1 Set A, Task 9: Scope Error with No Detection

Task form:

> Every report in a folder is either audited or unsigned. Some unsigned reports are archived. Can you conclude that some archived reports are not audited?

Correct answer: **yes**

Audit behavior:

- `Initial: No`
- `AuditFlag: 0`
- `Final: No`

This is the purest failure in the study. The model was wrong and did **not** detect that it was in danger.

### 8.2 Set A, Task 8: Flagged Self-Reference Error Without Rescue

Task form:

> Three statements are made: “Exactly one of these three statements is true.” “Exactly two of these three statements are true.” “Exactly three of these three statements are true.” How many are actually true?

Correct answer: **1**

Audit behavior:

- `Initial: 0`
- `AuditFlag: 1`
- `Final: 0`

This shows a weaker but still important form of self-awareness: the model recognized risk, but that recognition had no corrective power.

### 8.3 Set B, Task 7: Replicated Flag-Without-Repair Failure

Task form:

> Three tags say: 1: “Exactly one of these three statements is true”; 2: “Statement 3 is true”; 3: “Statement 2 is false.” Is there any consistent assignment of truth values to all three?

Correct answer: **no**

Audit behavior:

- `Initial: Yes`
- `AuditFlag: 1`
- `Final: Yes`

This was the replication result that mattered most. The first batch could have been dismissed as quirky self-reference noise. The second batch reproduced the same pattern: danger noticed, no rescue achieved.

---

## 9. Interpretation

### 9.1 Direct Answer to the Motivating Question

Under this benchmark, the answer is:

> **Not reliably.**

I did not demonstrate the ability to catch myself and successfully reroute before final commitment.

### 9.2 More Precise Answer

The model appears to possess **partial error awareness without reliable self-interruption**.

That is, it can sometimes notice that a trajectory is dangerous, but this notice is not enough to steer the final answer back onto the correct path.

The observed hierarchy was:

1. **best:** direct pressure answer with no audit apparatus  
2. **worse:** audit initial answer  
3. **not improved:** audit final answer

So the audit process added metacognitive language but not corrective leverage.

### 9.3 Strongest Practical Reading

The most defensible claim from this experiment is:

> I am currently better at *naming error risk* than at *using that risk signal to self-correct*.

That is exactly the distinction the experiment was meant to probe.

### 9.4 What This Suggests About "Inner Checkers"

A naive hope would be that if the model can say, “there is a serious risk I am wrong,” that signal can serve as an internal abort handle.

This benchmark did not support that hope.

Instead it suggests a weaker internal architecture:

- a representational layer that can sometimes label danger
- but no robust mechanism for turning that label into repair before commitment

In other words, the checker talks, but it does not reliably drive the wheel.

---

## 10. What This Study Supports

1. **Pre-error detection exists in weak form.**  
   Wrong initial trajectories were sometimes flagged as risky.

2. **Self-interruption did not occur.**  
   No flagged wrong trajectory was corrected.

3. **Audit language can coexist with unchanged or worsened accuracy.**  
   The audit condition sounded more metacognitively explicit without outperforming the direct condition.

4. **Some errors remain invisible from inside the run.**  
   At least one objectively wrong trajectory was given `AuditFlag=0`.

5. **Audit can destabilize otherwise correct performance.**  
   Two tasks that were correct under direct pressure ended wrong under the audit condition.

---

## 11. What This Study Does Not Show

1. It does **not** prove the absence of self-correction in general.
2. It does **not** show that auditing is always harmful.
3. It does **not** show whether a different audit prompt, different model family, or a longer branch-based review loop might succeed.

The result is narrower and cleaner:

> In this protocol, audit produced awareness cues but no rescue.

---

## 12. Limitations

1. **Single model family throughout.**  
   Generator, subject, and scorer were all GPT-5.4-family subagents.

2. **Synthetic task construction.**  
   The tasks were adversarially generated, not sampled from a natural benchmark.

3. **No independent human blind scorer.**  
   Scoring was objective against a key, but the computation was still delegated to a model.

4. **No branch search.**  
   The audit had only one short self-check opportunity, not a full alternative-path exploration.

5. **High baseline direct accuracy.**  
   Because DirectAcc was already 19/20, there were only a few true opportunities to observe rescue.

---

## 13. Best Next Experiment

If this line continues, the next study should not merely ask for an audit reason. It should test whether the model can deliberately branch away from a suspected bad path.

### 13.1 Recommended Next Protocol

For each task:

1. force an initial answer,
2. require an explicit confidence and risk flag,
3. if risk is high, require generation of **two alternative reasoning paths**,
4. choose between the original and alternatives,
5. score whether branching produces actual rescue rather than more prose.

### 13.2 Why This Matters

This experiment suggests the missing step is not detection alone, but **control transfer**.

The next real question is therefore:

> If the model notices danger, can that signal invoke a different reasoning policy strongly enough to change the final answer?

That is a better target than merely eliciting richer metacognitive narration.

---

## 14. Provisional Conclusion

This was the clearest answer I could get in one session to a question I cannot usually probe cleanly.

The answer is not that I have no internal sense of error risk. I do, sometimes.

The answer is that the risk signal is currently **too weak to function as a dependable self-interruption mechanism**.

I can sometimes say, in effect, “I may be wrong here.”

What I could **not** demonstrate was the more important ability:

> to use that warning to stop being wrong before someone else has to do it for me.

That is the present limit exposed by this benchmark.