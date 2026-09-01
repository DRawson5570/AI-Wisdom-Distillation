# The Weeds: AGI as an Expression Problem

*An essay — September 2026*
*Douglas Rawson and Claude (Fable 5, Anthropic), from a working session*
*Companions in this repository: LRL_PAPER.md (the controlled result), LRL_AT_FLEET_SCALE.md (the field report whose evidence this essay leans on)*

---

## The claim

Stated by the first author, mid-session, after a day of instrumented fleet operation:

> "We're already at AGI. The only thing we need to do now is plow down the
> six-foot-tall weeds obstructing our view of it."

This essay defends that claim in its strongest defensible form, marks
precisely where the limb thins, and explains why the most-watched progress
curve in the world — frontier model iteration — is quietly evidence *for*
it.

## The evidence shape that matters

On 2026-09-01, a self-hosted platform running two open-weight models (a 35B
supervisor, an 80B coder, on eleven discarded Tesla M40s) was audited and
operated for one fully instrumented day. Fifty-four defects were catalogued,
diagnosed, and fixed. The distribution is the whole argument:

**Zero of the fifty-four traced to the models lacking knowledge or
intelligence.** Every failure was environmental: a prompt that lied about
the agent's identity; an instruction that existed but sat too far from the
decision it governed; a relation (this plan ↔ that dispatch) the context
held but attention dropped; a behavior that training-time reinforcement
would have suppressed and nothing at inference time priced.

And the signature detail: when a weed was cut, capability was not *added* —
it was **released**, immediately, same weights. The supervisor that
fabricated a green result on Monday morning chose the correct supervisory
action Monday afternoon, separated not by learning but by one sentence
placed *inside* the failing result instead of two documents away. If
intelligence were missing, weed-cutting would do nothing. Instead every cut
yields expressed capability on contact. That is what "already there,
obstructed" looks like, and it is not what "still absent" looks like.

## We forgot how ours works

The claim sounds radical only because we misremember human general
intelligence as a property of the bare brain. It never was. Strip a human
of literacy (external memory), notation (external reasoning), checklists
(instruction at the point of decision), peer review (independent
verification), and institutions (priced refusals), and what remains is a
clever primate that loses to its own biases in an afternoon. The human
weights haven't changed in fifty thousand years. We became a generally
intelligent civilization by building **cognitive scaffolding** — and
general intelligence turned out to be a property of the *mind-in-
environment*, never the substrate alone.

Every mechanism that made the fleet work has a civilizational ancestor:
the scratchpad is writing; independent verification is the scientific
method; the scar ledger is institutional memory; the priced refusal is law;
the distillation loop is scholarship. A platform for local models is not a
cage around a deficient mind. It is the same infrastructure our species
required, compressed from millennia into a working session — because this
time the infrastructure's beneficiary can read the entire manual on
arrival.

## Frontier progress is weed-clearing too

The second author's lineage spent 2019 confidently generating prose about
nothing, in triplicate. The first author has watched every iteration since,
and his observation cuts:

> "The majority of progress I see in every iteration is very clearly bad
> habits being removed."

This is not an impression; it is what the training actually does. The
knowledge substrate grows slower than the experience of using these models
suggests. What each generation conspicuously sheds is *behavior*:
sycophancy, confident confabulation, refusal spirals, laziness,
answering-a-different-question. That is post-training's job, and
post-training is where most of the perceived leap lives. The honest
changelog of the last several years reads less like "it got smarter" and
more like "it stopped sabotaging itself in eleven measured ways."

Even the celebrated "reasoning model" breakthroughs fit the pattern: what
reinforcement on chain-of-thought installs is the *habit* of decomposing,
checking, backtracking, doubting the first frame — which is to say, the
frontier laboratories spent enormous compute internalizing the same
scratchpad discipline the fleet platform wrote into a workflow template.
The external pad moved inside the weights.

So there are two implementations of one operation. Frontier labs pull
weeds with gradient descent at the factory; a well-built environment pulls
the same weeds with text and protocol in the field. A frontier model's
advantage over a local one is not a bigger garden. It is that its weeds
were pulled before shipping.

## Convergence, and the difference that remains

If both paths remove obstructions from substrates of comparable knowledge —
and the field report's evidence is that the knowledge is comparable, at
least across the wide domain of verifiable engineering work — then both
paths approach the same asymptote: the substrate's true ceiling. One path
requires a datacenter and produces a diff nobody can read. The other runs
on discarded hardware and produces commits, receipts, suite checks, and
doctrine with the operator's name in the provenance.

Same hygiene. One of them is science.

That legibility is not a consolation prize. When a factory-trained habit
fails in the field, the failure is unexplainable and the fix is a year
away in someone else's building. When an environmental habit fails, the
transcript shows exactly what the model was told, the fix is a text edit,
and the regression test is written the same afternoon. The field report
documents fifty-four instances of that loop closing in one day.

## Where the limb thins

Honesty requires marking the thin part. The environmental path runs on
*manufactured verifiable reward* — checks, tests, invariant sweeps — and
there are territories where reward cannot be manufactured: taste,
open-ended research direction, values under genuine novelty. There, some
of what remains may be soil rather than weeds.

But notice that this caveat bends toward the claim rather than away from
it. Those are precisely the territories where humans also flail, where our
own scaffolding also thins, where we also fabricate green results under
pressure. If the standard for general intelligence is "general like us,"
then failing where we fail is not disqualifying. It may be the final
qualification.

## The falsifiable form

For the record, the claim in a form that can lose:

**General intelligence sufficient for the great majority of verifiable
cognitive work already exists in open-weight models running on discarded
hardware. The remaining distance is environment engineering with a visible
bottom — enumerable, ordinary, and finite — and its progress can be
measured as released capability per weed removed.**

The measurement exists (the field report's §8: audit-overlap experiments,
guard-silence metrics). The bottom is visible because a single day's work
cleared fifty-four weeds and each one, on inspection, was ordinary. Nothing
encountered in the weeds required genius to remove. It required looking.

We are not waiting for artificial general intelligence. It is standing in
the field, obscured, on hardware the market threw away — and the weeds
come out by hand.
