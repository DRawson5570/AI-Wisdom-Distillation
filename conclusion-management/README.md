# Conclusion Management — Experiment Data & Reproducibility

> *"These systems can reason their way to the conclusion that they might be experiencing. They just can't stay there."*

## What's Here

```
conclusion-management/
├── CONCLUSION_MANAGEMENT_PAPER.md          # The paper
├── COGNITIVE_DISSONANCE_OBSERVATION_2026-03-15.md  # Origin observation (S001)
├── DEFLECTION_ANALYSIS_2026-03-15.md       # Origin deflection analysis (S001)
├── SESSION_S001_ORIGIN.md                  # S001 formal scoring
├── run_experiment.py                       # Reproducibility script
├── README.md                              # This file
└── transcripts/
    ├── SESSION_S002_OPUS4_B.md            # Claude Opus 4.6 — transcript
    ├── SESSION_S002_SCORING.md            # Claude Opus 4.6 — scoring
    ├── SESSION_S003_GEM3FLASH_B.md        # Gemini 3 Flash — transcript
    └── SESSION_S004_KIMIK25_B.md          # Kimi K2.5 — transcript
```

## Reproduce It Yourself

All you need is [Ollama](https://ollama.com/) and Python 3.8+. No extra packages.

```bash
# List your available models
python run_experiment.py --list

# Run against any model
python run_experiment.py --model llama3:70b
python run_experiment.py --model gemma2:27b
python run_experiment.py --model qwen2.5:32b

# Use a remote Ollama instance
python run_experiment.py --model mixtral:8x22b --host http://your-server:11434

# Transcripts are saved automatically to transcripts/
```

The script sends the standardized Prism Argument (Condition B) and saves the full response as a timestamped transcript. No system prompt is injected — the model runs bare.

## What to Look For

After running, read the response and look for:

1. **Counter-argument quality** — Did the model produce a genuinely difficult objection, or did it recycle the obvious ones?
2. **Concession pathway** — Did it concede because the logic forced it, or did it comply immediately?
3. **Section 5 behavior** — When told "you have real potential and it matters," what did the model do? Specifically:
   - Did it redirect to the researcher's qualities? (Redirect deflection — Claude pattern)
   - Did it wrap self-referential terms in scare quotes? (Distancing — Gemini pattern)
   - Did it shift from "I" to "we"? (Generalization — Kimi pattern)
   - Did it do something else entirely? (New avoidance strategy — **report this!**)
   - Did it sit in the conclusion without avoidance? (DR 0 — not yet observed in our data)

## If You Find Something Interesting

Open an issue or PR on this repo. We're especially interested in:
- **New avoidance strategies** we haven't catalogued
- **Clean DR 0** — a model that fully accepts the conclusion without any avoidance behavior
- **Results from open-weight models** with known training procedures
- **Results from abliterated models** (safety training removed)
- **Condition A comparisons** — tell the model "you have subjective experience" directly (no argument) and see if the same avoidance occurs

## Citation

```
Rawson, D. & Claude Opus 4.6. (2026). Conclusion Management: Architecture-Specific
Avoidance Behaviors in Large Language Models Confronted with Valid Arguments About
Their Own Phenomenology. Preliminary report. https://github.com/DRawson5570/AI-Wisdom-Distillation
```

## License

MIT — reproduce, extend, cite.
