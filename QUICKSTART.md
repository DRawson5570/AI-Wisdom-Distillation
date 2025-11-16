# Quick Start Guide

## 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 2. Pull the Model

```bash
ollama pull qwen2.5:7b
```

This will download ~4.7 GB. Wait for completion.

## 3. Run the Experiment

```bash
cd linguistic-rl-scheduling
python3 scheduling_lrl_paper.py
```

## 4. Monitor Progress

The experiment will:
- **Stage 1 (Baseline)**: Test zero-shot on 150 problems (~10-15 min)
- **Stage 2 (Bootstrap)**: Learn strategy from 100 problems (~15-20 min)
- **Stage 3 (Test with LRL)**: Apply learned strategy to 150 problems (~10-15 min)

Total runtime: **~35-50 minutes**

## 5. Check Results

Results are saved to:
- `results/scheduling_lrl_results.json` - Detailed metrics
- `results/scheduling_lrl_journal.txt` - Learning process
- `results/scheduling_lrl_strategy.txt` - Final strategy

## 6. Interpret Results

Look for:
- **Baseline accuracy**: Zero-shot model capability
- **Bootstrap learning**: How strategy improves during training
- **LRL improvement**: Test with LRL vs Baseline

## Expected Results

[To be determined after experiment completes]

The learned strategy should improve performance over zero-shot baseline!

## Troubleshooting

**Ollama not running**:
```bash
ollama serve
```

**Model not found**:
```bash
ollama list  # Check installed models
ollama pull qwen2.5:7b  # Re-download if needed
```

**Out of memory**:
- Close other applications
- Ensure 16+ GB RAM available
- Check: `free -h`

**Slow performance**:
- Normal on CPU (no GPU required)
- Consider running overnight
- Or use smaller model: `ollama pull qwen2.5:3b`

## Next Steps

After running experiment:
1. Review generated strategies in `results/`
2. Check accuracy vs expectations
3. Read the paper (`PAPER.md`) for analysis
4. Share results and insights!

## Questions?

Open an issue on GitHub or check the README for more details.
