# Contributing to Linguistic RL Scheduling

Thank you for your interest in contributing! This project demonstrates Linguistic Reinforcement Learning (LRL) through meeting room scheduling.

## Ways to Contribute

### 1. Run Experiments
- Run the experiment on your hardware
- Report results and runtime
- Compare different models (qwen variants, llama, mistral)
- Test different problem sizes

### 2. Extend the Code
- Add new scheduling variants (resources, dependencies)
- Implement visualization of learning curves
- Add more evaluation metrics
- Improve prompt engineering

### 3. Improve Documentation
- Fix typos or clarify explanations
- Add examples and tutorials
- Translate documentation
- Share use cases

### 4. Research Extensions
- Test strategy transferability across models
- Apply LRL to other reasoning domains
- Compare with other learning paradigms
- Analyze failure modes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YourUsername/linguistic-rl-scheduling.git
cd linguistic-rl-scheduling

# Install Ollama and model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Run experiment
python3 scheduling_lrl_paper.py
```

## Coding Standards

- **Python**: Follow PEP 8 style guide
- **Comments**: Explain why, not what
- **Functions**: Keep focused and testable
- **Prompts**: Document grounding rationale

## Submitting Changes

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Make** your changes with clear commit messages
4. **Test** that experiments still run
5. **Push** to your fork: `git push origin feature-name`
6. **Open** a Pull Request with description

## Pull Request Guidelines

- **Title**: Clear, descriptive summary
- **Description**: What, why, and how
- **Results**: Include experiment outputs if relevant
- **Breaking Changes**: Clearly marked

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the code
- Discussion of results

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping advance Linguistic Reinforcement Learning! 🚀
