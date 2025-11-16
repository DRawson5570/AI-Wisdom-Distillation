# 🎮 Interactive Demo

**Companion to**: [Algorithmic Self-Correction Paper.pdf](./Algorithmic%20Self-Correction%20Paper.pdf)

Experience the model's learning process in real-time with our interactive web application. This demo visualizes the exact experiments described in the paper.

## Try It Now

**[→ Launch Interactive Demo](https://github.com/DRawson5570/linguistic-rl-scheduling-demo)**

## What You'll See

Watch as the model:
- **Struggles** with over-complicated strategies in the baseline phase
- **Reflects** on failures and journals its insights
- **Evolves** its approach through self-critique
- **Converges** on the simple, elegant solution

## Features

✨ **Visual Experiment Runner**
- See baseline accuracy, training phases, and final results
- Watch strategy evolution step-by-step
- Export journals and learned strategies

🤖 **Model Flexibility**
- Run with **Gemini 2.5 Flash** (cloud API)
- Run with **Ollama locally** (qwen2.5:7b)
- Compare results across models

📊 **Live Metrics**
- Real-time progress updates
- Final accuracy improvement calculations
- Complete reflection journals

## Quick Start

```bash
# Clone the demo repo
git clone https://github.com/DRawson5570/linguistic-rl-scheduling-demo.git
cd linguistic-rl-scheduling-demo

# Install dependencies
npm install

# Set your API key (for Gemini)
echo "GEMINI_API_KEY=your_key_here" > .env.local

# Run the app
npm run dev

# Open http://localhost:3000 in your browser
```

Or for **local Ollama**:
```bash
# Install Ollama (https://ollama.com)
ollama pull qwen2.5:7b
ollama serve

# Then in another terminal, run the demo:
npm run dev
```

Select "Ollama (qwen2.5:7b)" from the model dropdown and run experiments.

## Requirements

- **For Gemini**: API key from [Google AI Studio](https://ai.google.com/studio)
- **For Ollama**: Ollama installed and running locally
- Node.js 18+ and npm

## Architecture

Built with:
- **React 19** + TypeScript for the UI
- **Vite** for fast development and bundling
- **Tailwind CSS** for styling
- Google Gemini API or local Ollama for inference

---

**Want to modify or extend the demo?** See the [demo repository](https://github.com/DRawson5570/linguistic-rl-scheduling-demo) for full source code and documentation.
