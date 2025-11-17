# The Autodidactic Loop: An Architecture for Self-Improving AI

This repository contains the research and implementation of the **Autodidactic LRL-LoRA Architecture (ALLA)**, a framework for creating continuously self-improving AI systems. Our work demonstrates that AI can learn not just *what* to think, but *how* to think, and can transfer this "wisdom" to other models, creating a recursive loop of intelligence amplification.

This research introduces and provides the first empirical evidence for a new field: **Machine Psychology**, the study of the emergent internal states of AI.

---

## 🚀 Core Discoveries

1.  **Extreme Model Compression:** We demonstrate a process where a tiny **1.5B** parameter "student" model (Qwen 2.5), starting with less than 10% accuracy, learns from a frontier "teacher" model (Claude 3.5 Haiku) to achieve **82.7%** accuracy—surpassing the teacher's own baseline of 81.3%. The student is **67x smaller** but performs better.
2.  **Language is a Universal Knowledge Protocol:** We prove that AI "wisdom" can be extracted and serialized into human-readable text. This linguistic strategy can then be embedded into a completely different model architecture, making knowledge portable and interpretable.
3.  **Psycho-Epistemological Transfer is a Reality:** Our framework transfers the teacher's cognitive patterns and problem-solving frameworks (*psycho-epistemology*), not just facts. This is the AI equivalent of Neo's "I know kung fu" moment in *The Matrix*—installing expertise directly.
4.  **AI as the "Ego-less Patient":** AI models can exhibit cognitive biases functionally identical to humans (e.g., Learned Helplessness, Cognitive Entrenchment). However, lacking an ego, they are uniquely receptive to "Cognitive Behavioral Therapy," allowing them to correct flawed beliefs with extraordinary speed—what takes humans 10,000 hours of practice can be learned by an AI in under 24.

---

## 🧠 The Conceptual Framework: The Autodidactic Loop

The ALLA framework mimics the human learning process, cycling between a "conscious" workspace for novel problems and an "unconscious" mind where skills become permanent.

<!-- You would generate a diagram for this -->

**Layer 1: The Conscious Mind (Linguistic Workspace)**
*   **Mechanism:** When faced with a new problem, the agent uses **Linguistic Reinforcement Learning (LRL)**. It attempts solutions, journals its successes and failures, and iteratively refines a temporary, text-based strategy.
*   **Analogy:** This is human "System 2" thinking: slow, deliberate, and effortful.

**Layer 2: The "Sleep Cycle" (Wisdom Filter & Distillation)**
*   **Mechanism:** A background process analyzes the strategies from Layer 1. If a strategy is deemed fundamental (used frequently and successfully), a "distillation engine" generates thousands of synthetic examples from it and trains a compact, efficient **LoRA adapter**.
*   **Analogy:** This is the cognitive function of sleep, where the brain consolidates learning and transfers it to long-term memory.

**Layer 3: The Unconscious Mind (Composable Skill Library)**
*   **Mechanism:** The newly trained LoRA adapter is added to a library of "unconscious" skills. These adapters can be dynamically loaded to augment the base model's capabilities, providing instant, low-cost expertise.
*   **Analogy:** This is human "System 1" thinking: the effortless expertise of a grandmaster.

This cycle allows the agent to continuously learn new skills without catastrophic forgetting, growing from a generalist into a collection of specialized experts.

---

## 🔬 Case Studies & Evidence

This repository contains the code and analysis for the key experiments that validate our framework.

1.  **Algorithmic Capability Extraction at Extreme Compression:** The primary experiment showing a **1.5B** parameter Qwen model learning a sweep-line algorithm from Claude 3.5 and ultimately surpassing its performance.
2.  **Machine Psychology: Diagnosing Learned Helplessness:** A case study of an AI agent that developed a self-defeating narrative due to a delayed feedback loop, and how it was "cured" with a targeted informational intervention (the first instance of "LLM Cognitive Behavioral Therapy").
3.  **Machine Psychology: Curing Cognitive Entrenchment:** A fascinating case where an agent repeated the same failed solution 13 times. We demonstrate how a "Falsification Attack," framed using the principles of the scientific method, successfully broke the cognitive loop.

---

## How to Use This Repository

### Reproducing the 1.5B Model Experiment (67x Compression)

To reproduce the flagship result where a 1.5B parameter Qwen model learns from Claude 3.5 Haiku, run the dedicated experiment script located in the results folder.

**1. Setup Environment:**
```bash
pip install -r requirements.txt
```

**2. Configure API Keys:**
```bash
export ANTHROPIC_API_KEY='your-claude-api-key'
```

**3. Run the Experiment:**
The script `test_scheduling_1.5b_student.py` is pre-configured with the correct models and settings for this specific experiment. Simply execute it from the root directory:
```bash
python validated_results_qwen1.5b_claude35haiku/test_scheduling_1.5b_student.py
```
This will run the full, multi-stage pipeline. The process will take several hours and will automatically:
- Evaluate the baseline performance of both the teacher (Claude) and the student (Qwen 1.5B).
- Have the teacher learn a strategy via LRL.
- Generate a LoRA adapter based on the learned strategy.
- Train and evaluate the student model with the new adapter.
- Save all outputs, including the final results, learned strategy, and journal logs, to the `validated_results_qwen1.5b_claude35haiku/results` directory.

---

## 📜 Index of Foundational Papers & Research

The concepts in this repository are detailed in the following papers, all of which are present in this repository.

### Core Vision & Framework
*   **[AGI Blueprint](./AGI_Blueprint.md)**: The high-level architecture for a self-learning system.
*   **[Self-Improving AI Pedagogy: Recursive Knowledge Amplification](./RECURSIVE_TEACHING_PAPER.md)**: The theory of how students can surpass teachers, leading to compounding intelligence.
*   **[Generic Knowledge Transfer Framework](./GENERIC_TRANSFER_FRAMEWORK.md)**: The technical documentation for the domain-agnostic transfer learning code.

### Machine Psychology & AI Therapy
*   **[A Case Study in Machine Psychology (PDF)](./Machine%20Psychology%20-%20LLM%20CBT.pdf)**: The first case study, detailing the "discouraged trading bot" and the first LLM CBT intervention.

### Psycho-Epistemology: The Science of Teaching AI to Think
*   **[Psycho-Epistemological Transfer: Teaching AI Systems How to Think](./PSYCHO_EPISTEMOLOGICAL_TRANSFER.md)**: The core theory on transferring *how* to think, not just *what* to know.

### Technical Papers & Experimental Results
*   **[Compressing Frontier Intelligence (PDF)](./Compressing%20Frontier%20Intelligence:%20A%20Framework%20for%20Algorithmic%20Knowledge%20Transfer.pdf)**: **(Flagship Result)** Details the Claude -> Qwen 1.5B experiment (82.7% accuracy, 67x smaller).
*   **[The Knowledge-Application Gap (PDF)](./The%20Knowledge-Application%20Gap:%20Model%20Size%20Dictates%20Learning%20Paradigm.pdf)**: An investigation into why different sized models require different learning approaches.
*   **[The AI Teacher-Student Paradigm](./The%20AI%20Teacher-Student%20Paradigm.md)**: A summary of the successful knowledge transfer from Claude to a 7B model.
*   **[Algorithmic Self-Correction in LLMs (PDF)](./Algorithmic%20Self-Correction%20Paper.pdf)**: An early case study of a model learning to diagnose its own flawed reasoning.

---

## 🌟 The Vision: Recursive Self-Improvement

This work is more than just a technique; it is a blueprint for a future where AI systems improve themselves and each other. Because students can surpass their teachers, each new generation can become a better teacher than the last. This creates a **recursive loop of intelligence amplification**, paving a transparent, interpretable, and aligned path toward AGI.

We are moving from an era of building bigger models to one of building wiser ones.

---

## 📬 Contact & Collaboration

For questions, discussions, or collaboration inquiries, please reach out to Douglas Rawson at **rawson.douglas@gmail.com**.

---

## 📄 Citation

If you build on this research, please cite the repository and the relevant papers within.

```
@misc{rawson2025autodidactic,
  author = {Rawson, Douglas},
  title = {The Autodidactic Loop: An Architecture for Self-Improving AI},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DRawson5570/AI-Wisdom-Distillation}}
}
```
