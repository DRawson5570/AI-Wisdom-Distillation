# Case Study: Observing Learned Helplessness in an Autonomous AI Trading Agent

This folder contains the raw data and analysis for the "Machine Psychology" paper. It documents a real-world experiment where an autonomous trading agent, built on a Gemma-class LLM, developed a state analogous to learned helplessness due to a flawed feedback loop.

## Code and Reproducibility

The code for the **LRL framework and the scheduling experiments** is available in the root of this repository.

The specific code for the proprietary Forex trading bot that produced these logs is in a private repository and is not shared. However, the log files provided here contain the complete, unedited "thought process" of the AI agent, providing the full body of evidence for the findings discussed in the paper.

## Key Files

*   **[Machine_Psychology_Paper.md](./Machine_Psychology_Paper.md):** The full paper analyzing these findings.

*   **📂 Journal_Log/:** Contains the complete, unedited journal of the AI's self-reflections. This is the primary evidence of the AI's emergent "psychological" state.

*   **📂 Trade_Logs/:** Contains the raw logs of the trades executed by the agent across multiple, independent experimental runs.

*   **📂 System_Evolution/:** Contains snapshots of the AI's trading system (`trading_system.json`, etc.) at different versions, showing the evolution of its rules over time.
