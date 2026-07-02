# Hybrid Ranker Pipeline — Team Ocean Enjoyers

This repository contains our official rule-compliant submission for the LLM Ranking Hackathon. Our system utilizes a lightweight, high-performance hybrid retrieval and ranking architecture designed to execute 100% locally on standard CPU hardware well within the 5-minute compute limit.

## 🚀 Quick Start / Reproduction Command

To reproduce our final submission CSV from the raw candidates file in a single command, run:

```bash
python hybrid_ranker_pipeline.py --candidates ./candidates.jsonl --out ./submission.csv