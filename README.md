# 🌊Redrob India Runs Track 01: The Data and AI challenge
**Team Name:** ocean_enjoyers  
**Track:** Data and AI Challenge

An optimized hybrid ranking engine that filters, retrieves, and scores candidates using BM25 keyword matching and ChromaDB/HuggingFace dense vector semantic search.

---

## 🛠️ Setup & Installation

Ensure you have Python installed, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```


To run the pipeline and generate the final submission file, use the following single command:

```bash
python hybrid_ranker_pipeline.py --candidates ./candidates.jsonl --out ./ocean_enjoyers.csv
```