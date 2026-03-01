---
title: PharmAgent
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# PharmAgent

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/pharmagent)

Drug interaction and safety Q&A agent backed by real FDA label data. Ask something like "Can I take ibuprofen with warfarin?" and get an answer with citations pulled from openFDA drug labels — not generated from training data. Built as an agentic RAG pipeline with LangGraph, hybrid retrieval (FAISS + BM25), and input/output safety guards.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Agent framework | LangGraph |
| RAG framework | LangChain |
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536 dim) |
| Vector DB | FAISS (faiss-cpu) |
| Keyword search | rank-bm25 |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |
| HTTP | httpx (async) |
| Caching | diskcache |

## Evaluation

Evaluated on a 40-query golden dataset across interaction checks, side effects, contraindications, and out-of-scope queries.

| Metric | Score | Target |
|--------|-------|--------|
| Recall@5 | **0.95** | >= 0.75 |
| MRR | **0.80** | >= 0.50 |
| Faithfulness | **0.96** | >= 0.85 |
| Citation Accuracy | **0.96** | >= 0.80 |

Faithfulness is judged by GPT-4o-mini (each answer sentence scored as SUPPORTED / NOT_SUPPORTED against retrieved chunks). Citation accuracy measures the fraction of factual claims with valid `[Source N]` references.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pharmagent.git
cd pharmagent

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

python app.py
# Open http://localhost:7860
```

The FAISS and BM25 indexes are included in the repo (`data/indexes/`), so you don't need to run ingestion. If you want to rebuild them:

```bash
python -m src.ingestion.run_ingestion
```

## Project Structure

```
pharmagent/
├── app.py                      # Entry point — launches Gradio
├── src/
│   ├── config.py               # Env vars, paths, constants
│   ├── api_clients/
│   │   ├── openfda_client.py   # openFDA Label API
│   │   └── rxnorm_client.py    # RxNorm drug name resolution
│   ├── ingestion/
│   │   ├── fetch_and_parse.py  # Fetch + extract label sections
│   │   ├── chunker.py          # Section-aware chunking
│   │   ├── indexer.py          # Embed + build FAISS/BM25
│   │   └── run_ingestion.py    # Orchestrate pipeline
│   ├── retrieval/
│   │   └── retriever.py        # FAISS + BM25 + RRF fusion
│   ├── agents/
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── nodes.py            # 6 node functions
│   │   └── graph.py            # LangGraph wiring
│   ├── evaluation/
│   │   ├── golden_dataset.json # 40 test queries
│   │   └── run_eval.py         # Recall, MRR, faithfulness, citations
│   └── app/
│       └── main.py             # Gradio UI
├── data/
│   └── indexes/                # FAISS + BM25 indexes (committed)
└── tests/
```

## How It Works

Retrieval uses a hybrid approach: FAISS (semantic similarity via cosine search on OpenAI embeddings) and BM25 (keyword matching). Results from both are merged using Reciprocal Rank Fusion (RRF). Each retriever returns its top 20 candidates, RRF ranks them, and the top 5 go to the generator.

Drug names in the query are resolved to generic names via the RxNorm API before retrieval. This means "Tylenol" becomes "acetaminophen" and gets matched against the right FDA label chunks.

Every claim in the response maps to a `[Source N]` citation pointing to a specific drug label section. The output guard enforces this — no citation, no claim.

Why retrieval instead of just stuffing everything into a long context window? Cost and traceability. Embedding ~5,000 chunks once is cheap. Sending all of them per query is not. And citations need to point to specific source chunks, which a retrieval pipeline gives you for free.

## Limitations

- **Coverage**: ~500 drugs from openFDA. Rare or very new drugs may not be indexed.
- **Label lag**: FDA labels aren't updated in real time. The index reflects whatever openFDA had at ingestion time.
- **No dosage info**: Intentionally stripped by the output safety guard.
- **English only**: FDA labels are in English; no multilingual support.
- **Single LLM provider**: Tied to OpenAI (GPT-4o-mini + embeddings). No local model fallback.

## Future Work

- Add DailyMed as a second data source for broader label coverage
- Re-ranker model (e.g. cross-encoder) between retrieval and generation
- Query decomposition for multi-drug interactions (3+ drugs)
- Local embedding model option to remove OpenAI dependency for retrieval
- User feedback loop to flag bad answers and improve the golden dataset

## License

MIT
