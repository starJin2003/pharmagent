---
title: PharmAgent
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# PharmAgent

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-yellow)](https://huggingface.co/spaces/starJin2003/pharmagent)

Drug interaction and safety Q&A agent backed by real FDA label data. Ask something like "Can I take ibuprofen with warfarin?" and get an answer with citations pulled from openFDA drug labels, not generated from training data. Built as an agentic RAG pipeline with LangGraph, hybrid retrieval (FAISS + BM25), and input/output safety guards.

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

## Architecture

### Offline Pipeline (run once)

```
openFDA API
  -> Fetch top 500 drug labels (2 labels per drug, deduplicated by generic name + route)
  -> Extract 7 FDA label sections per drug
  -> Section-aware chunking (3,160 chunks total)
  -> Embed with text-embedding-3-small (1,536 dimensions)
  -> Build FAISS index (cosine similarity) + BM25 index (keyword matching)
  -> Save to data/indexes/
```

### Online Pipeline (per query)

```
User Query
  |
  v
[1] input_safety_check
    Regex and keyword matching, no LLM call.
    Detects emergency keywords and dosage/prescription requests.
    If flagged -> return safety message, skip all remaining nodes.
  |
  v
[2] resolve_drugs
    GPT-4o-mini extracts drug names from the query as a JSON list.
    Each name is sent to the RxNorm API to resolve brand names to generic names.
    "Tylenol" -> acetaminophen, "Advil" -> ibuprofen.
    If no drugs resolved -> return clarification message, skip remaining nodes.
  |
  v
[3] classify_query
    GPT-4o-mini classifies the query into one of four types:
    interaction_check | side_effect | contraindication | general_info
  |
  v
[4] retrieve_from_index
    Query is enriched by prepending resolved generic drug names.
    FAISS semantic search (top 20) + BM25 keyword search (top 20).
    Reciprocal Rank Fusion (RRF) merges both ranked lists.
    Returns top 5 chunks with metadata.
  |
  v
[5] generate_response
    GPT-4o-mini generates an answer using only the retrieved chunks as context.
    Every factual claim must include a [Source N] citation.
    If no relevant chunks found, says so instead of guessing.
  |
  v
[6] output_safety_check
    Strips any dosage recommendations via regex.
    Appends disclaimer if missing.
    Builds formatted sources list mapping each [Source N] to drug name, section, and source.
  |
  v
Answer with citations + disclaimer + sources
```

## Evaluation

Evaluated on a 40-query golden dataset across interaction checks, side effects, contraindications, and out-of-scope queries.

| Metric | Score | Target |
|--------|-------|--------|
| Recall@5 | **0.95** | >= 0.75 |
| MRR | **0.80** | >= 0.50 |
| Faithfulness | **0.96** | >= 0.85 |
| Citation Accuracy | **0.96** | >= 0.80 |

Faithfulness is judged by GPT-4o-mini (each answer sentence scored as SUPPORTED / NOT_SUPPORTED against retrieved chunks). Citation accuracy measures the fraction of factual claims with valid `[Source N]` references.

### Iteration Process

| Metric | Run 1 | Run 2 | Run 3 | Run 4 (final) | Target |
|--------|-------|-------|-------|----------------|--------|
| Recall@5 | 0.95 | 0.95 | 0.95 | 0.95 | >= 0.75 |
| MRR | 0.80 | 0.80 | 0.80 | 0.80 | >= 0.50 |
| Faithfulness | 0.73 | 0.73 | 0.975 | 0.96 | >= 0.85 |
| Citation Acc | 0.54 | 0.69 | 0.69 | 0.96 | >= 0.80 |

**Run 2.** Added "Every factual sentence MUST end with [Source N]" and a few-shot example to the generator prompt. Citation accuracy 0.54 to 0.69.

**Run 3.** Faithfulness judge was marking "no interaction found" answers as NOT_SUPPORTED because the source chunks never explicitly state "there is no interaction." Fixed the judge prompt to treat absence-of-evidence answers as supported. Faithfulness 0.73 to 0.975.

**Run 4.** Generator was not citing sources in safe combination answers where it described each drug individually. Added a rule: every sentence describing a drug must cite its source, even in "no interaction found" responses. Citation accuracy 0.69 to 0.96.

## Quick Start

```bash
git clone https://github.com/starJin2003/pharmagent.git
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
├── app.py                      # Entry point - launches Gradio
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
└── data/
    └── indexes/                # FAISS + BM25 indexes (committed)
```

## How It Works

Each query runs through a 6-node LangGraph pipeline. The first node checks for safety issues (emergencies, dosage requests) using regex, no LLM. If the query passes, the second node uses GPT-4o-mini to extract drug names and resolves them to generic names via the RxNorm API. The third node classifies the query type. The fourth node runs hybrid retrieval. The fifth node generates a cited answer. The sixth node appends a disclaimer and formats the sources list. If the safety check or drug resolution fails, the pipeline exits early without calling the LLM for generation.

Retrieval uses a hybrid approach: FAISS (semantic similarity via cosine search on OpenAI embeddings) and BM25 (keyword matching). Results from both are merged using Reciprocal Rank Fusion (RRF). Each retriever returns its top 20 candidates, RRF ranks them, and the top 5 go to the generator.

Drug names in the query are resolved to generic names via the RxNorm API before retrieval. This means "Tylenol" becomes "acetaminophen" and gets matched against the right FDA label chunks.

Every claim in the response maps to a `[Source N]` citation pointing to a specific drug label section. The output guard enforces this. No citation, no claim.

Why retrieval instead of just stuffing everything into a long context window? Cost and traceability. Embedding 3,160 chunks once is cheap. Sending all of them per query is not. And citations need to point to specific source chunks, which a retrieval pipeline gives you for free.

## Design Decisions

**Why RAG instead of direct LLM prompting.** LLMs hallucinate drug information. They will confidently state interactions that don't exist or miss ones that do. RAG forces every claim to come from a real FDA label. If the retriever finds nothing relevant, the system says "I could not find sufficient information" instead of making something up. The citations are verifiable because they point to actual label sections.

**Why hybrid retrieval (FAISS + BM25 + RRF).** FAISS catches semantic matches: a query about "blood thinners" will match chunks containing "anticoagulants" because the embeddings are close in vector space. BM25 catches exact keyword matches that embeddings might miss, like specific drug names or medical terms. Reciprocal Rank Fusion merges the two ranked lists into one without needing learned weights or tuning. Each method covers the other's blind spots.

**Why GPT-4o-mini.** It costs $0.15 per million input tokens and $0.60 per million output tokens. The entire project (ingestion, evaluation, testing) cost under $0.10. It handles drug name extraction, query classification, and response generation well enough for this use case. No reason to pay more for a larger model.

**Why section-aware chunking.** FDA drug labels have distinct sections: drug_interactions, warnings, adverse_reactions, contraindications, and others. Naive chunking that splits on token count alone would break across section boundaries, mixing interaction data with warning data in the same chunk. Section-aware chunking keeps each chunk within one section and tags it with the section type in metadata. This lets the retriever know whether a chunk is about interactions, warnings, or contraindications. Different sections also get different target chunk sizes: drug_interactions splits into 150-400 token chunks, while contraindications are kept whole since they tend to be short.

**Why FAISS over managed vector databases.** The index has 3,160 vectors at 1,536 dimensions. That fits in memory on a free Hugging Face Space with room to spare. A managed database like Pinecone or Weaviate would add network latency on every query, require a separate account and API key, and cost money, all for no benefit at this scale. FAISS loads from a local file and searches in milliseconds.

## Known Limitations and Future Work

The original design included the RxNorm Drug Interaction API for live drug-drug interaction lookups running in parallel with label retrieval. RxNorm deprecated that API in January 2024 with no free replacement. DrugBank requires a paid license and DDInter has no REST API. The system now relies entirely on the drug_interactions sections from FDA labels, which cover most common interactions but may miss newer findings not yet reflected in label updates.

**Current limitations:**

- Coverage is limited to roughly 500 drugs from openFDA. Rare or very new drugs may not be indexed.
- FDA labels are not updated in real time. The index reflects whatever openFDA had at ingestion time.
- Dosage information is intentionally stripped by the output safety guard.
- English only. FDA labels are in English and there is no multilingual support.
- Tied to OpenAI for both the LLM (GPT-4o-mini) and embeddings. No local model fallback.

**Future work:**

- Re-ranker model (cross-encoder) between retrieval and generation to improve chunk relevance
- Query decomposition for multi-drug interactions (3+ drugs)
- Local embedding model option to remove the OpenAI dependency for retrieval
- User feedback loop to flag bad answers and improve the golden dataset

## License

MIT
