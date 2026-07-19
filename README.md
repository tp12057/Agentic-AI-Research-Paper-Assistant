# 📚 AI Research Paper Intelligence System

A semantic search engine for Machine Learning research papers that goes beyond keyword matching — it understands the **meaning** of a query, retrieves the most relevant papers from a corpus of 50,000 ArXiv abstracts, then automatically **summarizes** each result and **extracts key topics**, so you can scan a paper's relevance in seconds instead of reading the full abstract.

On top of that, an **autonomous multi-agent research assistant** takes a broad topic, plans multiple focused search angles on its own, decides whether it has gathered enough coverage, and writes a structured literature-review report — with no LLM API key required.

Built as part of the Coding Blocks Internship program.

---

## 🎯 What it does

### 1. Semantic Search
Type a natural-language research query like:

> "deep learning for medical image analysis"

...and the system:

1. **Understands intent** — converts the query into a 384-dimensional semantic vector
2. **Searches by meaning, not keywords** — retrieves the top-k most similar papers using cosine similarity over a FAISS vector index
3. **Summarizes** — condenses each returned abstract into a short, readable summary using a BART-based transformer
4. **Extracts key phrases** — pulls out the core topics/keywords from each paper using KeyBERT

### 2. Agentic Research Assistant
Give it a **broad** topic like "attention mechanisms in NLP" instead, and a team of coordinated agents autonomously:

1. **Plans** — breaks the topic into ~5 focused sub-queries using a small local LLM (`flan-t5-small`)
2. **Retrieves** — searches the FAISS index for each sub-query, tracking unique papers found
3. **Decides** — checks whether enough distinct papers have been gathered to cover the topic; if not, asks the planner for more angles (this is the loop that makes it an *agent*, not a fixed script)
4. **Synthesizes** — summarizes findings per angle and detects cross-cutting themes vs. niche research gaps
5. **Writes** — compiles everything into a Markdown literature-review report

All of this is wrapped in a clean, interactive **Streamlit** web app with two tabs — one for direct search, one for the agent.

---

## 🧠 How it works (architecture)

```
                     ┌─────────────────────────┐
                     │   ArXiv ML Papers (HF)   │
                     │  ~50,000 title+abstract  │
                     └────────────┬─────────────┘
                                  │  clean & merge
                                  ▼
                     ┌─────────────────────────┐
                     │   Sentence-Transformer   │
                     │    (all-MiniLM-L6-v2)    │
                     │   text --> 384-dim vec   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      FAISS Index         │
                     │  (Inner Product / cosine)│
                     └────────────┬─────────────┘
                                  │
        user query ──encode──────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Top-K similar papers   │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────┴─────────────┐
                     ▼                           ▼
          ┌────────────────────┐     ┌────────────────────┐
          │   BART Summarizer   │     │   KeyBERT Keywords  │
          │ (distilbart-cnn-12) │     │  (n-gram phrases)   │
          └────────────────────┘     └────────────────────┘
                     │                           │
                     └────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │      Streamlit UI        │
                     └─────────────────────────┘
```

### Agent architecture

```
        broad topic
             │
             ▼
   ┌───────────────────┐
   │   Planner Agent     │  --> flan-t5-small generates ~5 focused sub-queries
   └─────────┬───────────┘
             │
             ▼
   ┌───────────────────┐
   │  Retrieval Agent    │  --> calls PaperSearchEngine.search() per sub-query
   │  (dedups papers)    │      tracks unique papers found so far
   └─────────┬───────────┘
             │
             ▼
   ┌───────────────────┐
   │    Orchestrator     │  --> "have we found enough unique papers?"
   │ (coverage decision) │      NO --> ask Planner for more angles, loop
   └─────────┬───────────┘      YES --> continue
             │
             ▼
   ┌───────────────────┐
   │   Analyst Agent     │  --> summarizes per angle, finds common themes
   └─────────┬───────────┘      vs. niche/gap signals across all angles
             │
             ▼
   ┌───────────────────┐
   │   Writer Agent      │  --> compiles the final Markdown report
   └───────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Dataset | [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Search | `faiss-cpu` — `IndexFlatIP` (cosine similarity via L2-normalized inner product) |
| Summarization | `transformers` — `sshleifer/distilbart-cnn-12-6` |
| Keyword Extraction | `keybert` |
| Agent Planning LLM | `transformers` — `google/flan-t5-small` (small, free, local — no API key) |
| Data handling | `pandas`, `numpy` |
| Web App | `streamlit` |

---

## 📂 Project Structure

```
AI-Research-Paper-Intelligence-System/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md              # explains how generated data/embeddings/index are created
├── notebooks/
│   ├── 01_EDA_and_Embeddings.ipynb        # data exploration + embedding generation walkthrough
│   ├── 02_Search_Engine.ipynb             # FAISS search + summarization + keyword extraction walkthrough
│   └── 03_Agentic_Research_Assistant.ipynb  # multi-agent pipeline walkthrough
└── src/
    ├── data_prep.py            # load & clean the raw dataset
    ├── build_index.py          # generate embeddings + build the FAISS index
    ├── search_engine.py        # PaperSearchEngine class: search, summarize, extract keywords
    ├── app.py                  # Streamlit web app (Search tab + Research Agent tab)
    ├── run_agent.py             # CLI entry point for the research agent
    └── agents/
        ├── planner_agent.py     # decomposes a broad topic into focused sub-queries
        ├── retrieval_agent.py   # wraps PaperSearchEngine, dedups papers across queries
        ├── analyst_agent.py     # synthesizes findings, detects cross-cutting themes
        ├── writer_agent.py      # compiles the final Markdown report
        └── orchestrator.py      # ResearchAgent: coordinates the full autonomous loop
```

The `notebooks/` walk through the reasoning behind every step (useful for
understanding or presenting the project). The `src/` folder holds the same
logic refactored into clean, reusable, production-style modules that power
the actual app.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the search index (one-time setup)

This downloads the dataset, generates embeddings for ~50,000 papers, and
builds the FAISS index. It's compute-heavy (~20-30 min on CPU) but only
needs to run once — results are cached in `data/`.

```bash
python src/data_prep.py
python src/build_index.py
```

### 3. Launch the app

```bash
streamlit run src/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) and
start searching.

### Or use it from Python directly

```python
from src.search_engine import PaperSearchEngine

engine = PaperSearchEngine()
results = engine.full_report("deep learning for medical image analysis", k=5)

for r in results:
    print(r["title"], "-", r["score"])
    print(r["summary"])
    print(r["keywords"])
```

### 4. Run the Research Agent

From the Streamlit app, use the **🤖 Research Agent** tab — enter a broad topic
and click "Run Research Agent."

Or from the command line:

```bash
python src/run_agent.py "attention mechanisms in NLP"
```

This prints the agent's step-by-step reasoning (planning, searching,
coverage checks) and saves the final report to `data/agent_report_<topic>.md`.

Or from Python directly:

```python
from src.search_engine import PaperSearchEngine
from src.agents.orchestrator import ResearchAgent

engine = PaperSearchEngine()
agent = ResearchAgent(engine, min_coverage=10)
report = agent.run("attention mechanisms in NLP")
print(report)
```

---

## 💡 Key Design Decisions

- **Why FAISS `IndexFlatIP` instead of a database?** For 50k papers, an
  exact (non-approximate) flat index is fast enough and guarantees perfect
  recall — no accuracy trade-off from approximate search.
- **Why normalize embeddings before indexing?** Cosine similarity depends
  only on vector *direction*, not magnitude. Normalizing every vector to
  unit length lets us use FAISS's fast Inner Product search to get
  mathematically identical results to cosine similarity.
- **Why `all-MiniLM-L6-v2`?** A strong balance of speed and semantic
  quality — 384 dimensions is small enough to index and query instantly,
  while still capturing rich sentence-level meaning.
- **Why cache embeddings/index to disk?** Re-encoding 50,000 papers takes
  ~20-30 minutes; caching means the app starts in seconds on every
  subsequent run.
- **Why `flan-t5-small` for the agent's planning step, instead of GPT-4/Claude?**
  It runs locally on CPU with no API key and no cost, which matters for a
  project meant to be runnable by anyone without a billing setup. It's small
  enough that its output is occasionally rough, so the planner always falls
  back to reliable templates if the model can't be loaded or returns nothing
  usable — the agent never breaks just because the LLM step underperforms.
- **What makes this "agentic" rather than just a multi-step script?** The
  orchestrator makes a genuine runtime decision: after each round of
  searches, it checks how many *unique* papers have actually been found and
  only proceeds to writing the report once a coverage threshold is met (or
  the planner has run out of new angles to try) — otherwise it loops back
  for more searches. A fixed script would run a hardcoded number of steps
  regardless of what it found.

---

## 🔮 Possible Extensions

- Swap `IndexFlatIP` for `IndexIVFFlat` / `IndexHNSW` to scale to millions of papers
- Add filters (year, category) alongside semantic search
- Deploy the Streamlit app publicly (Streamlit Community Cloud / HF Spaces)
- Add a citation graph / "papers similar to this one" feature
- Swap `flan-t5-small` for a hosted LLM (OpenAI/Claude/Gemini API) for
  noticeably smarter planning and synthesis, if an API key becomes available
- Let the agent also decide *which* papers within a sub-topic are most worth
  summarizing, rather than always taking the top-k by similarity

---

## 🙋 Author

Built by **Amandeep Singh** — Coding Blocks Internship, Projects 2 & 3.
