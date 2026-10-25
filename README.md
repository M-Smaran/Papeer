# Papeer

A research paper assistant. Load papers from PDFs, web pages or arXiv, then chat with them: ask questions grounded in the text, check whether a claim has been superseded by newer work, or search the web for recent findings.

Built with Streamlit, LangGraph, OpenAI, Qdrant and Tavily, and evaluated with DeepEval.

## Features

- **Ask your papers.** Questions are answered from the documents loaded into the current chat, with an agent that chooses between searching your papers and searching the web.
- **Verify a claim.** Paste a claim from a paper and Papeer searches the web and arXiv for more recent work that supersedes or challenges it, with links.
- **Multiple sources.** Upload PDF, TXT or Markdown files, load web pages by URL, or pull a paper from arXiv by ID or title.
- **Separate chats.** Each chat has its own document collection and history, and chats are named automatically from the first message.
- **Side questions.** Start a message with `/btw` to ask something off-topic without adding it to the chat history.

## How it works

Each message runs through a LangGraph pipeline:

```
router ──► retrieve ──► agent ⇄ tools (vector store / web search)
   │                       │
   │                       ▼
   │                relevancy check ──► rewrite query (once) ──► agent
   │                       │
   ├──► verify claim ──────┤
   │                       ▼
   └──► direct answer ──► generate answer
```

1. A **router** decides whether the question needs your documents, a claim check, or can be answered directly.
2. For retrieval, an **agent** calls tools to search the Qdrant vector store or the web (Tavily), up to 3 times.
3. A **relevancy check** grades the retrieved chunks. If they are off-topic, the query is rewritten and retried once.
4. The **answer** is generated from the retrieved context and streamed to the UI.

Conversation state is checkpointed to SQLite so chats survive restarts.

### Project layout

```
app.py                  Streamlit UI
backend/
  rag_graph.py          LangGraph pipeline (router, agent, tools, answer)
  vector_store.py       Qdrant collections per chat, cached OpenAI embeddings
  paper_loader.py       PDF / text / Markdown / web / arXiv loaders
  btw_handler.py        /btw side-question handler
  models.py             Pydantic schemas for structured LLM output
evaluate.py             DeepEval evaluation script
goldens.json            Evaluation questions and expected answers
.streamlit/config.toml  UI theme
```

## Setup

You need Python 3.12 and accounts for:

- [OpenAI](https://platform.openai.com/), for the LLM and embeddings
- [Qdrant Cloud](https://cloud.qdrant.io/), the vector database (the free tier is enough)
- [Tavily](https://tavily.com/), for web search (the free tier is enough)

**1. Clone and install**

```bash
git clone https://github.com/M-Smaran/Papeer.git
cd Papeer
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

**2. Add your keys**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=...
TAVILY_API_KEY=tvly-...
```

`.env` is in `.gitignore`. Never commit it.

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. Add documents from the **Add sources** panel in the sidebar, then start asking questions.

## Evaluate

`evaluate.py` runs every question in `goldens.json` through the pipeline and scores the results with DeepEval.

1. Put the PDF to evaluate against at `documents/Openclaw_Research_Report.pdf`, or change `PDF_PATH` in `evaluate.py`. The `documents/` folder is not in the repository.
2. Run:

   ```bash
   python evaluate.py
   ```

Results are written to `eval_results.json`. If `goldens.json` is deleted, the script generates a new set of questions from the PDF.

Each run calls the OpenAI API for every question and every metric, so it costs money.

### Latest results

From the committed `eval_results.json`: 10 questions, pass threshold 0.7. This run predates the switch to `gpt-5-mini`, so rerun the script for current numbers.

| Metric | Mean score |
|---|---|
| Faithfulness | 0.98 |
| Contextual Precision | 0.96 |
| Contextual Recall | 0.96 |
| Answer Relevancy | 0.72 |
| Contextual Relevancy | 0.62 |

## Troubleshooting

- **`Failed to obtain server version` from Qdrant.** The client can't reach your cluster. Free-tier clusters are suspended after a period of inactivity, so resume yours from the Qdrant Cloud console.
- **`Qdrant client version ... is incompatible with server version`.** Upgrade the client to match your server: `pip install "qdrant-client>=1.19,<1.20"`.
- **`KeyError: 'TAVILY_API_KEY'`.** Add `TAVILY_API_KEY` to `.env`. Web search, claim verification and `/btw` web lookups all need it.
