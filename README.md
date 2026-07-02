# 🧠 Personal Research Assistant

A fully local RAG (Retrieval-Augmented Generation) system that lets you chat
with your own PDF documents — no API keys, no cloud, runs entirely on your machine.

## Features
- Upload any PDF and ask questions about it
- Answers are grounded in your documents (no hallucination)
- Shows retrieved source chunks for full transparency
- Custom evaluation framework measuring faithfulness, relevancy, and context recall

## Tech Stack
| Layer | Tool |
|---|---|
| LLM | Llama 3.2 via Ollama (local) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |
| Eval | Custom LLM-as-judge + cosine similarity |

## Evaluation Results

| Metric | Baseline (chunk=500) | Small chunks (chunk=200) | Large chunks (chunk=1000) |
|---|---|---|---|
| Faithfulness | 0.00 | 0.00 | 0.00 |
| Answer Relevancy | 0.00 | 0.00 | 0.00 |
| Context Recall | 0.00 | 0.00 | 0.00 |

> Replace the 0.00s with your actual scores from running evaluate.py with different chunk sizes

## Setup

1. Install [Ollama](https://ollama.com) and pull the model:
```bash
   ollama pull llama3.2
```

2. Clone the repo and install dependencies:
```bash
   git clone https://github.com/yourusername/research-assistant
   cd research-assistant
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

3. Add PDFs to the `docs/` folder, then ingest:
```bash
   python3 ingest.py
```

4. Launch the app:
```bash
   streamlit run app.py
```

## Key Design Decisions
- **Local-first**: chose Ollama + HuggingFace embeddings over OpenAI API to eliminate
  cost and latency, and to keep data private
- **Custom evaluator**: built evaluation metrics from scratch after dependency conflicts
  with RAGAS, using LLM-as-judge for faithfulness and cosine similarity for relevancy
- **Chunking experiments**: tested chunk sizes of 200, 500, and 1000 characters —
  found that [write your finding here] performed best for this document type
