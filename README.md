# RAG Chat with Hugging Face + FAISS

This project builds a chatbot that answers from your own dataset using a Retrieval-Augmented Generation (RAG) pipeline.

## What you get

- Dataset ingestion from `jsonl`, `csv`, `txt`, and `pdf`
- Text chunking with overlap
- Embeddings via Sentence Transformers
- FAISS index for fast semantic search
- Confidence threshold to avoid weak/ungrounded answers
- Gradio chat interface with source snippets

## Project layout

```text
rag_chat_hf/
  data/
    sample_docs.jsonl
  artifacts/
  src/
    ingest.py
    chat_app.py
  requirements.txt
```

## 1) Setup

```bash
cd /Users/manuelapop/rag_chat_hf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add your dataset

Put your files in `data/` and run one of:

```bash
python src/ingest.py --input data/sample_docs.jsonl
python src/ingest.py --input data/my_docs.csv --text-column text --title-column title --source-column source
python src/ingest.py --input data/manuals.pdf
python src/ingest.py --input data/notes.txt
```

Artifacts are written to `artifacts/`:

- `index.faiss`
- `meta.json`

## 3) Start chat app

```bash
python src/chat_app.py --model google/gemma-2-2b-it
```

If your machine is limited, use a smaller instruct model:

```bash
python src/chat_app.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 4) Helpful options

```bash
python src/ingest.py --input data/sample_docs.jsonl --chunk-words 350 --chunk-overlap 60
python src/ingest.py --input data/NOTEEVENTS.csv --text-column TEXT --title-column CATEGORY --max-rows 20000
python src/ingest.py --input data/NOTEEVENTS.csv --text-column TEXT --title-column CATEGORY --max-rows 5000 --chunk-words 600 --chunk-overlap 80 --embedding-batch-size 32
python src/chat_app.py --top-k 5 --min-score 0.25
```

Recommended run for `NOTEEVENTS.csv`:

```bash
python src/ingest.py --input data/NOTEEVENTS.csv --text-column TEXT --title-column CATEGORY --max-rows 20000 --chunk-words 600 --chunk-overlap 80 --embedding-batch-size 32
python src/chat_app.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 160 --top-k 5 --min-score 0.2
```

Ingestion prints progress for loading rows, chunking, and embedding so long runs are visible.

## Quickstart for new users

1) Setup environment:

```bash
cd /Users/manuelapop/rag_chat_hf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put your own notes file in `data/` (for example `data/my_notes.csv`).

3) Build index artifacts from your notes:

```bash
python src/ingest.py --input data/my_notes.csv --text-column TEXT --title-column CATEGORY --chunk-words 600 --chunk-overlap 80 --embedding-batch-size 32
```

4) Start the chat app:

```bash
python src/chat_app.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 160 --top-k 5 --min-score 0.2
```

5) Open the local URL shown in terminal (usually `http://127.0.0.1:7860`) and ask questions.

Tip: run a smaller test first with `--max-rows 1000`, then increase once everything works.

## Notes

- Retrieval and queries must use the same embedding model.
- If no chunk passes `--min-score`, the app returns "I don't know" to reduce hallucinations.
- First model load may take time due to downloads.
