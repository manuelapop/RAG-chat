# Created with AI assistance.
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def print_progress(label: str, current: int, total: Optional[int] = None) -> None:
    if total is not None:
        message = f"\r{label}: {current}/{total}"
    else:
        message = f"\r{label}: {current}"
    print(message, end="", file=sys.stdout, flush=True)


def end_progress() -> None:
    print(file=sys.stdout, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from local data.")
    parser.add_argument("--input", required=True, help="Path to input file (jsonl, csv, txt, pdf).")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for both indexing and retrieval.",
    )
    parser.add_argument("--chunk-words", type=int, default=300, help="Chunk size in words.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in words.")
    parser.add_argument("--text-column", default="text", help="CSV text column.")
    parser.add_argument("--title-column", default="title", help="CSV title column.")
    parser.add_argument("--source-column", default="source", help="CSV source column.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of input rows/documents to ingest (useful for quick tests).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size used during embedding. Lower it if you run out of memory.",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts.")
    return parser.parse_args()


def chunk_text(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_words
        window = words[start:end]
        if window:
            chunks.append(" ".join(window))
        if end >= len(words):
            break
    return chunks


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
            if idx % 1000 == 0:
                print_progress("Loaded JSONL rows", idx)
    if rows:
        print_progress("Loaded JSONL rows", len(rows))
        end_progress()
    return rows


def load_csv(
    path: Path, text_column: str, title_column: str, source_column: str, max_rows: Optional[int] = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized = {name.lower(): name for name in fieldnames}

        def resolve_column(requested: str, fallbacks: List[str]) -> str:
            if requested in fieldnames:
                return requested
            if requested.lower() in normalized:
                return normalized[requested.lower()]
            for candidate in fallbacks:
                if candidate in fieldnames:
                    return candidate
                lowered = candidate.lower()
                if lowered in normalized:
                    return normalized[lowered]
            return requested

        text_key = resolve_column(text_column, ["TEXT", "note_text", "content", "body"])
        title_key = resolve_column(title_column, ["title", "CATEGORY", "category", "DESCRIPTION", "description"])
        source_key = resolve_column(source_column, ["source", "SOURCE", "filename", "FILE"])
        id_key = resolve_column("id", ["id", "ID", "ROW_ID", "row_id"])
        load_total = max_rows if max_rows is not None else None

        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(
                {
                    "id": str(row.get(id_key, idx)),
                    "title": row.get(title_key, ""),
                    "text": row.get(text_key, ""),
                    "source": row.get(source_key, path.name),
                }
            )
            if (idx + 1) % 1000 == 0:
                print_progress("Loaded CSV rows", idx + 1, load_total)
    if rows:
        print_progress("Loaded CSV rows", len(rows), load_total)
        end_progress()
    return rows


def load_txt(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [
        {
            "id": path.stem,
            "title": path.stem,
            "text": text,
            "source": path.name,
        }
    ]


def load_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return [
        {
            "id": path.stem,
            "title": path.stem,
            "text": "\n".join(pages),
            "source": path.name,
        }
    ]


def load_documents(args: argparse.Namespace) -> List[Dict[str, Any]]:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(input_path)
    if suffix == ".csv":
        return load_csv(input_path, args.text_column, args.title_column, args.source_column, args.max_rows)
    if suffix == ".txt":
        return load_txt(input_path)
    if suffix == ".pdf":
        return load_pdf(input_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def build_chunks(docs: List[Dict[str, Any]], chunk_words: int, overlap: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total_docs = len(docs)
    for idx, doc in enumerate(docs):
        doc_id = str(doc.get("id", idx))
        title = str(doc.get("title", ""))
        source = str(doc.get("source", ""))
        text = str(doc.get("text", ""))

        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_words, overlap)):
            out.append(
                {
                    "chunk_id": f"{doc_id}_{chunk_idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "source": source,
                    "text": chunk,
                }
            )
        if (idx + 1) % 500 == 0 or idx + 1 == total_docs:
            print_progress("Chunked documents", idx + 1, total_docs)
    if docs:
        end_progress()
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    docs = load_documents(args)
    print(f"Loaded {len(docs)} documents.")

    print("Building chunks...")
    chunks = build_chunks(docs, args.chunk_words, args.chunk_overlap)
    if not chunks:
        raise ValueError("No chunks produced. Check input content.")
    print(f"Built {len(chunks)} chunks.")

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    texts = [item["text"] for item in chunks]
    print(f"Embedding chunks (batch size={args.embedding_batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=args.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = output_dir / "index.faiss"
    meta_path = output_dir / "meta.json"
    config_path = output_dir / "config.json"

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(chunks, ensure_ascii=True, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "chunk_words": args.chunk_words,
                "chunk_overlap": args.chunk_overlap,
                "embedding_batch_size": args.embedding_batch_size,
                "num_documents": len(docs),
                "num_chunks": len(chunks),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print(f"Saved: {index_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
