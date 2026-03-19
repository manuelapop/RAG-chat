# Created with AI assistance.
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RAG chat app with Gradio.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with index.faiss/meta.json/config.json.")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="HF generation model name.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Minimum similarity threshold; below this returns 'I don't know'.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation token limit.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL.")
    return parser.parse_args()


def load_artifacts(artifacts_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    index_path = artifacts_dir / "index.faiss"
    meta_path = artifacts_dir / "meta.json"
    config_path = artifacts_dir / "config.json"

    if not index_path.exists() or not meta_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run ingestion first, for example:\n"
            "python src/ingest.py --input data/sample_docs.jsonl"
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return index, metadata, cfg


def retrieve(
    query: str,
    index: faiss.Index,
    metadata: Sequence[Dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(query_vec, top_k)

    matches: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        row = dict(metadata[idx])
        row["score"] = float(score)
        matches.append(row)
    return matches


def build_prompt(user_message: str, contexts: Sequence[Dict[str, Any]], history: Sequence[Tuple[str, str]]) -> str:
    turns = history[-4:]
    history_text = ""
    for user_text, assistant_text in turns:
        assistant_main = assistant_text.split("\n\nSources:\n", 1)[0]
        history_text += f"User: {user_text}\nAssistant: {assistant_main}\n"

    context_text = "\n\n".join(
        [
            f"[Source: {ctx.get('source', '')}; Title: {ctx.get('title', '')}; Score: {ctx.get('score', 0.0):.3f}] "
            f"{ctx.get('text', '')}"
            for ctx in contexts
        ]
    )

    return (
        "You are a helpful assistant.\n"
        "Answer only using the context below.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I don't know based on the provided data.\"\n\n"
        f"Recent conversation:\n{history_text}\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {user_message}\n"
        "Answer:"
    )


def normalize_history(history: Sequence[Any]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []

    # Old Gradio format: [(user, assistant), ...]
    if history and isinstance(history[0], (tuple, list)):
        for item in history:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                turns.append((str(item[0]), str(item[1])))
        return turns

    # New Gradio messages format: [{"role": "...", "content": "..."}, ...]
    pending_user: Optional[str] = None
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append((pending_user, content))
            pending_user = None
    return turns


def trim_prompt_to_model_limit(
    user_message: str,
    contexts: Sequence[Dict[str, Any]],
    history: Sequence[Tuple[str, str]],
    tokenizer: Any,
    max_new_tokens: int,
) -> str:
    model_limit = getattr(tokenizer, "model_max_length", 2048)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 1_000_000:
        model_limit = 2048
    input_budget = max(128, model_limit - max_new_tokens - 16)

    # Start with full context/history, then trim until prompt fits model token budget.
    current_contexts = [dict(ctx) for ctx in contexts]
    current_history = list(history)

    def fits(prompt_text: str) -> bool:
        token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return len(token_ids) <= input_budget

    prompt = build_prompt(user_message, current_contexts, current_history)
    if fits(prompt):
        return prompt

    if current_history:
        prompt = build_prompt(user_message, current_contexts, [])
        if fits(prompt):
            return prompt
        current_history = []

    # Iteratively reduce context text length and number of context chunks.
    char_limits = [1200, 800, 500, 300, 180]
    for char_limit in char_limits:
        shortened = []
        for ctx in current_contexts:
            cut = dict(ctx)
            text = str(cut.get("text", ""))
            if len(text) > char_limit:
                cut["text"] = text[:char_limit] + "..."
            shortened.append(cut)

        for keep_n in range(len(shortened), 0, -1):
            prompt = build_prompt(user_message, shortened[:keep_n], current_history)
            if fits(prompt):
                return prompt

    # Last resort: no retrieved context, just preserve user message.
    return build_prompt(user_message, [], [])


def format_sources(contexts: Sequence[Dict[str, Any]]) -> str:
    if not contexts:
        return "Sources:\n- none"
    lines = ["Sources:"]
    for ctx in contexts:
        snippet = ctx.get("text", "").replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(
            f"- {ctx.get('source', 'unknown')} | {ctx.get('title', '')} | score={ctx.get('score', 0.0):.3f}\n"
            f"  snippet: {snippet}"
        )
    return "\n".join(lines)


def build_generator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    use_device_map = has_accelerate and torch.cuda.is_available()

    if use_device_map:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # Prevent max_length/max_new_tokens conflicts from model defaults during generation.
    model.generation_config.max_length = None
    return tokenizer, model


def safe_model_limit(tokenizer: Any) -> int:
    model_limit = getattr(tokenizer, "model_max_length", 2048)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 1_000_000:
        return 2048
    return model_limit


def generate_text(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    temperature: float,
) -> str:
    model_limit = safe_model_limit(tokenizer)
    input_budget = max(64, model_limit - max_new_tokens - 8)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_budget)

    if hasattr(model, "device") and not hasattr(model, "hf_device_map"):
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    index, metadata, cfg = load_artifacts(artifacts_dir)

    embedder = SentenceTransformer(cfg["embedding_model"])
    generator_tokenizer, generator_model = build_generator(args.model)

    def chat_fn(message: str, history: List[Any]):
        turns = normalize_history(history or [])
        contexts = retrieve(message, index, metadata, embedder, args.top_k)
        best_score = max((ctx["score"] for ctx in contexts), default=-1.0)

        if best_score < args.min_score:
            answer = "I don't know based on the provided data."
            final = f"{answer}\n\n{format_sources(contexts)}"
            return final

        prompt = trim_prompt_to_model_limit(message, contexts, turns, generator_tokenizer, args.max_new_tokens)
        output = generate_text(prompt, generator_tokenizer, generator_model, args.max_new_tokens, args.temperature)
        answer = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
        final = f"{answer}\n\n{format_sources(contexts)}"
        return final

    app = gr.ChatInterface(
        fn=chat_fn,
        title="Hugging Face RAG Chat",
        description=(
            "This chat retrieves relevant chunks from your dataset before generating an answer."
        ),
    )
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
