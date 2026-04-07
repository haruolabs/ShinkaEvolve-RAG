"""Seed retrieval program for a ShinkaEvolve MultiHop-RAG project."""

from __future__ import annotations

# EVOLVE-BLOCK-START
import re
from typing import Any, Dict, List, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
COMMON_QUERY_TOKENS = {
    "about",
    "after",
    "before",
    "between",
    "during",
    "from",
    "into",
    "near",
    "news",
    "said",
    "says",
    "that",
    "their",
    "there",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "while",
    "whose",
    "with",
}


def get_retrieval_strategy() -> Dict[str, Any]:
    """Returns only the retrieval-side policy that evolution may change."""
    return {
        "candidate_pool_size": 24,
        "top_k": 10,
        "min_query_token_length": 3,
        "max_chunks_per_title": 2,
        "dense_weight": 1.0,
        "lexical_weight": 0.6,
        "metadata_weight": 0.25,
        "duplicate_penalty": 0.35,
        "diversity_penalty": 0.2,
    }


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _query_features(query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
    query_tokens = [
        token
        for token in _tokenize(query)
        if len(token) >= strategy["min_query_token_length"]
        and token not in COMMON_QUERY_TOKENS
    ]
    metadata_tokens = {
        token
        for token in query_tokens
        if any(char.isdigit() for char in token)
    }
    return {"query_tokens": set(query_tokens), "metadata_tokens": metadata_tokens}


def _candidate_score(
    candidate: Dict[str, Any],
    features: Dict[str, Any],
    strategy: Dict[str, Any],
    title_counts: Dict[str, int],
    chosen_texts: Sequence[str],
) -> float:
    score = strategy["dense_weight"] * float(candidate.get("dense_score") or 0.0)

    candidate_tokens = set(_tokenize(candidate["text"]))
    if features["query_tokens"]:
        lexical_overlap = len(features["query_tokens"] & candidate_tokens)
        score += strategy["lexical_weight"] * (
            lexical_overlap / len(features["query_tokens"])
        )

    metadata_blob = " ".join(
        [
            candidate.get("title", ""),
            candidate.get("source", ""),
            candidate.get("published_at", ""),
        ]
    ).lower()
    if features["metadata_tokens"]:
        metadata_hits = sum(
            1 for token in features["metadata_tokens"] if token in metadata_blob
        )
        score += strategy["metadata_weight"] * (
            metadata_hits / len(features["metadata_tokens"])
        )

    title = candidate.get("title", "")
    if title and title_counts.get(title, 0) >= strategy["max_chunks_per_title"]:
        score -= strategy["diversity_penalty"] * title_counts[title]

    if candidate.get("normalized_text") in chosen_texts:
        score -= strategy["duplicate_penalty"]

    return score


def rerank_candidates(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Adds lexical, metadata, and diversity-aware reranking."""
    features = _query_features(query, strategy)
    rescored = [dict(candidate) for candidate in candidates]

    for candidate in rescored:
        candidate["base_score"] = _candidate_score(
            candidate=candidate,
            features=features,
            strategy=strategy,
            title_counts={},
            chosen_texts=[],
        )

    rescored.sort(
        key=lambda item: (item["base_score"], item.get("dense_score", 0.0)),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    title_counts: Dict[str, int] = {}
    chosen_texts: List[str] = []
    for candidate in rescored:
        title = candidate.get("title", "")
        if (
            title
            and title_counts.get(title, 0) >= strategy["max_chunks_per_title"]
            and len(selected) < strategy["top_k"] - 1
        ):
            continue

        candidate["final_score"] = _candidate_score(
            candidate=candidate,
            features=features,
            strategy=strategy,
            title_counts=title_counts,
            chosen_texts=chosen_texts,
        )
        selected.append(candidate)
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1
        chosen_texts.append(candidate.get("normalized_text", ""))
        if len(selected) >= strategy["top_k"]:
            break

    selected.sort(
        key=lambda item: (item.get("final_score", 0.0), item.get("dense_score", 0.0)),
        reverse=True,
    )
    return selected


# EVOLVE-BLOCK-END

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from retrieval_runtime import (
    CORPUS_EMBED_MODEL_NAME,
    RUNTIME_PATHS,
    build_embed_model,
    build_run_payload,
    format_query_result,
    format_retrieval_item,
    load_json_payload,
    load_llama_index_modules,
    load_or_build_corpus_cache,
    retrieve_dense_candidates,
    sample_query_data,
    write_retrieval_results,
)


def run_retrieval(
    save_path: str | None = None,
    seed: int | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Loads fixed runtime assets, then reranks dense candidates for each query."""
    strategy = get_retrieval_strategy()
    modules = load_llama_index_modules()

    embed_model = build_embed_model(modules, CORPUS_EMBED_MODEL_NAME)
    node_records, corpus_embeddings, used_cached_corpus = load_or_build_corpus_cache(
        modules,
        embed_model,
    )
    if used_cached_corpus:
        print("[CACHE] Retrieval will use the cached corpus embeddings for this run.")
    else:
        print("[CACHE] Retrieval will use the newly built corpus embeddings for this run.")

    full_query_data = load_json_payload(RUNTIME_PATHS.query_path)
    query_data = sample_query_data(full_query_data, seed)
    print(
        f"Loaded {len(full_query_data)} queries and selected {len(query_data)} for this run."
    )

    retrieval_save_list: List[Dict[str, Any]] = []
    candidate_pool_size = max(strategy["candidate_pool_size"], strategy["top_k"])

    for query_record in tqdm(query_data, desc="Retrieving queries", unit="query"):
        dense_candidates = retrieve_dense_candidates(
            query=query_record["query"],
            embed_model=embed_model,
            node_records=node_records,
            corpus_embeddings=corpus_embeddings,
            top_n=candidate_pool_size,
            tokenize=_tokenize,
        )

        reranked_candidates = rerank_candidates(
            query=query_record["query"],
            candidates=dense_candidates,
            strategy=strategy,
        )

        retrieval_list = [
            format_retrieval_item(candidate, final_rank)
            for final_rank, candidate in enumerate(reranked_candidates, start=1)
        ]
        retrieval_save_list.append(format_query_result(query_record, retrieval_list))

    target_path = write_retrieval_results(retrieval_save_list, save_path=save_path)
    return build_run_payload(
        retrieval_save_list=retrieval_save_list,
        save_file=target_path,
        strategy=strategy,
        used_cached_corpus=used_cached_corpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ShinkaEvolve retrieval seed.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the retrieval results JSON.",
    )
    args = parser.parse_args()
    payload = run_retrieval(save_path=args.save_path)
    print(f"Saved {len(payload['retrieval_save_list'])} retrieval results to {payload['save_file']}")


if __name__ == "__main__":
    main()
