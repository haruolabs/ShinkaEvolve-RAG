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
    """Returns the chunking and reranking policy to evolve."""
    return {
        "embed_model_name": "BAAI/bge-base-en-v1.5",
        "embed_batch_size": 2,
        "candidate_pool_size": 24,
        "top_k": 10,
        "query_sample_size": 256,
        "chunking": [
            {"label": "small", "chunk_size": 160, "chunk_overlap": 40},
            {"label": "medium", "chunk_size": 256, "chunk_overlap": 48},
        ],
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
        or token
        in {
            "ap",
            "bbc",
            "cnn",
            "guardian",
            "july",
            "june",
            "march",
            "may",
            "reuters",
            "techcrunch",
            "verge",
        }
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
    """Adds lightweight lexical, metadata, and diversity-aware reranking."""
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
import json
import random
from pathlib import Path
from uuid import uuid4

from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
CORPUS_PATH = REPO_ROOT / "dataset" / "corpus.json"
QUERY_PATH = REPO_ROOT / "dataset" / "MultiHopRAG.json"


def _load_json_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    return json.loads(raw_text)


def _load_llama_index_modules() -> Dict[str, Any]:
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import MetadataMode
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "This project expects the modern llama-index package layout available in the current shinka environment."
        ) from exc

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        HuggingFaceEmbedding = None

    return {
        "Document": Document,
        "HuggingFaceEmbedding": HuggingFaceEmbedding,
        "MetadataMode": MetadataMode,
        "OpenAIEmbedding": OpenAIEmbedding,
        "SentenceSplitter": SentenceSplitter,
        "Settings": Settings,
        "VectorStoreIndex": VectorStoreIndex,
    }


def _build_embed_model(
    modules: Dict[str, Any],
    model_name: str,
    strategy: Dict[str, Any],
) -> Any:
    if model_name.startswith("text-embedding-") or model_name.startswith("text-search-"):
        return modules["OpenAIEmbedding"](
            model=model_name,
            embed_batch_size=strategy["embed_batch_size"],
            num_workers=1,
            reuse_client=False,
            timeout=120.0,
            max_retries=10,
        )

    if modules["HuggingFaceEmbedding"] is None:
        raise RuntimeError(
            "A Hugging Face embedding model was requested, but llama-index-huggingface is not installed. "
            "Install it in the shinka environment, for example with: "
            "`uv pip install llama-index-embeddings-huggingface sentence-transformers`."
        )

    return modules["HuggingFaceEmbedding"](model_name=model_name, trust_remote_code=True)


def _load_documents(modules: Dict[str, Any]) -> List[Any]:
    corpus = _load_json_payload(CORPUS_PATH)
    documents = []
    for record in corpus:
        documents.append(
            modules["Document"](
                text=record.get("body", ""),
                metadata={
                    "title": record.get("title", ""),
                    "source": record.get("source", ""),
                    "published_at": record.get("published_at", ""),
                },
            )
        )
    return documents


def _build_nodes(
    modules: Dict[str, Any],
    documents: Sequence[Any],
    strategy: Dict[str, Any],
) -> List[Any]:
    nodes: List[Any] = []
    for spec in tqdm(strategy["chunking"], desc="Chunking corpus", unit="spec"):
        splitter = modules["SentenceSplitter"](
            chunk_size=spec["chunk_size"],
            chunk_overlap=spec["chunk_overlap"],
        )
        spec_nodes = splitter.get_nodes_from_documents(list(documents))
        for ordinal, node in enumerate(spec_nodes):
            node.metadata = dict(node.metadata or {})
            node.metadata.update(
                {
                    "chunk_label": spec["label"],
                    "chunk_overlap": spec["chunk_overlap"],
                    "chunk_ordinal": ordinal,
                    "chunk_size": spec["chunk_size"],
                }
            )
            node.excluded_llm_metadata_keys = []
            node.excluded_embed_metadata_keys = sorted(node.metadata.keys())
            nodes.append(node)
    return nodes


def _configure_settings(modules: Dict[str, Any], embed_model: Any) -> None:
    modules["Settings"].embed_model = embed_model
    modules["Settings"].llm = None


def _default_save_path() -> Path:
    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"retrieval_results_{uuid4().hex}.json"


def _sample_query_data(
    query_data: List[Dict[str, Any]],
    strategy: Dict[str, Any],
    seed: int | None,
) -> List[Dict[str, Any]]:
    sample_size = strategy.get("query_sample_size")
    if sample_size is None or sample_size <= 0 or sample_size >= len(query_data):
        return query_data

    rng = random.Random(0 if seed is None else seed)
    sampled_indices = sorted(rng.sample(range(len(query_data)), sample_size))
    return [query_data[index] for index in sampled_indices]


def run_retrieval(
    save_path: str | None = None,
    seed: int | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Builds the vector index, retrieves for all queries, and saves the payload."""
    strategy = get_retrieval_strategy()
    modules = _load_llama_index_modules()
    documents = _load_documents(modules)
    print(f"Loaded {len(documents)} corpus documents.")
    nodes = _build_nodes(modules, documents, strategy)

    if not nodes:
        raise RuntimeError("No nodes were created from the corpus.")

    print(f"Built {len(nodes)} index nodes.")
    embed_model = _build_embed_model(
        modules,
        strategy["embed_model_name"],
        strategy,
    )
    _configure_settings(modules, embed_model)
    print("Building vector index...")
    index = modules["VectorStoreIndex"](nodes, show_progress=True)
    retriever = index.as_retriever(
        similarity_top_k=max(strategy["candidate_pool_size"], strategy["top_k"])
    )

    full_query_data = _load_json_payload(QUERY_PATH)
    query_data = _sample_query_data(full_query_data, strategy, seed)
    print(
        f"Loaded {len(full_query_data)} queries and selected {len(query_data)} for this run."
    )
    metadata_mode = modules["MetadataMode"].LLM
    retrieval_save_list: List[Dict[str, Any]] = []

    for query_record in tqdm(query_data, desc="Retrieving queries", unit="query"):
        dense_candidates = []
        for rank, node_with_score in enumerate(
            retriever.retrieve(query_record["query"]),
            start=1,
        ):
            node = getattr(node_with_score, "node", node_with_score)
            rendered_text = node.get_content(metadata_mode=metadata_mode)
            dense_candidates.append(
                {
                    "text": rendered_text,
                    "dense_score": float(node_with_score.get_score() or 0.0),
                    "dense_rank": rank,
                    "chunk_label": node.metadata.get("chunk_label", ""),
                    "title": node.metadata.get("title", ""),
                    "source": node.metadata.get("source", ""),
                    "published_at": node.metadata.get("published_at", ""),
                    "normalized_text": " ".join(_tokenize(rendered_text)),
                }
            )

        reranked_candidates = rerank_candidates(
            query=query_record["query"],
            candidates=dense_candidates,
            strategy=strategy,
        )

        retrieval_list = []
        for final_rank, candidate in enumerate(reranked_candidates, start=1):
            retrieval_list.append(
                {
                    "text": candidate["text"],
                    "score": float(candidate.get("final_score", 0.0)),
                    "dense_rank": candidate["dense_rank"],
                    "dense_score": float(candidate.get("dense_score", 0.0)),
                    "chunk_label": candidate["chunk_label"],
                    "metadata": {
                        "title": candidate["title"],
                        "source": candidate["source"],
                        "published_at": candidate["published_at"],
                    },
                    "rank": final_rank,
                }
            )

        retrieval_save_list.append(
            {
                "query": query_record["query"],
                "answer": query_record.get("answer", ""),
                "question_type": query_record.get("question_type", ""),
                "retrieval_list": retrieval_list,
                "gold_list": query_record.get("evidence_list", []),
            }
        )

    target_path = Path(save_path) if save_path else _default_save_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(retrieval_save_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset": {
            "corpus_path": str(CORPUS_PATH),
            "query_path": str(QUERY_PATH),
        },
        "retrieval_save_list": retrieval_save_list,
        "save_file": str(target_path),
        "strategy": strategy,
    }


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
