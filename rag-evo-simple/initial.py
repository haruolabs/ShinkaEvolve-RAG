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
import json
import os
import random
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm


def _resolve_repo_root() -> Path:
    env_repo_root = os.environ.get("SHINKA_RAG_REPO_ROOT")
    candidates: List[Path] = []
    if env_repo_root:
        candidates.append(Path(env_repo_root).expanduser().resolve())

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates.extend([here, cwd, *here.parents, *cwd.parents])

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)

        if (candidate / "dataset" / "corpus.json").exists() and (
            candidate / "dataset" / "MultiHopRAG.json"
        ).exists():
            return candidate

        if candidate.name == "rag-evo-simple":
            repo_root = candidate.parent
            if (repo_root / "dataset" / "corpus.json").exists() and (
                repo_root / "dataset" / "MultiHopRAG.json"
            ).exists():
                return repo_root

    raise RuntimeError(
        "Could not locate the repository root containing dataset/corpus.json and dataset/MultiHopRAG.json."
    )


def _resolve_project_dir(repo_root: Path) -> Path:
    env_project_dir = os.environ.get("SHINKA_RAG_PROJECT_DIR")
    if env_project_dir:
        project_dir = Path(env_project_dir).expanduser().resolve()
        if project_dir.exists():
            return project_dir

    candidate = repo_root / "rag-evo-simple"
    if candidate.exists():
        return candidate

    here = Path(__file__).resolve().parent
    if here.name == "rag-evo-simple":
        return here

    raise RuntimeError("Could not locate the canonical rag-evo-simple project directory.")


REPO_ROOT = _resolve_repo_root()
PROJECT_DIR = _resolve_project_dir(REPO_ROOT)
CORPUS_PATH = REPO_ROOT / "dataset" / "corpus.json"
QUERY_PATH = REPO_ROOT / "dataset" / "MultiHopRAG.json"
CACHE_ROOT = PROJECT_DIR / "cache"

# Fixed corpus-side configuration to make cache reuse stable across evolutions.
CORPUS_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
FIXED_CHUNKING_SPECS = (
    {"label": "small", "chunk_size": 160, "chunk_overlap": 40},
    {"label": "medium", "chunk_size": 256, "chunk_overlap": 48},
)
CORPUS_EMBED_BATCH_SIZE = 32
DEFAULT_QUERY_SAMPLE_SIZE = -1 #256


def _load_json_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_llama_index_modules() -> Dict[str, Any]:
    try:
        from llama_index.core import Document
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
    }


def _build_embed_model(modules: Dict[str, Any], model_name: str) -> Any:
    if model_name.startswith("text-embedding-") or model_name.startswith("text-search-"):
        return modules["OpenAIEmbedding"](
            model=model_name,
            embed_batch_size=2,
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


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        return array
    return array / norm


def _cache_dir() -> Path:
    safe_model_name = CORPUS_EMBED_MODEL_NAME.replace("/", "-")
    chunk_name = "-".join(
        f"{spec['chunk_size']}_{spec['chunk_overlap']}" for spec in FIXED_CHUNKING_SPECS
    )
    return CACHE_ROOT / f"corpus-{safe_model_name}-{chunk_name}"


def _cache_files(cache_dir: Path) -> Dict[str, Path]:
    return {
        "meta": cache_dir / "meta.json",
        "nodes": cache_dir / "nodes.jsonl",
        "embeddings": cache_dir / "embeddings.npy",
    }


def _cache_metadata() -> Dict[str, Any]:
    corpus_stat = CORPUS_PATH.stat()
    return {
        "corpus_path": str(CORPUS_PATH),
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "corpus_embed_model_name": CORPUS_EMBED_MODEL_NAME,
        "fixed_chunking_specs": list(FIXED_CHUNKING_SPECS),
        "cache_version": 1,
    }


def _cache_is_valid(cache_dir: Path) -> bool:
    files = _cache_files(cache_dir)
    if not all(path.exists() and path.stat().st_size > 0 for path in files.values()):
        return False

    try:
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return meta == _cache_metadata()


def _load_documents(modules: Dict[str, Any]) -> List[Any]:
    corpus = _load_json_payload(CORPUS_PATH)
    return [
        modules["Document"](
            text=record.get("body", ""),
            metadata={
                "title": record.get("title", ""),
                "source": record.get("source", ""),
                "published_at": record.get("published_at", ""),
            },
        )
        for record in corpus
    ]


def _build_nodes(modules: Dict[str, Any], documents: Sequence[Any]) -> List[Any]:
    nodes: List[Any] = []
    for spec in tqdm(FIXED_CHUNKING_SPECS, desc="Chunking corpus", unit="spec"):
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


def _build_node_records(
    nodes: Sequence[Any],
    metadata_mode: Any,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for node in nodes:
        rendered_text = node.get_content(metadata_mode=metadata_mode)
        records.append(
            {
                "text": rendered_text,
                "title": node.metadata.get("title", ""),
                "source": node.metadata.get("source", ""),
                "published_at": node.metadata.get("published_at", ""),
                "chunk_label": node.metadata.get("chunk_label", ""),
                "chunk_size": node.metadata.get("chunk_size", 0),
                "chunk_overlap": node.metadata.get("chunk_overlap", 0),
                "chunk_ordinal": node.metadata.get("chunk_ordinal", 0),
            }
        )
    return records


def _embed_text_batch(embed_model: Any, texts: Sequence[str]) -> np.ndarray:
    embeddings = embed_model.get_text_embedding_batch(list(texts))
    return np.vstack([_normalize_vector(embedding) for embedding in embeddings])


def _build_corpus_cache(
    modules: Dict[str, Any],
    embed_model: Any,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    print("No cached corpus embeddings found. Building corpus cache...")
    documents = _load_documents(modules)
    print(f"Loaded {len(documents)} corpus documents.")
    nodes = _build_nodes(modules, documents)
    if not nodes:
        raise RuntimeError("No nodes were created from the corpus.")

    print(f"Built {len(nodes)} index nodes.")
    metadata_mode = modules["MetadataMode"].LLM
    node_records = _build_node_records(nodes, metadata_mode)
    node_texts = [record["text"] for record in node_records]

    embedding_batches = []
    for start in tqdm(
        range(0, len(node_texts), CORPUS_EMBED_BATCH_SIZE),
        desc="Embedding corpus nodes",
        unit="batch",
    ):
        batch_texts = node_texts[start : start + CORPUS_EMBED_BATCH_SIZE]
        embedding_batches.append(_embed_text_batch(embed_model, batch_texts))
    embeddings = np.vstack(embedding_batches).astype(np.float32)

    files = _cache_files(cache_dir)
    temp_dir = cache_dir.with_name(f"{cache_dir.name}_building")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_files = _cache_files(temp_dir)
    with temp_files["nodes"].open("w", encoding="utf-8") as output_file:
        for record in node_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(temp_files["embeddings"], embeddings)
    temp_files["meta"].write_text(
        json.dumps(_cache_metadata(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not _cache_is_valid(temp_dir):
        raise RuntimeError(f"Corpus cache is incomplete at {temp_dir}.")

    temp_dir.replace(cache_dir)
    print(f"Saved corpus cache to {cache_dir}")
    return node_records, embeddings


def _load_cached_corpus(cache_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    files = _cache_files(cache_dir)
    node_records = []
    with files["nodes"].open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                node_records.append(json.loads(line))
    embeddings = np.load(files["embeddings"]).astype(np.float32)
    if len(node_records) != embeddings.shape[0]:
        raise RuntimeError(
            "Cached corpus nodes and embeddings have mismatched lengths. Delete the cache and rebuild."
        )
    return node_records, embeddings


def _load_or_build_corpus_cache(
    modules: Dict[str, Any],
    embed_model: Any,
) -> tuple[list[dict[str, Any]], np.ndarray, bool]:
    cache_dir = _cache_dir()
    if _cache_is_valid(cache_dir):
        print(f"[CACHE] Reusing cached corpus embeddings from {cache_dir}")
        node_records, embeddings = _load_cached_corpus(cache_dir)
        print(
            f"[CACHE] Loaded {len(node_records)} cached corpus nodes with embedding matrix shape {embeddings.shape}."
        )
        return node_records, embeddings, True

    if cache_dir.exists():
        print(f"[CACHE] Discarding incomplete cached corpus data at {cache_dir}")
        shutil.rmtree(cache_dir)
    node_records, embeddings = _build_corpus_cache(modules, embed_model, cache_dir)
    print(
        f"[CACHE] Built and saved corpus cache with {len(node_records)} nodes at {cache_dir}."
    )
    return node_records, embeddings, False


def _default_save_path() -> Path:
    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"retrieval_results_{uuid4().hex}.json"


def _sample_query_data(
    query_data: List[Dict[str, Any]],
    seed: int | None,
) -> List[Dict[str, Any]]:
    if DEFAULT_QUERY_SAMPLE_SIZE is None or DEFAULT_QUERY_SAMPLE_SIZE <= 0:
        return query_data
    if DEFAULT_QUERY_SAMPLE_SIZE >= len(query_data):
        return query_data

    rng = random.Random(0 if seed is None else seed)
    sampled_indices = sorted(rng.sample(range(len(query_data)), DEFAULT_QUERY_SAMPLE_SIZE))
    return [query_data[index] for index in sampled_indices]


def _retrieve_dense_candidates(
    query: str,
    embed_model: Any,
    node_records: Sequence[Dict[str, Any]],
    corpus_embeddings: np.ndarray,
    top_n: int,
) -> List[Dict[str, Any]]:
    query_embedding = _normalize_vector(embed_model.get_query_embedding(query))
    similarity_scores = corpus_embeddings @ query_embedding

    candidate_count = min(top_n, len(node_records))
    if candidate_count <= 0:
        return []

    if candidate_count == len(node_records):
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        top_indices = np.argpartition(similarity_scores, -candidate_count)[-candidate_count:]
        top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

    candidates = []
    for dense_rank, index in enumerate(top_indices, start=1):
        record = node_records[int(index)]
        candidates.append(
            {
                "text": record["text"],
                "dense_score": float(similarity_scores[int(index)]),
                "dense_rank": dense_rank,
                "chunk_label": record["chunk_label"],
                "title": record["title"],
                "source": record["source"],
                "published_at": record["published_at"],
                "normalized_text": " ".join(_tokenize(record["text"])),
            }
        )
    return candidates


def run_retrieval(
    save_path: str | None = None,
    seed: int | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Loads or builds the fixed corpus cache, then retrieves for a sampled query set."""
    strategy = get_retrieval_strategy()
    modules = _load_llama_index_modules()

    embed_model = _build_embed_model(modules, CORPUS_EMBED_MODEL_NAME)
    node_records, corpus_embeddings, used_cached_corpus = _load_or_build_corpus_cache(
        modules,
        embed_model,
    )
    if used_cached_corpus:
        print("[CACHE] Retrieval will use the cached corpus embeddings for this run.")
    else:
        print("[CACHE] Retrieval will use the newly built corpus embeddings for this run.")

    full_query_data = _load_json_payload(QUERY_PATH)
    query_data = _sample_query_data(full_query_data, seed)
    print(
        f"Loaded {len(full_query_data)} queries and selected {len(query_data)} for this run."
    )

    retrieval_save_list: List[Dict[str, Any]] = []
    candidate_pool_size = max(strategy["candidate_pool_size"], strategy["top_k"])

    for query_record in tqdm(query_data, desc="Retrieving queries", unit="query"):
        dense_candidates = _retrieve_dense_candidates(
            query=query_record["query"],
            embed_model=embed_model,
            node_records=node_records,
            corpus_embeddings=corpus_embeddings,
            top_n=candidate_pool_size,
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
        "strategy": {
            **strategy,
            "corpus_embed_model_name": CORPUS_EMBED_MODEL_NAME,
            "fixed_chunking": list(FIXED_CHUNKING_SPECS),
            "query_sample_size": DEFAULT_QUERY_SAMPLE_SIZE,
            "cache_dir": str(_cache_dir()),
            "used_cached_corpus": used_cached_corpus,
        },
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
