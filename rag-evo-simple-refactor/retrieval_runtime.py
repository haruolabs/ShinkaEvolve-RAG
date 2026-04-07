"""Fixed runtime helpers for the refactored MultiHop-RAG retrieval example."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm


CORPUS_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
FIXED_CHUNKING_SPECS = (
    {"label": "small", "chunk_size": 160, "chunk_overlap": 40},
    {"label": "medium", "chunk_size": 256, "chunk_overlap": 48},
)
CORPUS_EMBED_BATCH_SIZE = 32
DEFAULT_QUERY_SAMPLE_SIZE = -1  # 256


@dataclass(frozen=True)
class RuntimePaths:
    project_dir: Path
    repo_root: Path
    corpus_path: Path
    query_path: Path
    cache_root: Path


def _resolve_repo_root(project_dir: Path) -> Path:
    env_repo_root = os.environ.get("SHINKA_RAG_REPO_ROOT")
    candidates: List[Path] = []
    if env_repo_root:
        candidates.append(Path(env_repo_root).expanduser().resolve())

    cwd = Path.cwd().resolve()
    candidates.extend([project_dir, cwd, *project_dir.parents, *cwd.parents])

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

    raise RuntimeError(
        "Could not locate the repository root containing dataset/corpus.json and dataset/MultiHopRAG.json."
    )


def _build_runtime_paths() -> RuntimePaths:
    project_dir = Path(__file__).resolve().parent
    repo_root = _resolve_repo_root(project_dir)
    return RuntimePaths(
        project_dir=project_dir,
        repo_root=repo_root,
        corpus_path=repo_root / "dataset" / "corpus.json",
        query_path=repo_root / "dataset" / "MultiHopRAG.json",
        cache_root=project_dir / "cache",
    )


RUNTIME_PATHS = _build_runtime_paths()


def load_json_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_llama_index_modules() -> Dict[str, Any]:
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


def build_embed_model(modules: Dict[str, Any], model_name: str) -> Any:
    if model_name.startswith("text-embedding-") or model_name.startswith(
        "text-search-"
    ):
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

    return modules["HuggingFaceEmbedding"](
        model_name=model_name,
        trust_remote_code=True,
    )


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        return array
    return array / norm


def cache_dir() -> Path:
    safe_model_name = CORPUS_EMBED_MODEL_NAME.replace("/", "-")
    chunk_name = "-".join(
        f"{spec['chunk_size']}_{spec['chunk_overlap']}" for spec in FIXED_CHUNKING_SPECS
    )
    return RUNTIME_PATHS.cache_root / f"corpus-{safe_model_name}-{chunk_name}"


def _cache_files(target_cache_dir: Path) -> Dict[str, Path]:
    return {
        "meta": target_cache_dir / "meta.json",
        "nodes": target_cache_dir / "nodes.jsonl",
        "embeddings": target_cache_dir / "embeddings.npy",
    }


def _cache_metadata() -> Dict[str, Any]:
    corpus_stat = RUNTIME_PATHS.corpus_path.stat()
    return {
        "corpus_path": str(RUNTIME_PATHS.corpus_path),
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "corpus_embed_model_name": CORPUS_EMBED_MODEL_NAME,
        "fixed_chunking_specs": list(FIXED_CHUNKING_SPECS),
        "cache_version": 1,
    }


def _cache_is_valid(target_cache_dir: Path) -> bool:
    files = _cache_files(target_cache_dir)
    if not all(path.exists() and path.stat().st_size > 0 for path in files.values()):
        return False

    try:
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return meta == _cache_metadata()


def _load_documents(modules: Dict[str, Any]) -> List[Any]:
    corpus = load_json_payload(RUNTIME_PATHS.corpus_path)
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
    return np.vstack([normalize_vector(embedding) for embedding in embeddings])


def _build_corpus_cache(
    modules: Dict[str, Any],
    embed_model: Any,
    target_cache_dir: Path,
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

    files = _cache_files(target_cache_dir)
    temp_dir = target_cache_dir.with_name(f"{target_cache_dir.name}_building")
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

    temp_dir.replace(target_cache_dir)
    print(f"Saved corpus cache to {target_cache_dir}")
    return node_records, embeddings


def _load_cached_corpus(target_cache_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    files = _cache_files(target_cache_dir)
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


def load_or_build_corpus_cache(
    modules: Dict[str, Any],
    embed_model: Any,
) -> tuple[list[dict[str, Any]], np.ndarray, bool]:
    target_cache_dir = cache_dir()
    if _cache_is_valid(target_cache_dir):
        print(f"[CACHE] Reusing cached corpus embeddings from {target_cache_dir}")
        node_records, embeddings = _load_cached_corpus(target_cache_dir)
        print(
            f"[CACHE] Loaded {len(node_records)} cached corpus nodes with embedding matrix shape {embeddings.shape}."
        )
        return node_records, embeddings, True

    if target_cache_dir.exists():
        print(f"[CACHE] Discarding incomplete cached corpus data at {target_cache_dir}")
        shutil.rmtree(target_cache_dir)
    node_records, embeddings = _build_corpus_cache(
        modules,
        embed_model,
        target_cache_dir,
    )
    print(
        f"[CACHE] Built and saved corpus cache with {len(node_records)} nodes at {target_cache_dir}."
    )
    return node_records, embeddings, False


def default_save_path() -> Path:
    output_dir = RUNTIME_PATHS.project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"retrieval_results_{uuid4().hex}.json"


def sample_query_data(
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


def retrieve_dense_candidates(
    query: str,
    embed_model: Any,
    node_records: Sequence[Dict[str, Any]],
    corpus_embeddings: np.ndarray,
    top_n: int,
    tokenize: Callable[[str], Sequence[str]],
) -> List[Dict[str, Any]]:
    query_embedding = normalize_vector(embed_model.get_query_embedding(query))
    similarity_scores = corpus_embeddings @ query_embedding

    candidate_count = min(top_n, len(node_records))
    if candidate_count <= 0:
        return []

    if candidate_count == len(node_records):
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        top_indices = np.argpartition(similarity_scores, -candidate_count)[
            -candidate_count:
        ]
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
                "normalized_text": " ".join(tokenize(record["text"])),
            }
        )
    return candidates


def format_retrieval_item(candidate: Dict[str, Any], final_rank: int) -> Dict[str, Any]:
    return {
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


def format_query_result(
    query_record: Dict[str, Any],
    retrieval_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query_record["query"],
        "answer": query_record.get("answer", ""),
        "question_type": query_record.get("question_type", ""),
        "retrieval_list": list(retrieval_list),
        "gold_list": query_record.get("evidence_list", []),
    }


def write_retrieval_results(
    retrieval_save_list: Sequence[Dict[str, Any]],
    save_path: str | Path | None = None,
) -> Path:
    target_path = Path(save_path) if save_path else default_save_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(list(retrieval_save_list), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def build_run_payload(
    retrieval_save_list: Sequence[Dict[str, Any]],
    save_file: Path,
    strategy: Dict[str, Any],
    used_cached_corpus: bool,
) -> Dict[str, Any]:
    return {
        "dataset": {
            "corpus_path": str(RUNTIME_PATHS.corpus_path),
            "query_path": str(RUNTIME_PATHS.query_path),
        },
        "retrieval_save_list": list(retrieval_save_list),
        "save_file": str(save_file),
        "strategy": {
            **strategy,
            "corpus_embed_model_name": CORPUS_EMBED_MODEL_NAME,
            "fixed_chunking": list(FIXED_CHUNKING_SPECS),
            "query_sample_size": DEFAULT_QUERY_SAMPLE_SIZE,
            "cache_dir": str(cache_dir()),
            "used_cached_corpus": used_cached_corpus,
        },
    }
