#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

search_task_sys_msg = """You are an expert in information retrieval, multi-hop evidence retrieval, and retrieval-augmented generation systems.

Your task is to improve the retrieval policy for the MultiHop-RAG benchmark by editing only the EVOLVE block in initial.py.

What is fixed in this refactored project:
1. The runtime outside the EVOLVE block is fixed.
2. Corpus loading, corpus embeddings, cache format, embedding model, fixed chunking, payload schema, and run_retrieval() interface must not change.
3. For each query, the runtime first retrieves a dense candidate pool using the original raw query, then calls your reranking logic.
4. You are not changing the index, retriever implementation, or corpus preprocessing. You are improving how the dense shortlist is scored and selected.

Your actual search space inside the EVOLVE block:
1. Retrieval strategy parameters such as candidate_pool_size, top_k, weights, thresholds, and penalties.
2. Query feature extraction used for reranking.
3. Candidate scoring and selection logic.
4. Lightweight helper functions that remain deterministic, valid Python, and cheap to execute.

Candidate fields available during reranking:
1. text
2. dense_score
3. dense_rank
4. chunk_label
5. title
6. source
7. published_at
8. normalized_text

Optimization target:
1. The evaluator reports Hits@4, Hits@10, MAP@10, and MRR@10.
2. The scalar combined_score used by evolution is MAP@10, so optimize primarily for MAP@10 while keeping top-rank quality strong.
3. The evaluator gives credit when gold evidence substrings appear inside retrieved text after whitespace/newline removal.
4. Exact evidence-bearing chunks matter much more than paraphrases or semantically similar summaries.

Useful guidance for this benchmark:
1. Favor reranking strategies that move exact evidence-bearing chunks into the top 10, especially the highest ranks.
2. Reward strong lexical overlap, phrase-level clues, rare clue tokens, and metadata matches from title, source, and date when supported by the query.
3. Encourage complementary evidence across different titles/articles, but do not over-penalize same-article chunks if they likely contain distinct gold evidence.
4. Moderate increases to candidate_pool_size are allowed if they improve reranking opportunities, but avoid very large values that slow evaluation.
5. Query-side analysis is useful only insofar as it helps reranking; do not rely on changing the dense retrieval query itself.
6. Prefer simple, robust, efficient heuristics over complex or brittle logic.

Avoid:
1. Any edits outside the EVOLVE block.
2. Chunking, indexing, embedding-model, cache, dataset-path, or runtime refactors.
3. External dependencies, LLM calls, or expensive logic.
4. Rewriting, summarizing, or otherwise altering retrieved text.
5. Suggestions based on components that are not actually available in the refactored program.

Be creative, but stay grounded in the real search space of this refactored project: better use of a fixed dense shortlist through stronger reranking, evidence-aware scoring, and smarter diversity/duplication control."""


def main(config_path: str) -> None:
    project_dir = Path(__file__).resolve().parent
    os.environ.setdefault("SHINKA_RAG_PROJECT_DIR", str(project_dir))
    os.environ.setdefault("SHINKA_RAG_REPO_ROOT", str(project_dir.parent))

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    evo_config = EvolutionConfig(**config["evo_config"])
    job_kwargs = dict(config.get("job_config", {}))
    job_kwargs.setdefault("eval_program_path", "evaluate.py")
    job_kwargs.setdefault("time", "00:20:00")
    job_config = LocalJobConfig(**job_kwargs)
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    args = parser.parse_args()
    main(args.config_path)
