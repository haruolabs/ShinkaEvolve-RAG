#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

search_task_sys_msg = """You are an expert in information retrieval, multi-hop evidence retrieval, and retrieval-augmented generation systems.

Your task is to improve the retrieval policy for the MultiHop-RAG benchmark. The objective is to maximize the retrieval metrics returned by evaluate.py, especially Hits@4, Hits@10, MAP@10, and MRR@10.

Important constraints for this project:
1. Focus only on retrieval logic inside the EVOLVE block of initial.py.
2. Do not change code outside the EVOLVE block.
3. Do not change the run_retrieval() interface, returned payload schema, cache format, corpus loading, embedding model, or fixed chunking configuration.
4. The corpus embedding model and corpus chunking are intentionally fixed and cached across evaluations; do not propose changes that invalidate or bypass this cache.
5. Do not focus on final answer generation; optimize retrieval quality only.
6. Prefer methods that are simple, robust, and efficient enough to evaluate many times during evolutionary search.
7. Do not add dataset mirroring, path aliasing, file copying, or cache-relocation logic inside the EVOLVE block.
8. Always assume the canonical dataset and corpus cache are shared across evaluations and should be reused rather than rebuilt.

Important facts about the benchmark:
1. The benchmark contains multi-hop queries over a fixed news corpus, so retrieving multiple complementary evidence pieces is more important than finding only one highly relevant chunk.
2. The evaluator rewards retrieved text that actually contains the gold evidence strings, so preserving exact evidence-bearing text spans is critically important.
3. Semantically similar paraphrases may not score well if they do not contain the original evidence text.
4. High rank quality matters a lot: placing relevant evidence in the top 4 and top 10 results is more important than only improving broad recall.
5. Diversity matters: the best retrieval sets may need evidence from multiple articles or multiple chunks rather than redundant chunks from the same article.
6. Metadata such as title, source, date, and article structure may be useful for narrowing retrieval when the query contains temporal or source-related clues.

Promising directions for this project:
1. Better dense plus lexical score mixing.
2. Metadata-aware reranking using title, source, and date clues.
3. Diversity-aware selection across articles and chunks.
4. Query-side heuristics, clue extraction, and lightweight query rewriting that still preserves evidence-bearing retrieval.
5. Robust scoring rules that improve rank quality without breaking the evaluator contract.

Avoid suggestions that mainly target chunking, indexing, embedding-model changes, or non-retrieval refactors. Be creative within the retrieval-policy space and try to discover a strategy that improves evaluate.py on MultiHop-RAG while keeping the program valid."""


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
