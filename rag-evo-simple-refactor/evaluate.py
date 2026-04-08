"""Evaluator for the ShinkaEvolve MultiHop-RAG retrieval project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def calculate_metrics(
    retrieved_lists: Sequence[Sequence[str]],
    gold_lists: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Replicates the repository retrieval metric calculation."""
    hits_at_10_count = 0
    hits_at_4_count = 0
    map_at_10_list: List[float] = []
    mrr_list: List[float] = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        hits_at_10_flag = False
        hits_at_4_flag = False
        average_precision_sum = 0.0
        first_relevant_rank = None
        find_gold: List[str] = []

        gold_normalized = [
            item.replace(" ", "").replace("\n", "")
            for item in gold
            if item
        ]
        retrieved_normalized = [
            item.replace(" ", "").replace("\n", "")
            for item in retrieved
            if item
        ]

        for rank, retrieved_item in enumerate(retrieved_normalized[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold_normalized):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 4:
                        hits_at_4_flag = True

                    count = 0
                    for gold_item in gold_normalized:
                        if gold_item in retrieved_item and gold_item not in find_gold:
                            count += 1
                            find_gold.append(gold_item)
                    average_precision_sum += count / rank

        hits_at_10_count += int(hits_at_10_flag)
        hits_at_4_count += int(hits_at_4_flag)
        denominator = min(len(gold_normalized), 10) or 1
        map_at_10_list.append(average_precision_sum / denominator)
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

    num_queries = len(gold_lists) or 1
    return {
        "Hits@10": hits_at_10_count / num_queries,
        "Hits@4": hits_at_4_count / num_queries,
        "MAP@10": sum(map_at_10_list) / num_queries,
        "MRR@10": sum(mrr_list) / num_queries,
    }


def evaluate_retrieval_results(
    retrieval_save_list: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    retrieved_lists: List[List[str]] = []
    gold_lists: List[List[str]] = []
    skipped_null_queries = 0

    for item in retrieval_save_list:
        if item.get("question_type") == "null_query":
            skipped_null_queries += 1
            continue
        retrieved_lists.append(
            [entry.get("text", "") for entry in item.get("retrieval_list", [])]
        )
        gold_lists.append(
            [
                gold_item.get("fact", "")
                for gold_item in item.get("gold_list", [])
                if isinstance(gold_item, dict)
            ]
        )

    metrics = calculate_metrics(retrieved_lists, gold_lists)
    counts = {
        "num_queries_total": len(retrieval_save_list),
        "num_queries_scored": len(gold_lists),
        "num_null_queries": skipped_null_queries,
    }
    return metrics, counts


def validate_retrieval_run(run_output: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, dict):
        return False, "run_retrieval() must return a dictionary payload."

    retrieval_save_list = run_output.get("retrieval_save_list")
    if not isinstance(retrieval_save_list, list) or not retrieval_save_list:
        return False, "The payload must include a non-empty retrieval_save_list."

    required_keys = {"query", "question_type", "retrieval_list", "gold_list"}
    first_item = retrieval_save_list[0]
    missing_keys = sorted(required_keys - set(first_item.keys()))
    if missing_keys:
        return False, f"retrieval_save_list entries are missing keys: {missing_keys}"

    retrieval_list = first_item.get("retrieval_list")
    if not isinstance(retrieval_list, list):
        return False, "Each retrieval result must include a retrieval_list array."

    if retrieval_list:
        first_retrieved = retrieval_list[0]
        if "text" not in first_retrieved:
            return False, "Each retrieved item must include a text field."

    return True, "Retrieval payload looks valid."


def aggregate_retrieval_metrics(
    results: Sequence[Dict[str, Any]],
    results_dir: str,
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    run_output = results[0]
    retrieval_save_list = run_output["retrieval_save_list"]
    metrics, counts = evaluate_retrieval_results(retrieval_save_list)

    combined_score = metrics["MAP@10"] # This can be adjusted to weight different metrics but let's use MAP@10 as the primary score for now.

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    saved_results_path = results_path / "retrieval_results.json"
    saved_results_path.write_text(
        json.dumps(retrieval_save_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    extra_payload = {
        "counts": counts,
        "dataset": run_output.get("dataset", {}),
        "metrics": metrics,
        "save_file": run_output.get("save_file"),
        "strategy": run_output.get("strategy", {}),
    }
    extra_path = results_path / "extra.json"
    extra_path.write_text(
        json.dumps(extra_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "combined_score": combined_score,
        "public": {
            **metrics,
            **counts,
        },
        "private": {
            "artifact_file": str(saved_results_path),
            "extra_file": str(extra_path),
            "source_save_file": run_output.get("save_file"),
            "strategy": run_output.get("strategy", {}),
        },
    }


def main(program_path: str, results_dir: str) -> None:
    try:
        from shinka.core import run_shinka_eval
    except ImportError as exc:
        raise RuntimeError(
            "The shinka package must be installed to run this evaluator."
        ) from exc

    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    def _aggregator_with_context(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_retrieval_metrics(results, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_retrieval",
        num_runs=1,
        validate_fn=validate_retrieval_run,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and validation completed successfully.")
    else:
        print(f"Evaluation or validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a MultiHop-RAG retrieval program with ShinkaEvolve."
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program to evaluate (must expose run_retrieval).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save metrics and retrieval artifacts.",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
