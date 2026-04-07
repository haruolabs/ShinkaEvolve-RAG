# ShinkaEvolve MultiHop-RAG Example

This directory is a compact ShinkaEvolve project for improving retrieval on the repo's `MultiHop-RAG` dataset.

At a high level, the code does three things:

1. `initial.py` defines a baseline retrieval program and exposes `run_retrieval()`.
2. `evaluate.py` runs that program through Shinka's evaluation harness and converts raw retrieval results into retrieval metrics.
3. `run_evo.py` launches an evolutionary search loop that repeatedly edits the baseline program, evaluates candidates, and stores the best ones.

The setup is intentionally small, but the moving pieces are important because this directory is both:

- a runnable retrieval baseline, and
- an optimization target for ShinkaEvolve.

## Directory Map

- `initial.py`: seed retrieval program that Shinka evolves.
- `evaluate.py`: validator, scorer, and artifact writer for retrieval runs.
- `run_evo.py`: evolution runner entrypoint.
- `shinka_smoke.yaml`: tiny smoke-test configuration.
- `shinka_small.yaml`: fuller development configuration.
- `output/`: direct retrieval outputs written by `initial.py`.
- `results/`: evaluator outputs and ShinkaEvolve run artifacts.

## Execution Flow

The normal workflow looks like this:

```text
run_evo.py
  -> loads YAML config
  -> constructs Shinka runner
  -> asks Shinka to mutate the seed program
  -> candidate program exposes run_retrieval()
  -> evaluate.py validates and scores it
  -> metrics + artifacts are written under results/
```

If you are not running evolution, the simpler path is:

```text
initial.py
  -> builds index over ../dataset/corpus.json
  -> retrieves evidence for every query in ../dataset/MultiHopRAG.json
  -> saves raw retrieval output to output/*.json

evaluate.py
  -> calls run_retrieval() in a target program
  -> checks the payload shape
  -> computes Hits@4, Hits@10, MAP@10, and MRR@10
  -> saves retrieval artifacts + summary files
```

## Environment And Inputs

This example depends on assets outside this folder:

- Dataset corpus: [`/Users/hlabs/dev/ShinkaEvolve-RAG/dataset/corpus.json`](/Users/hlabs/dev/ShinkaEvolve-RAG/dataset/corpus.json)
- Query set: [`/Users/hlabs/dev/ShinkaEvolve-RAG/dataset/MultiHopRAG.json`](/Users/hlabs/dev/ShinkaEvolve-RAG/dataset/MultiHopRAG.json)

It also expects Python packages that are not vendored here:

- `shinka`
- `llama-index` with the modern module layout used in `initial.py`
- `llama-index-embeddings-openai`
- `llama-index-embeddings-huggingface`
- `tqdm`

Important note: the repo-level [`/Users/hlabs/dev/ShinkaEvolve-RAG/requirement.txt`](/Users/hlabs/dev/ShinkaEvolve-RAG/requirement.txt) pins `llama-index==0.9.40`, but [`/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/initial.py`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/initial.py) imports the newer `llama_index.core` package layout. In practice, this example wants the newer layout available in the Shinka environment.

The default embedding model documented for this example is `BAAI/bge-base-en-v1.5`, so the expected default path is the Hugging Face embedding backend rather than OpenAI embeddings.

If you switch the strategy to an OpenAI embedding model, you should expect to need `OPENAI_API_KEY` configured before running retrieval or evolution.

## Script-By-Script Explanation

### `initial.py`

[`initial.py`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/initial.py) is the baseline retrieval program. It is the file Shinka mutates during evolution, and it is also a normal standalone script you can run yourself.

Its responsibilities break down into five parts.

#### 1. Define the retrieval policy

The top of the file contains `get_retrieval_strategy()`, which returns a dictionary of tunable retrieval settings:

- embedding model name
- embedding batch size
- candidate pool size
- final `top_k`
- chunking specs
- token filtering thresholds
- reranking weights
- penalties for duplicates and low diversity

These values are the starting point for search. The `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers tell Shinka which part of the file it is allowed to modify.

In the default setup described by this README, the embedding model is `BAAI/bge-base-en-v1.5`.

#### 2. Create lightweight query features

The helper functions `_tokenize()` and `_query_features()` extract:

- normalized query tokens
- filtered tokens that ignore very common words
- metadata-oriented tokens such as dates, news sources, and source-like clues

This is used to add lexical and metadata-aware signals on top of dense retrieval.

#### 3. Rerank dense candidates

The baseline does not trust dense similarity alone. It first retrieves a larger candidate pool, then reranks those candidates with `_candidate_score()` and `rerank_candidates()`.

The reranker blends:

- dense embedding score
- lexical overlap between query and chunk text
- metadata matches from title, source, and publication date
- diversity control by limiting chunks from the same title
- duplicate penalties for near-repeated text

This is important for MultiHop-RAG because many queries need evidence from multiple documents, not just one very strong chunk.

#### 4. Build the retrieval index

Outside the evolve block, the file handles all runtime plumbing:

- `_load_json_payload()` loads JSON safely from disk.
- `_load_llama_index_modules()` imports the required LlamaIndex classes and raises a helpful error if the environment is missing them.
- `_build_embed_model()` selects either OpenAI embeddings or a Hugging Face embedding backend.
- `_load_documents()` converts the corpus into LlamaIndex `Document` objects with metadata.
- `_build_nodes()` chunks each document according to every chunking spec in the strategy and annotates each node with chunk metadata.
- `_configure_settings()` sets the embedding model and disables any LLM use in `Settings`.

One useful design detail: the code indexes the same corpus at multiple chunk sizes by looping over the `chunking` list and combining all generated nodes into one index. That gives the retriever access to mixed-granularity evidence.

#### 5. Run retrieval for the full benchmark

`run_retrieval()` is the main programmatic entrypoint. This is the function Shinka expects to exist when it evaluates a candidate.

It does the following:

1. Loads the current strategy.
2. Loads the dataset and corpus.
3. Builds chunked nodes and a `VectorStoreIndex`.
4. Creates a retriever with `similarity_top_k = max(candidate_pool_size, top_k)`.
5. Iterates over every query in `MultiHopRAG.json`.
6. Collects dense candidates with text, metadata, dense score, and dense rank.
7. Reranks them with the custom reranker.
8. Builds a `retrieval_list` for each query.
9. Saves all query-level retrieval outputs to JSON.
10. Returns a dictionary payload with dataset info, strategy, save path, and the full retrieval results.

The returned payload shape matters because `evaluate.py` validates it before scoring.

#### Output schema from `run_retrieval()`

The returned dictionary contains:

- `dataset`: paths to the corpus and query files
- `retrieval_save_list`: one entry per query
- `save_file`: the JSON file written to disk
- `strategy`: the retrieval strategy that produced the results

Each `retrieval_save_list` item contains:

- `query`
- `answer`
- `question_type`
- `retrieval_list`
- `gold_list`

Each retrieved item inside `retrieval_list` contains:

- `text`
- `score`
- `dense_rank`
- `dense_score`
- `chunk_label`
- `metadata`
- `rank`

This structure is what the evaluator uses to compute retrieval metrics.

#### CLI behavior

Running the file directly:

```bash
cd /Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple
python initial.py --save_path output/manual_run.json
```

If `--save_path` is omitted, the file writes to `output/retrieval_results_<uuid>.json`.

### `evaluate.py`

[`evaluate.py`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/evaluate.py) is the scoring layer between a retrieval program and Shinka's evolutionary loop.

Its job is not to build an index itself. Instead, it imports and runs a target program that exposes `run_retrieval()`, then turns that program's output into metrics and artifacts.

#### Metric computation

`calculate_metrics()` implements the retrieval metrics used for this task:

- `Hits@10`
- `Hits@4`
- `MAP@10`
- `MRR@10`

The scoring logic normalizes strings by removing spaces and newlines, then checks whether a gold evidence string is contained inside each retrieved text chunk. That means exact evidence-bearing text matters a lot. A semantically similar chunk may still score poorly if it does not include the gold substring.

This detail strongly shapes the search problem and explains why the baseline preserves raw evidence text instead of summarizing or rewriting chunks.

#### Filtering and validation

`evaluate_retrieval_results()` skips queries whose `question_type` is `null_query`, then prepares the retrieved texts and gold facts for scoring.

`validate_retrieval_run()` makes sure the target program returned the expected structure:

- top-level result must be a dictionary
- `retrieval_save_list` must exist and be non-empty
- each item must include `query`, `question_type`, `retrieval_list`, and `gold_list`
- retrieved items must contain `text`

This prevents Shinka from treating malformed outputs as successful evaluations.

#### Aggregation and artifact writing

`aggregate_retrieval_metrics()`:

- computes retrieval metrics for the first run
- averages the four public metrics into a single `combined_score`
- writes `retrieval_results.json`
- writes `extra.json` with counts, dataset info, metrics, source save path, and strategy

The `combined_score` is:

```text
(Hits@4 + Hits@10 + MAP@10 + MRR@10) / 4
```

That single number is what Shinka uses as the optimization objective.

#### Shinka integration

`main()` imports `run_shinka_eval` from `shinka.core` and passes:

- `program_path`
- `results_dir`
- `experiment_fn_name="run_retrieval"`
- `num_runs=1`
- `validate_fn=validate_retrieval_run`
- `aggregate_metrics_fn=...`

So from Shinka's point of view, this file is the task-specific adapter that says:

"Run `run_retrieval()` from the candidate program, make sure the payload is valid, then score it using retrieval metrics."

#### CLI behavior

Example:

```bash
cd /Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple
python evaluate.py --program_path initial.py --results_dir results/manual_eval
```

That command evaluates the current baseline without starting the evolutionary loop.

### `run_evo.py`

[`run_evo.py`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/run_evo.py) is the orchestration entrypoint for ShinkaEvolve.

It does three key things:

#### 1. Load a YAML config

`main(config_path)` reads a YAML file and splits it into:

- `evo_config`
- `job_config`
- `db_config`
- top-level worker counts such as `max_evaluation_jobs`

#### 2. Inject the task prompt

The `search_task_sys_msg` constant is a detailed system message describing how the LLM should improve retrieval for this benchmark.

It tells the model that:

- the task is retrieval-only
- top-4 and top-10 ranking quality matter
- exact evidence text matters
- multi-document diversity matters
- metadata-aware retrieval is useful
- chunking is a major design lever

This prompt is inserted into `config["evo_config"]["task_sys_msg"]` before the Shinka objects are created, so the YAML does not need to repeat it.

#### 3. Build and run the Shinka components

The file converts config sections into:

- `EvolutionConfig`
- `LocalJobConfig`
- `DatabaseConfig`

Then it creates `ShinkaEvolveRunner(...)` and calls `runner.run()`.

There are also a couple of practical defaults:

- `eval_program_path` defaults to `evaluate.py`
- `time` defaults to `00:20:00`

So even if the YAML omits those values, the runner still points at this task's evaluator.

#### CLI behavior

Example:

```bash
cd /Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple
python run_evo.py --config_path shinka_small.yaml
```

That launches a local Shinka evolution run using the baseline in `initial.py`.

## Config Files

### `shinka_smoke.yaml`

[`shinka_smoke.yaml`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/shinka_smoke.yaml) is the smallest useful profile. It is meant to verify that the end-to-end loop works.

Key characteristics:

- one evaluation worker
- one proposal worker
- one DB worker
- one island
- archive size `10`
- `10` generations
- only `gpt-5.4-mini` for proposals
- prompt evolution disabled
- results written to `results/results_rag_evo_simple_smoke`

Use this when you want a cheap sanity check before committing to a longer run.

### `shinka_small.yaml`

[`shinka_small.yaml`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/shinka_small.yaml) is the more complete development profile.

Compared with the smoke config, it increases search breadth:

- more proposal and DB workers
- larger archive
- `100` generations
- multiple proposal models
- multiple temperatures and reasoning levels
- dynamic island spawning enabled
- prompt evolution enabled
- results written to `results/results_rag_evo_simple_small`

This is the profile to read if you want to understand how the search strategy is tuned.

## Generated Outputs

### `output/`

This folder is for direct runs of `initial.py`.

Examples currently present:

- [`output/smoke_test_results.json`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/output/smoke_test_results.json)
- [`output/tiny_smoke_test_results.json`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/output/tiny_smoke_test_results.json)
- [`output/tiny_smoke_test_results_refactor.json`](/Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple/output/tiny_smoke_test_results_refactor.json)

These are raw retrieval payloads in the same shape returned by `run_retrieval()`.

### `results/`

This folder contains evaluator outputs and evolution artifacts.

You can already see two kinds of contents here:

- manual evaluation output such as `results/manual_initial_eval`
- full Shinka run state such as `results/results_rag_evo_simple_smoke`

Inside a Shinka run directory you may see:

- `programs.sqlite`: candidate/archive database
- `bandit_state.pkl`: dynamic model-selection state
- `evolution_run.log`: run log
- generation folders such as `gen_0`, `gen_1`, `gen_2`

Inside each generation folder you may find:

- `main.py`: the candidate program evaluated for that generation
- `original.py`: parent or pre-edit version
- `edit.diff`: applied code diff
- `rewrite.txt`: model-generated reasoning or rewrite notes
- `attempts/...`: proposal attempt metadata and raw patches
- `results/metrics.json`: final metrics for that candidate
- `results/correct.json`: validation/execution status
- `results/job_log.out` and `results/job_log.err`: execution logs

That folder structure makes it possible to inspect not only the best code, but also how Shinka got there.

## Practical Commands

From this directory:

```bash
cd /Users/hlabs/dev/ShinkaEvolve-RAG/rag-evo-simple
```

Run the baseline retrieval program:

```bash
python initial.py --save_path output/manual_run.json
```

Evaluate the baseline program:

```bash
python evaluate.py --program_path initial.py --results_dir results/manual_eval
```

Run the smoke evolution profile:

```bash
python run_evo.py --config_path shinka_smoke.yaml
```

Run the fuller development profile:

```bash
python run_evo.py --config_path shinka_small.yaml
```

## How The Pieces Fit Together Conceptually

If you only want the mental model, it is this:

- `initial.py` defines the search space in executable form.
- `evaluate.py` defines what "good" means.
- `run_evo.py` connects ShinkaEvolve to those two pieces.
- the YAML files define how aggressively to search.

In other words, this directory is a complete optimization task:

- a baseline program,
- an objective function,
- and a search configuration.

That is why the code is split this way.
