# How strict is the judge? Measuring the rigor of LLM hallucination judges

Code, raw model outputs and analyses for measuring the **rigor** (severity) of LLM judges
that detect hallucinations: how readily a judge flags an opinion as a hallucination,
separately from how well it discriminates. Four open judges (Qwen3.5-2B/4B,
Gemma-4-E2B/E4B) are run with nine prompts on the 4,238 human-labelled opinions of
[PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR) (504
hallucinations, 206 public hearings). Rigor is measured three ways:

* the raw flag rate `R_raw`;
* the signal-detection criterion `R_c = -c`;
* judge severity `R_rasch` in a many-facet Rasch model fitted by joint maximum likelihood.

Uncertainty comes from a bootstrap over hearings. In every measure, **larger = stricter**.

| Hypothesis | Result |
|---|---|
| H1 lenient < neutral < strict prompt (one-sentence intervention) | supported (24/24 contrasts) |
| H2 dataset prompts P1 < P2 < P3 | not supported (fails for Qwen3.5-2B only) |
| H3 the three measures converge (ρ ≥ 0.7) | supported (ρ ≥ 0.995) |
| H4 `R_c` less related to ability than `R_raw` | inconclusive |
| H5 neutral paraphrases move rigor less than the intervention | not supported (fails for Qwen3.5-2B only) |

## Repository layout

```
config.yaml        seed, dataset revision, models (pinned revisions), prompts, generation, analysis settings
prompts/           the nine system prompts, the user-message template, JSON schemas for guided decoding
data/              pilot_items.csv (pilot sample), paper_confusion_tables.csv (Tables 5-8 of the dataset paper)
src/
  data.py          dataset download and flattening, data checks, pilot sample, raw-output I/O
  generation.py    vLLM generation: prompts, guided JSON decoding, retry protocol, raw records
  measures.py      raw rate, criterion R_c and d' (signal detection)
  rasch.py         many-facet Rasch model (joint maximum likelihood), infit/outfit, separation
  bootstrap.py     cluster bootstrap over hearings
  analysis.py      hypotheses, variance decomposition, reliability, exploratory analyses
scripts/
  prepare_data.py        download + checks + pilot sample
  generate.py            run one judge on all prompts (GPU)
  pilot_inclusion.py     pilot inclusion criterion and diagnostics
  analyze.py             confirmatory analyses and sensitivity to missing verdicts
  analyze_exploratory.py reproducibility, recovered verdicts, log-probabilities, size ladder
  make_figures.py        figures and LaTeX tables of the paper
outputs/           raw judge outputs (gzipped JSONL): pilot/ main/ replication/ recovery/ size_ladder/
results/           analysis results (CSV/JSON), figures/ and tables/
```

## Setup

Python 3.12 on Linux. Generation needs an NVIDIA GPU (24 GB is enough for the four main
judges); analyses run on CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # direct dependencies, exact versions
# or, for the full pinned environment:  pip install -r requirements-lock.txt
```

Model weights are downloaded from the Hugging Face Hub at the revisions pinned in
`config.yaml`. Set `HF_HOME` to choose the cache location.

## Reproducing the results

**From the released outputs (no GPU, ~15 min):**

```bash
make reproduce     # = make data analysis figures
```

`make data` downloads `PublicHearingBR_NLI.jsonl` at a pinned revision, flattens it to
`data/items.parquet`, checks it (4,238 opinions, 504 hallucinations, all 48 cells of the
dataset paper's confusion matrices) and draws the pilot sample. The dataset is not
redistributed here. `datasets.load_dataset` is not used, because the dataset repository
holds two files with different schemas.

**From scratch (GPU):**

| Target | What it runs | Time on one RTX 4090 |
|---|---|---|
| `make quickstart` | 300-item pilot with Qwen3.5-2B, then the inclusion check | ~5 min |
| `make pilot` | pilot with the four judges; a judge is included if the 95% CI of d′ under prompt N excludes 0 | ~20 min |
| `make main` | four judges × nine prompts × 4,238 items | ~3 h |
| `make exploratory` | second run with verdict log-probabilities, regeneration of missing verdicts, size ladder (Qwen3.5-0.8B/9B, Gemma-4-12B) | ~10 h* |
| `make analysis figures` | all analyses, figures and tables | ~12 min (CPU) |

\* Qwen3.5-9B and Gemma-4-12B do not fit on one 24 GB GPU in bf16; use `make exploratory TP=2`
with two GPUs, or one GPU with ≥ 48 GB.

Every model is loaded once, and all its (prompt, item) requests go to vLLM together. Decoding is
greedy with JSON-schema guided decoding, and the verdict is generated last. Outputs that hit
the token limit or do not parse are regenerated once with 2,048 tokens and are otherwise
recorded as missing. To run models in parallel on several GPUs, call `scripts/generate.py`
once per model with different `CUDA_VISIBLE_DEVICES`.

Greedy decoding in vLLM is not bit-for-bit deterministic under dynamic batching. In a
complete second run, verdicts agreed on 92–99.9% of items per configuration, and every
measure moved by at most 0.006. The released outputs are therefore the reference for all
reported numbers.

## Raw outputs

`outputs/<run>/<model>__<prompt>.jsonl.gz` has one record per item with these fields:

* `item_uid`, `hearing_id`, `model_id`, `model_revision`, `prompt_id`, `prompt_sha256`;
* `raw_text` and `parsed` (the JSON output);
* `opiniao_inferida` (the verdict) and `flag` (1 = hallucination, null = missing);
* `finish_reason`, `n_output_tokens`, `attempt`, `timestamp`;
* `verdict_logprobs`, only in runs with `--logprobs`.

Each run directory has `*.attempt1_failures.jsonl.gz` with first attempts that were regenerated,
and `run_<model>.json` with diagnostics and timing. New runs write plain `.jsonl`; all readers
accept both formats.

## Paper figures and tables

`make figures` writes `results/figures` and `results/tables`; `make paper` writes the same files
into `paper/`.

| Paper | File | Input |
|---|---|---|
| Fig. 1 rigor by configuration | `fig_rigor_by_config.pdf` | `results/confirmatory/measures_by_config.csv`, `paper_judges.csv` |
| Fig. 2 convergence and ability (H3, H4) | `fig_convergence_ability.pdf` | `convergence.csv`, `ability.csv` |
| Fig. 3 variance decomposition | `fig_variance.pdf` | `variance_decomposition.csv` |
| Fig. 4 size ladder | `fig_size_ladder.pdf` | `results/exploratory/size_ladder_measures.csv` |
| Table 1 hypotheses | — | `results/confirmatory/summary.json` |
| Table 2 contrasts | `tables/contrasts.tex` | `results/confirmatory/contrasts.csv` |
| Appendix | `fig_triad.pdf`, `fig_stability.pdf`, `tables/{measures,paper_judges,size_ladder,prompts}.tex` | as above |

The remaining numbers in the text come from these sources:

* data checks: `results/data_checks/`;
* pilot inclusion: `results/pilot/inclusion.csv`;
* sensitivity to missing verdicts: `results/confirmatory/sensitivity_missing_as_{0,1}/`;
* Rasch fit: `results/confirmatory/rasch_config_fit.csv`;
* exploratory results: `results/exploratory/summary.json`.

## Method notes

* **Measures** (`src/measures.py`). H and F use the log-linear correction
  `(count + 0.5)/(n + 1)` in every configuration. `R_c = (z(H) + z(F))/2` and `d' = z(H) − z(F)`.
* **Rasch model** (`src/rasch.py`). `logit P(flag) = θ_item + α_model + γ_prompt + (αγ)`, with
  sum-to-zero constraints. It is fitted by unpenalized joint maximum likelihood using
  block-coordinate Newton steps, and gives the same estimates as an unpenalized logistic
  regression with item and configuration dummies. Items flagged by no judge or by every judge
  are not identifiable and are excluded.
* **Bootstrap** (`src/bootstrap.py`). One matrix of hearing-resampling counts (B = 1,000) is
  shared by all measures, so contrasts and correlations are bootstrapped jointly. The Rasch
  model is refitted in every replicate.
* **Decision rules** for supported / not supported / inconclusive are in the docstring of
  `src/analysis.py`.
* Three opinions have no usable context (empty or whitespace-only chunks). They are kept and
  flagged `no_context`.

## Troubleshooting

* *FlashInfer JIT compilation fails* (system `nvcc` older than the CUDA version of torch): set
  `VLLM_USE_FLASHINFER_SAMPLER=0`. Greedy decoding does not use that sampler.
* *Not enough KV-cache memory*: lower `max_num_seqs` for the model in `config.yaml`, or use
  `--tensor-parallel`.

## License and citation

Code: MIT (`LICENSE`). The dataset and model weights belong to their owners and are
downloaded, not redistributed. Please also cite the dataset:
Fernandes, Dobins, Lotufo and Pereira (2024), *PublicHearingBR: A Brazilian Portuguese Dataset
of Public Hearing Transcripts for Summarization of Long Documents*, arXiv:2410.07495.
