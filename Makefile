# Reproduction targets. Activate the Python environment first (see README).
PY      ?= python
MODELS  := qwen35-2b qwen35-4b gemma4-e2b gemma4-e4b
LADDER  := qwen35-0.8b qwen35-9b gemma4-12b
TP      ?= 1   # GPUs per model for qwen35-9b and gemma4-12b (2 on 24 GB GPUs)

.PHONY: help data quickstart pilot main analysis figures paper exploratory reproduce

help:        ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

data:        ## download the dataset, run data checks, draw the pilot sample (CPU)
	$(PY) scripts/prepare_data.py

quickstart:  ## pilot (300 items) with one judge, then the inclusion check (1 GPU)
	$(PY) scripts/generate.py --model qwen35-2b --split pilot
	$(PY) scripts/pilot_inclusion.py

pilot:       ## pilot with the four judges and inclusion check (1 GPU)
	for m in $(MODELS); do $(PY) scripts/generate.py --model $$m --split pilot || exit 1; done
	$(PY) scripts/pilot_inclusion.py

main:        ## main experiment: four judges x nine prompts x 4,238 items (1 GPU)
	for m in $(MODELS); do $(PY) scripts/generate.py --model $$m || exit 1; done

exploratory: ## re-run with log-probabilities, recovery of missing verdicts, size ladder (GPU)
	for m in $(MODELS); do $(PY) scripts/generate.py --model $$m --out outputs/replication --logprobs || exit 1; done
	for m in $(MODELS); do $(PY) scripts/generate.py --model $$m --out outputs/recovery --only-missing outputs/main --no-whitespace --logprobs || exit 1; done
	$(PY) scripts/generate.py --model qwen35-0.8b --out outputs/size_ladder --logprobs
	for m in qwen35-9b gemma4-12b; do $(PY) scripts/generate.py --model $$m --out outputs/size_ladder --logprobs --tensor-parallel $(TP) || exit 1; done
	for m in $(LADDER); do $(PY) scripts/generate.py --model $$m --out outputs/recovery --only-missing outputs/size_ladder --no-whitespace --logprobs || exit 1; done

analysis:    ## confirmatory and exploratory analyses from outputs/ (CPU)
	$(PY) scripts/analyze.py
	$(PY) scripts/analyze_exploratory.py

figures:     ## figures and LaTeX tables into results/ (CPU)
	$(PY) scripts/make_figures.py

paper:       ## figures and LaTeX tables into paper/ (CPU)
	$(PY) scripts/make_figures.py --out paper

reproduce: data analysis figures ## every result from the released outputs, no GPU
