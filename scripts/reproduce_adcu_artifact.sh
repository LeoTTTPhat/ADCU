#!/usr/bin/env bash
set -euo pipefail

export ADCU_ENABLE_HF_PEFT="${ADCU_ENABLE_HF_PEFT:-1}"
export ADCU_ENABLE_NEURAL_RETRIEVAL="${ADCU_ENABLE_NEURAL_RETRIEVAL:-1}"
export ADCU_ENABLE_PRETRAINED_LORA="${ADCU_ENABLE_PRETRAINED_LORA:-1}"
export ADCU_ENABLE_QWEN_TOFU="${ADCU_ENABLE_QWEN_TOFU:-1}"

python3.11 scripts/run_adcu_submission_experiments.py
python3.11 scripts/run_adcu_attack_suite.py
python3.11 scripts/run_adcu_advanced_experiments.py
python3.11 scripts/make_adcu_adjudication_draft.py
python3.11 scripts/make_adcu_submission_tables.py
python3.11 scripts/make_adcu_advanced_tables.py
python3.11 scripts/make_adcu_submission_figures.py
python3.11 scripts/make_adcu_pdf_figures.py
python3.11 -m pytest -q
(cd manuscript_adcu && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
