# LLM Training Sessions Material

## Motivation

Low-level experimentation material for ML engineers. The notebooks implement core LLM training concepts directly in PyTorch and PEFT (training loop, gradients, optimizer memory, mixed precision, gradient accumulation, LoRA) so each mechanism can be inspected and measured, not only used through high-level APIs.

## Quick start

Requirements: Python within `requires-python` in `pyproject.toml`, Poetry >= 2.0.

```bash
# Create .venv/ in the project root and install dependencies
poetry env use python3
poetry install

# Register the environment as a Jupyter kernel
poetry run python -m ipykernel install --user --name=llm-sessions --display-name "LLM Sessions"
```

Select the kernel:
- VS Code: open a notebook, Select Kernel > Jupyter Kernel > LLM Sessions
- Browser: `poetry run jupyter notebook` (pip setup: `jupyter notebook` in the activated venv), then Kernel > Change kernel > LLM Sessions

Without Poetry, GPU/CUDA wheel selection, download timeouts: see `set_up.md`.

No GPU: all notebooks run on CPU, roughly 10x slower. Mixed precision and GPU memory measurement sections are skipped.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/torch_llm.ipynb` | Session 0: PyTorch training loop on IMDb sentiment (TF-IDF + MLP). Gradients, Adam memory, learning rate, mixed precision, gradient accumulation, memory accounting, inference, exercises. |
| `notebooks/lora.ipynb` | Session 5: LoRA fine-tuning of DistilBERT on IMDb sentiment. LoRA from scratch, PEFT, rank ablation, adapter save/merge, cost analysis, exercises. |
| `documents/` | Session plans, background notes (floating point formats, TF-IDF, LoRA parameter counts, training cost estimates), AI tooling setup. |
| `pyproject.toml`, `poetry.toml` | Poetry project definition (source of truth for dependencies and Python version); venv created in `.venv/`. |
| `poetry.lock` | Exact tested versions. Commit it; do not delete it to rebuild. |
| `requirements.txt` | Direct dependencies pinned to `poetry.lock` versions, for a pip-only setup. |
| `set_up.md` | Detailed setup (Poetry and pip), CUDA notes, verification cell. |
| `.cursor/rules/`, `.cursorignore` | AI coding rules and ignore list for Cursor. |
