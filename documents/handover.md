# Agent Handover

Last updated: 2026-10-07. Read this first, then `documents/ai_setup.md` section 3 (repo rules).

## Project

Internal training material for ML engineers at Xomnia: Jupyter notebooks that teach LLM training mechanics at a low level (PyTorch, PEFT). The owner is Margot Razumeyeva.

The series is being reframed (see `documents/new_sessions_plan.md`, rough draft) from the old numbered track (Sessions 0-6: inference, quantization, LoRA, HPO) to "how would our company train a model itself": naive training, then hardware and dtypes, then efficient training, then business costs. The notebooks still reference the old session numbers.

## Working with the user

- Do not run console commands unless asked; suggest them instead. The user rejected shell calls used only for reading files. Use the Read tool.
- The user sets up environments themselves. Provide config files and commands, not install scripts.
- Do not create files or features that were not requested. Ask before editing docs that were not part of the request.
- Style: short, factual, no emojis or decorative symbols. Global rules are in `~/.claude/CLAUDE.md` (mirrored in `.cursor/rules/data-science.mdc`).
- The user writes quickly with typos; read for intent.

## Files

| Path | Notes |
|---|---|
| `notebooks/torch_llm.ipynb` | Session 0. IMDb, TF-IDF + MLP, training loop basics. Fits in one Read call. |
| `notebooks/lora.ipynb` | Session 5. DistilBERT + LoRA on IMDb. Over 25k tokens: Read with offset/limit, or ask the user before extracting cell sources with Python (`jq` is not installed). |
| `documents/notes.md` | Background notes: IEEE 754 floats, TF-IDF, LoRA parameter count (887,042 trainable), training cost ladder. |
| `documents/ai_setup.md` | Rules for AI agents, private-file handling, new-project checklist. |
| `pyproject.toml`, `poetry.toml` | Source of truth for dependencies. Poetry 2.x, `package-mode = false`, `requires-python >=3.12,<3.15`, venv in `.venv/`. Direct imports only, lower bound = latest stable as of 2026-10-07 (torch 2.14, transformers 5.19, datasets 5.1, peft 0.21, numpy 2.5, pandas 3.0). Do not change versions without asking. |
| `poetry.lock` | Generated 2026-10-07, matches the lower bounds in `pyproject.toml`. Commit it; never delete it to rebuild. |
| `requirements.txt` | Direct deps pinned (`==`) to `poetry.lock` versions. Regenerate when the lock changes. |
| `README.md` | Developer quick start (Poetry + kernel `llm-sessions`). |
| `set_up.md` | Detailed setup (Poetry and pip), CUDA notes. |

## Environment

- Manjaro Linux, zsh. Poetry and pip were not installed at first; install via `pacman` (PEP 668 blocks system pip).
- Python 3.12-3.14 required (numpy 2.5 lower bound, torch 2.14 upper bound).
- No GPU in recorded notebook outputs (`device: cpu`). Mixed precision and GPU memory sections are skipped on CPU.
- Branch: `1-lora-personalization`. Commit only when asked.

## Open items

1. Dependencies were upgraded to the latest major versions on 2026-10-07. Resolution succeeded (`poetry.lock` exists, Python 3.14); the first `poetry install` hit a download timeout on the `nvidia-*` CUDA wheels (fix documented in `set_up.md`). Neither notebook has been run on the new stack.
2. `torch_llm.ipynb` already adapted: removed `trust_remote_code=True`, wrapped `datasets` columns in `list()` (datasets >= 4 returns lazy Column objects), `torch.amp.GradScaler("cuda")`. Stored outputs are from the old stack.
3. `lora.ipynb` reviewed (not run) against the new stack: no API breakage found; dataset id changed to `stanfordnlp/imdb`. Unfixed bugs found in review: `OneCycleLR` stepped once per epoch instead of per optimizer step (LR stays near 8e-6), no scheduler in Part 11 ablation, no global seed, `warnings.filterwarnings("ignore")` hides deprecations, LLaMA comment says `v_lin` instead of `v_proj`, checkmark symbols in output, 2024 cost figures.
4. `torch_llm.ipynb` still open: no random seeds; recorded kernel crash (likely the dense TF-IDF `.toarray()`, about 1 GB per split); overfits from epoch 2 (final val acc 85.6%, comment claims 88-90%).
5. Untracked: `.gitignore`, `.cursorignore`, `.cursor/`, `pyproject.toml`, `poetry.toml`, `documents/handover.md`.
6. Xomnia wiki and Trello MCP connectors were not authorized in the first session; related plans there were not checked.
