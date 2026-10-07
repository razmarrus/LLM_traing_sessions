# AI Tooling and Private Files Setup

How AI coding rules and ignore files are set up, globally and per project.
Last updated: 2026-10-07 (switched to Poetry + `.venv/`, Python 3.12-3.14).

---

## 1. One-time global setup (per machine)

### Git
- [ ] Create the global ignore folder: `mkdir -p ~/.config/git`
- [ ] Copy patterns: `cp .gitignore ~/.config/git/ignore`
- [ ] Register it: `git config --global core.excludesFile ~/.config/git/ignore`
- [ ] Verify: `git check-ignore -v .env` prints the global file path
- [ ] Keep a project `.gitignore` in shared repos anyway: the global file only applies on your machine

### Cursor
- [ ] Cursor Settings > Rules > User Rules: paste the rules from `~/.claude/CLAUDE.md`
- [ ] Cursor Settings > General > Global Cursor Ignore List, add:
  `.env`, `.env.*`, `*.pem`, `*.key`, `credentials*.json`, `service-account*.json`, `secrets.*`, `.netrc`, `.venv/`, `venv/`
- [ ] Optional, large artifacts:
  `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.gguf`, `checkpoints/`, `outputs/`, `wandb/`, `mlruns/`

### Claude Code
- [ ] Global rules live in `~/.claude/CLAUDE.md` (already created)
- [ ] Add deny rules to `~/.claude/settings.json`:
  ```json
  {
    "permissions": {
      "deny": [
        "Read(**/.env)",
        "Read(**/.env.*)",
        "Read(**/*.pem)",
        "Read(**/*.key)",
        "Read(**/credentials*.json)",
        "Read(**/.venv/**)"
      ]
    }
  }
  ```
- [ ] Start a new session so the rules load
- [ ] Test: ask Claude to read a `.env` file, the read must be denied

---

## 2. New project checklist

### Repo basics
- [ ] `git init` or clone
- [ ] Copy a project `.gitignore` (start from this repo's)
- [ ] Add `.env.example` with variable names only, real values go in `.env`
- [ ] Add `README.md`: setup, usage, assumptions and limitations

### Environment
- [ ] Define dependencies in `pyproject.toml`; add `poetry.toml` with `virtualenvs.in-project = true` so the venv lands in `.venv/`
- [ ] `poetry install`, commit `poetry.lock`
- [ ] If the venv has a custom name, add it to `.gitignore` and `.cursorignore`
- [ ] Load config from `.env` via `python-dotenv` or `pydantic-settings`, never hardcode

### AI rules
- [ ] Claude Code: nothing to do. Add a project `CLAUDE.md` only for project specifics (stack, data paths, how to run)
- [ ] Cursor: nothing to do if User Rules are set. Add `.cursor/rules/data-science.mdc` only if teammates use Cursor
- [ ] `.cursorignore`: add only project-specific large folders (datasets, model outputs)

### Before the first commit
- [ ] `git status`: `.env`, venv, data and weights are not listed
- [ ] `git check-ignore -v .env` matches a rule

---

## 3. Instructions for AI models working in this repo

Read this section before making changes.

### Rules
- Coding rules are in `~/.claude/CLAUDE.md` (Claude Code) and `.cursor/rules/data-science.mdc` (Cursor). Both contain the same text. If you change one, change the other.
- Follow them: PEP8, type hints, one-line docstrings, `logging` over `print`, no emojis or decorative symbols, concise answers, code first.
- Exception: notebooks in `notebooks/` are teaching material and use `print` for cell output. Keep that style there; use `logging` in any `.py` modules.
- Do not run console commands unless the user explicitly asks. Suggest them instead.
- Do not create files or features the user did not request.

### Private files
- Never read, print, or commit `.env`, `.env.*`, keys, tokens, or credential files.
- The venv for this repo is `.venv/` (see `set_up.md`). Do not read or index it.
- When adding a new secret, data, or artifact location, add it to both `.gitignore` and `.cursorignore`.

### Repo layout
- `notebooks/`: `torch_llm.ipynb` (PyTorch basics), `lora.ipynb` (LoRA fine-tuning)
- `documents/`: session plans, notes, and this file
- `pyproject.toml`, `poetry.toml`: dependencies (source of truth), Python 3.12-3.14; do not change versions without asking
- `requirements.txt`: mirror of `pyproject.toml` for pip-only setups; keep in sync
- `set_up.md`: environment setup (Poetry and pip), Jupyter kernel `llm-sessions`

### Working style
- Match the existing notebook style: naming, cell structure, comment density.
- Prefer library implementations (`torch`, `transformers`, `peft`) over hand-written versions.
- Set random seeds in anything that trains or samples.
- Commit only when asked. Branch per topic (current: `1-lora-personalization`).

### Keeping this file current
- Update the date at the top and the relevant section when you change setup, rules, or ignore files.
