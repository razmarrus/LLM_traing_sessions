# Environment Setup

Dependencies are defined in `pyproject.toml` (source of truth). `requirements.txt` mirrors it for pip-only setups.

## Requirements

- Python version within `requires-python` in `pyproject.toml`
- Poetry >= 2.0

Install Poetry on Manjaro/Arch (system-wide `pip install` is blocked by PEP 668):

```bash
sudo pacman -S python-poetry
# or
sudo pacman -S python-pipx && pipx ensurepath && pipx install poetry
```

## Setup with Poetry (Linux / Mac)

```bash
# From the project root. poetry.toml places the venv in .venv/
python3 --version              # must satisfy requires-python in pyproject.toml
poetry env use python3         # or an explicit interpreter, e.g. python3.14
poetry install                 # creates .venv/, installs versions from poetry.lock

# Register as Jupyter kernel
poetry run python -m ipykernel install --user --name=llm-sessions --display-name "LLM Sessions"
```

## Setup with pip

### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=llm-sessions --display-name "LLM Sessions"
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=llm-sessions --display-name "LLM Sessions"
```

## Select the kernel

- VS Code: open a notebook, Select Kernel > Jupyter Kernel > LLM Sessions
- Browser (Poetry): `poetry run jupyter notebook`, then Kernel > Change kernel > LLM Sessions
- Browser (pip): with `.venv` activated, `jupyter notebook`, then Kernel > Change kernel > LLM Sessions

## Verify the install

Run in the first notebook cell:

```python
import torch
import transformers
import peft
import datasets

print(f"torch:          {torch.__version__}")
print(f"transformers:   {transformers.__version__}")
print(f"peft:           {peft.__version__}")
print(f"datasets:       {datasets.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Notes

**CUDA:** the default Linux torch wheel from PyPI bundles a recent CUDA runtime (about 3 GB of `nvidia-*` packages). If your driver is older (`nvidia-smi` shows the max supported CUDA version) or you have no GPU, use a different wheel index from https://pytorch.org/get-started/locally/ (e.g. `cu126`, `cpu`).

- Poetry: `poetry install` replaces a manually installed torch with the locked one, so the index must be declared in `pyproject.toml`, then re-lock:

  ```toml
  [[tool.poetry.source]]
  name = "pytorch"
  url = "https://download.pytorch.org/whl/cpu"   # or .../whl/cu126
  priority = "explicit"

  [tool.poetry.dependencies]
  torch = { source = "pytorch" }
  ```

  ```bash
  poetry lock && poetry install
  ```

  This changes torch for everyone using the repo; keep it local unless the team agrees.

- pip: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` (or `cu126`) before `pip install -r requirements.txt`.

**Download timeouts:** the CUDA packages are large. If `poetry install` fails with `Read timed out`, rerun with `POETRY_REQUESTS_TIMEOUT=600 POETRY_INSTALLER_MAX_WORKERS=2 poetry install`. Finished downloads are cached.

**No GPU:** the notebooks run on CPU, roughly 10x slower. Mixed precision and GPU memory measurement sections are skipped.

**Rebuild the environment:**

```bash
rm -rf .venv
poetry install
```

Do not delete `poetry.lock`: it pins the tested versions. Upgrade deliberately with `poetry update` or `poetry lock`, then re-run both notebooks and regenerate `requirements.txt`.

**Remove the kernel:**

```bash
jupyter kernelspec remove llm-sessions
```
