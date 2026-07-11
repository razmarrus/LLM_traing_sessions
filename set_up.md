# Requirements and Environment Setup


## Setup commands

### If you are on Linux / Mac

```bash
# Create venv
python3 -m venv lora-session

# Activate
source lora-session/bin/activate

# Upgrade pip first (old pip sometimes fails on torch)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Register as Jupyter kernel
python -m ipykernel install --user --name=lora-session --display-name "LoRA Session"

# Launch Jupyter
jupyter notebook
```

### If you are on Windows

```bash
# Create venv
python -m venv lora-session

# Activate
lora-session\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Register as Jupyter kernel
python -m ipykernel install --user --name=lora-session --display-name "LoRA Session"

# Launch Jupyter
jupyter notebook
```

---

## After launching Jupyter

```
Kernel → Change kernel → LoRA Session
```

---

## Verify the install

Run this in the first notebook cell before anything else:

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

---

## Notes

**Python version:** use 3.10 or 3.11. PyTorch 2.3 does not support 3.12 fully yet.

```bash
# Check your Python version before creating the venv
python3 --version
```

**CUDA version:** the torch version above pulls CUDA 12.1 by default. If your driver is older:

```bash
# Check your CUDA version
nvidia-smi

# Install torch for CUDA 11.8 instead
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

**No GPU:** the notebook runs on CPU. Expect roughly 10x slower training. All concepts are identical.

**Remove kernel when done:**

```bash
jupyter kernelspec remove lora-session
```

# Requirements and Environment Setup

## requirements.txt

```txt
# Core ML
torch==2.3.1
torchvision==0.18.1

# HuggingFace ecosystem
transformers==4.44.2
datasets==2.21.0
peft==0.12.0
accelerate==0.34.2
tokenizers==0.19.1
huggingface-hub==0.24.6

# Scientific Python
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.2

# Notebook
ipykernel==6.29.5
jupyter==1.0.0
```

---

## Setup commands

### If you are on Linux / Mac

```bash
# Create venv
python3 -m venv lora-session

# Activate
source lora-session/bin/activate

# Upgrade pip first (old pip sometimes fails on torch)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Register as Jupyter kernel
python -m ipykernel install --user --name=lora-session --display-name "LoRA Session"

# Launch Jupyter
jupyter notebook
```

### If you are on Windows

```bash
# Create venv
python -m venv lora-session

# Activate
lora-session\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Register as Jupyter kernel
python -m ipykernel install --user --name=lora-session --display-name "LoRA Session"

# Launch Jupyter
jupyter notebook
```

---

## After launching Jupyter

```
Kernel → Change kernel → LoRA Session
```

---

## Verify the install

Run this in the first notebook cell before anything else:

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

---

## Notes

**Python version:** use 3.10 or 3.11. PyTorch 2.3 does not support 3.12 fully yet.

```bash
# Check your Python version before creating the venv
python3 --version
```

**CUDA version:** the torch version above pulls CUDA 12.1 by default. If your driver is older:

```bash
# Check your CUDA version
nvidia-smi

# Install torch for CUDA 11.8 instead
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

**No GPU:** the notebook runs on CPU. Expect roughly 10x slower training. All concepts are identical.

**Remove kernel when done:**

```bash
jupyter kernelspec remove lora-session
```