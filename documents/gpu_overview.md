# GPUs for LLM work: overview

Status: autumn 2026. Prices are examples and change monthly. Always check live prices before renting.

## 1. What decides which GPU you need

| Spec | Software analogy | What it decides |
|---|---|---|
| VRAM (GPU memory) | Heap limit without swap; exceed it and you get `OutOfMemoryError` | Which model and batch size fit |
| Memory bandwidth | Read throughput | Generation speed; each generated token reads all weights once |
| Compute and number formats (bf16, fp8, fp4) | CPU speed and instruction-set extensions | Training speed; code only runs fast on formats the hardware supports |
| Interconnect (NVLink vs PCIe) | Network between nodes | Only matters when one model is split across several GPUs |
| Host CPU and RAM | The server around the GPU | Data loading and tokenization; rarely the bottleneck |

Useful formulas:

- Weights memory = parameters x bytes per parameter (bf16 = 2, 8-bit = 1, 4-bit = 0.5). An 8B model in bf16 is about 16 GB.
- Upper bound on tokens/s for one request = bandwidth / model size. 16 GB model: about 54 tok/s at 0.86 TB/s, about 210 tok/s at 3.35 TB/s.
- Full fine-tuning with Adam in mixed precision: about 16 bytes per parameter plus activations. LoRA avoids most of this because only a small fraction of parameters is trained.

## 2. Generations (architectures)

NVIDIA names each chip design after a scientist. Every GPU model belongs to one generation. A generation is like a CPU instruction-set version: it decides which features and number formats exist.

| Generation | Year | Example GPUs | Added for LLMs |
|---|---|---|---|
| Volta | 2017 | V100 | First tensor cores (matrix-multiply units) |
| Turing | 2018 | T4, RTX 20xx | int8 / int4 |
| Ampere | 2020 | A100, A10G, RTX 30xx | bf16 |
| Ada Lovelace | 2022 | L4, L40S, RTX 40xx | fp8 |
| Hopper | 2022 | H100, H200 | fp8 tuned for transformers, faster NVLink |
| Blackwell | 2024-25 | B200, B300, GB200/GB300, RTX 50xx, RTX PRO 6000 | fp4, larger memory |

The first letter of a datacenter GPU name usually gives the generation: V100 = Volta, A100 = Ampere, L40S = Lovelace, H100 = Hopper, B200 = Blackwell.

Why the generation matters:

1. Features: bf16 needs Ampere or newer (T4, V100 and free Colab GPUs cannot train in bf16). fp8 needs Ada or Hopper, fp4 needs Blackwell.
2. Software support: PyTorch is compiled for a list of generations. Very new GPUs need recent PyTorch and CUDA versions (Blackwell: CUDA 12.8 or later). Some optimized libraries support only recent generations.
3. Speed and efficiency: each generation does more work per second and per watt.

The generation does not tell you everything: memory size and type vary within one generation (L4 and H100 differ far more in memory than in generation).

Rule of thumb for LLM work: Ampere or newer, then choose by memory size and price.

## 3. GDDR vs HBM

Both are types of VRAM.

| | GDDR | HBM |
|---|---|---|
| Stands for | Graphics Double Data Rate | High Bandwidth Memory |
| Built as | Separate chips on the board around the GPU | Stacked chips next to the GPU in the same package |
| Bus width | Narrow (about 384 bits) | Very wide (thousands of bits) |
| Bandwidth | About 0.3-1.8 TB/s | About 2-8 TB/s |
| Cost | Cheaper | Much more expensive, limited supply |
| Found in | Gaming cards, L4, L40S, RTX PRO 6000 | A100, H100, H200, B200, AMD MI300X |
| Versions | GDDR6, GDDR6X, GDDR7 | HBM2e, HBM3, HBM3e |

HBM is the main reason datacenter training GPUs cost several times more per hour. If a model fits in a GDDR card and top speed is not needed, the GDDR card is usually the better buy.

## 4. Product tiers

Same generation, different products, like laptop vs server CPUs.

| Tier | Examples | VRAM | Memory | Typical use |
|---|---|---|---|---|
| Consumer | RTX 3090, 4090, 5090 | 24-32 GB | GDDR | Experiments, small fine-tunes |
| Datacenter inference | T4, A10G, L4, L40S | 16-48 GB | GDDR | Serving small and medium models |
| Pro / server | RTX 6000 Ada, RTX PRO 6000 | 48-96 GB | GDDR | Serving, mid-size fine-tuning |
| Datacenter training | A100, H100, H200, B200, B300 | 40-288 GB | HBM | Training, high-volume serving |
| Rack-scale | GB200 / GB300 NVL72 | 72 GPUs acting as one machine | HBM | Frontier-model training, contracts only |

Traps:

- NVIDIA's GeForce license forbids datacenter use, so big clouds do not offer consumer cards (RTX 4090). Marketplaces do.
- Same name, different hardware: H100 PCIe (2.0 TB/s, no NVLink) is slower than H100 SXM (3.35 TB/s, NVLink). "RTX 6000" names four different cards across four generations. Compare exact models.
- "L40S with Intel CPU" vs "with AMD CPU": same GPU, different host server.

## 5. Specs of common GPUs

| GPU | Generation | VRAM | Bandwidth | bf16 |
|---|---|---|---|---|
| V100 | Volta | 16/32 GB HBM2 | 0.9 TB/s | No |
| T4 | Turing | 16 GB GDDR6 | 0.32 TB/s | No |
| RTX 3090 | Ampere | 24 GB GDDR6X | 0.94 TB/s | Yes |
| A10G | Ampere | 24 GB GDDR6 | 0.6 TB/s | Yes |
| A100 | Ampere | 40/80 GB HBM2e | 1.6-2.0 TB/s | Yes |
| L4 | Ada | 24 GB GDDR6 | 0.3 TB/s | Yes |
| RTX 4090 | Ada | 24 GB GDDR6X | 1.0 TB/s | Yes |
| L40S | Ada | 48 GB GDDR6 | 0.86 TB/s | Yes |
| RTX PRO 6000 | Blackwell | 96 GB GDDR7 | ~1.6 TB/s | Yes |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | Yes |
| H200 | Hopper | 141 GB HBM3e | 4.8 TB/s | Yes |
| B200 | Blackwell | 180 GB HBM3e | ~8 TB/s | Yes |
| B300 | Blackwell Ultra | 288 GB HBM3e | ~8 TB/s | Yes |

## 6. Alternatives to NVIDIA

CUDA is the de facto standard: every ML library supports NVIDIA first. Other chips are like porting an app to a less common OS: it mostly works, with edge cases.

| Chip | Where to rent | Catch |
|---|---|---|
| AMD Instinct MI300X / MI325X / MI355X (192-288 GB) | Azure, Oracle, some GPU clouds | ROCm instead of CUDA; PyTorch works, some libraries lag |
| Google TPU (v5e, v6e Trillium, Ironwood) | Google Cloud only | Best with JAX; PyTorch via XLA |
| AWS Trainium / Inferentia | AWS only | Neuron SDK; code must be ported |

## 7. Where to rent

| Provider type | Examples | Good for | Watch out |
|---|---|---|---|
| Big clouds | AWS, Google Cloud, Azure, Oracle | Company accounts, compliance, other services nearby | Highest list prices, GPU quota requests, top GPUs often only in 8-GPU machines |
| GPU-specialist clouds | Nebius, CoreWeave, Lambda, Crusoe | Cheaper H100/H200/B200, single-GPU machines | Fewer managed services |
| European providers | Nebius (HQ Amsterdam), Scaleway, OVHcloud, Hetzner | EU data residency | Smaller GPU choice |
| Marketplaces | RunPod, Vast.ai | Cheapest, consumer GPUs, per-second billing | Community machines belong to third parties: no sensitive data; machines can disappear |
| Notebook platforms | Google Colab, Kaggle, Lightning AI | No setup, free tiers | Session limits; free GPUs (T4) have no bf16 |
| Serverless GPU | Modal, RunPod Serverless, Replicate, Baseten | Inference endpoints, batch jobs, scale to zero | Code packaged as function or container |
| No GPU | Managed fine-tuning and inference APIs (Together, Fireworks, Vertex AI, Bedrock, OpenAI) | No hardware to manage | Less control, only supported models |

## 8. Pricing models

- On-demand: per hour or second, stop any time.
- Spot / preemptible: often half price or less, can be reclaimed at any moment. Fine for jobs that checkpoint and resume.
- Reserved / committed: weeks to years, discounted. For steady use.
- Monthly dedicated servers (e.g. Hetzner): fixed price, always on.

Stopped VMs usually stop GPU billing, but disks keep billing. Forgetting to stop a VM is usually the largest cost.

## 9. Example prices (per GPU-hour, on-demand unless noted, autumn 2026)

| GPU | Price | Where |
|---|---|---|
| RTX 3090 | ~$0.20 | RunPod community |
| RTX 4090 | ~$0.34 community, ~$0.69 secure | RunPod |
| L4 | ~$0.71-0.74 (spot ~$0.37) | Google Cloud |
| L40S | $1.55 (preemptible $0.74 flat) | Nebius |
| RTX PRO 6000 | $1.80 | Nebius |
| H100 PCIe | ~$2.89 | RunPod |
| H100 SXM | $4.50 | Nebius |
| H200 | $5.40 | Nebius |
| B200 | $8.50 | Nebius |
| B300 | $9.50 | Nebius |
| GB200 / GB300 NVL72 | Contract only | Nebius |

RunPod and Google Cloud figures come from third-party trackers.

## 10. Sizing rules of thumb

| Workload | Memory needed | Typical rental |
|---|---|---|
| Inference, 7-8B, bf16 | ~16 GB + KV cache | One 24 GB GPU (L4, 4090, A10G) |
| Inference, 70B, fp8 | ~70 GB + KV cache | 1x H200 or 2x H100 |
| Inference, 70B, 4-bit | ~35-40 GB + KV cache | One 48 GB GPU (L40S), short contexts |
| Inference, 70B, bf16 | ~140 GB + KV cache | 2x H200 or 4x H100 |
| LoRA, 7-8B | ~16 GB + activations | 24-48 GB |
| QLoRA, 70B (4-bit base) | ~40-50 GB | One 48-80 GB GPU |
| Full fine-tune, 7-8B, Adam | ~130 GB + activations | 4-8x H100/H200 in one machine |
| Pretraining | Thousands of GPUs | Cluster contract |

## 11. Sources

- [Nebius prices](https://nebius.com/prices)
- [Nebius VM types](https://docs.nebius.com/compute/virtual-machines/types)
- [Google Cloud accelerator pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized?hl=en)
- [Devzero g2-standard-4](https://www.devzero.io/instances/gcp/g2-standard-4)
- [Spheron: RunPod vs Vast.ai 2026](https://www.spheron.network/blog/runpod-vs-vastai-2026/)
- [DIYAI: RunPod pricing 2026](https://diyai.io/ai-tools/hosting/runpod-pricing/)

## 12. What `notebooks/style_lora.ipynb` needs

The notebook fine-tunes `Qwen/Qwen3-8B` with LoRA on the last 8 decoder layers, in bf16, on one GPU.

### Hard requirements

| Requirement | Why | Consequence |
|---|---|---|
| NVIDIA GPU with CUDA | PART 0 asserts `torch.cuda.is_available()`; the model loads with `device_map={"": 0}` | No CPU run from PART 4 on |
| bf16 support (Ampere or newer) | PART 0 asserts `torch.cuda.is_bf16_supported()`; training uses `bf16=True` | T4, V100 and free Colab fail |
| At least 24 GB VRAM | Weights alone are about 16 GB (see below) | 16 GB cards do not fit Qwen3-8B |
| Messages file | JSONL, one `{"text": ...}` per line, at least ~500 rows recommended | `FileNotFoundError` without it |
| Internet access | Downloads the model from the Hugging Face Hub; no token needed for Qwen3 | |

Not required: a storage bucket (only printed and logged; PART 10 is manual), `HF_TOKEN`, the resume and re-train variables.

### GPU memory estimate

Estimates, not measurements. Check the "Peak GPU memory" line printed after training.

| Component | Size | Default settings (batch 8) | Reduced settings (batch 2, checkpointing) |
|---|---|---|---|
| Base weights, bf16 | 8.2B params x 2 bytes | ~16.4 GB | ~16.4 GB |
| LoRA weights, grads, Adam state | ~9.7M params x ~16 bytes | ~0.2 GB | ~0.2 GB |
| Logits and their gradients, fp32 | batch x tokens x 152k vocab x 4 bytes, about 2x | up to ~5 GB | up to ~1.3 GB |
| Activations, last 8 layers only | grows with batch x tokens | up to ~5 GB | under ~1 GB |
| CUDA context and allocator overhead | fixed | ~1-2 GB | ~1-2 GB |
| Total, worst case (512-token batches) | | ~25-30 GB | ~19-21 GB |

Batches are padded to the longest message in the batch, not to `max_length=512`, so short messages use far less than the worst case. Only the last 8 layers hold activations, because no layer before the first adapter needs gradients.

Effective batch stays 16 in both settings (8 x 2 and 2 x 8), so runs are comparable.

### GPU options

| GPU | VRAM | Settings | Fit |
|---|---|---|---|
| L4, RTX 3090/4090, A10G | 24 GB | Reduced, or switch to `Qwen/Qwen3-4B` | Tight for Qwen3-8B; unverified |
| L40S | 48 GB | Reduced as a safe start; defaults likely fit | Comfortable |
| RTX PRO 6000 | 96 GB | Defaults | Comfortable, notebook unchanged |
| H100 / A100 80 GB | 80 GB | Defaults (what the notebook was sized for) | Comfortable, more expensive |

Reduced settings, in PART 1:

```python
cfg = TrainConfig(
    per_device_batch_size=2,
    grad_accum_steps=8,
    gradient_checkpointing=True,
)
```

For a quick smoke test, also pass `n_epochs=1, eval_steps=10, save_steps=10` so evaluation and checkpointing run even on a small dataset.

### VM requirements

| Resource | Need | Note |
|---|---|---|
| GPU | One GPU from the table above | Single GPU; NVLink is irrelevant |
| vCPUs | 8 is enough | Only tokenization and data loading run on CPU |
| RAM | 32 GB is enough | Weights stream from disk straight to the GPU |
| Boot disk | ~100 GiB SSD | Model ~16 GB, CUDA PyTorch environment ~10 GB, OS, checkpoints |
| Image | Ubuntu with NVIDIA drivers preinstalled (e.g. Nebius `ubuntu24.04-cuda13.0`) | `nvidia-smi` must work; Ubuntu 24.04 ships Python 3.12, which meets `requires-python` |
| Region | EU | Messages and the trained adapter are personal data |
| Billing | On-demand for a first run; preemptible later | The notebook checkpoints every `save_steps` and resumes with `STYLE_LORA_RESUME=1` |

Example on Nebius: `gpu-l40s-a` with preset `1gpu-8vcpu-32gb` in `eu-north1` ($1.55/h, preemptible $0.74/h), or `gpu-rtx6000-a` with `1gpu-24vcpu-218gb` in `eu-south1` ($1.80/h).

### Software and setup on the VM

1. Copy the repo to the VM (`rsync`; the notebook may not be committed yet, so `git clone` can miss it).
2. `poetry install` and register the kernel (see `README.md`, `set_up.md`).
3. Check `torch.cuda.is_available()` is `True`; if not, the driver is older than the PyTorch CUDA runtime (see `set_up.md`).
4. Copy the messages file to `<root>/data/my_messages.jsonl`, outside the repo.
5. In a first notebook cell: `%env STYLE_LORA_ROOT=/home/<user>/style-lora` (the default `/mnt/data/style-lora` usually does not exist).
6. After training, copy `<root>/runs/<run_name>` off the VM, then stop or delete the VM from the console or CLI, not with `sudo shutdown`.

### Free step before renting

PARTs 1-3 (config, data validation, tokenization, loss-mask check) need only the tokenizer and run on a laptop CPU. Skip the PART 0 GPU check, fix data problems there, and rent the GPU only for PART 4 onwards.

### Expected cost

Training a few thousand short messages should take minutes (estimate, not measured). Setup (environment install, model download) takes longer. A first session of 1-2 hours costs roughly $1.50-3.60 on an L40S or RTX PRO 6000, excluding VAT and disk.

## 13. Prompt for further research

```
You are helping me research GPUs for working with large language models (LLMs).

About me:
- Data scientist / ML engineer with a software engineering background.
- I know Python, PyTorch and Hugging Face well; I know little about GPU hardware.
- Explain hardware in software terms (memory limits, throughput, instruction
  sets, networking), not gaming or electrical engineering terms.

What I already know (do not repeat it unless I ask):
- VRAM decides what fits; bandwidth decides generation speed; compute and
  number formats (bf16, fp8, fp4) decide training speed; NVLink matters only
  for multi-GPU.
- NVIDIA generations: Volta, Turing, Ampere (bf16), Ada (fp8), Hopper,
  Blackwell (fp4). Ampere or newer for LLM work.
- GDDR vs HBM: HBM is stacked on the package, 2-8 TB/s vs 0.3-1.8 TB/s, and
  explains most of the price gap.
- Product tiers (consumer, datacenter inference, pro, datacenter training,
  rack-scale) and provider types (big clouds, GPU clouds, EU providers,
  marketplaces, notebooks, serverless, managed APIs).
- Memory rule: params x bytes per param; full fine-tuning with Adam about
  16 bytes per param.

Why I need this:
- I fine-tune open-weight LLMs (LoRA, bf16) and rent GPUs in the cloud.
- I want to choose hardware myself and explain the reasoning to clients and
  colleagues at a consultancy.
- Some training data is personal data, so region (EU) and machine ownership
  matter as much as price.

Where I want to go deeper:
- Memory estimation beyond weights: activations, KV cache, logits, optimizer
  state, and how batch size, sequence length and gradient checkpointing change
  them.
- Throughput: how to estimate training time and cost (FLOPs, utilization) and
  inference tokens/s for a given GPU.
- Multi-GPU: data parallel, FSDP, tensor and pipeline parallelism, and when
  NVLink becomes necessary.
- Quantization (8-bit, 4-bit, fp8) and its effect on quality, memory and speed.
- Current prices and availability per provider, especially in the EU.
- Non-NVIDIA options (AMD ROCm, TPU, Trainium) in practice.

How to answer:
- Concise and direct. No filler, no marketing language, no emojis.
- Give numbers (GB, TB/s, TFLOPS, $/GPU-hour) with a date or source; say when
  a figure is approximate or may be outdated.
- Show calculations step by step, then give a recommendation.
- Point out traps (same name covering different hardware, free GPUs without
  bf16, spot reclamation, disks billing while VMs are stopped).
- If unsure or your knowledge may be outdated, say so instead of guessing.
```
