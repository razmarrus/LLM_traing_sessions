# Style LoRA: experiment plan for two model tiers

Status: autumn 2026. Prices are examples; check live prices before renting. Time and memory figures are estimates, not measurements.

## 1. Goal

Retrain the last decoder layers of open-weight models on my messages with `notebooks/style_lora.ipynb`, in two tiers, and compare how well each one picks up my writing style.

| Tier | Purpose | Models |
|---|---|---|
| Small (1-2B) | Cheap check that style transfers at all; pipeline test | Qwen3-1.7B, Llama 3.2 1B Instruct, SmolLM2-1.7B-Instruct |
| Large (7-8B) | Best quality; the models to actually use | Qwen3-8B, Llama 3.1 8B Instruct |

Everything else stays the same across runs: same messages file, same train/eval split, same prompts, same LoRA rank, same optimization settings.

## 2. Models

| Model | Hugging Face ID | Params | Layers | Released | Licence | Tokenizer family |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 1.7B | 28 | Apr 2025 | Apache 2.0, open | Qwen (152k vocab) |
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` | 1.24B | 16 | Sep 2024 | Llama licence, gated | Llama 3 (128k vocab) |
| SmolLM2-1.7B-Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | 24 | Nov 2024 | Apache 2.0, open | SmolLM2 (49k vocab) |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.2B | 36 | Apr 2025 | Apache 2.0, open | Qwen (152k vocab) |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8.0B | 32 | Jul 2024 | Llama licence, gated | Llama 3 (128k vocab) |

Notes:

- All five are drop-in for the notebook: same projection names (`q_proj` ... `down_proj`) and layer path (`model.layers.<i>`). The notebook's prefix check raises an error if a chat template does not fit the loss masking.
- Gated (Llama): accept the licence on each model's Hugging Face page, then log in once on the VM with `hf auth login`. Do not put the token in the notebook.
- Qwen3 has a thinking mode; the notebook disables it. Other templates ignore the `enable_thinking` flag.
- Llama and SmolLM2 have no pad token; the notebook falls back to the end-of-sequence token, and padded positions are excluded from the loss.
- Qwen3-1.7B and Qwen3-8B share a tokenizer, and so do Llama 3.2 1B and Llama 3.1 8B. This matters for comparing eval loss (section 7).

## 3. Run settings per model

Adapt about 25% of the layers in every model, so "last layers" means the same share of each model. Rank and all other settings stay at the notebook defaults (`lora_r=16`, `lora_alpha=32`, all 7 projections, 3 epochs, effective batch 16).

| Model | `n_last_layers` | Params retrained (r=16) | Share of model | Batch settings on 48 GB |
|---|---|---|---|---|
| Qwen3-1.7B | 7 of 28 | ~4.4M | ~0.26% | Defaults |
| Llama 3.2 1B | 4 of 16 | ~2.8M | ~0.23% | Defaults |
| SmolLM2-1.7B | 6 of 24 | ~4.5M | ~0.26% | Defaults |
| Qwen3-8B | 9 of 36 | ~10.9M | ~0.13% | Defaults; on OOM use reduced |
| Llama 3.1 8B | 8 of 32 | ~10.5M | ~0.13% | Defaults; on OOM use reduced |

Defaults: `per_device_batch_size=8`, `grad_accum_steps=2`. Reduced: `per_device_batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`. Both give effective batch 16, so results stay comparable.

Config line per run (PART 1, replace `cfg = TrainConfig()`):

```python
cfg = TrainConfig(model_name="Qwen/Qwen3-1.7B", n_last_layers=7)
cfg = TrainConfig(model_name="meta-llama/Llama-3.2-1B-Instruct", n_last_layers=4)
cfg = TrainConfig(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", n_last_layers=6)
cfg = TrainConfig(model_name="Qwen/Qwen3-8B", n_last_layers=9)
cfg = TrainConfig(model_name="meta-llama/Llama-3.1-8B-Instruct", n_last_layers=8)
```

Use one line per run. Restart the kernel between runs, so GPU memory is freed and the run name and paths are rebuilt. The run name includes the model name, layer count and timestamp, so runs never overwrite each other.

For a first smoke test only, add `n_epochs=1, eval_steps=10, save_steps=10`.

## 4. GPU

One 48 GB VM runs both tiers. Small models need far less memory, but renting a second, smaller GPU saves little and costs a second setup.

| Model | Weights in bf16 | Estimated peak memory, defaults | Fits on 48 GB |
|---|---|---|---|
| Llama 3.2 1B | ~2.5 GB | ~5-8 GB | Yes, large margin |
| Qwen3-1.7B, SmolLM2-1.7B | ~3.4 GB | ~6-10 GB | Yes, large margin |
| Qwen3-8B, Llama 3.1 8B | ~16 GB | ~25-30 GB worst case (512-token batches) | Yes |

Options (Nebius, one GPU, EU region):

| GPU | VRAM | Platform / preset | Region | On-demand | Preemptible |
|---|---|---|---|---|---|
| L40S (recommended) | 48 GB GDDR6 | `gpu-l40s-a` / `1gpu-8vcpu-32gb` | `eu-north1` | $1.55/h | $0.74/h flat |
| RTX PRO 6000 | 96 GB GDDR7 | `gpu-rtx6000-a` / `1gpu-24vcpu-218gb` | `eu-south1` | $1.80/h | from $0.79/h |

The RTX PRO 6000 is about twice as fast in memory bandwidth and leaves room for larger ablations (more layers, higher rank), for $0.25/h more. Either works for this plan.

Not usable: T4, V100 and free Colab GPUs (no bf16). 24 GB GPUs (L4, RTX 3090/4090) are fine for the small tier but tight for the 8B models.

Use on-demand for the first session. Switch to preemptible for repeat runs: the notebook checkpoints every `save_steps` and resumes with `STYLE_LORA_RESUME=1`.

## 5. VM and storage

### VM

| Resource | Size | Why |
|---|---|---|
| GPU | 1x L40S 48 GB (or RTX PRO 6000) | Section 4 |
| vCPUs / RAM | 8 vCPUs / 32 GB | Only tokenization and data loading run on CPU |
| Boot disk | 150 GiB network SSD | Model downloads for all five models ~42 GB, Python environment ~10 GB, OS ~10 GB, checkpoints, headroom |
| Image | `ubuntu24.04-cuda13.0` (NVIDIA drivers preinstalled) | Ubuntu 24.04 ships Python 3.12 |
| Region | EU (`eu-north1` for L40S) | Messages and adapters are personal data |

To use a smaller disk (~100 GiB), delete a model's cache folder under `<root>/hf_cache` after its run is finished.

### What is stored where

| Item | Size | Recreate from | Keep in |
|---|---|---|---|
| Messages file | A few MB | Only I have it | Bucket (private) |
| Model downloads (`hf_cache`) | ~3-16 GB per model | Hugging Face | VM disk only |
| Python environment | ~10 GB | `poetry install` | VM disk only |
| Checkpoints | ~0.1-0.2 GB each, max 3 per run | Re-training | VM disk; only to resume a run |
| Adapter + `run_config.json` | ~15-45 MB per run | Re-training (GPU time) | Bucket, after every run |

All five adapters together are well under 1 GB.

### Storage options

| Option | What | Lifetime | Price | Use |
|---|---|---|---|---|
| Boot disk | VM's own disk | Deleted with the VM | ~$0.05-0.12/GiB-month | Working files |
| Separate data disk or shared filesystem | Disk mounted at `/mnt/data`, the notebook's default root | Survives VM deletion; attach to the next VM | ~$0.05-0.12/GiB-month (shared filesystem $0.08) | Keep `hf_cache` and runs between VMs without re-downloading |
| Object storage bucket | S3-compatible file store | Independent of VMs | Nebius $0.0147/GiB-month | Durable copies of messages and adapters |

Recommended: a private Nebius bucket in the same region as the VM, plus the boot disk. Same provider and region means fast, cheap transfers and data staying in one EU region.

Other bucket providers if needed (prices approximate, from memory; check live pricing): Google Cloud Storage and AWS S3 (~$0.02-0.025/GiB-month in the EU), Cloudflare R2 (~$0.015, no download fees), Backblaze B2 and Hetzner (~$0.005-0.015).

Bucket layout:

```
s3://<bucket>/style-lora/data/my_messages.jsonl
s3://<bucket>/style-lora/runs/<run_name>/adapter/
s3://<bucket>/style-lora/runs/<run_name>/run_config.json
```

Upload and sync commands are in PART 10 of the notebook. Give the VM bucket access through the provider's VM identity or environment variables, never keys in the notebook. Keep public access off.

### How long to keep what

| Item | Keep for | Then |
|---|---|---|
| Bucket (messages, adapters) | The whole project | Delete when no longer needed (GDPR: do not keep personal data longer than needed) |
| VM | Only during a session | Stop it from the console or CLI, not with `sudo shutdown` (Nebius reboots and keeps billing) |
| Boot disk | Between sessions, if the next one is within a week or two (~$8-18/month for 150 GiB) | Otherwise delete the VM; recreating it costs ~20-30 minutes of setup |

## 6. Setup and run order

### Before renting (free, on the laptop)

1. Create the messages file: JSONL, one `{"text": ...}` per line, only my messages, at least ~500 rows. Keep it outside the repo.
2. Accept the Llama 3.1 8B and Llama 3.2 1B licences on Hugging Face.
3. Run PARTs 1-3 on the laptop CPU with each model's tokenizer (skip the PART 0 GPU check). This validates the data and each chat template's loss masking without a GPU.

### Session on the VM

1. Create the VM (section 5) and the bucket; upload the messages file to the bucket.
2. Copy the repo to the VM with `rsync` (the notebook may not be committed yet), then `poetry install` and register the kernel.
3. Check `nvidia-smi` and `torch.cuda.is_available()`.
4. `hf auth login` for the Llama models.
5. Download the messages from the bucket to `<root>/data/my_messages.jsonl`.
6. In the first notebook cell: `%env STYLE_LORA_ROOT=/home/<user>/style-lora`.
7. Smoke test: Qwen3-1.7B with `n_epochs=1, eval_steps=10, save_steps=10`.
8. Small tier: Qwen3-1.7B, Llama 3.2 1B, SmolLM2-1.7B. Restart the kernel between runs. Sync each run folder to the bucket.
9. Check the small tier (section 7). If no model shows any style change, fix the data before continuing.
10. Large tier: Qwen3-8B, Llama 3.1 8B. Check "Peak GPU memory" after the first one. Sync to the bucket.
11. Stop or delete the VM.

### Time and cost (estimates)

Assumes ~5,000 messages of ~100 tokens, 3 epochs, L40S on-demand.

| Step | Time |
|---|---|
| VM setup, environment install | ~20-30 min |
| Model downloads (all five, ~42 GB) | ~10-20 min |
| Training, each small model | ~1-5 min |
| Training, each 8B model | ~6-15 min |
| Evaluation, generation, saving, per run | ~2-5 min |
| Total session | ~2-3 hours, ~$3-5 on L40S (excl. VAT, disk) |

Repeat sessions (new data, ablations) are cheaper: no setup if the disk was kept, and preemptible pricing.

## 7. Comparing quality

### Eval loss: only comparable within a tokenizer

Eval loss is the average loss per token. Different tokenizers split the same text into different numbers of tokens, so loss is not comparable across model families.

| Comparison | Eval loss comparable? |
|---|---|
| Baseline vs adapted, same model | Yes: the main signal that the adapter learned something |
| Qwen3-1.7B vs Qwen3-8B | Yes: same tokenizer |
| Llama 3.2 1B vs Llama 3.1 8B | Yes: same tokenizer |
| Qwen vs Llama vs SmolLM2 | No |

For a cross-family number, convert loss to bits per character: total loss over all eval tokens / number of eval characters / ln 2. The notebook does not compute this yet.

The train/eval split depends only on the messages file and the seed, so every run evaluates on the same held-out messages.

### Per run: what to record

From `run_config.json`: baseline and final eval loss, peak GPU memory, adapted layers, train and eval sizes. Add training time from the PART 6 output.

Training curves (PART 7): eval loss rising while train loss falls means memorization. Use fewer epochs or a lower rank.

### Generated text: the real test

1. Same prompts for every model: the `EVAL_PROMPTS` in PART 1, plus a few more covering typical situations (short reply, longer update, informal and formal).
2. Generate several samples per prompt per model, base and adapted (PART 8 uses a fixed seed and temperature 0.7).
3. Simple style statistics, compared against held-out real messages: message length, sentence length, punctuation and capitalization habits, emoji use, most frequent words and phrases.
4. Blind test: mix held-out real messages with generated ones from each model, without labels. Guess which are real, or rate "sounds like me" from 1 to 5. Repeat with someone who knows my writing.
5. Memorization check: search generated texts for long exact matches with training messages. A good adapter copies style, not sentences.

### Questions to answer

| Question | Evidence |
|---|---|
| Does style transfer at all? | Small tier: adapted vs base output, style statistics |
| How much does model size add? | Qwen3-1.7B vs Qwen3-8B, Llama 3.2 1B vs Llama 3.1 8B: eval loss (same tokenizer) and blind test |
| Which family writes most like me? | Qwen3-8B vs Llama 3.1 8B: blind test, style statistics |
| Is the extra cost of 8B justified? | Quality gap vs time and cost per run |

## 8. Possible follow-ups

- Ablate `n_last_layers` (e.g. 12.5%, 25%, 50%, all) and `lora_r` (8, 16, 32) on the best model.
- Frozen eval file across runs, and confidence intervals on eval loss (bootstrap).
- Context pairs (the message I replied to plus my reply) so the model learns how I answer, not only how I write.
- Bits-per-character metric in the notebook for cross-family comparison.
