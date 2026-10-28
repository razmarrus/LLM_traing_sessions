# Checklist: retrain a small model with `style_lora.ipynb`

Model: `Qwen/Qwen3-1.7B` (open, no access request). GPU: one Nebius L40S 48 GB in `eu-north1`.
Result: a trained LoRA adapter plus `run_config.json`, stored in a private bucket.

Details and reasoning: `documents/style_lora_experiment_plan.md`.

## 1. Accounts and access

- [ ] Nebius account created, payment method added
- [ ] Project in `eu-north1` selected
- [ ] Capacity advisor shows an L40S available for my quota
- [ ] SSH key exists on the laptop: `ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519`
- [ ] VS Code extensi on "Remote - SSH" installed

## 2. Data (laptop)

- [ ] Messages exported: only my own messages, English
- [ ] Saved as JSONL, one `{"text": "..."}` per line, UTF-8
- [ ] At least ~500 messages
- [ ] File stored outside the repo (not in `data/`)
- [ ] `git status` does not show the messages file

## 3. Free check before renting (laptop, CPU)

- [ ] First notebook cell added: `%env STYLE_LORA_ROOT=<local folder>`
- [ ] `STYLE_LORA_MESSAGES` points to my file, or the file is at `<root>/data/my_messages.jsonl`
- [ ] PART 1 config line set: `cfg = TrainConfig(model_name="Qwen/Qwen3-1.7B", n_last_layers=7)`
- [ ] PART 0 GPU check cell skipped
- [ ] PART 1 runs; printed paths look right
- [ ] PART 2 runs; row counts, lengths and samples look right, no warning about too few messages
- [ ] PART 3 runs; no chat template error
- [ ] "TRAINED ON" output shows only my message plus the end-of-turn token
- [ ] Notebook outputs cleared (they contain my messages)

## 4. Storage bucket

- [ ] Private bucket created in Nebius Object Storage, `eu-north1`
- [ ] Access key created for the bucket (Nebius docs: service account + access key)
- [ ] AWS CLI configured on the laptop with that key
- [ ] Messages uploaded:
  `aws s3 cp my_messages.jsonl s3://<bucket>/style-lora/data/ --endpoint-url https://storage.eu-north1.nebius.cloud`
- [ ] Upload visible: `aws s3 ls s3://<bucket>/style-lora/data/ --endpoint-url https://storage.eu-north1.nebius.cloud`

## 5. Create the VM

- [ ] Compute > Virtual machines > Create: With GPUs, Regular (on-demand)
- [ ] Platform `gpu-l40s-a` (L40S with Intel), preset `1gpu-8vcpu-32gb`
- [ ] Boot disk: image `ubuntu24.04-cuda13.0`, 100 GiB
- [ ] Public IP: Auto (dynamic)
- [ ] Username (not `root` or `admin`) and my public SSH key added
- [ ] VM created and status is Running
- [ ] Public IP noted; billing has started

## 6. Connect

- [ ] `~/.ssh/config` entry added on the laptop:
  ```
  Host style-lora
      HostName <public_ip>
      User <username>
      IdentityFile ~/.ssh/id_ed25519
  ```
- [ ] `ssh style-lora` works
- [ ] `nvidia-smi` on the VM shows the L40S

## 7. Environment on the VM

- [ ] Repo copied from the laptop:
  `rsync -av --exclude .venv --exclude lora-session ~/projects/llm_xomnia_course/ style-lora:~/llm_xomnia_course/`
- [ ] Poetry installed: `sudo apt-get update && sudo apt-get install -y pipx && pipx install poetry && pipx ensurepath && source ~/.bashrc`
- [ ] Dependencies installed: `cd ~/llm_xomnia_course && poetry install`
- [ ] Kernel registered: `poetry run python -m ipykernel install --user --name=llm-sessions --display-name "LLM Sessions"`
- [ ] CUDA works: `poetry run python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] AWS CLI installed and configured on the VM with the bucket key
- [ ] Messages downloaded:
  `aws s3 cp s3://<bucket>/style-lora/data/my_messages.jsonl ~/style-lora/data/ --endpoint-url https://storage.eu-north1.nebius.cloud`

## 8. Open the notebook on the VM

- [ ] VS Code: Remote-SSH: Connect to Host > `style-lora`
- [ ] Folder `~/llm_xomnia_course` opened
- [ ] Python and Jupyter extensions installed on the remote when prompted
- [ ] `notebooks/style_lora.ipynb` opened, kernel "LLM Sessions" selected
- [ ] First cell: `%env STYLE_LORA_ROOT=/home/<username>/style-lora`

## 9. Smoke test (short run)

- [ ] PART 1 config line: `cfg = TrainConfig(model_name="Qwen/Qwen3-1.7B", n_last_layers=7, n_epochs=1, eval_steps=10, save_steps=10)`
- [ ] Run all cells
- [ ] PART 0 shows the L40S, ~48 GB, bf16 OK
- [ ] PART 4 downloads the model and loads it without errors
- [ ] PART 5 shows layers 21-27 adapted, ~4.4M trainable parameters
- [ ] PART 6 trains without OOM; eval loss is printed during training
- [ ] PART 9 saves an adapter folder

## 10. Full run

- [ ] Kernel restarted
- [ ] PART 1 config line: `cfg = TrainConfig(model_name="Qwen/Qwen3-1.7B", n_last_layers=7)`
- [ ] Run all cells
- [ ] Run name noted (printed in PART 1 as `run_dir`)
- [ ] Final eval loss is lower than the baseline eval loss
- [ ] Training curves (PART 7): eval loss does not rise steadily while train loss falls
- [ ] Peak GPU memory noted
- [ ] PART 8: adapted output differs from base output and sounds more like me
- [ ] PART 9: `adapter/` and `run_config.json` saved under `~/style-lora/runs/<run_name>/`

## 11. Save the result

- [ ] Run folder synced to the bucket (on the VM):
  `aws s3 sync ~/style-lora/runs/<run_name> s3://<bucket>/style-lora/runs/<run_name> --endpoint-url https://storage.eu-north1.nebius.cloud`
- [ ] Sync verified from the laptop:
  `aws s3 ls s3://<bucket>/style-lora/runs/<run_name>/adapter/ --endpoint-url https://storage.eu-north1.nebius.cloud`
- [ ] Notes kept: run name, baseline and final eval loss, peak memory, training time, a few generated samples

## 12. Shut down

- [ ] Optional, for the other small models: repeat section 10 with `meta-llama/Llama-3.2-1B-Instruct` (`n_last_layers=4`, after `hf auth login`) and `HuggingFaceTB/SmolLM2-1.7B-Instruct` (`n_last_layers=6`), and sync each run
- [ ] VM stopped from the console or with `nebius compute instance stop --id <VM_ID>` (not `sudo shutdown`)
- [ ] Or, if no session follows within a week or two: VM deleted with `nebius compute instance delete --id <VM_ID>`
- [ ] Console shows the VM as Stopped or gone
- [ ] Notebook outputs cleared before any commit

## Done when

- [ ] Adapter and `run_config.json` are in the bucket
- [ ] Final eval loss is below the baseline
- [ ] Adapted output reads more like me than base output
- [ ] No VM is running
