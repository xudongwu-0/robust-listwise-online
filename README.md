# Robust Listwise Preference Optimization — Online Experiments

## Online Robust Listwise Experiments

This repository contains the online policy-induced / reward-model-driven experiments for Distributionally Robust Listwise Preference Optimization. It builds on the SAIL-style online alignment pipeline and adds Robust PL / RPL experiments with K=2/K=4 candidate lists and different robustness radii.

For full details, see: [docs/online_reproduction.md](docs/online_reproduction.md)

### Setup

```bash
# Python 3.10.x, CUDA 12.x
pip install -e .
# Fill in your credentials in (these files are .gitignore'd):
#   configs/services/huggingface.yaml
#   configs/services/wandb.yaml
#   configs/services/openai.yaml   (only needed for GPT-as-judge)
```

### Reproduction

```bash
# Quick syntax/smoke check
bash scripts/online/run_quick_test.sh

# Run the online main-table experiments
GPUS=0,1,2,3 NPROCS=4 NUM_STEPS=200 PUSH=1 bash scripts/online/run_main_table.sh

# Evaluate generated models
GPUS=0,1,2,3 python scripts/online/evaluate.py --ranking

# Optional GPT-as-judge evaluation
python scripts/online/evaluate.py --gpt --gpt-ver gpt-35-turbo

# Collect results
python scripts/online/collect_results.py
```

### Hardware note

These runs require GPU resources and may download large language models and reward models. The quick test is intended only to verify that the pipeline is wired correctly.

### Security note

API keys and service credentials should be provided through local untracked config files or environment variables. Do not commit `configs/services/*.yaml` with real secrets.

### Citation

```bibtex
@inproceedings{anonymous2026robustlistwise,
  title={Distributionally Robust Listwise Preference Optimization via Pointwise Total-Variation Ambiguity over Ranking Labels},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

---

## Upstream: SAIL — Self-Improving Efficient Online Alignment of Large Language Models

A lightweight plugin for Huggingface's DPOTrainer, achieving up to 11.6% win-rate improvement with minimal overhead compared to fast DPO.

### Step 1: Filling YAML Configs
1. We need HuggingFace Hub and WanDB to manage experiments. Please fill in `./configs/services/huggingface.yaml` and `./configs/services/wandb.yaml` with your account info.
2. We need OpenAI api to evaluate models. Please fill in `./configs/services/openai.yaml` with your account info.
3. We use a HuggingFace Space App to retrieve and review results. Please fill in `./viewer/.env` with your account info.

### Step 2: Installation
1. With `python 3.10.*` and `CUDA 12.*` installed. You can run `pip install -e .` to install this package called `cdpo`.

### Step 3: Config the Experiments to Run
1. Fill in or modify `./configs/tasks.yaml` for the set of experiments to run.

### Step 4: Run Experiments
1. Run command `cdpo execute` to run all experiments specified.

### Step 5: View Results
1. Inside `./viewer` folder, run `streamlit run app.py` to start the result viewer. Using the UI there to analyse the results.

### Training time and memory requirements
The approximate training time and memory requirements of each SAIL training on three models are: Qwen1.5-0.5B: 1–4 hours with 4×A40 GPUs; Phi-3-3.8B: 2–8 hours with 4×RTX6000Ada GPUs; Llama-3-8B: 2–12 hours with 4×A100 GPUs.