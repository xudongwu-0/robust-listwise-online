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

## Built on SAIL

This online reproduction code is built on the SAIL framework for self-improving online alignment. The original SAIL work is:

```bibtex
@article{ding2024sail,
  title={SAIL: Self-Improving Efficient Online Alignment of Large Language Models},
  author={Ding, Mucong and Chakraborty, Souradip and Agrawal, Vibhu and Che, Zora and Koppel, Alec and Wang, Mengdi and Bedi, Amrit and Huang, Furong},
  journal={arXiv preprint arXiv:2406.15567},
  year={2024},
  url={https://arxiv.org/abs/2406.15567}
}