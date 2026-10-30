# GuideVul: CWE-Guided Cross-Language Transfer for Vulnerability Detection in Low-Resource Languages

## Setup

```bash
conda create -n guidevul python=3.11 -y
conda activate guidevul
pip install -r requirements.txt
huggingface-cli download microsoft/codebert-base --local-dir models/codebert-base
huggingface-cli download Salesforce/codet5-base --local-dir models/codet5-base
```

## Run

```bash
python run.py --list                                  # all experiments of the paper
python run.py guidevul_codebert_applicable            # Phase 1 + Phase 2, seeds 42, 1234, 7 x folds 1-5
python run.py baseline_codebert --seeds 42 --folds 1  # one run
python run.py all                                     # every experiment
python report.py                                      # tables of the paper from results/
```

Each run uses one GPU with 16 GB of memory. Results are written to `results/<experiment>/seed_<seed>/fold<k>.json`, logs to `logs/`, and checkpoints to `checkpoints/`.
