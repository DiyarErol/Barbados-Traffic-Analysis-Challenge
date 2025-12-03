# Barbados Traffic Analysis – Modular Pipeline

## Quick Start

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
python run_pipeline.py
```

- Config: `config/config.yaml`
- Outputs: `reports/profiling_report.md`, `experiments/experiments.jsonl`, `models/gb_enter.pkl`
- Sample data: `sample_data/Train.csv`

## Architecture Overview

![Architecture](docs/architecture.png)

- `src/modules/data_loader.py` – CSV loading
- `src/modules/preprocessing.py` – cleaning and NA handling
- `src/modules/feature_engineering.py` – basic time features
- `src/modules/model_training.py` – baseline Gradient Boosting
- `src/modules/inference.py` – predictions
- `src/modules/evaluation.py` – metrics and confusion matrix

## Folders
- `config/` – central configuration
- `src/modules/` – modular code
- `models/` – saved models + model_card
- `experiments/` – experiment logs
- `reports/` – profiling and evaluation
- `tests/` – unit tests
- `assets/branding/` – logos/banners
