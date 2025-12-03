# Barbados Traffic Analysis – Modular Pipeline

## Quick Start (PowerShell)

```powershell
# From the project root
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the modular pipeline (uses sample_data by default)
python run_pipeline.py

# Run tests
pytest -q tests
```

- Config: `config/config.yaml`
- Sample data: `sample_data/Train.csv` (used by default)
- Outputs:
	- Profiling: `reports/profiling_report.md` and `reports/profiling.jsonl`
	- Experiments: `experiments/experiments.jsonl`
	- Model: `models/gb_enter.pkl` and `models/model_card.md`

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
- `reports/` – profiling (`profiling_report.md`, `profiling.jsonl`) and evaluation
- `tests/` – unit tests
- `assets/branding/` – logos/banners
