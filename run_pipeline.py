import time
import json
from pathlib import Path
import yaml
import psutil

import pandas as pd

from src.modules.data_loader import DataLoader
from src.modules.preprocessing import Preprocessing
from src.modules.feature_engineering import FeatureEngineering
from src.modules.model_training import ModelTraining
from src.modules.inference import Inference
from src.modules.evaluation import Evaluation


def _profile_section(name: str, start_time: float, outputs_dir: Path, extra: dict = None) -> dict:
    duration = time.time() - start_time
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=None)
    info = {
        'section': name,
        'duration_sec': round(duration, 4),
        'rss_mb': round(mem_info.rss / (1024 ** 2), 2),
        'cpu_percent': cpu_percent,
    }
    if extra:
        info.update(extra)
    (outputs_dir / 'profiling.jsonl').write_text(
        ((outputs_dir / 'profiling.jsonl').read_text() if (outputs_dir / 'profiling.jsonl').exists() else '') + json.dumps(info) + "\n"
    )
    return info


def main():
    cfg = yaml.safe_load(Path('config/config.yaml').read_text())

    outputs_dir = Path(cfg['paths']['reports_dir'])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    profiling_md = outputs_dir / 'profiling_report.md'
    profiling_md.write_text('# Profiling Report\n\n')

    # Load
    t0 = time.time()
    dl = DataLoader(train_csv=Path(cfg['paths']['train_csv']), test_csv=Path(cfg['paths']['test_csv']))
    train_df = dl.load_train()
    test_df = dl.load_test()
    p_load = _profile_section('data_load', t0, outputs_dir, {'train_rows': len(train_df), 'test_rows': len(test_df or [])})
    profiling_md.write_text(profiling_md.read_text() + f"- data_load: {p_load}\n")

    # Preprocess
    t1 = time.time()
    pp = Preprocessing()
    train_df = pp.clean(train_df)
    if test_df is not None:
        test_df = pp.clean(test_df)
    p_prep = _profile_section('preprocessing', t1, outputs_dir)
    profiling_md.write_text(profiling_md.read_text() + f"- preprocessing: {p_prep}\n")

    # Features
    t2 = time.time()
    fe = FeatureEngineering()
    train_df = fe.add_basic_time_features(train_df, cfg['features']['time']['time_col'])
    if test_df is not None:
        test_df = fe.add_basic_time_features(test_df, cfg['features']['time']['time_col'])
    feature_cols = ['hour', 'day_of_week']
    p_feat = _profile_section('feature_engineering', t2, outputs_dir, {'features': feature_cols})
    profiling_md.write_text(profiling_md.read_text() + f"- feature_engineering: {p_feat}\n")

    # Train
    t3 = time.time()
    mt = ModelTraining()
    params = cfg['model']['params']
    model, score = mt.train_simple(train_df, feature_cols, target_col='congestion_enter_rating', params=params)
    models_dir = Path(cfg['paths']['models_dir']); models_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(model, models_dir / 'gb_enter.pkl')
    p_train = _profile_section('model_training', t3, outputs_dir, {'f1_macro': score})
    profiling_md.write_text(profiling_md.read_text() + f"- model_training: {p_train}\n")

    # Inference + Eval (if test labels available simulate)
    if test_df is not None and 'congestion_enter_rating' in test_df.columns:
        t4 = time.time()
        inf = Inference()
        pred = inf.predict(model, test_df, feature_cols)
        ev = Evaluation()
        metrics = ev.evaluate(test_df['congestion_enter_rating'], pred)
        (outputs_dir / 'evaluation.json').write_text(json.dumps(metrics, indent=2))
        p_inf = _profile_section('inference_evaluation', t4, outputs_dir, {'f1_macro_test': metrics['f1_macro']})
        profiling_md.write_text(profiling_md.read_text() + f"- inference_evaluation: {p_inf}\n")

    # Experiment tracking
    exp_dir = Path(cfg['paths']['experiments_dir']); exp_dir.mkdir(exist_ok=True)
    exp = {
        'id': int(time.time()),
        'date': pd.Timestamp.now().isoformat(),
        'model': {'type': cfg['model']['type'], 'params': params},
        'features': feature_cols,
        'metrics': {'f1_macro_val': score},
    }
    (exp_dir / 'experiments.jsonl').write_text(
        ((exp_dir / 'experiments.jsonl').read_text() if (exp_dir / 'experiments.jsonl').exists() else '') + json.dumps(exp) + "\n"
    )

    print('Pipeline completed. See reports/profiling_report.md and experiments/experiments.jsonl')


if __name__ == '__main__':
    main()
