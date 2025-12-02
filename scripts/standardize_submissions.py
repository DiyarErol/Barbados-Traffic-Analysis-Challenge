import os
import glob
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
SAMPLE_PATH = os.path.join(ROOT, 'SampleSubmission.csv')
SUB_DIRS = [ROOT, os.path.join(ROOT, 'submissions')]
OUT_DIR = os.path.join(ROOT, 'submissions_fixed')

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(SAMPLE_PATH):
    raise FileNotFoundError('SampleSubmission.csv not found in project root')

sample = pd.read_csv(SAMPLE_PATH)
required_cols = ['ID','Target','Target_Accuracy']

# Allowed canonical labels
CANONICAL = {
    'free flowing': 'free flowing',
    'free': 'free flowing',
    'free_flowing': 'free flowing',
    'free-flowing': 'free flowing',
    'light delay': 'light delay',
    'light': 'light delay',
    'light_delay': 'light delay',
    'light-delay': 'light delay',
    'moderate delay': 'moderate delay',
    'moderate': 'moderate delay',
    'moderate_delay': 'moderate delay',
    'moderate-delay': 'moderate delay',
    'heavy delay': 'heavy delay',
    'heavy': 'heavy delay',
    'heavy_delay': 'heavy delay',
    'heavy-delay': 'heavy delay',
}

processed = []

for base in SUB_DIRS:
    pattern = os.path.join(base, 'submission*.csv')
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        out_path = os.path.join(OUT_DIR, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            processed.append({'file': name, 'status': 'read_error', 'detail': str(e)})
            continue

        # Normalize headers
        colmap = {c.lower(): c for c in df.columns}
        if 'id' not in colmap or 'target' not in colmap:
            processed.append({'file': name, 'status': 'skip', 'detail': 'missing ID/Target columns'})
            continue
        # Ensure Target_Accuracy exists and equals Target
        if 'target_accuracy' in colmap:
            df = df.rename(columns={colmap['id']:'ID', colmap['target']:'Target', colmap['target_accuracy']:'Target_Accuracy'})
        else:
            df = df.rename(columns={colmap['id']:'ID', colmap['target']:'Target'})
            df['Target_Accuracy'] = df['Target']

        # Canonicalize label values
        def canon(v: str) -> str:
            key = str(v).strip().lower()
            return CANONICAL.get(key, str(v))
        df['Target'] = df['Target'].apply(canon)
        df['Target_Accuracy'] = df['Target']  # enforce equality

        # Reorder columns exactly
        df = df[required_cols]

        # Check ID set vs sample
        sample_ids = set(sample['ID'])
        df_ids = set(df['ID'])
        if sample_ids != df_ids:
            missing = len(sample_ids - df_ids)
            extra = len(df_ids - sample_ids)
            processed.append({'file': name, 'status': 'skip', 'detail': f'id_mismatch missing={missing} extra={extra}'})
            continue

        # Reorder rows to sample order for determinism
        df = df.set_index('ID').loc[sample['ID']].reset_index()

        df.to_csv(out_path, index=False)
        processed.append({'file': name, 'status': 'fixed', 'detail': f'written {os.path.relpath(out_path, ROOT)}'})

# Write summary
summary_path = os.path.join(OUT_DIR, f'standardize_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
pd.DataFrame(processed).to_csv(summary_path, index=False)
print(f"✓ Standardization complete. Summary: {summary_path}")
