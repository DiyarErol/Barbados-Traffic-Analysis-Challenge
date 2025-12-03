import os
import json
import glob
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)  # go up from scripts/
REPORTS_DIR = os.path.join(ROOT, 'reports')
CONFIG_PATH = os.path.join(ROOT, 'config', 'best_submission.json')

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load best submission name if provided
best_name = None
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            best_name = json.load(f).get('best_submission')
    except Exception:
        best_name = None

# Locate submission CSVs in repo root and submissions/
csv_files = []
root_candidates = sorted([
    f for f in glob.glob(os.path.join(ROOT, '*.csv'))
    if os.path.basename(f).lower().startswith('submission')
])
csv_files.extend(root_candidates)

sub_dir = os.path.join(ROOT, 'submissions')
if os.path.isdir(sub_dir):
    sub_candidates = sorted([
        f for f in glob.glob(os.path.join(sub_dir, '**', '*.csv'), recursive=True)
        if os.path.basename(f).lower().startswith('submission')
    ])
    csv_files.extend(sub_candidates)

rows = []
summary_lines = []

summary_lines.append(f"# Submission Index Report\n\nGenerated: {datetime.now().isoformat()}\n")
summary_lines.append("- Best (from config): %s\n" % (best_name or 'N/A'))

for path in csv_files:
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        rows.append({
            'file': name,
            'rows': 0,
            'error': f'read_error: {e}'
        })
        continue

    # Normalize columns
    cols = [c.lower() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    has_id = 'id' in cols
    has_target = 'target' in cols
    has_acc = 'target_accuracy' in cols

    if not has_id or not has_target:
        rows.append({
            'file': name,
            'rows': len(df),
            'error': 'missing_required_columns'
        })
        continue

    # Compute distributions
    target_col = colmap['target']
    vc = df[target_col].value_counts(normalize=True) * 100
    free = float(vc.get('free flowing', 0.0))
    light = float(vc.get('light delay', 0.0))
    moderate = float(vc.get('moderate delay', 0.0))
    heavy = float(vc.get('heavy delay', 0.0))

    # Enter/Exit split heuristic via ID
    id_col = colmap['id']
    enter_mask = df[id_col].astype(str).str.contains('enter', case=False, na=False)
    exit_mask = df[id_col].astype(str).str.contains('exit', case=False, na=False)

    def dist(sub):
        v = sub[target_col].value_counts(normalize=True) * 100
        return float(v.get('free flowing', 0.0)), float(v.get('heavy delay', 0.0))

    e_free, e_heavy = dist(df[enter_mask])
    x_free, x_heavy = dist(df[exit_mask])

    is_best = (best_name is not None and name == best_name)

    rows.append({
        'file': name,
        'rows': len(df),
        'has_accuracy': has_acc,
        'free': round(free, 2),
        'light': round(light, 2),
        'moderate': round(moderate, 2),
        'heavy': round(heavy, 2),
        'enter_free': round(e_free, 2),
        'enter_heavy': round(e_heavy, 2),
        'exit_free': round(x_free, 2),
        'exit_heavy': round(x_heavy, 2),
        'is_best': is_best,
        'mtime': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    })

# Save CSV index
index_path = os.path.join(REPORTS_DIR, 'submission_index.csv')
pd.DataFrame(rows).sort_values(by=['is_best','mtime'], ascending=[False, False]).to_csv(index_path, index=False)

# Save markdown summary
# Top-10 by closest to (free ~ 70-74, heavy ~ 6-8)
import math

def score_row(r):
    free = r.get('free', 0.0)
    heavy = r.get('heavy', 0.0)
    # distance to sweet spots
    free_pen = min(abs(free-72), abs(free-73), abs(free-71)) / 100.0
    heavy_pen = min(abs(heavy-7), abs(heavy-6), abs(heavy-8)) / 100.0
    return 1.0 - (free_pen*2 + heavy_pen*1.5)

scored = sorted(rows, key=score_row, reverse=True)[:10]

summary_lines.append("\n## Top Candidates (distribution-based)\n")
for i, r in enumerate(scored, start=1):
    summary_lines.append(f"{i}. {r['file']} | F {r.get('free',0)}% | H {r.get('heavy',0)}% | Enter/Exit F {r.get('enter_free',0)}/{r.get('exit_free',0)} | best={r.get('is_best',False)}")

summary_path = os.path.join(REPORTS_DIR, 'submission_summary.md')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"Written: {index_path}")
print(f"Written: {summary_path}")
