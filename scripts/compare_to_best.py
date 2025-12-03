import os
import json
import argparse
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
REPORTS_DIR = os.path.join(ROOT, 'reports')
CONFIG_PATH = os.path.join(ROOT, 'config', 'best_submission.json')

os.makedirs(REPORTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True, help='Path to candidate submission CSV')
args = parser.parse_args()

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError('config/best_submission.json missing')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    best_name = json.load(f).get('best_submission')

best_path = None
# search in root and submissions/
for p in [os.path.join(ROOT, best_name), os.path.join(ROOT, 'submissions', best_name)]:
    if p and os.path.exists(p):
        best_path = p
        break
if best_path is None:
    raise FileNotFoundError(f'Best submission {best_name} not found')

best = pd.read_csv(best_path)
cand = pd.read_csv(args.candidate)

# Normalize columns
bcols = {c.lower(): c for c in best.columns}
ccols = {c.lower(): c for c in cand.columns}
for col in ['id','target']:
    if col not in bcols or col not in ccols:
        raise ValueError('Both files must contain ID and Target columns')

bid, btarget = bcols['id'], bcols['target']
cid, ctarget = ccols['id'], ccols['target']

# Align by ID
merged = best[[bid, btarget]].merge(cand[[cid, ctarget]], left_on=bid, right_on=cid, how='inner', suffixes=('_best','_cand'))

# Distributions
def dist(df, col):
    vc = df[col].value_counts(normalize=True) * 100
    return {
        'free': float(vc.get('free flowing', 0.0)),
        'light': float(vc.get('light delay', 0.0)),
        'moderate': float(vc.get('moderate delay', 0.0)),
        'heavy': float(vc.get('heavy delay', 0.0)),
    }

best_d = dist(best, btarget)
cand_d = dist(cand, ctarget)

# Differences by class label
merged['same'] = (merged[f'{btarget}_best'] == merged[f'{ctarget}_cand'])
acc_est = merged['same'].mean() * 100.0

by_label = merged.groupby([f'{btarget}_best', f'{ctarget}_cand']).size().reset_index(name='count')

lines = []
lines.append(f"# Compare Candidate vs Best\n\nGenerated: {datetime.now().isoformat()}\n")
lines.append(f"- Best: {best_name}\n- Candidate: {os.path.basename(args.candidate)}\n- Overlap rows: {len(merged)}\n- Agreement (exact label match): {acc_est:.2f}%\n")
lines.append("\n## Distributions\n")
lines.append(f"- Best: {best_d}")
lines.append(f"- Candidate: {cand_d}\n")

lines.append("## Confusion-like counts (best vs cand)\n")
lines.append(by_label.to_string(index=False))

out_name = f"comparison_{os.path.basename(args.candidate).replace('.csv','')}.md"
out_path = os.path.join(REPORTS_DIR, out_name)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Written: {out_path}")
