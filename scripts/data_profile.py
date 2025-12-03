import os
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

train_path = os.path.join(ROOT, 'Train.csv')
test_path = os.path.join(ROOT, 'TestInputSegments.csv')

lines = []
lines.append(f"# Data Profile\n\nGenerated: {datetime.now().isoformat()}\n")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    lines.append("Missing Train.csv or TestInputSegments.csv\n")
else:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Try to infer label columns for train
    tcols = [c.lower() for c in train.columns]
    cand_enter = [c for c in train.columns if ('enter' in c.lower() and 'rating' in c.lower()) or ('enter' in c.lower() and 'target' in c.lower())]
    cand_exit  = [c for c in train.columns if ('exit' in c.lower() and 'rating' in c.lower()) or ('exit' in c.lower() and 'target' in c.lower())]

    def class_dist(series):
        vc = series.value_counts(normalize=True) * 100
        return {
            'free': float(vc.get('free flowing', 0.0)),
            'light': float(vc.get('light delay', 0.0)),
            'moderate': float(vc.get('moderate delay', 0.0)),
            'heavy': float(vc.get('heavy delay', 0.0)),
        }

    lines.append("## Train Distributions\n")
    if cand_enter:
        d = class_dist(train[cand_enter[0]])
        lines.append(f"- Enter: {d}\n")
    else:
        lines.append("- Enter: (label column not found)\n")
    if cand_exit:
        d = class_dist(train[cand_exit[0]])
        lines.append(f"- Exit: {d}\n")
    else:
        lines.append("- Exit: (label column not found)\n")

    # Test time-of-day and signaling (if available)
    lines.append("\n## Test Meta\n")
    if 'video_time' in test.columns:
        test['hour'] = pd.to_datetime(test['video_time'], errors='coerce').dt.hour
        bins = [0, 6, 10, 14, 18, 24]
        labels = ['Night', 'Morning', 'Midday', 'Afternoon', 'Evening']
        tod = pd.cut(test['hour'].where(test['hour']>0, 0.1), bins=bins, labels=labels, include_lowest=True)
        lines.append("- Time-of-day distribution:\n")
        lines.append(tod.value_counts(normalize=True).to_string())
        lines.append("\n")
    else:
        lines.append("- Time-of-day: (video_time not found)\n")

    if 'signaling' in test.columns:
        lines.append("- Signaling distribution:\n")
        lines.append(test['signaling'].astype(str).value_counts(normalize=True).to_string())
        lines.append("\n")
    else:
        lines.append("- Signaling: (column not found)\n")

profile_path = os.path.join(REPORTS_DIR, 'data_profile.md')
with open(profile_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Written: {profile_path}")
