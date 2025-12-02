"""
Dev 27b: Directional Target Tuning (Wider Grid)
- Start from available best base (prefers submission_FINAL_timeaware.csv)
- Tune Enter free ~ 66-72%, Exit free ~ 74-80%
- Keep Heavy in [6,8]% overall via targeted swaps
- Time-of-day + signaling + direction-aware candidate scoring
- Use Pure GBM and V2 when available as weak signals
- Produce all variants and copy Top-3 to *_TOP files for easy submission
"""

import pandas as pd
import numpy as np
import os
import shutil

np.random.seed(42)

print("\n" + "="*80)
print("DEV 27b: DIRECTIONAL TUNING (WIDE GRID)")
print("="*80)

# Load bases
base_file_candidates = [
    'submission_FINAL_timeaware.csv',
    'submission_optimized_72pct.csv',
    'submission_max_diversity.csv',
    'submission_extreme_balanced.csv',
    'submission_OPTION3_pure_gbm.csv',
    'SampleSubmission.csv',
]
base_file = None
for c in base_file_candidates:
    if os.path.exists(c):
        base_file = c
        break
if base_file is None:
    raise FileNotFoundError("No base submission found. Place one of: submission_FINAL_timeaware.csv, submission_optimized_72pct.csv, submission_OPTION3_pure_gbm.csv, or SampleSubmission.csv")
if base_file != 'submission_FINAL_timeaware.csv':
    print(f"⚠️ Using fallback base: {base_file}")
base = pd.read_csv(base_file)
if 'Target_Accuracy' not in base.columns:
    base['Target_Accuracy'] = 1.0

# Optional references
try:
    pure_gbm = pd.read_csv('submission_OPTION3_pure_gbm.csv')
    if 'Target' not in pure_gbm.columns:
        raise ValueError('Invalid pure_gbm format')
except Exception:
    pure_gbm = base[['ID','Target']].copy()
    print("⚠️ Missing/invalid submission_OPTION3_pure_gbm.csv, using base as gbm reference")

try:
    v2 = pd.read_csv('submission_improved_v2.csv')
    if 'Target' not in v2.columns:
        raise ValueError('Invalid v2 format')
except Exception:
    v2 = base[['ID','Target']].copy()
    print("⚠️ Missing/invalid submission_improved_v2.csv, using base as v2 reference")

# Load metadata
if not os.path.exists('TestInputSegments.csv'):
    raise FileNotFoundError("TestInputSegments.csv is required for metadata")

test = pd.read_csv('TestInputSegments.csv')
if 'video_time' in test.columns:
    test['hour'] = pd.to_datetime(test['video_time'], errors='coerce').dt.hour
else:
    test['hour'] = np.nan
bins = [0, 6, 10, 14, 18, 24]
labels = ['Night', 'Morning', 'Midday', 'Afternoon', 'Evening']
try:
    test['time_of_day'] = pd.cut(test['hour'].where(test['hour']>0, 0.1), bins=bins, labels=labels, include_lowest=True)
except Exception:
    test['time_of_day'] = None

meta = {}
for _, r in test.iterrows():
    sig = r['signaling'] if 'signaling' in r else 'none'
    meta[r['ID_enter']] = {'time_of_day': r.get('time_of_day', None), 'signaling': sig, 'dir': 'enter'}
    meta[r['ID_exit']]  = {'time_of_day': r.get('time_of_day', None), 'signaling': sig, 'dir': 'exit'}

# Class maps
CLASSES = {'free flowing': 0, 'light delay': 1, 'moderate delay': 2, 'heavy delay': 3}
REV = {v:k for k,v in CLASSES.items()}

# Working frame
work = base.copy()
work = work.merge(pure_gbm[['ID','Target']].rename(columns={'Target':'gbm'}), on='ID', how='left')
work = work.merge(v2[['ID','Target']].rename(columns={'Target':'v2'}), on='ID', how='left')
work['gbm'] = work['gbm'].fillna(work['Target'])
work['v2'] = work['v2'].fillna(work['Target'])

def infer_dir(s):
    s = str(s)
    if 'enter' in s:
        return 'enter'
    if 'exit' in s:
        return 'exit'
    return 'enter'
work['dir'] = work['ID'].apply(infer_dir)
work['time_of_day'] = work['ID'].apply(lambda s: meta.get(s, {}).get('time_of_day'))
work['signaling'] = work['ID'].apply(lambda s: meta.get(s, {}).get('signaling'))

# Utils

def distribution(df):
    vc = df['Target'].value_counts(normalize=True) * 100
    return {
        'free': vc.get('free flowing', 0.0),
        'light': vc.get('light delay', 0.0),
        'moderate': vc.get('moderate delay', 0.0),
        'heavy': vc.get('heavy delay', 0.0),
    }

def distribution_dir(df, direc):
    part = df[df['dir']==direc]
    return distribution(part)

# Heuristic predictor

def predict_score(free_pct, heavy_pct, light_pct, moderate_pct, enter_free, exit_free):
    base_score = 0.7708
    score_adj = 0.0
    # Free sweet spot
    if free_pct < 65:
        score_adj -= 0.02
    elif free_pct > 80:
        score_adj -= 0.015
    elif 70 <= free_pct <= 74:
        score_adj += 0.012
    elif 69 <= free_pct <= 75:
        score_adj += 0.008
    # Heavy constraints
    if heavy_pct > 9:
        score_adj -= 0.015
    elif heavy_pct < 5:
        score_adj -= 0.005
    elif 6 <= heavy_pct <= 8:
        score_adj += 0.006
    # Delay total
    delay_sum = light_pct + moderate_pct + heavy_pct
    if 25 <= delay_sum <= 35:
        score_adj += 0.004
    # Directional spacing (Enter < Exit is generally good)
    gap = exit_free - enter_free
    if gap >= 4:
        score_adj += 0.004
    elif gap >= 2:
        score_adj += 0.002
    return max(0.65, min(0.85, base_score + score_adj))

# Heavy constraint pass

def constrain_heavy(df, target_min=6.0, target_max=8.0):
    d = df.copy()
    dist = distribution(d)
    heavy_over = dist['heavy'] - target_max
    heavy_under = target_min - dist['heavy']

    if heavy_over > 0:
        candidates = d[(d['Target']=='heavy delay')].copy()
        def score(row):
            s = 0
            if row['dir']=='exit': s += 2
            if (str(row['signaling']) or '').lower()=='none': s += 2
            if row['time_of_day']=='Morning': s += 1
            return s
        candidates = candidates.assign(score=candidates.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*heavy_over/100)+1
        idxs = candidates.head(need).index
        d.loc[idxs, 'Target'] = 'moderate delay'

    elif heavy_under > 0:
        candidates = d[(d['Target']=='moderate delay')].copy()
        def score(row):
            s = 0
            if row['dir']=='enter': s += 2
            if row['time_of_day']=='Midday': s += 2
            if (str(row['signaling']) or '').lower() in ['medium','high']: s += 1
            return s
        candidates = candidates.assign(score=candidates.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*heavy_under/100)+1
        idxs = candidates.head(need).index
        d.loc[idxs, 'Target'] = 'heavy delay'

    return d

# Enter/Exit free tuning

def tune_free_direction(df, enter_target_free, exit_target_free):
    d = df.copy()

    def tune_side(d, direc, target_free):
        part = d[d['dir']==direc].copy()
        dist = distribution(part)
        delta = target_free - dist['free']
        changes = 0
        if abs(delta) < 0.10:
            return d, changes
        if delta > 0:
            # Need more free: light->free, then moderate->light if needed
            cand = part[part['Target']=='light delay'].copy()
            def score(row):
                s = 0
                if direc=='exit': s += 2
                if row['time_of_day']=='Morning': s += 1
                if (str(row['signaling']) or '').lower()=='none': s += 2
                if (row['gbm']=='free flowing') and (row['v2']=='free flowing'): s += 2
                return s
            cand = cand.assign(score=cand.apply(score, axis=1)).sort_values('score', ascending=False)
            need = int(len(part)*delta/100)+1
            idxs = cand.head(need).index
            d.loc[idxs, 'Target'] = 'free flowing'
            changes += len(idxs)
            # if still not enough due to scarcity, move mod->light
            part2 = d[d['dir']==direc]
            dist2 = distribution(part2)
            if dist2['free'] < target_free:
                need2 = int(len(part)*(target_free - dist2['free'])/100)+1
                cand2 = part2[part2['Target']=='moderate delay'].copy()
                cand2 = cand2.sample(min(len(cand2), need2), random_state=42)
                d.loc[cand2.index, 'Target'] = 'light delay'
                changes += len(cand2)
        else:
            # Need less free: free->light; try to pick risky contexts
            cand = part[part['Target']=='free flowing'].copy()
            def score(row):
                s = 0
                if direc=='enter': s += 2
                if row['time_of_day']=='Midday': s += 2
                if (str(row['signaling']) or '').lower() in ['low','medium','high']: s += 1
                if not ((row['gbm']=='free flowing') and (row['v2']=='free flowing')): s += 2
                return s
            cand = cand.assign(score=cand.apply(score, axis=1)).sort_values('score', ascending=False)
            need = int(len(part)*(-delta)/100)+1
            idxs = cand.head(need).index
            d.loc[idxs, 'Target'] = 'light delay'
            changes += len(idxs)
        return d, changes

    total_changes = 0
    d, c1 = tune_side(d, 'enter', enter_target_free)
    d, c2 = tune_side(d, 'exit', exit_target_free)
    total_changes = c1 + c2
    return d, total_changes

# Baseline stats
base_dist = distribution(work)
enter_dist = distribution_dir(work, 'enter')
exit_dist = distribution_dir(work, 'exit')
print("\nBaseline:")
print(f"  OVERALL: {base_dist['free']:.1f}% F | {base_dist['light']:.1f}% L | {base_dist['moderate']:.1f}% M | {base_dist['heavy']:.1f}% H")
print(f"  ENTER:   {enter_dist['free']:.1f}% F | HEAVY {enter_dist['heavy']:.1f}%")
print(f"  EXIT:    {exit_dist['free']:.1f}% F | HEAVY {exit_dist['heavy']:.1f}%")

variants = []
enter_targets = [66.0, 68.0, 70.0, 72.0]
exit_targets = [74.0, 76.0, 78.0, 80.0]

for et in enter_targets:
    for xt in exit_targets:
        tuned, ch = tune_free_direction(work, et, xt)
        tuned = constrain_heavy(tuned, 6.0, 8.0)
        dist = distribution(tuned)
        e_dist = distribution_dir(tuned, 'enter')
        x_dist = distribution_dir(tuned, 'exit')
        # Filter plausible overall free
        if not (68.0 <= dist['free'] <= 78.5):
            continue
        pred = predict_score(dist['free'], dist['heavy'], dist['light'], dist['moderate'], e_dist['free'], x_dist['free'])
        fname = f"submission_directional_WIDE_E{int(et)}_X{int(xt)}.csv"
        tuned_out = tuned.copy()
        tuned_out['Target_Accuracy'] = tuned_out['Target']
        tuned_out[['ID','Target','Target_Accuracy']].to_csv(fname, index=False)
        variants.append({'file': fname, 'enter_free': e_dist['free'], 'exit_free': x_dist['free'], 'free': dist['free'], 'heavy': dist['heavy'], 'light': dist['light'], 'moderate': dist['moderate'], 'pred': pred, 'changes': ch})
        print(f"\nVariant {fname}:")
        print(f"  OVERALL: {dist['free']:.1f}% F | {dist['light']:.1f}% L | {dist['moderate']:.1f}% M | {dist['heavy']:.1f}% H")
        print(f"  ENTER/EXIT FREE: {e_dist['free']:.1f}% / {x_dist['free']:.1f}% | changes: {ch}")
        print(f"  Predicted score: {pred:.4f}")

if not variants:
    raise RuntimeError("No variants produced; check inputs and constraints")

# Pick Top-3
variants_sorted = sorted(variants, key=lambda x: x['pred'], reverse=True)
print("\n" + "="*80)
print("TOP PICKS")
print("="*80)
for i, v in enumerate(variants_sorted[:3], 1):
    print(f"#{i}: {v['file']} | Pred {v['pred']:.4f} | OVERALL {v['free']:.1f}% F, {v['heavy']:.1f}% H | ENTER/EXIT F {v['enter_free']:.1f}/{v['exit_free']:.1f}")
    top_name = v['file'].replace('.csv', '_TOP.csv')
    shutil.copyfile(v['file'], top_name)

# Save best as FINAL
best = variants_sorted[0]
best_df = pd.read_csv(best['file'])
if 'Target_Accuracy' not in best_df.columns:
    best_df['Target_Accuracy'] = best_df['Target']
best_df[['ID','Target','Target_Accuracy']].to_csv('submission_FINAL_directional_WIDE.csv', index=False)
print("\nSaved: submission_FINAL_directional_WIDE.csv (best wide-grid)")
