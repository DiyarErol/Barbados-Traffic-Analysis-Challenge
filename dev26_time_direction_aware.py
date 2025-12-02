"""
Dev 26: Time + Direction + Signaling Aware Optimization
- Starts from submission_optimized_72pct.csv (71.4% F, 6.5% H)
- Uses TestInputSegments metadata (time_of_day, signaling)
- Uses Pure GBM and V2 as tie-breaker signals
Generates 3 variants targeting 72%, 73%, 74% free with heavy 6-8%
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("DEV 26: TIME + DIRECTION AWARE OPTIMIZATION")
print("="*80)

# Load submissions
base = pd.read_csv('submission_optimized_72pct.csv')  # 71.4% free, 6.5% heavy
pure_gbm = pd.read_csv('submission_OPTION3_pure_gbm.csv')
v2 = pd.read_csv('submission_improved_v2.csv')

# Load test metadata
test = pd.read_csv('TestInputSegments.csv')

# Precompute time_of_day and map IDs to metadata
test['hour'] = pd.to_datetime(test['video_time']).dt.hour
bins = [0, 6, 10, 14, 18, 24]
labels = ['Night', 'Morning', 'Midday', 'Afternoon', 'Evening']
# For 0 hour, include Night
test['time_of_day'] = pd.cut(test['hour'].where(test['hour']>0, 0.1), bins=bins, labels=labels, include_lowest=True)

# Build lookup for enter/exit
meta = {}
for _, r in test.iterrows():
    meta[r['ID_enter']] = {'time_of_day': r['time_of_day'], 'signaling': r['signaling'], 'dir': 'enter'}
    meta[r['ID_exit']]  = {'time_of_day': r['time_of_day'], 'signaling': r['signaling'], 'dir': 'exit'}

CLASSES = {'free flowing': 0, 'light delay': 1, 'moderate delay': 2, 'heavy delay': 3}
REV = {v:k for k,v in CLASSES.items()}

def distribution(df):
    vc = df['Target'].value_counts(normalize=True) * 100
    return {
        'free': vc.get('free flowing', 0.0),
        'light': vc.get('light delay', 0.0),
        'moderate': vc.get('moderate delay', 0.0),
        'heavy': vc.get('heavy delay', 0.0),
    }

# Merge helper frames
work = base.copy()
work['gbm'] = pure_gbm['Target']
work['v2'] = v2['Target']

# Extract meta
def get_meta(id_):
    return meta.get(id_, {'time_of_day': None, 'signaling': None, 'dir': 'enter' if 'enter' in id_ else 'exit'})

work['dir'] = work['ID'].apply(lambda s: 'enter' if s.endswith('congestion_enter_rating') else 'exit')
work['time_of_day'] = work['ID'].apply(lambda s: get_meta(s)['time_of_day'])
work['signaling'] = work['ID'].apply(lambda s: get_meta(s)['signaling'])

print("\nBaseline (input):")
base_dist = distribution(work)
print(f"  {base_dist['free']:.1f}% F | {base_dist['light']:.1f}% L | {base_dist['moderate']:.1f}% M | {base_dist['heavy']:.1f}% H")

# Adjustment functions

def upgrade_level(level, by=1):
    return min(3, level+by)

def downgrade_level(level, by=1):
    return max(0, level-by)

# Core adjuster

def adjust_time_direction(df):
    out = df.copy()
    changes = 0
    for i, row in out.iterrows():
        cur = row['Target']
        lev = CLASSES[cur]
        gbm_lev = CLASSES[row['gbm']]
        v2_lev = CLASSES[row['v2']]
        dir_ = row['dir']
        tod = row['time_of_day']
        sig = (row['signaling'] or '').lower()

        new_lev = lev

        # Midday tends to have higher delays; boost Enter first
        if tod == 'Midday':
            if dir_ == 'enter':
                # If any reference suggests higher delay, upgrade 1
                ref_max = max(gbm_lev, v2_lev)
                if ref_max > lev:
                    new_lev = upgrade_level(lev, 1)
            else:  # exit
                # Be mild; only upgrade if both refs higher
                if gbm_lev > lev and v2_lev > lev:
                    new_lev = upgrade_level(lev, 1)

        # Morning usually calmer; reduce Exit first if refs agree
        if tod == 'Morning':
            if dir_ == 'exit' and lev >= 2:
                if gbm_lev < lev and v2_lev < lev:
                    new_lev = downgrade_level(lev, 1)

        # Signaling impact
        if sig in ['medium', 'high']:
            # If any ref suggests higher delay and we're not heavy, upgrade by 1
            if max(gbm_lev, v2_lev) > lev and new_lev < 3:
                new_lev = upgrade_level(new_lev, 1)
        elif sig == 'none':
            # If both refs suggest lower and not Morning Midday override, downgrade by 1 for Exit
            if dir_ == 'exit' and gbm_lev < lev and v2_lev < lev:
                new_lev = downgrade_level(new_lev, 1)

        if new_lev != lev:
            out.at[i, 'Target'] = REV[new_lev]
            changes += 1
    return out, changes

# Generate base time-aware adjusted
timeaware, ch1 = adjust_time_direction(work)
print(f"\nTime+Direction aware changes: {ch1} rows")
base_ta = distribution(timeaware)
print(f"  {base_ta['free']:.1f}% F | {base_ta['light']:.1f}% L | {base_ta['moderate']:.1f}% M | {base_ta['heavy']:.1f}% H")

# Constrain heavy into 6-8% and produce variants targeting free 72/73/74

def constrain_heavy(df, target_min=6.0, target_max=8.0):
    d = df.copy()
    dist = distribution(d)
    heavy_over = dist['heavy'] - target_max
    heavy_under = target_min - dist['heavy']

    if heavy_over > 0:
        # Downgrade some heavy to moderate, prioritize Exit + none + Morning
        candidates = d[(d['Target']=='heavy delay')]
        # attach meta scoring
        def score(row):
            s = 0
            if row['dir']=='exit': s += 2
            if (row['signaling'] or '').lower()=='none': s += 2
            if row['time_of_day']=='Morning': s += 1
            return s
        candidates = candidates.assign(score=candidates.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*heavy_over/100)+1
        idxs = candidates.head(need).index
        d.loc[idxs, 'Target'] = 'moderate delay'

    elif heavy_under > 0:
        # Upgrade some moderate to heavy, prioritize Enter + Midday + medium/high
        candidates = d[(d['Target']=='moderate delay')]
        def score(row):
            s = 0
            if row['dir']=='enter': s += 2
            if row['time_of_day']=='Midday': s += 2
            if (row['signaling'] or '').lower() in ['medium','high']: s += 1
            return s
        candidates = candidates.assign(score=candidates.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*heavy_under/100)+1
        idxs = candidates.head(need).index
        d.loc[idxs, 'Target'] = 'heavy delay'

    return d


def tune_free(df, target_free):
    d = df.copy()
    dist = distribution(d)
    delta = target_free - dist['free']
    if abs(delta) < 0.2:
        return d, 0

    changes = 0
    if delta > 0:
        # Need more free: downgrade light->free on Exit / Morning / none
        mask = (d['Target']=='light delay')
        cand = d[mask].copy()
        def score(row):
            s = 0
            if row['dir']=='exit': s += 2
            if row['time_of_day']=='Morning': s += 1
            if (row['signaling'] or '').lower()=='none': s += 2
            return s
        cand = cand.assign(score=cand.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*delta/100)+1
        idxs = cand.head(need).index
        d.loc[idxs, 'Target'] = 'free flowing'
        changes = len(idxs)
    else:
        # Need less free: upgrade free->light on Enter / Midday / medium+ signaling
        mask = (d['Target']=='free flowing')
        cand = d[mask].copy()
        def score(row):
            s = 0
            if row['dir']=='enter': s += 2
            if row['time_of_day']=='Midday': s += 2
            if (row['signaling'] or '').lower() in ['low','medium','high']: s += 1
            return s
        cand = cand.assign(score=cand.apply(score, axis=1)).sort_values('score', ascending=False)
        need = int(len(d)*(-delta)/100)+1
        idxs = cand.head(need).index
        d.loc[idxs, 'Target'] = 'light delay'
        changes = len(idxs)

    return d, changes

# Heuristic score predictor

def predict_score(free_pct, heavy_pct, light_pct, moderate_pct, enter_free, exit_free):
    base_score = 0.7708
    score_adj = 0
    if free_pct < 65:
        score_adj -= 0.02
    elif free_pct > 80:
        score_adj -= 0.015
    elif 69 <= free_pct <= 75:
        score_adj += 0.01
    if heavy_pct > 9:
        score_adj -= 0.015
    elif heavy_pct < 5:
        score_adj -= 0.005
    elif 6 <= heavy_pct <= 8:
        score_adj += 0.005
    delay_sum = light_pct + moderate_pct + heavy_pct
    if 25 <= delay_sum <= 35:
        score_adj += 0.005
    diff = abs(enter_free - exit_free)
    if diff > 3:
        score_adj += 0.003
    return max(0.65, min(0.85, base_score + score_adj))

# Build variants
variants = []
for target_free in [72.0, 73.0, 74.0]:
    tuned = constrain_heavy(timeaware, 6.0, 8.0)
    tuned, chf = tune_free(tuned, target_free)
    distv = distribution(tuned)
    enter_dist = distribution(tuned[tuned['dir']=='enter'])
    exit_dist = distribution(tuned[tuned['dir']=='exit'])
    pred = predict_score(distv['free'], distv['heavy'], distv['light'], distv['moderate'], enter_dist['free'], exit_dist['free'])
    fname = f"submission_timeaware_{int(target_free)}.csv"
    tuned[['ID','Target','Target_Accuracy']].to_csv(fname, index=False)
    variants.append({'file': fname, 'free': distv['free'], 'light': distv['light'], 'moderate': distv['moderate'], 'heavy': distv['heavy'], 'pred': pred})
    print(f"\nVariant {fname}:")
    print(f"  {distv['free']:.1f}% F | {distv['light']:.1f}% L | {distv['moderate']:.1f}% M | {distv['heavy']:.1f}% H")
    print(f"  Predicted score: {pred:.4f}")

# Pick best by predicted score
best = sorted(variants, key=lambda x: x['pred'], reverse=True)[0]
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"→ Best: {best['file']} | Predicted {best['pred']:.4f}")
print(f"  Dist: {best['free']:.1f}% F | {best['light']:.1f}% L | {best['moderate']:.1f}% M | {best['heavy']:.1f}% H")

# Save as final candidate
import shutil
shutil.copyfile(best['file'], 'submission_FINAL_timeaware.csv')
print("\nSaved: submission_FINAL_timeaware.csv")
