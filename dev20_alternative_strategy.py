"""
Dev 20: Alternative Strategy - Boost Delay Classes
===================================================
If minimal changes didn't work, try more aggressive
delay class boosting based on GBM's strength
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("DEV 20: ALTERNATIVE STRATEGY - AGGRESSIVE DELAY BOOST")
print("="*80)

# Load submissions
print("\n[LOAD] Loading submissions...")
best = pd.read_csv('submission_final_ensemble.csv')  # 0.7708
gbm = pd.read_csv('submission_gbm_blend.csv')
cond = pd.read_csv('submission_conditional_calibrated.csv')

print(f"  Best (0.7708): {best['Target'].value_counts(normalize=True)['free flowing']*100:.1f}% free")
print(f"  GBM:           {gbm['Target'].value_counts(normalize=True)['free flowing']*100:.1f}% free")
print(f"  Cond:          {cond['Target'].value_counts(normalize=True)['free flowing']*100:.1f}% free")

# Parse segment info
best['SegmentID'] = best['ID'].str.extract(r'(\d+)').astype(int)
best['Direction'] = best['ID'].str.extract(r'_(enter|exit)')[0]
best['pred_gbm'] = gbm['Target'].values
best['pred_cond'] = cond['Target'].values

print("\n[ANALYSIS] Finding patterns...")

# Strategy: GBM has 69.1% free (best balance)
# Let's trust GBM more aggressively for delay predictions

new_predictions = []

for idx, row in best.iterrows():
    best_pred = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # Strategy 1: If GBM predicts delay and best predicts free, trust GBM
    if best_pred == 'free flowing' and gbm_pred != 'free flowing':
        # Check if at least one other model agrees with GBM
        if cond_pred == gbm_pred:
            # Strong signal - both GBM and Cond say delay
            new_predictions.append(gbm_pred)
        elif cond_pred != 'free flowing':
            # Both predict delay but different types - trust GBM
            new_predictions.append(gbm_pred)
        else:
            # Only GBM predicts delay - be conservative, keep original
            new_predictions.append(best_pred)
    
    # Strategy 2: If all three disagree, prefer GBM (best single model)
    elif best_pred != gbm_pred and best_pred != cond_pred and gbm_pred != cond_pred:
        new_predictions.append(gbm_pred)
    
    # Strategy 3: If best and cond agree on free, but GBM says delay
    elif best_pred == 'free flowing' and cond_pred == 'free flowing' and gbm_pred != 'free flowing':
        # 2 vs 1, but GBM is strongest - compromise: keep free
        new_predictions.append(best_pred)
    
    else:
        # Keep original
        new_predictions.append(best_pred)

# Analyze
changes = sum([1 for i in range(len(new_predictions)) if new_predictions[i] != best['Target'].values[i]])
print(f"\n[CHANGES] {changes} predictions changed")

dist = pd.Series(new_predictions).value_counts(normalize=True)
orig_dist = best['Target'].value_counts(normalize=True)

print(f"\nDistribution comparison:")
print(f"  Free:     {dist.get('free flowing', 0)*100:.1f}% (was {orig_dist.get('free flowing', 0)*100:.1f}%)")
print(f"  Light:    {dist.get('light delay', 0)*100:.1f}% (was {orig_dist.get('light delay', 0)*100:.1f}%)")
print(f"  Moderate: {dist.get('moderate delay', 0)*100:.1f}% (was {orig_dist.get('moderate delay', 0)*100:.1f}%)")
print(f"  Heavy:    {dist.get('heavy delay', 0)*100:.1f}% (was {orig_dist.get('heavy delay', 0)*100:.1f}%)")

# Save
submission1 = pd.DataFrame({
    'ID': best['ID'],
    'Target': new_predictions,
    'Target_Accuracy': new_predictions
})
submission1.to_csv('submission_strategy_v2.csv', index=False)
print("\n[OK] Saved submission_strategy_v2.csv")

# Alternative: Pure GBM weighted approach
print("\n[ALTERNATIVE] Creating GBM-dominant blend...")

gbm_dominant = []
for idx, row in best.iterrows():
    best_pred = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    direction = row['Direction']
    
    # Give GBM 60% weight, best 30%, cond 10%
    votes = Counter()
    votes[gbm_pred] += 0.60
    votes[best_pred] += 0.30
    votes[cond_pred] += 0.10
    
    # Bonus: if it's a delay prediction from GBM, add extra weight
    if gbm_pred != 'free flowing':
        votes[gbm_pred] += 0.15
    
    gbm_dominant.append(votes.most_common(1)[0][0])

dist2 = pd.Series(gbm_dominant).value_counts(normalize=True)
changes2 = sum([1 for i in range(len(gbm_dominant)) if gbm_dominant[i] != best['Target'].values[i]])

print(f"\nGBM-Dominant Distribution:")
print(f"  Free:     {dist2.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist2.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist2.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist2.get('heavy delay', 0)*100:.1f}%")
print(f"  Changes: {changes2}")

submission2 = pd.DataFrame({
    'ID': best['ID'],
    'Target': gbm_dominant,
    'Target_Accuracy': gbm_dominant
})
submission2.to_csv('submission_gbm_dominant.csv', index=False)
print("[OK] Saved submission_gbm_dominant.csv")

# Ultra-conservative: Only change most obvious cases
print("\n[ULTRA-CONSERVATIVE] Minimal changes only...")

ultra_conservative = best['Target'].copy()
ultra_changes = 0

for idx, row in best.iterrows():
    best_pred = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # Only change if:
    # 1. Best says free
    # 2. Both GBM and Cond predict same delay
    # 3. That delay is moderate or heavy (more obvious)
    
    if (best_pred == 'free flowing' and 
        gbm_pred == cond_pred and 
        gbm_pred in ['moderate delay', 'heavy delay']):
        ultra_conservative.iloc[idx] = gbm_pred
        ultra_changes += 1

dist3 = ultra_conservative.value_counts(normalize=True)
print(f"\nUltra-Conservative Distribution:")
print(f"  Free:     {dist3.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist3.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist3.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist3.get('heavy delay', 0)*100:.1f}%")
print(f"  Changes: {ultra_changes}")

submission3 = pd.DataFrame({
    'ID': best['ID'],
    'Target': ultra_conservative,
    'Target_Accuracy': ultra_conservative
})
submission3.to_csv('submission_ultra_conservative.csv', index=False)
print("[OK] Saved submission_ultra_conservative.csv")

print("\n" + "="*80)
print("RECOMMENDATION - 3 NEW STRATEGIES")
print("="*80)
print("\n1. submission_strategy_v2.csv")
print(f"   - {changes} changes, trust GBM for delay predictions")
print(f"   - Distribution: {dist.get('free flowing', 0)*100:.1f}% F")

print("\n2. submission_gbm_dominant.csv")
print(f"   - {changes2} changes, GBM-weighted (60%)")
print(f"   - Distribution: {dist2.get('free flowing', 0)*100:.1f}% F")

print("\n3. submission_ultra_conservative.csv (RECOMMENDED)")
print(f"   - {ultra_changes} changes, only obvious moderate/heavy cases")
print(f"   - Distribution: {dist3.get('free flowing', 0)*100:.1f}% F")
print(f"   - Most conservative approach")

print("\n[STRATEGY] Try ultra_conservative first - smallest risk")
print("="*80)
