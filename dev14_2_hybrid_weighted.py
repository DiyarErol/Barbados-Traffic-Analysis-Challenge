"""
Dev 14.2: Hybrid Ensemble with Multiple Strategies
===================================================
Combine multiple successful approaches:
1. GBM blend (best F1)
2. Conditional calibration (best distribution)
3. Fine-tuned ensemble weights
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DEV 14.2: ADVANCED HYBRID ENSEMBLE")
print("="*70)

# Load all three submissions
print("\n[LOAD] Reading existing submissions...")
sub_gbm = pd.read_csv('submission_gbm_blend.csv')
sub_cond = pd.read_csv('submission_conditional_calibrated.csv')
sub_final = pd.read_csv('submission_final_ensemble.csv')

print(f"  - GBM blend: {len(sub_gbm)} rows")
print(f"  - Conditional calibrated: {len(sub_cond)} rows")
print(f"  - Final ensemble: {len(sub_final)} rows")

# Analyze distributions
def get_distribution(df):
    """Get overall distribution from submission"""
    all_preds = pd.concat([df['EnterTrafficCondition'], df['ExitTrafficCondition']])
    return all_preds.value_counts(normalize=True).to_dict()

print("\n[ANALYSIS] Current submission distributions:")
for name, df in [('GBM', sub_gbm), ('Conditional', sub_cond), ('Final', sub_final)]:
    dist = get_distribution(df)
    print(f"\n{name}:")
    for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        print(f"  {cls}: {dist.get(cls, 0)*100:.1f}%")

# Load train for target distribution
train = pd.read_csv('Train.csv')
target_dist_enter = train['congestion_enter_rating'].value_counts(normalize=True)
target_dist_exit = train['congestion_exit_rating'].value_counts(normalize=True)

print("\n[TARGET] Train data distributions:")
print("Enter:")
for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
    print(f"  {cls}: {target_dist_enter.get(cls, 0)*100:.1f}%")
print("Exit:")
for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
    print(f"  {target_dist_exit.get(cls, 0)*100:.1f}%")

# Strategy: Weighted voting with strategic tie-breaking
print("\n[ENSEMBLE] Creating advanced hybrid ensemble...")

class_names = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']

# Assign weights based on validation performance and distribution quality
# GBM: best F1 (0.46)
# Conditional: best distribution match
# Final: balanced approach
weights = {
    'gbm': 0.40,      # Strong F1 performance
    'cond': 0.35,     # Excellent distribution
    'final': 0.25     # Balanced middle ground
}

print(f"  Weights: GBM={weights['gbm']}, Conditional={weights['cond']}, Final={weights['final']}")

# Weighted voting
def weighted_vote(preds_list, weights_list):
    """Weighted majority vote with tie-breaking"""
    vote_scores = defaultdict(float)
    for pred, weight in zip(preds_list, weights_list):
        vote_scores[pred] += weight
    return max(vote_scores, key=vote_scores.get)

enter_preds = []
exit_preds = []

for idx in range(len(sub_gbm)):
    # Enter predictions
    enter_votes = [
        sub_gbm.loc[idx, 'EnterTrafficCondition'],
        sub_cond.loc[idx, 'EnterTrafficCondition'],
        sub_final.loc[idx, 'EnterTrafficCondition']
    ]
    enter_weights = [weights['gbm'], weights['cond'], weights['final']]
    enter_pred = weighted_vote(enter_votes, enter_weights)
    enter_preds.append(enter_pred)
    
    # Exit predictions
    exit_votes = [
        sub_gbm.loc[idx, 'ExitTrafficCondition'],
        sub_cond.loc[idx, 'ExitTrafficCondition'],
        sub_final.loc[idx, 'ExitTrafficCondition']
    ]
    exit_weights = [weights['gbm'], weights['cond'], weights['final']]
    exit_pred = weighted_vote(exit_votes, exit_weights)
    exit_preds.append(exit_pred)

# Create hybrid submission
hybrid = pd.DataFrame({
    'SegmentID': sub_gbm['SegmentID'],
    'EnterTrafficCondition': enter_preds,
    'ExitTrafficCondition': exit_preds
})

# Check distribution and apply soft adjustment if needed
hybrid_dist = get_distribution(hybrid)
print("\n[INITIAL] Hybrid ensemble distribution:")
for cls in class_names:
    print(f"  {cls}: {hybrid_dist.get(cls, 0)*100:.1f}%")

# Soft distribution adjustment - move some predictions to underrepresented classes
# if deviation is too large
target_overall = pd.concat([target_dist_enter, target_dist_exit]).groupby(level=0).mean()

print("\n[ADJUST] Applying soft distribution calibration...")

# Identify predictions where models disagree and confidence might be low
disagreements = []
for idx in range(len(sub_gbm)):
    enter_unique = len(set([
        sub_gbm.loc[idx, 'EnterTrafficCondition'],
        sub_cond.loc[idx, 'EnterTrafficCondition'],
        sub_final.loc[idx, 'EnterTrafficCondition']
    ]))
    exit_unique = len(set([
        sub_gbm.loc[idx, 'ExitTrafficCondition'],
        sub_cond.loc[idx, 'ExitTrafficCondition'],
        sub_final.loc[idx, 'ExitTrafficCondition']
    ]))
    
    if enter_unique >= 2 or exit_unique >= 2:
        disagreements.append(idx)

print(f"  Found {len(disagreements)} samples with model disagreement")

# For disagreement cases, favor conditional calibration (better distribution)
for idx in disagreements:
    # 60% chance to use conditional prediction for better distribution
    if np.random.random() < 0.6:
        hybrid.loc[idx, 'EnterTrafficCondition'] = sub_cond.loc[idx, 'EnterTrafficCondition']
        hybrid.loc[idx, 'ExitTrafficCondition'] = sub_cond.loc[idx, 'ExitTrafficCondition']

# Final distribution
final_dist = get_distribution(hybrid)
print("\n[FINAL] Adjusted hybrid distribution:")
for cls in class_names:
    print(f"  {cls}: {final_dist.get(cls, 0)*100:.1f}%")

# Save submission
hybrid.to_csv('submission_hybrid_weighted.csv', index=False)
print("\n[OK] Saved submission_hybrid_weighted.csv")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Combined 3 best submissions with strategic weighting")
print("✓ Applied weighted voting (GBM:0.40, Cond:0.35, Final:0.25)")
print("✓ Leveraged model disagreement for distribution adjustment")
print("✓ Expected performance: Between best F1 and best distribution")
print("\n[RECOMMENDATION] Try submission_hybrid_weighted.csv")
print("This combines:")
print("  - GBM's strong classification performance (F1≈0.46)")
print("  - Conditional calibration's excellent distribution match")
print("  - Final ensemble's balanced approach")
print("  - Smart handling of prediction uncertainty")
