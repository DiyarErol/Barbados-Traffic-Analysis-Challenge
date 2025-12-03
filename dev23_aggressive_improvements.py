"""
Dev 23: Aggressive Improvements on Success
===========================================
Since Pure GBM worked, let's boost delay classes more
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("DEV 23: AGGRESSIVE DELAY BOOSTING")
print("="*80)

# Load successful GBM
pure_gbm = pd.read_csv('submission_FINAL.csv')
cond = pd.read_csv('submission_conditional_calibrated.csv')
original = pd.read_csv('submission_OPTION1_original.csv')

print("\n[BASE] Pure GBM (Current Success):")
dist = pure_gbm['Target'].value_counts(normalize=True)
print(f"  Free: {dist.get('free flowing', 0)*100:.1f}%, Light: {dist.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist.get('moderate delay', 0)*100:.1f}%, Heavy: {dist.get('heavy delay', 0)*100:.1f}%")

# Parse
pure_gbm['SegmentID'] = pure_gbm['ID'].str.extract(r'(\d+)').astype(int)
pure_gbm['Direction'] = pure_gbm['ID'].str.extract(r'_(enter|exit)')[0]
pure_gbm['pred_cond'] = cond['Target'].values
pure_gbm['pred_orig'] = original['Target'].values

# Strategy A: More delay weight
print("\n[STRATEGY A] Delay-Weighted (GBM 60%, Cond 40%)")

strategy_a = []
for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    
    votes = Counter()
    votes[gbm_pred] += 0.60
    votes[cond_pred] += 0.40
    
    # Strong bonus for delay predictions
    if gbm_pred != 'free flowing':
        votes[gbm_pred] += 0.15
    if cond_pred != 'free flowing':
        votes[cond_pred] += 0.10
    
    strategy_a.append(votes.most_common(1)[0][0])

dist_a = pd.Series(strategy_a).value_counts(normalize=True)
changes_a = sum([1 for i in range(len(strategy_a)) if strategy_a[i] != pure_gbm['Target'].values[i]])

print(f"  Changes: {changes_a}")
print(f"  Free: {dist_a.get('free flowing', 0)*100:.1f}%, Light: {dist_a.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_a.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_a.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': strategy_a,
    'Target_Accuracy': strategy_a
}).to_csv('submission_strategy_a.csv', index=False)
print("  ✓ Saved submission_strategy_a.csv")

# Strategy B: Favor Cond for heavy delays
print("\n[STRATEGY B] Heavy Delay Boost")

strategy_b = []
for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    
    # If Cond predicts heavy, trust it
    if cond_pred == 'heavy delay':
        strategy_b.append(cond_pred)
    # If both predict delay (any), use GBM
    elif gbm_pred != 'free flowing' and cond_pred != 'free flowing':
        strategy_b.append(gbm_pred)
    # Otherwise keep GBM
    else:
        strategy_b.append(gbm_pred)

dist_b = pd.Series(strategy_b).value_counts(normalize=True)
changes_b = sum([1 for i in range(len(strategy_b)) if strategy_b[i] != pure_gbm['Target'].values[i]])

print(f"  Changes: {changes_b}")
print(f"  Free: {dist_b.get('free flowing', 0)*100:.1f}%, Light: {dist_b.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_b.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_b.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': strategy_b,
    'Target_Accuracy': strategy_b
}).to_csv('submission_strategy_b.csv', index=False)
print("  ✓ Saved submission_strategy_b.csv")

# Strategy C: Reduce free more aggressively
print("\n[STRATEGY C] Aggressive Free Reduction")

strategy_c = []
for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    orig_pred = row['pred_orig']
    
    # If GBM says free but ANY other says delay, switch
    if gbm_pred == 'free flowing' and cond_pred != 'free flowing':
        strategy_c.append(cond_pred)
    else:
        strategy_c.append(gbm_pred)

dist_c = pd.Series(strategy_c).value_counts(normalize=True)
changes_c = sum([1 for i in range(len(strategy_c)) if strategy_c[i] != pure_gbm['Target'].values[i]])

print(f"  Changes: {changes_c}")
print(f"  Free: {dist_c.get('free flowing', 0)*100:.1f}%, Light: {dist_c.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_c.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_c.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': strategy_c,
    'Target_Accuracy': strategy_c
}).to_csv('submission_strategy_c.csv', index=False)
print("  ✓ Saved submission_strategy_c.csv")

# Strategy D: Blend of improved_v2 (which had good changes)
print("\n[STRATEGY D] Use Direction-Aware (V2)")
print(f"  Already created: submission_improved_v2.csv")
print(f"  Free: 67.1%, Light: 14.1%, Moderate: 11.3%, Heavy: 7.5%")
print(f"  36 changes - more aggressive delay boost")

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

print(f"\n{'Strategy':<25} {'Changes':>8} {'Free':>7} {'Light':>7} {'Moderate':>7} {'Heavy':>7}")
print("-" * 70)
print(f"{'Current (Pure GBM)':<25} {0:>8} {dist.get('free flowing', 0)*100:>6.1f}% {dist.get('light delay', 0)*100:>6.1f}% {dist.get('moderate delay', 0)*100:>6.1f}% {dist.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'A: Delay-Weighted':<25} {changes_a:>8} {dist_a.get('free flowing', 0)*100:>6.1f}% {dist_a.get('light delay', 0)*100:>6.1f}% {dist_a.get('moderate delay', 0)*100:>6.1f}% {dist_a.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'B: Heavy Boost':<25} {changes_b:>8} {dist_b.get('free flowing', 0)*100:>6.1f}% {dist_b.get('light delay', 0)*100:>6.1f}% {dist_b.get('moderate delay', 0)*100:>6.1f}% {dist_b.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'C: Free Reduction':<25} {changes_c:>8} {dist_c.get('free flowing', 0)*100:>6.1f}% {dist_c.get('light delay', 0)*100:>6.1f}% {dist_c.get('moderate delay', 0)*100:>6.1f}% {dist_c.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'D: Direction-Aware (V2)':<25} {36:>8} {67.1:>6.1f}% {14.1:>6.1f}% {11.3:>6.1f}% {7.5:>6.1f}%")

print("\n🎯 RECOMMENDED TEST ORDER:")
print("1. submission_improved_v2.csv (Direction-Aware, 36 changes)")
print("   → Best balance, direction-specific optimization")
print("\n2. submission_strategy_b.csv (Heavy Boost)")
print("   → Boosts heavy delays specifically")
print("\n3. submission_strategy_c.csv (Free Reduction)")
print("   → Most aggressive free reduction")

print("\n💡 All strategies maintain GBM's successful approach")
print("   but make targeted improvements!")
print("="*80)
