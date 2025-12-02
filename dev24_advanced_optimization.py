"""
Dev 24: Advanced Optimization - Push Boundaries
================================================
Continue improving on successful V2 approach
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("DEV 24: ADVANCED OPTIMIZATION - PUSHING BOUNDARIES")
print("="*80)

# Load current successful submissions
v2 = pd.read_csv('submission_improved_v2.csv')  # Current best direction-aware
pure_gbm = pd.read_csv('submission_OPTION3_pure_gbm.csv')
cond = pd.read_csv('submission_conditional_calibrated.csv')
original = pd.read_csv('submission_OPTION1_original.csv')

print("\n[BASELINE] Current V2 (Direction-Aware):")
dist_v2 = v2['Target'].value_counts(normalize=True)
print(f"  Free: {dist_v2.get('free flowing', 0)*100:.1f}%, Light: {dist_v2.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_v2.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_v2.get('heavy delay', 0)*100:.1f}%")

# Parse all data
v2['SegmentID'] = v2['ID'].str.extract(r'(\d+)').astype(int)
v2['Direction'] = v2['ID'].str.extract(r'_(enter|exit)')[0]
v2['pred_gbm'] = pure_gbm['Target'].values
v2['pred_cond'] = cond['Target'].values
v2['pred_orig'] = original['Target'].values

# Strategy 1: Even more aggressive delay boosting
print("\n[STRATEGY 1] Ultra-Aggressive Delay Boost")

strategy_1 = []
for idx, row in v2.iterrows():
    current = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # If ANY model predicts delay (not free), boost it
    votes = Counter()
    votes[current] += 0.50  # Current V2 prediction
    votes[gbm_pred] += 0.30
    votes[cond_pred] += 0.20
    
    # Heavy bonus for delay predictions
    if current != 'free flowing':
        votes[current] += 0.20
    if cond_pred in ['moderate delay', 'heavy delay']:
        votes[cond_pred] += 0.15
    
    strategy_1.append(votes.most_common(1)[0][0])

dist_1 = pd.Series(strategy_1).value_counts(normalize=True)
changes_1 = sum([1 for i in range(len(strategy_1)) if strategy_1[i] != v2['Target'].values[i]])
print(f"  Changes: {changes_1}")
print(f"  Free: {dist_1.get('free flowing', 0)*100:.1f}%, Light: {dist_1.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_1.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_1.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({'ID': v2['ID'], 'Target': strategy_1, 'Target_Accuracy': strategy_1}).to_csv('submission_ultra_aggressive.csv', index=False)
print("  ✓ Saved submission_ultra_aggressive.csv")

# Strategy 2: Maximize diversity (reduce free even more)
print("\n[STRATEGY 2] Maximum Diversity")

strategy_2 = []
for idx, row in v2.iterrows():
    current = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    direction = row['Direction']
    
    # For enter: very aggressive on delays
    if direction == 'enter':
        if cond_pred != 'free flowing':
            strategy_2.append(cond_pred)
        elif gbm_pred != 'free flowing':
            strategy_2.append(gbm_pred)
        else:
            strategy_2.append(current)
    # For exit: more conservative
    else:
        if gbm_pred != 'free flowing' and cond_pred != 'free flowing':
            strategy_2.append(gbm_pred)
        else:
            strategy_2.append(current)

dist_2 = pd.Series(strategy_2).value_counts(normalize=True)
changes_2 = sum([1 for i in range(len(strategy_2)) if strategy_2[i] != v2['Target'].values[i]])
print(f"  Changes: {changes_2}")
print(f"  Free: {dist_2.get('free flowing', 0)*100:.1f}%, Light: {dist_2.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_2.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_2.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({'ID': v2['ID'], 'Target': strategy_2, 'Target_Accuracy': strategy_2}).to_csv('submission_max_diversity.csv', index=False)
print("  ✓ Saved submission_max_diversity.csv")

# Strategy 3: Fine-tune V2 with small adjustments
print("\n[STRATEGY 3] Fine-Tuned V2 (Small Adjustments)")

strategy_3 = []
for idx, row in v2.iterrows():
    current = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # Only change if there's strong evidence
    if current == 'free flowing':
        # If both GBM and Cond agree on same delay type
        if gbm_pred == cond_pred and gbm_pred != 'free flowing':
            strategy_3.append(gbm_pred)
        else:
            strategy_3.append(current)
    elif current == 'light delay':
        # If both suggest heavier delay, upgrade
        if gbm_pred in ['moderate delay', 'heavy delay'] and cond_pred in ['moderate delay', 'heavy delay']:
            strategy_3.append(gbm_pred)
        else:
            strategy_3.append(current)
    else:
        # Keep moderate/heavy as is
        strategy_3.append(current)

dist_3 = pd.Series(strategy_3).value_counts(normalize=True)
changes_3 = sum([1 for i in range(len(strategy_3)) if strategy_3[i] != v2['Target'].values[i]])
print(f"  Changes: {changes_3}")
print(f"  Free: {dist_3.get('free flowing', 0)*100:.1f}%, Light: {dist_3.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_3.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_3.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({'ID': v2['ID'], 'Target': strategy_3, 'Target_Accuracy': strategy_3}).to_csv('submission_fine_tuned.csv', index=False)
print("  ✓ Saved submission_fine_tuned.csv")

# Strategy 4: Weighted ensemble with bias toward delays
print("\n[STRATEGY 4] Delay-Biased Ensemble")

strategy_4 = []
for idx, row in v2.iterrows():
    current = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # Weight: Current 50%, GBM 30%, Cond 20%
    votes = Counter()
    votes[current] += 0.50
    votes[gbm_pred] += 0.30
    votes[cond_pred] += 0.20
    
    # Strong bias: if it's a delay, add extra weight
    for pred in [current, gbm_pred, cond_pred]:
        if pred == 'heavy delay':
            votes[pred] += 0.25
        elif pred == 'moderate delay':
            votes[pred] += 0.15
        elif pred == 'light delay':
            votes[pred] += 0.10
    
    strategy_4.append(votes.most_common(1)[0][0])

dist_4 = pd.Series(strategy_4).value_counts(normalize=True)
changes_4 = sum([1 for i in range(len(strategy_4)) if strategy_4[i] != v2['Target'].values[i]])
print(f"  Changes: {changes_4}")
print(f"  Free: {dist_4.get('free flowing', 0)*100:.1f}%, Light: {dist_4.get('light delay', 0)*100:.1f}%, "
      f"Moderate: {dist_4.get('moderate delay', 0)*100:.1f}%, Heavy: {dist_4.get('heavy delay', 0)*100:.1f}%")

pd.DataFrame({'ID': v2['ID'], 'Target': strategy_4, 'Target_Accuracy': strategy_4}).to_csv('submission_delay_biased.csv', index=False)
print("  ✓ Saved submission_delay_biased.csv")

# Load Strategy C from before (most aggressive)
strategy_c = pd.read_csv('submission_strategy_c.csv')
dist_c = strategy_c['Target'].value_counts(normalize=True)

print("\n" + "="*80)
print("COMPLETE STRATEGY COMPARISON")
print("="*80)

strategies = [
    ('Current V2', dist_v2, 0),
    ('Strategy 1: Ultra-Aggressive', dist_1, changes_1),
    ('Strategy 2: Max Diversity', dist_2, changes_2),
    ('Strategy 3: Fine-Tuned', dist_3, changes_3),
    ('Strategy 4: Delay-Biased', dist_4, changes_4),
    ('Strategy C (Previous)', dist_c, 65)
]

print(f"\n{'Strategy':<30} {'Changes':>8} {'Free':>7} {'Light':>7} {'Moderate':>7} {'Heavy':>7}")
print("-" * 75)
for name, dist, changes in strategies:
    print(f"{name:<30} {changes:>8} "
          f"{dist.get('free flowing', 0)*100:>6.1f}% "
          f"{dist.get('light delay', 0)*100:>6.1f}% "
          f"{dist.get('moderate delay', 0)*100:>6.1f}% "
          f"{dist.get('heavy delay', 0)*100:>6.1f}%")

print("\n🎯 RECOMMENDED PROGRESSION:")
print("\n1. submission_fine_tuned.csv (RECOMMENDED NEXT)")
print("   → Conservative, builds on V2 success")
print(f"   → Only {changes_3} changes, minimal risk")
print(f"   → {dist_3.get('free flowing', 0)*100:.1f}% free")

print("\n2. submission_delay_biased.csv")
print("   → Weighted approach with delay preference")
print(f"   → {changes_4} changes")
print(f"   → {dist_4.get('free flowing', 0)*100:.1f}% free")

print("\n3. submission_max_diversity.csv")
print("   → More aggressive, especially on Enter")
print(f"   → {changes_2} changes")
print(f"   → {dist_2.get('free flowing', 0)*100:.1f}% free")

print("\n4. submission_strategy_c.csv")
print("   → Most aggressive free reduction")
print(f"   → 65 changes from Pure GBM")
print(f"   → {dist_c.get('free flowing', 0)*100:.1f}% free")

print("\n💡 STRATEGY:")
print("   Test fine_tuned first → safest progression")
print("   If it works → Try delay_biased")
print("   If that works → Try max_diversity")
print("   If need even more → strategy_c")

print("\n🎯 Target: Continue improving toward 0.8013!")
print("="*80)
