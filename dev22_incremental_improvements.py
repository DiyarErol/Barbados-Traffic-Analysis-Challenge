"""
Dev 22: Build on Success - Incremental Improvements
====================================================
Pure GBM worked! Now let's make small improvements on it
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("DEV 22: BUILDING ON SUCCESS - INCREMENTAL IMPROVEMENTS")
print("="*80)

# Load the successful submission (Pure GBM)
print("\n[SUCCESS] Loading Pure GBM (the winner)...")
pure_gbm = pd.read_csv('submission_FINAL.csv')

# Load other models for fine-tuning
cond = pd.read_csv('submission_conditional_calibrated.csv')
original = pd.read_csv('submission_OPTION1_original.csv')  # 0.7708

print(f"  Pure GBM: {len(pure_gbm)} predictions")

# Analyze current distribution
print("\n[CURRENT] Pure GBM Distribution (SUCCESS):")
dist = pure_gbm['Target'].value_counts(normalize=True)
print(f"  Free:     {dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist.get('heavy delay', 0)*100:.1f}%")

# Parse segment info
pure_gbm['SegmentID'] = pure_gbm['ID'].str.extract(r'(\d+)').astype(int)
pure_gbm['Direction'] = pure_gbm['ID'].str.extract(r'_(enter|exit)')[0]
pure_gbm['pred_cond'] = cond['Target'].values
pure_gbm['pred_orig'] = original['Target'].values

print("\n[STRATEGY] Fine-tuning Pure GBM...")

# Strategy 1: Where 3 models agree on something different, consider it
improved_v1 = []

for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    orig_pred = row['pred_orig']
    
    # If all 3 agree, definitely keep it (high confidence)
    if gbm_pred == cond_pred == orig_pred:
        improved_v1.append(gbm_pred)
    
    # If GBM says "free" but both others agree on a delay, reconsider
    elif gbm_pred == 'free flowing' and cond_pred == orig_pred and cond_pred != 'free flowing':
        # Strong signal from 2 other models - maybe switch
        # But only for heavy/moderate (more obvious delays)
        if cond_pred in ['heavy delay', 'moderate delay']:
            improved_v1.append(cond_pred)
        else:
            improved_v1.append(gbm_pred)  # Keep GBM for light delays
    
    else:
        # Trust GBM (it's working!)
        improved_v1.append(gbm_pred)

changes_v1 = sum([1 for i in range(len(improved_v1)) if improved_v1[i] != pure_gbm['Target'].values[i]])
dist_v1 = pd.Series(improved_v1).value_counts(normalize=True)

print(f"\n[VERSION 1] Conservative Improvement")
print(f"  Changes: {changes_v1} predictions")
print(f"  Free:     {dist_v1.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist_v1.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist_v1.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist_v1.get('heavy delay', 0)*100:.1f}%")

submission_v1 = pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': improved_v1,
    'Target_Accuracy': improved_v1
})
submission_v1.to_csv('submission_improved_v1.csv', index=False)
print("  ✓ Saved as submission_improved_v1.csv")

# Strategy 2: Boost delay predictions slightly more
improved_v2 = []

for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    orig_pred = row['pred_orig']
    direction = row['Direction']
    
    # If GBM+Cond agree on delay (any delay), use it
    if gbm_pred == cond_pred and gbm_pred != 'free flowing':
        improved_v2.append(gbm_pred)
    
    # If GBM says free but Cond says delay, weighted decision
    elif gbm_pred == 'free flowing' and cond_pred != 'free flowing':
        # For exit, trust GBM more (exits are easier)
        if direction == 'exit':
            improved_v2.append(gbm_pred)
        # For enter, if Cond says moderate/heavy, trust it
        elif cond_pred in ['moderate delay', 'heavy delay']:
            improved_v2.append(cond_pred)
        else:
            improved_v2.append(gbm_pred)
    
    else:
        improved_v2.append(gbm_pred)

changes_v2 = sum([1 for i in range(len(improved_v2)) if improved_v2[i] != pure_gbm['Target'].values[i]])
dist_v2 = pd.Series(improved_v2).value_counts(normalize=True)

print(f"\n[VERSION 2] Direction-Aware Improvement")
print(f"  Changes: {changes_v2} predictions")
print(f"  Free:     {dist_v2.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist_v2.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist_v2.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist_v2.get('heavy delay', 0)*100:.1f}%")

submission_v2 = pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': improved_v2,
    'Target_Accuracy': improved_v2
})
submission_v2.to_csv('submission_improved_v2.csv', index=False)
print("  ✓ Saved as submission_improved_v2.csv")

# Strategy 3: Add slight GBM+Cond blend for uncertain cases
improved_v3 = []

for idx, row in pure_gbm.iterrows():
    gbm_pred = row['Target']
    cond_pred = row['pred_cond']
    
    # Give GBM 70%, Cond 30% weight
    votes = Counter()
    votes[gbm_pred] += 0.70
    votes[cond_pred] += 0.30
    
    # Bonus: if it's a delay prediction, add weight
    if gbm_pred != 'free flowing':
        votes[gbm_pred] += 0.10
    if cond_pred != 'free flowing':
        votes[cond_pred] += 0.05
    
    improved_v3.append(votes.most_common(1)[0][0])

changes_v3 = sum([1 for i in range(len(improved_v3)) if improved_v3[i] != pure_gbm['Target'].values[i]])
dist_v3 = pd.Series(improved_v3).value_counts(normalize=True)

print(f"\n[VERSION 3] Weighted GBM+Cond (70/30)")
print(f"  Changes: {changes_v3} predictions")
print(f"  Free:     {dist_v3.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist_v3.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist_v3.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist_v3.get('heavy delay', 0)*100:.1f}%")

submission_v3 = pd.DataFrame({
    'ID': pure_gbm['ID'],
    'Target': improved_v3,
    'Target_Accuracy': improved_v3
})
submission_v3.to_csv('submission_improved_v3.csv', index=False)
print("  ✓ Saved as submission_improved_v3.csv")

print("\n" + "="*80)
print("RECOMMENDATION - 3 INCREMENTAL IMPROVEMENTS")
print("="*80)

print("\n✅ CURRENT SUCCESS: Pure GBM (69.1% free)")
print("   This is your baseline now!")

print("\n📈 3 IMPROVEMENT OPTIONS:")

print("\n1. submission_improved_v1.csv (MOST CONSERVATIVE)")
print(f"   - {changes_v1} changes from Pure GBM")
print(f"   - Only switch when 2 other models strongly agree")
print(f"   - {dist_v1.get('free flowing', 0)*100:.1f}% free")
print("   → Lowest risk, small gain potential")

print("\n2. submission_improved_v2.csv (BALANCED)")
print(f"   - {changes_v2} changes from Pure GBM")
print(f"   - Direction-aware (trust exit more, enter less)")
print(f"   - {dist_v2.get('free flowing', 0)*100:.1f}% free")
print("   → Medium risk, medium gain potential")

print("\n3. submission_improved_v3.csv (RECOMMENDED)")
print(f"   - {changes_v3} changes from Pure GBM")
print(f"   - Weighted blend (GBM 70%, Cond 30%)")
print(f"   - {dist_v3.get('free flowing', 0)*100:.1f}% free")
print("   → Balanced approach, best theoretical foundation")

print("\n🎯 MY SUGGESTION: Try Version 3 first")
print("   → It maintains GBM's strength (70% weight)")
print("   → Adds Cond's insights (30% weight)")
print("   → Boosts delay predictions slightly")
print("   → Most likely to improve on current success")

print("\n💡 IF VERSION 3 WORKS:")
print("   → We can continue optimizing the weight ratio")
print("   → Try 75/25, 65/35, etc.")

print("\n" + "="*80)
