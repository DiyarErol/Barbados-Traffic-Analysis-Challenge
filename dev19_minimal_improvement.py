"""
Dev 19: Minimal Targeted Improvements on Best Submission
=========================================================
Take submission_final_ensemble (best: 0.7708) and apply
ONLY minimal, evidence-based improvements
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("DEV 19: MINIMAL IMPROVEMENTS ON BEST SUBMISSION")
print("="*80)

# Load the best submission
print("\n[LOAD] Loading best submission (final_ensemble - 0.7708)...")
best = pd.read_csv('submission_final_ensemble.csv')

# Load other good submissions for comparison
gbm = pd.read_csv('submission_gbm_blend.csv')
cond = pd.read_csv('submission_conditional_calibrated.csv')

print(f"  Best submission: {len(best)} predictions")

# Analyze current distribution
print("\n[ANALYSIS] Current Distribution (0.7708 scorer):")
dist = best['Target'].value_counts(normalize=True)
print(f"  Free:     {dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist.get('heavy delay', 0)*100:.1f}%")

# Parse segment info
best['SegmentID'] = best['ID'].str.extract(r'(\d+)').astype(int)
best['Direction'] = best['ID'].str.extract(r'_(enter|exit)')[0]

# Add other predictions for comparison
best['pred_gbm'] = gbm['Target'].values
best['pred_cond'] = cond['Target'].values

# Strategy: Find low-confidence predictions where GBM disagrees
print("\n[STRATEGY] Finding improvement opportunities...")

# Rule 1: Where GBM and Cond BOTH disagree with final, check if they agree
improvements = []

for idx, row in best.iterrows():
    final_pred = row['Target']
    gbm_pred = row['pred_gbm']
    cond_pred = row['pred_cond']
    
    # If GBM and Cond both agree but differ from final, consider switching
    if gbm_pred == cond_pred and gbm_pred != final_pred:
        # Additional check: only if final predicts "free flowing" and others don't
        if final_pred == 'free flowing' and gbm_pred != 'free flowing':
            improvements.append({
                'idx': idx,
                'id': row['ID'],
                'direction': row['Direction'],
                'old': final_pred,
                'new': gbm_pred,
                'reason': 'GBM+Cond agree, reduce free-bias'
            })

print(f"\n  Found {len(improvements)} improvement opportunities")

if len(improvements) > 0:
    # Show samples
    print("\n  Sample improvements:")
    for i, imp in enumerate(improvements[:10]):
        print(f"    {imp['id']:<50} {imp['direction']:5} {imp['old']:14} → {imp['new']:14}")
    
    # Analyze by direction
    enter_improvements = [x for x in improvements if x['direction'] == 'enter']
    exit_improvements = [x for x in improvements if x['direction'] == 'exit']
    print(f"\n  Enter improvements: {len(enter_improvements)}")
    print(f"  Exit improvements:  {len(exit_improvements)}")
    
    # Analyze by target class
    print("\n  Changes by class:")
    change_summary = pd.DataFrame(improvements).groupby(['old', 'new']).size().reset_index(name='count')
    print(change_summary.to_string(index=False))

# Create improved submission - VERY CONSERVATIVE
print("\n[CREATE] Creating minimally improved submission...")

improved_predictions = best['Target'].copy()

# Apply only the most confident improvements (where both GBM and Cond agree)
for imp in improvements:
    improved_predictions.iloc[imp['idx']] = imp['new']

# Analyze new distribution
new_dist = improved_predictions.value_counts(normalize=True)
print(f"\nNew Distribution:")
print(f"  Free:     {new_dist.get('free flowing', 0)*100:.1f}% (was {dist.get('free flowing', 0)*100:.1f}%)")
print(f"  Light:    {new_dist.get('light delay', 0)*100:.1f}% (was {dist.get('light delay', 0)*100:.1f}%)")
print(f"  Moderate: {new_dist.get('moderate delay', 0)*100:.1f}% (was {dist.get('moderate delay', 0)*100:.1f}%)")
print(f"  Heavy:    {new_dist.get('heavy delay', 0)*100:.1f}% (was {dist.get('heavy delay', 0)*100:.1f}%)")

change_count = (improved_predictions != best['Target']).sum()
print(f"\n  Total changes: {change_count} predictions ({change_count/len(best)*100:.1f}%)")

# Save improved submission
submission = pd.DataFrame({
    'ID': best['ID'],
    'Target': improved_predictions,
    'Target_Accuracy': improved_predictions
})

submission.to_csv('submission_improved_final.csv', index=False)
print("\n[OK] Saved submission_improved_final.csv")

# Create an even more conservative version (only 50% of improvements)
print("\n[BONUS] Creating ultra-conservative version...")

# Only apply improvements where the difference is most clear
conservative_predictions = best['Target'].copy()

# Sort improvements by confidence (prefer exit over enter, heavy/moderate over light)
confidence_scores = []
for imp in improvements:
    score = 0
    if imp['direction'] == 'exit':
        score += 2  # Exit predictions generally more reliable
    if imp['new'] in ['heavy delay', 'moderate delay']:
        score += 2  # More confident about significant delays
    elif imp['new'] == 'light delay':
        score += 1
    confidence_scores.append(score)

# Sort improvements by confidence
sorted_improvements = sorted(zip(improvements, confidence_scores), 
                            key=lambda x: x[1], reverse=True)

# Apply only top 50% most confident
n_changes = len(sorted_improvements) // 2
for imp, score in sorted_improvements[:n_changes]:
    conservative_predictions.iloc[imp['idx']] = imp['new']

cons_dist = conservative_predictions.value_counts(normalize=True)
print(f"\nUltra-Conservative Distribution:")
print(f"  Free:     {cons_dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {cons_dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {cons_dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {cons_dist.get('heavy delay', 0)*100:.1f}%")
print(f"  Changes: {n_changes} predictions")

conservative_submission = pd.DataFrame({
    'ID': best['ID'],
    'Target': conservative_predictions,
    'Target_Accuracy': conservative_predictions
})

conservative_submission.to_csv('submission_conservative_improved.csv', index=False)
print("[OK] Saved submission_conservative_improved.csv")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print("\n🎯 BEST CHOICE: submission_improved_final.csv")
print(f"   - Based on your best submission (0.7708)")
print(f"   - {change_count} minimal, evidence-based improvements")
print(f"   - Only changes where GBM+Cond both agree")
print(f"   - Reduces free-flowing bias slightly")
print(f"   - Distribution: {new_dist.get('free flowing', 0)*100:.1f}% F, "
      f"{new_dist.get('light delay', 0)*100:.1f}% L, "
      f"{new_dist.get('moderate delay', 0)*100:.1f}% M, "
      f"{new_dist.get('heavy delay', 0)*100:.1f}% H")

print("\n🛡️ BACKUP: submission_conservative_improved.csv")
print(f"   - Even more conservative ({n_changes} changes)")
print(f"   - Only highest-confidence improvements")
print(f"   - Distribution: {cons_dist.get('free flowing', 0)*100:.1f}% F, "
      f"{cons_dist.get('light delay', 0)*100:.1f}% L, "
      f"{cons_dist.get('moderate delay', 0)*100:.1f}% M, "
      f"{cons_dist.get('heavy delay', 0)*100:.1f}% H")

print("\n[STRATEGY] Start with submission_improved_final.csv")
print("[TARGET] Hedef: 0.8013, Mevcut: 0.7708, Gap: +3.05%")
print("[APPROACH] Minimal changes to proven winner")
print("="*80)
