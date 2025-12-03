"""
Dev 21: Multiple Final Options
================================
Create 3 very different approaches for user to choose
"""

import pandas as pd

print("="*80)
print("DEV 21: CREATING MULTIPLE FINAL OPTIONS")
print("="*80)

# Load originals
best = pd.read_csv('submission_final_ensemble.csv')  # 0.7708 - the winner
gbm = pd.read_csv('submission_gbm_blend.csv')

print("\n[OPTION 1] Keep Original Best (0.7708)")
print("   - NO changes at all")
print("   - Maybe your original was already optimal")

best.to_csv('submission_OPTION1_original.csv', index=False)
print("   ✓ Saved as submission_OPTION1_original.csv")

orig_dist = best['Target'].value_counts(normalize=True)
print(f"   Distribution: F={orig_dist.get('free flowing', 0)*100:.1f}%, "
      f"L={orig_dist.get('light delay', 0)*100:.1f}%, "
      f"M={orig_dist.get('moderate delay', 0)*100:.1f}%, "
      f"H={orig_dist.get('heavy delay', 0)*100:.1f}%")

print("\n[OPTION 2] GBM-Dominant (Aggressive)")
print("   - 237 changes (13.5%)")
print("   - Completely different distribution")
print("   - High risk, high reward")

# Already created as submission_gbm_dominant.csv
gbm_dom = pd.read_csv('submission_gbm_dominant.csv')
gbm_dom.to_csv('submission_OPTION2_aggressive.csv', index=False)
print("   ✓ Saved as submission_OPTION2_aggressive.csv")

gbm_dist = gbm_dom['Target'].value_counts(normalize=True)
print(f"   Distribution: F={gbm_dist.get('free flowing', 0)*100:.1f}%, "
      f"L={gbm_dist.get('light delay', 0)*100:.1f}%, "
      f"M={gbm_dist.get('moderate delay', 0)*100:.1f}%, "
      f"H={gbm_dist.get('heavy delay', 0)*100:.1f}%")

print("\n[OPTION 3] Pure GBM")
print("   - Use GBM predictions directly")
print("   - GBM was the best single model")
print("   - Maximum difference from original")

gbm.to_csv('submission_OPTION3_pure_gbm.csv', index=False)
print("   ✓ Saved as submission_OPTION3_pure_gbm.csv")

pure_dist = gbm['Target'].value_counts(normalize=True)
print(f"   Distribution: F={pure_dist.get('free flowing', 0)*100:.1f}%, "
      f"L={pure_dist.get('light delay', 0)*100:.1f}%, "
      f"M={pure_dist.get('moderate delay', 0)*100:.1f}%, "
      f"H={pure_dist.get('heavy delay', 0)*100:.1f}%")

print("\n" + "="*80)
print("SUMMARY - CHOOSE YOUR APPROACH")
print("="*80)

print("\n📊 DISTRIBUTION COMPARISON:")
print(f"{'Strategy':<25} {'Free':>7} {'Light':>7} {'Moderate':>7} {'Heavy':>7}")
print("-" * 60)
print(f"{'Option 1: Original (0.7708)':<25} {orig_dist.get('free flowing', 0)*100:>6.1f}% "
      f"{orig_dist.get('light delay', 0)*100:>6.1f}% "
      f"{orig_dist.get('moderate delay', 0)*100:>6.1f}% "
      f"{orig_dist.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'Option 2: GBM-Dominant':<25} {gbm_dist.get('free flowing', 0)*100:>6.1f}% "
      f"{gbm_dist.get('light delay', 0)*100:>6.1f}% "
      f"{gbm_dist.get('moderate delay', 0)*100:>6.1f}% "
      f"{gbm_dist.get('heavy delay', 0)*100:>6.1f}%")
print(f"{'Option 3: Pure GBM':<25} {pure_dist.get('free flowing', 0)*100:>6.1f}% "
      f"{pure_dist.get('light delay', 0)*100:>6.1f}% "
      f"{pure_dist.get('moderate delay', 0)*100:>6.1f}% "
      f"{pure_dist.get('heavy delay', 0)*100:>6.1f}%")

print("\n💡 RECOMMENDATIONS:")
print("\n1. If you want SAFETY: Try Option 1 (Original)")
print("   → Maybe 0.7708 is already very good")
print("   → Test set might be similar to train")

print("\n2. If you want BALANCED RISK: Try Option 2 (GBM-Dominant)")
print("   → Weighted approach with GBM emphasis")
print("   → 69.1% free (10% less than original)")

print("\n3. If you want MAXIMUM DIFFERENT: Try Option 3 (Pure GBM)")
print("   → Pure GBM predictions")
print("   → If test set is very different, this might win")

print("\n🎯 MY SUGGESTION: Try Option 3 (Pure GBM) first")
print("   → Most different from what you tried")
print("   → GBM was your best single model")
print("   → Conservative approaches keep failing")

print("\n" + "="*80)
