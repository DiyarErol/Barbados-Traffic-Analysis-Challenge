"""
Dev 16: Optimize Best Performing Submissions
=============================================
Take our 3 best submissions and find optimal blending weights
to maximize performance toward 0.8013 target
"""

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from collections import defaultdict

print("="*80)
print("DEV 16: OPTIMAL SUBMISSION BLENDING")
print("="*80)

# Load all best submissions
print("\n[LOAD] Loading best submissions...")
subs = {
    'final': pd.read_csv('submission_final_ensemble.csv'),
    'cond': pd.read_csv('submission_conditional_calibrated.csv'),
    'gbm': pd.read_csv('submission_gbm_blend.csv'),
    'calibrated': pd.read_csv('submission_calibrated.csv'),
    'strategic': pd.read_csv('submission_strategic.csv'),
    'stacking': pd.read_csv('submission_stacking.csv')
}

for name, df in subs.items():
    print(f"  {name}: {len(df)} rows")

# Analyze distributions
print("\n[ANALYSIS] Distributions:")
for name, df in subs.items():
    dist = df['Target'].value_counts(normalize=True)
    free = dist.get('free flowing', 0) * 100
    light = dist.get('light delay', 0) * 100
    moderate = dist.get('moderate delay', 0) * 100
    heavy = dist.get('heavy delay', 0) * 100
    print(f"  {name:12s}: F={free:5.1f}% L={light:5.1f}% M={moderate:5.1f}% H={heavy:5.1f}%")

# Try different blending strategies
def majority_vote_with_weights(predictions_list, weights):
    """Weighted majority vote"""
    n_samples = len(predictions_list[0])
    final_preds = []
    
    for i in range(n_samples):
        votes = defaultdict(float)
        for preds, weight in zip(predictions_list, weights):
            votes[preds[i]] += weight
        final_preds.append(max(votes, key=votes.get))
    
    return final_preds

print("\n[BLEND] Creating optimally weighted ensembles...")

# Strategy 1: Equal weights
print("\n--- Equal Weights (baseline) ---")
equal_weights = [1/6] * 6
predictions = [df['Target'].values for df in subs.values()]
equal_blend = majority_vote_with_weights(predictions, equal_weights)

dist = pd.Series(equal_blend).value_counts(normalize=True)
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

# Strategy 2: Top 3 only (final, cond, gbm)
print("\n--- Top 3 (final, cond, gbm) with tuned weights ---")
top3_predictions = [
    subs['final']['Target'].values,
    subs['cond']['Target'].values,
    subs['gbm']['Target'].values
]

# Try different weight combinations
best_weights = None
best_diversity = 0

for w1 in np.arange(0.2, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue
        
        weights = [w1, w2, w3]
        blend = majority_vote_with_weights(top3_predictions, weights)
        dist = pd.Series(blend).value_counts(normalize=True)
        
        # Diversity score (we want balanced distribution but not too extreme)
        # Target roughly: 70-80% free, 6-10% light, 8-12% moderate, 6-10% heavy
        free_score = 1.0 - abs(dist.get('free flowing', 0) - 0.75)
        light_score = 1.0 - abs(dist.get('light delay', 0) - 0.08)
        moderate_score = 1.0 - abs(dist.get('moderate delay', 0) - 0.10)
        heavy_score = 1.0 - abs(dist.get('heavy delay', 0) - 0.07)
        
        diversity = (free_score + light_score + moderate_score + heavy_score) / 4
        
        if diversity > best_diversity:
            best_diversity = diversity
            best_weights = weights
            best_blend = blend
            best_dist = dist

print(f"Best weights: final={best_weights[0]:.2f}, cond={best_weights[1]:.2f}, gbm={best_weights[2]:.2f}")
print(f"Diversity score: {best_diversity:.4f}")
print(f"Distribution: F={best_dist.get('free flowing', 0)*100:.1f}% "
      f"L={best_dist.get('light delay', 0)*100:.1f}% "
      f"M={best_dist.get('moderate delay', 0)*100:.1f}% "
      f"H={best_dist.get('heavy delay', 0)*100:.1f}%")

# Create optimized submission
optimized = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': best_blend,
    'Target_Accuracy': best_blend
})

optimized.to_csv('submission_optimized_blend.csv', index=False)
print("\n[OK] Saved submission_optimized_blend.csv")

# Strategy 3: Conservative blend (favor historical best performer)
print("\n--- Conservative (weighted toward best historical) ---")
# If we knew which was 0.7708, we'd weight it heavily
# For now, assume 'final' was best, give it 50% weight
conservative_weights = [0.50, 0.30, 0.20]  # final, cond, gbm
conservative_blend = majority_vote_with_weights(top3_predictions, conservative_weights)

dist = pd.Series(conservative_blend).value_counts(normalize=True)
print(f"Weights: final=0.50, cond=0.30, gbm=0.20")
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

conservative = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': conservative_blend,
    'Target_Accuracy': conservative_blend
})

conservative.to_csv('submission_conservative_blend.csv', index=False)
print("[OK] Saved submission_conservative_blend.csv")

# Strategy 4: Aggressive blend (favor gbm for performance)
print("\n--- Aggressive (weighted toward GBM performance) ---")
aggressive_weights = [0.30, 0.25, 0.45]  # final, cond, gbm
aggressive_blend = majority_vote_with_weights(top3_predictions, aggressive_weights)

dist = pd.Series(aggressive_blend).value_counts(normalize=True)
print(f"Weights: final=0.30, cond=0.25, gbm=0.45")
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

aggressive = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': aggressive_blend,
    'Target_Accuracy': aggressive_blend
})

aggressive.to_csv('submission_aggressive_blend.csv', index=False)
print("[OK] Saved submission_aggressive_blend.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✓ Created 3 new optimized blends:")
print("  1. submission_optimized_blend.csv - Diversity-optimized weights")
print("  2. submission_conservative_blend.csv - Favor final ensemble (if it was 0.7708)")
print("  3. submission_aggressive_blend.csv - Favor GBM performance")
print(f"\n[STRATEGY] Test these in order:")
print(f"  1st: Conservative (if final was 0.7708)")
print(f"  2nd: Optimized (best diversity)")
print(f"  3rd: Aggressive (performance-focused)")
print(f"\n[TARGET] Hedef: 0.8013, Mevcut: 0.7708, Gap: +3.05%")
