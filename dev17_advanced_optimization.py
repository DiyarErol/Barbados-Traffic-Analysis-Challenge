"""
Dev 17: Advanced Submission Optimization
=========================================
Analyze submission agreements/disagreements and create
intelligent weighted ensemble with segment-level optimization
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score

print("="*80)
print("DEV 17: ADVANCED SUBMISSION OPTIMIZATION")
print("="*80)

# Load submissions
print("\n[LOAD] Loading submissions...")
subs = {
    'final': pd.read_csv('submission_final_ensemble.csv'),
    'cond': pd.read_csv('submission_conditional_calibrated.csv'),
    'gbm': pd.read_csv('submission_gbm_blend.csv'),
    'calibrated': pd.read_csv('submission_calibrated.csv')
}

# Analyze agreement patterns
print("\n[ANALYSIS] Agreement Analysis:")
print("-" * 80)

# Create agreement matrix
n_subs = len(subs)
sub_names = list(subs.keys())
predictions = {name: df['Target'].values for name, df in subs.items()}

# Calculate pairwise agreement
print("\nPairwise Agreement (Cohen's Kappa):")
for i, name1 in enumerate(sub_names):
    for j, name2 in enumerate(sub_names):
        if i < j:
            kappa = cohen_kappa_score(predictions[name1], predictions[name2])
            agreement = np.mean(predictions[name1] == predictions[name2]) * 100
            print(f"  {name1:12s} vs {name2:12s}: {agreement:5.1f}% agree, κ={kappa:.3f}")

# Analyze confidence patterns
print("\n[CONFIDENCE] Identifying high-confidence predictions...")

# For each segment, count how many models agree
n_samples = len(subs['final'])
consensus_level = np.zeros(n_samples)
final_predictions = []

for i in range(n_samples):
    votes = [predictions[name][i] for name in sub_names]
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)[0]
    consensus_level[i] = most_common[1] / len(votes)
    final_predictions.append(most_common[0])

print(f"  4/4 consensus: {np.sum(consensus_level == 1.0):.0f} segments ({np.mean(consensus_level == 1.0)*100:.1f}%)")
print(f"  3/4 consensus: {np.sum(consensus_level == 0.75):.0f} segments ({np.mean(consensus_level == 0.75)*100:.1f}%)")
print(f"  2/4 consensus: {np.sum(consensus_level == 0.5):.0f} segments ({np.mean(consensus_level == 0.5)*100:.1f}%)")

# Strategy 1: Confidence-weighted ensemble
print("\n[STRATEGY 1] Confidence-Weighted Ensemble")
print("-" * 80)

# For high consensus (3-4 models agree), use majority vote
# For low consensus (2 models agree), prefer GBM (typically best performer)
smart_predictions = []

for i in range(n_samples):
    votes = [predictions[name][i] for name in sub_names]
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)[0]
    
    if most_common[1] >= 3:  # 3 or 4 models agree
        smart_predictions.append(most_common[0])
    else:  # Tie or weak consensus - use GBM
        smart_predictions.append(predictions['gbm'][i])

dist = pd.Series(smart_predictions).value_counts(normalize=True)
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

# Save Strategy 1
strategy1 = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': smart_predictions,
    'Target_Accuracy': smart_predictions
})
strategy1.to_csv('submission_smart_consensus.csv', index=False)
print("[OK] Saved submission_smart_consensus.csv")

# Strategy 2: Segment-aware blending
print("\n[STRATEGY 2] Segment-Aware Blending")
print("-" * 80)

# Parse segment info from ID
df_analysis = subs['final'].copy()
df_analysis['SegmentID'] = df_analysis['ID'].str.extract(r'(\d+)').astype(int)
df_analysis['Direction'] = df_analysis['ID'].str.extract(r'_(enter|exit)')[0]

# Add predictions from all models
for name in sub_names:
    df_analysis[f'pred_{name}'] = predictions[name]

# For Enter: prefer models with better enter performance
# For Exit: prefer models with better exit performance
segment_smart = []

for idx, row in df_analysis.iterrows():
    direction = row['Direction']
    votes = [row[f'pred_{name}'] for name in sub_names]
    
    if direction == 'enter':
        # For Enter, weight: gbm=0.4, cond=0.3, final=0.2, calibrated=0.1
        vote_counts = Counter()
        vote_counts[row['pred_gbm']] += 0.4
        vote_counts[row['pred_cond']] += 0.3
        vote_counts[row['pred_final']] += 0.2
        vote_counts[row['pred_calibrated']] += 0.1
    else:  # exit
        # For Exit, weight: gbm=0.45, final=0.25, cond=0.2, calibrated=0.1
        vote_counts = Counter()
        vote_counts[row['pred_gbm']] += 0.45
        vote_counts[row['pred_final']] += 0.25
        vote_counts[row['pred_cond']] += 0.2
        vote_counts[row['pred_calibrated']] += 0.1
    
    segment_smart.append(vote_counts.most_common(1)[0][0])

dist = pd.Series(segment_smart).value_counts(normalize=True)
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

# Analyze by direction
enter_preds = [segment_smart[i] for i in range(0, len(segment_smart), 2)]
exit_preds = [segment_smart[i] for i in range(1, len(segment_smart), 2)]

enter_dist = pd.Series(enter_preds).value_counts(normalize=True)
exit_dist = pd.Series(exit_preds).value_counts(normalize=True)

print(f"\nEnter: F={enter_dist.get('free flowing', 0)*100:.1f}% "
      f"L={enter_dist.get('light delay', 0)*100:.1f}% "
      f"M={enter_dist.get('moderate delay', 0)*100:.1f}% "
      f"H={enter_dist.get('heavy delay', 0)*100:.1f}%")
print(f"Exit:  F={exit_dist.get('free flowing', 0)*100:.1f}% "
      f"L={exit_dist.get('light delay', 0)*100:.1f}% "
      f"M={exit_dist.get('moderate delay', 0)*100:.1f}% "
      f"H={exit_dist.get('heavy delay', 0)*100:.1f}%")

# Save Strategy 2
strategy2 = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': segment_smart,
    'Target_Accuracy': segment_smart
})
strategy2.to_csv('submission_segment_aware.csv', index=False)
print("[OK] Saved submission_segment_aware.csv")

# Strategy 3: Hybrid - Confidence + Segment-aware
print("\n[STRATEGY 3] Hybrid Approach (Confidence + Segment-aware)")
print("-" * 80)

hybrid_predictions = []

for i in range(n_samples):
    # If high consensus (3-4 agree), trust it
    if consensus_level[i] >= 0.75:
        hybrid_predictions.append(smart_predictions[i])
    else:
        # Otherwise use segment-aware prediction
        hybrid_predictions.append(segment_smart[i])

dist = pd.Series(hybrid_predictions).value_counts(normalize=True)
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

# Save Strategy 3
strategy3 = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': hybrid_predictions,
    'Target_Accuracy': hybrid_predictions
})
strategy3.to_csv('submission_hybrid_smart.csv', index=False)
print("[OK] Saved submission_hybrid_smart.csv")

# Strategy 4: Conservative-Optimized (slight adjustment to optimized)
print("\n[STRATEGY 4] Conservative-Optimized Blend")
print("-" * 80)

# Load the optimized blend we created earlier
optimized = pd.read_csv('submission_optimized_blend.csv')

# Apply slight calibration - reduce free flowing by 1-2% if too high
conservative_opt = []
for pred in optimized['Target'].values:
    # Keep prediction as is - already well-calibrated
    conservative_opt.append(pred)

# But we can create a version that's slightly more conservative
# by preferring GBM more in ties
refined_predictions = []

for i in range(n_samples):
    votes = {
        'final': predictions['final'][i],
        'cond': predictions['cond'][i],
        'gbm': predictions['gbm'][i]
    }
    
    # Weight votes: gbm=0.45, cond=0.35, final=0.20
    weighted_votes = Counter()
    weighted_votes[votes['gbm']] += 0.45
    weighted_votes[votes['cond']] += 0.35
    weighted_votes[votes['final']] += 0.20
    
    refined_predictions.append(weighted_votes.most_common(1)[0][0])

dist = pd.Series(refined_predictions).value_counts(normalize=True)
print(f"Weights: gbm=0.45, cond=0.35, final=0.20")
print(f"Distribution: F={dist.get('free flowing', 0)*100:.1f}% "
      f"L={dist.get('light delay', 0)*100:.1f}% "
      f"M={dist.get('moderate delay', 0)*100:.1f}% "
      f"H={dist.get('heavy delay', 0)*100:.1f}%")

# Save Strategy 4
strategy4 = pd.DataFrame({
    'ID': subs['final']['ID'],
    'Target': refined_predictions,
    'Target_Accuracy': refined_predictions
})
strategy4.to_csv('submission_refined_optimized.csv', index=False)
print("[OK] Saved submission_refined_optimized.csv")

print("\n" + "="*80)
print("SUMMARY - 4 NEW ADVANCED SUBMISSIONS")
print("="*80)
print("\n✓ Created 4 advanced optimized submissions:")
print("\n1. submission_smart_consensus.csv")
print("   - High consensus (3-4 agree): Use majority vote")
print("   - Low consensus (tie): Prefer GBM")
print(f"   - Distribution: Most balanced for confident predictions")

print("\n2. submission_segment_aware.csv")
print("   - Enter predictions: Weight GBM=0.40, Cond=0.30, Final=0.20")
print("   - Exit predictions: Weight GBM=0.45, Final=0.25, Cond=0.20")
print(f"   - Strategy: Direction-specific optimization")

print("\n3. submission_hybrid_smart.csv")
print("   - Combines consensus + segment-aware strategies")
print("   - Uses consensus when available, segment-aware for ties")
print(f"   - Best of both approaches")

print("\n4. submission_refined_optimized.csv")
print("   - Refined weights: GBM=0.45, Cond=0.35, Final=0.20")
print("   - More aggressive GBM weighting")
print(f"   - Conservative but performance-focused")

print("\n" + "="*80)
print("🎯 RECOMMENDATION")
print("="*80)
print("Test order:")
print("  1st: submission_hybrid_smart.csv - Best theoretical approach")
print("  2nd: submission_refined_optimized.csv - More GBM weight")
print("  3rd: submission_segment_aware.csv - Direction-specific")
print("  4th: submission_smart_consensus.csv - Consensus-based")
print("\n[TARGET] Hedef: 0.8013, Mevcut: 0.7708, Gap: +3.05%")
print("="*80)
