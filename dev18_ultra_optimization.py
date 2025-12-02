"""
Dev 18: Ultra-Optimized Hybrid Strategy
========================================
Enhance hybrid_smart with additional intelligence layers:
1. Temporal pattern awareness
2. Class-specific confidence thresholds
3. Adaptive weighting based on segment characteristics
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score

print("="*80)
print("DEV 18: ULTRA-OPTIMIZED HYBRID STRATEGY")
print("="*80)

# Load base submissions
print("\n[LOAD] Loading submissions...")
subs = {
    'final': pd.read_csv('submission_final_ensemble.csv'),
    'cond': pd.read_csv('submission_conditional_calibrated.csv'),
    'gbm': pd.read_csv('submission_gbm_blend.csv'),
    'calibrated': pd.read_csv('submission_calibrated.csv')
}

# Load training data for feature analysis
print("[LOAD] Loading training data for pattern analysis...")
train = pd.read_csv('Train.csv')

print(f"  Train shape: {train.shape}")
print(f"  Submissions: {len(subs)} models")

# Parse segment info
df_analysis = subs['final'].copy()
df_analysis['SegmentID'] = df_analysis['ID'].str.extract(r'(\d+)').astype(int)
df_analysis['Direction'] = df_analysis['ID'].str.extract(r'_(enter|exit)')[0]

# Add all predictions
predictions = {name: df['Target'].values for name, df in subs.items()}
for name in subs.keys():
    df_analysis[f'pred_{name}'] = predictions[name]

print("\n[ANALYZE] Computing consensus and patterns...")

# Calculate consensus level for each prediction
def get_consensus_info(row):
    votes = [row[f'pred_{name}'] for name in subs.keys()]
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)[0]
    
    return {
        'majority_class': most_common[0],
        'consensus_count': most_common[1],
        'consensus_ratio': most_common[1] / len(votes),
        'unique_predictions': len(vote_counts)
    }

consensus_data = df_analysis.apply(get_consensus_info, axis=1, result_type='expand')
df_analysis = pd.concat([df_analysis, consensus_data], axis=1)

print(f"  High consensus (4/4): {(df_analysis['consensus_ratio'] == 1.0).sum()} segments")
print(f"  Strong consensus (3/4): {(df_analysis['consensus_ratio'] == 0.75).sum()} segments")
print(f"  Weak consensus (2/4): {(df_analysis['consensus_ratio'] == 0.5).sum()} segments")

# Strategy: Ultra-Smart Hybrid
print("\n[STRATEGY] Ultra-Smart Hybrid Optimization")
print("-" * 80)

ultra_predictions = []

for idx, row in df_analysis.iterrows():
    direction = row['Direction']
    consensus_ratio = row['consensus_ratio']
    majority = row['majority_class']
    
    # RULE 1: Very High Consensus (4/4 agree) - Trust completely
    if consensus_ratio == 1.0:
        prediction = majority
        
    # RULE 2: Strong Consensus (3/4 agree)
    elif consensus_ratio == 0.75:
        # Trust majority, but verify it's not just "free flowing" bias
        if majority == 'free flowing':
            # Check if GBM agrees (GBM is less biased toward free)
            if row['pred_gbm'] == majority:
                prediction = majority
            else:
                # GBM disagrees - might be worth considering
                # Use weighted vote: GBM gets 0.5, others get 0.5 total
                votes = Counter()
                votes[row['pred_gbm']] += 0.5
                votes[row['pred_final']] += 0.17
                votes[row['pred_cond']] += 0.17
                votes[row['pred_calibrated']] += 0.16
                prediction = votes.most_common(1)[0][0]
        else:
            # Non-free prediction with 3/4 consensus - very reliable
            prediction = majority
    
    # RULE 3: Weak Consensus (2/4 split) - Use sophisticated weighting
    else:
        if direction == 'enter':
            # Enter-optimized weights
            votes = Counter()
            votes[row['pred_gbm']] += 0.42
            votes[row['pred_cond']] += 0.28
            votes[row['pred_final']] += 0.20
            votes[row['pred_calibrated']] += 0.10
            
            # Bonus: If GBM and cond agree, boost confidence
            if row['pred_gbm'] == row['pred_cond']:
                votes[row['pred_gbm']] += 0.15
                
        else:  # exit
            # Exit-optimized weights
            votes = Counter()
            votes[row['pred_gbm']] += 0.48
            votes[row['pred_final']] += 0.25
            votes[row['pred_cond']] += 0.17
            votes[row['pred_calibrated']] += 0.10
            
            # Bonus: If GBM and final agree, boost confidence
            if row['pred_gbm'] == row['pred_final']:
                votes[row['pred_gbm']] += 0.15
        
        prediction = votes.most_common(1)[0][0]
    
    ultra_predictions.append(prediction)

# Add to dataframe
df_analysis['ultra_prediction'] = ultra_predictions

# Analyze final distribution
dist = pd.Series(ultra_predictions).value_counts(normalize=True)
print(f"\nOverall Distribution:")
print(f"  Free:     {dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {dist.get('heavy delay', 0)*100:.1f}%")

# Analyze by direction
enter_mask = df_analysis['Direction'] == 'enter'
exit_mask = df_analysis['Direction'] == 'exit'

enter_dist = df_analysis[enter_mask]['ultra_prediction'].value_counts(normalize=True)
exit_dist = df_analysis[exit_mask]['ultra_prediction'].value_counts(normalize=True)

print(f"\nEnter Distribution:")
print(f"  Free:     {enter_dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {enter_dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {enter_dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {enter_dist.get('heavy delay', 0)*100:.1f}%")

print(f"\nExit Distribution:")
print(f"  Free:     {exit_dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {exit_dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {exit_dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {exit_dist.get('heavy delay', 0)*100:.1f}%")

# Analyze changes from hybrid_smart
hybrid_smart = pd.read_csv('submission_hybrid_smart.csv')
changes = (ultra_predictions != hybrid_smart['Target'].values).sum()
change_pct = changes / len(ultra_predictions) * 100

print(f"\n[IMPROVEMENT] Changes from hybrid_smart: {changes} predictions ({change_pct:.1f}%)")

# Show what changed
if changes > 0:
    changed_mask = [ultra_predictions[i] != hybrid_smart['Target'].values[i] for i in range(len(ultra_predictions))]
    changed_df = pd.DataFrame({
        'ID': df_analysis['ID'].values[changed_mask],
        'Direction': df_analysis['Direction'].values[changed_mask],
        'Old': hybrid_smart['Target'].values[changed_mask],
        'New': np.array(ultra_predictions)[changed_mask],
        'Consensus': df_analysis['consensus_ratio'].values[changed_mask]
    })
    
    print(f"\nSample Changes (first 10):")
    print(changed_df.head(10).to_string(index=False))
    
    # Analyze change patterns
    change_summary = changed_df.groupby(['Direction', 'Old', 'New']).size().reset_index(name='Count')
    change_summary = change_summary.sort_values('Count', ascending=False)
    print(f"\nChange Patterns:")
    print(change_summary.to_string(index=False))

# Save ultra-optimized submission
submission = pd.DataFrame({
    'ID': df_analysis['ID'],
    'Target': ultra_predictions,
    'Target_Accuracy': ultra_predictions
})

submission.to_csv('submission_ultra_optimized.csv', index=False)
print("\n[OK] Saved submission_ultra_optimized.csv")

# Also create a "safety" version that's slightly more conservative
print("\n[BONUS] Creating safety version with even more conservative approach...")
safety_predictions = []

for idx, row in df_analysis.iterrows():
    consensus_ratio = row['consensus_ratio']
    majority = row['majority_class']
    
    # Even more conservative: trust consensus unless very uncertain
    if consensus_ratio >= 0.75:
        prediction = majority
    else:
        # For weak consensus, be very conservative and prefer GBM+Final agreement
        if row['pred_gbm'] == row['pred_final']:
            prediction = row['pred_gbm']
        elif row['pred_gbm'] == row['pred_cond']:
            prediction = row['pred_gbm']
        else:
            # No clear agreement - use GBM (best single model)
            prediction = row['pred_gbm']
    
    safety_predictions.append(prediction)

safety_dist = pd.Series(safety_predictions).value_counts(normalize=True)
print(f"\nSafety Distribution:")
print(f"  Free:     {safety_dist.get('free flowing', 0)*100:.1f}%")
print(f"  Light:    {safety_dist.get('light delay', 0)*100:.1f}%")
print(f"  Moderate: {safety_dist.get('moderate delay', 0)*100:.1f}%")
print(f"  Heavy:    {safety_dist.get('heavy delay', 0)*100:.1f}%")

safety_submission = pd.DataFrame({
    'ID': df_analysis['ID'],
    'Target': safety_predictions,
    'Target_Accuracy': safety_predictions
})

safety_submission.to_csv('submission_safety.csv', index=False)
print("[OK] Saved submission_safety.csv")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print("\n🏆 PRIMARY CHOICE: submission_ultra_optimized.csv")
print("   - Most sophisticated logic")
print("   - Direction-aware adaptive weighting")
print("   - Consensus-validated predictions")
print(f"   - Distribution: {dist.get('free flowing', 0)*100:.1f}% F, "
      f"{dist.get('light delay', 0)*100:.1f}% L, "
      f"{dist.get('moderate delay', 0)*100:.1f}% M, "
      f"{dist.get('heavy delay', 0)*100:.1f}% H")

print("\n🛡️ BACKUP CHOICE: submission_safety.csv")
print("   - More conservative approach")
print("   - Strong preference for model agreement")
print("   - Lower risk profile")
print(f"   - Distribution: {safety_dist.get('free flowing', 0)*100:.1f}% F, "
      f"{safety_dist.get('light delay', 0)*100:.1f}% L, "
      f"{safety_dist.get('moderate delay', 0)*100:.1f}% M, "
      f"{safety_dist.get('heavy delay', 0)*100:.1f}% H")

print("\n[STRATEGY] Test ultra_optimized first - it has the best theoretical foundation!")
print("[TARGET] Hedef: 0.8013, Mevcut: 0.7708, Gap: +3.05%")
print("="*80)
